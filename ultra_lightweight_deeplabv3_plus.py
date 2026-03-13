# models/lightweight_deeplabv3_plus_ultra.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv

try:
    from torchvision.models import MobileNet_V2_Weights
    _HAS_ENUM_WEIGHTS = True
except Exception:
    _HAS_ENUM_WEIGHTS = False

# 导入CBAM注意力模块
try:
    from utils.attention_modules import CBAM
    _HAS_CBAM = True
except ImportError:
    _HAS_CBAM = False

# 导入SEBlock注意力模块
try:
    from utils.attention_modules import SEBlock
    _HAS_SE = True
except ImportError:
    _HAS_SE = False


# ---------- 基础模块 ----------
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, k, s, p, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.bn = nn.GroupNorm(1, out_ch)  # 使用GroupNorm替代BatchNorm
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)


class MiniASPP(nn.Module):
    """
    极简 ASPP：1x1 + 多个空洞卷积 + 全局池化 分支
    通道小到 64，显存占用极低。
    """
    def __init__(self, in_ch, out_ch=64, aspp_rates=None):
        super().__init__()
        # 默认使用单个空洞卷积，rate=6
        if aspp_rates is None:
            aspp_rates = [6]
        
        # 1x1卷积分支
        self.b0 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(1, out_ch), nn.ReLU(inplace=True)  # 使用GroupNorm替代BatchNorm
        )
        
        # 多个空洞卷积分支
        self.branches = nn.ModuleList()
        for rate in aspp_rates:
            branch = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=rate, dilation=rate, bias=False),
                nn.GroupNorm(1, out_ch), nn.ReLU(inplace=True)  # 使用GroupNorm替代BatchNorm
            )
            self.branches.append(branch)
        
        # 全局池化分支
        self.gp = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(1, out_ch), nn.ReLU(inplace=True)  # 使用GroupNorm替代BatchNorm
        )
        
        # 输出卷积，分支数量 = 1 (1x1) + len(aspp_rates) (空洞卷积) + 1 (全局池化)
        total_branches = 2 + len(aspp_rates)
        self.out = nn.Sequential(
            nn.Conv2d(out_ch * total_branches, out_ch, 1, bias=False),
            nn.GroupNorm(1, out_ch), nn.ReLU(inplace=True),  # 使用GroupNorm替代BatchNorm
            nn.Dropout(0.05)
        )

    def forward(self, x):
        h, w = x.shape[-2:]
        
        # 1x1卷积分支
        b0 = self.b0(x)
        
        # 空洞卷积分支
        branch_outputs = [b0]
        for branch in self.branches:
            branch_outputs.append(branch(x))
        
        # 全局池化分支
        gp = F.adaptive_avg_pool2d(x, 1)
        gp = self.gp(gp)
        gp = F.interpolate(gp, size=(h, w), mode="bilinear", align_corners=False)
        branch_outputs.append(gp)
        
        # 拼接所有分支
        y = torch.cat(branch_outputs, dim=1)
        return self.out(y)


# ---------- MobileNetV2：6通道 & OS=32 ----------
def _mobilenet_v2_6ch(pretrained=True, n_channels=6):
    if pretrained and _HAS_ENUM_WEIGHTS:
        backbone = tv.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    elif pretrained:
        backbone = tv.mobilenet_v2(pretrained=True)
    else:
        backbone = tv.mobilenet_v2(pretrained=False)

    # 改首层为指定通道数
    first = backbone.features[0][0]
    new_first = nn.Conv2d(n_channels, first.out_channels, kernel_size=first.kernel_size,
                          stride=first.stride, padding=first.padding, bias=False)
    with torch.no_grad():
        if pretrained:
            w = first.weight  # [32,3,k,k]
            if n_channels == 3:
                # 如果是3通道，直接使用原始权重
                new_first.weight.copy_(w)
            elif n_channels > 3:
                # 如果通道数大于3，复制RGB通道的权重到前3个通道，其余通道使用平均权重
                mean_w = w.mean(dim=1, keepdim=True)
                new_w = mean_w.repeat(1, n_channels, 1, 1)
                new_w[:, 0:3] = w
                new_first.weight.copy_(new_w + 0.01 * torch.randn_like(new_w))
            else:
                # 如果通道数小于3，使用原始权重的平均值
                mean_w = w.mean(dim=1, keepdim=True)
                new_w = mean_w.repeat(1, n_channels, 1, 1)
                new_first.weight.copy_(new_w)
        else:
            # 如果不使用预训练权重，使用Kaiming初始化
            nn.init.kaiming_normal_(new_first.weight, mode='fan_out', nonlinearity='relu')
    backbone.features[0][0] = new_first
    return backbone

def _set_output_stride(backbone, output_stride=32):
    """
    根据output_stride参数设置MobileNetV2的输出步长：
    - output_stride=8: 修改features[7]和features[14]的stride和dilation
    - output_stride=16: 修改features[14]的stride和dilation
    - output_stride=32: 保持默认stride，不做修改
    
    注意：需要直接修改卷积层的参数，而不是仅仅修改属性
    """
    if output_stride == 32:
        # 默认情况，不做修改
        return backbone
    
    # MobileNetV2的下采样层索引（包含stride=2的InvertedResidual模块）
    # features[2]: stride=2
    # features[4]: stride=2
    # features[7]: stride=2
    # features[14]: stride=2
    
    if output_stride == 16:
        # 修改features[14]的stride从2改为1，dilation从1改为2
        idx = 14
        module = backbone.features[idx]
        
        # 直接修改3x3深度可分离卷积的参数
        conv_layer = module.conv[1][0]  # 获取3x3卷积层
        
        # 创建新的卷积层
        new_conv = nn.Conv2d(
            conv_layer.in_channels,
            conv_layer.out_channels,
            kernel_size=conv_layer.kernel_size,
            stride=(1, 1),  # 将stride从(2,2)改为(1,1)
            padding=(2, 2),  # 将padding从(1,1)改为(2,2)
            dilation=(2, 2),  # 将dilation从(1,1)改为(2,2)
            groups=conv_layer.groups,
            bias=conv_layer.bias is not None
        )
        
        # 复制权重
        with torch.no_grad():
            new_conv.weight.copy_(conv_layer.weight)
            if conv_layer.bias is not None:
                new_conv.bias.data.copy_(conv_layer.bias.data)
        
        # 替换卷积层
        module.conv[1][0] = new_conv
        
        # 修改模块的stride属性
        module.stride = 1
    
    elif output_stride == 8:
        # 修改features[7]和features[14]的stride从2改为1，dilation从1改为2和4
        for idx, dilation in [(7, 2), (14, 4)]:
            module = backbone.features[idx]
            
            # 直接修改3x3深度可分离卷积的参数
            conv_layer = module.conv[1][0]  # 获取3x3卷积层
            
            # 创建新的卷积层
            new_conv = nn.Conv2d(
                conv_layer.in_channels,
                conv_layer.out_channels,
                kernel_size=conv_layer.kernel_size,
                stride=(1, 1),  # 将stride从(2,2)改为(1,1)
                padding=(dilation, dilation),  # 根据dilation调整padding
                dilation=(dilation, dilation),  # 设置dilation
                groups=conv_layer.groups,
                bias=conv_layer.bias is not None
            )
            
            # 复制权重
            with torch.no_grad():
                new_conv.weight.copy_(conv_layer.weight)
                if conv_layer.bias is not None:
                    new_conv.bias.data.copy_(conv_layer.bias.data)
            
            # 替换卷积层
            module.conv[1][0] = new_conv
            
            # 修改模块的stride属性
            module.stride = 1
    
    return backbone


# ---------- Ultra-Light DeepLabV3+ ----------
class UltraLightDeepLabV3Plus(nn.Module):
    """
    Ultra 省显存版：
      - OS=32（backbone 输出 1/32）
      - ASPP 通道 64，分支 3
      - 解码器一层深度可分离 + 轻量低层跳连（从 /4 尺度）
      - 可选CBAM注意力机制
    """
    def __init__(self, n_channels=6, n_classes=1,
                 pretrained_backbone=True, aspp_out=64, dec_ch=64, low_ch_out=32, low_ch_in=32,
                 use_cbam=False, cbam_reduction_ratio=16, output_stride=32, aspp_rates=None, class_prior=None, use_se=False):
        super().__init__()

        self.backbone = _mobilenet_v2_6ch(pretrained=pretrained_backbone, n_channels=n_channels)
        _set_output_stride(self.backbone, output_stride)

        # low-level 采用 /4 尺度输出（一般≈24ch 或 32ch）
        self.low_slice_end = 4
        self.high_slice_start = 5

        # MobileNetV2 最后层 ≈1280ch
        self.aspp = MiniASPP(in_ch=1280, out_ch=aspp_out, aspp_rates=aspp_rates)

        # 解码器：仅一层
        self.decoder = DepthwiseSeparableConv(aspp_out + low_ch_out, dec_ch)
        self.classifier = nn.Conv2d(dec_ch, n_classes, 1)

        # 初始化时创建low_proj层，使用low_ch_out参数
        self.low_proj = nn.Sequential(
            nn.Conv2d(low_ch_in, low_ch_out, 1, bias=False),  # 使用可配置的输入通道数
            nn.GroupNorm(1, low_ch_out),  # 使用GroupNorm替代BatchNorm
            nn.ReLU(inplace=True)
        )
        
        # 兼容旧代码 - 使用low_reduce作为别名
        self.low_reduce = self.low_proj
        
        # 添加CBAM注意力机制（如果启用且可用）
        self.use_cbam = use_cbam and _HAS_CBAM
        if self.use_cbam:
            self.cbam_after_aspp = CBAM(aspp_out, reduction_ratio=cbam_reduction_ratio)
            self.cbam_after_decoder = CBAM(dec_ch, reduction_ratio=cbam_reduction_ratio)
        
        # 添加SE注意力机制（如果启用）
        self.use_se = use_se and _HAS_SE
        if self.use_se:
            self.se_after_aspp = SEBlock(aspp_out, reduction_ratio=16)
            self.se_after_decoder = SEBlock(dec_ch, reduction_ratio=16)
        
        # 保存class_prior参数，用于损失函数
        self.class_prior = class_prior
        
        self._init_head()
    
    def freeze_backbone(self):
        """冻结骨干网络参数"""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """解冻骨干网络参数"""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def _forward_slices(self, x):
        feats = self.backbone.features
        low = x
        for i in range(0, self.low_slice_end + 1):
            low = feats[i](low)
        high = low
        for i in range(self.high_slice_start, len(feats)):
            high = feats[i](high)
        return low, high

    def forward(self, x):
        b, c, h, w = x.shape
        low, high = self._forward_slices(x)

        aspp = self.aspp(high)                                  # /32
        aspp = F.interpolate(aspp, size=low.shape[-2:], mode="bilinear", align_corners=False)  # 到 /4
        
        # 应用CBAM注意力机制（如果启用）
        if self.use_cbam:
            aspp = self.cbam_after_aspp(aspp)
            
        # 应用SE注意力机制（如果启用）
        if self.use_se:
            aspp = self.se_after_aspp(aspp)
            
        low = self.low_reduce(low)                              # /4, low_ch_out
        y = torch.cat([aspp, low], dim=1)                       # /4, (aspp_out + low_ch_out)
        y = self.decoder(y)                                     # /4, dec_ch
        
        # 应用CBAM注意力机制（如果启用）
        if self.use_cbam:
            y = self.cbam_after_decoder(y)
            
        # 应用SE注意力机制（如果启用）
        if self.use_se:
            y = self.se_after_decoder(y)
            
        y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
        y = self.classifier(y)                                  # logits
        return y

    def _init_head(self):
        for m in [self.aspp, self.decoder, self.classifier]:
            for mod in m.modules() if isinstance(m, nn.Module) else []:
                if isinstance(mod, nn.Conv2d):
                    nn.init.kaiming_normal_(mod.weight, mode='fan_out', nonlinearity='relu')
                    if mod.bias is not None:
                        nn.init.zeros_(mod.bias)
                elif isinstance(mod, nn.GroupNorm):  # 修改为GroupNorm初始化
                    nn.init.ones_(mod.weight); nn.init.zeros_(mod.bias)


def get_ultra_light_deeplabv3_plus(n_channels=6, n_classes=1,
                                   pretrained_backbone=True,
                                   aspp_out=64, dec_ch=64, low_ch_out=32,
                                   use_cbam=False, cbam_reduction_ratio=16,
                                   output_stride=32, aspp_rates=None, class_prior=None, use_se=False):
    return UltraLightDeepLabV3Plus(
        n_channels=n_channels, n_classes=n_classes,
        pretrained_backbone=pretrained_backbone,
        aspp_out=aspp_out, dec_ch=dec_ch, low_ch_out=low_ch_out,
        use_cbam=use_cbam, cbam_reduction_ratio=cbam_reduction_ratio,
        output_stride=output_stride, aspp_rates=aspp_rates, 
        class_prior=class_prior, use_se=use_se
    )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = get_ultra_light_deeplabv3_plus(pretrained_backbone=True)
    x = torch.randn(1, 6, 256, 256)
    y = model.train()(x)
    print("input:", x.shape, "output:", y.shape)
    print("params:", f"{count_parameters(model):,}")
