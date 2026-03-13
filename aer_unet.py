# models/aer_unet.py
"""
Implementation of AER U-Net (Attention-Enhanced Multi-Scale Residual U-Net) for Sentinel-2 water segmentation.
Modified to support 6-channel input (blue, green, red, near-infrared, SWIR1, SWIR2).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionGate(nn.Module):
    """
    Attention Gate (AG) for U-Net to focus on relevant structures.
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class ResidualBlock(nn.Module):
    """
    A residual block with two convolutional layers.
    """
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class AERUNet(nn.Module):
    """
    Attention-Enhanced Multi-Scale Residual U-Net (AER U-Net) for water segmentation.
    Modified to support 6-channel input (Sentinel-2 bands).
    """
    def __init__(self, n_channels=6, n_classes=1, base_features=32, dropout_rate=0.3):
        super(AERUNet, self).__init__()
        
        # 编码器部分 (3个层级)
        # 编码器第1层: 32个过滤器
        self.enc1 = ResidualBlock(n_channels, base_features, dropout_rate)
        self.pool1 = nn.MaxPool2d(2)
        
        # 编码器第2层: 64个过滤器
        self.enc2 = ResidualBlock(base_features, base_features * 2, dropout_rate)
        self.pool2 = nn.MaxPool2d(2)
        
        # 编码器第3层: 128个过滤器
        self.enc3 = ResidualBlock(base_features * 2, base_features * 4, dropout_rate)
        self.pool3 = nn.MaxPool2d(2)

        # 瓶颈层 (256个过滤器)
        self.bottleneck = ResidualBlock(base_features * 4, base_features * 8, dropout_rate)
        
        # 解码器部分 (3个层级)
        # 解码器第3层 (输入: 256个过滤器, 输出: 128个过滤器)
        self.up3 = nn.ConvTranspose2d(base_features * 8, base_features * 4, kernel_size=2, stride=2)
        self.attn3 = AttentionGate(F_g=base_features * 4, F_l=base_features * 4, F_int=base_features * 2)
        self.dec3 = ResidualBlock(base_features * 8, base_features * 4, dropout_rate)

        # 解码器第2层 (输入: 128个过滤器, 输出: 64个过滤器)
        self.up2 = nn.ConvTranspose2d(base_features * 4, base_features * 2, kernel_size=2, stride=2)
        self.attn2 = AttentionGate(F_g=base_features * 2, F_l=base_features * 2, F_int=base_features)
        self.dec2 = ResidualBlock(base_features * 4, base_features * 2, dropout_rate)
        
        # 解码器第1层 (输入: 64个过滤器, 输出: 32个过滤器)
        self.up1 = nn.ConvTranspose2d(base_features * 2, base_features, kernel_size=2, stride=2)
        self.attn1 = AttentionGate(F_g=base_features, F_l=base_features, F_int=base_features // 2)
        self.dec1 = ResidualBlock(base_features * 2, base_features, dropout_rate)

        # 输出层
        self.out_conv = nn.Conv2d(base_features, n_classes, kernel_size=1)

    def forward(self, x):
        # 编码器路径
        e1 = self.enc1(x)  # 输入6通道图像
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        # 瓶颈层
        b = self.bottleneck(self.pool3(e3))
        
        # 带注意力机制的解码器路径
        d3 = self.up3(b)
        e3_attn = self.attn3(g=d3, x=e3)
        d3 = torch.cat((e3_attn, d3), dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        e2_attn = self.attn2(g=d2, x=e2)
        d2 = torch.cat((e2_attn, d2), dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        e1_attn = self.attn1(g=d1, x=e1)
        d1 = torch.cat((e1_attn, d1), dim=1)
        d1 = self.dec1(d1)

        return self.out_conv(d1)

def get_aer_unet_model(n_channels=6, n_classes=1, base_features=32, dropout_rate=0.3):
    """
    工厂函数，创建适用于Sentinel-2水体分割的AER U-Net模型。
    默认配置为6通道输入，适合Sentinel-2数据。
    """
    return AERUNet(
        n_channels=n_channels,
        n_classes=n_classes,
        base_features=base_features,
        dropout_rate=dropout_rate
    )