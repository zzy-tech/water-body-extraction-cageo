# utils/losses.py
"""
损失函数模块，包含focal loss和lovasz loss等高级损失函数。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union

class BCEDiceLoss(nn.Module):
    """
    BCE + Dice（二值分割）
    - 训练/验证共用同一实现（不要在loss里分叉）
    - BCE: reduction='mean'，可选 pos_weight
    - Dice: 用概率计算，FP32归约，和BCE同量纲（平均）
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0,
                 pos_weight: Union[float, None] = None, debug_interval: int = 0):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.smooth = float(smooth)
        self.debug_interval = int(debug_interval)
        self.debug_counter = 0

        if pos_weight is not None:
            self.register_buffer("pos_weight_tensor",
                                 torch.tensor([pos_weight], dtype=torch.float32))
            self.bce = nn.BCEWithLogitsLoss(reduction='mean',
                                            pos_weight=self.pos_weight_tensor)
        else:
            self.pos_weight_tensor = None
            self.bce = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, inputs, targets):
        # inputs: logits；targets: {0,1} 或 [0,1]
        targets = targets.float()
        
        # 确保pos_weight_tensor与输入在同一设备上
        if self.pos_weight_tensor is not None and self.pos_weight_tensor.device != inputs.device:
            self.pos_weight_tensor = self.pos_weight_tensor.to(inputs.device)
            # 重新创建BCE损失函数
            self.bce = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=self.pos_weight_tensor)

        # 添加数值稳定性检查
        if torch.isnan(inputs).any():
            print("Warning: inputs contain NaN values in BCEDiceLoss")
            inputs = torch.nan_to_num(inputs, nan=0.0)
        
        if torch.isinf(inputs).any():
            print("Warning: inputs contain Inf values in BCEDiceLoss")
            inputs = torch.nan_to_num(inputs, nan=0.0, posinf=1.0, neginf=-1.0)
            
        # 1) BCE（与训练一致；不要先 sigmoid）
        bce_loss = self.bce(inputs, targets)
        
        # 检查BCE是否为NaN
        if torch.isnan(bce_loss):
            print("Warning: BCE loss is NaN in BCEDiceLoss")
            bce_loss = torch.tensor(0.0).to(inputs.device)

        # 2) Dice（概率，FP32 归约）
        probs = torch.sigmoid(inputs).float().clamp(1e-7, 1 - 1e-7)
        inter = (probs * targets).sum(dtype=torch.float32)
        p_sum = probs.sum(dtype=torch.float32)
        t_sum = targets.sum(dtype=torch.float32)

        # 防止除零
        denominator = p_sum + t_sum + self.smooth
        if denominator == 0:
            dice_coeff = torch.tensor(0.0).to(inputs.device)
        else:
            dice_coeff = (2.0 * inter + self.smooth) / denominator
        dice_loss = (1.0 - dice_coeff).clamp(0.0, 1.0)
        
        # 检查Dice是否为NaN
        if torch.isnan(dice_loss):
            print("Warning: Dice loss is NaN in BCEDiceLoss")
            dice_loss = torch.tensor(0.0).to(inputs.device)

        # 3) 组合（训/验同权重）
        loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        
        # 最终检查
        if torch.isnan(loss):
            print("Warning: Combined BCEDiceLoss is NaN, using BCE only")
            loss = self.bce(inputs, targets)

        if self.debug_interval and self.training:
            self.debug_counter += 1
            if self.debug_counter % self.debug_interval == 0:
                print(f"DEBUG Dice [Batch {self.debug_counter}]: "
                      f"intersection={inter.item():.3f}, inputs_sum={p_sum.item():.3f}, "
                      f"targets_sum={t_sum.item():.3f}, smooth={self.smooth}")
                print(f"DEBUG Loss [Batch {self.debug_counter}]: "
                      f"bce={bce_loss.item():.6f}, dice={dice_loss.item():.6f}, "
                      f"combined={loss.item():.6f}")
        return loss

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    修改为适用于二值分割的Focal Loss。
    
    Args:
        alpha (float): 平衡因子，默认为1
        gamma (float): 聚焦参数，默认为2
        reduction (str): 损失聚合方式，'mean', 'sum' 或 'none'
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # 确保输入和目标在同一设备上
        if inputs.device != targets.device:
            targets = targets.to(inputs.device)
            
        # 使用sigmoid将logits转换为概率
        probs = torch.sigmoid(inputs)
        
        # 确保概率在合理范围内，避免log(0)
        probs = torch.clamp(probs, min=1e-7, max=1-1e-7)
        
        # 计算BCE损失
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # 计算概率权重
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = torch.pow(1 - pt, self.gamma)
        
        # 计算focal loss
        focal_loss = self.alpha * focal_weight * bce_loss
        
        # 检查损失是否为NaN
        if torch.isnan(focal_loss).any():
            print("Warning: FocalLoss contains NaN values, using BCE loss instead")
            focal_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # 聚合损失
        if self.reduction == 'mean':
            loss = focal_loss.mean()
        elif self.reduction == 'sum':
            loss = focal_loss.sum()
        else:  # 'none'
            loss = focal_loss
            
        # 最终检查
        if self.reduction != 'none' and torch.isnan(loss):
            print("Warning: Final FocalLoss is NaN, using BCE loss instead")
            loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction=self.reduction)
            
        return loss

class LovaszLoss(nn.Module):
    """
    Lovasz-Softmax Loss: A tractable surrogate for the optimization of the intersection-over-union measure in neural networks.
    修改为适用于二值分割的Lovasz-Hinge Loss。
    """
    def __init__(self, per_image=True, ignore=None):
        super(LovaszLoss, self).__init__()
        self.per_image = per_image
        self.ignore = ignore
        
    def forward(self, inputs, targets):
        # 确保输入和目标在同一设备上
        if inputs.device != targets.device:
            targets = targets.to(inputs.device)
            
        # 使用sigmoid将logits转换为概率
        probs = torch.sigmoid(inputs)
        
        # 确保概率在合理范围内
        probs = torch.clamp(probs, min=1e-7, max=1-1e-7)
        
        # 计算Lovasz-Hinge Loss
        if self.per_image:
            loss = self._lovasz_hinge_per_image(probs, targets)
        else:
            loss = self._lovasz_hinge_flat(probs, targets)
            
        # 检查损失是否为NaN
        if torch.isnan(loss):
            print("Warning: LovaszLoss is NaN, using BCE loss instead")
            loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
            
        return loss
    
    def _lovasz_hinge_per_image(self, probs, targets):
        losses = []
        for prob, target in zip(probs, targets):
            losses.append(self._lovasz_hinge_flat(prob.unsqueeze(0), target.unsqueeze(0)))
        
        # 检查是否有NaN
        losses_tensor = torch.stack(losses)
        if torch.isnan(losses_tensor).any():
            print("Warning: Some per-image Lovasz losses are NaN")
            losses_tensor = torch.nan_to_num(losses_tensor, nan=0.0)
            
        return torch.mean(losses_tensor)
    
    def _lovasz_hinge_flat(self, probs, targets):
        # 展平张量
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        # 忽略特定值（如果有）
        if self.ignore is not None:
            mask = targets != self.ignore
            if mask.sum() == 0:
                return torch.tensor(0.0).to(probs.device)
            probs = probs[mask]
            targets = targets[mask]
        
        # 计算边际
        margins = targets * (2 * probs - 1) + (1 - targets) * (1 - 2 * probs)
        
        # 排序边际
        sorted_margins, indices = torch.sort(margins, descending=True)
        sorted_targets = targets[indices]
        
        # 计算Lovasz扩展
        grad = torch.zeros_like(sorted_margins)
        grad[sorted_targets == 1] = 1
        
        # 计算累积和
        cumsum = torch.cumsum(grad, dim=0)
        
        # 防止除零
        if cumsum.sum() == 0:
            return torch.tensor(0.0).to(probs.device)
        
        # 计算损失 - 修复：确保损失为正值
        loss = torch.sum(torch.abs(sorted_margins) * cumsum) / cumsum.sum()
        
        # 检查损失是否为NaN
        if torch.isnan(loss):
            print("Warning: Lovasz hinge loss is NaN")
            loss = torch.tensor(0.0).to(probs.device)
        
        return loss

class FocalLovaszLoss(nn.Module):
    """
    Focal Loss和Lovasz Loss的组合损失函数。
    
    Args:
        focal_weight (float): Focal Loss的权重
        lovasz_weight (float): Lovasz Loss的权重
        focal_alpha (float): Focal Loss的alpha参数
        focal_gamma (float): Focal Loss的gamma参数
    """
    def __init__(self, focal_weight=0.5, lovasz_weight=0.5, focal_alpha=1, focal_gamma=2):
        super(FocalLovaszLoss, self).__init__()
        self.focal_weight = focal_weight
        self.lovasz_weight = lovasz_weight
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.lovasz_loss = LovaszLoss()
        
    def forward(self, inputs, targets):
        # 确保输入和目标在同一设备上
        if inputs.device != targets.device:
            targets = targets.to(inputs.device)
        
        # 添加数值稳定性检查
        if torch.isnan(inputs).any():
            print("Warning: inputs contain NaN values")
            inputs = torch.nan_to_num(inputs, nan=0.0)
        
        if torch.isinf(inputs).any():
            print("Warning: inputs contain Inf values")
            inputs = torch.nan_to_num(inputs, nan=0.0, posinf=1.0, neginf=-1.0)
            
        # 计算Focal Loss
        focal_loss = self.focal_loss(inputs, targets)
        
        # 检查Focal Loss是否为NaN
        if torch.isnan(focal_loss):
            print("Warning: focal_loss is NaN, using BCE loss instead")
            focal_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        
        # 计算Lovasz Loss
        lovasz_loss = self.lovasz_loss(inputs, targets)
        
        # 检查Lovasz Loss是否为NaN
        if torch.isnan(lovasz_loss):
            print("Warning: lovasz_loss is NaN, using BCE loss instead")
            lovasz_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        
        # 组合损失 - 修复：确保损失为正值
        combined_loss = torch.abs(self.focal_weight * focal_loss) + torch.abs(self.lovasz_weight * lovasz_loss)
        
        # 最终检查
        if torch.isnan(combined_loss):
            print("Warning: combined_loss is NaN, using BCE loss instead")
            combined_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        
        return combined_loss

class BCEFocalLovaszLoss(nn.Module):
    """
    BCE + Focal + Lovasz组合损失函数，结合了三种损失函数的优势。
    
    Args:
        bce_weight (float): BCE Loss的权重
        focal_weight (float): Focal Loss的权重
        lovasz_weight (float): Lovasz Loss的权重
        focal_alpha (float): Focal Loss的alpha参数
        focal_gamma (float): Focal Loss的gamma参数
        pos_weight (float): BCE Loss的正样本权重
        smooth (float): 平滑因子，用于数值稳定性
    """
    def __init__(self, bce_weight=0.33, focal_weight=0.33, lovasz_weight=0.34, 
                 focal_alpha=1, focal_gamma=2, pos_weight=None, smooth=1e-6):
        super(BCEFocalLovaszLoss, self).__init__()
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.lovasz_weight = lovasz_weight
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.lovasz_loss = LovaszLoss()
        self.smooth = smooth
        
        if pos_weight is not None:
            self.register_buffer("pos_weight_tensor",
                                 torch.tensor([pos_weight], dtype=torch.float32))
            self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean',
                                                 pos_weight=self.pos_weight_tensor)
        else:
            self.pos_weight_tensor = None
            self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
        
    def forward(self, inputs, targets):
        # 确保输入和目标在同一设备上
        if inputs.device != targets.device:
            targets = targets.to(inputs.device)
        
        # 确保pos_weight_tensor与输入在同一设备上
        if self.pos_weight_tensor is not None and self.pos_weight_tensor.device != inputs.device:
            self.pos_weight_tensor = self.pos_weight_tensor.to(inputs.device)
            # 重新创建BCE损失函数
            self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=self.pos_weight_tensor)
        
        # 添加数值稳定性检查
        if torch.isnan(inputs).any():
            print("Warning: inputs contain NaN values in BCEFocalLovaszLoss")
            inputs = torch.nan_to_num(inputs, nan=0.0)
        
        if torch.isinf(inputs).any():
            print("Warning: inputs contain Inf values in BCEFocalLovaszLoss")
            inputs = torch.nan_to_num(inputs, nan=0.0, posinf=1.0, neginf=-1.0)
            
        # 计算BCE Loss
        bce_loss = self.bce_loss(inputs, targets)
        
        # 检查BCE Loss是否为NaN
        if torch.isnan(bce_loss):
            print("Warning: bce_loss is NaN, using default BCE loss")
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        
        # 计算Focal Loss
        focal_loss = self.focal_loss(inputs, targets)
        
        # 检查Focal Loss是否为NaN
        if torch.isnan(focal_loss):
            print("Warning: focal_loss is NaN, using BCE loss instead")
            focal_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        
        # 计算Lovasz Loss
        lovasz_loss = self.lovasz_loss(inputs, targets)
        
        # 检查Lovasz Loss是否为NaN
        if torch.isnan(lovasz_loss):
            print("Warning: lovasz_loss is NaN, using BCE loss instead")
            lovasz_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        
        # 组合损失 - 确保损失为正值
        combined_loss = (torch.abs(self.bce_weight * bce_loss) + 
                         torch.abs(self.focal_weight * focal_loss) + 
                         torch.abs(self.lovasz_weight * lovasz_loss))
        
        # 最终检查
        if torch.isnan(combined_loss):
            print("Warning: combined_loss is NaN, using BCE loss instead")
            combined_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        
        return combined_loss