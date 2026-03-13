import torch
import numpy as np
import random

def mixup_data(x, y, alpha=1.0):
    """
    实现MixUp数据增强
    
    Args:
        x: 输入图像张量 (batch_size, C, H, W)
        y: 目标掩码张量 (batch_size, 1, H, W) 或 (batch_size, H, W)
        alpha: MixUp强度参数
        
    Returns:
        mixed_x: 混合后的图像
        mixed_y: 混合后的掩码
        lam: 混合比例
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index, :]
    
    return mixed_x, mixed_y, lam

def cutmix_data(x, y, alpha=1.0):
    """
    实现CutMix数据增强
    
    Args:
        x: 输入图像张量 (batch_size, C, H, W)
        y: 目标掩码张量 (batch_size, 1, H, W) 或 (batch_size, H, W)
        alpha: CutMix强度参数
        
    Returns:
        mixed_x: 混合后的图像
        mixed_y: 混合后的掩码
        lam: 混合比例
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    _, _, H, W = x.shape
    
    # 生成随机边界框
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    # 随机中心点
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    # 边界框边界
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    # 应用CutMix
    mixed_x = x.clone()
    mixed_y = y.clone()
    
    mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    mixed_y[:, :, bbx1:bbx2, bby1:bby2] = y[index, :, bbx1:bbx2, bby1:bby2]
    
    # 调整lambda以反映实际混合区域
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (H * W))
    
    return mixed_x, mixed_y, lam

def apply_random_augmentation(x, y, mixup_prob=0.5, cutmix_prob=0.5, mixup_alpha=1.0, cutmix_alpha=1.0):
    """
    随机应用MixUp或CutMix数据增强
    
    Args:
        x: 输入图像张量
        y: 目标掩码张量
        mixup_prob: 应用MixUp的概率
        cutmix_prob: 应用CutMix的概率
        mixup_alpha: MixUp的alpha参数
        cutmix_alpha: CutMix的alpha参数
        
    Returns:
        augmented_x: 增强后的图像
        augmented_y: 增强后的掩码
        aug_type: 应用的增强类型 ('none', 'mixup', 'cutmix')
        lam: 混合比例 (如果没有应用增强则为1.0)
    """
    r = random.random()
    
    if r < mixup_prob:
        # 应用MixUp
        augmented_x, augmented_y, lam = mixup_data(x, y, alpha=mixup_alpha)
        return augmented_x, augmented_y, 'mixup', lam
    elif r < mixup_prob + cutmix_prob:
        # 应用CutMix
        augmented_x, augmented_y, lam = cutmix_data(x, y, alpha=cutmix_alpha)
        return augmented_x, augmented_y, 'cutmix', lam
    else:
        # 不应用增强
        return x, y, 'none', 1.0