# utils/crf_utils.py
"""
CRF (Conditional Random Field) 后处理工具函数，用于优化分割结果。
"""
import torch
import numpy as np
import torch.nn.functional as F
from typing import Optional, Tuple

class DenseCRF:
    """
    稠密条件随机场(DenseCRF)实现，用于分割后处理。
    这是一个简化版的CRF实现，使用PyTorch进行加速。
    """
    def __init__(self, iter_max: int = 10, 
                 pos_w: float = 3.0, pos_xy_std: float = 1.0,
                 bi_w: float = 4.0, bi_xy_std: float = 67.0, bi_rgb_std: float = 3.0):
        """
        初始化CRF参数
        
        Args:
            iter_max: 迭代次数
            pos_w: 位置权重
            pos_xy_std: 位置空间标准差
            bi_w: 双边权重
            bi_xy_std: 双边位置空间标准差
            bi_rgb_std: 双边颜色空间标准差
        """
        self.iter_max = iter_max
        self.pos_w = pos_w
        self.pos_xy_std = pos_xy_std
        self.bi_w = bi_w
        self.bi_xy_std = bi_xy_std
        self.bi_rgb_std = bi_rgb_std
    
    def _create_affinity_kernel(self, image: torch.Tensor, 
                                xy_std: float, rgb_std: Optional[float] = None) -> torch.Tensor:
        """
        创建亲和性核
        
        Args:
            image: 输入图像 [C, H, W]
            xy_std: 位置空间标准差
            rgb_std: 颜色空间标准差，如果为None则只考虑位置
            
        Returns:
            亲和性核 [H*W, H*W]
        """
        device = image.device
        H, W = image.shape[1], image.shape[2]
        
        # 创建位置网格
        xx, yy = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        positions = torch.stack([xx.flatten(), yy.flatten()], dim=0)  # [2, H*W]
        
        # 计算位置差异
        pos_diff = positions.unsqueeze(2) - positions.unsqueeze(1)  # [2, H*W, H*W]
        pos_dist = torch.sum(pos_diff ** 2, dim=0)  # [H*W, H*W]
        
        # 位置亲和性
        pos_affinity = torch.exp(-pos_dist / (2 * xy_std ** 2))
        
        if rgb_std is not None:
            # 计算颜色差异
            if image.shape[0] == 1:  # 灰度图像
                pixels = image.flatten().unsqueeze(0)  # [1, H*W]
            else:  # 彩色图像
                pixels = image.view(image.shape[0], -1)  # [C, H*W]
            
            rgb_diff = pixels.unsqueeze(2) - pixels.unsqueeze(1)  # [C, H*W, H*W]
            rgb_dist = torch.sum(rgb_diff ** 2, dim=0)  # [H*W, H*W]
            
            # 双边亲和性
            bi_affinity = torch.exp(-rgb_dist / (2 * rgb_std ** 2))
            
            # 组合位置和颜色亲和性
            affinity = pos_affinity * bi_affinity
        else:
            affinity = pos_affinity
            
        return affinity
    
    def __call__(self, image: torch.Tensor, unary: torch.Tensor) -> torch.Tensor:
        """
        应用CRF后处理
        
        Args:
            image: 输入图像 [C, H, W]
            unary: 一元势能 [H, W] 或 [1, H, W]
            
        Returns:
            CRF处理后的概率 [H, W]
        """
        device = image.device
        
        # 确保unary形状正确
        if unary.dim() == 3:
            unary = unary.squeeze(0)  # [H, W]
        
        # 将一元势能转换为概率
        if unary.min() < 0:  # 如果是logits
            prob = torch.sigmoid(unary)
        else:  # 如果已经是概率
            prob = unary
        
        # 创建Q矩阵 [2, H*W]，其中Q[0]是背景概率，Q[1]是前景概率
        Q = torch.stack([1 - prob, prob], dim=0)  # [2, H*W]
        
        # 创建亲和性核
        pos_kernel = self._create_affinity_kernel(image, self.pos_xy_std)
        bi_kernel = self._create_affinity_kernel(image, self.bi_xy_std, self.bi_rgb_std)
        
        # 迭代优化
        for _ in range(self.iter_max):
            # 计算消息传递
            pos_message = self.pos_w * torch.matmul(pos_kernel, Q)
            bi_message = self.bi_w * torch.matmul(bi_kernel, Q)
            
            # 更新Q
            Q = torch.exp(unary.flatten() + pos_message + bi_message)
            
            # 归一化
            Q = Q / (Q.sum(dim=0, keepdim=True) + 1e-10)
        
        # 返回前景概率
        return Q[1].view_as(unary)


def apply_crf_postprocessing(image: torch.Tensor, prediction: torch.Tensor, 
                           iterations: int = 10) -> torch.Tensor:
    """
    应用CRF后处理到预测结果
    
    Args:
        image: 原始图像 [C, H, W]
        prediction: 模型预测 [1, H, W] 或 [H, W]
        iterations: CRF迭代次数
        
    Returns:
        CRF处理后的预测 [H, W]
    """
    # 确保输入在正确的设备上
    device = image.device
    
    # 如果预测是logits，转换为概率
    if prediction.min() < 0 or prediction.max() > 1:
        prediction = torch.sigmoid(prediction)
    
    # 确保预测形状正确
    if prediction.dim() == 3:
        prediction = prediction.squeeze(0)
    
    # 如果图像是单通道，复制为三通道（对于双边滤波）
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    
    # 初始化CRF
    crf = DenseCRF(iter_max=iterations)
    
    # 应用CRF
    refined_prediction = crf(image, prediction)
    
    return refined_prediction


def batch_apply_crf(images: torch.Tensor, predictions: torch.Tensor, 
                   iterations: int = 10) -> torch.Tensor:
    """
    批量应用CRF后处理
    
    Args:
        images: 批量图像 [B, C, H, W]
        predictions: 批量预测 [B, 1, H, W] 或 [B, H, W]
        iterations: CRF迭代次数
        
    Returns:
        CRF处理后的批量预测 [B, H, W]
    """
    batch_size = images.shape[0]
    device = images.device
    
    # 确保预测形状正确
    if predictions.dim() == 4:
        predictions = predictions.squeeze(1)  # [B, H, W]
    
    # 如果预测是logits，转换为概率
    if predictions.min() < 0 or predictions.max() > 1:
        predictions = torch.sigmoid(predictions)
    
    # 初始化结果张量
    refined_predictions = torch.zeros_like(predictions)
    
    # 对每个样本应用CRF
    for i in range(batch_size):
        refined_predictions[i] = apply_crf_postprocessing(
            images[i], predictions[i], iterations
        )
    
    return refined_predictions