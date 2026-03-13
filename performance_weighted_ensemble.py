# utils/performance_weighted_ensemble.py
"""
基于性能的加权集成策略实现
根据模型在验证集上的性能指标动态调整权重
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import numpy as np
import os
import pandas as pd
from pathlib import Path

class PerformanceWeightedEnsemble(nn.Module):
    """
    基于性能的加权集成类
    根据模型在验证集上的性能指标动态调整权重
    """
    def __init__(self, models: List[nn.Module], 
                 performance_metrics: Dict[str, float],
                 metric_name: str = 'iou',
                 temperature: float = 2.0):
        """
        初始化基于性能的加权集成器
        
        Args:
            models: 要集成的模型列表
            performance_metrics: 各模型的性能指标字典，格式为 {model_name: metric_value}
            metric_name: 使用的性能指标名称，如 'iou', 'dice', 'f1_score' 等
            temperature: 温度参数，用于调整权重分布的尖锐程度
        """
        super(PerformanceWeightedEnsemble, self).__init__()
        self.models = nn.ModuleList(models)
        self.performance_metrics = performance_metrics
        self.metric_name = metric_name
        self.temperature = temperature
        
        # 根据性能指标计算权重
        self._compute_weights()
    
    def _compute_weights(self) -> None:
        """
        根据性能指标计算模型权重
        使用softmax函数将性能指标转换为权重
        """
        # 提取性能指标值
        metric_values = list(self.performance_metrics.values())
        
        # 转换为tensor
        metric_tensor = torch.tensor(metric_values, dtype=torch.float32)
        
        # 应用温度参数调整分布
        scaled_metrics = metric_tensor / self.temperature
        
        # 使用softmax计算权重
        weights = F.softmax(scaled_metrics, dim=0)
        
        # 存储权重
        self.register_buffer('weights', weights)
        
        print(f"基于{self.metric_name}指标的模型权重: {weights.tolist()}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        执行前向传播，使用基于性能的权重进行集成
        
        Args:
            x: 输入数据，形状为 [batch_size, channels, height, width]
        
        Returns:
            集成后的预测结果
        """
        # 获取所有模型的预测
        outputs = []
        probs = []
        with torch.no_grad():
            for model in self.models:
                model.eval()  # 设置为评估模式
                logits = model(x)
                outputs.append(logits)
                probs.append(torch.sigmoid(logits))  # 先转概率
        
        # 在概率域进行加权平均
        weighted_probs = [prob * weight for prob, weight in zip(probs, self.weights)]
        combined_prob = torch.sum(torch.stack(weighted_probs), dim=0)
        combined_prob = torch.clamp(combined_prob, 1e-6, 1 - 1e-6)
        
        return torch.logit(combined_prob)  # 再转回 logits
    
    def update_metrics(self, new_metrics: Dict[str, float]) -> None:
        """
        更新性能指标并重新计算权重
        
        Args:
            new_metrics: 新的性能指标字典
        """
        self.performance_metrics = new_metrics
        self._compute_weights()
    
    def get_weights(self) -> torch.Tensor:
        """
        获取当前权重
        
        Returns:
            当前权重tensor
        """
        return self.weights

def load_performance_metrics(csv_paths: List[str], 
                           model_names: List[str],
                           metric_name: str = 'iou') -> Dict[str, float]:
    """
    从CSV文件中加载各模型的性能指标
    
    Args:
        csv_paths: 各模型评估结果CSV文件路径列表
        model_names: 模型名称列表
        metric_name: 要加载的指标名称
        
    Returns:
        性能指标字典 {model_name: metric_value}
    """
    metrics = {}
    
    for csv_path, model_name in zip(csv_paths, model_names):
        if os.path.exists(csv_path):
            # 读取CSV文件
            df = pd.read_csv(csv_path)
            
            # 检查CSV文件结构
            if 'Metric' in df.columns and 'Value' in df.columns:
                # 处理Type,Category,Metric,Value结构的CSV文件
                metric_row = df[df['Metric'] == metric_name]
                if not metric_row.empty:
                    metric_value = metric_row['Value'].iloc[0]
                    metrics[model_name] = metric_value
                else:
                    print(f"警告: 在 {csv_path} 中未找到指标 {metric_name}")
                    metrics[model_name] = 0.5
            elif metric_name in df.columns:
                # 处理指标直接作为列名的CSV文件
                metric_value = df[metric_name].iloc[-1]
                metrics[model_name] = metric_value
            else:
                print(f"警告: 在 {csv_path} 中未找到指标 {metric_name}")
                metrics[model_name] = 0.5
        else:
            print(f"警告: 文件 {csv_path} 不存在")
            metrics[model_name] = 0.5
    
    return metrics

def create_performance_weighted_ensemble(model_paths: List[str],
                                       model_classes: List,
                                       model_names: List[str],
                                       csv_paths: List[str],
                                       metric_name: str = 'iou',
                                       temperature: float = 2.0,
                                       device: str = 'cuda'):
    """
    创建基于性能的加权集成模型
    
    Args:
        model_paths: 模型权重文件路径列表
        model_classes: 模型类列表
        model_names: 模型名称列表
        csv_paths: 各模型评估结果CSV文件路径列表
        metric_name: 使用的性能指标名称
        temperature: 温度参数
        device: 设备类型
        
    Returns:
        基于性能的加权集成模型实例
    """
    # 加载性能指标
    performance_metrics = load_performance_metrics(csv_paths, model_names, metric_name)
    
    # 加载模型
    models = []
    for model_path, model_class in zip(model_paths, model_classes):
        model = model_class()
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # 处理不同结构的checkpoint文件
        if isinstance(checkpoint, dict):
            # 尝试不同的键名来获取模型状态字典
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                # 如果没有找到特定的键，假设整个checkpoint就是状态字典
                state_dict = checkpoint
        else:
            # 如果checkpoint不是字典，假设它直接是状态字典
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        models.append(model)
    
    # 创建集成模型
    ensemble = PerformanceWeightedEnsemble(
        models=models,
        performance_metrics=performance_metrics,
        metric_name=metric_name,
        temperature=temperature
    )
    
    return ensemble.to(device)