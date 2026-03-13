import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import logging
from typing import List, Dict, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

def _normalize_model_name(name: str) -> str:
    if not name:
        return ""
    normalized = name.strip().lower().replace(" ", "_").replace("-", "_")
    if normalized == "aer_u_net":
        normalized = "aer_unet"
    elif normalized == "ultra_lightweight_deeplabv3_":
        normalized = "ultra_lightweight_deeplabv3_plus"
    return normalized

class ImprovedPerformanceWeightedEnsemble(nn.Module):
    """
    改进的基于性能的加权集成类
    解决了原版本中模型预测相互抵消的问题
    支持多指标加权融合
    """
    def __init__(self, models: List[nn.Module], 
                 performance_metrics: Dict[str, Dict[str, float]],
                 metric_weights: Dict[str, float] = None,
                 metric_name: str = 'iou',
                 temperature: float = 1.0,
                 power: float = 2.0,
                 ensemble_method: str = 'gated_ensemble',
                 diff_threshold: float = 0.20,
                 conf_threshold: float = 0.22,
                 binary_threshold: float = 0.5,
                 model_names: List[str] = None):
        """
        初始化改进的基于性能的加权集成器
        
        Args:
            models: 要集成的模型列表
            performance_metrics: 各模型的性能指标字典，格式为 {model_name: {'iou': ..., 'dice': ..., 'f1': ...}}
            metric_weights: 各指标权重字典，格式为 {'iou': 0.5, 'dice': 0.3, 'f1': 0.2}，如果不提供则默认等权重
            metric_name: 主要指标名称（用于日志记录和向后兼容）
            temperature: 温度参数，用于调整权重分布的尖锐程度
            power: 幂函数参数，用于放大性能差异
            ensemble_method: 集成方法，可选 'logits_weighted'、'prob_weighted' 或 'gated_ensemble'
            diff_threshold: 差异阈值，超过此阈值认为两个模型预测分歧较大
            conf_threshold: 置信度阈值，低于此值认为模型不确定
            binary_threshold: 二值化阈值，用于置信度计算
            model_names: 模型名称列表（用于门控集成顺序识别）
        """
        super(ImprovedPerformanceWeightedEnsemble, self).__init__()
        self.models = nn.ModuleList(models)
        self.model_names = [_normalize_model_name(name) for name in model_names] if model_names else []
        self.aer_index = None
        self.ultra_index = None
        if self.model_names:
            for idx, name in enumerate(self.model_names):
                if name == 'aer_unet' and self.aer_index is None:
                    self.aer_index = idx
                if name == 'ultra_lightweight_deeplabv3_plus' and self.ultra_index is None:
                    self.ultra_index = idx
            if self.aer_index is None or self.ultra_index is None:
                logger.warning("gated_ensemble: could not find both 'aer_unet' and 'ultra_lightweight_deeplabv3_plus' in model_names; falling back to first two models.")
        self.performance_metrics = performance_metrics
        
        # 设置默认指标权重
        if metric_weights is None:
            # 获取所有可用的指标
            all_metrics = set()
            for model_metrics in performance_metrics.values():
                all_metrics.update(model_metrics.keys())
            
            # 默认等权重分配
            num_metrics = len(all_metrics)
            self.metric_weights = {metric: 1.0 / num_metrics for metric in all_metrics}
        else:
            self.metric_weights = metric_weights
        
        self.metric_name = metric_name
        self.temperature = temperature
        self.power = power
        self.ensemble_method = ensemble_method
        self.diff_threshold = diff_threshold
        self.conf_threshold = conf_threshold
        self.binary_threshold = binary_threshold  # 添加二值化阈值
        
        # 初始化权重buffer
        initial_weights = torch.ones(len(models)) / len(models)
        self.register_buffer('weights', initial_weights)
        
        # 根据性能指标计算权重
        self._compute_weights()
    
    def _compute_weights(self) -> None:
        """
        根据性能指标计算模型权重
        使用改进的多指标加权融合方法
        """
        # 归一化metric_weights以确保总和为1
        metric_weight_sum = sum(self.metric_weights.values())
        if metric_weight_sum > 0:
            normalized_metric_weights = {k: v / metric_weight_sum for k, v in self.metric_weights.items()}
            # 记录归一化后的指标权重
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"归一化后的指标权重: {normalized_metric_weights}")
        else:
            # 如果权重总和为0，使用默认等权重
            num_metrics = len(self.metric_weights)
            normalized_metric_weights = {k: 1.0 / num_metrics for k in self.metric_weights.keys()}
            print(f"警告: 指标权重总和为0，使用默认等权重: {normalized_metric_weights}")
        
        # 提取所有模型的综合指标分数
        composite_scores = []
        
        for model_name in self.performance_metrics.keys():
            model_metrics = self.performance_metrics[model_name]
            
            # 计算有效权重总和（只考虑该模型实际拥有的指标）
            total_weight = sum(weight for metric, weight in self.metric_weights.items() 
                             if metric in model_metrics)
            
            # 只有当有效权重总和大于0时才计算加权平均
            if total_weight > 0:
                # 先对权重进行归一化，确保权重之和为1
                normalized_weights = {k: v/total_weight for k, v in self.metric_weights.items() 
                                     if k in model_metrics}
                
                # 使用归一化权重计算加权平均分数
                recalc_weighted_sum = 0.0
                for metric_name, weight in normalized_weights.items():
                    recalc_weighted_sum += model_metrics[metric_name] * weight
                
                composite_scores.append(recalc_weighted_sum)
            else:
                # 如果所有指标都缺失，使用默认分数0.5
                composite_scores.append(0.5)
                print(f"警告: 模型 {model_name} 所有指标都缺失，使用默认分数 0.5")
        
        # 转换为tensor
        score_tensor = torch.tensor(composite_scores, dtype=torch.float32)
        
        # 改进的权重计算方法：
        # 1. 首先归一化到0-1范围
        min_score = torch.min(score_tensor)
        max_score = torch.max(score_tensor)
        if max_score > min_score:
            normalized_scores = (score_tensor - min_score) / (max_score - min_score)
        else:
            normalized_scores = torch.ones_like(score_tensor) / len(score_tensor)
        
        # 2. 应用幂函数放大差异
        amplified_scores = torch.pow(normalized_scores, self.power)
        
        # 3. 应用温度参数调整分布
        scaled_scores = amplified_scores / self.temperature
        
        # 4. 使用softmax计算权重
        weights = F.softmax(scaled_scores, dim=0)
        
        # 使用copy_更新已注册的buffer，而不是重复注册
        self.weights.copy_(weights)
        
        # 添加performance_weights属性，用于在evaluate.py中显示实际权重
        self.performance_weights = weights.tolist()
        
        # 导入logger以确保日志正确记录
        import logging
        logger = logging.getLogger(__name__)
        
        # 记录详细信息
        logger.info(f"使用多指标加权融合，指标权重: {self.metric_weights}")
        logger.info(f"模型综合分数: {composite_scores}")
        logger.info(f"基于综合指标的模型权重: {weights.tolist()}")
        logger.info(f"温度参数: {self.temperature}, 幂函数参数: {self.power}")
        
        # 记录原始性能指标详情
        for i, (model_name, model_metrics) in enumerate(self.performance_metrics.items()):
            # 使用正确的归一化权重计算加权综合分数
            total_weight = sum(weight for metric, weight in self.metric_weights.items() 
                             if metric in model_metrics)
            if total_weight > 0:
                normalized_weights = {k: v/total_weight for k, v in self.metric_weights.items() 
                                     if k in model_metrics}
                weighted_sum = sum(model_metrics.get(metric, 0) * weight 
                                 for metric, weight in normalized_weights.items())
            else:
                weighted_sum = 0.5
            
            logger.info(f"{model_name}: 原始指标={model_metrics}, 加权综合分数={weighted_sum:.4f}, 最终权重={weights[i].item():.4f}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        执行前向传播，使用改进的集成方法
        
        Args:
            x: 输入数据，形状为 [batch_size, channels, height, width]
        
        Returns:
            集成后的预测结果
        """
        # 获取所有模型的预测
        outputs = []
        with torch.no_grad():
            for model in self.models:
                model.eval()  # 设置为评估模式
                logits = model(x)
                outputs.append(logits)
        
        if self.ensemble_method == 'logits_weighted':
            # 方法1：在logits域进行加权平均
            weighted_logits = [logit * weight for logit, weight in zip(outputs, self.weights)]
            combined_logits = torch.sum(torch.stack(weighted_logits), dim=0)
            return combined_logits
        elif self.ensemble_method == 'prob_weighted':
            # 方法2：在概率域进行加权平均（原方法）
            probs = [torch.sigmoid(logits) for logits in outputs]
            weighted_probs = [prob * weight for prob, weight in zip(probs, self.weights)]
            combined_prob = torch.sum(torch.stack(weighted_probs), dim=0)
            combined_prob = torch.clamp(combined_prob, 1e-6, 1 - 1e-6)
            # 将概率转换回logits，确保与管线其他部分兼容
            return torch.logit(combined_prob)
        elif self.ensemble_method == 'gated_ensemble':
            # 方法3：改进的门控/条件集成策略
            probs = [torch.sigmoid(logits) for logits in outputs]

            if len(probs) < 2:
                return outputs[0]

            aer_idx = self.aer_index if self.aer_index is not None else 0
            ultra_idx = self.ultra_index if self.ultra_index is not None else 1
            if aer_idx == ultra_idx:
                aer_idx, ultra_idx = 0, 1

            # 使用显式模型顺序（如果可用）
            aer_prob = probs[aer_idx]
            ultra_prob = probs[ultra_idx]
            
            # 计算两个模型预测的差异（绝对差）
            prob_diff = torch.abs(aer_prob - ultra_prob)
            
            # 设定差异阈值，超过此阈值认为两个模型预测分歧较大
            diff_threshold = self.diff_threshold
            
            # 计算两个模型的置信度（离binary_threshold的距离）
            aer_conf = torch.abs(aer_prob - self.binary_threshold)
            ultra_conf = torch.abs(ultra_prob - self.binary_threshold)
            
            # 计算平均置信度，用于判断两个模型是否都不确定
            avg_conf = (aer_conf + ultra_conf) / 2
            
            # 设定置信度阈值，低于此值认为模型不确定
            conf_threshold = self.conf_threshold  # 预测值在(binary_threshold-conf_threshold)到(binary_threshold+conf_threshold)范围内时认为不确定
            
            # 改进的门控策略：
            # 1. 两个模型都不确定（平均置信度低）：使用加权平均
            # 2. 两个模型预测一致（差异小且至少一个确定）：使用置信度更高的模型
            # 3. 两个模型预测不一致（差异大）：使用置信度更高的模型
            
            # 情况1：两个模型都不确定
            both_uncertain_mask = avg_conf < conf_threshold
            
            # 情况2：两个模型预测一致且至少一个确定
            consistent_mask = (prob_diff < diff_threshold) & (avg_conf >= conf_threshold)
            
            # 情况3：两个模型预测不一致
            inconsistent_mask = (prob_diff >= diff_threshold) & (avg_conf >= conf_threshold)
            
            # 在一致或不确定的情况下，选择置信度更高的模型
            use_aer_in_confident = aer_conf >= ultra_conf
            
            # 应用改进的门控策略
            gated_prob = torch.where(
                both_uncertain_mask,  # 两个都不确定：使用加权平均
                (aer_prob * aer_conf + ultra_prob * ultra_conf) / (aer_conf + ultra_conf + 1e-8),
                torch.where(
                    consistent_mask,  # 一致且至少一个确定：使用置信度更高的模型
                    torch.where(use_aer_in_confident, aer_prob, ultra_prob),
                    torch.where(  # 不一致：使用置信度更高的模型
                        inconsistent_mask,
                        torch.where(use_aer_in_confident, aer_prob, ultra_prob),
                        # 默认情况：使用加权平均
                        (aer_prob * aer_conf + ultra_prob * ultra_conf) / (aer_conf + ultra_conf + 1e-8)
                    )
                )
            )
            
            # 确保概率值在合理范围内
            gated_prob = torch.clamp(gated_prob, 1e-6, 1 - 1e-6)
            
            # 将概率转换回logits，确保与管线其他部分兼容
            return torch.logit(gated_prob)
        else:
            raise ValueError(f"未知的集成方法: {self.ensemble_method}")
    
    def update_metrics(self, new_metrics: Dict[str, Dict[str, float]]) -> None:
        """
        更新性能指标并重新计算权重
        
        Args:
            new_metrics: 新的性能指标字典 {model_name: {'iou': ..., 'dice': ..., 'f1': ...}}
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

def create_improved_performance_weighted_ensemble(model_paths: List[str],
                                                 model_classes: List,
                                                 model_names: List[str],
                                                 csv_paths: List[str],
                                                 metric_names: List[str] = None,
                                                 metric_weights: Dict[str, float] = None,
                                                 metric_name: str = 'iou',
                                                 temperature: float = 1.0,
                                                 power: float = 2.0,
                                                 ensemble_method: str = 'gated_ensemble',
                                                 device: str = 'cuda',
                                                 n_channels: int = 6,
                                                 n_classes: int = 1,
                                                 model_configs: List[Dict] = None,
                                                 diff_threshold: float = 0.20,
                                                 conf_threshold: float = 0.22,
                                                 binary_threshold: float = 0.5):
    """
    创建改进的基于性能的加权集成模型
    
    Args:
        model_paths: 模型权重文件路径列表
        model_classes: 模型类列表
        model_names: 模型名称列表
        csv_paths: 各模型评估结果CSV文件路径列表
        metric_names: 要使用的性能指标名称列表，默认为['iou', 'dice', 'f1']
        metric_weights: 各指标权重字典，格式为 {'iou': 0.5, 'dice': 0.3, 'f1': 0.2}
        metric_name: 主要指标名称（用于日志记录和向后兼容）
        temperature: 温度参数，用于调整权重分布的尖锐程度
        power: 幂函数参数，用于放大性能差异
        ensemble_method: 集成方法
        device: 设备类型
        n_channels: 输入通道数
        n_classes: 输出类别数
        model_configs: 模型配置列表
        diff_threshold: 差异阈值
        conf_threshold: 置信度阈值
        binary_threshold: 二值化阈值
        
    Returns:
        改进的基于性能的加权集成模型实例
    """
    # 设置默认指标列表
    if metric_names is None:
        metric_names = ['iou', 'dice', 'f1']
    
    # 加载多指标性能数据
    performance_metrics = load_performance_metrics(csv_paths, model_names, metric_names)
    
    # 加载模型
    models = []
    for i, (model_path, model_class) in enumerate(zip(model_paths, model_classes)):
        # 获取当前模型的配置
        cfg = model_configs[i] if model_configs and i < len(model_configs) else {}
        
        # 规范化模型名称，使其能够匹配配置中的名称
        normalized_model_name = model_names[i].strip().lower().replace(" ", "_").replace("-", "_")
        if normalized_model_name == "aer_u_net":
            normalized_model_name = "aer_unet"
        elif normalized_model_name == "ultra_lightweight_deeplabv3_":
            normalized_model_name = "ultra_lightweight_deeplabv3_plus"
        
        # 根据模型类型和配置创建模型实例
        if normalized_model_name == 'aer_unet':
            model = model_class(
                n_channels=cfg.get('n_channels', n_channels),
                n_classes=cfg.get('n_classes', n_classes),
                base_features=cfg.get('base_features', 32),  # AER U-Net默认使用32个基础通道
                dropout_rate=cfg.get('dropout_rate', 0.3)
            )
        elif normalized_model_name in ['lightweight_unet', 'lightweight_deeplabv3_plus']:
            model = model_class(
                n_channels=cfg.get('n_channels', n_channels),
                n_classes=cfg.get('n_classes', n_classes),
                base_features=cfg.get('base_features', 64),
                dropout_rate=cfg.get('dropout_rate', 0.3),
                output_stride=cfg.get('output_stride', 16),
                pretrained_backbone=cfg.get('pretrained_backbone', True)
            )
        elif normalized_model_name == 'deeplabv3_plus':
            model = model_class(
                n_channels=cfg.get('n_channels', n_channels),
                n_classes=cfg.get('n_classes', n_classes),
                output_stride=cfg.get('output_stride', 16),
                pretrained_backbone=cfg.get('pretrained_backbone', True)
            )
        elif normalized_model_name == 'ultra_lightweight_deeplabv3_plus':
            model = model_class(
                n_channels=cfg.get('n_channels', n_channels),
                n_classes=cfg.get('n_classes', n_classes),
                pretrained_backbone=cfg.get('pretrained_backbone', True),
                aspp_out=cfg.get('aspp_out', 64),
                dec_ch=cfg.get('dec_ch', 64),
                low_ch_out=cfg.get('low_ch_out', 32),
                use_cbam=cfg.get('use_cbam', False),
                cbam_reduction_ratio=cfg.get('cbam_reduction_ratio', 16),
                output_stride=cfg.get('output_stride', 16)
            )
        else:
            # 默认参数
            model = model_class(
                n_channels=cfg.get('n_channels', n_channels),
                n_classes=cfg.get('n_classes', n_classes)
            )
        
        # 加载检查点
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
            
        # 特殊处理Ultra-Light DeepLabV3+模型的键名映射
        if model_names[i] == 'ultra_lightweight_deeplabv3_plus':
            state_dict = {k.replace('low_reduce', 'low_proj'): v for k, v in state_dict.items()}
        
        # 使用strict=False加载，以兼容不同版本的模型
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[Load] {model_names[i]} loaded with strict=False | missing: {len(missing)} | unexpected: {len(unexpected)}")
        
        if len(missing) > 0:
            print("  missing (sample):")
            for m in list(missing)[:5]:  # 只显示前5个
                print(f"    - {m}")
        
        if len(unexpected) > 0:
            print("  unexpected (sample):")
            for u in list(unexpected)[:5]:  # 只显示前5个
                print(f"    - {u}")
        
        model.to(device)
        model.eval()
        models.append(model)
    
    # 创建集成模型
    ensemble = ImprovedPerformanceWeightedEnsemble(
        models=models,
        performance_metrics=performance_metrics,
        metric_weights=metric_weights,
        metric_name=metric_name,
        temperature=temperature,
        power=power,
        ensemble_method=ensemble_method,
        diff_threshold=diff_threshold,
        conf_threshold=conf_threshold,
        binary_threshold=binary_threshold,
        model_names=model_names
    )
    
    return ensemble.to(device)

def load_performance_metrics(csv_paths: List[str], 
                           model_names: List[str],
                           metric_names: List[str] = None) -> Dict[str, Dict[str, float]]:
    """
    从CSV文件中加载各模型的性能指标
    
    Args:
        csv_paths: 各模型评估结果CSV文件路径列表
        model_names: 模型名称列表
        metric_names: 要加载的指标名称列表，如果为None则默认加载['iou', 'dice', 'f1']
        
    Returns:
        性能指标字典 {model_name: {'iou': value, 'dice': value, 'f1': value, ...}}
    """
    # 默认指标列表
    if metric_names is None:
        metric_names = ['iou', 'dice', 'f1']
    
    # 如果传入的是字符串，转换为列表
    if isinstance(metric_names, str):
        metric_names = [metric_names]
    
    # 获取缺失指标默认值（从环境变量或配置中获取）
    missing_metric_default = 0.5
    
    metrics = {}
    
    for csv_path, model_name in zip(csv_paths, model_names):
        if os.path.exists(csv_path):
            try:
                # 读取CSV文件
                df = pd.read_csv(csv_path)
                
                # 初始化该模型的指标字典
                model_metrics = {}
                
                # 检查CSV文件结构
                if 'Metric' in df.columns and 'Value' in df.columns:
                    # 处理Type,Category,Metric,Value结构的CSV文件
                    for metric_name in metric_names:
                        metric_row = df[df['Metric'] == metric_name]
                        if not metric_row.empty:
                            metric_value = float(metric_row['Value'].iloc[0])  # 确保转换为float
                            # 检查值是否有效（不为NaN或None）
                            if pd.notna(metric_value) and isinstance(metric_value, (int, float)):
                                # 确保值在合理范围内
                                if 0 <= metric_value <= 1:
                                    model_metrics[metric_name] = metric_value
                                else:
                                    print(f"警告: 在 {csv_path} 中指标 {metric_name} 的值 {metric_value} 超出合理范围 [0,1]，使用默认值")
                                    model_metrics[metric_name] = missing_metric_default
                            else:
                                print(f"警告: 在 {csv_path} 中指标 {metric_name} 的值无效，使用默认值")
                                model_metrics[metric_name] = missing_metric_default
                        else:
                            print(f"警告: 在 {csv_path} 中未找到指标 {metric_name}，使用默认值{missing_metric_default}")
                            model_metrics[metric_name] = missing_metric_default
                            
                    # 尝试从其他可能的指标名称中查找
                    possible_names = {
                        'dice': ['dice', 'dice_coefficient', 'dice_coeff'],
                        'f1': ['f1', 'f1_score', 'f1score'],
                        'precision': ['precision', 'prec'],
                        'recall': ['recall', 'rec']
                    }
                    
                    for target_name, possible_names_list in possible_names.items():
                        if target_name not in model_metrics or model_metrics[target_name] == missing_metric_default:
                            for possible_name in possible_names_list:
                                metric_row = df[df['Metric'] == possible_name]
                                if not metric_row.empty:
                                    try:
                                        metric_value = float(metric_row['Value'].iloc[0])
                                        if pd.notna(metric_value) and 0 <= metric_value <= 1:
                                            model_metrics[target_name] = metric_value
                                            print(f"从 {csv_path} 中找到替代指标 {possible_name} 作为 {target_name}")
                                            break
                                    except (ValueError, TypeError):
                                        continue
                    
                elif all(metric_name in df.columns for metric_name in metric_names):
                    # 处理指标直接作为列名的CSV文件
                    for metric_name in metric_names:
                        try:
                            metric_value = df[metric_name].iloc[-1]  # 获取最后一个值
                            if pd.notna(metric_value):
                                if isinstance(metric_value, str):
                                    # 处理百分比字符串（如"85.2%"）
                                    if '%' in metric_value:
                                        metric_value = float(metric_value.replace('%', '')) / 100
                                    else:
                                        metric_value = float(metric_value)
                                
                                if 0 <= metric_value <= 1:
                                    model_metrics[metric_name] = float(metric_value)
                                else:
                                    print(f"警告: 在 {csv_path} 中指标 {metric_name} 的值 {metric_value} 超出合理范围 [0,1]，使用默认值")
                                    model_metrics[metric_name] = missing_metric_default
                            else:
                                print(f"警告: 在 {csv_path} 中指标 {metric_name} 的值缺失，使用默认值")
                                model_metrics[metric_name] = missing_metric_default
                        except (ValueError, TypeError):
                            print(f"警告: 在 {csv_path} 中无法解析指标 {metric_name}，使用默认值")
                            model_metrics[metric_name] = missing_metric_default
                else:
                    print(f"警告: 在 {csv_path} 中未找到所需的指标列，使用默认值{missing_metric_default}")
                    for metric_name in metric_names:
                        model_metrics[metric_name] = missing_metric_default
                
                metrics[model_name] = model_metrics
                
                # 打印成功加载的指标
                print(f"从 {csv_path} 为模型 {model_name} 加载指标:")
                for metric, value in model_metrics.items():
                    print(f"  {metric}: {value:.4f}")
                    
            except Exception as e:
                print(f"警告: 读取 {csv_path} 时发生错误: {e}，使用默认指标")
                # 为该模型创建默认指标字典
                default_metrics = {}
                for metric_name in metric_names:
                    default_metrics[metric_name] = missing_metric_default
                metrics[model_name] = default_metrics
        else:
            print(f"警告: 文件 {csv_path} 不存在，使用默认指标")
            # 为该模型创建默认指标字典
            default_metrics = {}
            for metric_name in metric_names:
                default_metrics[metric_name] = missing_metric_default
            metrics[model_name] = default_metrics
    
    return metrics
