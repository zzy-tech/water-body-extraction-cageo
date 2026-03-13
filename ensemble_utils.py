# utils/ensemble_utils.py
"""
工具函数用于模型集成，支持多种集成策略，如平均、投票、加权平均等。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import os

class ModelEnsemble(nn.Module):
    """
    模型集成类，支持多种集成策略，结合多个模型的预测结果。
    """
    def __init__(self, models: List[nn.Module], strategy: str = 'mean', weights: Optional[List[float]] = None):
        """
        初始化模型集成器。
        
        Args:
            models: 要集成的模型列表
            strategy: 集成策略，可选 'mean', 'weighted_mean', 'vote', 'logits_mean'
            weights: 用于加权平均策略的权重列表
        """
        super(ModelEnsemble, self).__init__()
        self.models = nn.ModuleList(models)
        self.strategy = strategy.lower()
        
        # 验证策略
        valid_strategies = ['mean', 'weighted_mean', 'vote', 'logits_mean', 'performance_weighted']
        if self.strategy not in valid_strategies:
            raise ValueError(f"无效的集成策略 '{strategy}'。可选: {', '.join(valid_strategies)}")
        
        # 初始化权重
        if self.strategy == 'weighted_mean':
            if weights is None:
                # 默认等权重
                self.weights = nn.Parameter(torch.ones(len(models)) / len(models), requires_grad=False)
            else:
                if len(weights) != len(models):
                    raise ValueError("权重数量必须与模型数量匹配")
                self.weights = nn.Parameter(torch.tensor(weights), requires_grad=False)
        else:
            self.weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        执行前向传播，通过集成策略合并多个模型的预测结果。
        
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
                outputs.append(model(x))
        
        # 执行集成策略
        if self.strategy == 'mean':
            return self._mean_ensemble(outputs)
        elif self.strategy == 'weighted_mean':
            return self._weighted_mean_ensemble(outputs)
        elif self.strategy == 'vote':
            return self._vote_ensemble(outputs)
        elif self.strategy == 'logits_mean':
            return self._logits_mean_ensemble(outputs)
        elif self.strategy == 'performance_weighted':
            return self._performance_weighted_ensemble(outputs)
        
        # 不应该到达这里，因为初始化时已经验证了策略
        raise ValueError(f"未知的集成策略: {self.strategy}")
    
    def _mean_ensemble(self, outputs: List[torch.Tensor]) -> torch.Tensor:
        """简单平均集成"""
        # 将logits转换为概率后再进行平均
        probs = [torch.sigmoid(output) for output in outputs]
        # 在概率域进行平均，然后将结果转换回logits
        mean_prob = torch.mean(torch.stack(probs), dim=0)
        # 将概率转换回logits，使用clamp避免数值问题
        return torch.logit(torch.clamp(mean_prob, 1e-6, 1-1e-6))
    
    def _weighted_mean_ensemble(self, outputs: List[torch.Tensor]) -> torch.Tensor:
        """加权平均集成"""
        # 将logits转换为概率后再进行加权平均
        probs = [torch.sigmoid(output) for output in outputs]
        # 确保权重正确归一化
        normalized_weights = F.softmax(self.weights, dim=0)
        weighted_outputs = [prob * weight for prob, weight in zip(probs, normalized_weights)]
        # 在概率域进行加权平均，然后将结果转换回logits
        weighted_prob = torch.sum(torch.stack(weighted_outputs), dim=0)
        # 将概率转换回logits，使用clamp避免数值问题
        return torch.logit(torch.clamp(weighted_prob, 1e-6, 1-1e-6))
    
    def _vote_ensemble(self, outputs: List[torch.Tensor]) -> torch.Tensor:
        """投票集成，适用于分类任务"""
        # 对于二值分割，我们将输出转换为0/1预测
        predictions = []
        for output in outputs:
            # 对于单通道输出，使用0.5作为阈值
            if output.shape[1] == 1:
                pred = (torch.sigmoid(output) > 0.5).float()
            else:
                # 对于多通道输出，选择概率最高的类别
                pred = torch.argmax(torch.softmax(output, dim=1), dim=1, keepdim=True).float()
            predictions.append(pred)
        
        # 计算投票结果（平均值，然后取整）
        vote_result = torch.mean(torch.stack(predictions), dim=0) >= 0.5
        return vote_result.float()
    
    def _logits_mean_ensemble(self, outputs: List[torch.Tensor]) -> torch.Tensor:
        """在logits级别上进行平均，然后应用激活函数"""
        # 直接对logits进行平均
        mean_logits = torch.mean(torch.stack(outputs), dim=0)
        # 不再应用sigmoid，直接返回logits以保持接口一致性
        return mean_logits
    
    def _performance_weighted_ensemble(self, outputs: List[torch.Tensor]) -> torch.Tensor:
        """基于性能的加权集成，使用模型在验证集上的性能指标作为权重"""
        # 如果没有提供性能权重，则使用默认权重
        if self.weights is None:
            # 默认等权重
            weights = torch.ones(len(outputs)) / len(outputs)
        else:
            weights = self.weights
        
        # 将logits转换为概率后再进行加权
        probs = [torch.sigmoid(output) for output in outputs]
        
        # 确保权重正确归一化
        normalized_weights = F.softmax(weights, dim=0)
        
        # 应用权重
        weighted_outputs = [prob * weight for prob, weight in zip(probs, normalized_weights)]
        # 在概率域进行加权平均，然后将结果转换回logits
        weighted_prob = torch.sum(torch.stack(weighted_outputs), dim=0)
        # 将概率转换回logits，使用clamp避免数值问题
        return torch.logit(torch.clamp(weighted_prob, 1e-6, 1-1e-6))
    
    def set_strategy(self, strategy: str, weights: Optional[List[float]] = None) -> None:
        """
        更新集成策略
        
        Args:
            strategy: 新的集成策略
            weights: 用于加权平均策略的新权重
        """
        self.strategy = strategy.lower()
        
        if self.strategy == 'weighted_mean' and weights is not None:
            if len(weights) != len(self.models):
                raise ValueError("权重数量必须与模型数量匹配")
            self.weights = nn.Parameter(torch.tensor(weights), requires_grad=False)
    
    def get_model_predictions(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        获取每个模型的单独预测结果，不进行集成
        
        Args:
            x: 输入数据
        
        Returns:
            每个模型的预测结果列表
        """
        predictions = []
        with torch.no_grad():
            for model in self.models:
                model.eval()
                predictions.append(model(x))
        return predictions

class WeightedModelEnsemble(ModelEnsemble):
    """
    带可学习权重的模型集成类。
    权重可以通过训练进行优化，以找到最佳组合策略。
    """
    def __init__(self, models: List[nn.Module], n_classes: int = 1):
        """
        初始化带可学习权重的模型集成器。
        
        Args:
            models: 要集成的模型列表
            n_classes: 类别数量，默认为1（二值分割）
        """
        super(WeightedModelEnsemble, self).__init__(models, strategy='weighted_mean')
        self.n_classes = n_classes
        
        # 可学习的模型权重
        self.learnable_weights = nn.Parameter(torch.ones(len(models)) / len(models))
        
        # 类别级别的权重（可选）
        if n_classes > 1:
            self.class_weights = nn.Parameter(torch.ones(len(models), n_classes) / len(models))
        else:
            self.class_weights = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，使用可学习的权重进行集成。
        """
        # 获取所有模型的预测
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        
        # 应用可学习的权重
        normalized_weights = F.softmax(self.learnable_weights, dim=0)
        
        if self.class_weights is not None and self.n_classes > 1:
            # 类别级别的加权
            weighted_outputs = []
            for i, output in enumerate(outputs):
                # 对每个类别的权重进行softmax归一化
                class_norm_weights = F.softmax(self.class_weights[i], dim=0)
                weighted_output = output * class_norm_weights.view(1, -1, 1, 1)
                weighted_outputs.append(weighted_output * normalized_weights[i])
            return torch.sum(torch.stack(weighted_outputs), dim=0)
        else:
            # 模型级别的加权
            weighted_outputs = [output * weight for output, weight in zip(outputs, normalized_weights)]
            return torch.sum(torch.stack(weighted_outputs), dim=0)

class MultiHeadEnsemble(nn.Module):
    """
    多头集成模型，通过额外的融合层学习如何最好地组合多个模型的特征。
    """
    def __init__(self, models: List[nn.Module], input_channels: int, n_classes: int = 1):
        """
        初始化多头集成模型。
        
        Args:
            models: 要集成的模型列表
            input_channels: 输入通道数
            n_classes: 输出类别数，默认为1（二值分割）
        """
        super(MultiHeadEnsemble, self).__init__()
        self.models = nn.ModuleList(models)
        self.n_classes = n_classes
        
        # 获取每个模型的输出通道数
        # 使用一个测试输入来确定
        with torch.no_grad():
            dummy_input = torch.randn(1, input_channels, 64, 64)
            output_sizes = []
            for model in self.models:
                output = model(dummy_input)
                output_sizes.append(output.size(1))  # 获取通道数
        
        # 创建特征融合层
        total_channels = sum(output_sizes)
        
        # 融合网络 - 简单的卷积块
        self.fusion = nn.Sequential(
            nn.Conv2d(total_channels, total_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(total_channels // 2),
            nn.ReLU(),
            nn.Conv2d(total_channels // 2, n_classes, kernel_size=1)
        )
        
        # 初始化权重
        for m in self.fusion.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，通过融合层组合多个模型的输出特征。
        """
        # 获取所有模型的输出
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        
        # 确保所有输出的空间尺寸相同
        target_size = outputs[0].shape[2:]
        for i in range(1, len(outputs)):
            if outputs[i].shape[2:] != target_size:
                outputs[i] = F.interpolate(outputs[i], size=target_size, mode='bilinear', align_corners=False)
        
        # 沿着通道维度连接所有输出
        combined_features = torch.cat(outputs, dim=1)
        
        # 通过融合层
        result = self.fusion(combined_features)
        
        return result

def load_ensemble_models(model_paths: List[str], model_classes: List[nn.Module], device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> List[nn.Module]:
    """
    加载多个模型用于集成。
    
    Args:
        model_paths: 模型权重文件路径列表
        model_classes: 对应的模型类列表
        device: 设备类型
        
    Returns:
        加载好的模型列表
    """
    models = []
    
    if len(model_paths) != len(model_classes):
        raise ValueError("模型路径数量必须与模型类数量匹配")
    
    for path, model_class in zip(model_paths, model_classes):
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件不存在: {path}")
        
        # 初始化模型
        model = model_class()
        
        # 加载权重
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        
        # 处理不同格式的checkpoint
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        # 移至指定设备
        model.to(device)
        model.eval()
        
        models.append(model)
    
    return models

class StackingEnsemble(nn.Module):
    """
    堆叠集成模型，通过训练一个元模型来学习如何组合多个模型的输出。
    """
    def __init__(self, models: List[nn.Module], n_classes: int = 1, 
                 fusion_layers: int = 2, hidden_units: int = 64, 
                 dropout_rate: float = 0.2, use_batch_norm: bool = True):
        """
        初始化堆叠集成模型。
        
        Args:
            models: 要集成的模型列表
            n_classes: 输出类别数，默认为1（二值分割）
            fusion_layers: 融合层的数量
            hidden_units: 融合层中的隐藏单元数
            dropout_rate: 融合层的dropout率
            use_batch_norm: 是否使用批归一化
        """
        super(StackingEnsemble, self).__init__()
        self.models = nn.ModuleList(models)
        self.n_classes = n_classes
        
        # 获取每个模型的输出通道数
        with torch.no_grad():
            dummy_input = torch.randn(1, 6, 64, 64)  # 假设输入有6个通道
            output_sizes = []
            for model in self.models:
                output = model(dummy_input)
                output_sizes.append(output.size(1))  # 获取通道数
        
        # 创建堆叠融合网络
        total_channels = sum(output_sizes)
        
        # 构建融合层
        layers = []
        input_dim = total_channels
        
        for i in range(fusion_layers):
            layers.append(nn.Linear(input_dim, hidden_units))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_units
        
        # 输出层
        layers.append(nn.Linear(input_dim, n_classes))
        
        self.fusion_network = nn.Sequential(*layers)
        
        # 初始化权重
        for m in self.fusion_network.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，通过融合网络组合多个模型的输出。
        """
        # 获取所有模型的输出
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        
        # 确保所有输出的空间尺寸相同
        target_size = outputs[0].shape[2:]
        for i in range(1, len(outputs)):
            if outputs[i].shape[2:] != target_size:
                outputs[i] = F.interpolate(outputs[i], size=target_size, mode='bilinear', align_corners=False)
        
        # 全局平均池化将空间维度降为1
        pooled_outputs = []
        for output in outputs:
            # 如果是二值分割，应用sigmoid
            if output.shape[1] == 1:
                pooled = F.adaptive_avg_pool2d(torch.sigmoid(output), (1, 1))
            else:
                pooled = F.adaptive_avg_pool2d(torch.softmax(output, dim=1), (1, 1))
            pooled_outputs.append(pooled.view(pooled.size(0), -1))  # 展平
        
        # 沿着特征维度连接所有输出
        combined_features = torch.cat(pooled_outputs, dim=1)
        
        # 通过融合网络
        result = self.fusion_network(combined_features)
        
        # 如果是二值分割，返回单个logit
        if self.n_classes == 1:
            return result.view(-1, 1, 1, 1)  # 重塑为 [batch, 1, 1, 1]
        else:
            # 对于多类别分割，需要将结果扩展回空间维度
            # 这里我们简单地将结果复制到整个空间
            batch_size = x.size(0)
            spatial_size = target_size
            return result.view(batch_size, self.n_classes, 1, 1).expand(
                batch_size, self.n_classes, spatial_size[0], spatial_size[1]
            )

def create_water_segmentation_ensemble(aer_unet_model=None, lightweight_unet_model=None, deeplabv3_plus_model=None, strategy='mean'):
    """
    创建适用于水体分割的模型集成。
    支持AER U-Net、Lightweight U-Net和DeepLabV3+模型的任意组合。
    
    Args:
        aer_unet_model: 训练好的AER U-Net模型（可选）
        lightweight_unet_model: 训练好的Lightweight U-Net模型（可选）
        deeplabv3_plus_model: 训练好的DeepLabV3+模型（可选）
        strategy: 集成策略
        
    Returns:
        模型集成实例
    """
    models = []
    
    # 添加提供的模型
    if aer_unet_model is not None:
        models.append(aer_unet_model)
    if lightweight_unet_model is not None:
        models.append(lightweight_unet_model)
    if deeplabv3_plus_model is not None:
        models.append(deeplabv3_plus_model)
    
    # 确保至少提供了一个模型
    if len(models) == 0:
        raise ValueError("必须至少提供一个模型")
    
    return ModelEnsemble(
        models=models,
        strategy=strategy
    )

def compute_ensemble_metrics(predictions: List[torch.Tensor], targets: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """
    计算集成模型的性能指标。
    
    Args:
        predictions: 每个模型的预测结果列表
        targets: 真实标签
        threshold: 二值化阈值
        
    Returns:
        包含各种评估指标的字典
    """
    from .metrics import compute_iou, compute_dice, compute_precision_recall, compute_accuracy
    
    # 计算集成预测
    ensemble_pred = torch.mean(torch.stack(predictions), dim=0)
    
    # 应用阈值得到二值预测
    if ensemble_pred.shape[1] == 1:
        # 二值分割
        binary_pred = (torch.sigmoid(ensemble_pred) > threshold).float()
    else:
        # 多类别分割
        binary_pred = torch.argmax(torch.softmax(ensemble_pred, dim=1), dim=1, keepdim=True).float()
    
    # 计算各项指标
    metrics = {
        'iou': compute_iou(binary_pred, targets).item(),
        'dice': compute_dice(binary_pred, targets).item(),
    }
    
    # 计算精确率和召回率
    precision, recall = compute_precision_recall(binary_pred, targets)
    metrics['precision'] = precision.item()
    metrics['recall'] = recall.item()
    
    # 计算准确率
    metrics['accuracy'] = compute_accuracy(binary_pred, targets).item()
    
    return metrics

class AdvancedAdaptiveWeightedEnsemble(nn.Module):
    """
    高级自适应权重集成模型，根据输入图像的特征和模型输出动态调整各个模型的权重。
    该方法不仅考虑原始输入特征，还考虑各模型的中间特征，并为图像的不同区域生成不同的权重。
    """
    def __init__(self, models: List[nn.Module], n_classes: int = 1, 
                 input_channels: int = 6, hidden_units: int = 128, 
                 dropout_rate: float = 0.3, use_attention: bool = True):
        """
        初始化高级自适应权重集成模型。
        
        Args:
            models: 要集成的模型列表
            n_classes: 输出类别数，默认为1（二值分割）
            input_channels: 输入图像的通道数
            hidden_units: 权重生成网络的隐藏单元数
            dropout_rate: 权重生成网络的dropout率
            use_attention: 是否使用注意力机制
        """
        super(AdvancedAdaptiveWeightedEnsemble, self).__init__()
        self.models = nn.ModuleList(models)
        self.n_classes = n_classes
        self.num_models = len(models)
        self.use_attention = use_attention
        
        # 创建输入特征提取网络
        self.input_feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, hidden_units, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )
        
        # 创建权重生成网络
        # 该网络将根据输入图像的特征和模型输出生成每个模型的权重
        self.weight_generator = nn.Sequential(
            nn.Conv2d(hidden_units + self.num_models * n_classes, hidden_units, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(hidden_units, hidden_units // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units // 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(hidden_units // 2, self.num_models, kernel_size=1),
            nn.Sigmoid()  # 使用Sigmoid确保权重在[0,1]范围内
        )
        
        # 可选的注意力机制
        if use_attention:
            self.attention = nn.Sequential(
                nn.Conv2d(self.num_models, self.num_models, kernel_size=1),
                nn.Softmax(dim=1)
            )
        else:
            self.attention = None
        
        # 初始化权重
        for m in self.input_feature_extractor.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
        for m in self.weight_generator.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，根据输入图像的特征和模型输出动态调整各个模型的权重。
        """
        # 提取输入特征
        input_features = self.input_feature_extractor(x)
        
        # 获取所有模型的输出
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        
        # 确保所有输出的空间尺寸相同
        target_size = outputs[0].shape[2:]
        for i in range(1, len(outputs)):
            if outputs[i].shape[2:] != target_size:
                outputs[i] = F.interpolate(outputs[i], size=target_size, mode='bilinear', align_corners=False)
        
        # 将所有模型输出连接起来
        concatenated_outputs = torch.cat(outputs, dim=1)  # [batch_size, num_models * n_classes, height, width]
        
        # 将输入特征和模型输出连接起来
        combined_features = torch.cat([input_features, concatenated_outputs], dim=1)
        
        # 生成每个模型的权重图
        model_weights = self.weight_generator(combined_features)  # [batch_size, num_models, height, width]
        
        # 可选的注意力机制
        if self.attention is not None:
            # 对权重图应用注意力机制
            model_weights = self.attention(model_weights)
        else:
            # 对权重进行归一化，确保每个位置的权重和为1
            model_weights = F.softmax(model_weights, dim=1)
        
        # 应用自适应权重
        weighted_outputs = []
        for i, output in enumerate(outputs):
            weight = model_weights[:, i:i+1, :, :]  # [batch_size, 1, height, width]
            weighted_output = output * weight
            weighted_outputs.append(weighted_output)
        
        # 求和得到最终输出
        result = torch.sum(torch.stack(weighted_outputs), dim=0)
        
        return result

class AdaptiveWeightedEnsemble(nn.Module):
    """
    自适应权重集成模型，根据输入图像的特征动态调整各个模型的权重。
    这种方法可以让集成模型对不同类型的输入图像使用不同的模型组合策略。
    """
    def __init__(self, models: List[nn.Module], n_classes: int = 1, 
                 input_channels: int = 6, hidden_units: int = 64, 
                 dropout_rate: float = 0.2):
        """
        初始化自适应权重集成模型。
        
        Args:
            models: 要集成的模型列表
            n_classes: 输出类别数，默认为1（二值分割）
            input_channels: 输入图像的通道数
            hidden_units: 权重生成网络的隐藏单元数
            dropout_rate: 权重生成网络的dropout率
        """
        super(AdaptiveWeightedEnsemble, self).__init__()
        self.models = nn.ModuleList(models)
        self.n_classes = n_classes
        self.num_models = len(models)
        
        # 创建权重生成网络
        # 该网络将根据输入图像的特征生成每个模型的权重
        self.weight_generator = nn.Sequential(
            nn.Conv2d(input_channels, hidden_units, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(hidden_units, hidden_units // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units // 2),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(hidden_units // 2, self.num_models, kernel_size=1),
            nn.AdaptiveAvgPool2d(1)  # 全局平均池化得到每个模型的权重
        )
        
        # 初始化权重
        for m in self.weight_generator.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，根据输入图像的特征动态调整各个模型的权重。
        """
        # 生成每个模型的权重
        model_weights = self.weight_generator(x)  # [batch_size, num_models, 1, 1]
        model_weights = F.softmax(model_weights, dim=1)  # 对权重进行softmax归一化
        
        # 获取所有模型的输出
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        
        # 确保所有输出的空间尺寸相同
        target_size = outputs[0].shape[2:]
        for i in range(1, len(outputs)):
            if outputs[i].shape[2:] != target_size:
                outputs[i] = F.interpolate(outputs[i], size=target_size, mode='bilinear', align_corners=False)
        
        # 应用自适应权重
        weighted_outputs = []
        for i, output in enumerate(outputs):
            weight = model_weights[:, i:i+1, :, :]  # [batch_size, 1, 1, 1]
            weighted_output = output * weight
            weighted_outputs.append(weighted_output)
        
        # 求和得到最终输出
        result = torch.sum(torch.stack(weighted_outputs), dim=0)
        
        return result