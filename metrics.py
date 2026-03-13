
# utils/metrics.py
"""
评估指标工具函数，用于计算水体分割模型的性能。
包括IoU、Dice系数、精确率、召回率、准确率等指标。
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional

def compute_iou_no_threshold(prediction: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    计算IoU (Intersection over Union)指标，不进行二值化。
    用于阈值分析，预测值应该是已经二值化的。
    
    Args:
        prediction: 已经二值化的预测结果，形状为 [batch, 1, height, width]
        target: 真实标签，形状为 [batch, 1, height, width]
        smooth: 平滑项，避免除零错误
        
    Returns:
        平均IoU值
    """
    # 确保预测和目标在同一设备上
    device = prediction.device
    target = target.to(device)
    
    # 确保预测和目标是FP32类型
    prediction = prediction.float()
    target = target.float()
    
    # 确保预测和目标形状匹配
    if prediction.shape != target.shape:
        prediction = torch.nn.functional.interpolate(
            prediction, size=target.shape[2:], mode='bilinear', align_corners=False
        )
    
    # 展平张量以计算IoU
    prediction = prediction.view(-1)
    target = target.view(-1)
    
    # 计算交集和并集，使用FP32进行归约
    intersection = (prediction * target).sum(dtype=torch.float32)
    union = prediction.sum(dtype=torch.float32) + target.sum(dtype=torch.float32) - intersection
    
    # 计算IoU
    iou = (intersection + smooth) / (union + smooth)
    
    return iou

def compute_dice_no_threshold(prediction: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    计算Dice系数（F1分数的一种变体），不进行二值化。
    用于阈值分析，预测值应该是已经二值化的。
    
    Args:
        prediction: 已经二值化的预测结果
        target: 真实标签
        smooth: 平滑项，避免除零错误
        
    Returns:
        平均Dice系数
    """
    # 确保预测和目标在同一设备上
    device = prediction.device
    target = target.to(device)
    
    # 确保预测和目标是FP32类型
    prediction = prediction.float()
    target = target.float()
    
    # 确保预测和目标形状匹配
    if prediction.shape != target.shape:
        prediction = torch.nn.functional.interpolate(
            prediction, size=target.shape[2:], mode='bilinear', align_corners=False
        )
    
    # 展平张量以计算Dice系数
    prediction = prediction.view(-1)
    target = target.view(-1)
    
    # 计算Dice系数，使用FP32进行归约
    intersection = (prediction * target).sum(dtype=torch.float32)
    dice = (2. * intersection + smooth) / (prediction.sum(dtype=torch.float32) + target.sum(dtype=torch.float32) + smooth)
    
    return dice

# utils/metrics.py
"""
评估指标工具函数，用于计算水体分割模型的性能。
包括IoU、Dice系数、精确率、召回率、准确率等指标。
"""
def compute_iou(prediction: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6, threshold: float = 0.5) -> torch.Tensor:
    """
    计算IoU (Intersection over Union)指标。
    
    Args:
        prediction: 模型预测结果，形状为 [batch, 1, height, width]
        target: 真实标签，形状为 [batch, 1, height, width]
        smooth: 平滑项，避免除零错误
        threshold: 二值化阈值
        
    Returns:
        平均IoU值
    """
    # 确保预测和目标在同一设备上
    device = prediction.device
    target = target.to(device)
    
    # 确保预测和目标是FP32类型
    prediction = prediction.float()
    target = target.float()
    
    # 确保预测和目标形状匹配
    if prediction.shape != target.shape:
        prediction = torch.nn.functional.interpolate(
            prediction, size=target.shape[2:], mode='bilinear', align_corners=False
        )
    
    # 将预测转换为二值掩码
    if prediction.min() < 0 or prediction.max() > 1:
        prediction = torch.sigmoid(prediction)
    prediction = (prediction > threshold).float()
    
    # 确保目标掩码是二值化的（处理可能的255等值）
    target = (target > 0.5).float()
    
    # 展平张量以计算IoU
    prediction = prediction.view(-1)
    target = target.view(-1)
    
    # 计算交集和并集，使用FP32进行归约
    intersection = (prediction * target).sum(dtype=torch.float32)
    union = prediction.sum(dtype=torch.float32) + target.sum(dtype=torch.float32) - intersection
    
    # 计算IoU
    iou = (intersection + smooth) / (union + smooth)
    
    return iou

def compute_dice(prediction: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6, threshold: float = 0.5) -> torch.Tensor:
    """
    计算Dice系数（F1分数的一种变体）。
    
    Args:
        prediction: 模型预测结果
        target: 真实标签
        smooth: 平滑项，避免除零错误
        threshold: 二值化阈值
        
    Returns:
        平均Dice系数
    """
    # 确保预测和目标在同一设备上
    device = prediction.device
    target = target.to(device)
    
    # 确保预测和目标是FP32类型
    prediction = prediction.float()
    target = target.float()
    
    # 确保预测和目标形状匹配
    if prediction.shape != target.shape:
        prediction = torch.nn.functional.interpolate(
            prediction, size=target.shape[2:], mode='bilinear', align_corners=False
        )
    
    # 将预测转换为二值掩码
    if prediction.min() < 0 or prediction.max() > 1:
        prediction = torch.sigmoid(prediction)
    prediction = (prediction > threshold).float()
    
    # 确保目标掩码是二值化的（处理可能的255等值）
    target = (target > 0.5).float()
    
    # 展平张量以计算Dice系数
    prediction = prediction.view(-1)
    target = target.view(-1)
    
    # 计算Dice系数，使用FP32进行归约
    intersection = (prediction * target).sum(dtype=torch.float32)
    dice = (2. * intersection + smooth) / (prediction.sum(dtype=torch.float32) + target.sum(dtype=torch.float32) + smooth)
    
    return dice

def compute_precision_recall(prediction: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算精确率和召回率。
    
    Args:
        prediction: 模型预测结果（可以是logits或概率）
        target: 真实标签
        threshold: 二值化阈值
        
    Returns:
        精确率和召回率的元组
    """
    # 确保预测和目标在同一设备上
    device = prediction.device
    target = target.to(device)
    
    # 确保预测和目标是FP32类型
    prediction = prediction.float()
    target = target.float()
    
    # 确保预测和目标形状匹配
    if prediction.shape != target.shape:
        prediction = torch.nn.functional.interpolate(
            prediction, size=target.shape[2:], mode='bilinear', align_corners=False
        )
    
    # 将预测转换为概率（如果需要）
    if prediction.min() < 0 or prediction.max() > 1:
        prediction = torch.sigmoid(prediction)
    
    # 应用阈值进行二值化
    prediction = (prediction > threshold).float()
    target = (target > 0.5).float()
    
    # 展平张量
    prediction = prediction.view(-1)
    target = target.view(-1)
    
    # 计算True Positive, False Positive, False Negative，使用FP32进行归约
    tp = (prediction * target).sum(dtype=torch.float32)
    fp = (prediction * (1 - target)).sum(dtype=torch.float32)
    fn = ((1 - prediction) * target).sum(dtype=torch.float32)
    
    # 计算精确率和召回率，使用更稳定的epsilon值
    eps = 1e-7
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    
    # 确保返回值在合理范围内
    precision = torch.clamp(precision, 0.0, 1.0)
    recall = torch.clamp(recall, 0.0, 1.0)
    
    return precision, recall

def compute_accuracy(prediction: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """
    计算准确率。
    
    Args:
        prediction: 模型预测结果（可以是logits或概率）
        target: 真实标签
        threshold: 二值化阈值
        
    Returns:
        准确率
    """
    # 确保预测和目标在同一设备上
    device = prediction.device
    target = target.to(device)
    
    # 确保预测和目标是FP32类型
    prediction = prediction.float()
    target = target.float()
    
    # 确保预测和目标形状匹配
    if prediction.shape != target.shape:
        prediction = torch.nn.functional.interpolate(
            prediction, size=target.shape[2:], mode='bilinear', align_corners=False
        )
    
    # 将预测转换为概率（如果需要）
    if prediction.min() < 0 or prediction.max() > 1:
        prediction = torch.sigmoid(prediction)
    
    # 应用阈值进行二值化
    prediction = (prediction > threshold).float()
    target = (target > 0.5).float()
    
    # 展平张量
    prediction = prediction.view(-1)
    target = target.view(-1)
    
    # 计算准确率，使用FP32进行归约
    correct = (prediction == target).sum(dtype=torch.float32)
    accuracy = correct / prediction.numel()
    
    # 确保返回值在合理范围内
    accuracy = float(torch.clamp(accuracy, 0.0, 1.0))
    
    return accuracy

def compute_metrics_from_counts(tp: float, fp: float, fn: float, tn: float, eps: float = 1e-6) -> Dict[str, float]:
    """
    根据混淆矩阵的计数计算IoU、Dice、Precision、Recall、F1和Accuracy。
    """
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1_score = (2.0 * precision * recall) / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    accuracy = (tp + tn) / (tp + fp + fn + tn + eps)
    false_discovery_rate = fp / (fp + tp + eps)  # 误检率 (False Discovery Rate)
    false_negative_rate = fn / (fn + tp + eps)   # 漏检率 (False Negative Rate)

    return {
        'iou': float(iou),
        'dice': float(dice),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1_score),
        'accuracy': float(accuracy),
        'false_discovery_rate': float(false_discovery_rate),
        'false_negative_rate': float(false_negative_rate)
    }


def compute_classification_report_from_counts(tp: float, fp: float, fn: float, tn: float, eps: float = 1e-6) -> Dict[str, Dict[str, float]]:
    """
    根据混淆矩阵的计数构建与 sklearn classification_report 类似的结构。
    """
    # Class 1 (foreground)
    precision_1 = tp / (tp + fp + eps)
    recall_1 = tp / (tp + fn + eps)
    f1_1 = (2.0 * precision_1 * recall_1) / (precision_1 + recall_1 + eps)
    support_1 = tp + fn

    # Class 0 (background)
    tp_0 = tn
    fp_0 = fn
    fn_0 = fp
    precision_0 = tp_0 / (tp_0 + fp_0 + eps)
    recall_0 = tp_0 / (tp_0 + fn_0 + eps)
    f1_0 = (2.0 * precision_0 * recall_0) / (precision_0 + recall_0 + eps)
    support_0 = tp_0 + fn_0

    total_support = support_0 + support_1 + eps

    macro_precision = (precision_0 + precision_1) / 2.0
    macro_recall = (recall_0 + recall_1) / 2.0
    macro_f1 = (f1_0 + f1_1) / 2.0

    weighted_precision = (precision_0 * support_0 + precision_1 * support_1) / total_support
    weighted_recall = (recall_0 * support_0 + recall_1 * support_1) / total_support
    weighted_f1 = (f1_0 * support_0 + f1_1 * support_1) / total_support

    return {
        '0': {
            'precision': float(precision_0),
            'recall': float(recall_0),
            'f1-score': float(f1_0),
            'support': float(support_0)
        },
        '1': {
            'precision': float(precision_1),
            'recall': float(recall_1),
            'f1-score': float(f1_1),
            'support': float(support_1)
        },
        'macro avg': {
            'precision': float(macro_precision),
            'recall': float(macro_recall),
            'f1-score': float(macro_f1),
            'support': float(support_0 + support_1)
        },
        'weighted avg': {
            'precision': float(weighted_precision),
            'recall': float(weighted_recall),
            'f1-score': float(weighted_f1),
            'support': float(support_0 + support_1)
        }
    }


def compute_global_binary_metrics(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    在整个数据集上汇总计算二值分割相关指标。

    Args:
        predictions: 模型输出的概率或logits，形状为 [N, 1, H, W]
        targets: 真实标签，形状为 [N, 1, H, W]
        threshold: 二值化阈值
        eps: 数值稳定项

    Returns:
        metrics: 包含IoU、Dice、Precision、Recall、F1、Accuracy的字典
        counts: 包含TP、FP、FN、TN的字典
    """
    predictions = predictions.float()
    targets = targets.float()

    if predictions.shape != targets.shape:
        predictions = torch.nn.functional.interpolate(
            predictions, size=targets.shape[2:], mode='bilinear', align_corners=False
        )

    if predictions.min() < 0 or predictions.max() > 1:
        predictions = torch.sigmoid(predictions)
    predictions = predictions.clamp(0.0, 1.0)
    
    # 确保目标掩码是二值化的（处理可能的255等值）
    targets = (targets > 0.5).float()

    binary_preds = (predictions > threshold).float()

    tp = (binary_preds * targets).sum(dtype=torch.float64).item()
    fp = (binary_preds * (1.0 - targets)).sum(dtype=torch.float64).item()
    fn = ((1.0 - binary_preds) * targets).sum(dtype=torch.float64).item()
    tn = ((1.0 - binary_preds) * (1.0 - targets)).sum(dtype=torch.float64).item()

    metrics = compute_metrics_from_counts(tp, fp, fn, tn, eps=eps)

    counts = {
        'tp': float(tp),
        'fp': float(fp),
        'fn': float(fn),
        'tn': float(tn)
    }

    return metrics, counts

def compute_f1_score(precision: torch.Tensor, recall: torch.Tensor, smooth: float = 1e-6) -> float:
    """
    计算F1分数。
    
    Args:
        precision: 精确率
        recall: 召回率
        smooth: 平滑因子，防止除零
        
    Returns:
        F1分数
    """
    f1 = (2 * precision * recall) / (precision + recall + smooth)
    return f1.item()

def compute_confusion_matrix(prediction: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    """
    计算混淆矩阵。
    
    Args:
        prediction: 模型预测结果（二值化结果，0或1）
        target: 真实标签
        threshold: 二值化阈值（仅当输入为概率时使用）
        
    Returns:
        2x2的混淆矩阵
    """
    # 确保预测和目标在同一设备上
    device = prediction.device
    target = target.to(device)
    
    # 确保预测和目标是FP32类型
    prediction = prediction.float()
    target = target.float()
    
    # 确保预测和目标形状匹配
    if prediction.shape != target.shape:
        prediction = torch.nn.functional.interpolate(
            prediction, size=target.shape[2:], mode='bilinear', align_corners=False
        )
    
    # 如果预测不是二值化的，则进行二值化
    if prediction.min() >= 0 and prediction.max() <= 1 and not torch.all((prediction == 0) | (prediction == 1)):
        if prediction.min() < 0 or prediction.max() > 1:
            prediction = torch.sigmoid(prediction)
        prediction = (prediction > threshold).float()
    
    # 确保目标掩码是二值化的（处理可能的255等值）
    target = (target > 0.5).float()
    
    # 转换为numpy数组
    pred_np = prediction.cpu().detach().numpy().flatten()
    target_np = target.cpu().detach().numpy().flatten()
    
    # 计算混淆矩阵的各个元素
    tp = np.sum((pred_np == 1) & (target_np == 1))
    fp = np.sum((pred_np == 1) & (target_np == 0))
    fn = np.sum((pred_np == 0) & (target_np == 1))
    tn = np.sum((pred_np == 0) & (target_np == 0))
    
    # 构建混淆矩阵
    confusion_matrix = np.array([[tn, fp], [fn, tp]])
    
    return confusion_matrix

def compute_metrics_from_prob(prediction: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """
    从概率值计算所有评估指标并返回结果字典。
    与compute_metrics函数的区别在于，此函数假设输入已经是概率值，不需要进行sigmoid转换。
    
    Args:
        prediction: 模型预测结果（概率值，范围在[0,1]）
        target: 真实标签
        threshold: 二值化阈值
        
    Returns:
        包含所有评估指标的字典
    """
    # 确保预测和目标在同一设备上
    device = prediction.device
    target = target.to(device)
    
    # 确保预测和目标是FP32类型
    prediction = prediction.float()
    target = target.float()
    
    # 确保预测和目标形状匹配
    if prediction.shape != target.shape:
        prediction = torch.nn.functional.interpolate(
            prediction, size=target.shape[2:], mode='bilinear', align_corners=False
        )
    
    # 确保目标掩码是二值化的（处理可能的255等值）
    target = (target > 0.5).float()
    
    # 将预测转换为二值掩码（已经是概率值，直接应用阈值）
    binary_prediction = (prediction > threshold).float()
    
    # 展平张量以计算IoU
    binary_prediction_flat = binary_prediction.view(-1)
    target_flat = target.view(-1)
    
    # 计算IoU
    smooth = 1e-6
    intersection = (binary_prediction_flat * target_flat).sum(dtype=torch.float32)
    union = binary_prediction_flat.sum(dtype=torch.float32) + target_flat.sum(dtype=torch.float32) - intersection
    iou = (intersection + smooth) / (union + smooth)
    
    # 计算Dice系数
    dice = (2. * intersection + smooth) / (binary_prediction_flat.sum(dtype=torch.float32) + target_flat.sum(dtype=torch.float32) + smooth)
    
    # 计算混淆矩阵元素
    tp = (binary_prediction_flat * target_flat).sum(dtype=torch.float32)
    fp = (binary_prediction_flat * (1 - target_flat)).sum(dtype=torch.float32)
    fn = ((1 - binary_prediction_flat) * target_flat).sum(dtype=torch.float32)
    tn = ((1 - binary_prediction_flat) * (1 - target_flat)).sum(dtype=torch.float32)
    
    eps = 1e-7
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    
    # 计算F1分数
    f1 = (2 * precision * recall) / (precision + recall + smooth)
    
    # 计算准确率
    correct = (binary_prediction_flat == target_flat).sum(dtype=torch.float32)
    accuracy = correct / binary_prediction_flat.numel()
    
    # 计算误检率和漏检率
    false_discovery_rate = fp / (fp + tp + eps)  # 误检率 (False Discovery Rate)
    false_negative_rate = fn / (fn + tp + eps)   # 漏检率 (False Negative Rate)
    
    # 确保所有指标在合理范围内
    iou = float(torch.clamp(iou, 0.0, 1.0))
    dice = float(torch.clamp(dice, 0.0, 1.0))
    precision = float(torch.clamp(precision, 0.0, 1.0))
    recall = float(torch.clamp(recall, 0.0, 1.0))
    f1 = float(torch.clamp(f1, 0.0, 1.0))
    accuracy = float(torch.clamp(accuracy, 0.0, 1.0))
    false_discovery_rate = float(torch.clamp(false_discovery_rate, 0.0, 1.0))
    false_negative_rate = float(torch.clamp(false_negative_rate, 0.0, 1.0))
    
    return {
        'iou': iou,
        'dice': dice,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy,
        'false_discovery_rate': false_discovery_rate,
        'false_negative_rate': false_negative_rate
    }

def compute_metrics(prediction: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """
    计算所有评估指标并返回结果字典。
    
    Args:
        prediction: 模型预测结果
        target: 真实标签
        threshold: 二值化阈值
        
    Returns:
        包含所有评估指标的字典
    """
    # 确保预测和目标在同一设备上
    device = prediction.device
    target = target.to(device)
    
    # 确保预测和目标是FP32类型
    prediction = prediction.float()
    target = target.float()
    
    # 确保预测和目标形状匹配
    if prediction.shape != target.shape:
        prediction = torch.nn.functional.interpolate(
            prediction, size=target.shape[2:], mode='bilinear', align_corners=False
        )
    
    # 将预测转换为概率（如果需要）
    if prediction.min() < 0 or prediction.max() > 1:
        prediction = torch.sigmoid(prediction)
    
    # 应用阈值进行二值化
    pred_bin = (prediction > threshold).float()
    mask_bin = (target > 0.5).float()
    
    # 计算混淆矩阵元素
    tp = ((pred_bin == 1) & (mask_bin == 1)).sum()
    fp = ((pred_bin == 1) & (mask_bin == 0)).sum()
    fn = ((pred_bin == 0) & (mask_bin == 1)).sum()
    tn = ((pred_bin == 0) & (mask_bin == 0)).sum()
    
    # 计算IoU，传递threshold参数
    iou = compute_iou(prediction, target, threshold=threshold)
    
    # 计算Dice系数，传递threshold参数
    dice = compute_dice(prediction, target, threshold=threshold)
    
    # 计算精确率和召回率
    precision, recall = compute_precision_recall(prediction, target, threshold)
    
    # 计算F1分数
    f1 = compute_f1_score(precision, recall)
    
    # 计算准确率
    accuracy = compute_accuracy(prediction, target, threshold)
    
    # 计算误检率和漏检率
    eps = 1e-7
    false_discovery_rate = fp / (fp + tp + eps)  # 误检率 (False Discovery Rate)
    false_negative_rate = fn / (fn + tp + eps)   # 漏检率 (False Negative Rate)
    
    # 计算混淆矩阵
    confusion_mat = compute_confusion_matrix(prediction, target, threshold)
    
    # 构建结果字典
    metrics = {
        'iou': iou if isinstance(iou, float) else iou.item(),
        'dice': dice if isinstance(dice, float) else dice.item(),
        'precision': precision if isinstance(precision, float) else precision.item(),
        'recall': recall if isinstance(recall, float) else recall.item(),
        'f1_score': f1 if isinstance(f1, float) else f1.item(),
        'accuracy': accuracy if isinstance(accuracy, float) else accuracy.item(),
        'false_discovery_rate': false_discovery_rate if isinstance(false_discovery_rate, float) else false_discovery_rate.item(),
        'false_negative_rate': false_negative_rate if isinstance(false_negative_rate, float) else false_negative_rate.item(),
        'confusion_matrix': confusion_mat.tolist() if hasattr(confusion_mat, 'tolist') else confusion_mat
    }
    
    return metrics

def compute_classification_report(prediction: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, batch_size: int = 1000) -> Dict[str, float]:
    """
    计算分类报告，包含精确率、召回率和F1分数等指标。
    使用分批处理来减少内存使用。
    """
    # 确保预测和目标在同一设备上
    device = prediction.device
    target = target.to(device)
    
    # 确保预测和目标形状匹配
    if prediction.shape != target.shape:
        prediction = torch.nn.functional.interpolate(
            prediction, size=target.shape[2:], mode='bilinear', align_corners=False
        )

    # 确保预测是概率值
    if prediction.min() < 0 or prediction.max() > 1:
        prediction = torch.sigmoid(prediction)

    # 获取总样本数并确定批次大小
    total_samples = prediction.shape[0]
    batch_size = max(1, min(batch_size, total_samples))

    # 初始化计数器
    tp_0_total = 0.0
    fp_0_total = 0.0
    fn_0_total = 0.0
    tp_1_total = 0.0
    fp_1_total = 0.0
    fn_1_total = 0.0
    support_0_total = 0.0
    support_1_total = 0.0

    # 分批处理数据
    for i in range(0, total_samples, batch_size):
        end_idx = min(i + batch_size, total_samples)
        batch_preds = prediction[i:end_idx]
        batch_targets = target[i:end_idx]

        # 生成二值预测和目标（确保在相同设备上）
        binary_preds = (batch_preds > threshold).to(torch.float32)
        targets_bin = (batch_targets > 0.5).to(torch.float32)
        inv_preds = 1.0 - binary_preds
        inv_targets = 1.0 - targets_bin

        # 计算类别1的真阳性、假阳性和假阴性
        tp_1 = float(torch.sum(binary_preds * targets_bin))
        fp_1 = float(torch.sum(binary_preds * inv_targets))
        fn_1 = float(torch.sum(inv_preds * targets_bin))

        # 计算类别0的真阳性、假阳性和假阴性
        tp_0 = float(torch.sum(inv_preds * inv_targets))
        fp_0 = fn_1  # 类别0的假阳性是类别1的假阴性
        fn_0 = fp_1  # 类别0的假阴性是类别1的假阳性

        tp_0_total += tp_0
        fp_0_total += fp_0
        fn_0_total += fn_0
        tp_1_total += tp_1
        fp_1_total += fp_1
        fn_1_total += fn_1

        support_0_total += float(torch.sum(inv_targets))
        support_1_total += float(torch.sum(targets_bin))

    eps = 1e-6

    precision_0 = tp_0_total / (tp_0_total + fp_0_total + eps)
    recall_0 = tp_0_total / (tp_0_total + fn_0_total + eps)
    f1_0 = (2 * precision_0 * recall_0) / (precision_0 + recall_0 + eps)

    precision_1 = tp_1_total / (tp_1_total + fp_1_total + eps)
    recall_1 = tp_1_total / (tp_1_total + fn_1_total + eps)
    f1_1 = (2 * precision_1 * recall_1) / (precision_1 + recall_1 + eps)

    macro_precision = (precision_0 + precision_1) / 2
    macro_recall = (recall_0 + recall_1) / 2
    macro_f1 = (f1_0 + f1_1) / 2

    total_support = support_0_total + support_1_total
    safe_total_support = total_support if total_support > 0 else eps

    weighted_precision = (precision_0 * support_0_total + precision_1 * support_1_total) / safe_total_support
    weighted_recall = (recall_0 * support_0_total + recall_1 * support_1_total) / safe_total_support
    weighted_f1 = (f1_0 * support_0_total + f1_1 * support_1_total) / safe_total_support

    report = {
        '0': {
            'precision': precision_0,
            'recall': recall_0,
            'f1-score': f1_0,
            'support': support_0_total
        },
        '1': {
            'precision': precision_1,
            'recall': recall_1,
            'f1-score': f1_1,
            'support': support_1_total
        },
        'macro avg': {
            'precision': macro_precision,
            'recall': macro_recall,
            'f1-score': macro_f1,
            'support': total_support
        },
        'weighted avg': {
            'precision': weighted_precision,
            'recall': weighted_recall,
            'f1-score': weighted_f1,
            'support': total_support
        }
    }

    return report

def calculate_threshold_metrics(predictions: torch.Tensor, targets: torch.Tensor, thresholds: Optional[List[float]] = None, batch_size: int = 1000) -> Dict[str, List[float]]:
    """
    计算不同阈值下的性能指标，以帮助选择最佳阈值。
    使用分批处理来减少内存使用。
    
    Args:
        predictions: 模型预测结果（概率值，范围在0-1之间）
        targets: 真实标签
        thresholds: 要测试的阈值列表，如果为None，则使用默认阈值
        batch_size: 分批处理的大小，用于减少内存使用
        
    Returns:
        包含不同阈值下各项指标的字典
    """
    # 确保预测和目标在同一设备上
    device = predictions.device
    targets = targets.to(device)
    
    # 确保预测和目标是FP32类型
    predictions = predictions.float()
    targets = targets.float()
    
    # 设置默认阈值
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 9).tolist()
    
    # 初始化结果字典
    results = {
        'thresholds': thresholds,
        'iou': [],
        'dice': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'accuracy': []
    }
    
    # 确保预测是概率值（范围在0-1之间）
    if predictions.min() < 0 or predictions.max() > 1:
        prob_preds = torch.sigmoid(predictions)
    else:
        prob_preds = predictions
    prob_preds = prob_preds.clamp(0.0, 1.0)
    
    # 获取数据大小
    total_samples = prob_preds.shape[0]
    
    # 分批处理数据以减少内存使用
    for threshold in thresholds:
        # 初始化当前阈值的指标累加器
        iou_sum = 0.0
        dice_sum = 0.0
        precision_sum = 0.0
        recall_sum = 0.0
        f1_sum = 0.0
        accuracy_sum = 0.0
        valid_batches = 0
        
        # 分批处理
        for i in range(0, total_samples, batch_size):
            end_idx = min(i + batch_size, total_samples)
            batch_preds = prob_preds[i:end_idx]
            batch_targets = targets[i:end_idx]
            
            # 对概率进行二值化
            binary_preds = (batch_preds > threshold).float()
            
            try:
                # 使用不进行二值化的版本计算IoU和Dice
                iou = compute_iou_no_threshold(binary_preds, batch_targets)
                dice = compute_dice_no_threshold(binary_preds, batch_targets)
                
                # 计算精确率和召回率
                precision, recall = compute_precision_recall(binary_preds, batch_targets, threshold=0.5)  # 使用固定阈值0.5，因为输入已经是二值化的
                f1 = compute_f1_score(precision, recall)
                accuracy = compute_accuracy(binary_preds, batch_targets, threshold=0.5)  # 使用固定阈值0.5，因为输入已经是二值化的
                
                # 累加指标
                iou_sum += iou.item() if hasattr(iou, 'item') else iou
                dice_sum += dice.item() if hasattr(dice, 'item') else dice
                precision_sum += precision.item() if hasattr(precision, 'item') else precision
                recall_sum += recall.item() if hasattr(recall, 'item') else recall
                f1_sum += f1 if isinstance(f1, float) else f1.item()
                accuracy_sum += accuracy if isinstance(accuracy, float) else accuracy.item()
                
                valid_batches += 1
            except Exception as e:
                # 如果当前批次处理失败，跳过并继续
                print(f"处理批次 {i}-{end_idx} 时出错: {e}")
                continue
        
        # 计算当前阈值的平均指标
        if valid_batches > 0:
            results['iou'].append(iou_sum / valid_batches)
            results['dice'].append(dice_sum / valid_batches)
            results['precision'].append(precision_sum / valid_batches)
            results['recall'].append(recall_sum / valid_batches)
            results['f1_score'].append(f1_sum / valid_batches)
            results['accuracy'].append(accuracy_sum / valid_batches)
        else:
            # 如果没有有效批次，添加0值
            results['iou'].append(0.0)
            results['dice'].append(0.0)
            results['precision'].append(0.0)
            results['recall'].append(0.0)
            results['f1_score'].append(0.0)
            results['accuracy'].append(0.0)
    
    # 找到性能最佳的阈值
    best_f1_index = np.argmax(results['f1_score'])
    best_threshold = thresholds[best_f1_index]
    
    results['best_threshold'] = best_threshold
    results['best_f1_score'] = results['f1_score'][best_f1_index]
    
    return results
