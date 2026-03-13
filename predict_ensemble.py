#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型集成预测脚本
集成两个最好的模型（AER U-Net和Ultra-Light DeepLabV3+）进行预测
"""
import os
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# 导入自定义模块
from models.aer_unet import get_aer_unet_model
from models.ultra_lightweight_deeplabv3_plus import get_ultra_light_deeplabv3_plus
from utils.ensemble_utils import ModelEnsemble, load_ensemble_models, StackingEnsemble, AdaptiveWeightedEnsemble, AdvancedAdaptiveWeightedEnsemble
from utils.performance_weighted_ensemble import PerformanceWeightedEnsemble, create_performance_weighted_ensemble
from utils.improved_performance_weighted_ensemble import create_improved_performance_weighted_ensemble
from utils.data_utils import normalize_image, load_sentinel2_image, save_predictions
from config import get_config

# 导入CRF工具函数
from utils.crf_utils import apply_crf_postprocessing

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ensemble_prediction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_ensemble_models(model_paths, model_classes, weights=[0.4, 0.6], strategy='weighted_mean', device='cuda', 
                         stacking_config=None, adaptive_config=None, advanced_adaptive_config=None,
                         performance_weighted_config=None, improved_performance_weighted_config=None, binary_threshold=0.5):
    """
    设置集成模型
    
    Args:
        model_paths: 模型权重文件路径列表
        model_classes: 对应的模型类列表
        weights: 模型权重列表
        strategy: 集成策略
        device: 设备类型
        stacking_config: 堆叠集成配置（可选）
        adaptive_config: 自适应权重集成配置（可选）
        advanced_adaptive_config: 高级自适应权重集成配置（可选）
        performance_weighted_config: 基于性能的加权集成配置（可选）
        improved_performance_weighted_config: 改进的基于性能的加权集成配置（可选）
        binary_threshold: 二值化阈值，用于置信度计算
        
    Returns:
        集成模型实例
    """
    logger.info("加载集成模型...")
    
    # 根据策略创建不同类型的集成模型
    if strategy == 'stacking':
        if stacking_config is None:
            stacking_config = {
                'fusion_layers': 3,
                'hidden_units': 128,
                'dropout_rate': 0.3,
                'use_batch_norm': True
            }
        
        logger.info(f"使用堆叠集成策略，参数: {stacking_config}")
        
        # 加载模型
        models = load_ensemble_models(model_paths, model_classes, device=device)
        
        ensemble = StackingEnsemble(
            models=models,
            n_classes=1,  # 二值分割
            **stacking_config
        )
    elif strategy == 'adaptive':
        if adaptive_config is None:
            adaptive_config = {
                'input_channels': 6,
                'hidden_units': 64,
                'dropout_rate': 0.2
            }
        
        logger.info(f"使用自适应权重集成策略，参数: {adaptive_config}")
        
        # 加载模型
        models = load_ensemble_models(model_paths, model_classes, device=device)
        
        ensemble = AdaptiveWeightedEnsemble(
            models=models,
            n_classes=1,  # 二值分割
            **adaptive_config
        )
    elif strategy == 'advanced_adaptive':
        if advanced_adaptive_config is None:
            advanced_adaptive_config = {
                'input_channels': 6,
                'hidden_units': 128,
                'dropout_rate': 0.3,
                'use_attention': True
            }
        
        logger.info(f"使用高级自适应权重集成策略，参数: {advanced_adaptive_config}")
        
        # 加载模型
        models = load_ensemble_models(model_paths, model_classes, device=device)
        
        ensemble = AdvancedAdaptiveWeightedEnsemble(
            models=models,
            n_classes=1,  # 二值分割
            **advanced_adaptive_config
        )
    elif strategy == 'performance_weighted':
        if performance_weighted_config is None:
            performance_weighted_config = {
                'model_names': ['aer_unet', 'ultra_lightweight_deeplabv3_plus'],
                'csv_paths': [
                    'predictions/examples/aer_unet/evaluation_results_aer_unet_20251022_144056.csv',
                    'predictions/examples/ultra_lightweight_deeplabv3_plus/evaluation_results_ultra_lightweight_deeplabv3_plus_20251022_143831.csv'
                ],
                'metric_name': 'iou',
                'temperature': 2.0
            }
        
        logger.info(f"使用基于性能的加权集成策略，参数: {performance_weighted_config}")
        
        # 创建基于性能的加权集成模型
        ensemble = create_performance_weighted_ensemble(
            model_paths=model_paths,
            model_classes=model_classes,
            model_names=performance_weighted_config['model_names'],
            csv_paths=performance_weighted_config['csv_paths'],
            metric_name=performance_weighted_config['metric_name'],
            temperature=performance_weighted_config['temperature'],
            device=device
        )
    elif strategy == 'improved_performance_weighted':
        if improved_performance_weighted_config is None:
            improved_performance_weighted_config = {
                'model_names': ['aer_unet', 'ultra_lightweight_deeplabv3_plus'],
                'csv_paths': [
                    'predictions/examples/aer_unet/evaluation_results_aer_unet_20251022_144056.csv',
                    'predictions/examples/ultra_lightweight_deeplabv3_plus/evaluation_results_ultra_lightweight_deeplabv3_plus_20251022_143831.csv'
                ],
                'metric_name': 'iou',
                'temperature': 1.0,
                'ensemble_method': 'gated_ensemble',
                'diff_threshold': 0.20,
                'conf_threshold': 0.22
            }
        
        logger.info(f"使用改进的基于性能的加权集成策略，参数: {improved_performance_weighted_config}")
        
        # 创建改进的基于性能的加权集成模型
        # 直接调用模型类函数，而不是传递函数本身
        models = []
        for i, (model_path, model_class_func) in enumerate(zip(model_paths, model_classes)):
            # 调用函数获取模型实例
            model = model_class_func()
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
                
            # 在加载checkpoint前统一做键名映射
            # 根据模型名称而不是索引来判断
            model_name = improved_performance_weighted_config['model_names'][i]
            if model_name == 'ultra_lightweight_deeplabv3_plus':
                # 处理权重命名不一致问题：确保low_proj和low_reduce两个前缀的键都存在
                # 首先检查哪些键存在
                has_low_proj = any(k.startswith('low_proj.') for k in state_dict.keys())
                has_low_reduce = any(k.startswith('low_reduce.') for k in state_dict.keys())
                
                # 如果只有low_proj键，复制为low_reduce键
                if has_low_proj and not has_low_reduce:
                    for k, v in list(state_dict.items()):
                        if k.startswith('low_proj.'):
                            new_key = k.replace('low_proj.', 'low_reduce.')
                            state_dict[new_key] = v
                # 如果只有low_reduce键，复制为low_proj键
                elif has_low_reduce and not has_low_proj:
                    for k, v in list(state_dict.items()):
                        if k.startswith('low_reduce.'):
                            new_key = k.replace('low_reduce.', 'low_proj.')
                            state_dict[new_key] = v
            
            # 使用strict=True加载，因为我们已经处理了键名映射
            missing, unexpected = model.load_state_dict(state_dict, strict=True)
            print(f"[Load] {model_name} loaded with strict=True | missing: {len(missing)} | unexpected: {len(unexpected)}")
            
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
        
        # 加载性能指标
        from utils.improved_performance_weighted_ensemble import load_performance_metrics
        metric_weights = improved_performance_weighted_config.get('metric_weights')
        if isinstance(metric_weights, dict):
            # normalize possible key name for f1
            if 'f1_score' in metric_weights and 'f1' not in metric_weights:
                metric_weights['f1'] = metric_weights.pop('f1_score')
        performance_metrics = load_performance_metrics(
            improved_performance_weighted_config['csv_paths'], 
            improved_performance_weighted_config['model_names'],
            improved_performance_weighted_config['metric_name']
        )
        
        # 创建集成模型
        from utils.improved_performance_weighted_ensemble import ImprovedPerformanceWeightedEnsemble
        ensemble = ImprovedPerformanceWeightedEnsemble(
            models=models,
            performance_metrics=performance_metrics,
            metric_weights=metric_weights,
            metric_name=improved_performance_weighted_config['metric_name'],
            temperature=improved_performance_weighted_config['temperature'],
            ensemble_method=improved_performance_weighted_config['ensemble_method'],
            diff_threshold=improved_performance_weighted_config.get('diff_threshold', 0.20),
            conf_threshold=improved_performance_weighted_config.get('conf_threshold', 0.22),
            binary_threshold=binary_threshold,  # 使用函数参数中的binary_threshold,
            model_names=improved_performance_weighted_config.get('model_names')
        )
    else:
        # 创建传统集成模型，使用指定策略和权重
        logger.info(f"使用传统集成策略: {strategy}")
        
        # 加载模型
        models = load_ensemble_models(model_paths, model_classes, device=device)
        
        ensemble = ModelEnsemble(
            models=models,
            strategy=strategy,
            weights=weights
        )
    
    return ensemble.to(device)

def predict_with_ensemble(model, image_path, args, image_index=None):
    """
    使用集成模型进行预测
    
    Args:
        model: 集成模型
        image_path: 图像文件路径
        args: 参数
        image_index: 图像索引，用于限制可视化数量
        
    Returns:
        预测结果路径
    """
    logger.info(f'预测图像: {image_path}')
    
    # 加载图像
    image, profile = load_sentinel2_image(image_path, args.bands)
    
    # 图像归一化
    normalized_image = normalize_image(image, method='sentinel')
    
    # 获取图像尺寸
    channels, height, width = normalized_image.shape
    
    # 检查是否需要分块处理
    if height <= args.tile_size and width <= args.tile_size:
        # 直接处理整幅图像
        logger.info('图像尺寸合适，直接处理整幅图像')
        prediction = predict_on_tile(model, normalized_image, args.device)
    else:
        # 分块处理
        logger.info(f'图像尺寸较大，分块处理 ({height}x{width})')
        prediction = process_large_image(model, normalized_image, args)
    
    # 应用CRF后处理
    if args.crf_iterations > 0:
        logger.info(f'应用CRF后处理，迭代次数: {args.crf_iterations}')
        
        # 将预测结果转换为tensor
        pred_tensor = torch.from_numpy(prediction).float().to(args.device)
        # 将图像数据转换为tensor并归一化到[0,1]范围
        img_tensor = torch.from_numpy(normalized_image).float().to(args.device)
        
        # 应用CRF
        with torch.no_grad():
            refined_pred = apply_crf_postprocessing(
                img_tensor, pred_tensor, args.crf_iterations
            )
        
        # 转换回numpy
        prediction = refined_pred.cpu().numpy()
    
    # 应用阈值进行二值化
    binary_prediction = (prediction > args.threshold).astype(np.float32)
    
    # 获取文件名
    filename = os.path.basename(image_path)
    
    # 保存预测结果（根据save_predictions参数控制）
    prediction_path = None
    if args.save_predictions:
        prediction_path = save_prediction_to_geotiff(binary_prediction, profile, args.output_dir, filename)
    
    # 如果需要，保存预测概率图（根据max_visualizations参数限制数量）
    if args.save_visualization:
        # 检查是否设置了最大可视化数量限制
        if hasattr(args, 'max_visualizations') and args.max_visualizations is not None and image_index is not None:
            if image_index < args.max_visualizations:
                visualize_prediction(normalized_image, binary_prediction, prediction, args.output_dir, filename)
                logger.info(f'保存可视化结果 ({image_index+1}/{args.max_visualizations})')
            else:
                logger.info(f'跳过可视化结果 (已达到最大数量 {args.max_visualizations})')
        else:
            # 如果没有设置限制，则保存所有可视化结果
            visualize_prediction(normalized_image, binary_prediction, prediction, args.output_dir, filename)
    
    return prediction_path

def predict_on_tile(model, tile, device):
    """在单个瓦片上进行预测"""
    # 将瓦片转换为tensor并添加批次维度
    tile_tensor = torch.from_numpy(tile).float().to(device)
    tile_tensor = tile_tensor.unsqueeze(0)
    
    # 进行预测
    with torch.no_grad():
        output = model(tile_tensor)
        # 仅当输出还在logit空间时才应用sigmoid转换为概率
        if output.min() < 0 or output.max() > 1:
            output = torch.sigmoid(output)
    
    # 移除批次维度并转换回numpy数组
    output = output.squeeze(0).cpu().numpy()
    
    return output

def process_large_image(model, image, args):
    """分块处理大图像"""
    # 获取图像尺寸和批次大小
    channels, height, width = image.shape
    tile_size = args.tile_size
    overlap = args.overlap
    batch_size = args.batch_size
    
    # 计算分块数量
    stride = tile_size - overlap
    num_tiles_h = (height - overlap + stride - 1) // stride
    num_tiles_w = (width - overlap + stride - 1) // stride
    
    logger.info(f'分块: {num_tiles_h}x{num_tiles_w} 瓦片')
    
    # 初始化结果数组
    prediction = np.zeros((1, height, width), dtype=np.float32)
    weight_map = np.zeros((height, width), dtype=np.float32)
    
    # 准备所有瓦片
    tiles = []
    positions = []
    
    for i in range(num_tiles_h):
        for j in range(num_tiles_w):
            # 计算瓦片的起始和结束坐标
            start_h = min(i * stride, height - tile_size)
            end_h = min(start_h + tile_size, height)
            start_w = min(j * stride, width - tile_size)
            end_w = min(start_w + tile_size, width)
            
            # 提取瓦片
            tile = image[:, start_h:end_h, start_w:end_w]
            tiles.append(tile)
            positions.append((start_h, end_h, start_w, end_w))
    
    # 批量处理瓦片
    for i in tqdm(range(0, len(tiles), batch_size), desc='处理瓦片'):
        # 获取当前批次的瓦片
        batch_tiles = tiles[i:i+batch_size]
        batch_positions = positions[i:i+batch_size]
        
        # 将瓦片转换为tensor并添加批次维度
        batch_tensor = torch.from_numpy(np.array(batch_tiles)).float().to(args.device)
        
        # 进行预测
        with torch.no_grad():
            batch_output = model(batch_tensor)
            # 仅当输出还在logit空间时才应用sigmoid转换为概率
            if batch_output.min() < 0 or batch_output.max() > 1:
                batch_output = torch.sigmoid(batch_output)
        
        # 处理预测结果
        for j in range(len(batch_tiles)):
            # 获取当前瓦片的预测结果
            output = batch_output[j].cpu().numpy()
            
            # 获取瓦片在原图中的位置
            start_h, end_h, start_w, end_w = batch_positions[j]
            
            # 计算当前瓦片的权重（用于重叠区域的加权平均）
            tile_h, tile_w = end_h - start_h, end_w - start_w
            weight = create_weight_map(tile_h, tile_w, overlap)
            
            # 将预测结果添加到总结果中（使用加权平均处理重叠区域）
            prediction[0, start_h:end_h, start_w:end_w] += output[0] * weight
            weight_map[start_h:end_h, start_w:end_w] += weight
    
    # 归一化预测结果
    mask = weight_map > 0
    prediction[0, mask] /= weight_map[mask]
    
    return prediction

def create_weight_map(height, width, overlap):
    """创建用于重叠区域加权平均的权重图"""
    # 创建线性权重图
    weight_h = np.ones((height, 1), dtype=np.float32)
    weight_w = np.ones((1, width), dtype=np.float32)
    
    # 处理垂直方向的边界
    if overlap > 0:
        # 顶部边界权重
        top_weight = np.linspace(0, 1, overlap, dtype=np.float32)[:, np.newaxis]
        weight_h[:overlap] = top_weight
        
        # 底部边界权重
        bottom_weight = np.linspace(1, 0, overlap, dtype=np.float32)[:, np.newaxis]
        weight_h[-overlap:] = bottom_weight
    
    # 处理水平方向的边界
    if overlap > 0:
        # 左侧边界权重
        left_weight = np.linspace(0, 1, overlap, dtype=np.float32)[np.newaxis, :]
        weight_w[:, :overlap] = left_weight
        
        # 右侧边界权重
        right_weight = np.linspace(1, 0, overlap, dtype=np.float32)[np.newaxis, :]
        weight_w[:, -overlap:] = right_weight
    
    # 计算最终权重（两个方向的乘积）
    weight = weight_h * weight_w
    
    return weight

def save_prediction_to_geotiff(prediction, profile, output_dir, filename):
    """保存预测结果为GeoTIFF文件"""
    import rasterio
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建输出文件路径
    base_name = os.path.splitext(filename)[0]
    output_path = os.path.join(output_dir, f'{base_name}_prediction.tif')
    
    # 确保预测结果形状正确 [height, width]
    if prediction.ndim == 4:  # 形状为 [batch, channels, height, width]
        prediction = prediction.squeeze(0).squeeze(0)  # 移除批次和通道维度
    elif prediction.ndim == 3:  # 形状为 [channels, height, width]
        prediction = prediction.squeeze(0)  # 移除通道维度
    
    # 更新profile
    profile.update(
        dtype=rasterio.float32,
        count=1,
        compress='lzw'
    )
    
    # 保存预测结果
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(prediction.astype(np.float32), 1)
    
    logger.info(f'预测结果已保存到: {output_path}')
    
    return output_path

def visualize_prediction(image, binary_prediction, probability_prediction, output_dir, filename):
    """可视化预测结果"""
    import matplotlib.pyplot as plt
    
    # 确保中文显示正常
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    
    # 创建可视化输出目录
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # 构建输出文件路径
    base_name = os.path.splitext(filename)[0]
    viz_path = os.path.join(viz_dir, f'{base_name}_visualization.png')
    
    # 创建画布
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 绘制RGB图像
    if image.shape[0] >= 3:
        # 重组波段以显示RGB图像（假设波段顺序是蓝、绿、红或其他）
        rgb_image = np.zeros((image.shape[1], image.shape[2], 3))
        
        # 根据实际波段顺序调整
        # 这里假设波段顺序是：蓝、绿、红、近红、短波红外1、短波红外2
        if image.shape[0] >= 3:
            rgb_image[:, :, 0] = image[2]  # 红波段
            rgb_image[:, :, 1] = image[1]  # 绿波段
            rgb_image[:, :, 2] = image[0]  # 蓝波段
        
        # 归一化到[0, 1]范围以便显示
        rgb_min = rgb_image.min()
        rgb_max = rgb_image.max()
        if rgb_max - rgb_min > 0:
            rgb_image = (rgb_image - rgb_min) / (rgb_max - rgb_min)
        
        axes[0].imshow(rgb_image)
    else:
        # 单波段图像
        axes[0].imshow(image[0], cmap='gray')
    axes[0].set_title('输入图像')
    axes[0].axis('off')
    
    # 绘制预测概率图
    if probability_prediction.shape[0] == 1:
        im = axes[1].imshow(probability_prediction[0], cmap='jet')
    else:
        # 多类别情况，显示概率最高的类别
        im = axes[1].imshow(np.argmax(probability_prediction, axis=0), cmap='jet')
    axes[1].set_title('预测概率')
    axes[1].axis('off')
    fig.colorbar(im, ax=axes[1])
    
    # 绘制二值预测结果
    axes[2].imshow(binary_prediction[0], cmap='Blues')
    axes[2].set_title('二值预测结果')
    axes[2].axis('off')
    
    # 添加文件名
    plt.suptitle(f'文件: {filename}', fontsize=16)
    plt.tight_layout()
    
    # 保存可视化结果
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    logger.info(f'可视化结果已保存到: {viz_path}')
    
    plt.close(fig)  # 关闭图形以释放内存

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Sentinel-2影像水体分割集成预测')
    
    # 配置文件参数
    parser.add_argument('--config', type=str, default=None,
                        help='配置文件路径')
    
    # 输入/输出参数
    parser.add_argument('--input_dir', type=str, default='data/Images',
                        help='????????')
    parser.add_argument('--input_file', type=str, default=None,
                        help='??????????????input_dir?')
    parser.add_argument('--output_dir', type=str, default='poyanghu',
                        help='预测结果保存目录')
    parser.add_argument('--file_pattern', type=str, default='*.tif',
                        help='输入影像文件匹配模式（仅在输入为目录时使用）')
    parser.add_argument('--bands', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6],
                        help='要使用的波段索引列表')
    
    # 模型参数
    parser.add_argument('--aer_unet_path', type=str, 
                        default='checkpoints/aer_unet_1/aer_unet_best.pth',
                        help='AER U-Net模型检查点文件路径')
    parser.add_argument('--deeplab_path', type=str, 
                        default='checkpoints/ultra_lightweight_deeplabv3_plus_3/ultra_lightweight_deeplabv3_plus_best.pth',
                        help='DeepLabV3+模型检查点文件路径')
    parser.add_argument('--aer_unet_weight', type=float, default=0.4,
                        help='AER U-Net模型在集成中的权重')
    parser.add_argument('--deeplab_weight', type=float, default=0.6,
                        help='DeepLabV3+模型在集成中的权重')
    parser.add_argument('--ensemble_strategy', type=str, default='weighted_mean',
                        choices=['mean', 'weighted_mean', 'vote', 'logits_mean', 'stacking', 'adaptive', 'advanced_adaptive', 'performance_weighted', 'improved_performance_weighted'],
                        help='集成策略')
    
    # 预测参数
    parser.add_argument('--threshold', type=float, default=0.7,
                        help='二值化阈值')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='批量大小')
    parser.add_argument('--tile_size', type=int, default=256,
                        help='滑窗推理的瓦片大小')
    parser.add_argument('--overlap', type=int, default=64,
                        help='处理大影像时的瓦片重叠大小')
    
    # 其他参数
    parser.add_argument('--crf_iterations', type=int, default=10,
                        help='CRF迭代次数，0表示不使用CRF')
    parser.add_argument('--save_visualization', action='store_true',
                        help='保存可视化结果')
    parser.add_argument('--save_predictions', action='store_true', default=True,
                        help='保存预测结果(.tif文件)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='预测设备')
    
    args = parser.parse_args()
    
    # 设置默认的配置占位，避免未提供config时报未定义变量
    config = {}
    models_config = []
    
    # 如果提供了配置文件，加载配置
    if args.config:
        config = get_config(yaml_config_path=args.config)
        
        # 从配置文件中更新参数
        if 'predict' in config:
            predict_config = config['predict']
            args.threshold = predict_config.get('threshold', args.threshold)
            args.tile_size = predict_config.get('tile_size', args.tile_size)
            args.overlap = predict_config.get('overlap', args.overlap)
            args.batch_size = predict_config.get('batch_size', args.batch_size)
            args.crf_iterations = predict_config.get('crf_iterations', args.crf_iterations)
            args.save_visualization = predict_config.get('save_visualization', args.save_visualization)
            args.save_predictions = predict_config.get('save_predictions', args.save_predictions)
            args.bands = predict_config.get('bands', args.bands)
            args.max_visualizations = predict_config.get('max_visualizations', None)  # 添加最大可视化图像数量参数
            
            # 集成策略参数
            args.ensemble_strategy = predict_config.get('ensemble_strategy', args.ensemble_strategy)
            
            # 从模型配置中更新参数
            if 'models' in config:
                models_config = config['models']
                if len(models_config) >= 2:
                    args.aer_unet_path = models_config[0].get('checkpoint_path', args.aer_unet_path)
                    args.deeplab_path = models_config[1].get('checkpoint_path', args.deeplab_path)
                    args.aer_unet_weight = models_config[0].get('ens_weight', args.aer_unet_weight)
                    args.deeplab_weight = models_config[1].get('ens_weight', args.deeplab_weight)
    
    # 从配置文件中获取输入路径这是那个概率图
    if args.config and 'data' in config:
        args.input_dir = config['data'].get('input_dir', args.input_dir)
        # 获取测试集文件列表（如果存在）
        args.test_list = config['data'].get('test_list', None)
    
    # 在这里直接修改影像路径（已注释，使用配置文件中的路径）
    # 如果要使用单个影像文件，请取消下面的注释并修改路径
    # args.input_dir = "F:\\deeplearning\\sentinel2_water_segmentation\\datasets\\test\\images\\sentinel12_s2_1_img_17.tif"
    
    # 如果要使用目录中的所有影像，请取消下面的注释并修改路径
    # args.input_dir = "data\\Images"
    
    # 检查输入是文件还是目录
    if args.input_file:
        if not os.path.exists(args.input_file):
            raise ValueError(f"???????: {args.input_file}")
        image_files = [args.input_file]
        logger.info(f'????????: {args.input_file}')
    elif os.path.isfile(args.input_dir):
        # 输入是单个文件
        image_files = [args.input_dir]
        logger.info(f'处理单个影像文件: {args.input_dir}')
    elif os.path.isdir(args.input_dir):
        # 输入是目录，收集所有匹配的影像文件
        image_files = []
        
        # 检查是否有测试集文件列表
        if hasattr(args, 'test_list') and args.test_list is not None and os.path.exists(args.test_list):
            # 从测试集文件列表中读取文件名
            logger.info(f'从测试集文件列表中读取: {args.test_list}')
            with open(args.test_list, 'r') as f:
                test_filenames = [line.strip() for line in f.readlines() if line.strip()]
            
            # 构建完整的文件路径
            for filename in test_filenames:
                # 检查文件名是否已经包含扩展名
                if filename.lower().endswith(('.tif', '.tiff')):
                    # 如果文件名包含扩展名，先移除扩展名
                    base_filename = filename[:-4] if filename.lower().endswith('.tif') else filename[:-5]
                    # 将掩码文件名转换为图像文件名（将msk替换为img）
                    if '_msk_' in base_filename:
                        img_filename = base_filename.replace('_msk_', '_img_')
                        full_path = os.path.join(args.input_dir, img_filename + '.tif')
                    else:
                        full_path = os.path.join(args.input_dir, filename)
                else:
                    # 如果没有扩展名，先将掩码文件名转换为图像文件名（将msk替换为img），然后添加.tif
                    if '_msk_' in filename:
                        img_filename = filename.replace('_msk_', '_img_')
                        full_path = os.path.join(args.input_dir, img_filename + '.tif')
                    else:
                        full_path = os.path.join(args.input_dir, filename + '.tif')
                
                if os.path.exists(full_path):
                    image_files.append(full_path)
                else:
                    logger.warning(f'测试集文件不存在: {full_path}')
            
            logger.info(f'从测试集文件列表中找到 {len(image_files)} 个影像文件')
        else:
            # 如果没有测试集文件列表，收集目录中所有匹配的影像文件
            logger.info(f'在目录 {args.input_dir} 中搜索所有影像文件')
            try:
                from fnmatch import fnmatch
                pattern = args.file_pattern or '*.tif'
            except Exception:
                pattern = '*.tif'
            for root, _, files in os.walk(args.input_dir):
                for file in files:
                    if not file.lower().endswith(('.tif', '.tiff')):
                        continue
                    if not fnmatch(file, pattern):
                        continue
                        image_files.append(os.path.join(root, file))
            logger.info(f'在目录 {args.input_dir} 中找到 {len(image_files)} 个影像文件')
    else:
        raise ValueError(f"输入路径不存在或不是有效的文件/目录: {args.input_dir}")
    
    # 验证是否找到了图像文件
    if len(image_files) == 0:
        raise ValueError(f"找不到匹配的影像文件: {args.input_dir}")
    
    # 检查模型文件是否存在
    if not os.path.exists(args.aer_unet_path):
        raise ValueError(f"AER U-Net模型文件不存在: {args.aer_unet_path}")
    
    if not os.path.exists(args.deeplab_path):
        raise ValueError(f"DeepLabV3+模型文件不存在: {args.deeplab_path}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置模型类
    def get_aer_unet_model_class():
        return get_aer_unet_model(
            n_channels=len(args.bands),
            n_classes=1,
            base_features=32,
            dropout_rate=0.3
        )
    
    def get_deeplabv3_plus_model_class():
        # 从配置中获取DeepLabV3+模型的参数
        deeplab_config = models_config[1] if 'models' in config and len(models_config) > 1 else {}
        
        return get_ultra_light_deeplabv3_plus(
            n_channels=len(args.bands),
            n_classes=1,
            pretrained_backbone=deeplab_config.get('pretrained_backbone', True),
            use_cbam=deeplab_config.get('use_cbam', True),
            aspp_out=deeplab_config.get('aspp_out', 128),
            dec_ch=deeplab_config.get('dec_ch', 128),
            low_ch_out=deeplab_config.get('low_ch_out', 32),
            cbam_reduction_ratio=deeplab_config.get('cbam_reduction_ratio', 16)
        )
    
    # 设置集成模型
    stacking_config = None
    adaptive_config = None
    advanced_adaptive_config = None
    performance_weighted_config = None
    improved_performance_weighted_config = None
    
    if args.ensemble_strategy == 'stacking':
        # 堆叠集成配置
        if args.config and 'predict' in config and 'stacking' in config['predict']:
            # 从配置文件中加载堆叠参数
            stacking_config = config['predict']['stacking']
            logger.info(f"从配置文件加载堆叠集成配置: {stacking_config}")
        else:
            # 使用默认配置
            stacking_config = {
                'fusion_layers': 3,
                'hidden_units': 128,
                'dropout_rate': 0.3,
                'use_batch_norm': True
            }
            logger.info(f"使用默认堆叠集成配置: {stacking_config}")
    elif args.ensemble_strategy == 'adaptive':
        # 自适应权重集成配置
        if args.config and 'predict' in config and 'adaptive' in config['predict']:
            # 从配置文件中加载自适应参数
            adaptive_config = config['predict']['adaptive']
            logger.info(f"从配置文件加载自适应权重集成配置: {adaptive_config}")
        else:
            # 使用默认配置
            adaptive_config = {
                'input_channels': len(args.bands),
                'hidden_units': 64,
                'dropout_rate': 0.2
            }
            logger.info(f"使用默认自适应权重集成配置: {adaptive_config}")
    elif args.ensemble_strategy == 'advanced_adaptive':
        # 高级自适应权重集成配置
        if args.config and 'predict' in config and 'advanced_adaptive' in config['predict']:
            # 从配置文件中加载高级自适应参数
            advanced_adaptive_config = config['predict']['advanced_adaptive']
            logger.info(f"从配置文件加载高级自适应权重集成配置: {advanced_adaptive_config}")
        else:
            # 使用默认配置
            advanced_adaptive_config = {
                'input_channels': len(args.bands),
                'hidden_units': 128,
                'dropout_rate': 0.3,
                'use_attention': True
            }
            logger.info(f"使用默认高级自适应权重集成配置: {advanced_adaptive_config}")
    elif args.ensemble_strategy == 'performance_weighted':
        # 基于性能的加权集成配置
        if args.config and 'predict' in config and 'performance_weighted' in config['predict']:
            # 从配置文件中加载基于性能的加权参数
            performance_weighted_config = config['predict']['performance_weighted']
            logger.info(f"从配置文件加载基于性能的加权集成配置: {performance_weighted_config}")
        else:
            # 使用默认配置
            performance_weighted_config = {
                'model_names': ['aer_unet', 'ultra_lightweight_deeplabv3_plus'],
                'csv_paths': [
                    'predictions/examples/aer_unet/evaluation_results_aer_unet_20251022_144056.csv',
                    'predictions/examples/ultra_lightweight_deeplabv3_plus/evaluation_results_ultra_lightweight_deeplabv3_plus_20251022_143831.csv'
                ],
                'metric_name': 'iou',
                'temperature': 2.0
            }
            logger.info(f"使用默认基于性能的加权集成配置: {performance_weighted_config}")
    elif args.ensemble_strategy == 'improved_performance_weighted':
        # 改进的基于性能的加权集成配置
        if args.config and 'ensemble' in config and 'improved_performance_weighted' in config['ensemble']:
            # 从配置文件中加载改进的基于性能的加权参数
            improved_performance_weighted_config = config['ensemble']['improved_performance_weighted']
            logger.info(f"从配置文件加载改进的基于性能的加权集成配置: {improved_performance_weighted_config}")
        else:
            # 使用默认配置
            improved_performance_weighted_config = {
                'model_names': ['aer_unet', 'ultra_lightweight_deeplabv3_plus'],
                'csv_paths': [
                    'predictions/examples/aer_unet/evaluation_results_aer_unet_20251022_144056.csv',
                    'predictions/examples/ultra_lightweight_deeplabv3_plus/evaluation_results_ultra_lightweight_deeplabv3_plus_20251022_143831.csv'
                ],
                'metric_name': 'iou',
                'metric_weights': {'iou': 0.5, 'dice': 0.3, 'f1': 0.2},
                'temperature': 1.0,
                'ensemble_method': 'gated_ensemble',
                'diff_threshold': 0.20,
                'conf_threshold': 0.15
            }
            logger.info(f"使用默认改进的基于性能的加权集成配置: {improved_performance_weighted_config}")
    
    model = setup_ensemble_models(
        model_paths=[args.aer_unet_path, args.deeplab_path],
        model_classes=[get_aer_unet_model_class, get_deeplabv3_plus_model_class],
        weights=[args.aer_unet_weight, args.deeplab_weight],
        strategy=args.ensemble_strategy,
        device=args.device,
        stacking_config=stacking_config,
        adaptive_config=adaptive_config,
        advanced_adaptive_config=advanced_adaptive_config,
        performance_weighted_config=performance_weighted_config,
        improved_performance_weighted_config=improved_performance_weighted_config,
        binary_threshold=args.threshold  # 传入二值化阈值
    )
    
    # 处理每个图像
    for i, image_file in enumerate(tqdm(image_files, desc='处理图像')):
        predict_with_ensemble(model, image_file, args, image_index=i)
    
    logger.info("集成预测完成!")

if __name__ == '__main__':
    main()
