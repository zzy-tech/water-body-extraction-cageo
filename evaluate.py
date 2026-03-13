# evaluate.py
"""
Sentinel-2水体分割模型评估脚本。
支持评估单个模型或集成多个模型的性能，以及生成预测结果。
"""
import os
import argparse
import logging
import numpy as np
import torch
import matplotlib
# 设置matplotlib后端，避免在没有GUI的环境中出错
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime
import pandas as pd
import rasterio

# 导入自定义模块
from models.aer_unet import get_aer_unet_model
from utils.data_utils import Sentinel2WaterDataset, create_data_loader, save_predictions
from utils.metrics import compute_metrics, compute_classification_report, calculate_threshold_metrics, compute_metrics_from_prob
from utils.postprocessing_utils import apply_postprocessing_pipeline
from utils.ensemble_utils import ModelEnsemble, load_ensemble_models, create_water_segmentation_ensemble, StackingEnsemble, AdaptiveWeightedEnsemble, AdvancedAdaptiveWeightedEnsemble
from utils.performance_weighted_ensemble import PerformanceWeightedEnsemble, create_performance_weighted_ensemble, load_performance_metrics
from utils.improved_performance_weighted_ensemble import ImprovedPerformanceWeightedEnsemble, create_improved_performance_weighted_ensemble
from config import get_config
from predict import sliding_window_inference, process_large_image_with_sliding_window

# 创建评估日志目录
eval_logs_dir = "evaluation_logs"
os.makedirs(eval_logs_dir, exist_ok=True)

# 生成带时间戳的日志文件名
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(eval_logs_dir, f"evaluation_{timestamp}.log")

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def _normalize_low_proj_keys(state_dict):
    """Handle low_proj/low_reduce aliasing for ultra_lightweight_deeplabv3_plus checkpoints."""
    if state_dict is None:
        return state_dict
    sd = dict(state_dict)
    has_low_proj = any("low_proj" in k for k in sd)
    has_low_reduce = any("low_reduce" in k for k in sd)
    if has_low_reduce and not has_low_proj:
        for k, v in list(sd.items()):
            if "low_reduce" in k:
                sd[k.replace("low_reduce", "low_proj")] = v
    if has_low_proj and not has_low_reduce:
        for k, v in list(sd.items()):
            if "low_proj" in k:
                sd[k.replace("low_proj", "low_reduce")] = v
    return sd



def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='评估Sentinel-2水体分割模型')
    
    # 新增参数：配置文件路径
    parser.add_argument('--config', type=str, default=None,
                        help='YAML配置文件路径')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='data',
                        help='数据集目录路径')
    parser.add_argument('--split', type=str, choices=['val', 'test'], default='test',
                        help='要评估的数据集分割')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批量大小')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载器工作进程数')
    parser.add_argument('--bands', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6],
                        help='要使用的波段索引列表')
    
    # 模型参数
    parser.add_argument('--model', type=str, choices=['aer_unet', 'lightweight_unet', 'unet', 'deeplabv3_plus', 'deeplabv3_plus_legacy', 'lightweight_deeplabv3_plus', 'ultra_lightweight_deeplabv3_plus'],
                        help='要评估的模型类型')
    parser.add_argument('--checkpoint_path', type=str,
                        help='模型检查点文件路径')
    parser.add_argument('--n_classes', type=int, default=1,
                        help='输出类别数')
    parser.add_argument('--base_features', type=int, default=64,
                        help='AER U-Net的基础特征数')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                        help='Dropout率')
    parser.add_argument('--output_stride', type=int, default=16, choices=[8, 16],
                        help='DeepLabV3+的输出步长')
    parser.add_argument('--pretrained_backbone', action='store_true',
                        help='是否使用预训练的骨干网络')
    parser.add_argument('--backbone_type', type=str, default='resnet50', choices=['resnet50', 'mobilenet_v2'],
                        help='骨干网络类型')
    
    # 集成参数
    parser.add_argument('--ensemble', action='store_true',
                        help='启用模型集成')
    parser.add_argument('--models', type=str, nargs='+', choices=['aer_unet', 'lightweight_unet', 'unet', 'deeplabv3_plus', 'lightweight_deeplabv3_plus', 'ultra_lightweight_deeplabv3_plus'],
                        help='要集成的模型类型列表')
    parser.add_argument('--checkpoint_paths', type=str, nargs='+',
                        help='要集成的模型检查点文件路径列表')
    parser.add_argument('--ensemble_strategy', type=str, choices=['mean', 'weighted_mean', 'vote', 'logits_mean', 'stacking', 'adaptive', 'advanced_adaptive', 'performance_weighted', 'improved_performance_weighted'], default='mean',
                    help='集成策略')
    parser.add_argument('--weights', type=float, nargs='+',
                        help='用于加权平均策略的权重列表')
    parser.add_argument('--ensemble_method', type=str, choices=['logits_weighted', 'prob_weighted', 'gated_ensemble'], default='gated_ensemble',
                    help='改进性能加权集成方法：logits_weighted(在logits域加权), prob_weighted(在概率域加权), gated_ensemble(门控/条件集成)')
    parser.add_argument('--gated_ensemble', action='store_true',
                        help='启用门控集成策略(当两个模型预测一致时直接取AER U-Net的结果，在分歧大的区域选择置信度更高的模型)')
    
    # 门控集成特定参数
    parser.add_argument('--diff_threshold', type=float, default=None,
                        help='模型预测差异阈值 (用于门控集成)')
    parser.add_argument('--conf_threshold', type=float, default=None,
                        help='置信度阈值 (用于门控集成)')
    parser.add_argument('--binary_threshold', type=float, default=None,
                        help='二值化阈值 (用于门控集成中的置信度计算)')
    parser.add_argument('--performance_metric_name', type=str, default='iou',
                        help='用于计算权重的性能指标名称 (iou, dice, f1_score等)')
    parser.add_argument('--metric_weights', type=str, default=None,
                        help='多指标权重配置，格式为JSON字符串，如 "{\\"iou\\": 0.5, \\"dice\\": 0.3, \\"f1\\": 0.2}"')
    parser.add_argument('--temperature', type=float, default=None,
                        help='温度参数，用于调整权重分布的尖锐程度')
    parser.add_argument('--power', type=float, default=None,
                        help='幂函数参数，用于放大性能差异')
    parser.add_argument('--model_names', type=str, nargs='+', default=None,
                        help='模型名称列表 (用于improved_performance_weighted策略)')
    parser.add_argument('--csv_paths', type=str, nargs='+', default=None,
                        help='模型性能CSV文件路径列表 (用于improved_performance_weighted策略)')
    
    # 预测参数
    parser.add_argument('--predict', action='store_true',
                        help='生成预测结果')
    parser.add_argument('--output_dir', type=str, default='predictions',
                        help='预测结果保存目录')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='二值化阈值')
    parser.add_argument('--plot_examples', action='store_true',
                        help='绘制预测示例')
    parser.add_argument('--num_examples', type=int, default=30,
                        help='要绘制的示例数量')
    
    # 滑窗预测参数
    parser.add_argument('--use_sliding_window', action='store_true',
                        help='启用滑窗预测')
    parser.add_argument('--tile_size', type=int, default=256,
                        help='滑窗瓦片大小')
    parser.add_argument('--overlap', type=int, default=32,
                        help='滑窗重叠大小')
    

    # 阈值搜索参数
    parser.add_argument('--enable_threshold_search', action='store_true',
                        help='启用阈值搜索')
    parser.add_argument('--threshold_search_interval', type=float, default=0.1,
                        help='阈值搜索间隔')
    parser.add_argument('--threshold_range', type=float, nargs=2, default=[0.1, 0.9],
                        help='阈值搜索范围')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='评估设备')
    
    # 单张影像评估参数
    parser.add_argument('--single_image', action='store_true',
                        help='启用单张影像评估模式')
    parser.add_argument('--image_path', type=str,
                        help='单张影像路径（单张影像评估模式）')
    parser.add_argument('--mask_path', type=str,
                        help='单张影像对应的真实掩膜路径（单张影像评估模式）')
    
    # 训练阶段后处理控制参数
    parser.add_argument('--apply_postprocessing_during_training', action='store_true',
                        help='训练阶段是否应用后处理（默认不应用，提高训练速度）')
    
    return parser.parse_args()

def update_args_with_config(args):
    """使用配置文件更新命令行参数"""
    # 在函数开头保存命令行传入的策略值
    cli_strategy = args.ensemble_strategy
    
    # 获取ensemble_strategy的默认值
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--ensemble_strategy', default='mean')  # 设置与parse_args中相同的默认值
    default_strategy = parser.get_default('ensemble_strategy')
    
    if args.config is not None:
        logger.info(f"从配置文件加载参数: {args.config}")
        # 获取配置
        config = get_config(yaml_config_path=args.config)
        
        # 打印配置的所有键
        logger.info(f"配置中的所有键: {list(config.keys())}")
        
        # 更新数据参数
        if 'data' in config and isinstance(config['data'], dict):
            if 'BATCH_SIZE' in config['data']:
                args.batch_size = config['data']['BATCH_SIZE']
            if 'NUM_WORKERS' in config['data']:
                args.num_workers = config['data']['NUM_WORKERS']
            if 'batch_size' in config['data']:
                args.batch_size = config['data']['batch_size']
            if 'num_workers' in config['data']:
                args.num_workers = config['data']['num_workers']
            # 处理数据路径配置
            if 'data_dir' in config['data']:
                args.data_dir = config['data']['data_dir']
            if 'images_dir' in config['data']:
                args.images_dir = config['data']['images_dir']
            if 'masks_dir' in config['data']:
                args.masks_dir = config['data']['masks_dir']
            if 'splits_dir' in config['data']:
                args.splits_dir = config['data']['splits_dir']
            if 'output_dir' in config['data']:
                args.output_dir = config['data']['output_dir']
        
        # 更新预测参数
        if 'predict' in config and isinstance(config['predict'], dict):
            if 'tile_size' in config['predict']:
                args.tile_size = config['predict']['tile_size']
            if 'overlap' in config['predict']:
                args.overlap = config['predict']['overlap']
            if 'batch_size' in config['predict']:
                args.batch_size = config['predict']['batch_size']
            if 'threshold' in config['predict']:
                args.threshold = config['predict']['threshold']
            if 'n_classes' in config['predict']:
                args.n_classes = config['predict']['n_classes']
            if 'use_sliding_window' in config['predict']:
                args.use_sliding_window = config['predict']['use_sliding_window']
            # 只有当命令行没有指定策略或使用默认值时才使用配置文件的值
            if 'ensemble_strategy' in config['predict'] and (cli_strategy is None or cli_strategy == default_strategy):
                args.ensemble_strategy = config['predict']['ensemble_strategy']
            if 'stacking' in config['predict']:
                args.stacking_config = config['predict']['stacking']
            if 'adaptive' in config['predict']:
                args.adaptive_config = config['predict']['adaptive']
            if 'advanced_adaptive' in config['predict']:
                args.advanced_adaptive_config = config['predict']['advanced_adaptive']
            if 'performance_weighted' in config['predict']:
                args.performance_weighted_config = config['predict']['performance_weighted']
            if 'improved_performance_weighted' in config['predict']:
                args.improved_performance_weighted_config = config['predict']['improved_performance_weighted']
                # 从improved_performance_weighted配置中读取ensemble_method
                if 'ensemble_method' in config['predict']['improved_performance_weighted']:
                    args.ensemble_method = config['predict']['improved_performance_weighted']['ensemble_method']
            if 'use_crf' in config['predict']:
                args.use_crf = config['predict']['use_crf']
            if 'use_tta' in config['predict']:
                args.use_tta = config['predict']['use_tta']
            if 'tta_types' in config['predict']:
                args.tta_types = config['predict']['tta_types']
            if 'bands' in config['predict']:
                args.bands = config['predict']['bands']
            if 'postprocessing' in config['predict']:
                args.postprocessing = config['predict']['postprocessing']
            if 'apply_postprocessing_during_training' in config['predict']:
                args.apply_postprocessing_during_training = config['predict']['apply_postprocessing_during_training']
        
        # 更新评估参数
        if 'eval' in config and isinstance(config['eval'], dict):
            if 'THRESHOLD' in config['eval']:
                args.threshold = config['eval']['THRESHOLD']
            if 'PLOT_EXAMPLES' in config['eval']:
                args.plot_examples = config['eval']['PLOT_EXAMPLES']
            if 'NUM_EXAMPLES' in config['eval']:
                args.num_examples = config['eval']['NUM_EXAMPLES']
        
        # 更新训练参数中的阈值搜索配置
        if 'train' in config and isinstance(config['train'], dict):
            if 'enable_threshold_search' in config['train']:
                args.enable_threshold_search = config['train']['enable_threshold_search']
            if 'threshold_search_interval' in config['train']:
                args.threshold_search_interval = config['train']['threshold_search_interval']
            if 'threshold_range' in config['train']:
                args.threshold_range = config['train']['threshold_range']
        
        # 更新模型参数
        if 'models' in config and isinstance(config['models'], list):
            # 处理集成模型配置
            args.models = config['models']
            args.use_ensemble = True
        else:
            # 处理单个模型配置 - 与train.py保持一致
            if args.model == 'aer_unet' and 'model' in config:
                model_config = config['model']
                if 'base_features' in model_config:
                    args.base_features = model_config['base_features']
                if 'dropout_rate' in model_config:
                    args.dropout_rate = model_config['dropout_rate']
            elif args.model == 'unet' and 'model' in config:
                model_config = config['model']
                if 'bilinear' in model_config:
                    args.unet_bilinear = bool(model_config['bilinear'])
            elif args.model == 'aer_unet' and 'aer_unet' in config:
                # 兼容旧配置格式
                aer_unet_config = config['aer_unet']
                if 'BASE_FEATURES' in aer_unet_config:
                    args.base_features = aer_unet_config['BASE_FEATURES']
                if 'DROPOUT_RATE' in aer_unet_config:
                    args.dropout_rate = aer_unet_config['DROPOUT_RATE']
                if 'base_features' in aer_unet_config:
                    args.base_features = aer_unet_config['base_features']
                if 'dropout_rate' in aer_unet_config:
                    args.dropout_rate = aer_unet_config['dropout_rate']
            
            if 'deeplabv3_plus' in config and isinstance(config['deeplabv3_plus'], dict):
                if 'OUTPUT_STRIDE' in config['deeplabv3_plus']:
                    args.output_stride = config['deeplabv3_plus']['OUTPUT_STRIDE']
                if 'PRETRAINED_BACKBONE' in config['deeplabv3_plus']:
                    args.pretrained_backbone = config['deeplabv3_plus']['PRETRAINED_BACKBONE']
                if 'output_stride' in config['deeplabv3_plus']:
                    args.output_stride = config['deeplabv3_plus']['output_stride']
                if 'pretrained_backbone' in config['deeplabv3_plus']:
                    args.pretrained_backbone = config['deeplabv3_plus']['pretrained_backbone']
                if 'BACKBONE_TYPE' in config['deeplabv3_plus']:
                    args.backbone_type = config['deeplabv3_plus']['BACKBONE_TYPE']
                if 'backbone_type' in config['deeplabv3_plus']:
                    args.backbone_type = config['deeplabv3_plus']['backbone_type']
            
            # 从model配置中读取backbone_type
            if 'model' in config and isinstance(config['model'], dict):
                if 'backbone' in config['model']:
                    args.backbone_type = config['model']['backbone']
                if 'backbone_type' in config['model']:
                    args.backbone_type = config['model']['backbone_type']
        
        # 更新集成参数
        if 'ensemble' in config and isinstance(config['ensemble'], dict):
            # 只有当命令行没有指定策略或使用默认值时才使用配置文件的值
            if 'STRATEGY' in config['ensemble'] and (cli_strategy is None or cli_strategy == default_strategy):
                args.ensemble_strategy = config['ensemble']['STRATEGY']
            if 'WEIGHTS' in config['ensemble']:
                args.weights = config['ensemble']['WEIGHTS']
            # 使用保存的cli_strategy判断
            if 'strategy' in config['ensemble'] and (cli_strategy is None or cli_strategy == default_strategy):
                args.ensemble_strategy = config['ensemble']['strategy']
            if 'weights' in config['ensemble']:
                args.weights = config['ensemble']['weights']
    
    return args

def get_model_class(model_type):
    """根据模型类型返回对应的模型类"""
    if model_type == 'aer_unet':
        return get_aer_unet_model
    elif model_type == 'lightweight_unet':
        from models.lightweight_unet import get_lightweight_unet_model
        return get_lightweight_unet_model
    elif model_type == 'unet':
        from models.unet_model import get_unet_model
        return get_unet_model
    elif model_type == 'deeplabv3_plus':
        from models.deeplabv3_plus import get_deeplabv3_plus_model
        return get_deeplabv3_plus_model
    elif model_type == 'lightweight_deeplabv3_plus':
        from models.lightweight_deeplabv3_plus import get_lightweight_deeplabv3_plus_model
        return get_lightweight_deeplabv3_plus_model
    elif model_type == 'ultra_lightweight_deeplabv3_plus':
        from models.ultra_lightweight_deeplabv3_plus import get_ultra_light_deeplabv3_plus
        return get_ultra_light_deeplabv3_plus
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


def setup_single_model(args):
    """根据参数设置单个模型"""
    logger.info(f"设置{args.model}模型...")
    
    if args.model == 'aer_unet':
        model = get_aer_unet_model(
            n_channels=len(args.bands),
            n_classes=args.n_classes,
            base_features=args.base_features,
            dropout_rate=args.dropout_rate
        )
        
        # 加载检查点
        if args.checkpoint_path is not None:
            logger.info(f'从检查点加载模型: {args.checkpoint_path}')
            checkpoint = torch.load(args.checkpoint_path, map_location=args.device, weights_only=False)
            
            # 与train.py保持一致的加载逻辑
            model_dict = model.state_dict()
            if 'model_state_dict' in checkpoint:
                sd = checkpoint['model_state_dict']
            else:
                sd = checkpoint
            
            # 键名映射 - 与train.py保持一致
            sd = _normalize_low_proj_keys(sd)
            
            # 使用strict=False加载，与train.py保持一致
            missing, unexpected = model.load_state_dict(sd, strict=False)
            logger.info(f"[Load] loaded with strict=False | missing: {len(missing)} | unexpected: {len(unexpected)}")
            
            if len(missing) > 0:
                logger.warning("  missing (sample):")
                for m in list(missing)[:5]:  # 只显示前5个
                    logger.warning(f"    - {m}")
            
            if len(unexpected) > 0:
                logger.warning("  unexpected (sample):")
                for u in list(unexpected)[:5]:  # 只显示前5个
                    logger.warning(f"    - {u}")
    elif args.model == 'lightweight_unet':
        from models.lightweight_unet import get_lightweight_unet_model
        model = get_lightweight_unet_model(
            n_channels=len(args.bands),
            n_classes=args.n_classes,
            base_features=args.base_features,
            dropout_rate=args.dropout_rate
        )
        
        # 加载检查点
        if args.checkpoint_path is not None:
            logger.info(f'从检查点加载模型: {args.checkpoint_path}')
            checkpoint = torch.load(args.checkpoint_path, map_location=args.device, weights_only=False)
    elif args.model == 'unet':
        from models.unet_model import get_unet_model
        model = get_unet_model(
            n_channels=len(args.bands),
            n_classes=args.n_classes,
            bilinear=getattr(args, 'unet_bilinear', False)
        )

        # 加载检查点
        if args.checkpoint_path is not None:
            logger.info(f'从检查点加载模型: {args.checkpoint_path}')
            checkpoint = torch.load(args.checkpoint_path, map_location=args.device, weights_only=False)
            
            # 与train.py保持一致的加载逻辑
            model_dict = model.state_dict()
            if 'model_state_dict' in checkpoint:
                sd = checkpoint['model_state_dict']
            else:
                sd = checkpoint
            
            # 键名映射 - 与train.py保持一致
            sd = _normalize_low_proj_keys(sd)
            
            # 使用strict=False加载，与train.py保持一致
            missing, unexpected = model.load_state_dict(sd, strict=False)
            logger.info(f"[Load] loaded with strict=False | missing: {len(missing)} | unexpected: {len(unexpected)}")
            
            if len(missing) > 0:
                logger.warning("  missing (sample):")
                for m in list(missing)[:5]:  # 只显示前5个
                    logger.warning(f"    - {m}")
            
            if len(unexpected) > 0:
                logger.warning("  unexpected (sample):")
                for u in list(unexpected)[:5]:  # 只显示前5个
                    logger.warning(f"    - {u}")
    elif args.model == 'deeplabv3_plus':
        from models.deeplabv3_plus import get_deeplabv3_plus_model
        
        # 从配置文件或命令行获取backbone_type
        backbone_type = getattr(args, 'backbone_type', 'resnet50')
        
        model = get_deeplabv3_plus_model(
            n_channels=len(args.bands),
            n_classes=args.n_classes,
            output_stride=args.output_stride,
            pretrained_backbone=args.pretrained_backbone,
            backbone_type=backbone_type
        )
        
        # 加载检查点
        if args.checkpoint_path is not None:
            logger.info(f'从检查点加载模型: {args.checkpoint_path}')
            checkpoint = torch.load(args.checkpoint_path, map_location=args.device, weights_only=False)
            
            # 与train.py保持一致的加载逻辑
            model_dict = model.state_dict()
            if 'model_state_dict' in checkpoint:
                sd = checkpoint['model_state_dict']
            else:
                sd = checkpoint
            
            # 键名映射 - 与train.py保持一致
            sd = _normalize_low_proj_keys(sd)
            
            # 使用strict=False加载，与train.py保持一致
            missing, unexpected = model.load_state_dict(sd, strict=False)
            logger.info(f"[Load] loaded with strict=False | missing: {len(missing)} | unexpected: {len(unexpected)}")
            
            if len(missing) > 0:
                logger.warning("  missing (sample):")
                for m in list(missing)[:5]:  # 只显示前5个
                    logger.warning(f"    - {m}")
            
            if len(unexpected) > 0:
                logger.warning("  unexpected (sample):")
                for u in list(unexpected)[:5]:  # 只显示前5个
                    logger.warning(f"    - {u}")
    elif args.model == 'lightweight_deeplabv3_plus':
        from models.lightweight_deeplabv3_plus import get_lightweight_deeplabv3_plus_model
        model = get_lightweight_deeplabv3_plus_model(
            n_channels=len(args.bands),
            n_classes=args.n_classes,
            output_stride=args.output_stride,
            pretrained_backbone=args.pretrained_backbone
        )
        
        # 加载检查点
        if args.checkpoint_path is not None:
            logger.info(f'从检查点加载模型: {args.checkpoint_path}')
            checkpoint = torch.load(args.checkpoint_path, map_location=args.device, weights_only=False)
            
            # 与train.py保持一致的加载逻辑
            model_dict = model.state_dict()
            if 'model_state_dict' in checkpoint:
                sd = checkpoint['model_state_dict']
            else:
                sd = checkpoint
            
            # 键名映射 - 与train.py保持一致
            sd = _normalize_low_proj_keys(sd)
            
            # 使用strict=False加载，与train.py保持一致
            missing, unexpected = model.load_state_dict(sd, strict=False)
            logger.info(f"[Load] loaded with strict=False | missing: {len(missing)} | unexpected: {len(unexpected)}")
            
            if len(missing) > 0:
                logger.warning("  missing (sample):")
                for m in list(missing)[:5]:  # 只显示前5个
                    logger.warning(f"    - {m}")
            
            if len(unexpected) > 0:
                logger.warning("  unexpected (sample):")
                for u in list(unexpected)[:5]:  # 只显示前5个
                    logger.warning(f"    - {u}")
    elif args.model == 'ultra_lightweight_deeplabv3_plus':
        from models.ultra_lightweight_deeplabv3_plus import get_ultra_light_deeplabv3_plus
        
        # 首先尝试使用默认参数创建模型
        model = get_ultra_light_deeplabv3_plus(
            n_channels=len(args.bands),
            n_classes=args.n_classes,
            pretrained_backbone=args.pretrained_backbone
        )
        
        # 加载检查点
        if args.checkpoint_path is not None:
            logger.info(f'从检查点加载模型: {args.checkpoint_path}')
            checkpoint = torch.load(args.checkpoint_path, map_location=args.device, weights_only=False)
            
            # 检查检查点中是否包含模型配置信息
            model_config = {}
            if 'model_config' in checkpoint:
                model_config = checkpoint['model_config']
                logger.info(f"从检查点加载模型配置: {model_config}")
            
            # 检查检查点中是否包含训练参数
            if 'args' in checkpoint and not model_config:
                # 从训练参数中提取模型配置
                train_args = checkpoint['args']
                if hasattr(train_args, 'aspp_out'):
                    model_config['aspp_out'] = train_args.aspp_out
                if hasattr(train_args, 'dec_ch'):
                    model_config['dec_ch'] = train_args.dec_ch
                if hasattr(train_args, 'low_ch_out'):
                    model_config['low_ch_out'] = train_args.low_ch_out
                if hasattr(train_args, 'use_cbam'):
                    model_config['use_cbam'] = train_args.use_cbam
                logger.info(f"从训练参数中提取模型配置: {model_config}")
            
            # 如果有配置信息，使用这些配置重新创建模型
            if model_config:
                try:
                    model = get_ultra_light_deeplabv3_plus(
                        n_channels=len(args.bands),
                        n_classes=args.n_classes,
                        pretrained_backbone=args.pretrained_backbone,
                        aspp_out=model_config.get('aspp_out', 64),
                        dec_ch=model_config.get('dec_ch', 64),
                        low_ch_out=model_config.get('low_ch_out', 32),
                        use_cbam=model_config.get('use_cbam', False)
                    )
                    logger.info("使用检查点中的配置重新创建模型")
                except Exception as e:
                    logger.warning(f"使用检查点配置创建模型失败: {e}，使用默认配置")
                    # 确保model变量已经被定义
                    model = get_ultra_light_deeplabv3_plus(
                        n_channels=len(args.bands),
                        n_classes=args.n_classes,
                        pretrained_backbone=args.pretrained_backbone
                    )
            
            # 尝试加载状态字典 - 与train.py保持一致的加载逻辑
            if 'model_state_dict' in checkpoint:
                sd = checkpoint['model_state_dict']
            else:
                sd = checkpoint
            
            # 键名映射 - 与train.py保持一致
            sd = _normalize_low_proj_keys(sd)
            
            # 使用strict=False加载，与train.py保持一致
            missing, unexpected = model.load_state_dict(sd, strict=False)
            logger.info(f"[Load] loaded with strict=False | missing: {len(missing)} | unexpected: {len(unexpected)}")
            
            if len(missing) > 0:
                logger.warning("  missing (sample):")
                for m in list(missing)[:5]:  # 只显示前5个
                    logger.warning(f"    - {m}")
            
            if len(unexpected) > 0:
                logger.warning("  unexpected (sample):")
                for u in list(unexpected)[:5]:  # 只显示前5个
                    logger.warning(f"    - {u}")
    else:
        raise ValueError(f"不支持的模型类型: {args.model}")
    
    # 设置为评估模式
    model.to(args.device)
    model.eval()
    
    return model

def setup_ensemble_model(args):
    """设置集成模型"""
    logger.info("设置集成模型...")
    
    # 验证参数
    if args.models is None or len(args.models) == 0:
        raise ValueError("集成模式需要指定模型类型")
    
    if args.checkpoint_paths is None or len(args.checkpoint_paths) == 0:
        raise ValueError("集成模式需要指定检查点路径")
    
    if len(args.models) != len(args.checkpoint_paths):
        raise ValueError("模型类型数量必须与检查点路径数量匹配")
    
    # 创建模型列表
    models = []
    model_configs = []  # 存储每个模型的配置信息
    
    for i, (model_type, checkpoint_path) in enumerate(zip(args.models, args.checkpoint_paths)):
        logger.info(f"加载{model_type}模型: {checkpoint_path}")
        
        # 规范化模型类型，使其能够匹配展示名称
        normalized_model_type = model_type.strip().lower().replace(" ", "_").replace("-", "_")
        
        # 特殊处理一些常见的展示名称
        if normalized_model_type == "aer_u_net":
            normalized_model_type = "aer_unet"
        elif normalized_model_type == "ultra_lightweight_deeplabv3_":
            normalized_model_type = "ultra_lightweight_deeplabv3_plus"
        elif normalized_model_type == "ultra_lightweight_deeplabv3+":
            normalized_model_type = "ultra_lightweight_deeplabv3_plus"
        
        # 创建模型配置字典，默认使用全局参数
        model_config = {
            'name': normalized_model_type,
            'n_channels': len(args.bands),
            'n_classes': args.n_classes,
            'base_features': args.base_features,
            'dropout_rate': args.dropout_rate,
            'output_stride': args.output_stride,
            'pretrained_backbone': args.pretrained_backbone
        }
        
        # 尝试从检查点加载模型特定配置
        checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
        
        # 检查检查点中是否包含模型配置信息
        checkpoint_model_config = {}
        if 'model_config' in checkpoint:
            checkpoint_model_config = checkpoint['model_config']
            logger.info(f"从检查点加载模型配置: {checkpoint_model_config}")
        
        # 检查检查点中是否包含训练参数
        if 'args' in checkpoint and not checkpoint_model_config:
            # 从训练参数中提取模型配置
            train_args = checkpoint['args']
            if hasattr(train_args, 'base_features'):
                checkpoint_model_config['base_features'] = train_args.base_features
            if hasattr(train_args, 'dropout_rate'):
                checkpoint_model_config['dropout_rate'] = train_args.dropout_rate
            if hasattr(train_args, 'output_stride'):
                checkpoint_model_config['output_stride'] = train_args.output_stride
            if hasattr(train_args, 'aspp_out'):
                checkpoint_model_config['aspp_out'] = train_args.aspp_out
            if hasattr(train_args, 'dec_ch'):
                checkpoint_model_config['dec_ch'] = train_args.dec_ch
            if hasattr(train_args, 'low_ch_out'):
                checkpoint_model_config['low_ch_out'] = train_args.low_ch_out
            if hasattr(train_args, 'use_cbam'):
                checkpoint_model_config['use_cbam'] = train_args.use_cbam
            if hasattr(train_args, 'pretrained_backbone'):
                checkpoint_model_config['pretrained_backbone'] = train_args.pretrained_backbone
            logger.info(f"从训练参数中提取模型配置: {checkpoint_model_config}")
        
        # 更新模型配置，使用检查点中的特定配置
        model_config.update(checkpoint_model_config)
        model_configs.append(model_config)
        
        if normalized_model_type == 'aer_unet':
            model = get_aer_unet_model(
                n_channels=model_config.get('n_channels', len(args.bands)),
                n_classes=model_config.get('n_classes', args.n_classes),
                base_features=model_config.get('base_features', args.base_features),
                dropout_rate=model_config.get('dropout_rate', args.dropout_rate)
            )
        elif normalized_model_type == 'lightweight_unet':
            from models.lightweight_unet import get_lightweight_unet_model
            model = get_lightweight_unet_model(
                n_channels=model_config.get('n_channels', len(args.bands)),
                n_classes=model_config.get('n_classes', args.n_classes),
                base_features=model_config.get('base_features', args.base_features),
                dropout_rate=model_config.get('dropout_rate', args.dropout_rate)
            )
        elif normalized_model_type == 'unet':
            from models.unet_model import get_unet_model
            model = get_unet_model(
                n_channels=model_config.get('n_channels', len(args.bands)),
                n_classes=model_config.get('n_classes', args.n_classes),
                bilinear=model_config.get('bilinear', False)
            )
        elif normalized_model_type == 'deeplabv3_plus':
            from models.deeplabv3_plus import get_deeplabv3_plus_model
            model = get_deeplabv3_plus_model(
                n_channels=model_config.get('n_channels', len(args.bands)),
                n_classes=model_config.get('n_classes', args.n_classes),
                output_stride=model_config.get('output_stride', args.output_stride),
                pretrained_backbone=model_config.get('pretrained_backbone', args.pretrained_backbone),
                backbone_type=model_config.get('backbone', getattr(args, 'backbone_type', 'resnet50'))
            )
        elif normalized_model_type == 'lightweight_deeplabv3_plus':
            from models.lightweight_deeplabv3_plus import get_lightweight_deeplabv3_plus_model
            model = get_lightweight_deeplabv3_plus_model(
                n_channels=model_config.get('n_channels', len(args.bands)),
                n_classes=model_config.get('n_classes', args.n_classes),
                output_stride=model_config.get('output_stride', args.output_stride),
                pretrained_backbone=model_config.get('pretrained_backbone', args.pretrained_backbone)
            )
        elif normalized_model_type == 'ultra_lightweight_deeplabv3_plus':
            from models.ultra_lightweight_deeplabv3_plus import get_ultra_light_deeplabv3_plus
            
            # 使用检查点中的配置创建模型
            model = get_ultra_light_deeplabv3_plus(
                n_channels=model_config.get('n_channels', len(args.bands)),
                n_classes=model_config.get('n_classes', args.n_classes),
                pretrained_backbone=model_config.get('pretrained_backbone', args.pretrained_backbone),
                aspp_out=model_config.get('aspp_out', 64),
                dec_ch=model_config.get('dec_ch', 64),
                low_ch_out=model_config.get('low_ch_out', 32),
                use_cbam=model_config.get('use_cbam', False)
            )
            
            # 尝试加载状态字典
            if 'model_state_dict' in checkpoint:
                # 尝试加载状态字典，只加载形状匹配的键
                model_dict = model.state_dict()
                pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() 
                                  if k in model_dict and v.shape == model_dict[k].shape}
                
                if len(pretrained_dict) == len(model_dict):
                    logger.info(f"完全加载{normalized_model_type}模型检查点（所有{len(pretrained_dict)}个参数形状匹配）")
                    model_dict.update(pretrained_dict)
                    model.load_state_dict(model_dict)
                elif pretrained_dict:
                    logger.warning(f"部分加载{normalized_model_type}模型检查点（{len(pretrained_dict)}/{len(model_dict)}个参数形状匹配）")
                    logger.warning("以下参数不匹配:")
                    for k in model_dict:
                        if k not in pretrained_dict:
                            logger.warning(f"  - {k}: 模型期望形状 {model_dict[k].shape}")
                    model_dict.update(pretrained_dict)
                    model.load_state_dict(model_dict)
                else:
                    logger.warning("没有找到形状匹配的参数，模型将使用随机初始化的参数")
            else:
                # 尝试加载整个检查点，只加载形状匹配的键
                model_dict = model.state_dict()
                pretrained_dict = {k: v for k, v in checkpoint.items() 
                                  if k in model_dict and v.shape == model_dict[k].shape}
                
                if len(pretrained_dict) == len(model_dict):
                    logger.info(f"完全加载{normalized_model_type}模型检查点（所有{len(pretrained_dict)}个参数形状匹配）")
                    model_dict.update(pretrained_dict)
                    model.load_state_dict(model_dict)
                elif pretrained_dict:
                    logger.warning(f"部分加载{normalized_model_type}模型检查点（{len(pretrained_dict)}/{len(model_dict)}个参数形状匹配）")
                    logger.warning("以下参数不匹配:")
                    for k in model_dict:
                        if k not in pretrained_dict:
                            logger.warning(f"  - {k}: 模型期望形状 {model_dict[k].shape}")
                    model_dict.update(pretrained_dict)
                    model.load_state_dict(model_dict)
                else:
                    logger.warning("没有找到形状匹配的参数，模型将使用随机初始化的参数")
        else:
            raise ValueError(f"不支持的模型类型: {normalized_model_type}")
        
        # 对于其他模型类型，也尝试加载检查点
        if normalized_model_type != 'ultra_lightweight_deeplabv3_plus':
            # 尝试加载状态字典 - 与train.py保持一致的加载逻辑
            if 'model_state_dict' in checkpoint:
                sd = checkpoint['model_state_dict']
            else:
                sd = checkpoint
            
            # 键名映射 - 与train.py保持一致
            sd = _normalize_low_proj_keys(sd)
            
            # 使用strict=False加载，与train.py保持一致
            missing, unexpected = model.load_state_dict(sd, strict=False)
            logger.info(f"[Load] {normalized_model_type} loaded with strict=False | missing: {len(missing)} | unexpected: {len(unexpected)}")
            
            if len(missing) > 0:
                logger.warning("  missing (sample):")
                for m in list(missing)[:5]:  # 只显示前5个
                    logger.warning(f"    - {m}")
            
            if len(unexpected) > 0:
                logger.warning("  unexpected (sample):")
                for u in list(unexpected)[:5]:  # 只显示前5个
                    logger.warning(f"    - {u}")
        
        # 设置为评估模式
        model.to(args.device)
        model.eval()
        
        models.append(model)
    
    # 创建集成模型
    if args.ensemble_strategy == 'stacking':
        # 堆叠集成
        # 从配置中获取堆叠参数，如果没有则使用默认值
        stacking_config = getattr(args, 'stacking_config', {})
        fusion_layers = stacking_config.get('fusion_layers', 3)
        hidden_units = stacking_config.get('hidden_units', 128)
        dropout_rate = stacking_config.get('dropout_rate', 0.3)
        use_batch_norm = stacking_config.get('use_batch_norm', True)
        
        logger.info(f"使用堆叠集成策略，参数: fusion_layers={fusion_layers}, hidden_units={hidden_units}, dropout_rate={dropout_rate}")
        
        ensemble = StackingEnsemble(
            models=models,
            n_classes=args.n_classes,
            fusion_layers=fusion_layers,
            hidden_units=hidden_units,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm
        )
    elif args.ensemble_strategy == 'adaptive':
        # 自适应权重集成
        # 从配置中获取自适应参数，如果没有则使用默认值
        adaptive_config = getattr(args, 'adaptive_config', {})
        input_channels = adaptive_config.get('input_channels', 6)
        hidden_units = adaptive_config.get('hidden_units', 64)
        dropout_rate = adaptive_config.get('dropout_rate', 0.2)
        
        logger.info(f"使用自适应权重集成策略，参数: input_channels={input_channels}, hidden_units={hidden_units}, dropout_rate={dropout_rate}")
        
        ensemble = AdaptiveWeightedEnsemble(
            models=models,
            n_classes=args.n_classes,
            input_channels=input_channels,
            hidden_units=hidden_units,
            dropout_rate=dropout_rate
        )
    elif args.ensemble_strategy == 'advanced_adaptive':
        # 高级自适应权重集成策略
        advanced_adaptive_config = getattr(args, 'advanced_adaptive_config', {})
        input_channels = advanced_adaptive_config.get('input_channels', 6)
        hidden_units = advanced_adaptive_config.get('hidden_units', 128)
        dropout_rate = advanced_adaptive_config.get('dropout_rate', 0.3)
        use_attention = advanced_adaptive_config.get('use_attention', True)
        
        logger.info(f"使用高级自适应权重集成策略，参数: input_channels={input_channels}, hidden_units={hidden_units}, dropout_rate={dropout_rate}, use_attention={use_attention}")
        
        ensemble = AdvancedAdaptiveWeightedEnsemble(
            models=models,
            n_classes=args.n_classes,
            input_channels=input_channels,
            hidden_units=hidden_units,
            dropout_rate=dropout_rate,
            use_attention=use_attention
        )
    elif args.ensemble_strategy == 'performance_weighted':
        # 基于性能的加权集成策略
        performance_weighted_config = getattr(args, 'performance_weighted_config', {})
        metrics_path = performance_weighted_config.get('metrics_path', 'model_performance_metrics.csv')
        metric_name = performance_weighted_config.get('metric_name', 'iou')
        temperature = performance_weighted_config.get('temperature', 2.0)
        
        logger.info(f"使用基于性能的加权集成策略，参数: metrics_path={metrics_path}, metric_name={metric_name}, temperature={temperature}")
        
        # 加载模型性能指标
        try:
            performance_metrics = load_performance_metrics(metrics_path)
            logger.info(f"成功加载模型性能指标: {performance_metrics}")
        except Exception as e:
            logger.warning(f"无法加载性能指标文件 {metrics_path}: {e}")
            logger.info("使用默认性能指标（基于模型类型）")
            # 根据模型类型设置默认性能指标
            performance_metrics = {}
            for i, model_type in enumerate(args.models):
                # 这里可以根据模型类型设置不同的默认值
                # 例如，AER_UNET通常性能较好，可以设置较高的默认值
                if model_type == 'aer_unet':
                    performance_metrics[model_type] = 0.9
                elif model_type == 'ultra_lightweight_deeplabv3_plus':
                    performance_metrics[model_type] = 0.8
                else:
                    performance_metrics[model_type] = 0.75
        
        # 创建基于性能的加权集成模型
        ensemble = PerformanceWeightedEnsemble(
            models=models,
            performance_metrics=performance_metrics,
            metric_name=metric_name,
            temperature=temperature
        )
    elif args.ensemble_strategy == 'improved_performance_weighted':
        # 改进的基于性能的加权集成策略
        improved_cfg = getattr(args, 'improved_performance_weighted_config', {}) or {}
        model_names = improved_cfg.get('model_names', ['aer_unet', 'ultra_lightweight_deeplabv3_plus'])
        csv_paths = improved_cfg.get('csv_paths', [])
        
        # 如果csv_paths为空或命令行指定了--checkpoint_paths，尝试从命令行参数推断
        if not csv_paths and hasattr(args, 'checkpoint_paths'):
            logger.warning("csv_paths未设置，尝试从checkpoint_paths推断")
            # 为每个模型生成默认的CSV路径
            csv_paths = []
            for i, model_type in enumerate(args.models):
                default_path = f'test_evaluation/examples/{model_type}/evaluation_results_{model_type}_*.csv'
                csv_paths.append(default_path)
        
        metric_name = improved_cfg.get('metric_name', 'iou')
        temperature = improved_cfg.get('temperature', 1.0)
        
        # 从改进性能加权配置中读取参数
        metric_name = improved_cfg.get('metric_name', 'iou')
        temperature = improved_cfg.get('temperature', 1.0)
        power = improved_cfg.get('power', 2.0)  # 新增power参数
        metric_weights = improved_cfg.get('metric_weights', {'iou': 0.5, 'dice': 0.3, 'f1': 0.2})  # 新增多指标权重
        if isinstance(metric_weights, dict):
            # normalize possible key name for f1
            if 'f1_score' in metric_weights and 'f1' not in metric_weights:
                metric_weights['f1'] = metric_weights.pop('f1_score')
        
        # 从配置文件中获取门控阈值参数
        diff_threshold = improved_cfg.get('diff_threshold', 0.20)
        conf_threshold = improved_cfg.get('conf_threshold', 0.22)
        
        # 使用命令行参数覆盖配置文件中的值（如果提供了命令行参数）
        if hasattr(args, 'diff_threshold') and args.diff_threshold is not None:
            diff_threshold = args.diff_threshold
            logger.info(f"使用命令行参数覆盖diff_threshold: {diff_threshold}")
            
        if hasattr(args, 'conf_threshold') and args.conf_threshold is not None:
            conf_threshold = args.conf_threshold
            logger.info(f"使用命令行参数覆盖conf_threshold: {conf_threshold}")
        
        # 处理命令行参数覆盖配置中的多指标权重和power参数
        if hasattr(args, 'metric_weights') and args.metric_weights is not None:
            try:
                import json
                metric_weights = json.loads(args.metric_weights)
                logger.info(f"使用命令行参数覆盖metric_weights: {metric_weights}")
            except json.JSONDecodeError:
                logger.warning(f"无法解析metric_weights参数: {args.metric_weights}，使用配置文件中的值")
                
        if hasattr(args, 'power') and args.power is not None:
            power = args.power
            logger.info(f"使用命令行参数覆盖power: {power}")
            
        # 覆盖温度参数（如果命令行指定了）
        if hasattr(args, 'temperature') and args.temperature is not None:
            temperature = args.temperature
            logger.info(f"使用命令行参数覆盖temperature: {temperature}")
        
        # 从命令行参数获取集成方法，如果没有指定则使用配置文件中的值
        ensemble_method = getattr(args, 'ensemble_method', improved_cfg.get('ensemble_method', 'gated_ensemble'))
        
        # 如果指定了--gated_ensemble参数，则使用gated_ensemble方法
        if getattr(args, 'gated_ensemble', False):
            ensemble_method = 'gated_ensemble'
        
        logger.info(f"使用改进的基于性能的加权集成策略，参数: model_names={model_names}, csv_paths={csv_paths}, metric_name={metric_name}, temperature={temperature}, ensemble_method={ensemble_method}, diff_threshold={diff_threshold}, conf_threshold={conf_threshold}")
        
        # 验证CSV路径的有效性
        for csv_path in csv_paths:
            # 检查文件是否存在，如果使用通配符则检查目录
            if '*' in csv_path:
                import glob
                matching_files = glob.glob(csv_path)
                if not matching_files:
                    logger.warning(f"未找到匹配CSV文件: {csv_path}")
                else:
                    logger.info(f"找到{len(matching_files)}个匹配CSV文件: {matching_files[:2]}...")
            elif not os.path.exists(csv_path):
                logger.warning(f"CSV文件不存在: {csv_path}")
            else:
                logger.info(f"CSV文件存在: {csv_path}")
        
        # 获取二值化阈值，优先使用命令行参数
        binary_threshold = getattr(args, 'binary_threshold', None)
        if binary_threshold is None:
            # 如果命令行没有指定，使用配置文件中的值
            binary_threshold = improved_cfg.get('binary_threshold', 0.5)
        else:
            logger.info(f"使用命令行参数覆盖binary_threshold: {binary_threshold}")
        
        # 创建改进的基于性能的加权集成模型
        ensemble = create_improved_performance_weighted_ensemble(
            model_paths=args.checkpoint_paths,
            model_classes=[get_model_class(model_name) for model_name in args.models],
            model_names=model_names,
            csv_paths=csv_paths,
            metric_names=improved_cfg.get('metric_names', ['iou', 'dice', 'f1']),  # 新增多指标列表
            metric_weights=metric_weights,  # 传递多指标权重
            metric_name=metric_name,
            temperature=temperature,
            power=power,  # 传递power参数
            ensemble_method=ensemble_method,
            diff_threshold=diff_threshold,
            conf_threshold=conf_threshold,
            binary_threshold=binary_threshold,
            device=args.device,
            n_channels=len(args.bands),
            n_classes=args.n_classes,
            model_configs=model_configs  # 传递模型配置列表
        )
        
        # 记录从CSV提取的实际权重和门控参数 - 使用print确保立即在控制台显示
        print(f"从CSV提取的实际模型权重: {ensemble.weights}")
        # 添加显示性能加权结果的日志（实际用于集成的权重）
        if hasattr(ensemble, 'performance_weights'):
            print(f"性能加权结果（实际集成权重）: {ensemble.performance_weights}")
        else:
            print("警告: 未找到performance_weights属性，请检查ImprovedPerformanceWeightedEnsemble类的实现")
        print(f"门控集成参数 - ensemble_method: {ensemble_method}, diff_threshold: {diff_threshold}, "
              f"conf_threshold: {conf_threshold}, temperature: {temperature}")
        
        # 同时记录到日志文件
        logger.info(f"从CSV提取的实际模型权重: {ensemble.weights}")
        if hasattr(ensemble, 'performance_weights'):
            logger.info(f"性能加权结果（实际集成权重）: {ensemble.performance_weights}")
        else:
            logger.info("警告: 未找到performance_weights属性，请检查ImprovedPerformanceWeightedEnsemble类的实现")
        logger.info(f"门控集成参数 - ensemble_method: {ensemble_method}, diff_threshold: {diff_threshold}, "
                    f"conf_threshold: {conf_threshold}, temperature: {temperature}")
    else:
        # 其他集成策略
        ensemble = ModelEnsemble(
            models=models,
            strategy=args.ensemble_strategy,
            weights=args.weights
        )
    
    return ensemble.to(args.device)

def evaluate_model(model, dataloader, device, args):
    """评估模型性能"""
    model.eval()
    all_metrics = []
    all_predictions = []
    all_masks = []
    all_filenames = []
    
    # 添加内存监控
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info(f"评估开始前GPU内存使用: {torch.cuda.memory_allocated() / 1024**3:.2f}GB / {torch.cuda.memory_reserved() / 1024**3:.2f}GB")
    
    # 检查是否使用滑窗推理
    use_sliding_window = getattr(args, 'use_sliding_window', False)
    
    with torch.no_grad():
        # 创建进度条
        pbar = tqdm(dataloader, desc='评估', leave=False)
        
        for batch_idx, batch in enumerate(pbar):
            try:
                # 获取数据
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                filenames = batch['filename']
                
                if use_sliding_window:
                    # 使用滑窗推理处理每张图像
                    for i in range(images.size(0)):
                        try:
                            # 获取单张图像和掩码
                            image = images[i].unsqueeze(0)  # 添加批次维度
                            mask = masks[i].unsqueeze(0)
                            filename = filenames[i]
                            
                            # 使用滑窗推理
                            tile_size = getattr(args, 'tile_size', 256)
                            overlap = getattr(args, 'overlap', tile_size // 2)  # 默认使用50%重叠
                            stride = tile_size - overlap  # 通过tile_size和overlap计算stride
                            
                            # 检查图像尺寸，如果图像尺寸小于或等于tile_size，直接使用整图推理
                            _, _, H, W = image.shape
                            if H <= tile_size and W <= tile_size:
                                # 直接使用整图推理，避免滑窗推理导致的边界反射问题
                                with torch.no_grad():
                                    logits = model(image)
                                    prob_tensor = torch.sigmoid(logits)
                            else:
                                # 使用滑窗推理获取整图预测
                                prob_tensor = sliding_window_inference(
                                    model, 
                                    image, 
                                    tile_size=tile_size, 
                                    stride=stride, 
                                    device=device
                                )
                            
                            # 将概率图转换回logits，以便后续函数正确处理
                            # sliding_window_inference返回的是概率图（已应用sigmoid）
                            # 但compute_metrics等函数期望输入是logits，会在内部再次应用sigmoid
                            prob_tensor = prob_tensor.clamp(1e-6, 1-1e-6)  # 避免log(0)或log(1)
                            logits = torch.log(prob_tensor / (1 - prob_tensor))  # 将概率转换回logits
                            
                            # 将预测结果和掩码移到CPU
                            output_cpu = logits.cpu()  # 使用logits而不是prob_tensor
                            mask_cpu = mask.cpu()
                            
                            # 应用后处理
                            if hasattr(args, 'postprocessing') and args.postprocessing and args.apply_postprocessing_during_training:
                                # 获取后处理参数
                                postprocessing_config = args.postprocessing
                                # 确保预测值是概率值（如果输入是logits，应用sigmoid）
                                if output_cpu.min() < 0 or output_cpu.max() > 1:
                                    prob_tensor = torch.sigmoid(output_cpu)
                                else:
                                    prob_tensor = output_cpu
                                
                                # 根据概率值计算二值预测
                                binary_pred = (prob_tensor > args.threshold).float()
                                
                                # 应用后处理，传入二值预测而非GT mask
                                processed_output = apply_postprocessing_pipeline(
                                    prob_tensor, 
                                    binary_pred,
                                    gaussian_sigma=postprocessing_config.get('gaussian_sigma', 1.0),
                                    median_kernel_size=postprocessing_config.get('median_kernel_size', 3),
                                    morph_close_kernel_size=postprocessing_config.get('morph_close_kernel_size', 3),
                                    morph_open_kernel_size=postprocessing_config.get('morph_open_kernel_size', 0),
                                    min_object_size=postprocessing_config.get('min_object_size', 50),
                                    hole_area_threshold=postprocessing_config.get('hole_area_threshold', 30),
                                    adaptive_threshold=postprocessing_config.get('adaptive_threshold', False)
                                )
                                
                                # 将numpy结果转换回torch张量
                                output_cpu = torch.from_numpy(processed_output).to(mask_cpu.device)
                            
                            # 保存预测结果和掩码
                            all_predictions.append(output_cpu)
                            all_masks.append(mask_cpu)
                            all_filenames.append(filename)
                            
                            # 计算评估指标
                            metrics = compute_metrics(output_cpu, mask_cpu, args.threshold)
                            # 验证指标是否在合理范围内
                            if (0 <= metrics['precision'] <= 1 and 
                                0 <= metrics['recall'] <= 1 and 
                                0 <= metrics['accuracy'] <= 1 and
                                0 <= metrics['iou'] <= 1 and
                                0 <= metrics['dice'] <= 1 and
                                0 <= metrics['f1_score'] <= 1):
                                all_metrics.append(metrics)
                            else:
                                logger.warning(f"样本 {filename} 的指标超出合理范围，已跳过")
                        except Exception as e:
                            logger.warning(f"处理样本 {filenames[i]} 时出错: {e}")
                            continue
                else:
                    # 原有的批次处理逻辑
                    # 前向传播
                    outputs = model(images)
                    
                    # 立即将预测结果和掩码移到CPU，释放GPU内存
                    outputs_cpu = outputs.cpu()
                    masks_cpu = masks.cpu()
                    
                    # 创建临时列表存储经过后处理的预测结果
                    processed_predictions = []
                    
                    # 计算评估指标
                    for i in range(images.size(0)):
                        try:
                            # 应用后处理
                            if hasattr(args, 'postprocessing') and args.postprocessing and args.apply_postprocessing_during_training:
                                # 获取后处理参数
                                postprocessing_config = args.postprocessing
                                # 确保预测值是概率值（如果输入是logits，应用sigmoid）
                                if outputs_cpu[i:i+1].min() < 0 or outputs_cpu[i:i+1].max() > 1:
                                    prob_tensor = torch.sigmoid(outputs_cpu[i:i+1])
                                else:
                                    prob_tensor = outputs_cpu[i:i+1]
                                
                                # 根据概率值计算二值预测
                                binary_pred = (prob_tensor > args.threshold).float()
                                
                                # 应用后处理，传入二值预测而非GT mask
                                processed_output = apply_postprocessing_pipeline(
                                    prob_tensor, 
                                    binary_pred,
                                    gaussian_sigma=postprocessing_config.get('gaussian_sigma', 1.0),
                                    median_kernel_size=postprocessing_config.get('median_kernel_size', 3),
                                    morph_close_kernel_size=postprocessing_config.get('morph_close_kernel_size', 3),
                                    morph_open_kernel_size=postprocessing_config.get('morph_open_kernel_size', 0),
                                    min_object_size=postprocessing_config.get('min_object_size', 50),
                                    hole_area_threshold=postprocessing_config.get('hole_area_threshold', 30),
                                    adaptive_threshold=postprocessing_config.get('adaptive_threshold', False)
                                )
                                
                                # 将numpy结果转换回torch张量
                                processed_output = torch.from_numpy(processed_output).to(masks_cpu[i:i+1].device)
                                processed_predictions.append(processed_output)
                                metrics = compute_metrics(processed_output, masks_cpu[i:i+1], args.threshold)
                            else:
                                # 如果没有后处理，直接使用原始输出
                                processed_predictions.append(outputs_cpu[i:i+1])
                                metrics = compute_metrics(outputs_cpu[i:i+1], masks_cpu[i:i+1], args.threshold)
                            
                            # 验证指标是否在合理范围内
                            if (0 <= metrics['precision'] <= 1 and 
                                0 <= metrics['recall'] <= 1 and 
                                0 <= metrics['accuracy'] <= 1 and
                                0 <= metrics['iou'] <= 1 and
                                0 <= metrics['dice'] <= 1 and
                                0 <= metrics['f1_score'] <= 1):
                                all_metrics.append(metrics)
                            else:
                                logger.warning(f"样本 {filenames[i]} 的指标超出合理范围，已跳过")
                        except Exception as e:
                            logger.warning(f"计算样本 {filenames[i]} 的指标时出错: {e}")
                            continue
                    
                    # 保存经过后处理的预测结果和掩码
                    if processed_predictions:
                        all_predictions.append(torch.cat(processed_predictions, dim=0))
                    all_masks.append(masks_cpu)
                    all_filenames.extend(filenames)
                
                # 定期清理GPU内存
                if torch.cuda.is_available() and batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.error(f"GPU内存不足在批次 {batch_idx}: {e}")
                    # 尝试清理内存并继续
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            except Exception as e:
                logger.error(f"处理批次 {batch_idx} 时出错: {e}")
                continue
    
    # 如果没有有效的指标，返回默认值
    if not all_metrics:
        logger.warning("没有有效的指标计算结果，返回默认值")
        avg_metrics = {
            'iou': 0.0,
            'dice': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'accuracy': 0.0
        }
        classification_report = {}
        return avg_metrics, classification_report, torch.tensor([]), torch.tensor([]), []
    
    # 使用更稳健的方法计算平均指标
    try:
        # 收集所有有效指标
        valid_ious = [m['iou'] for m in all_metrics if not np.isnan(m['iou']) and 0 <= m['iou'] <= 1]
        valid_dices = [m['dice'] for m in all_metrics if not np.isnan(m['dice']) and 0 <= m['dice'] <= 1]
        valid_precisions = [m['precision'] for m in all_metrics if not np.isnan(m['precision']) and 0 <= m['precision'] <= 1]
        valid_recalls = [m['recall'] for m in all_metrics if not np.isnan(m['recall']) and 0 <= m['recall'] <= 1]
        valid_f1s = [m['f1_score'] for m in all_metrics if not np.isnan(m['f1_score']) and 0 <= m['f1_score'] <= 1]
        valid_accuracies = [m['accuracy'] for m in all_metrics if not np.isnan(m['accuracy']) and 0 <= m['accuracy'] <= 1]
        
        # 计算平均值，如果没有有效值则使用0
        avg_metrics = {
            'iou': np.mean(valid_ious) if valid_ious else 0.0,
            'dice': np.mean(valid_dices) if valid_dices else 0.0,
            'precision': np.mean(valid_precisions) if valid_precisions else 0.0,
            'recall': np.mean(valid_recalls) if valid_recalls else 0.0,
            'f1_score': np.mean(valid_f1s) if valid_f1s else 0.0,
            'accuracy': np.mean(valid_accuracies) if valid_accuracies else 0.0
        }
    except Exception as e:
        logger.error(f"计算平均指标时出错: {e}")
        avg_metrics = {
            'iou': 0.0,
            'dice': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'accuracy': 0.0
        }
    
    # 计算总体混淆矩阵
    try:
        # 检查数据量大小，如果太大则分批处理
        total_elements = sum(pred.numel() for pred in all_predictions)
        logger.info(f"预测数据总大小: {total_elements} 元素")
        

        # 如果数据量很大，使用分批处理
        if total_elements > 100000000:  # 100M 元素
            logger.info("数据量较大，使用分批处理计算分类报告")
            
            # 分批处理，避免一次性加载所有数据
            batch_size = 5  # 每次处理5个批次
            all_preds_list = []
            all_masks_list = []
            
            for i in range(0, len(all_predictions), batch_size):
                end_idx = min(i + batch_size, len(all_predictions))
                batch_preds = torch.cat(all_predictions[i:end_idx])
                batch_masks = torch.cat(all_masks[i:end_idx])
                
                # 确保预测值是概率值（如果输入是logits，应用sigmoid）
                if batch_preds.min() < 0 or batch_preds.max() > 1:
                    batch_preds = torch.sigmoid(batch_preds)
                
                # 限制预测值范围
                batch_preds = torch.clamp(batch_preds, 0.0, 1.0)
                
                all_preds_list.append(batch_preds)
                all_masks_list.append(batch_masks)
                
                # 释放内存
                del batch_preds, batch_masks
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # 合并所有批次
            all_preds_tensor = torch.cat(all_preds_list)
            all_masks_tensor = torch.cat(all_masks_list)
            
            # 释放临时列表
            del all_preds_list, all_masks_list
        else:
            # 数据量不大，直接处理
            all_preds_tensor = torch.cat(all_predictions)
            all_masks_tensor = torch.cat(all_masks)
            
            # 确保预测值是概率值（如果输入是logits，应用sigmoid）
            # 确保预测值是概率值（如果输入是logits，应用sigmoid）
            # 注意：对于improved_performance_weighted且ensemble_method为prob_weighted的情况，模型已经返回概率值
            if args.ensemble_strategy == 'improved_performance_weighted':
                # 检查配置文件中的ensemble_method
                ensemble_method = 'logits_weighted'  # 默认值
                if hasattr(args, 'improved_performance_weighted_config') and args.improved_performance_weighted_config:
                    ensemble_method = args.improved_performance_weighted_config.get('ensemble_method', 'logits_weighted')
                
                if ensemble_method == 'prob_weighted' or ensemble_method == 'gated_ensemble':
                    # 模型已经返回概率值，不需要转换
                    pass
                else:
                    # 模型返回logits，需要转换为概率
                    if all_preds_tensor.min() < 0 or all_preds_tensor.max() > 1:
                        all_preds_tensor = torch.sigmoid(all_preds_tensor)
            else:
                # 其他集成策略，检查是否需要转换
                if all_preds_tensor.min() < 0 or all_preds_tensor.max() > 1:
                    all_preds_tensor = torch.sigmoid(all_preds_tensor)
            
            # 限制预测值范围
            all_preds_tensor = torch.clamp(all_preds_tensor, 0.0, 1.0)
    except Exception as e:
        logger.error(f"准备预测数据时出错: {e}")
        # 使用空数据继续
        all_preds_tensor = torch.tensor([])
        all_masks_tensor = torch.tensor([])
    
    # 使用更稳健的方法计算分类报告
    try:
        # 计算分类报告
        if args.n_classes == 1:
            # 二值分割
            # 确保预测值是概率值（如果输入是logits，应用sigmoid）
            if all_preds_tensor.numel() > 0:  # 确保张量不为空
                if all_preds_tensor.min() < 0 or all_preds_tensor.max() > 1:
                    all_preds_tensor = torch.sigmoid(all_preds_tensor)
                
                # 限制预测值范围
                all_preds_tensor = torch.clamp(all_preds_tensor, 0.0, 1.0)
            
            # 计算指标
            if all_preds_tensor.numel() > 0:
                # 对于集成模型，我们已经有了概率值，所以直接使用
                # 根据集成方法和阈值类型决定如何处理预测值
                if args.ensemble_strategy == 'improved_performance_weighted':
                    # 检查配置文件中的ensemble_method
                    ensemble_method = 'logits_weighted'  # 默认值
                    if hasattr(args, 'improved_performance_weighted_config') and args.improved_performance_weighted_config:
                        ensemble_method = args.improved_performance_weighted_config.get('ensemble_method', 'logits_weighted')
                    
                    if ensemble_method == 'prob_weighted' or ensemble_method == 'gated_ensemble':
                        # 模型已经返回概率值，直接使用概率与阈值比较
                        # 生成二值预测
                        binary_preds = (all_preds_tensor > args.threshold).float()
                        # 直接使用概率计算指标
                        metrics = compute_metrics_from_prob(all_preds_tensor, all_masks_tensor, args.threshold)
                    else:
                        # 模型返回logits，需要转换为logits以与阈值比较
                        logits_tensor = all_preds_tensor.clamp(1e-6, 1-1e-6)  # 避免log(0)或log(1)
                        logits_tensor = torch.log(logits_tensor / (1 - logits_tensor))  # 将概率转换回logits
                        # 使用compute_metrics函数计算指标，这与单模型评估流程保持一致
                        metrics = compute_metrics(logits_tensor, all_masks_tensor, args.threshold)
                        # 生成二值预测
                        binary_preds = (torch.sigmoid(logits_tensor) > args.threshold).float()
                else:
                    # 其他集成策略，需要将概率转换为logits
                    logits_tensor = all_preds_tensor.clamp(1e-6, 1-1e-6)  # 避免log(0)或log(1)
                    logits_tensor = torch.log(logits_tensor / (1 - logits_tensor))  # 将概率转换回logits
                    # 使用compute_metrics函数计算指标，这与单模型评估流程保持一致
                    metrics = compute_metrics(logits_tensor, all_masks_tensor, args.threshold)
                    # 生成二值预测
                    binary_preds = (torch.sigmoid(logits_tensor) > args.threshold).float()
                
                avg_metrics = metrics  # 更新avg_metrics
            else:
                binary_preds = torch.tensor([])
        else:
            # 多类别分割
            binary_preds = torch.argmax(torch.softmax(all_preds_tensor, dim=1), dim=1, keepdim=True).float()
        
        # 确保所有张量在同一个设备上
        if all_masks_tensor.numel() > 0:
            all_masks_tensor = all_masks_tensor.to(device)
        
        # 计算分类报告
        if binary_preds.numel() > 0 and all_masks_tensor.numel() > 0:
            classification_report = compute_classification_report(binary_preds, all_masks_tensor, args.threshold, batch_size=500)
        else:
            classification_report = {}
    except Exception as e:
        logger.error(f"计算分类报告时出错: {e}")
        classification_report = {}
    
    # 打印评估结果
    logger.info('\n===== 评估结果 =====')
    logger.info(f'iou: {avg_metrics["iou"]:.4f}')
    logger.info(f'dice: {avg_metrics["dice"]:.4f}')
    logger.info(f'precision: {avg_metrics["precision"]:.4f}')
    logger.info(f'recall: {avg_metrics["recall"]:.4f}')
    logger.info(f'f1_score: {avg_metrics["f1_score"]:.4f}')
    logger.info(f'accuracy: {avg_metrics["accuracy"]:.4f}')
    
    # 打印误检率和漏检率（如果存在）
    if 'false_discovery_rate' in avg_metrics:
        logger.info(f'false_discovery_rate (误检率): {avg_metrics["false_discovery_rate"]:.4f}')
    if 'false_negative_rate' in avg_metrics:
        logger.info(f'false_negative_rate (漏检率): {avg_metrics["false_negative_rate"]:.4f}')
    
    logger.info('\n===== 分类报告 =====')
    for class_name, metrics in classification_report.items():
        logger.info(f'类别 {class_name}:')
        for metric_name, metric_value in metrics.items():
            logger.info(f'  {metric_name}: {metric_value:.4f}')
    
    # 如果需要，计算不同阈值下的性能
    if args.n_classes == 1 and len(all_predictions) > 0:
        logger.info('\n===== 阈值分析 =====')
        try:
            # 分析预测值分布
            logger.info("分析预测值分布...")
            all_preds_flat = all_preds_tensor.flatten()
            logger.info(f"预测值统计: min={all_preds_flat.min():.4f}, max={all_preds_flat.max():.4f}, mean={all_preds_flat.mean():.4f}, std={all_preds_flat.std():.4f}")
            
            # 计算不同百分位的预测值
            percentiles = [10, 25, 50, 75, 90, 95, 99]
            for p in percentiles:
                logger.info(f"预测值第{p}百分位: {torch.quantile(all_preds_flat, p/100):.4f}")
            
            # 计算高于不同阈值的像素比例
            high_thresholds = [0.5, 0.6, 0.68, 0.7, 0.8, 0.9]
            for thresh in high_thresholds:
                ratio = (all_preds_flat > thresh).float().mean().item()
                logger.info(f"高于阈值{thresh}的像素比例: {ratio:.6f}")
            
            # 检查是否启用阈值搜索
            if hasattr(args, 'enable_threshold_search') and args.enable_threshold_search:
                # 从配置中获取阈值搜索参数
                if hasattr(args, 'threshold_range') and hasattr(args, 'threshold_search_interval'):
                    min_threshold, max_threshold = args.threshold_range
                    interval = args.threshold_search_interval
                    thresholds = np.arange(min_threshold, max_threshold + interval, interval).tolist()
                    logger.info(f"使用自定义阈值范围: {min_threshold} 到 {max_threshold}, 间隔: {interval}")
                    logger.info(f"总共将测试 {len(thresholds)} 个阈值点")
                else:
                    thresholds = None
                    logger.info("使用默认阈值列表")
                
                # 检查数据大小，如果太大则减少阈值点
                total_elements = all_preds_tensor.numel()
                logger.info(f"预测数据大小: {total_elements} 元素")
                
                # 如果数据量很大，减少阈值点以加快处理
                if total_elements > 100000000 and len(thresholds) > 10:
                    # 使用更大的间隔
                    new_interval = interval * 2
                    thresholds = np.arange(min_threshold, max_threshold + new_interval, new_interval).tolist()
                    logger.info(f"数据量较大，调整阈值间隔为 {new_interval}，减少到 {len(thresholds)} 个阈值点")
                
                # 如果数据量极大，进一步减少阈值点
                if total_elements > 500000000 and len(thresholds) > 5:
                    # 使用更大的间隔
                    new_interval = interval * 4
                    thresholds = np.arange(min_threshold, max_threshold + new_interval, new_interval).tolist()
                    logger.info(f"数据量极大，调整阈值间隔为 {new_interval}，减少到 {len(thresholds)} 个阈值点")
                
                # 使用较小的批次大小来减少内存使用
                logger.info("开始计算不同阈值下的性能指标...")
                
                # 根据数据量动态调整批次大小
                if total_elements > 500000000:
                    batch_size = 50  # 极大数据集使用更小的批次
                elif total_elements > 100000000:
                    batch_size = 100  # 大数据集使用小批次
                else:
                    batch_size = 200  # 默认批次大小
                
                logger.info(f"使用批次大小: {batch_size}")
                
                # 分批处理阈值分析，避免内存不足
                try:
                    threshold_metrics = calculate_threshold_metrics(
                        all_preds_tensor, 
                        all_masks_tensor,
                        thresholds=thresholds,
                        batch_size=batch_size
                    )
                    
                    logger.info(f'最佳阈值: {threshold_metrics["best_threshold"]:.4f}')
                    logger.info(f'最佳F1分数: {threshold_metrics["best_f1_score"]:.4f}')
                    
                    # 打印不同阈值下的指标
                    for i, threshold in enumerate(threshold_metrics['thresholds']):
                        logger.info(f"阈值 {threshold:.2f}: IoU={threshold_metrics['iou'][i]:.4f}, "
                                   f"Dice={threshold_metrics['dice'][i]:.4f}, "
                                   f"F1={threshold_metrics['f1_score'][i]:.4f}")
                except Exception as e:
                    logger.error(f"阈值分析计算失败: {e}")
                    # 如果阈值分析失败，尝试只计算默认阈值
                    logger.info("尝试只计算默认阈值...")
                    try:
                        default_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
                        threshold_metrics = calculate_threshold_metrics(
                            all_preds_tensor, 
                            all_masks_tensor,
                            thresholds=default_thresholds,
                            batch_size=50  # 使用更小的批次大小
                        )
                        
                        logger.info(f'最佳阈值: {threshold_metrics["best_threshold"]:.4f}')
                        logger.info(f'最佳F1分数: {threshold_metrics["best_f1_score"]:.4f}')
                    except Exception as e2:
                        logger.error(f"默认阈值计算也失败: {e2}")
                        logger.info("跳过阈值分析，继续执行...")
            else:
                logger.info("阈值搜索已禁用，跳过阈值分析")
        except Exception as e:
            logger.error(f"阈值分析失败: {e}")
            logger.info("跳过阈值分析，继续执行...")
    
    return avg_metrics, classification_report, all_preds_tensor, all_masks_tensor, all_filenames

def generate_predictions(model, dataloader, output_dir, device, args, reference_dir=None):
    """生成并保存预测结果"""
    logger.info(f'生成预测结果并保存到: {output_dir}')
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 确保模型在评估模式
    model.eval()
    
    # 用于保存所有预测结果的列表
    all_predictions = []
    all_filenames = []
    
    with torch.no_grad():
        # 创建进度条
        pbar = tqdm(dataloader, desc='生成预测', leave=False)
        
        if args.use_sliding_window:
            # 滑窗模式：逐张图像处理
            for batch in pbar:
                # 获取数据
                images = batch['image'].to(device)
                filenames = batch['filename']
                
                # 逐张图像处理
                for i, filename in enumerate(filenames):
                    # 提取单张图像
                    image = images[i].unsqueeze(0)  # 添加批次维度
                    
                    # 设置滑窗参数
                    tile_size = getattr(args, 'tile_size', 256)
                    stride = getattr(args, 'overlap', 128)
                    
                    # 检查图像尺寸是否小于等于tile_size
                    _, _, h, w = image.shape
                    if h <= tile_size and w <= tile_size:
                        # 小尺寸图像使用整图推理
                        try:
                            logger.info(f"图像尺寸({h}x{w})小于等于tile_size({tile_size})，使用整图推理: {filename}")
                            with torch.no_grad():
                                logits = model(image)
                                prob_tensor = torch.sigmoid(logits)
                            prediction = prob_tensor
                            
                            # 将预测结果移到CPU并保存
                            prediction_cpu = prediction.cpu()
                            all_predictions.append(prediction_cpu)
                            all_filenames.append(filename)
                            
                            # 保存预测结果
                            postprocessing_config = getattr(args, 'postprocessing', None)
                            save_predictions(prediction_cpu, [filename], output_dir, args.threshold, is_probabilities=True, postprocessing_config=postprocessing_config, reference_dir=reference_dir)
                            
                        except Exception as e:
                            logger.error(f"整图推理处理图像 {filename} 失败: {e}")
                            continue
                    else:
                        # 大尺寸图像使用滑窗推理
                        try:
                            logger.info(f"使用滑窗推理处理图像: {filename}")
                            prediction = sliding_window_inference(
                                model, 
                                image, 
                                tile_size=tile_size, 
                                stride=stride,
                                device=device
                            )
                            
                            # 将预测结果移到CPU并保存
                            prediction_cpu = prediction.cpu()
                            all_predictions.append(prediction_cpu)
                            all_filenames.append(filename)
                            
                            # 保存预测结果
                            postprocessing_config = getattr(args, 'postprocessing', None)
                            save_predictions(prediction_cpu, [filename], output_dir, args.threshold, is_probabilities=True, postprocessing_config=postprocessing_config, reference_dir=reference_dir)
                            
                        except Exception as e:
                            logger.error(f"滑窗推理处理图像 {filename} 失败: {e}")
                            # 回退到普通推理
                            try:
                                logger.info(f"回退到普通推理处理图像: {filename}")
                                output = model(image)
                                output_cpu = output.cpu()
                                all_predictions.append(output_cpu)
                                all_filenames.append(filename)
                                postprocessing_config = getattr(args, 'postprocessing', None)
                                save_predictions(output_cpu, [filename], output_dir, args.threshold, is_probabilities=False, postprocessing_config=postprocessing_config, reference_dir=reference_dir)
                            except Exception as e2:
                                logger.error(f"普通推理处理图像 {filename} 也失败: {e2}")
        else:
            # 原有的批次处理逻辑
            for batch in pbar:
                # 获取数据
                images = batch['image'].to(device)
                filenames = batch['filename']
                
                # 前向传播
                outputs = model(images)
                
                # 保存预测结果和文件名
                all_predictions.append(outputs.cpu())
                all_filenames.extend(filenames)
                
                # 保存当前批次的预测结果
                postprocessing_config = getattr(args, 'postprocessing', None)
                save_predictions(outputs, filenames, output_dir, args.threshold, is_probabilities=False, postprocessing_config=postprocessing_config, reference_dir=reference_dir)
    
    logger.info(f'预测结果已保存到 {output_dir}')
    

    return torch.cat(all_predictions), all_filenames

def save_evaluation_results_to_csv(avg_metrics, classification_report, output_dir, model_name, timestamp=None):
    """将评估结果保存为CSV文件"""
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建CSV文件名
    csv_filename = f"evaluation_results_{model_name}_{timestamp}.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    
    # 准备数据
    data = []
    
    # 添加平均指标
    for metric_name, metric_value in avg_metrics.items():
        # 为误检率和漏检率添加中文说明，保持与其他指标一致的格式
        if metric_name == 'false_discovery_rate':
            display_name = 'false_discovery_rate (误检率)'
        elif metric_name == 'false_negative_rate':
            display_name = 'false_negative_rate (漏检率)'
        else:
            display_name = metric_name
            
        data.append({
            'Type': 'Average Metrics',
            'Category': 'Overall',
            'Metric': display_name,
            'Value': metric_value
        })
    
    # 添加分类报告
    for class_name, metrics in classification_report.items():
        for metric_name, metric_value in metrics.items():
            # 为误检率和漏检率添加中文说明，保持与其他指标一致的格式
            if metric_name == 'false_discovery_rate':
                display_name = 'false_discovery_rate (误检率)'
            elif metric_name == 'false_negative_rate':
                display_name = 'false_negative_rate (漏检率)'
            else:
                display_name = metric_name
                
            data.append({
                'Type': 'Classification Report',
                'Category': class_name,
                'Metric': display_name,
                'Value': metric_value
            })
    
    # 创建DataFrame并保存为CSV
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    
    logger.info(f"评估结果已保存到: {csv_path}")
    
    return csv_path

def plot_prediction_examples(images, masks, predictions, filenames, num_examples=5, threshold=0.5, output_dir=None, transforms=None, crs=None):
    """绘制预测示例"""
    logger.info(f'绘制 {num_examples} 个预测示例...')
    
    # 设置字体（使用try-except处理可能的字体错误）
    try:
        # 尝试设置支持中文的字体，移除不存在的Arial Unicode MS
        plt.rcParams["font.family"] = ["DejaVu Sans", "Microsoft YaHei", "SimHei"]
    except Exception as e:
        logger.warning(f"设置字体时出错，使用默认字体: {e}")
        # 使用默认字体
        plt.rcParams["font.family"] = ["DejaVu Sans"]
    
    # 顺序遍历示例
    num_samples = min(num_examples, len(images))
    
    # 对每个示例进行绘图
    for idx in range(num_samples):
        # 获取数据
        # 如果传入的是torch tensor，需要转换为numpy
        if hasattr(images[idx], 'cpu'):
            image = images[idx].cpu().numpy()
        else:
            image = images[idx]
            
        if hasattr(masks[idx], 'cpu'):
            mask = masks[idx].cpu().numpy()
        else:
            mask = masks[idx]
            
        if hasattr(predictions[idx], 'cpu'):
            prediction = predictions[idx].cpu().numpy()
        else:
            prediction = predictions[idx]
            
        filename = filenames[idx]
        
        # 创建画布
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # 绘制RGB图像（如果有多个波段）
        if image.shape[0] >= 3:
            # 选择红、绿、蓝三个波段（假设顺序是蓝、绿、红或其他）
            rgb_image = np.zeros((image.shape[1], image.shape[2], 3))
            
            # 根据实际波段顺序调整
            # 这里假设波段顺序是：蓝、绿、红、近红、短波红外1、短波红外2
            if image.shape[0] >= 3:
                rgb_image[:, :, 0] = image[2]  # 红波段
                rgb_image[:, :, 1] = image[1]  # 绿波段
                rgb_image[:, :, 2] = image[0]  # 蓝波段
            
            # 归一化到[0, 1]范围
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
        
        # 绘制真实掩码
        # 确保掩码值在0-1范围内，并使用适当的颜色映射
        mask_display = mask[0]
        axes[1].imshow(mask_display, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title('真实掩码')
        axes[1].axis('off')
        
        # 绘制预测概率图
        if prediction.shape[0] == 1:
            pred = prediction[0]
            # 使用5%和95%分位数作为显示范围，以便更好地区分高概率区域的细微差别
            axes[2].imshow(
                pred,
                cmap='jet',
                vmin=np.percentile(pred, 5),
                vmax=np.percentile(pred, 95)
            )
        else:
            # 多类别情况，显示概率最高的类别
            axes[2].imshow(np.argmax(prediction, axis=0), cmap='jet')
        axes[2].set_title('预测概率')
        axes[2].axis('off')
        
        # 绘制二值预测结果
        if prediction.shape[0] == 1:
            pred = prediction[0]
            binary_pred = (pred > threshold).astype(np.float32)
            
            # 创建差异图
            mask_bin = mask_display.astype(np.uint8)
            pred_bin = binary_pred.astype(np.uint8)
            
            diff = np.zeros_like(pred_bin, dtype=int)
            diff[(pred_bin == 1) & (mask_bin == 0)] = 1    # 误检
            diff[(pred_bin == 0) & (mask_bin == 1)] = -1   # 漏检
            
            # 创建5个子图以显示差异图
            fig, axes = plt.subplots(1, 5, figsize=(25, 5))
            
            # 重新绘制前4个图
            if image.shape[0] >= 3:
                axes[0].imshow(rgb_image)
            else:
                axes[0].imshow(image[0], cmap='gray')
            axes[0].set_title('输入图像')
            axes[0].axis('off')
            
            axes[1].imshow(mask_display, cmap='gray', vmin=0, vmax=1)
            axes[1].set_title('真实掩码')
            axes[1].axis('off')
            
            im = axes[2].imshow(
                pred,
                cmap='jet',
                vmin=np.percentile(pred, 5),
                vmax=np.percentile(pred, 95)
            )
            axes[2].set_title('预测概率')
            axes[2].axis('off')
            # 为预测概率图添加颜色条
            plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
            
            axes[3].imshow(binary_pred, cmap='gray', vmin=0, vmax=1)
            axes[3].set_title('二值预测')
            axes[3].axis('off')
            
            # 绘制差异图
            diff_im = axes[4].imshow(diff, cmap='bwr', vmin=-1, vmax=1)
            axes[4].set_title('差异图 (红=预测多出的水体, 蓝=漏检区域)')
            axes[4].axis('off')
            # 为差异图添加颜色条
            cbar = plt.colorbar(diff_im, ax=axes[4], fraction=0.046, pad=0.04)
            cbar.set_label('预测差异')
            cbar.set_ticks([-1, 0, 1])
            cbar.set_ticklabels(['漏检', '正确', '误检'])
            
            # 导出差异GeoTIFF
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
                image_stem = os.path.splitext(filename)[0]
                diff_tif = os.path.join(output_dir, f"{image_stem}_difference.tif")
                
                # 使用传入的transform和crs，如果没有传入则使用默认值
                from rasterio.transform import from_origin
                current_transform = transforms[idx] if transforms is not None and idx < len(transforms) else from_origin(0, 0, 1, 1)
                current_crs = crs[idx] if crs is not None and idx < len(crs) else 'EPSG:4326'
                
                diff_profile = {
                    "driver": "GTiff",
                    "height": diff.shape[0],
                    "width": diff.shape[1],
                    "count": 1,
                    "dtype": "int8",
                    "crs": current_crs,
                    "transform": current_transform,
                    "compress": "lzw"
                }
                
                with rasterio.open(diff_tif, "w", **diff_profile) as dst:
                    dst.write(diff.astype(np.int8), 1)
                logger.info(f'差异GeoTIFF已保存到: {diff_tif}')
                
                # 导出差异PNG
                diff_fig = plt.figure(figsize=(6, 6), dpi=300)
                plt.imshow(diff, cmap="bwr", vmin=-1, vmax=1)
                plt.axis("off")
                diff_png = os.path.join(output_dir, f"{image_stem}_difference.png")
                diff_fig.savefig(diff_png, bbox_inches="tight", pad_inches=0)
                plt.close(diff_fig)
                logger.info(f'差异PNG已保存到: {diff_png}')
        else:
            # 多类别情况，显示概率最高的类别
            binary_pred = np.argmax(prediction, axis=0).astype(np.float32)
            axes[3].imshow(binary_pred, cmap='gray', vmin=0, vmax=1)
            axes[3].set_title('二值预测')
            axes[3].axis('off')
        
        # 添加文件名
        plt.suptitle(f'文件: {filename}', fontsize=16)
        plt.tight_layout()
        
        # 保存图像
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            plot_filename = os.path.splitext(filename)[0] + '.png'
            plot_path = os.path.join(output_dir, plot_filename)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f'预测示例已保存到: {plot_path}')
        
        plt.close(fig)  # 关闭图形以释放内存


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 使用配置文件更新参数（如果提供了配置文件）
    args = update_args_with_config(args)
    
    # 调试输出：确认滑窗参数是否正确加载
    logger.info(f"use_sliding_window={args.use_sliding_window}, tile_size={getattr(args, 'tile_size', None)}, overlap={getattr(args, 'overlap', None)}")
    
    # 记录最终使用的集成策略
    logger.info(f"最终 ensemble_strategy: {args.ensemble_strategy}")
    
    # 验证参数
    if hasattr(args, 'use_ensemble') and args.use_ensemble:
        # 从配置文件加载的集成模型配置
        if not hasattr(args, 'models') or not args.models:
            raise ValueError("集成模式需要指定models参数")
        
        # 从models配置中提取模型名称和检查点路径
        model_names = []
        checkpoint_paths = []
        weights = []
        
        for model_config in args.models:
            if isinstance(model_config, dict):
                model_names.append(model_config['name'])
                checkpoint_paths.append(model_config['checkpoint_path'])
                weights.append(model_config.get('ens_weight', 1.0))
        
        args.models = model_names
        args.checkpoint_paths = checkpoint_paths
        args.weights = weights
        args.ensemble = True
        
        logger.info(f"使用集成模型: {model_names}")
        logger.info(f"模型权重: {weights}")
    elif args.ensemble:
        # 命令行指定的集成模型配置
        if args.models is None or len(args.models) == 0 or args.checkpoint_paths is None or len(args.checkpoint_paths) == 0:
            raise ValueError("集成模式需要指定models和checkpoint_paths参数")
        if len(args.models) != len(args.checkpoint_paths):
            raise ValueError("模型类型数量必须与检查点路径数量匹配")
    else:
        # 单模型配置
        if args.model is None or args.checkpoint_path is None:
            raise ValueError("单模型评估需要指定model和checkpoint_path参数")
    
    # 设置随机种子以确保可重复性
    if hasattr(get_config(), 'train') and hasattr(get_config()['train'], 'SEED'):
        seed = get_config()['train'].SEED
    else:
        seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 加载数据集
    if args.single_image:
        logger.info(f"加载单张影像数据集...")
    else:
        logger.info(f"加载{args.split}数据集...")
    
    # 检查是否启用单张影像评估模式
    if args.single_image:
        # 验证单张影像评估所需参数
        if not args.image_path:
            raise ValueError("单张影像评估模式需要指定--image_path参数。例如：--image_path path/to/image.tif")
        if not args.mask_path:
            raise ValueError("单张影像评估模式需要指定--mask_path参数。例如：--mask_path path/to/mask.tif")
        
        # 检查文件是否存在，提供更详细的错误信息
        if not os.path.exists(args.image_path):
            raise FileNotFoundError(f"图像文件不存在: {args.image_path}。请检查文件路径是否正确，或确认文件扩展名是否为.tif、.tiff或.png。")
        if not os.path.exists(args.mask_path):
            raise FileNotFoundError(f"掩膜文件不存在: {args.mask_path}。请检查文件路径是否正确，或确认文件扩展名是否为.tif、.tiff、.png、.jpg或.jpeg。")
        
        # 创建单张影像数据集
        from utils.data_utils import SingleImageDataset
        try:
            dataset = SingleImageDataset(
                image_path=args.image_path,
                mask_path=args.mask_path,
                bands=args.bands,
                normalize_method='sentinel'
            )
        except Exception as e:
            raise ValueError(f"创建单张影像数据集时出错: {e}。请检查图像和掩膜文件格式是否正确。")
        
        logger.info(f"使用单张影像评估模式:")
        logger.info(f"  - 图像文件: {args.image_path}")
        logger.info(f"  - 掩膜文件: {args.mask_path}")
        logger.info(f"  - 使用的波段: {args.bands if args.bands else [1, 2, 3, 4, 5, 6]}")
    else:
        # 从配置文件获取图像和掩码路径
        images_dir = None
        masks_dir = None
        splits_dir = 'splits'  # 默认使用 splits 目录
        data_dir = args.data_dir  # 默认使用命令行参数中的data_dir
        
        if args.config is not None:
            config = get_config(yaml_config_path=args.config)
            if 'data' in config and isinstance(config['data'], dict):
                # 优先使用配置文件中的路径设置
                if 'data_dir' in config['data']:
                    data_dir = config['data']['data_dir']
                if 'images_dir' in config['data']:
                    images_dir = config['data']['images_dir']
                if 'masks_dir' in config['data']:
                    masks_dir = config['data']['masks_dir']
                if 'splits_dir' in config['data']:
                    splits_dir = config['data']['splits_dir']
                # 兼容旧的配置格式
                if 'images' in config['data']:
                    images_dir = config['data']['images']
                if 'masks' in config['data']:
                    masks_dir = config['data']['masks']
                if 'splits' in config['data']:
                    splits_config = config['data']['splits']
                    # 如果splits是字典，提取目录路径
                    if isinstance(splits_config, dict):
                        # 从train/val/test路径中提取目录路径
                        for split_type in ['train', 'val', 'test']:
                            if split_type in splits_config:
                                split_path = splits_config[split_type]
                                # 提取目录部分
                                splits_dir = os.path.dirname(split_path)
                                break
                    else:
                        # 如果splits是字符串，直接使用
                        splits_dir = splits_config
        
        dataset = Sentinel2WaterDataset(
            data_dir=data_dir,
            split=args.split,
            bands=args.bands,
            augment=False,
            normalize_method='sentinel',
            splits_dir=splits_dir,
            images_dir=images_dir,
            masks_dir=masks_dir
        )

    reference_dir = os.path.dirname(args.image_path) if args.single_image else getattr(dataset, 'image_dir', None)

    # 记录数据集大小
    logger.info(f"数据集大小: {len(dataset)} 个样本")
    
    # 创建数据加载器
    # 根据可用内存调整数据加载器配置
    # 使用更小的批大小和工作进程数以减少内存使用
    effective_batch_size = min(args.batch_size, 2)  # 限制最大批大小为2
    effective_num_workers = min(args.num_workers, 1)  # 限制最大工作进程数为1
    
    logger.info(f"使用优化的数据加载器配置: batch_size={effective_batch_size}, num_workers={effective_num_workers}")
    
    dataloader = create_data_loader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=effective_num_workers,
        pin_memory=False,  # 确保不使用pin_memory以减少内存压力
        memory_optimized=True  # 启用内存优化模式
    )
    
    # 设置模型
    if args.ensemble:
        model = setup_ensemble_model(args)
    else:
        model = setup_single_model(args)
    
    # 评估模型
    if args.single_image:
        logger.info("开始评估单张影像...")
    else:
        logger.info("开始评估模型...")
    avg_metrics, classification_report, all_preds, all_masks, all_filenames = evaluate_model(model, dataloader, args.device, args)
    
    # 生成预测结果
    if args.predict:
        # 确定输出目录
        if args.ensemble:
            output_dir = os.path.join(args.output_dir, f'ensemble_{"_+".join(args.models)}')
        else:
            output_dir = os.path.join(args.output_dir, args.model)
        
        # 生成预测
        predictions, filenames = generate_predictions(model, dataloader, output_dir, args.device, args, reference_dir=reference_dir)
    
    # 绘制预测示例
    if args.plot_examples and len(all_preds) > 0:
        # 确定示例输出目录
        examples_dir = os.path.join(args.output_dir, 'examples')
        if args.ensemble:
            examples_dir = os.path.join(examples_dir, f'ensemble_{"_+".join(args.models)}')
        else:
            examples_dir = os.path.join(examples_dir, args.model)
        
        # 只抽取需要展示的样本，避免将整套数据都堆进内存
        num_samples = min(args.num_examples, all_preds.shape[0])
        selected_indices = np.random.choice(all_preds.shape[0], num_samples, replace=False)
        
        # 将numpy索引转换为PyTorch张量
        index_tensor = torch.as_tensor(selected_indices, dtype=torch.long, device='cpu')
        
        # 获取选中的原始图像数据
        selected_images = []
        for idx in selected_indices:
            sample = dataset[int(idx)]
            selected_images.append(sample['image'])
        
        # 将图像列表转换为张量
        selected_images = torch.stack(selected_images)
        
        # 确保所有张量在同一设备上
        masks_device = all_masks.device
        preds_device = all_preds.device
        
        # 将索引张量移动到与掩码相同的设备
        index_tensor = index_tensor.to(masks_device)
        
        # 使用index_select获取选中的掩码和预测
        selected_masks = all_masks.index_select(0, index_tensor)
        
        # 将索引张量移动到与预测相同的设备
        index_tensor = index_tensor.to(preds_device)
        selected_preds = all_preds.index_select(0, index_tensor)
        selected_filenames = [all_filenames[int(i)] for i in selected_indices]
        
        # 获取选中样本的地理参考信息
        selected_transforms = []
        selected_crs = []
        for idx in selected_indices:
            sample = dataset[int(idx)]
            # 检查dataset是否有transform和crs属性
            if hasattr(dataset, 'transforms') and dataset.transforms is not None:
                selected_transforms.append(dataset.transforms[int(idx)])
            else:
                # 如果是SingleImageDataset或直接打开的rasterio对象，尝试获取transform
                if hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'transform'):
                    selected_transforms.append(dataset.dataset.transform)
                else:
                    # 默认仿射变换
                    from rasterio.transform import from_origin
                    selected_transforms.append(from_origin(0, 0, 1, 1))
            
            if hasattr(dataset, 'crs') and dataset.crs is not None:
                selected_crs.append(dataset.crs[int(idx)])
            else:
                # 如果是SingleImageDataset或直接打开的rasterio对象，尝试获取crs
                if hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'crs'):
                    selected_crs.append(dataset.dataset.crs)
                else:
                    # 默认坐标系
                    selected_crs.append('EPSG:4326')
        
        # 绘制示例
        plot_prediction_examples(
            selected_images,  # 使用选中的原始图像
            selected_masks,
            selected_preds,
            selected_filenames,
            num_examples=num_samples,  # 使用实际抽取的样本数
            threshold=args.threshold,
            output_dir=examples_dir,
            transforms=selected_transforms,
            crs=selected_crs
        )
        
        # 将评估结果保存为CSV文件，与可视化图像放在同一个文件夹下
        model_name = f'ensemble_{"_+".join(args.models)}' if args.ensemble else args.model
        save_evaluation_results_to_csv(avg_metrics, classification_report, examples_dir, model_name)
    else:
        # 如果不绘制示例，仍然保存评估结果为CSV文件
        results_dir = os.path.join(args.output_dir, 'results')
        if args.ensemble:
            results_dir = os.path.join(results_dir, f'ensemble_{"_+".join(args.models)}')
        else:
            results_dir = os.path.join(results_dir, args.model)
        
        model_name = f'ensemble_{"_+".join(args.models)}' if args.ensemble else args.model
        save_evaluation_results_to_csv(avg_metrics, classification_report, results_dir, model_name)
    
    if args.single_image:
        logger.info("单张影像评估完成!")
    else:
        logger.info("评估完成!")

if __name__ == '__main__':
    main()
