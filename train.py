import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import yaml
import os
from datetime import datetime
from tqdm import tqdm
import logging
import time
import numpy as np
import json
from torch.optim import lr_scheduler

# 多进程保护措施
if __name__ == '__main__':
    # 设置多进程启动方法
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

from utils.losses import FocalLoss, LovaszLoss, FocalLovaszLoss, BCEDiceLoss, BCEFocalLovaszLoss
from utils.data_utils import create_data_loader
from utils.metrics import compute_metrics, calculate_threshold_metrics, compute_global_binary_metrics
from models.aer_unet import get_aer_unet_model
from config import get_config

# ================== 日志 ==================
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler("training.log"),
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
    parser = argparse.ArgumentParser(description='训练Sentinel-2水体分割模型')

    # 配置文件路径
    parser.add_argument('--config', type=str, default=None, help='YAML配置文件路径')

    # 数据参数
    parser.add_argument('--data_dir', type=str, default='datasets', help='数据集目录路径')
    parser.add_argument('--images_dir', type=str, default=None, help='图像目录路径（可选）')
    parser.add_argument('--masks_dir', type=str, default=None, help='掩码目录路径（可选）')
    parser.add_argument('--splits_dir', type=str, default=None, help='splits 目录（可选，包含 train/val/test.txt）')
    parser.add_argument('--normalize_method', type=str, default='sentinel', choices=['minmax', 'zscore', 'sentinel'], help='归一化方法')
    parser.add_argument('--prefetch_factor', type=int, default=None, help='DataLoader 预取因子（num_workers>0 时生效）')
    parser.add_argument('--augment', action='store_true', help='是否启用基础数据增强')
    parser.add_argument('--no_augment', dest='augment', action='store_false', help='禁用基础数据增强')
    parser.set_defaults(augment=None)
    parser.add_argument('--batch_size', type=int, default=1, help='批量大小')
    parser.add_argument('--num_workers', type=int, default=0, help='数据加载器工作进程数（Windows上建议使用0）')
    parser.add_argument('--pin_memory', action='store_true', help='是否启用内存锁定以加速数据传输到GPU')
    parser.add_argument('--no_pin_memory', action='store_true', help='是否禁用内存锁定')
    parser.add_argument('--persistent_workers', action='store_true', help='是否使用持久化工作进程（Windows上建议禁用）')
    parser.add_argument('--no_persistent_workers', action='store_true', help='是否禁用持久化工作进程')
    parser.add_argument('--force_multiprocess', action='store_true', default=False, help='在Windows上强制使用多进程数据加载（可能导致问题）')
    parser.add_argument('--bands', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6], help='要使用的波段索引列表')

    # 模型参数
    parser.add_argument('--model', type=str, choices=[
        'aer_unet', 'lightweight_unet', 'unet', 'deeplabv3_plus',
        'lightweight_deeplabv3_plus', 'ultra_lightweight_deeplabv3_plus'
    ], required=True, help='要训练的模型类型')
    parser.add_argument('--backbone_type', type=str, choices=['resnet50', 'mobilenet_v2'], 
                        default='resnet50', help='骨干网络类型 (仅对deeplabv3_plus有效)')
    parser.add_argument('--n_classes', type=int, default=1, help='输出类别数')
    parser.add_argument('--base_features', type=int, default=32, help='AER U-Net的基础特征数')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout率')
    parser.add_argument('--output_stride', type=int, default=16, choices=[8, 16, 32], help='DeepLabV3+的输出步长')
    parser.add_argument('--pretrained_backbone', action='store_true', help='是否使用预训练的骨干网络')
    parser.add_argument('--no_pretrained_backbone', dest='pretrained_backbone', action='store_false', help='禁用预训练骨干')
    parser.set_defaults(pretrained_backbone=None)
    parser.add_argument('--freeze_backbone', action='store_true', help='是否冻结骨干网络')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--lr_backbone', type=float, default=None, help='骨干网络学习率（分离学习率设置）')
    parser.add_argument('--lr_head', type=float, default=None, help='头部学习率（分离学习率设置）')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--optimizer', type=str, choices=['adam', 'adamw', 'sgd'], default='adam', help='优化器类型')
    parser.add_argument('--scheduler', type=str, choices=['cosine', 'step', 'plateau'], default='cosine', help='学习率调度器类型')
    parser.add_argument('--step_size', type=int, default=10, help='学习率衰减步长（用于step调度器）')
    parser.add_argument('--gamma', type=float, default=0.1, help='学习率衰减因子')
    parser.add_argument('--patience', type=int, default=5, help='调度器耐心值（兼容旧字段）')
    parser.add_argument('--scheduler_patience', type=int, default=None, help='Plateau专用耐心')
    parser.add_argument('--min_lr', type=float, default=None, help='Plateau专用最小学习率')
    parser.add_argument('--scheduler_cooldown', type=int, default=0, help='Plateau降LR后的冷却期')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='梯度累积步数')

    # 损失函数参数
    parser.add_argument('--loss_type', type=str, choices=['bce', 'bce_logits', 'focal', 'lovasz', 'focal_lovasz', 'bce_dice', 'bce_focal_lovasz'], default='bce', help='损失函数类型')
    parser.add_argument('--focal_weight', type=float, default=0.5, help='Focal Loss的权重（用于组合损失）')
    parser.add_argument('--lovasz_weight', type=float, default=0.5, help='Lovasz Loss的权重（用于组合损失）')
    parser.add_argument('--bce_weight', type=float, default=0.5, help='BCE Loss的权重（用于组合损失）')
    parser.add_argument('--dice_weight', type=float, default=0.5, help='Dice Loss的权重（用于组合损失）')
    parser.add_argument('--dice_smooth', type=float, default=1e-6, help='Dice Loss的平滑因子，防止分母为零')
    parser.add_argument('--focal_alpha', type=float, default=1.0, help='Focal Loss的alpha参数')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal Loss的gamma参数')

    # 训练技巧参数
    parser.add_argument('--use_amp', action='store_true', help='是否使用自动混合精度训练')
    parser.add_argument('--early_stopping', action='store_true', help='是否使用早停')
    parser.add_argument('--early_stopping_patience', type=int, default=20, help='早停耐心值')
    parser.add_argument('--gradient_clipping', action='store_true', help='是否使用梯度裁剪')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='梯度裁剪的最大范数')

    # 其他参数
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='检查点保存目录')
    parser.add_argument('--resume_from', type=str, default=None, help='从指定的检查点恢复训练')
    parser.add_argument('--save_all_checkpoints', action='store_true', help='保存所有检查点，如果未设置则只保存最佳模型')
    parser.add_argument('--use_timestamp', action='store_true', help='在检查点文件名中添加时间戳')
    parser.add_argument('--tensorboard', action='store_true', help='启用TensorBoard可视化')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='训练设备')
    parser.add_argument('--debug', action='store_true', help='打印调试信息')

    # 检查点保存策略参数
    parser.add_argument('--save_improvement_threshold', type=float, default=0.01, help='Dice提升阈值，超过则保存')
    parser.add_argument('--save_interval_epochs', type=int, default=10, help='固定间隔保存')

    # 骨干网络冻结参数
    parser.add_argument('--freeze_backbone_epochs', type=int, default=0, help='冻结骨干网络的epoch数，之后解冻进行微调')

    # 类别不平衡 & 推理增强 & EMA
    parser.add_argument('--pos_weight', type=float, default=None, help='BCEWithLogitsLoss 的 pos_weight')
    parser.add_argument('--use_tta', action='store_true', help='验证/推理时使用 TTA')
    parser.add_argument('--tta_types', type=str, nargs='+', default=['h_flip', 'v_flip', 'transpose'], help='TTA类型列表')
    parser.add_argument('--ema', action='store_true', help='训练中维护 EMA 权重（验证/保存时使用）')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA 衰减系数')
    parser.add_argument('--rdrop_alpha', type=float, default=0.0, help='R-Drop正则化系数，0表示不使用')

    # BN 冻结（可选）
    parser.add_argument('--freeze_bn_stats', action='store_true', help='是否冻结BN的统计量/仿射参数')

    # CRF后处理参数
    parser.add_argument('--crf_iterations', type=int, default=0, help='CRF后处理迭代次数，0表示不使用CRF')

    # 高级增强参数
    parser.add_argument('--use_advanced_aug', action='store_true', help='是否使用MixUp/CutMix等高级数据增强')
    parser.add_argument('--mixup_alpha', type=float, default=1.0, help='MixUp增强的alpha参数')
    parser.add_argument('--cutmix_alpha', type=float, default=1.0, help='CutMix增强的alpha参数')
    parser.add_argument('--mixup_prob', type=float, default=0.5, help='应用MixUp增强的概率')
    parser.add_argument('--cutmix_prob', type=float, default=0.5, help='应用CutMix增强的概率')
    
    # CBAM注意力参数
    parser.add_argument('--use_cbam', action='store_true', help='是否使用CBAM注意力机制')
    parser.add_argument('--cbam_reduction_ratio', type=int, default=16, help='CBAM通道注意力的降维比例')
    
    # 阈值搜索参数
    parser.add_argument('--enable_threshold_search', action='store_true', help='是否启用阈值搜索')
    parser.add_argument('--threshold_search_interval', type=int, default=5, help='阈值搜索间隔（epoch数）')
    parser.add_argument('--threshold_range', type=float, nargs='+', default=[0.1, 0.9], help='阈值搜索范围')
    parser.add_argument('--minimal_threshold_search', action='store_true', help='是否使用最小化阈值搜索（只测试关键阈值点）')

    return parser.parse_args()


def update_args_with_config(args):
    """使用配置文件更新命令行参数"""
    debug_enabled = getattr(args, 'debug', False)
    if args.config is not None:
        logger.info(f"从配置文件加载参数: {args.config}")
        config = get_config(yaml_config_path=args.config)

        if debug_enabled:
            print("DEBUG loaded config keys:", list(config.keys()))
            import pprint
            print("DEBUG full config:")
            pprint.pp(config)
            print("DEBUG config file absolute path:", os.path.abspath(args.config))

        # 数据参数
        if 'data' in config:
            data_config = config['data']
            if debug_enabled:
                print("DEBUG data config:", data_config)
            if 'data_dir' in data_config:
                args.data_dir = data_config['data_dir']
            if 'images' in data_config:
                args.images_dir = data_config['images']
            if 'masks' in data_config:
                args.masks_dir = data_config['masks']
            if 'images_dir' in data_config:
                args.images_dir = data_config['images_dir']
            if 'masks_dir' in data_config:
                args.masks_dir = data_config['masks_dir']
            if 'normalize_method' in data_config:
                args.normalize_method = data_config['normalize_method']
            if 'NORMALIZE_METHOD' in data_config:
                args.normalize_method = data_config['NORMALIZE_METHOD']
            if 'splits_dir' in data_config:
                args.splits_dir = data_config['splits_dir']
            if 'splits' in data_config:
                splits_config = data_config['splits']
                if isinstance(splits_config, dict):
                    for split_type in ['train', 'val', 'test']:
                        if split_type in splits_config:
                            args.splits_dir = os.path.dirname(splits_config[split_type])
                            break
                elif isinstance(splits_config, str):
                    args.splits_dir = splits_config
            if 'prefetch_factor' in data_config:
                args.prefetch_factor = int(data_config['prefetch_factor'])
            if 'augment' in data_config:
                args.augment = bool(data_config['augment'])
            if 'AUGMENT' in data_config:
                args.augment = bool(data_config['AUGMENT'])
            if 'num_workers' in data_config:
                args.num_workers = int(data_config['num_workers'])
            if 'pin_memory' in data_config:
                args.pin_memory = bool(data_config['pin_memory'])
            if 'persistent_workers' in data_config:
                args.persistent_workers = bool(data_config['persistent_workers'])
            if 'force_multiprocess' in data_config:
                # 确保force_multiprocess参数正确处理
                force_multiprocess_value = data_config['force_multiprocess']
                if isinstance(force_multiprocess_value, str):
                    # 处理字符串形式的布尔值
                    args.force_multiprocess = force_multiprocess_value.lower() in ('true', '1', 'yes', 'on')
                else:
                    args.force_multiprocess = bool(force_multiprocess_value)
                if debug_enabled:
                    print(f"DEBUG set force_multiprocess to {args.force_multiprocess} from config value {force_multiprocess_value}")

        # 训练参数
        if 'train' in config:
            train_config = config['train']
            if 'batch_size' in train_config:
                if debug_enabled:
                    print("DEBUG setting batch_size from train config:", train_config['batch_size'])
                args.batch_size = int(train_config['batch_size'])
            if 'epochs' in train_config:
                args.epochs = int(train_config['epochs'])
            if 'lr' in train_config:
                args.learning_rate = float(train_config['lr'])
            if 'weight_decay' in train_config:
                args.weight_decay = float(train_config['weight_decay'])
            if 'ema' in train_config:
                args.ema = bool(train_config['ema'])
            if 'ema_decay' in train_config:
                args.ema_decay = float(train_config['ema_decay'])
            if 'scheduler' in train_config:
                args.scheduler = train_config['scheduler']
            # 兼容配置文件中的scheduler_type字段
            elif 'scheduler_type' in train_config:
                args.scheduler = train_config['scheduler_type']
            if 'step_size' in train_config:
                args.step_size = int(train_config['step_size'])
            if 'gamma' in train_config:
                args.gamma = float(train_config['gamma'])
            # 兼容配置文件中的scheduler_factor字段
            elif 'scheduler_factor' in train_config:
                args.gamma = float(train_config['scheduler_factor'])
            if 'patience' in train_config:
                args.patience = int(train_config['patience'])
            if 'scheduler_patience' in train_config:
                args.scheduler_patience = int(train_config['scheduler_patience'])
            if 'min_lr' in train_config:
                args.min_lr = float(train_config['min_lr'])
            if 'scheduler_cooldown' in train_config:
                args.scheduler_cooldown = int(train_config['scheduler_cooldown'])
            if 'gradient_accumulation_steps' in train_config:
                args.gradient_accumulation_steps = int(train_config['gradient_accumulation_steps'])
            if 'save_interval_epochs' in train_config:
                args.save_interval_epochs = int(train_config['save_interval_epochs'])
            # 注意：optimizer配置在model部分，不在train部分
            # 添加分离学习率支持
            if 'lr_backbone' in train_config:
                args.lr_backbone = float(train_config['lr_backbone'])
            if 'lr_head' in train_config:
                args.lr_head = float(train_config['lr_head'])
            if 'weight_decay' in train_config:
                args.weight_decay = float(train_config['weight_decay'])
            if 'gradient_accumulation_steps' in train_config:
                args.gradient_accumulation_steps = int(train_config['gradient_accumulation_steps'])
            if 'scheduler_type' in train_config:
                args.scheduler = train_config['scheduler_type']
            if 'scheduler_patience' in train_config:
                args.scheduler_patience = int(train_config['scheduler_patience'])
            if 'scheduler_factor' in train_config:
                args.gamma = float(train_config['scheduler_factor'])
            if 'patience' in train_config:
                args.patience = int(train_config['patience'])
            if 'min_lr' in train_config:
                args.min_lr = float(train_config['min_lr'])
            if 'scheduler_cooldown' in train_config:
                args.scheduler_cooldown = int(train_config['scheduler_cooldown'])
            # 添加优化器类型支持
            if 'optimizer' in train_config:
                args.optimizer = train_config['optimizer']

            # 损失函数参数
            if 'loss_type' in train_config:
                lt = str(train_config['loss_type']).lower()
                if lt in ('bce_logits', 'bcewithlogits', 'bcewithlogitsloss'):
                    lt = 'bce'  # 这里沿用你原来的映射行为（BCEWithLogitsLoss）
                args.loss_type = lt
            if 'focal_weight' in train_config:
                args.focal_weight = float(train_config['focal_weight'])
            if 'lovasz_weight' in train_config:
                args.lovasz_weight = float(train_config['lovasz_weight'])
            if 'bce_weight' in train_config:
                args.bce_weight = float(train_config['bce_weight'])
            if 'dice_weight' in train_config:
                args.dice_weight = float(train_config['dice_weight'])
            if 'dice_smooth' in train_config:
                args.dice_smooth = float(train_config['dice_smooth'])
            if 'focal_alpha' in train_config:
                args.focal_alpha = float(train_config['focal_alpha'])
            if 'focal_gamma' in train_config:
                args.focal_gamma = float(train_config['focal_gamma'])

            # 训练技巧参数
            if 'use_amp' in train_config:
                args.use_amp = bool(train_config['use_amp'])
            if 'early_stopping' in train_config:
                args.early_stopping = bool(train_config['early_stopping'])
            if 'early_stopping_patience' in train_config:
                args.early_stopping_patience = int(train_config['early_stopping_patience'])
            if 'gradient_clipping' in train_config:
                args.gradient_clipping = bool(train_config['gradient_clipping'])
            if 'max_grad_norm' in train_config:
                args.max_grad_norm = float(train_config['max_grad_norm'])
            if 'pos_weight' in train_config:
                args.pos_weight = float(train_config['pos_weight'])
            if 'freeze_backbone_epochs' in train_config:
                args.freeze_backbone_epochs = int(train_config['freeze_backbone_epochs'])
            if 'freeze_backbone' in train_config:
                args.freeze_backbone = bool(train_config['freeze_backbone'])
            # 添加R-Drop正则化支持
            if 'rdrop_alpha' in train_config:
                args.rdrop_alpha = float(train_config['rdrop_alpha'])
            
            # 添加MixUp和CutMix数据增强参数支持
            if 'mixup_alpha' in train_config:
                args.mixup_alpha = float(train_config['mixup_alpha'])
            if 'cutmix_alpha' in train_config:
                args.cutmix_alpha = float(train_config['cutmix_alpha'])
            if 'mixup_prob' in train_config:
                args.mixup_prob = float(train_config['mixup_prob'])
            if 'cutmix_prob' in train_config:
                args.cutmix_prob = float(train_config['cutmix_prob'])
            
            # 阈值搜索参数
            if 'enable_threshold_search' in train_config:
                args.enable_threshold_search = bool(train_config['enable_threshold_search'])
            if 'threshold_search_interval' in train_config:
                args.threshold_search_interval = int(train_config['threshold_search_interval'])
            if 'threshold_range' in train_config:
                args.threshold_range = train_config['threshold_range']
            if 'minimal_threshold_search' in train_config:
                args.minimal_threshold_search = bool(train_config['minimal_threshold_search'])

            # 保存策略
            if 'save_improvement_threshold' in train_config:
                args.save_improvement_threshold = float(train_config['save_improvement_threshold'])
            if 'save_interval_epochs' in train_config:
                args.save_interval_epochs = int(train_config['save_interval_epochs'])

        # 模型参数
        if args.model == 'aer_unet' and 'model' in config:
            model_config = config['model']
            if 'base_features' in model_config:
                args.base_features = model_config['base_features']
            if 'dropout_rate' in model_config:
                args.dropout_rate = model_config['dropout_rate']
            # 优化器配置在model部分
            if 'optimizer' in model_config:
                args.optimizer = model_config['optimizer']
        elif args.model == 'deeplabv3_plus' and 'model' in config:
            model_config = config['model']
            if 'output_stride' in model_config:
                args.output_stride = model_config['output_stride']
            if 'pretrained_backbone' in model_config and args.pretrained_backbone is None:
                args.pretrained_backbone = model_config['pretrained_backbone']
            # 添加backbone_type参数支持
            if 'backbone' in model_config:
                args.backbone_type = model_config['backbone']
        elif args.model == 'lightweight_deeplabv3_plus' and 'lightweight_deeplabv3_plus' in config:
            model_config = config['lightweight_deeplabv3_plus']
            if 'output_stride' in model_config:
                args.output_stride = model_config['output_stride']
            if 'pretrained_backbone' in model_config and args.pretrained_backbone is None:
                args.pretrained_backbone = model_config['pretrained_backbone']
        elif args.model == 'ultra_lightweight_deeplabv3_plus':
            model_config = config.get('model', config.get('ultra_lightweight_deeplabv3_plus', {}))
            if debug_enabled:
                print("DEBUG model config:", model_config)
            if 'output_stride' in model_config:
                args.output_stride = int(model_config['output_stride'])
            if 'pretrained_backbone' in model_config:
                if args.pretrained_backbone is None:
                    args.pretrained_backbone = bool(model_config['pretrained_backbone'])
                if debug_enabled:
                    print("DEBUG pretrained_backbone(final):", args.pretrained_backbone)
            if 'aspp_out' in model_config:
                args.aspp_out = model_config['aspp_out']
            if 'dec_ch' in model_config:
                args.dec_ch = model_config['dec_ch']
            if 'low_ch_out' in model_config:
                args.low_ch_out = model_config['low_ch_out']
            # 添加CBAM注意力机制支持
            if 'use_cbam' in model_config:
                args.use_cbam = bool(model_config['use_cbam'])
            if 'cbam_reduction_ratio' in model_config:
                args.cbam_reduction_ratio = int(model_config['cbam_reduction_ratio'])
            # 添加base_features和dropout_rate支持
            if 'base_filters' in model_config:
                args.base_features = int(model_config['base_filters'])
            if 'dropout_rate' in model_config:
                args.dropout_rate = float(model_config['dropout_rate'])
            # 添加ASPP速率、类别先验和SE注意力机制支持
            if 'aspp_rates' in model_config:
                args.aspp_rates = model_config['aspp_rates']
            if 'class_prior' in model_config:
                args.class_prior = model_config['class_prior']
            if 'use_se' in model_config:
                args.use_se = bool(model_config['use_se'])
        elif args.model == 'unet' and 'model' in config:
            model_config = config['model']
            if 'bilinear' in model_config:
                args.unet_bilinear = bool(model_config['bilinear'])

        # 推理/验证 TTA
        if 'inference' in config:
            infer_cfg = config['inference']
            if 'use_tta' in infer_cfg:
                args.use_tta = bool(infer_cfg['use_tta'])
            # 添加TTA类型支持
            if 'tta_types' in infer_cfg:
                args.tta_types = infer_cfg['tta_types']
            # 添加CRF后处理支持
            if 'crf_iterations' in infer_cfg:
                args.crf_iterations = int(infer_cfg['crf_iterations'])
        
        # 数据增强参数
        if 'data' in config:
            data_config = config['data']
            # 添加高级数据增强支持
            if 'use_advanced_aug' in data_config:
                args.use_advanced_aug = bool(data_config['use_advanced_aug'])
            if 'mixup_alpha' in data_config:
                args.mixup_alpha = float(data_config['mixup_alpha'])
            if 'cutmix_alpha' in data_config:
                args.cutmix_alpha = float(data_config['cutmix_alpha'])
            if 'mixup_prob' in data_config:
                args.mixup_prob = float(data_config['mixup_prob'])
            if 'cutmix_prob' in data_config:
                args.cutmix_prob = float(data_config['cutmix_prob'])
        
        # 从augment部分加载高级数据增强参数（如果data部分没有）
        if 'augment' in config:
            aug_config = config['augment']
            # 只有在data部分没有设置的情况下，才从augment部分加载
            if not hasattr(args, 'use_advanced_aug') or args.use_advanced_aug is None:
                if 'use_advanced_aug' in aug_config:
                    args.use_advanced_aug = bool(aug_config['use_advanced_aug'])
            if 'mixup_alpha' in aug_config:
                args.mixup_alpha = float(aug_config['mixup_alpha'])
            if 'cutmix_alpha' in aug_config:
                args.cutmix_alpha = float(aug_config['cutmix_alpha'])
            if 'mixup_prob' in aug_config:
                args.mixup_prob = float(aug_config['mixup_prob'])
            if 'cutmix_prob' in aug_config:
                args.cutmix_prob = float(aug_config['cutmix_prob'])
            # 处理噪声参数：将配置文件中的gauss_noise_std映射到代码中的noise_std
            if 'gauss_noise_std' in aug_config:
                args.noise_std = float(aug_config['gauss_noise_std'])
            elif 'noise_std' in aug_config:
                args.noise_std = float(aug_config['noise_std'])

    if getattr(args, 'augment', None) is None:
        args.augment = True

    # 打印关键参数
    if debug_enabled:
        print("DEBUG final args.batch_size:", args.batch_size)
        print("DEBUG final args.gradient_accumulation_steps:", args.gradient_accumulation_steps)
        print("DEBUG final args.pretrained_backbone:", args.pretrained_backbone)
        print("DEBUG final args.loss_type:", args.loss_type)
        print("DEBUG final args.pos_weight:", getattr(args, 'pos_weight', None))
        print("DEBUG final args.dice_smooth:", getattr(args, 'dice_smooth', 1e-6))
        print("DEBUG final args.use_tta:", getattr(args, 'use_tta', False))
        print("DEBUG final args.ema:", getattr(args, 'ema', False), "decay:", getattr(args, 'ema_decay', None))
        print("DEBUG final args.num_workers:", args.num_workers)
        print("DEBUG final args.persistent_workers:", args.persistent_workers)
        print("DEBUG final args.force_multiprocess:", args.force_multiprocess)
        # 打印MixUp和CutMix参数
        print("DEBUG final args.mixup_alpha:", getattr(args, 'mixup_alpha', None))
        print("DEBUG final args.cutmix_alpha:", getattr(args, 'cutmix_alpha', None))
        print("DEBUG final args.mixup_prob:", getattr(args, 'mixup_prob', None))
        print("DEBUG final args.cutmix_prob:", getattr(args, 'cutmix_prob', None))

    return args


def setup_model(args):
    """根据参数设置模型"""
    if args.model == 'aer_unet':
        model = get_aer_unet_model(
            n_channels=len(args.bands),
            n_classes=args.n_classes,
            base_features=args.base_features,
            dropout_rate=args.dropout_rate
        )
    elif args.model == 'lightweight_unet':
        from models.lightweight_unet import get_lightweight_unet_model
        model = get_lightweight_unet_model(
            n_channels=len(args.bands),
            n_classes=args.n_classes
        )
    elif args.model == 'unet':
        from models.unet_model import get_unet_model
        model = get_unet_model(
            n_channels=len(args.bands),
            n_classes=args.n_classes,
            bilinear=getattr(args, 'unet_bilinear', False)
        )
    elif args.model == 'deeplabv3_plus':
        from models.deeplabv3_plus import get_deeplabv3_plus_model
        # Get backbone_type from args if it exists, default to 'resnet50'
        backbone_type = getattr(args, 'backbone_type', 'resnet50')
        model = get_deeplabv3_plus_model(
            n_channels=len(args.bands),
            n_classes=args.n_classes,
            output_stride=args.output_stride,
            pretrained_backbone=args.pretrained_backbone,
            backbone_type=backbone_type
        )
    elif args.model == 'lightweight_deeplabv3_plus':
        from models.lightweight_deeplabv3_plus import get_lightweight_deeplabv3_plus_model
        model = get_lightweight_deeplabv3_plus_model(
            n_channels=len(args.bands),
            n_classes=args.n_classes,
            output_stride=args.output_stride,
            pretrained_backbone=args.pretrained_backbone
        )
    elif args.model == 'ultra_lightweight_deeplabv3_plus':
        from models.ultra_lightweight_deeplabv3_plus import get_ultra_light_deeplabv3_plus
        model = get_ultra_light_deeplabv3_plus(
            n_channels=len(args.bands),
            n_classes=args.n_classes,
            pretrained_backbone=args.pretrained_backbone,
            aspp_out=getattr(args, 'aspp_out', 64),
            dec_ch=getattr(args, 'dec_ch', 64),
            low_ch_out=getattr(args, 'low_ch_out', 32),
            use_cbam=getattr(args, 'use_cbam', False),
            cbam_reduction_ratio=getattr(args, 'cbam_reduction_ratio', 16),
            output_stride=getattr(args, 'output_stride', 32),
            aspp_rates=getattr(args, 'aspp_rates', None),
            class_prior=getattr(args, 'class_prior', None),
            use_se=getattr(args, 'use_se', False)
        )

    # 给模型补充 freeze/unfreeze 方法
    def _freeze_backbone(m):
        if hasattr(m, 'backbone'):
            for p in m.backbone.parameters():
                p.requires_grad = False

    def _unfreeze_backbone(m):
        if hasattr(m, 'backbone'):
            for p in m.backbone.parameters():
                p.requires_grad = True

    setattr(model, 'freeze_backbone', lambda: _freeze_backbone(model))
    setattr(model, 'unfreeze_backbone', lambda: _unfreeze_backbone(model))

    # 可选：冻结 BN 统计与仿射
    if getattr(args, 'freeze_bn_stats', False):
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                m.eval()
                m.weight.requires_grad_(False)
                m.bias.requires_grad_(False)

    return model.to(args.device)


def setup_optimizer_and_scheduler(model, args):
    """设置优化器和学习率调度器"""
    # 检查是否有有效的分离学习率设置
    has_separate_lr = (hasattr(args, 'lr_backbone') and hasattr(args, 'lr_head') and 
                      args.lr_backbone is not None and args.lr_head is not None)
    
    if has_separate_lr:
        # 分离学习率设置
        print(f"使用分离学习率: 骨干网络 {args.lr_backbone}, 头部 {args.lr_head}")
        
        # 分离骨干网络和头部的参数
        backbone_params = []
        head_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
                
            # 判断参数是否属于骨干网络
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
        
        # 为不同部分设置不同学习率
        params = [
            {'params': backbone_params, 'lr': args.lr_backbone},
            {'params': head_params, 'lr': args.lr_head}
        ]
        
        # 使用统一的学习率作为默认值（如果优化器需要）
        lr = args.lr_head  # 使用头部学习率作为主要学习率
    else:
        # 统一学习率设置
        params = [p for p in model.parameters() if p.requires_grad]
        lr = args.learning_rate
        print(f"使用统一学习率: {lr}")
    
    # 设置优化器
    if args.optimizer == 'adam':
        optimizer = optim.Adam(params, lr=lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(params, lr=lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"不支持的优化器类型: {args.optimizer}")

    # 设置学习率调度器
    if args.scheduler == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == 'plateau':
        patience = args.scheduler_patience if args.scheduler_patience is not None else max(1, args.patience // 1)
        min_lr = args.min_lr if args.min_lr is not None else 0.0
        cooldown = getattr(args, 'scheduler_cooldown', 0)
        
        # 根据配置文件中的scheduler_metric参数设置监控指标
        scheduler_metric = getattr(args, 'scheduler_metric', 'dice').lower()
        if scheduler_metric == 'recall':
            mode = 'max'
            print("学习率调度器将监控验证召回率 (recall)")
        elif scheduler_metric == 'f1_score' or scheduler_metric == 'f1':
            mode = 'max'
            print("学习率调度器将监控验证F1分数 (f1_score)")
        elif scheduler_metric == 'precision':
            mode = 'max'
            print("学习率调度器将监控验证精确率 (precision)")
        elif scheduler_metric == 'iou':
            mode = 'max'
            print("学习率调度器将监控验证IoU (iou)")
        elif scheduler_metric == 'loss':
            mode = 'min'
            print("学习率调度器将监控验证损失 (loss)")
        else:  # 默认使用dice
            mode = 'max'
            print("学习率调度器将监控验证Dice系数 (dice)")
            
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=mode,                  # 根据配置动态设置
            factor=args.gamma,           # 来自 scheduler_factor
            patience=patience,
            min_lr=min_lr,
            cooldown=cooldown,
            verbose=False
        )
    else:
        raise ValueError(f"不支持的调度器类型: {args.scheduler}")

    return optimizer, scheduler


# ================== EMA 工具（健壮版） ==================
def init_ema(model):
    # 不依赖 requires_grad；把所有参数都克隆进去，避免解冻后缺键
    return {n: p.detach().clone() for n, p in model.named_parameters()}

@torch.no_grad()
def update_ema(model, ema_dict, decay):
    for n, p in model.named_parameters():
        if n not in ema_dict:
            # 新增/解冻的参数，先补齐
            ema_dict[n] = p.detach().clone()
        # 只对需要训练的参数做 EMA（如需对全部参数做EMA，请去掉下面判断）
        if p.requires_grad:
            ema_dict[n].mul_(decay).add_(p.detach(), alpha=1.0 - decay)

@torch.no_grad()
def apply_ema_weights(model, ema_dict):
    backup = {}
    for n, p in model.named_parameters():
        if n in ema_dict:
            backup[n] = p.detach().clone()
            p.copy_(ema_dict[n])
    return backup

@torch.no_grad()
def restore_weights(model, backup):
    for n, p in model.named_parameters():
        if n in backup:
            p.copy_(backup[n])


# ================== TTA 推理 ==================
@torch.no_grad()
def predict_tta(model, x, use_amp=True):
    # x: [B,C,H,W]
    logits_list = []
    from torch.amp import autocast
    
    # 确保在FP32下进行TTA计算，防止数值溢出
    with autocast('cuda', enabled=False):  # 禁用AMP，全程使用FP32
        # 原图
        logits = model(x)
        logits_list.append(logits)  # 保存logits而不是概率
        
        # 水平翻转
        logits_flip_h = model(torch.flip(x, dims=[-1]))
        logits_list.append(logits_flip_h.flip([-1]))  # 保存logits而不是概率
        
        # 垂直翻转
        logits_flip_v = model(torch.flip(x, dims=[-2]))
        logits_list.append(logits_flip_v.flip([-2]))  # 保存logits而不是概率
        
        # 轴交换（转置）
        logits_transpose = model(x.transpose(-1, -2))
        logits_list.append(logits_transpose.transpose(-1, -2))  # 保存logits而不是概率
    
    # 在FP32下计算logits的平均值
    avg_logits = torch.stack(logits_list, 0).mean(0)
    return avg_logits


def train_one_epoch(model, data_loader, optimizer, criterion, device, use_amp=False, gradient_accumulation_steps=1, args=None, ema_dict=None, ema_decay=0.999):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0

    from torch.amp import GradScaler, autocast
    scaler = GradScaler('cuda', enabled=use_amp)

    torch.cuda.empty_cache()
    pbar = tqdm(data_loader, desc="训练中")

    for batch_idx, batch in enumerate(pbar):
        try:
            if batch_idx > 0:
                # 安全删除变量，避免未定义错误
                if 'images' in locals():
                    del images
                if 'masks' in locals():
                    del masks
                if 'outputs' in locals():
                    del outputs
                if 'loss' in locals():
                    del loss
                torch.cuda.empty_cache()

            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # 应用高级数据增强（如果数据集支持且配置启用）
            aug_info = {'type': 'none', 'lambda': 1.0}
            if hasattr(data_loader.dataset, 'apply_batch_augmentation') and getattr(args, 'use_advanced_aug', False):
                images, masks, aug_info = data_loader.dataset.apply_batch_augmentation(images, masks)

            with autocast('cuda', enabled=use_amp):
                # R-Drop正则化：进行两次前向传播
                if args and getattr(args, 'rdrop_alpha', 0.0) > 0:
                    # 第一次前向传播
                    outputs1 = model(images)
                    # 第二次前向传播（使用dropout）
                    outputs2 = model(images)
                    
                    # 设置outputs变量用于内存清理（使用第一次前向传播的结果）
                    outputs = outputs1
                    
                    # 计算任务损失（使用第一次前向传播的结果）
                    if aug_info['type'] != 'none' and aug_info['type'] in ['mixup', 'cutmix']:
                        task_loss = criterion(outputs1, masks) / gradient_accumulation_steps
                    else:
                        task_loss = criterion(outputs1, masks) / gradient_accumulation_steps
                    
                    # 计算KL散度损失（R-Drop正则化）
                    # 使用log_softmax确保数值稳定性
                    log_probs1 = torch.nn.functional.log_softmax(outputs1, dim=1)
                    probs2 = torch.nn.functional.softmax(outputs2, dim=1)
                    kl_loss = torch.nn.functional.kl_div(log_probs1, probs2, reduction='batchmean')
                    
                    # 总损失 = 任务损失 + rdrop_alpha * KL散度损失
                    loss = task_loss + args.rdrop_alpha * kl_loss
                else:
                    # 常规训练流程
                    outputs = model(images)
                    
                    # 如果使用了MixUp/CutMix，调整损失计算
                    if aug_info['type'] != 'none' and aug_info['type'] in ['mixup', 'cutmix']:
                        # 对于混合增强，使用原始损失函数，因为标签已经是混合的
                        loss = criterion(outputs, masks) / gradient_accumulation_steps
                    else:
                        loss = criterion(outputs, masks) / gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if args and args.gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # EMA 更新
                if ema_dict is not None and getattr(args, 'ema', False):
                    update_ema(model, ema_dict, ema_decay)

                if (batch_idx + 1) % (gradient_accumulation_steps * 5) == 0:
                    torch.cuda.empty_cache()

            total_loss += loss.item() * gradient_accumulation_steps
            pbar.set_postfix({'loss': total_loss / (batch_idx + 1), 'aug': aug_info['type']})

        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"警告: 批次 {batch_idx} 内存不足，跳过此批次")
                torch.cuda.empty_cache()
                continue
            else:
                raise e

    torch.cuda.empty_cache()
    return total_loss / len(data_loader)


def validate(model, data_loader, criterion, device, args=None, val_results_cache=None):
    """验证模型（支持TTA与IoU）"""
    model.eval()
    total_loss = 0.0
    n_batches = 0  # 记录实际处理的batch数量
    dice_scores = []
    iou_scores = []
    
    # 完全关闭验证缓存，每轮都进行完整验证
    use_cache = False
    
    if use_cache:
        print(f"使用验证结果缓存 (上次更新于epoch {val_results_cache['last_epoch']})")
        # 直接使用缓存的预测和目标
        all_probs = val_results_cache['probs']
        all_masks = val_results_cache['masks']
        
        # 计算指标
        for i in range(len(all_probs)):
            probs = all_probs[i].to(device)  # 将概率张量移动到设备
            masks = all_masks[i].to(device)  # 将掩码张量移动到设备
            
            # 计算损失
            outputs = torch.log(probs / (1 - probs + 1e-7))  # 从概率转换回logits
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            n_batches += 1
            
            # 计算Dice和IoU
            preds = (probs > 0.5).to(masks.dtype)
            masks_f = masks.float()
            intersection = (probs * masks_f).sum(dtype=torch.float32)
            union = preds.sum(dtype=torch.float32) + masks_f.sum(dtype=torch.float32) - intersection
            
            dice_denominator = preds.sum(dtype=torch.float32) + masks_f.sum(dtype=torch.float32)
            dice = (2 * intersection) / (dice_denominator + 1e-8)
            iou = (intersection + 1e-8) / (union + 1e-8)
            
            dice_scores.append(dice.item())
            iou_scores.append(iou.item())
            
    else:
        print("执行完整验证过程...")
        # 收集所有预测和目标用于缓存
        all_probs = []
        all_masks = []

    # 只有在不使用缓存时才执行DataLoader循环
    if not use_cache:
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
            for batch_idx, batch in enumerate(tqdm(data_loader, desc="验证中")):
                try:
                    if batch_idx > 0:
                        # 安全删除变量，避免未定义错误
                        if 'images' in locals():
                            del images
                        if 'masks' in locals():
                            del masks
                        if 'outputs' in locals():
                            del outputs
                        if 'preds' in locals():
                            del preds
                        if 'probs' in locals():
                            del probs
                        if 'loss' in locals():
                            del loss
                        torch.cuda.empty_cache()

                    images = batch['image'].to(device)
                    masks = batch['mask'].to(device)

                    # 评估指标用 TTA 的概率
                    if getattr(args, 'use_tta', False):
                        # TTA推理使用FP32确保数值稳定性，返回logits
                        # 注意：TTA只用于指标计算，不用于损失计算
                        outputs_tta = predict_tta(model, images, use_amp=False)
                        
                        # 计算损失 - 使用非TTA的logits
                        outputs = model(images)
                        loss = criterion(outputs, masks)
                        
                        # 检查损失是否为NaN或Inf
                        if torch.isnan(loss) or torch.isinf(loss):
                            print(f"警告: 验证批次 {batch_idx} 损失为 {loss}，跳过此批次")
                            continue
                        
                        total_loss += loss.item()
                        n_batches += 1  # 增加实际处理的batch计数
                        
                        # 计算概率并确保数值稳定（使用TTA的logits）
                        probs = torch.sigmoid(outputs_tta)
                        probs = probs.clamp(1e-7, 1 - 1e-7)  # 防止极值
                        
                        # 计算指标 - 确保所有sum操作在FP32下进行
                        preds = (probs > 0.5).to(masks.dtype)
                        
                        # 使用FP32进行所有归约操作
                        masks_f = masks.float()
                        intersection = (probs * masks_f).sum(dtype=torch.float32)
                        union = preds.sum(dtype=torch.float32) + masks_f.sum(dtype=torch.float32) - intersection
                        
                        # 防止除零
                        dice_denominator = preds.sum(dtype=torch.float32) + masks_f.sum(dtype=torch.float32)
                        dice = (2 * intersection) / (dice_denominator + 1e-8)
                        iou = (intersection + 1e-8) / (union + 1e-8)

                        dice_scores.append(dice.item())
                        iou_scores.append(iou.item())

                        # 收集预测和目标用于缓存
                        all_probs.append(probs.detach().cpu())
                        all_masks.append(masks.detach().cpu())
                    else:
                        # 非TTA推理，直接使用模型输出
                        outputs = model(images)
                        
                        # 计算损失 - 直接使用logits，不在loss函数内进行clamp
                        loss = criterion(outputs, masks)
                        
                        # 检查损失是否为NaN或Inf
                        if torch.isnan(loss) or torch.isinf(loss):
                            print(f"警告: 验证批次 {batch_idx} 损失为 {loss}，跳过此批次")
                            continue
                        
                        total_loss += loss.item()
                        n_batches += 1  # 增加实际处理的batch计数
                        
                        # 计算概率并确保数值稳定
                        probs = torch.sigmoid(outputs)
                        probs = probs.clamp(1e-7, 1 - 1e-7)  # 防止极值
                        
                        # 计算指标 - 确保所有sum操作在FP32下进行
                        preds = (probs > 0.5).to(masks.dtype)
                        
                        # 使用FP32进行所有归约操作
                        masks_f = masks.float()
                        intersection = (probs * masks_f).sum(dtype=torch.float32)
                        union = preds.sum(dtype=torch.float32) + masks_f.sum(dtype=torch.float32) - intersection
                        
                        # 防止除零
                        dice_denominator = preds.sum(dtype=torch.float32) + masks_f.sum(dtype=torch.float32)
                        dice = (2 * intersection) / (dice_denominator + 1e-8)
                        iou = (intersection + 1e-8) / (union + 1e-8)

                        dice_scores.append(dice.item())
                        iou_scores.append(iou.item())

                        # 收集预测和目标用于缓存
                        all_probs.append(probs.detach().cpu())
                        all_masks.append(masks.detach().cpu())

            
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower():
                        print(f"警告: 验证批次 {batch_idx} 内存不足，跳过此批次")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        print(f"错误: 验证批次 {batch_idx} 发生运行时错误: {e}")
                        raise e
                except Exception as e:
                    print(f"错误: 验证批次 {batch_idx} 发生未知错误: {e}")
                    continue

    torch.cuda.empty_cache()

    # 修复：将验证损失和指标都按实际处理的batch数量平均
    avg_loss = total_loss / max(1, n_batches)  # 使用实际处理的batch数量
    patch_mean_dice = sum(dice_scores) / max(1, len(dice_scores)) if dice_scores else 0.0
    patch_mean_iou = sum(iou_scores) / max(1, len(iou_scores)) if iou_scores else 0.0

    # 初始化全局指标
    global_dice = 0.0
    global_iou = 0.0

    # 聚合全量预测计算验证指标
    avg_metrics = {
        'iou': 0.0,
        'dice': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1_score': 0.0,
        'accuracy': 0.0
    }
    
    print(f"验证阶段: all_probs长度={len(all_probs)}, all_masks长度={len(all_masks)}")
    
    if len(all_probs) > 0 and len(all_masks) > 0:
        try:
            probs_tensor = torch.cat([p if isinstance(p, torch.Tensor) else torch.as_tensor(p) for p in all_probs], dim=0)
            masks_tensor = torch.cat([m if isinstance(m, torch.Tensor) else torch.as_tensor(m) for m in all_masks], dim=0)
            threshold = 0.5
            if args is not None:
                threshold = getattr(args, 'best_threshold', getattr(args, 'threshold', 0.5))
            
            print(f"计算全局指标: 使用阈值 {threshold}, 概率张量形状 {probs_tensor.shape}, 掩码张量形状 {masks_tensor.shape}")
            
            # 检查张量值范围
            print(f"概率值范围: [{probs_tensor.min().item():.6f}, {probs_tensor.max().item():.6f}]")
            print(f"掩码值范围: [{masks_tensor.min().item():.6f}, {masks_tensor.max().item():.6f}]")
            
            # 检查掩码是否为二值
            unique_masks = torch.unique(masks_tensor)
            print(f"掩码唯一值: {unique_masks.tolist()}")
            
            # 检查正负样本比例
            total_pixels = masks_tensor.numel()
            positive_pixels = masks_tensor.sum().item()
            negative_pixels = total_pixels - positive_pixels
            print(f"正样本比例: {positive_pixels/total_pixels*100:.2f}%, 负样本比例: {negative_pixels/total_pixels*100:.2f}%")
            
            # 检查概率分布
            binary_preds = (probs_tensor > threshold).float()
            pred_positive_pixels = binary_preds.sum().item()
            pred_negative_pixels = total_pixels - pred_positive_pixels
            print(f"预测正样本比例: {pred_positive_pixels/total_pixels*100:.2f}%, 预测负样本比例: {pred_negative_pixels/total_pixels*100:.2f}%")
            
            # 计算混淆矩阵
            tp = (binary_preds * masks_tensor).sum().item()
            fp = (binary_preds * (1.0 - masks_tensor)).sum().item()
            fn = ((1.0 - binary_preds) * masks_tensor).sum().item()
            tn = ((1.0 - binary_preds) * (1.0 - masks_tensor)).sum().item()
            
            print(f"混淆矩阵: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
            
            # 手动计算指标
            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            f1_score = (2.0 * precision * recall) / (precision + recall + 1e-6)
            iou = tp / (tp + fp + fn + 1e-6)
            dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-6)
            accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-6)
            
            print(f"手动计算指标: IoU={iou:.6f}, Dice={dice:.6f}, Precision={precision:.6f}, Recall={recall:.6f}, F1={f1_score:.6f}, Accuracy={accuracy:.6f}")
            
            # 使用compute_global_binary_metrics函数计算指标
            avg_metrics, counts = compute_global_binary_metrics(probs_tensor, masks_tensor, threshold)
            
            # 保存全局Dice和IoU值
            global_dice = dice
            global_iou = iou
            
            # 将patch均值Dice和IoU添加到avg_metrics中，以便在训练循环中访问
            avg_metrics['patch_mean_dice'] = patch_mean_dice
            avg_metrics['patch_mean_iou'] = patch_mean_iou
            
            print(f"compute_global_binary_metrics计算结果: {avg_metrics}")
            print(f"compute_global_binary_metrics混淆矩阵计数: TP={counts['tp']}, FP={counts['fp']}, FN={counts['fn']}, TN={counts['tn']}")
            
            # 打印全局Dice和patch均值Dice，明确区分
            print(f"全局Dice: {global_dice:.6f} (基于所有像素的混淆矩阵计算 - 最终评估指标)")
            print(f"Patch均值Dice: {patch_mean_dice:.6f} (基于各patch Dice的平均值 - 仅供参考)")
            
        except Exception as exc:
            print(f"警告: 聚合验证指标时出错: {exc}")
            import traceback
            traceback.print_exc()
    else:
        print("警告: 没有收集到预测或掩码数据，跳过全局指标计算")

    # 不再更新缓存，每轮都进行完整验证
    # if not use_cache and val_results_cache is not None and len(all_probs) > 0:
    #     val_results_cache['probs'] = all_probs
    #     val_results_cache['masks'] = all_masks
    #     print("验证结果已缓存")

    return avg_loss, global_dice, global_iou, avg_metrics


def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, args):
    """训练模型"""
    # 确保使用全局os模块
    global os
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    best_dice = 0.0
    epochs_no_improve = 0
    start_epoch = 0
    
    # 验证结果缓存机制（已禁用）
    val_results_cache = {
        'probs': None,
        'masks': None,
        'last_epoch': -1,
        'cache_interval': 1  # 虽然设置了间隔，但实际已禁用缓存
    }

    # 恢复训练（如果有检查点）
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location='cpu', weights_only=False)
        # ---- 关键：键名映射 + strict=False ----
        sd = ckpt.get('model_state_dict', ckpt)
        sd = _normalize_low_proj_keys(sd)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[Resume] loaded with strict=False | missing: {len(missing)} | unexpected: {len(unexpected)}")
        if len(missing) > 0:
            print("  missing (sample):", [m for m in list(missing)[:5]])
        start_epoch = int(ckpt.get('epoch', -1)) + 1
        
        # 尝试从检查点获取best_dice
        best_dice = float(ckpt.get('best_dice', 0.0))
        print(f"[Resume] Initial best_dice from checkpoint: {best_dice:.4f}")
        
        # 如果best_dice为0或很小，尝试从其他地方获取
        if best_dice <= 0.0:
            # 尝试从检查点的val_metrics中获取dice值
            val_metrics = ckpt.get('val_metrics', {})
            if isinstance(val_metrics, dict):
                metrics_dice = float(val_metrics.get('dice', 0.0))
                if metrics_dice > best_dice:
                    best_dice = metrics_dice
                    print(f"[Resume] Updated best_dice from val_metrics: {best_dice:.4f}")
            
            # 尝试从训练结果JSON文件中获取最佳Dice系数
            if best_dice <= 0.0:
                checkpoint_dir = os.path.dirname(args.resume_from)
                json_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('_training_results.json')]
                if json_files:
                    json_path = os.path.join(checkpoint_dir, json_files[0])
                    try:
                        with open(json_path, 'r') as f:
                            results = json.load(f)
                            json_best_dice = float(results.get('training_results', {}).get('best_validation_dice', 0.0))
                            if json_best_dice > best_dice:
                                best_dice = json_best_dice
                                print(f"[Resume] Updated best_dice from JSON: {best_dice:.4f}")
                    except Exception as e:
                        print(f"[Resume] Failed to load best_dice from JSON: {e}")
            
            # 尝试从检查点目录中的其他检查点文件获取
            if best_dice <= 0.0:
                checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth') and 'best' in f]
                for ckpt_file in checkpoint_files:
                    if ckpt_file != os.path.basename(args.resume_from):
                        try:
                            other_ckpt_path = os.path.join(checkpoint_dir, ckpt_file)
                            other_ckpt = torch.load(other_ckpt_path, map_location='cpu', weights_only=False)
                            other_best_dice = float(other_ckpt.get('best_dice', 0.0))
                            if other_best_dice > best_dice:
                                best_dice = other_best_dice
                                print(f"[Resume] Updated best_dice from {ckpt_file}: {best_dice:.4f}")
                            
                            # 也尝试从其他检查点的val_metrics中获取
                            if other_best_dice <= 0.0:
                                other_val_metrics = other_ckpt.get('val_metrics', {})
                                if isinstance(other_val_metrics, dict):
                                    other_metrics_dice = float(other_val_metrics.get('dice', 0.0))
                                    if other_metrics_dice > best_dice:
                                        best_dice = other_metrics_dice
                                        print(f"[Resume] Updated best_dice from {ckpt_file} val_metrics: {best_dice:.4f}")
                        except Exception as e:
                            print(f"[Resume] Failed to load best_dice from {ckpt_file}: {e}")
            
            # 如果所有尝试都失败了，使用一个合理的默认值
            if best_dice <= 0.0:
                best_dice = 0.5  # 设置一个合理的默认值
                print(f"[Resume] Using default best_dice: {best_dice:.4f}")
        
        print(f"[Resume] start_epoch={start_epoch}, prev_best_dice={best_dice:.4f}")
        # 不加载旧优化器/调度器（结构/超参可能已变）

    # 修正骨干网络冻结逻辑
    # 只在start_epoch < freeze_backbone_epochs时才冻结骨干网络
    if getattr(args, 'freeze_backbone', False) and hasattr(model, 'freeze_backbone'):
        if start_epoch < getattr(args, 'freeze_backbone_epochs', 0):
            print(f"冻结骨干网络 {getattr(args, 'freeze_backbone_epochs', 0)} 轮...")
            model.freeze_backbone()
        else:
            print(f"start_epoch ({start_epoch}) >= freeze_backbone_epochs ({getattr(args, 'freeze_backbone_epochs', 0)})，不冻结骨干网络")
            # 如果骨干网络当前被冻结，解冻它
            if hasattr(model, 'unfreeze_backbone'):
                print("解冻骨干网络...")
                model.unfreeze_backbone()
                # 解冻后重建优化器/调度器
                optimizer, scheduler = setup_optimizer_and_scheduler(model, args)
                # 解冻后补齐 EMA 键，避免 KeyError
                if getattr(args, 'ema', False):
                    ema_dict = init_ema(model)

    # 初始化 EMA
    ema_dict = None
    if getattr(args, 'ema', False):
        ema_dict = init_ema(model)

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        # 解冻逻辑：在指定 epoch 解冻并重建优化器/调度器
        # 如果当前epoch等于freeze_backbone_epochs，并且骨干网络需要解冻
        if epoch == args.freeze_backbone_epochs and hasattr(model, 'unfreeze_backbone'):
            print(f"在第 {epoch+1} 轮解冻骨干网络进行微调...")
            model.unfreeze_backbone()
            optimizer, scheduler = setup_optimizer_and_scheduler(model, args)
            # 解冻后补齐 EMA 键，避免 KeyError
            if ema_dict is not None and getattr(args, 'ema', False):
                for n, p in model.named_parameters():
                    if n not in ema_dict:
                        ema_dict[n] = p.detach().clone()

        # 训练
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, args.device,
            use_amp=args.use_amp, gradient_accumulation_steps=args.gradient_accumulation_steps, args=args,
            ema_dict=ema_dict, ema_decay=args.ema_decay
        )

        # 验证（若开启 EMA，临时切到 EMA 权重）
        backup_weights = None
        if ema_dict is not None and getattr(args, 'ema', False):
            backup_weights = apply_ema_weights(model, ema_dict)

        # 完全关闭验证缓存，每轮都进行完整验证
        use_cache = False
        
        val_loss, val_dice, val_iou, val_metrics = validate(model, val_loader, criterion, args.device, args=args, val_results_cache=val_results_cache)

        if backup_weights is not None:
            restore_weights(model, backup_weights)

        # 调度器
        if args.scheduler == 'plateau':
            # 根据配置的监控指标调整学习率
            scheduler_metric = getattr(args, 'scheduler_metric', 'dice').lower()
            if scheduler_metric == 'recall':
                scheduler.step(val_metrics.get('recall', 0.0))
            elif scheduler_metric == 'f1_score' or scheduler_metric == 'f1':
                scheduler.step(val_metrics.get('f1_score', 0.0))
            elif scheduler_metric == 'precision':
                scheduler.step(val_metrics.get('precision', 0.0))
            elif scheduler_metric == 'iou':
                scheduler.step(val_iou)
            elif scheduler_metric == 'loss':
                scheduler.step(val_loss)
            else:  # 默认使用dice
                scheduler.step(val_dice)
        else:
            scheduler.step()

        print(f"训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}, 验证Dice(全局): {val_dice:.4f}, 验证IOU(全局): {val_iou:.4f}")
        print(f"验证Dice(Patch均值): {val_metrics.get('patch_mean_dice', 0.0):.4f}, 验证IOU(Patch均值): {val_metrics.get('patch_mean_iou', 0.0):.4f}")

        # 打印当前学习率
        cur_lr = optimizer.param_groups[0]['lr']
        print(f"[LR] current learning rate: {cur_lr:.6g}")

        # 保存最佳（保存 EMA 权重）
        should_save_model = False
        should_search_threshold = False
        
        # 根据配置的监控指标决定是否保存模型
        scheduler_metric = getattr(args, 'scheduler_metric', 'dice').lower()
        if scheduler_metric == 'recall':
            current_metric = val_metrics.get('recall', 0.0)
            best_metric = getattr(args, 'best_recall', 0.0)
            metric_name = "召回率"
        elif scheduler_metric == 'f1_score' or scheduler_metric == 'f1':
            current_metric = val_metrics.get('f1_score', 0.0)
            best_metric = getattr(args, 'best_f1', 0.0)
            metric_name = "F1分数"
        elif scheduler_metric == 'precision':
            current_metric = val_metrics.get('precision', 0.0)
            best_metric = getattr(args, 'best_precision', 0.0)
            metric_name = "精确率"
        elif scheduler_metric == 'iou':
            current_metric = val_iou
            best_metric = getattr(args, 'best_iou', 0.0)
            metric_name = "IoU"
        elif scheduler_metric == 'loss':
            current_metric = val_loss
            best_metric = getattr(args, 'best_loss', float('inf'))
            metric_name = "损失"
            # 对于损失，越小越好
            if current_metric < best_metric - args.save_improvement_threshold:
                best_metric = current_metric
                epochs_no_improve = 0
                should_save_model = True
                setattr(args, 'best_loss', best_metric)
                print(f"新的最佳{metric_name}: {best_metric:.4f}")
            else:
                epochs_no_improve += 1
        else:  # 默认使用dice
            current_metric = val_dice
            best_metric = best_dice
            metric_name = "Dice系数"
            
        # 对于需要最大化的指标（recall, f1, precision, iou, dice）
        if scheduler_metric != 'loss':
            if current_metric > best_metric + args.save_improvement_threshold:
                best_metric = current_metric
                epochs_no_improve = 0
                should_save_model = True
                
                # 更新对应的最佳指标
                if scheduler_metric == 'recall':
                    setattr(args, 'best_recall', best_metric)
                elif scheduler_metric == 'f1_score' or scheduler_metric == 'f1':
                    setattr(args, 'best_f1', best_metric)
                elif scheduler_metric == 'precision':
                    setattr(args, 'best_precision', best_metric)
                elif scheduler_metric == 'iou':
                    setattr(args, 'best_iou', best_metric)
                else:  # dice
                    best_dice = best_metric
                    
                print(f"新的最佳{metric_name}: {best_metric:.4f}")
                
                # 检查是否应该进行阈值搜索
                if hasattr(args, 'enable_threshold_search') and args.enable_threshold_search:
                    if hasattr(args, 'threshold_search_interval') and (epoch + 1) % args.threshold_search_interval == 0:
                        should_search_threshold = True
                    elif not hasattr(args, 'threshold_search_interval'):
                        should_search_threshold = True
            else:
                epochs_no_improve += 1

        # 执行阈值搜索（如果需要）
        best_threshold = 0.5  # 默认阈值
        best_f1_score = 0.0
        
        if should_search_threshold:
            print(f"[阈值搜索] 开始进行阈值搜索...")
            
            # 阈值搜索（用"当前模型权重"进行）
            from utils.metrics import calculate_threshold_metrics
            model.eval()
            all_predictions = []
            all_targets = []

            with torch.no_grad():
                for batch in val_loader:
                    images = batch['image'].to(args.device)
                    masks = batch['mask'].to(args.device)

                    # 禁用AMP，全程使用FP32计算阈值搜索
                    with torch.amp.autocast('cuda', enabled=False):
                        outputs = model(images)
                        probs = torch.sigmoid(outputs).float()  # 确保FP32
                        probs = probs.clamp_(1e-7, 1 - 1e-7)  # 防止极值

                    all_predictions.append(probs.cpu())
                    all_targets.append(masks.cpu())

            all_predictions = torch.cat(all_predictions, dim=0)
            all_targets = torch.cat(all_targets, dim=0)

            # 使用最小化阈值搜索（如果配置了）
            if hasattr(args, 'minimal_threshold_search') and args.minimal_threshold_search:
                # 只测试关键阈值点，减少计算量
                threshold_results = calculate_threshold_metrics(
                    all_predictions, all_targets, 
                    thresholds=[0.4, 0.5, 0.6, 0.7, 0.8]  # 调整为更高的阈值范围，控制过度预测
                )
            else:
                # 使用配置文件中的阈值范围
                if hasattr(args, 'threshold_range') and isinstance(args.threshold_range, list) and len(args.threshold_range) == 2:
                    min_threshold, max_threshold = args.threshold_range
                    thresholds = np.linspace(min_threshold, max_threshold, 9).tolist()
                else:
                    thresholds = np.linspace(0.4, 0.8, 9).tolist()  # 默认使用更高的阈值范围
                
                threshold_results = calculate_threshold_metrics(all_predictions, all_targets, thresholds=thresholds)
                
            best_threshold = float(threshold_results['best_threshold'])
            best_f1_score = float(threshold_results['best_f1_score'])

            print(f"[阈值搜索] 最佳阈值: {best_threshold:.3f}, 最佳F1分数: {best_f1_score:.4f}")

            del all_predictions, all_targets
            torch.cuda.empty_cache()
        elif should_save_model and hasattr(args, 'best_threshold') and hasattr(args, 'best_f1_score'):
            # 如果不进行阈值搜索但需要保存模型，使用之前的最佳阈值和F1分数
            best_threshold = getattr(args, 'best_threshold', 0.5)
            best_f1_score = getattr(args, 'best_f1_score', 0.0)

        # 保存模型（如果需要）
        if should_save_model:
            # 应用 EMA 后保存（若开启 EMA）
            to_save_state = model.state_dict()
            if ema_dict is not None and getattr(args, 'ema', False):
                backup2 = apply_ema_weights(model, ema_dict)
                to_save_state = model.state_dict()
                restore_weights(model, backup2)

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': to_save_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
                'best_dice': float(best_dice),
                'best_iou': float(val_iou),
                'best_threshold': best_threshold,
                'best_f1_score': best_f1_score,
                # 添加其他最佳指标
                'best_recall': float(getattr(args, 'best_recall', val_metrics.get('recall', 0.0))),
                'best_precision': float(getattr(args, 'best_precision', val_metrics.get('precision', 0.0))),
                'best_loss': float(getattr(args, 'best_loss', val_loss)),
                'args': args
            }

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_dir = os.path.join(args.checkpoint_dir, args.model)
            os.makedirs(model_dir, exist_ok=True)
            timestamp_dir = os.path.join(model_dir, timestamp)
            os.makedirs(timestamp_dir, exist_ok=True)

            timestamp_save_path = os.path.join(timestamp_dir, f"{args.model}_best.pth")
            torch.save(checkpoint, timestamp_save_path)

            timestamped_save_path = os.path.join(model_dir, f"{args.model}_best_{timestamp}.pth")
            torch.save(checkpoint, timestamped_save_path)

            latest_save_path = os.path.join(model_dir, f"{args.model}_best.pth")
            torch.save(checkpoint, latest_save_path)

            print(f"保存最佳模型到 {timestamp_save_path}")
            print(f"保存带时间戳的版本到 {timestamped_save_path}")
            print(f"保存最新版本到 {latest_save_path}")

            # 训练结果写JSON（异步方式，减少阻塞）
            def save_training_results_async(val_iou, val_dice, val_metrics, train_loss, val_loss, best_dice, timestamp, timestamp_dir, model_dir):
                # 将args转换为字典，过滤掉不可序列化的对象
                args_dict = {}
                for arg, value in vars(args).items():
                    # 跳过不可序列化的对象
                    if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                        args_dict[arg] = value
                
                training_results = {
                    'training_results': {
                        'best_validation_dice': float(best_dice),
                        'final_train_loss': float(train_loss),
                        'final_val_loss': float(val_loss),
                        'final_val_metrics': {
                            'iou': float(val_iou),
                            'dice': float(val_dice),
                            'precision': float(val_metrics.get('precision', 0.0)),
                            'recall': float(val_metrics.get('recall', 0.0)),
                            'f1_score': float(val_metrics.get('f1_score', 0.0)),
                            'accuracy': float(val_metrics.get('accuracy', 0.0))
                        }
                    },
                    'training_parameters': args_dict,
                    'timestamp': timestamp
                }
                timestamp_json_path = os.path.join(timestamp_dir, f"{args.model}_training_results.json")
                with open(timestamp_json_path, 'w') as f:
                    json.dump(training_results, f, indent=4)
                timestamped_json_path = os.path.join(model_dir, f"{args.model}_training_results_{timestamp}.json")
                with open(timestamped_json_path, 'w') as f:
                    json.dump(training_results, f, indent=4)
            
            # 使用线程异步保存JSON结果
            import threading
            json_thread = threading.Thread(
                target=save_training_results_async, 
                args=(val_iou, val_dice, val_metrics, train_loss, val_loss, best_dice, timestamp, timestamp_dir, model_dir)
            )
            json_thread.daemon = True
            json_thread.start()
            
            # 更新args中的最佳阈值和F1分数，以便后续使用
            args.best_threshold = best_threshold
            args.best_f1_score = best_f1_score

        # 定期保存（异步方式，减少阻塞）
        if (epoch + 1) % args.save_interval_epochs == 0:
            def save_checkpoint_async():
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
                    'best_dice': float(best_dice),
                    'args': args
                }
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_dir = os.path.join(args.checkpoint_dir, args.model)
                os.makedirs(model_dir, exist_ok=True)
                timestamp_dir = os.path.join(model_dir, timestamp)
                os.makedirs(timestamp_dir, exist_ok=True)
                timestamp_save_path = os.path.join(timestamp_dir, f"{args.model}_epoch{epoch+1}.pth")
                torch.save(checkpoint, timestamp_save_path)
                timestamped_save_path = os.path.join(model_dir, f"{args.model}_epoch{epoch+1}_{timestamp}.pth")
                torch.save(checkpoint, timestamped_save_path)
                print(f"保存检查点到 {timestamp_save_path}")
                print(f"保存带时间戳的版本到 {timestamped_save_path}")
            
            # 使用线程异步保存检查点
            import threading
            checkpoint_thread = threading.Thread(target=save_checkpoint_async)
            checkpoint_thread.daemon = True
            checkpoint_thread.start()
            
            print(f"开始异步保存检查点 (epoch {epoch+1})...")

        # 早停
        if args.early_stopping and epochs_no_improve >= args.early_stopping_patience:
            print(f"早停：{args.early_stopping_patience} 轮验证Dice系数(全局)没有改善")
            break

    print(f"训练完成，最佳验证Dice系数(全局): {best_dice:.4f}")
    return model


def main():
    """主函数"""
    # 显存安全设置
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    if torch.cuda.is_available():
        try:
            torch.cuda.set_per_process_memory_fraction(0.8, device=0)
        except Exception:
            pass
        torch.cuda.empty_cache()
        # 移除不支持的expandable_segments选项
        # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    args = parse_args()

    if args.config:
        print("Using config abs path:", os.path.abspath(args.config))

    args = update_args_with_config(args)

    print("训练参数:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    
    # 特别打印数据加载相关参数
    print(f"数据加载参数:")
    print(f"  num_workers: {args.num_workers}")
    print(f"  persistent_workers: {args.persistent_workers}")
    print(f"  force_multiprocess: {args.force_multiprocess}")
    print(f"  pin_memory: {args.pin_memory}")
    print(f"  batch_size: {args.batch_size}")

    device = torch.device(args.device)
    print(f"使用设备: {device}")

    model = setup_model(args)

    # 损失函数（含 pos_weight）
    if args.loss_type in ('bce', 'bce_logits'):
        pw = torch.tensor([getattr(args, 'pos_weight', 1.0)], device=args.device, dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    elif args.loss_type == 'focal':
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    elif args.loss_type == 'lovasz':
        criterion = LovaszLoss()
    elif args.loss_type == 'focal_lovasz':
        criterion = FocalLovaszLoss(
            focal_weight=args.focal_weight,
            lovasz_weight=args.lovasz_weight,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma
        )
    elif args.loss_type == 'bce_focal_lovasz':
        criterion = BCEFocalLovaszLoss(
            bce_weight=getattr(args, 'bce_weight', 0.2),
            focal_weight=getattr(args, 'focal_weight', 0.4),
            lovasz_weight=getattr(args, 'lovasz_weight', 0.4),
            focal_alpha=getattr(args, 'focal_alpha', 1.0),
            focal_gamma=getattr(args, 'focal_gamma', 2.0),
            pos_weight=getattr(args, 'pos_weight', None)
        )
    elif args.loss_type == 'bce_dice':
        criterion = BCEDiceLoss(
            bce_weight=getattr(args, 'bce_weight', 0.5),
            dice_weight=getattr(args, 'dice_weight', 0.5),
            smooth=getattr(args, 'dice_smooth', 1e-6),
            pos_weight=getattr(args, 'pos_weight', None)
        )
    elif args.loss_type == 'combined':
        criterion = BCEFocalLovaszLoss(
            bce_weight=getattr(args, 'bce_weight', 0.33),
            focal_weight=getattr(args, 'focal_weight', 0.33),
            lovasz_weight=getattr(args, 'lovasz_weight', 0.34),
            focal_alpha=getattr(args, 'focal_alpha', 1.0),
            focal_gamma=getattr(args, 'focal_gamma', 2.0),
            pos_weight=getattr(args, 'pos_weight', None)
        )
    else:
        raise ValueError(f"不支持的损失函数类型: {args.loss_type}")

    optimizer, scheduler = setup_optimizer_and_scheduler(model, args)

    from utils.data_utils import Sentinel2WaterDataset, Sentinel2WaterDatasetWithAdvancedAug

    # 从配置中获取数据增强配置
    augmentation_config = {}
    if hasattr(args, 'config') and args.config:
        # 从配置文件中读取数据增强参数
        import yaml
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            # 检查augment部分
            if 'augment' in config:
                augmentation_config = config['augment']
            # 或者检查data.augmentation部分（兼容性）
            elif 'data' in config and 'augmentation' in config['data']:
                augmentation_config = config['data']['augmentation']
    
    splits_dir = args.splits_dir or 'splits'

    # 根据是否使用高级数据增强选择数据集类
    if getattr(args, 'use_advanced_aug', False):
        print("使用高级数据增强 (MixUp/CutMix)")
        train_dataset = Sentinel2WaterDatasetWithAdvancedAug(
            data_dir=args.data_dir,
            split='train',
            bands=args.bands,
            augment=bool(args.augment),
            normalize_method=args.normalize_method,
            splits_dir=splits_dir,
            images_dir=getattr(args, 'images_dir', None),
            masks_dir=getattr(args, 'masks_dir', None),
            mixup_alpha=getattr(args, 'mixup_alpha', 1.0),
            cutmix_alpha=getattr(args, 'cutmix_alpha', 1.0),
            mixup_prob=getattr(args, 'mixup_prob', 0.5),
            cutmix_prob=getattr(args, 'cutmix_prob', 0.5),
            augmentation_config=augmentation_config
        )
    else:
        train_dataset = Sentinel2WaterDataset(
            data_dir=args.data_dir,
            split='train',
            bands=args.bands,
            augment=bool(args.augment),
            normalize_method=args.normalize_method,
            splits_dir=splits_dir,
            images_dir=getattr(args, 'images_dir', None),
            masks_dir=getattr(args, 'masks_dir', None),
            augmentation_config=augmentation_config
        )

    val_dataset = Sentinel2WaterDataset(
        data_dir=args.data_dir,
        split='val',
        bands=args.bands,
        augment=False,
        normalize_method=args.normalize_method,
        splits_dir=splits_dir,
        images_dir=getattr(args, 'images_dir', None),
        masks_dir=getattr(args, 'masks_dir', None)
    )

    train_loader = create_data_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers if os.name != 'nt' or args.force_multiprocess else 0,  # 根据参数决定是否使用多进程
        pin_memory=args.pin_memory and not args.no_pin_memory,
        persistent_workers=args.persistent_workers if os.name != 'nt' or args.force_multiprocess else False,  # 根据参数决定是否使用持久化工作进程
        prefetch_factor=getattr(args, 'prefetch_factor', None)
    )

    val_loader = create_data_loader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,  # 验证集使用0以避免内存问题
        pin_memory=args.pin_memory and not args.no_pin_memory,
        persistent_workers=False  # 验证集禁用以避免内存问题
    )
    
    # 打印数据加载器参数
    print(f"训练数据加载器参数:")
    print(f"  batch_size: {train_loader.batch_size}")
    print(f"  num_workers: {train_loader.num_workers}")
    print(f"  pin_memory: {train_loader.pin_memory}")
    print(f"  persistent_workers: {train_loader.persistent_workers}")
    
    print(f"验证数据加载器参数:")
    print(f"  batch_size: {val_loader.batch_size}")
    print(f"  num_workers: {val_loader.num_workers}")
    print(f"  pin_memory: {val_loader.pin_memory}")
    print(f"  persistent_workers: {val_loader.persistent_workers}")

    model = train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, args)


if __name__ == "__main__":
    main()
