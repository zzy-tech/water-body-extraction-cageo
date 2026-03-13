# predict.py
"""
Sentinel-2水体分割预测脚本。
支持使用训练好的模型或集成模型对新的Sentinel-2影像进行水体提取预测。
"""
import os
import argparse
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import rasterio

# 导入自定义模块
from models.aer_unet import get_aer_unet_model
from utils.data_utils import normalize_image, load_sentinel2_image, save_predictions
from utils.ensemble_utils import ModelEnsemble
from config import get_config

# 导入CRF工具函数
from utils.crf_utils import apply_crf_postprocessing

# 导入后处理工具函数
from utils.postprocessing_utils import apply_postprocessing_pipeline

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("prediction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Sentinel-2影像水体分割预测')
    
    # 配置文件参数
    parser.add_argument('--config', type=str, default=None, help='YAML配置文件路径')
    
    # 输入/输出参数
    parser.add_argument('--input_dir', type=str,
                        help='输入影像目录路径')
    parser.add_argument('--output_dir', type=str, default='poyanghu',
                        help='预测结果保存目录')
    parser.add_argument('--file_pattern', type=str, default='*.tif',
                        help='输入影像文件匹配模式')
    parser.add_argument('--bands', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6],
                        help='要使用的波段索引列表')
    
    # 模型参数
    parser.add_argument('--model', type=str, choices=['aer_unet', 'lightweight_unet', 'unet', 'deeplabv3_plus', 'lightweight_deeplabv3_plus', 'ultra_lightweight_deeplabv3_plus'],
                         help='要使用的模型类型')
    parser.add_argument('--checkpoint_path', type=str,
                        help='模型检查点文件路径')
    parser.add_argument('--n_classes', type=int, default=1,
                        help='输出类别数')
    parser.add_argument('--base_features', type=int, default=32,
                        help='AER U-Net的基础特征数')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                        help='Dropout率')
    parser.add_argument('--output_stride', type=int, default=16, choices=[8, 16, 32],
                        help='DeepLabV3+的输出步长')
    parser.add_argument('--pretrained_backbone', action='store_true',
                        help='是否使用预训练的骨干网络')
    
    # UltraLightDeepLabV3Plus特定参数
    parser.add_argument('--aspp_rates', type=int, nargs='+', default=None,
                        help='ASPP模块的空洞卷积速率列表')
    parser.add_argument('--class_prior', type=float, nargs='+', default=None,
                        help='类别先验概率列表')
    parser.add_argument('--use_se', type=lambda x: str(x).lower() == 'true', default=None,
                        help='是否使用SE注意力机制')
    parser.add_argument('--aspp_out', type=int, default=None,
                        help='ASPP模块输出通道数')
    parser.add_argument('--dec_ch', type=int, default=None,
                        help='解码器通道数')
    parser.add_argument('--low_ch_out', type=int, default=None,
                        help='低层特征输出通道数')
    parser.add_argument('--use_cbam', type=lambda x: str(x).lower() == 'true', default=None,
                        help='是否使用CBAM注意力机制')
    parser.add_argument('--cbam_reduction_ratio', type=int, default=None,
                        help='CBAM注意力机制的降维比率')
    
    # 集成参数
    parser.add_argument('--ensemble', action='store_true',
                        help='启用模型集成')
    parser.add_argument('--models', type=str, nargs='+', choices=['aer_unet', 'lightweight_unet', 'unet', 'deeplabv3_plus', 'lightweight_deeplabv3_plus', 'ultra_lightweight_deeplabv3_plus'],
                         help='要集成的模型类型列表')
    parser.add_argument('--checkpoint_paths', type=str, nargs='+',
                        help='要集成的模型检查点文件路径列表')
    parser.add_argument('--ensemble_strategy', type=str, choices=['mean', 'weighted_mean', 'vote', 'logits_mean'], default='mean',
                        help='集成策略')
    parser.add_argument('--weights', type=float, nargs='+',
                        help='用于加权平均策略的权重列表')
    
    # 预测参数
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='二值化阈值')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批量大小')
    parser.add_argument('--tile_size', type=int, default=256,
                        help='处理大影像时的瓦片大小，也用于滑窗推理的窗口大小')
    parser.add_argument('--overlap', type=int, default=32,
                        help='处理大影像时的瓦片重叠大小')
    
    # 滑窗推理和余弦加权融合参数
    parser.add_argument('--use_sliding_window', action='store_true',
                        help='使用滑窗推理和余弦加权融合')
    
    parser.add_argument('--show_results', action='store_true',
                        help='显示预测结果')
    parser.add_argument('--save_visualization', action='store_true',
                        help='保存可视化结果')
    parser.add_argument('--save_probability', action='store_true',
                        help='保存概率图')
    
    # CRF参数
    parser.add_argument('--crf_iterations', type=int, default=0,
                        help='CRF迭代次数，0表示不使用CRF')
    
    # 后处理参数
    parser.add_argument('--median_kernel_size', type=int, default=0,
                        help='中值滤波核大小，0表示不应用')
    parser.add_argument('--gaussian_sigma', type=float, default=0.0,
                        help='高斯滤波标准差，0表示不应用')
    parser.add_argument('--morph_close_kernel_size', type=int, default=0,
                        help='形态学闭运算核大小，0表示不应用')
    parser.add_argument('--morph_open_kernel_size', type=int, default=0,
                        help='形态学开运算核大小，0表示不应用')
    parser.add_argument('--min_object_size', type=int, default=0,
                        help='最小对象大小（像素数），0表示不应用')
    parser.add_argument('--hole_area_threshold', type=int, default=0,
                        help='小孔面积阈值，0表示不应用')
    parser.add_argument('--adaptive_threshold', action='store_true',
                        help='使用自适应阈值（Otsu）')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='预测设备')
    
    return parser.parse_args()

def update_args_with_config(args):
    """从YAML配置文件更新命令行参数"""
    if args.config is None:
        return args
    
    # 加载YAML配置
    config = get_config(yaml_config_path=args.config)
    
    # 更新数据相关参数
    if hasattr(config, 'data'):
        if hasattr(config.data, 'images') and args.input_dir is None:
            args.input_dir = config.data.images
        if hasattr(config.data, 'bands') and args.bands is None:
            args.bands = config.data.bands
    
    # 更新模型相关参数
    if hasattr(config, 'models') and config.models:
        # 单模型模式下，使用第一个模型
        if not args.ensemble and args.model is None:
            args.model = config.models[0].name
            if hasattr(config.models[0], 'kwargs'):
                if 'base_features' in config.models[0].kwargs and args.base_features is None:
                    args.base_features = config.models[0].kwargs['base_features']
                if 'dropout_rate' in config.models[0].kwargs and args.dropout_rate is None:
                    args.dropout_rate = config.models[0].kwargs['dropout_rate']
                if 'output_stride' in config.models[0].kwargs and args.output_stride is None:
                    args.output_stride = config.models[0].kwargs['output_stride']
                if 'pretrained_backbone' in config.models[0].kwargs and args.pretrained_backbone is None:
                    args.pretrained_backbone = config.models[0].kwargs['pretrained_backbone']
                # 添加UltraLightDeepLabV3Plus特定参数
                if 'aspp_rates' in config.models[0].kwargs:
                    args.aspp_rates = config.models[0].kwargs['aspp_rates']
                if 'class_prior' in config.models[0].kwargs:
                    args.class_prior = config.models[0].kwargs['class_prior']
                if 'use_se' in config.models[0].kwargs:
                    args.use_se = config.models[0].kwargs['use_se']
                if 'aspp_out' in config.models[0].kwargs:
                    args.aspp_out = config.models[0].kwargs['aspp_out']
                if 'dec_ch' in config.models[0].kwargs:
                    args.dec_ch = config.models[0].kwargs['dec_ch']
                if 'low_ch_out' in config.models[0].kwargs:
                    args.low_ch_out = config.models[0].kwargs['low_ch_out']
                if 'use_cbam' in config.models[0].kwargs:
                    args.use_cbam = config.models[0].kwargs['use_cbam']
                if 'cbam_reduction_ratio' in config.models[0].kwargs:
                    args.cbam_reduction_ratio = config.models[0].kwargs['cbam_reduction_ratio']
        # 集成模式下
        elif args.ensemble:
            args.models = [model.name for model in config.models]
            args.weights = [model.ens_weight for model in config.models] if hasattr(config.models[0], 'ens_weight') else None
    
    # 更新预测相关参数
    if hasattr(config, 'predict'):
        predict_config = config.predict
        if hasattr(predict_config, 'tile_size'):
            args.tile_size = predict_config.tile_size
        if hasattr(predict_config, 'overlap'):
            args.overlap = predict_config.overlap
        if hasattr(predict_config, 'batch_size'):
            args.batch_size = predict_config.batch_size
        if hasattr(predict_config, 'threshold'):
            args.threshold = predict_config.threshold
        if hasattr(predict_config, 'n_classes'):
            args.n_classes = predict_config.n_classes
        if hasattr(predict_config, 'save_visualization'):
            args.save_visualization = predict_config.save_visualization
        if hasattr(predict_config, 'show_results'):
            args.show_results = predict_config.show_results
        if hasattr(predict_config, 'crf_iterations'):
            args.crf_iterations = predict_config.crf_iterations
        
        # 更新后处理参数
        if hasattr(predict_config, 'postprocessing'):
            post_config = predict_config.postprocessing
            if hasattr(post_config, 'median_kernel_size'):
                args.median_kernel_size = post_config.median_kernel_size
            if hasattr(post_config, 'gaussian_sigma'):
                args.gaussian_sigma = post_config.gaussian_sigma
            if hasattr(post_config, 'morph_close_kernel_size'):
                args.morph_close_kernel_size = post_config.morph_close_kernel_size
            if hasattr(post_config, 'morph_open_kernel_size'):
                args.morph_open_kernel_size = post_config.morph_open_kernel_size
            if hasattr(post_config, 'min_object_size'):
                args.min_object_size = post_config.min_object_size
            if hasattr(post_config, 'hole_area_threshold'):
                args.hole_area_threshold = post_config.hole_area_threshold
            if hasattr(post_config, 'adaptive_threshold'):
                args.adaptive_threshold = post_config.adaptive_threshold
    
    # 更新训练相关参数（用于随机种子）
    if hasattr(config, 'train') and hasattr(config.train, 'batch_size') and args.batch_size is None:
        args.batch_size = config.train.batch_size
    
    return args

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
    elif args.model == 'lightweight_unet':
        from models.lightweight_unet import get_lightweight_unet_model
        model = get_lightweight_unet_model(
            n_channels=len(args.bands),
            n_classes=args.n_classes,
            base_features=args.base_features,
            dropout_rate=args.dropout_rate
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
        model = get_deeplabv3_plus_model(
            n_channels=len(args.bands),
            n_classes=args.n_classes,
            output_stride=args.output_stride,
            pretrained_backbone=args.pretrained_backbone
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
        
        # 为Ultra-Light DeepLabV3+参数设置默认值
        if args.aspp_out is None:
            args.aspp_out = 64
        if args.dec_ch is None:
            args.dec_ch = 64
        if args.low_ch_out is None:
            args.low_ch_out = 32
        if args.use_cbam is None:
            args.use_cbam = False
        if args.cbam_reduction_ratio is None:
            args.cbam_reduction_ratio = 16
        if args.use_se is None:
            args.use_se = False
            
        model = get_ultra_light_deeplabv3_plus(
            n_channels=len(args.bands),
            n_classes=args.n_classes,
            pretrained_backbone=args.pretrained_backbone,
            aspp_out=args.aspp_out,
            dec_ch=args.dec_ch,
            low_ch_out=args.low_ch_out,
            use_cbam=args.use_cbam,
            cbam_reduction_ratio=args.cbam_reduction_ratio,
            output_stride=getattr(args, 'output_stride', 32),
            aspp_rates=args.aspp_rates,
            class_prior=args.class_prior,
            use_se=args.use_se
        )
    else:
        raise ValueError(f"不支持的模型类型: {args.model}")
    
    # 加载检查点
    if args.checkpoint_path is not None:
        logger.info(f'从检查点加载模型: {args.checkpoint_path}')
        checkpoint = torch.load(args.checkpoint_path, map_location=args.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
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
    for model_type, checkpoint_path in zip(args.models, args.checkpoint_paths):
        logger.info(f"加载{model_type}模型: {checkpoint_path}")
        
        if model_type == 'aer_unet':
            model = get_aer_unet_model(
                n_channels=len(args.bands),
                n_classes=args.n_classes,
                base_features=args.base_features,
                dropout_rate=args.dropout_rate
            )
        elif model_type == 'lightweight_unet':
            from models.lightweight_unet import get_lightweight_unet_model
            model = get_lightweight_unet_model(
                n_channels=len(args.bands),
                n_classes=args.n_classes,
                base_features=args.base_features,
                dropout_rate=args.dropout_rate
            )
        elif model_type == 'unet':
            from models.unet_model import get_unet_model
            model = get_unet_model(
                n_channels=len(args.bands),
                n_classes=args.n_classes,
                bilinear=getattr(args, 'unet_bilinear', False)
            )
        elif model_type == 'deeplabv3_plus':
            from models.deeplabv3_plus import get_deeplabv3_plus_model
            model = get_deeplabv3_plus_model(
                n_channels=len(args.bands),
                n_classes=args.n_classes,
                output_stride=args.output_stride,
                pretrained_backbone=args.pretrained_backbone
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        # 设置为评估模式
        model.to(args.device)
        model.eval()
        
        models.append(model)
    
    # 创建集成模型
    ensemble = ModelEnsemble(
        models=models,
        strategy=args.ensemble_strategy,
        weights=args.weights
    )
    
    return ensemble.to(args.device)

def predict_on_tile(model, tile, device):
    """在单个瓦片上进行预测"""
    # 将瓦片转换为tensor并添加批次维度
    tile_tensor = torch.from_numpy(tile).float().to(device)
    tile_tensor = tile_tensor.unsqueeze(0)
    
    # 进行预测
    with torch.no_grad():
        output = model(tile_tensor)
    
    # 移除批次维度并转换回numpy数组
    output = output.squeeze(0).cpu().numpy()
    
    return output

def predict_on_image(model, image_path, args):
    """对单个图像进行预测"""
    logger.info(f'预测图像: {image_path}')
    
    # 加载图像
    image, profile = load_sentinel2_image(image_path, args.bands)
    
    # 图像归一化
    normalized_image = normalize_image(image, method='sentinel')
    
    # 获取图像尺寸
    height, width = normalized_image.shape[1], normalized_image.shape[2]
    
    # 检查是否需要分块处理
    if height <= args.tile_size and width <= args.tile_size:
        # 直接处理整幅图像
        logger.info('图像尺寸合适，直接处理整幅图像')
        prediction = predict_on_tile(model, normalized_image, args.device)
    else:
        # 分块处理
        logger.info(f'图像尺寸较大，分块处理 ({height}x{width})')
        
        # 根据参数选择处理方法
        if hasattr(args, 'use_sliding_window') and args.use_sliding_window:
            logger.info('使用滑窗推理和余弦加权融合')
            prediction = process_large_image_with_sliding_window(model, normalized_image, args)
        else:
            logger.info('使用传统分块处理方法')
            prediction = process_large_image(model, normalized_image, args)
    
    # 应用CRF后处理
    if args.crf_iterations > 0:
        logger.info(f'应用CRF后处理，迭代次数: {args.crf_iterations}')
        
        # 如果是单类预测，直接应用CRF
        if args.n_classes == 1:
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
        else:
            # 多类情况下，对每个类别应用CRF
            refined_prediction = np.zeros_like(prediction)
            for c in range(args.n_classes):
                # 将当前类别的预测结果转换为tensor
                pred_tensor = torch.from_numpy(prediction[c]).float().to(args.device)
                # 将图像数据转换为tensor并归一化到[0,1]范围
                img_tensor = torch.from_numpy(normalized_image).float().to(args.device)
                
                # 应用CRF
                with torch.no_grad():
                    refined_pred = apply_crf_postprocessing(
                        img_tensor, pred_tensor, args.crf_iterations
                    )
                
                # 转换回numpy
                refined_prediction[c] = refined_pred.cpu().numpy()
            
            prediction = refined_prediction
    
    # 应用阈值进行二值化
    if args.n_classes == 1:
        # 对于大图像，使用更节省内存的方式
        try:
            binary_prediction = (prediction > args.threshold).astype(np.float32)
        except MemoryError:
            # 如果内存不足，使用分块处理
            logger.info("内存不足，使用分块处理进行二值化")
            binary_prediction = np.zeros_like(prediction, dtype=np.float32)
            h, w = prediction.shape[-2:]
            tile_size = 1024  # 使用较大的块以减少处理时间
            
            for i in range(0, h, tile_size):
                for j in range(0, w, tile_size):
                    end_i = min(i + tile_size, h)
                    end_j = min(j + tile_size, w)
                    binary_prediction[..., i:end_i, j:end_j] = (prediction[..., i:end_i, j:end_j] > args.threshold).astype(np.float32)
    else:
        # 多类别情况下，选择概率最高的类别
        binary_prediction = np.argmax(prediction, axis=0).astype(np.float32)
    
    # 应用后处理
    if args.n_classes == 1 and (
        args.median_kernel_size > 0 or 
        args.gaussian_sigma > 0.0 or 
        args.morph_close_kernel_size > 0 or 
        args.morph_open_kernel_size > 0 or 
        args.min_object_size > 0 or 
        args.hole_area_threshold > 0 or 
        args.adaptive_threshold
    ):
        logger.info('应用后处理平滑预测结果')
        
        # 对于单类预测，使用第一个通道的概率图
        prob_map = prediction[0] if prediction.ndim == 3 else prediction
        
        # 对于大图像，使用分块处理以避免内存问题
        try:
            # 应用后处理管道
            binary_prediction = apply_postprocessing_pipeline(
                probability_prediction=prob_map,
                binary_prediction=binary_prediction,
                median_kernel_size=args.median_kernel_size,
                gaussian_sigma=args.gaussian_sigma,
                morph_close_kernel_size=args.morph_close_kernel_size,
                morph_open_kernel_size=args.morph_open_kernel_size,
                min_object_size=args.min_object_size,
                hole_area_threshold=args.hole_area_threshold,
                adaptive_threshold=args.adaptive_threshold,
                threshold=args.threshold
            )
        except MemoryError:
            logger.info("内存不足，使用分块处理后处理")
            # 对于大图像，分块处理
            h, w = binary_prediction.shape[-2:]
            tile_size = 1024  # 使用较大的块以减少处理时间
            
            for i in range(0, h, tile_size):
                for j in range(0, w, tile_size):
                    end_i = min(i + tile_size, h)
                    end_j = min(j + tile_size, w)
                    
                    # 提取当前块
                    prob_tile = prob_map[..., i:end_i, j:end_j]
                    binary_tile = binary_prediction[..., i:end_i, j:end_j]
                    
                    # 应用后处理管道
                    processed_tile = apply_postprocessing_pipeline(
                        probability_prediction=prob_tile,
                        binary_prediction=binary_tile,
                        median_kernel_size=args.median_kernel_size,
                        gaussian_sigma=args.gaussian_sigma,
                        morph_close_kernel_size=args.morph_close_kernel_size,
                        morph_open_kernel_size=args.morph_open_kernel_size,
                        min_object_size=args.min_object_size,
                        hole_area_threshold=args.hole_area_threshold,
                        adaptive_threshold=args.adaptive_threshold,
                        threshold=args.threshold
                    )
                    
                    # 将处理后的块放回原位置
                    binary_prediction[..., i:end_i, j:end_j] = processed_tile
    
    # 保存预测结果
    filename = os.path.basename(image_path)
    prediction_path = save_prediction_to_geotiff(binary_prediction, profile, args.output_dir, filename)
    
    # 如果需要，保存概率图
    if args.save_probability:
        # 对于单类预测，使用第一个通道的概率图
        prob_map = prediction[0] if prediction.ndim == 3 else prediction
        save_probability_to_geotiff(prob_map, profile, args.output_dir, filename)
    
    # 如果需要，保存预测概率图
    if args.save_visualization:
        visualize_prediction(normalized_image, binary_prediction, prediction, args.output_dir, filename)
    
    # 如果需要，显示结果
    if args.show_results:
        show_prediction_result(normalized_image, binary_prediction, filename)
    
    return prediction_path

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
    prediction = np.zeros((args.n_classes, height, width), dtype=np.float32)
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
            for c in range(args.n_classes):
                prediction[c, start_h:end_h, start_w:end_w] += output[c] * weight
            weight_map[start_h:end_h, start_w:end_w] += weight
    
    # 归一化预测结果
    for c in range(args.n_classes):
        # 避免除零错误
        mask = weight_map > 0
        prediction[c, mask] /= weight_map[mask]
    
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

def make_cosine_window(tile_size):
    """创建余弦窗口用于边界平滑"""
    import math
    y = 0.5 - 0.5 * np.cos(np.linspace(0, math.pi, tile_size))
    w = np.outer(y, y).astype(np.float32)
    return w

def sliding_window_inference(model, img_tensor, tile_size=256, stride=128, device="cuda"):
    """
    滑窗推理 + 余弦加权融合
    
    Args:
        model: 模型
        img_tensor: (1, C, H, W) tensor, float 已归一化
        tile_size: 瓦片大小
        stride: 滑动步长
        device: 设备
    
    Returns:
        prob: (1, 1, H, W) tensor
    """
    model.eval().to(device)
    img_tensor = img_tensor.to(device)
    _, _, H, W = img_tensor.shape
    
    # 检查图像尺寸，如果图像尺寸小于或等于tile_size，直接使用整图推理
    if H <= tile_size and W <= tile_size:
        # 直接使用整图推理，避免滑窗推理导致的边界反射问题
        with torch.no_grad():
            logits = model(img_tensor)
            prob = torch.sigmoid(logits)
        return prob
    
    # 初始化输出概率图和归一化因子
    try:
        prob = torch.zeros((1, 1, H, W), device=device)
        norm = torch.zeros_like(prob)
    except torch.cuda.OutOfMemoryError:
        # 如果GPU内存不足，使用CPU
        logger.warning("GPU内存不足，切换到CPU进行滑窗推理")
        device = "cpu"
        img_tensor = img_tensor.to(device)
        model = model.to(device)
        prob = torch.zeros((1, 1, H, W), device=device)
        norm = torch.zeros_like(prob)
    
    # 创建余弦窗口
    win = make_cosine_window(tile_size)
    win = torch.from_numpy(win).to(device)
    win = win.unsqueeze(0).unsqueeze(0)  # (1, 1, tile_size, tile_size)
    
    # 滑窗处理
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            # 计算当前瓦片的实际大小（考虑边界）
            actual_tile_size_h = min(tile_size, H - y)
            actual_tile_size_w = min(tile_size, W - x)
            
            # 提取瓦片
            patch = img_tensor[:, :, y:y+actual_tile_size_h, x:x+actual_tile_size_w]
            
            # 处理边界瓦片（可能小于tile_size）
            if patch.shape[-2] < tile_size or patch.shape[-1] < tile_size:
                # 计算需要的填充量
                ph = tile_size - patch.shape[-2] if patch.shape[-2] < tile_size else 0
                pw = tile_size - patch.shape[-1] if patch.shape[-1] < tile_size else 0
                
                # 确保填充量不超过输入维度
                if ph > 0 or pw > 0:
                    # 检查填充量是否合理
                    if ph >= patch.shape[-2] or pw >= patch.shape[-1]:
                        # 如果填充量过大，使用镜像复制的方式扩展瓦片
                        # 这样可以避免填充过多导致的错误
                        if ph >= patch.shape[-2]:
                            # 在高度方向上进行镜像复制
                            patch = torch.cat([patch, torch.flip(patch, dims=(-2,))], dim=-2)
                            # 重新计算填充量
                            ph = tile_size - patch.shape[-2]
                        
                        if pw >= patch.shape[-1]:
                            # 在宽度方向上进行镜像复制
                            patch = torch.cat([patch, torch.flip(patch, dims=(-1,))], dim=-1)
                            # 重新计算填充量
                            pw = tile_size - patch.shape[-1]
                    
                    # 如果仍然需要填充（但现在已经安全），应用填充
                    if ph > 0 or pw > 0:
                        patch = torch.nn.functional.pad(patch, (0, pw, 0, ph), mode="reflect")
            
            # 推理
            try:
                # 使用新的torch.amp.autocast API替代已弃用的torch.cuda.amp.autocast
                with torch.amp.autocast(device_type=device):
                    logit = model(patch)
                    p = torch.sigmoid(logit)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                # 如果推理时内存不足，尝试使用CPU
                if "out of memory" in str(e).lower():
                    logger.warning(f"推理时GPU内存不足，切换到CPU处理瓦片 ({y},{x})")
                    patch_cpu = patch.to("cpu")
                    model_cpu = model.to("cpu")
                    with torch.no_grad():
                        logit = model_cpu(patch_cpu)
                        p = torch.sigmoid(logit)
                    # 将结果移回原设备
                    p = p.to(device)
                    # 如果之前切换到了CPU，将模型移回GPU
                    if device != "cpu":
                        model = model.to(device)
                else:
                    raise e
            
            # 调整预测结果大小（去除填充部分）
            p = p[:, :, :actual_tile_size_h, :actual_tile_size_w]
            
            # 调整窗口大小以匹配实际瓦片大小
            w = win[:, :, :actual_tile_size_h, :actual_tile_size_w]
            
            # 加权融合
            prob[:, :, y:y+actual_tile_size_h, x:x+actual_tile_size_w] += p * w
            norm[:, :, y:y+actual_tile_size_h, x:x+actual_tile_size_w] += w
    
    # 归一化
    return prob / (norm + 1e-6)

def process_large_image_with_sliding_window(model, image, args):
    """使用滑窗推理和余弦加权融合处理大图像"""
    # 获取图像尺寸
    channels, height, width = image.shape
    
    # 将图像转换为tensor并添加批次维度
    img_tensor = torch.from_numpy(image).float().unsqueeze(0)
    
    # 计算stride：通过tile_size和overlap计算
    overlap = getattr(args, 'overlap', args.tile_size // 2)  # 默认使用50%重叠
    stride = args.tile_size - overlap
    
    # 使用滑窗推理
    with torch.no_grad():
        prob_tensor = sliding_window_inference(
            model, 
            img_tensor, 
            tile_size=args.tile_size, 
            stride=stride,
            device=args.device
        )
    
    # 转换回numpy数组
    prediction = prob_tensor.squeeze(0).cpu().numpy()
    
    return prediction

def save_probability_to_geotiff(probability, profile, output_dir, filename):
    """保存预测概率图为GeoTIFF文件"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建输出文件路径
    base_name = os.path.splitext(filename)[0]
    output_path = os.path.join(output_dir, f'{base_name}_probability.tif')
    
    # 如果文件已存在，尝试删除它
    if os.path.exists(output_path):
        try:
            os.remove(output_path)
            logger.info(f'已删除现有的概率图文件: {output_path}')
        except PermissionError:
            # 如果无法删除，尝试使用时间戳创建新文件名
            import time
            timestamp = int(time.time())
            output_path = os.path.join(output_dir, f'{base_name}_probability_{timestamp}.tif')
            logger.warning(f'无法删除现有文件，使用新文件名: {output_path}')
    
    # 更新profile
    profile.update(
        dtype=rasterio.float32,
        count=1,
        compress='lzw'
    )
    
    # 保存概率图
    with rasterio.open(output_path, 'w', **profile) as dst:
        # 如果概率图是(1, H, W)形状，则去除第一个维度
        if probability.ndim == 3 and probability.shape[0] == 1:
            dst.write(probability[0].astype(np.float32), 1)
        else:
            dst.write(probability.astype(np.float32), 1)
    
    logger.info(f'概率图已保存到: {output_path}')
    
    return output_path

def save_prediction_to_geotiff(prediction, profile, output_dir, filename):
    """保存预测结果为GeoTIFF文件"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建输出文件路径
    base_name = os.path.splitext(filename)[0]
    output_path = os.path.join(output_dir, f'{base_name}_prediction.tif')
    
    # 如果文件已存在，尝试删除它
    if os.path.exists(output_path):
        try:
            os.remove(output_path)
            logger.info(f'已删除现有的预测结果文件: {output_path}')
        except PermissionError:
            # 如果无法删除，尝试使用时间戳创建新文件名
            import time
            timestamp = int(time.time())
            output_path = os.path.join(output_dir, f'{base_name}_prediction_{timestamp}.tif')
            logger.warning(f'无法删除现有文件，使用新文件名: {output_path}')
    
    # 更新profile
    profile.update(
        dtype=rasterio.float32,
        count=1,
        compress='lzw'
    )
    
    # 保存预测结果
    with rasterio.open(output_path, 'w', **profile) as dst:
        # 如果预测结果是(1, H, W)形状，则去除第一个维度
        if prediction.ndim == 3 and prediction.shape[0] == 1:
            dst.write(prediction[0].astype(np.float32), 1)
        else:
            dst.write(prediction.astype(np.float32), 1)
    
    logger.info(f'预测结果已保存到: {output_path}')
    
    return output_path

def visualize_prediction(image, binary_prediction, probability_prediction, output_dir, filename):
    """可视化预测结果"""
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

def show_prediction_result(image, binary_prediction, filename):
    """显示预测结果"""
    # 确保中文显示正常
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    
    # 创建画布
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # 绘制RGB图像
    if image.shape[0] >= 3:
        # 重组波段以显示RGB图像
        rgb_image = np.zeros((image.shape[1], image.shape[2], 3))
        
        # 根据实际波段顺序调整
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
    
    # 绘制二值预测结果
    axes[1].imshow(binary_prediction[0], cmap='Blues')
    axes[1].set_title('二值预测结果')
    axes[1].axis('off')
    
    # 添加文件名
    plt.suptitle(f'文件: {filename}', fontsize=16)
    plt.tight_layout()
    
    # 显示结果
    plt.show(block=False)
    plt.pause(3)  # 显示3秒
    plt.close(fig)

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 从YAML配置文件更新参数
    args = update_args_with_config(args)
    
    # 验证输入目录
    if args.input_dir is None:
        raise ValueError("必须指定输入图像目录")
    
    # 验证参数
    if args.ensemble:
        if args.models is None or len(args.models) == 0 or args.checkpoint_paths is None or len(args.checkpoint_paths) == 0:
            raise ValueError("集成模式需要指定models和checkpoint_paths参数")
        if len(args.models) != len(args.checkpoint_paths):
            raise ValueError("模型类型数量必须与检查点路径数量匹配")
    else:
        if args.model is None or args.checkpoint_path is None:
            raise ValueError("单模型预测需要指定model和checkpoint_path参数")
    
    # 设置随机种子
    seed = 42
    if args.config is not None:
        config = get_config(yaml_config_path=args.config)
        if hasattr(config, 'train') and hasattr(config.train, 'seed'):
            seed = config.train.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 检查输入目录是否存在
    if not os.path.exists(args.input_dir):
        raise ValueError(f"输入目录不存在: {args.input_dir}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置模型
    if args.ensemble:
        model = setup_ensemble_model(args)
    else:
        model = setup_single_model(args)
    
    # 收集所有要处理的图像文件
    import fnmatch
    image_files = []
    for root, _, files in os.walk(args.input_dir):
        for file in files:
            if file.lower().endswith(('.tif', '.tiff')):
                # 使用file_pattern进行过滤
                if fnmatch.fnmatch(file, args.file_pattern):
                    image_files.append(os.path.join(root, file))
    
    # 验证是否找到了图像文件
    if len(image_files) == 0:
        raise ValueError(f"在输入目录中找不到匹配的图像文件: {args.input_dir}")
    
    logger.info(f'找到 {len(image_files)} 个图像文件')
    
    # 处理每个图像
    for image_file in tqdm(image_files, desc='处理图像'):
        predict_on_image(model, image_file, args)
    
    logger.info("预测完成!")

if __name__ == '__main__':
    main()
