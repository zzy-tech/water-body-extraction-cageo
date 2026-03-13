# config.py
"""
Sentinel-2水体分割项目配置文件。
集中管理模型训练、评估和预测的相关参数，支持从YAML文件加载配置。
"""
import os
import yaml

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 数据相关配置
class DataConfig:
    # 数据集路径
    DATA_DIR = os.path.join(PROJECT_ROOT, 'datasets')
    
    # 检查点和预测结果保存路径
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
    PREDICTION_DIR = os.path.join(PROJECT_ROOT, 'predictions')
    
    # 训练、验证、测试数据目录
    TRAIN_DIR = os.path.join(DATA_DIR, 'train')
    VAL_DIR = os.path.join(DATA_DIR, 'val')
    TEST_DIR = os.path.join(DATA_DIR, 'test')
    
    # Sentinel-2波段配置（蓝、绿、红、近红、短波红外1、短波红外2）
    BANDS = [1, 2, 3, 4, 5, 6]  # 默认使用所有6个波段
    N_CHANNELS = len(BANDS)
    
    # 图像尺寸配置
    IMAGE_HEIGHT = 256  # 默认图像高度
    IMAGE_WIDTH = 256   # 默认图像宽度
    
    # 数据增强配置
    AUGMENT = True      # 是否使用数据增强
    
    # 数据归一化方法（'minmax', 'zscore', 'sentinel'）
    NORMALIZE_METHOD = 'sentinel'
    
    # 批量大小和工作进程数
    BATCH_SIZE = 8
    NUM_WORKERS = 4
    
    # 水体类别数
    N_CLASSES = 1

# AER U-Net模型配置
class AerUNetConfig:
    # 基础特征数
    BASE_FEATURES = 32
    
    # Dropout率
    DROPOUT_RATE = 0.3
    
    # 下采样方法
    DOWNSAMPLE_METHOD = 'maxpool'
    
    # 上采样方法
    UPSAMPLE_METHOD = 'bilinear'
    
    # 检查点保存路径
    CHECKPOINT_PATH = os.path.join(DataConfig.CHECKPOINT_DIR, 'aer_unet_best.pth')

# Lightweight U-Net模型配置
class LightweightUNetConfig:
    # 基础特征数
    BASE_FEATURES = 16
    
    # Dropout率
    DROPOUT_RATE = 0.2
    
    # 检查点保存路径
    CHECKPOINT_PATH = os.path.join(DataConfig.CHECKPOINT_DIR, 'lightweight_unet_best.pth')

# DeepLabV3+配置
class DeepLabV3PlusConfig:
    # 输出步长，控制特征图下采样比例
    OUTPUT_STRIDE = 16
    
    # 是否使用预训练骨干网络
    PRETRAINED_BACKBONE = True

# 轻量级DeepLabV3+配置
class LightweightDeepLabV3PlusConfig:
    # 输出步长，控制特征图下采样比例
    OUTPUT_STRIDE = 16
    
    # 是否使用预训练骨干网络
    PRETRAINED_BACKBONE = True
    
    # 检查点保存路径
    CHECKPOINT_PATH = os.path.join(DataConfig.CHECKPOINT_DIR, 'lightweight_deeplabv3_plus_best.pth')

# 超轻量级DeepLabV3+配置
class UltraLightweightDeepLabV3PlusConfig:
    # 输出步长，控制特征图下采样比例
    OUTPUT_STRIDE = 32
    
    # 是否使用预训练骨干网络
    PRETRAINED_BACKBONE = True
    
    # ASPP模块输出通道数
    ASPP_OUT = 64
    
    # 解码器通道数
    DEC_CH = 64
    
    # 低层特征通道数
    LOW_CH_OUT = 32
    
    # 检查点保存路径
    CHECKPOINT_PATH = os.path.join(DataConfig.CHECKPOINT_DIR, 'ultra_lightweight_deeplabv3_plus_best.pth')

# 模型集成配置
class EnsembleConfig:
    # 要集成的模型类型列表
    MODELS = ['aer_unet', 'lightweight_unet']
    
    # 要集成的模型检查点路径列表
    CHECKPOINT_PATHS = [
        AerUNetConfig.CHECKPOINT_PATH,
        LightweightUNetConfig.CHECKPOINT_PATH
    ]
    
    # 集成策略（'mean', 'weighted_mean', 'vote', 'logits_mean'）
    STRATEGY = 'weighted_mean'
    
    # 用于加权平均策略的权重列表
    WEIGHTS = [0.5, 0.5]  # 相等权重
    
    # 集成模型检查点保存路径
    CHECKPOINT_PATH = os.path.join(DataConfig.CHECKPOINT_DIR, 'ensemble_best.pth')

# 训练配置
class TrainConfig:
    # 训练设备
    DEVICE = 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') is not None else 'cpu'
    
    # 训练轮数
    EPOCHS = 50
    
    # 学习率
    LEARNING_RATE = 1e-4
    
    # 学习率调度器配置
    SCHEDULER_TYPE = 'cosine'
    MIN_LR = 1e-6
    
    # 权重衰减
    WEIGHT_DECAY = 1e-5
    
    # 优化器类型
    OPTIMIZER_TYPE = 'adamw'
    
    # 梯度裁剪
    GRAD_CLIP = 1.0
    
    # 早停机制
    EARLY_STOPPING = True
    PATIENCE = 10
    
    # 检查点保存配置
    SAVE_CHECKPOINT = True
    SAVE_EVERY_N_EPOCHS = 5
    
    # TensorBoard配置
    TENSORBOARD = True
    TENSORBOARD_DIR = os.path.join(PROJECT_ROOT, 'runs')
    
    # 随机种子（确保可重复性）
    SEED = 42

# 评估配置
class EvalConfig:
    # 评估设备
    DEVICE = TrainConfig.DEVICE
    
    # 评估数据集分割
    SPLIT = 'test'  # 可选: 'val', 'test'
    
    # 批量大小
    BATCH_SIZE = DataConfig.BATCH_SIZE
    
    # 二值化阈值
    THRESHOLD = 0.5
    
    # 是否绘制预测示例
    PLOT_EXAMPLES = True
    
    # 预测示例数量
    NUM_EXAMPLES = 5

# 预测配置
class PredictConfig:
    # 预测设备
    DEVICE = TrainConfig.DEVICE
    
    # 二值化阈值
    THRESHOLD = 0.5
    
    # 批量大小
    BATCH_SIZE = 8
    
    # 处理大影像时的瓦片大小
    TILE_SIZE = 256
    
    # 处理大影像时的瓦片重叠大小
    OVERLAP = 32
    
    # 是否显示预测结果
    SHOW_RESULTS = False
    
    # 是否保存可视化结果
    SAVE_VISUALIZATION = True
    
    # 可视化结果保存目录
    VISUALIZATION_DIR = os.path.join(DataConfig.PREDICTION_DIR, 'visualizations')

# 日志配置
class LogConfig:
    # 日志级别
    LOG_LEVEL = 'INFO'
    
    # 日志文件
    LOG_FILE = os.path.join(PROJECT_ROOT, 'sentinel2_water_segmentation.log')
    
    # 日志格式
    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

# 阈值分析配置
class ThresholdConfig:
    # 阈值搜索范围
    THRESHOLD_RANGE = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # 是否进行阈值分析
    ANALYZE = True

# 自定义配置函数（允许用户覆盖默认配置）
def get_custom_config(config_dict=None):
    """
    根据字典覆盖默认配置。
    
    参数:
        config_dict: 包含要覆盖的配置项的字典
    
    返回:
        包含所有配置的字典
    """
    # 构建默认配置字典
    config = {
        'data': DataConfig,
        'aer_unet': AerUNetConfig,
        'lightweight_unet': LightweightUNetConfig,
        'deeplabv3_plus': DeepLabV3PlusConfig,
        'lightweight_deeplabv3_plus': LightweightDeepLabV3PlusConfig,
        'ultra_lightweight_deeplabv3_plus': UltraLightweightDeepLabV3PlusConfig,
        'ensemble': EnsembleConfig,
        'train': TrainConfig,
        'eval': EvalConfig,
        'predict': PredictConfig,
        'log': LogConfig,
        'threshold': ThresholdConfig,
        'model': DeepLabV3PlusConfig  # 添加model配置，默认使用DeepLabV3+配置
    }
    
    # 如果提供了自定义配置，更新默认配置
    if config_dict is not None:
        for section, section_config in config_dict.items():
            if section in config:
                for key, value in section_config.items():
                    if hasattr(config[section], key):
                        setattr(config[section], key, value)
    
    # 将配置类转换为字典
    config_dict_result = {}
    for section_name, section_class in config.items():
        section_dict = {}
        for attr_name in dir(section_class):
            # 过滤掉特殊方法和私有属性
            if not attr_name.startswith('_') and not callable(getattr(section_class, attr_name)):
                section_dict[attr_name] = getattr(section_class, attr_name)
        config_dict_result[section_name] = section_dict
    
    return config_dict_result

# 示例配置（用于快速开始）
# 可以创建不同的配置集来适应不同的实验
EXAMPLE_CONFIGS = {
    # 基础配置
    'basic': {
        'data': {
            'batch_size': 8,
            'n_channels': 6
        },
        'train': {
            'epochs': 30,
            'lr': 1e-4
        }
    },
    
    # 快速测试配置
    'quick_test': {
        'data': {
            'batch_size': 4
        },
        'train': {
            'epochs': 5,
            'lr': 1e-3
        }
    },
    
    # 高精度配置
    'high_precision': {
        'data': {
            'batch_size': 8,
            'image_height': 256,
            'image_width': 256
        },
        'train': {
            'epochs': 100,
            'lr': 5e-5,
            'weight_decay': 5e-5
        },
        'aer_unet': {
            'base_features': 64
        }
    }
}

# 获取指定的示例配置
def get_example_config(config_name):
    """
    获取指定名称的示例配置。
    
    参数:
        config_name: 示例配置名称
    
    返回:
        示例配置字典
    """
    if config_name in EXAMPLE_CONFIGS:
        return EXAMPLE_CONFIGS[config_name]
    else:
        print(f"警告: 未找到名为'{config_name}'的示例配置，使用默认配置。")
        return {}

def load_config_from_yaml(yaml_path):
    """
    从YAML文件加载配置。
    
    参数:
        yaml_path: YAML配置文件路径
    
    返回:
        配置字典
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"配置文件不存在: {yaml_path}")
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # 直接返回YAML文件的内容，不做转换
    # 这样可以保留原始配置文件的结构
    return config_dict

# 导出所有配置供其他脚本使用
def get_config(config_name=None, custom_config=None, yaml_config_path=None):
    """
    获取最终配置，支持使用示例配置、自定义配置或YAML配置文件。
    
    参数:
        config_name: 示例配置名称（可选）
        custom_config: 自定义配置字典（可选）
        yaml_config_path: YAML配置文件路径（可选）
    
    返回:
        包含所有配置的字典
    """
    # 首先获取基础配置
    config = get_custom_config()
    
    # 如果指定了示例配置，应用它
    if config_name is not None:
        example_config = get_example_config(config_name)
        config = get_custom_config(example_config)
    
    # 如果指定了YAML配置文件，将其合并到基础配置中
    if yaml_config_path is not None:
        yaml_config = load_config_from_yaml(yaml_config_path)
        # 合并YAML配置到基础配置中
        for section, section_config in yaml_config.items():
            if section not in config:
                config[section] = {}
            
            # 检查section_config是否是字典类型
            if isinstance(section_config, dict):
                for key, value in section_config.items():
                    config[section][key] = value
            else:
                # 如果不是字典类型（如列表），直接赋值
                config[section] = section_config
    
    # 如果提供了自定义配置，应用它（会覆盖示例配置和YAML配置中的冲突项）
    if custom_config is not None:
        # 合并自定义配置到现有配置中
        for section, section_config in custom_config.items():
            if section not in config:
                config[section] = {}
            for key, value in section_config.items():
                config[section][key] = value
    
    return config