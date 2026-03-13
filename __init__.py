# models/__init__.py
"""
模型模块初始化文件，提供统一的模型创建接口。
"""

# 导入所有模型创建函数
from models.aer_unet import get_aer_unet_model
try:
    from models.lightweight_unet import get_lightweight_unet_model
    _HAS_LIGHTWEIGHT_UNET = True
except ModuleNotFoundError:
    get_lightweight_unet_model = None
    _HAS_LIGHTWEIGHT_UNET = False
from models.unet_model import get_unet_model
from models.deeplabv3_plus import get_deeplabv3_plus_model

# 模型类型枚举
class ModelType:
    AER_UNET = "aer_unet"
    LIGHTWEIGHT_UNET = "lightweight_unet"
    UNET = "unet"
    DEEPLABV3_PLUS = "deeplabv3_plus"

# 模型创建工厂
def create_model(model_type: str, **kwargs):
    """
    根据模型类型创建模型实例
    
    Args:
        model_type: 模型类型，可选值为:
            - "aer_unet": AER U-Net模型
            - "lightweight_unet": 轻量级U-Net模型
            - "deeplabv3_plus": DeepLabV3+模型
        **kwargs: 模型特定参数
    
    Returns:
        创建的模型实例
    
    Raises:
        ValueError: 当模型类型不支持时
    """
    if model_type == ModelType.AER_UNET:
        return get_aer_unet_model(**kwargs)
    elif model_type == ModelType.LIGHTWEIGHT_UNET:
        if not _HAS_LIGHTWEIGHT_UNET:
            raise ValueError("lightweight_unet 模型文件缺失，无法创建该模型。")
        return get_lightweight_unet_model(**kwargs)
    elif model_type == ModelType.UNET:
        return get_unet_model(**kwargs)
    elif model_type == ModelType.DEEPLABV3_PLUS:
        return get_deeplabv3_plus_model(**kwargs)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

# 获取所有支持的模型类型
def get_supported_model_types():
    """获取所有支持的模型类型列表"""
    return [
        ModelType.AER_UNET,
        *( [ModelType.LIGHTWEIGHT_UNET] if _HAS_LIGHTWEIGHT_UNET else [] ),
        ModelType.UNET,
        ModelType.DEEPLABV3_PLUS
    ]

# 获取模型信息
def get_model_info(model_type: str):
    """
    获取模型信息
    
    Args:
        model_type: 模型类型
    
    Returns:
        包含模型信息的字典
    
    Raises:
        ValueError: 当模型类型不支持时
    """
    model_info = {
        ModelType.AER_UNET: {
            "name": "AER U-Net",
            "description": "Attention-Enhanced Multi-Scale Residual U-Net",
            "architecture": "CNN",
            "complexity": "Medium",
            "parameters": "约5-10M",
            "suitable_for": "高精度水体分割，需要中等计算资源"
        },
        ModelType.LIGHTWEIGHT_UNET: {
            "name": "Lightweight U-Net",
            "description": "使用深度可分离卷积的轻量级U-Net",
            "architecture": "CNN",
            "complexity": "Low",
            "parameters": "约1-2M",
            "suitable_for": "快速水体分割，适合资源受限环境"
        },
        ModelType.UNET: {
            "name": "U-Net",
            "description": "Standard U-Net encoder-decoder with skip connections",
            "architecture": "CNN",
            "complexity": "Medium",
            "parameters": "约30M",
            "suitable_for": "通用分割基线对比"
        },
        ModelType.DEEPLABV3_PLUS: {
            "name": "DeepLabV3+",
            "description": "基于空洞卷积和ASPP模块的语义分割模型",
            "architecture": "CNN with Atrous Convolutions",
            "complexity": "High",
            "parameters": "约40-50M",
            "suitable_for": "高精度语义分割，需要较强计算资源"
        }
    }
    
    if model_type in model_info:
        return model_info[model_type]
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

# 打印所有模型信息
def print_all_models_info():
    """打印所有模型的信息"""
    print("支持的模型类型:")
    print("=" * 80)
    
    for model_type in get_supported_model_types():
        info = get_model_info(model_type)
        print(f"模型类型: {model_type}")
        print(f"名称: {info['name']}")
        print(f"描述: {info['description']}")
        print(f"架构: {info['architecture']}")
        print(f"复杂度: {info['complexity']}")
        print(f"参数量: {info['parameters']}")
        print(f"适用场景: {info['suitable_for']}")
        print("-" * 80)

if __name__ == "__main__":
    print_all_models_info()
