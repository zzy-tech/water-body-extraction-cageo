# utils/data_utils.py
"""
数据处理工具函数和类，用于Sentinel-2影像的加载、预处理、增强和数据集创建。
特别支持6波段数据（蓝、绿、红、近红、短波红外1、短波红外2）。
"""
import os
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms as transforms  # 暂时注释掉以避免导入问题
import rasterio
import logging
import random
from typing import List, Tuple, Dict, Optional, Union

logger = logging.getLogger(__name__)

def load_sentinel2_image(image_path: str, bands: List[int] = None) -> Tuple[np.ndarray, dict]:
    """
    加载Sentinel-2影像，支持指定波段选择。
    
    Args:
        image_path: 影像文件路径
        bands: 要加载的波段索引列表，如果为None则加载所有波段
        
    Returns:
        tuple: (加载的影像数据，形状为 [bands, height, width]; 影像元数据profile)
    """
    try:
        with rasterio.open(image_path) as src:
            # 读取元数据
            height, width = src.shape
            count = src.count
            
            # 如果未指定波段，则加载所有波段
            if bands is None:
                bands = list(range(1, count + 1))
            else:
                # 确保波段索引有效
                valid_bands = [b for b in bands if 1 <= b <= count]
                if not valid_bands:
                    raise ValueError(f"没有有效的波段索引，文件包含 {count} 个波段")
                bands = valid_bands
            
            # 尝试分批读取波段以减少内存峰值使用
            try:
                # 首先尝试创建完整的数组
                image = np.zeros((len(bands), height, width), dtype=np.float32)
                
                # 分批处理，每次处理1个波段以减少内存使用
                for i, band_idx in enumerate(bands):
                    # 直接读取到预分配的数组中，避免中间变量
                    try:
                        band_data = src.read(band_idx)
                        image[i] = band_data.astype(np.float32)
                        # 立即释放临时变量
                        del band_data
                    except MemoryError:
                        # 如果仍然内存不足，尝试逐行读取
                        logger.warning(f"内存不足，尝试逐行读取波段 {band_idx}")
                        for row in range(height):
                            try:
                                row_data = src.read(band_idx, window=((row, row+1), (0, width)))
                                image[i, row, :] = row_data.astype(np.float32)
                                del row_data
                            except MemoryError:
                                # 如果逐行读取仍然失败，尝试分块读取
                                logger.warning(f"逐行读取仍然失败，尝试分块读取波段 {band_idx}")
                                chunk_size = max(1, height // 10)  # 将图像分成10块
                                for chunk_start in range(0, height, chunk_size):
                                    chunk_end = min(chunk_start + chunk_size, height)
                                    try:
                                        chunk_data = src.read(band_idx, window=((chunk_start, chunk_end), (0, width)))
                                        image[i, chunk_start:chunk_end, :] = chunk_data.astype(np.float32)
                                        del chunk_data
                                    except MemoryError as e:
                                        raise IOError(f"无法为波段 {band_idx} 分配内存，即使使用分块读取也失败: {str(e)}")
                    
                    # 每处理完一个波段后进行垃圾回收
                    if i % 2 == 0:  # 每处理2个波段回收一次
                        import gc
                        gc.collect()
                
                # 获取元数据profile
                profile = src.profile
                
                return image, profile
                
            except MemoryError:
                # 如果创建完整数组失败，尝试使用更小的数据类型或分块处理
                logger.warning(f"无法为完整图像分配内存，尝试使用分块处理")
                
                # 创建一个空列表来存储每个波段的分块数据
                band_chunks = []
                
                # 分块大小 - 根据可用内存调整
                chunk_height = max(1, height // 4)  # 将图像分成4块
                
                for band_idx in bands:
                    band_data = np.zeros((height, width), dtype=np.float32)
                    
                    # 分块读取当前波段
                    for chunk_start in range(0, height, chunk_height):
                        chunk_end = min(chunk_start + chunk_height, height)
                        
                        try:
                            # 读取当前块
                            chunk_data = src.read(band_idx, window=((chunk_start, chunk_end), (0, width)))
                            band_data[chunk_start:chunk_end, :] = chunk_data.astype(np.float32)
                            del chunk_data
                        except MemoryError as e:
                            raise IOError(f"无法为波段 {band_idx} 的块 [{chunk_start}:{chunk_end}] 分配内存: {str(e)}")
                        
                        # 每处理完一个块后进行垃圾回收
                        import gc
                        gc.collect()
                    
                    band_chunks.append(band_data)
                
                # 将所有波段堆叠成一个数组
                try:
                    image = np.stack(band_chunks, axis=0)
                    # 清理临时变量
                    del band_chunks
                    
                    # 获取元数据profile
                    profile = src.profile
                    
                    return image, profile
                except MemoryError as e:
                    raise IOError(f"无法将波段数据堆叠成完整图像: {str(e)}")
                    
    except Exception as e:
        raise IOError(f"加载影像文件失败: {image_path}. 错误: {str(e)}")

def load_mask(mask_path: str) -> np.ndarray:
    """
    加载水体掩码文件。
    
    Args:
        mask_path: 掩码文件路径
        
    Returns:
        加载的掩码数据，形状为 [1, height, width]，值为0或1
    """
    try:
        # 尝试使用rasterio加载（如果是GeoTIFF格式）
        try:
            with rasterio.open(mask_path) as src:
                mask = src.read(1).astype(np.float32)
        except:
            # 如果失败，尝试使用OpenCV加载
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        
        # 确保掩码是二值的
        mask = (mask > 0.5).astype(np.float32)
        
        # 添加通道维度
        mask = np.expand_dims(mask, axis=0)
        
        return mask
    except Exception as e:
        raise IOError(f"加载掩码文件失败: {mask_path}. 错误: {str(e)}")

def normalize_image(image: np.ndarray, method: str = 'minmax', params: Dict = None) -> np.ndarray:
    """
    对影像进行归一化处理。
    
    Args:
        image: 输入影像数据
        method: 归一化方法，可选 'minmax', 'zscore', 'sentinel'
        params: 归一化参数，如min, max, mean, std等
        
    Returns:
        归一化后的影像数据
    """
    # 直接在原始数组上操作，避免创建副本以减少内存使用
    normalized_image = image
    
    if method == 'minmax':
        # 0-1范围归一化
        for i in range(image.shape[0]):
            band_min = params.get('min', {}).get(i, image[i].min()) if params else image[i].min()
            band_max = params.get('max', {}).get(i, image[i].max()) if params else image[i].max()
            if band_max - band_min > 0:
                normalized_image[i] = (image[i] - band_min) / (band_max - band_min)
            else:
                normalized_image[i] = 0
    
    elif method == 'zscore':
        # Z-score归一化（均值为0，标准差为1）
        for i in range(image.shape[0]):
            band_mean = params.get('mean', {}).get(i, image[i].mean()) if params else image[i].mean()
            band_std = params.get('std', {}).get(i, image[i].std()) if params else image[i].std()
            if band_std > 0:
                normalized_image[i] = (image[i] - band_mean) / band_std
            else:
                normalized_image[i] = image[i] - band_mean
    
    elif method == 'sentinel':
        # Sentinel-2特定的归一化（基于常用的统计值）
        # 顺序：蓝、绿、红、近红、短波红外1、短波红外2
        sentinel_means = np.array([1379.82, 1287.89, 1210.28, 1055.91, 2199.17, 1595.94])
        sentinel_stds = np.array([156.83, 235.39, 298.42, 462.18, 537.47, 503.59])
        
        # 确保输入影像有足够的通道
        num_bands = min(image.shape[0], len(sentinel_means))
        
        for i in range(num_bands):
            mean = params.get('mean', {}).get(i, sentinel_means[i]) if params else sentinel_means[i]
            std = params.get('std', {}).get(i, sentinel_stds[i]) if params else sentinel_stds[i]
            if std > 0:
                normalized_image[i] = (image[i] - mean) / std
            else:
                normalized_image[i] = image[i] - mean
            
        # 限制范围在[-3, 3]之间
        normalized_image = np.clip(normalized_image, -3, 3)
    
    else:
        raise ValueError(f"不支持的归一化方法: {method}")
    
    return normalized_image

def augment_image_and_mask(image: np.ndarray, mask: np.ndarray, 
                          rotation_range: float = 15, 
                          width_shift_range: float = 0.1, 
                          height_shift_range: float = 0.1, 
                          scale_range: Tuple[float, float] = (0.9, 1.1), 
                          horizontal_flip: bool = True, 
                          vertical_flip: bool = True, 
                          brightness_range: Tuple[float, float] = (0.9, 1.1), 
                          contrast_range: Tuple[float, float] = (0.9, 1.1),
                          saturation_range: Optional[Tuple[float, float]] = None,
                          hue_range: Optional[Tuple[float, float]] = None,
                          noise_std: float = 0.0,
                          elastic_alpha: float = 1000,
                          elastic_sigma: float = 8) -> Tuple[np.ndarray, np.ndarray]:
    """
    对影像和掩码进行数据增强。
    
    Args:
        image: 输入影像数据
        mask: 输入掩码数据
        rotation_range: 旋转角度范围
        width_shift_range: 水平平移范围
        height_shift_range: 垂直平移范围
        scale_range: 缩放范围
        horizontal_flip: 是否进行水平翻转
        vertical_flip: 是否进行垂直翻转
        brightness_range: 亮度调整范围
        contrast_range: 对比度调整范围
        saturation_range: 饱和度调整范围（仅对RGB影像有效）
        hue_range: 色调调整范围（仅对RGB影像有效）
        noise_std: 高斯噪声标准差
        elastic_alpha: 弹性变形的alpha参数
        elastic_sigma: 弹性变形的sigma参数
        
    Returns:
        增强后的影像和掩码数据
    """
    # 获取影像尺寸
    _, height, width = image.shape
    
    # 创建变换矩阵
    transform_matrix = np.eye(3)
    
    # 随机旋转
    if rotation_range > 0:
        angle = random.uniform(-rotation_range, rotation_range)
        rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        transform_matrix[:2, :] = rotation_matrix
    
    # 随机平移
    if width_shift_range > 0 or height_shift_range > 0:
        tx = random.uniform(-width_shift_range, width_shift_range) * width
        ty = random.uniform(-height_shift_range, height_shift_range) * height
        transform_matrix[0, 2] += tx
        transform_matrix[1, 2] += ty
    
    # 随机缩放
    if scale_range is not None and len(scale_range) == 2:
        zoom = random.uniform(scale_range[0], scale_range[1])
        transform_matrix[0, 0] *= zoom
        transform_matrix[1, 1] *= zoom
        
        # 调整平移以保持图像中心不变
        transform_matrix[0, 2] = (1 - zoom) * width / 2 + transform_matrix[0, 2]
        transform_matrix[1, 2] = (1 - zoom) * height / 2 + transform_matrix[1, 2]
    
    # 应用仿射变换到每个波段
    augmented_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        augmented_image[i] = cv2.warpAffine(
            image[i], transform_matrix[:2, :], (width, height),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )
    
    # 应用仿射变换到掩码（使用最近邻插值保持二值性）
    augmented_mask = cv2.warpAffine(
        mask[0], transform_matrix[:2, :], (width, height),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT
    )
    augmented_mask = np.expand_dims(augmented_mask, axis=0)
    
    # 随机水平翻转
    if horizontal_flip and random.random() > 0.5:
        augmented_image = np.flip(augmented_image, axis=2).copy()
        augmented_mask = np.flip(augmented_mask, axis=2).copy()
    
    # 随机垂直翻转
    if vertical_flip and random.random() > 0.5:
        augmented_image = np.flip(augmented_image, axis=1).copy()
        augmented_mask = np.flip(augmented_mask, axis=1).copy()
    
    # 随机亮度调整
    if brightness_range is not None and random.random() > 0.5:
        brightness_factor = random.uniform(brightness_range[0], brightness_range[1])
        augmented_image = augmented_image * brightness_factor
    
    # 随机对比度调整
    if contrast_range is not None and random.random() > 0.5:
        contrast_factor = random.uniform(contrast_range[0], contrast_range[1])
        # 对每个波段分别调整对比度
        for i in range(augmented_image.shape[0]):
            mean = augmented_image[i].mean()
            augmented_image[i] = (augmented_image[i] - mean) * contrast_factor + mean
    
    # 饱和度调整（仅对前3个波段，假设是RGB）
    # 注意：对于Sentinel-2多光谱数据，HSV变换可能不适用，因为数据不是RGB格式
    # 这里保留实现但添加警告，建议在配置中禁用saturation_range和hue_range
    if (saturation_range is not None and 
        random.random() > 0.5 and 
        augmented_image.shape[0] >= 3 and
        saturation_range[0] != saturation_range[1]):  # 添加范围检查
        print("警告：饱和度调整可能不适用于Sentinel-2多光谱数据，建议在配置中禁用saturation_range")
        saturation_factor = random.uniform(saturation_range[0], saturation_range[1])
        # 转换为HSV调整饱和度
        rgb_image = np.transpose(augmented_image[:3], (1, 2, 0))
        # 归一化到0-1范围
        rgb_norm = np.clip(rgb_image / 255.0, 0, 1) if rgb_image.max() > 1 else rgb_image
        # 转换为HSV
        hsv = cv2.cvtColor(rgb_norm, cv2.COLOR_RGB2HSV)
        # 调整饱和度
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 1)
        # 转换回RGB
        rgb_adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        # 恢复原始范围
        if rgb_image.max() > 1:
            rgb_adjusted = rgb_adjusted * 255
        # 更新前3个波段
        augmented_image[:3] = np.transpose(rgb_adjusted, (2, 0, 1))
    
    # 色调调整（仅对前3个波段，假设是RGB）
    # 注意：对于Sentinel-2多光谱数据，HSV变换可能不适用，因为数据不是RGB格式
    # 这里保留实现但添加警告，建议在配置中禁用saturation_range和hue_range
    if (hue_range is not None and 
        random.random() > 0.5 and 
        augmented_image.shape[0] >= 3 and
        hue_range[0] != hue_range[1]):  # 添加范围检查
        print("警告：色调调整可能不适用于Sentinel-2多光谱数据，建议在配置中禁用hue_range")
        hue_shift = random.uniform(hue_range[0], hue_range[1])
        # 转换为HSV调整色调
        rgb_image = np.transpose(augmented_image[:3], (1, 2, 0))
        # 归一化到0-1范围
        rgb_norm = np.clip(rgb_image / 255.0, 0, 1) if rgb_image.max() > 1 else rgb_image
        # 转换为HSV
        hsv = cv2.cvtColor(rgb_norm, cv2.COLOR_RGB2HSV)
        # 调整色调（转换为0-180范围，因为OpenCV使用0-180）
        hue_shift_scaled = hue_shift * 90  # 将[-1, 1]范围映射到[-90, 90]
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift_scaled) % 180
        # 转换回RGB
        rgb_adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        # 恢复原始范围
        if rgb_image.max() > 1:
            rgb_adjusted = rgb_adjusted * 255
        # 更新前3个波段
        augmented_image[:3] = np.transpose(rgb_adjusted, (2, 0, 1))
    
    # 添加高斯噪声
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, augmented_image.shape).astype(np.float32)
        augmented_image = augmented_image + noise
    
    # 弹性变形
    if elastic_alpha > 0 and elastic_sigma > 0:
        # 生成位移场
        shape = augmented_image.shape[1:]  # H, W
        dx = cv2.GaussianBlur((np.random.rand(*shape) * 2 - 1), (0, 0), elastic_sigma) * elastic_alpha
        dy = cv2.GaussianBlur((np.random.rand(*shape) * 2 - 1), (0, 0), elastic_sigma) * elastic_alpha
        
        # 创建网格
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)
        
        # 应用弹性变形到每个波段
        for i in range(augmented_image.shape[0]):
            augmented_image[i] = cv2.remap(augmented_image[i], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        # 应用弹性变形到掩码
        augmented_mask[0] = cv2.remap(augmented_mask[0], map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
    
    return augmented_image, augmented_mask

def create_sentinel2_transform(normalize_method: str = 'sentinel'):
    """
    创建适用于Sentinel-2数据的变换组合。
    
    Args:
        normalize_method: 归一化方法
        
    Returns:
        变换函数
    """
    def transform(image):
        # 转换为Tensor
        if isinstance(image, np.ndarray):
            # 如果是numpy数组，转换为torch tensor
            image_tensor = torch.from_numpy(image).float()
            # 确保维度顺序为 (C, H, W)
            if image_tensor.dim() == 3 and image_tensor.shape[2] <= image_tensor.shape[0]:
                image_tensor = image_tensor.permute(2, 0, 1)
        else:
            # 如果已经是tensor，确保是float类型
            image_tensor = image.float()
        
        # 根据选择的方法添加归一化
        if normalize_method == 'sentinel':
            # 注意：Sentinel特定的归一化在load_sentinel2_image函数中已经实现
            # 这里可以添加其他需要的变换
            pass
        
        return image_tensor
    
    return transform

class Sentinel2WaterDataset(Dataset):
    """
    Sentinel-2水体分割数据集类。
    支持从指定目录加载Sentinel-2图像和对应的水体掩码。
    """
    def __init__(self, data_dir: str, split: str = 'train', bands: List[int] = None, 
                 augment: bool = False, normalize_method: str = 'sentinel', 
                 splits_dir: str = None, images_dir: str = None, masks_dir: str = None,
                 augmentation_config: Dict = None):
        """
        初始化数据集。
        
        Args:
            data_dir: 数据目录路径（如果images_dir和masks_dir未指定，则用于构建路径）
            split: 数据集分割，可选 'train', 'val', 'test'
            bands: 要使用的波段索引列表，如果为None则使用所有波段
            augment: 是否进行数据增强
            normalize_method: 归一化方法
            splits_dir: 数据集划分文件目录路径，如果为None则直接加载目录中的所有文件
            images_dir: 图像目录路径（可选，覆盖默认路径构建）
            masks_dir: 掩码目录路径（可选，覆盖默认路径构建）
            augmentation_config: 数据增强配置字典，包含各种增强参数
        """
        self.data_dir = data_dir
        self.split = split.lower()
        self.bands = bands
        self.augment = augment
        self.normalize_method = normalize_method
        self.splits_dir = splits_dir
        
        # 设置默认增强参数
        if augmentation_config is None:
            self.augmentation_config = {
                'rotation_range': 15,
                'width_shift_range': 0.1,
                'height_shift_range': 0.1,
                'scale_range': [0.9, 1.1],  # 修改为scale_range
                'horizontal_flip': True,
                'vertical_flip': True,
                'brightness_range': [0.9, 1.1],
                'contrast_range': [0.9, 1.1],
                'saturation_range': [0.9, 1.1],
                'hue_range': [-0.1, 0.1],
                'noise_std': 0.0,
                'elastic_alpha': 1000,
                'elastic_sigma': 8
            }
        else:
            self.augmentation_config = augmentation_config
        
        # 验证分割类型
        if self.split not in ['train', 'val', 'test']:
            raise ValueError(f"无效的分割类型: {self.split}，可选: train, val, test")
        
        # 获取图像和掩码文件路径
        if images_dir is not None:
            self.image_dir = images_dir
        else:
            self.image_dir = os.path.join(data_dir, self.split, 'images')
            
        if masks_dir is not None:
            self.mask_dir = masks_dir
        else:
            self.mask_dir = os.path.join(data_dir, self.split, 'masks')
        
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"图像目录不存在: {self.image_dir}")
        if not os.path.exists(self.mask_dir):
            raise FileNotFoundError(f"掩码目录不存在: {self.mask_dir}")
        
        # 获取图像文件列表
        if self.splits_dir is not None:
            # 从splits目录中的文件列表加载
            split_file = os.path.join(self.splits_dir, f"{self.split}.txt")
            if not os.path.exists(split_file):
                raise FileNotFoundError(f"找不到划分文件: {split_file}")
            
            # 读取划分文件中的文件名列表
            with open(split_file, 'r') as f:
                file_names = [line.strip() for line in f.readlines() if line.strip()]
            
            # 构建完整的图像文件列表
            self.image_files = []
            for name in file_names:
                # 首先尝试直接匹配（图像文件名）
                for ext in ['.tif', '.tiff', '.png']:
                    img_file = name + ext
                    if os.path.exists(os.path.join(self.image_dir, img_file)):
                        self.image_files.append(img_file)
                        break
                else:
                    # 如果直接匹配失败，尝试将'msk'替换为'img'
                    if '_msk_' in name:
                        img_name = name.replace('_msk_', '_img_')
                        for ext in ['.tif', '.tiff', '.png']:
                            img_file = img_name + ext
                            if os.path.exists(os.path.join(self.image_dir, img_file)):
                                self.image_files.append(img_file)
                                break
                        else:
                            raise FileNotFoundError(f"在 {self.image_dir} 中找不到图像文件: {name} 或 {img_name}")
                    else:
                        raise FileNotFoundError(f"在 {self.image_dir} 中找不到图像文件: {name}")
        else:
            # 获取所有图像文件（原始行为）
            self.image_files = sorted([f for f in os.listdir(self.image_dir) 
                                      if f.endswith('.tif') or f.endswith('.tiff') or f.endswith('.png')])
        
        # 检查图像和掩码文件是否匹配
        self.mask_files = []
        for img_file in self.image_files:
            # 假设掩码文件具有相同的基本名称，但可能有不同的扩展名
            base_name = os.path.splitext(img_file)[0]
            mask_file = None
            
            # 首先尝试直接匹配（相同文件名）
            for ext in ['.tif', '.tiff', '.png', '.jpg', '.jpeg']:
                potential_mask = os.path.join(self.mask_dir, base_name + ext)
                if os.path.exists(potential_mask):
                    mask_file = base_name + ext
                    break
            
            # 如果直接匹配失败，尝试将'img'替换为'msk'
            if mask_file is None and '_img_' in base_name:
                mask_base_name = base_name.replace('_img_', '_msk_')
                for ext in ['.tif', '.tiff', '.png', '.jpg', '.jpeg']:
                    potential_mask = os.path.join(self.mask_dir, mask_base_name + ext)
                    if os.path.exists(potential_mask):
                        mask_file = mask_base_name + ext
                        break
            
            # 如果仍然失败，检查是否是从splits文件加载的掩码文件名
            if mask_file is None and self.splits_dir is not None:
                # 如果图像文件名是从掩码文件名转换而来的，那么直接使用原始掩码文件名
                img_base_name = os.path.splitext(img_file)[0]
                if '_img_' in img_base_name:
                    original_mask_name = img_base_name.replace('_img_', '_msk_')
                    for ext in ['.tif', '.tiff', '.png', '.jpg', '.jpeg']:
                        potential_mask = os.path.join(self.mask_dir, original_mask_name + ext)
                        if os.path.exists(potential_mask):
                            mask_file = original_mask_name + ext
                            break
            
            if mask_file is None:
                raise FileNotFoundError(f"找不到与图像 '{img_file}' 匹配的掩码文件")
            
            self.mask_files.append(mask_file)
        
        if len(self.image_files) == 0:
            raise ValueError(f"在 {self.image_dir} 中未找到图像文件")
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本，可能包含高级增强。
        
        Args:
            idx: 样本索引
            
        Returns:
            包含图像和掩码的字典
        """
        # 获取文件路径
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        
        # 加载图像和掩码
        image, _ = load_sentinel2_image(image_path, self.bands)
        mask = load_mask(mask_path)
        
        # 数据增强（在归一化之前进行）
        if self.augment and self.split == 'train':
            image, mask = augment_image_and_mask(
                image, mask, 
                rotation_range=self.augmentation_config.get('rotation_range', 15),
                width_shift_range=self.augmentation_config.get('width_shift_range', 0.1),
                height_shift_range=self.augmentation_config.get('height_shift_range', 0.1),
                scale_range=self.augmentation_config.get('scale_range', [0.9, 1.1]),
                horizontal_flip=self.augmentation_config.get('horizontal_flip', True),
                vertical_flip=self.augmentation_config.get('vertical_flip', True),
                brightness_range=self.augmentation_config.get('brightness_range', [0.9, 1.1]),
                contrast_range=self.augmentation_config.get('contrast_range', [0.9, 1.1]),
                saturation_range=self.augmentation_config.get('saturation_range', [0.9, 1.1]),
                hue_range=self.augmentation_config.get('hue_range', [-0.1, 0.1]),
                noise_std=self.augmentation_config.get('noise_std', 0.0),
                elastic_alpha=self.augmentation_config.get('elastic_alpha', 1000),
                elastic_sigma=self.augmentation_config.get('elastic_sigma', 8)
            )
        
        # 归一化图像（在增强之后进行）
        image = normalize_image(image, method=self.normalize_method)
        
        # 转换为Tensor并立即释放numpy数组内存
        image_tensor = torch.from_numpy(image).float()
        mask_tensor = torch.from_numpy(mask).float()
        
        # 显式删除numpy数组以释放内存
        del image
        del mask
        
        # 仅在训练模式下或每10个样本进行一次垃圾回收，减少频繁调用
        if self.split == 'train' or idx % 10 == 0:
            import gc
            gc.collect()
        
        # 返回样本
        return {
            'image': image_tensor,
            'mask': mask_tensor,
            'filename': self.image_files[idx]
        }

class SingleImageDataset(Dataset):
    """
    单张影像数据集类，用于评估单张影像。
    直接从指定路径加载单张影像和对应的掩膜。
    确保与Sentinel2WaterDataset使用相同的数据处理逻辑。
    """
    def __init__(self, image_path: str, mask_path: str, bands: List[int] = None, 
                 normalize_method: str = 'sentinel'):
        """
        初始化单张影像数据集。
        
        Args:
            image_path: 影像文件路径
            mask_path: 掩膜文件路径
            bands: 要使用的波段索引列表，如果为None则使用默认波段[1, 2, 3, 4, 5, 6]
            normalize_method: 归一化方法，必须与Sentinel2WaterDataset保持一致
        """
        self.image_path = image_path
        self.mask_path = mask_path
        self.normalize_method = normalize_method
        self.image_filename = os.path.basename(image_path)
        
        # 如果没有指定波段，使用与Sentinel2WaterDataset相同的默认波段
        if bands is None:
            self.bands = [1, 2, 3, 4, 5, 6]  # 与Sentinel2WaterDataset默认值一致
        else:
            self.bands = bands
        
        # 检查文件是否存在，提供更清晰的错误信息
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}。请检查文件路径是否正确。")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"掩膜文件不存在: {mask_path}。请检查文件路径是否正确。")
        
        # 验证归一化方法
        valid_normalize_methods = ['minmax', 'zscore', 'sentinel']
        if normalize_method not in valid_normalize_methods:
            raise ValueError(f"无效的归一化方法: {normalize_method}。可选: {valid_normalize_methods}")
    
    def __len__(self) -> int:
        """返回数据集大小（始终为1）"""
        return 1
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本，确保与Sentinel2WaterDataset返回的数据格式一致。
        
        Args:
            idx: 样本索引（必须为0）
            
        Returns:
            包含图像和掩码的字典，格式与Sentinel2WaterDataset一致
        """
        if idx != 0:
            raise IndexError("单张影像数据集只有一个样本，索引必须为0")
        
        try:
            # 加载图像和掩码，使用与Sentinel2WaterDataset相同的函数
            image, _ = load_sentinel2_image(self.image_path, self.bands)
            mask = load_mask(self.mask_path)
            
            # 应用归一化，使用与Sentinel2WaterDataset相同的归一化方法
            image = normalize_image(image, method=self.normalize_method)
            
            # 转换为Tensor并立即释放numpy数组内存
            image_tensor = torch.from_numpy(image).float()
            mask_tensor = torch.from_numpy(mask).float()
            
            # 确保维度顺序正确 (C, H, W) for image and (1, H, W) for mask
            if image_tensor.dim() == 3 and image_tensor.shape[2] <= image_tensor.shape[0]:
                image_tensor = image_tensor.permute(2, 0, 1)
            
            if mask_tensor.dim() == 2:
                mask_tensor = mask_tensor.unsqueeze(0)
            
            # 显式删除numpy数组以释放内存
            del image
            del mask
            
            # 返回与Sentinel2WaterDataset相同格式的数据
            return {
                'image': image_tensor,
                'mask': mask_tensor,
                'filename': self.image_filename
            }
        except Exception as e:
            logger.error(f"加载单张影像数据时出错: {e}")
            raise

class Sentinel2WaterDatasetWithAdvancedAug(Sentinel2WaterDataset):
    """
    带有高级数据增强的Sentinel-2水体分割数据集类。
    支持MixUp和CutMix等高级增强技术。
    """
    def __init__(self, data_dir: str, split: str = 'train', bands: List[int] = None, 
                 augment: bool = False, normalize_method: str = 'sentinel', 
                 splits_dir: str = None, images_dir: str = None, masks_dir: str = None,
                 augmentation_config: Dict = None,
                 mixup_alpha=1.0, cutmix_alpha=1.0, mixup_prob=0.5, cutmix_prob=0.5):
        """
        初始化数据集。
        
        Args:
            data_dir: 数据目录路径（如果images_dir和masks_dir未指定，则用于构建路径）
            split: 数据集分割，可选 'train', 'val', 'test'
            bands: 要使用的波段索引列表，如果为None则使用所有波段
            augment: 是否进行数据增强
            normalize_method: 归一化方法
            splits_dir: 数据集划分文件目录路径，如果为None则直接加载目录中的所有文件
            images_dir: 图像目录路径（可选，覆盖默认路径构建）
            masks_dir: 掩码目录路径（可选，覆盖默认路径构建）
            augmentation_config: 数据增强配置字典，包含各种增强参数
            mixup_alpha: MixUp增强的alpha参数
            cutmix_alpha: CutMix增强的alpha参数
            mixup_prob: 应用MixUp增强的概率
            cutmix_prob: 应用CutMix增强的概率
        """
        super().__init__(data_dir, split, bands, augment, normalize_method, splits_dir, images_dir, masks_dir, augmentation_config)
        
        # 高级增强参数
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mixup_prob = mixup_prob
        self.cutmix_prob = cutmix_prob
        
        # 尝试导入增强工具
        try:
            from utils.augmentation_utils import apply_random_augmentation
            self.apply_random_augmentation = apply_random_augmentation
            self.has_advanced_aug = True
        except ImportError:
            self.has_advanced_aug = False
            print("警告: 无法导入高级增强工具，将使用基本增强")
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本，可能包含高级增强。
        
        Args:
            idx: 样本索引
            
        Returns:
            包含图像和掩码的字典
        """
        # 调用父类的__getitem__方法获取基本处理后的数据
        # 这确保了数据增强和归一化按照正确的顺序执行
        item = super().__getitem__(idx)
        
        # 如果需要高级增强且是训练模式，可以在批量级别应用
        # 注意：MixUp和CutMix等高级增强通常在批量级别应用，而不是单个样本级别
        # 这些增强在apply_batch_augmentation方法中处理
        
        return item
    
    def apply_batch_augmentation(self, images, masks):
        """
        对批量数据应用高级增强（MixUp/CutMix）。
        注意：此方法应在DataLoader之后调用，在训练循环中。
        
        Args:
            images: 图像批量 (batch_size, C, H, W)
            masks: 掩码批量 (batch_size, 1, H, W)
            
        Returns:
            augmented_images: 增强后的图像
            augmented_masks: 增强后的掩码
            aug_info: 增强信息字典，包含增强类型和lambda值
        """
        if not (self.augment and self.split == 'train' and self.has_advanced_aug):
            return images, masks, {'type': 'none', 'lambda': 1.0}
        
        # 应用随机增强
        augmented_images, augmented_masks, aug_type, lam = self.apply_random_augmentation(
            images, masks, 
            mixup_prob=self.mixup_prob, 
            cutmix_prob=self.cutmix_prob,
            mixup_alpha=self.mixup_alpha, 
            cutmix_alpha=self.cutmix_alpha
        )
        
        return augmented_images, augmented_masks, {'type': aug_type, 'lambda': lam}

def collate_fn(batch):
    """
    数据加载器的批处理函数。
    优化版本，减少内存使用。
    
    Args:
        batch: 批量数据
        
    Returns:
        处理后的批量数据
    """
    # 使用生成器表达式减少内存使用
    images = list(item['image'] for item in batch)
    masks = list(item['mask'] for item in batch)
    filenames = list(item['filename'] for item in batch)
    
    # 堆叠张量
    stacked_images = torch.stack(images)
    stacked_masks = torch.stack(masks)
    
    # 立即清理中间变量以释放内存
    del images
    del masks
    
    # 强制垃圾回收以释放内存
    import gc
    gc.collect()
    
    return {
        'image': stacked_images,
        'mask': stacked_masks,
        'filename': filenames
    }

def create_data_loader(dataset: Dataset, batch_size: int = 8, shuffle: bool = True, 
                      num_workers: int = 4, pin_memory: bool = True, persistent_workers: bool = False,
                      memory_optimized: bool = False, prefetch_factor: int = None) -> DataLoader:
    """
    创建数据加载器。
    
    Args:
        dataset: 数据集实例
        batch_size: 批量大小
        shuffle: 是否打乱数据
        num_workers: 工作进程数
        pin_memory: 是否锁定内存
        persistent_workers: 是否使用持久化工作进程
        memory_optimized: 是否启用内存优化模式
        prefetch_factor: 预取因子（num_workers>0时生效）
        
    Returns:
        数据加载器实例
    """
    # 如果启用内存优化模式，调整参数以减少内存使用
    if memory_optimized:
        # 减少批大小和工作进程数
        batch_size = min(batch_size, 2)
        num_workers = min(num_workers, 1)
        # 禁用pin_memory以减少内存压力
        pin_memory = False
        # 禁用persistent_workers以减少内存占用
        persistent_workers = False
    
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        persistent_workers=persistent_workers
    )
    if num_workers > 0 and prefetch_factor is not None:
        loader_kwargs['prefetch_factor'] = int(prefetch_factor)

    return DataLoader(**loader_kwargs)

def calculate_dataset_statistics(dataset: Dataset, num_samples: int = 100) -> Dict[str, np.ndarray]:
    """
    计算数据集的统计信息（均值、标准差等）。
    
    Args:
        dataset: 数据集实例
        num_samples: 用于计算统计信息的样本数量
        
    Returns:
        包含统计信息的字典
    """
    # 随机选择样本
    sample_indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    # 初始化统计变量
    means = []
    stds = []
    mins = []
    maxs = []
    
    # 遍历选定的样本
    for idx in sample_indices:
        sample = dataset[idx]
        image = sample['image'].numpy()
        
        # 计算每个波段的统计信息
        band_means = np.mean(image, axis=(1, 2))
        band_stds = np.std(image, axis=(1, 2))
        band_mins = np.min(image, axis=(1, 2))
        band_maxs = np.max(image, axis=(1, 2))
        
        means.append(band_means)
        stds.append(band_stds)
        mins.append(band_mins)
        maxs.append(band_maxs)
    
    # 计算总体统计信息
    results = {
        'mean': np.mean(means, axis=0),
        'std': np.mean(stds, axis=0),
        'min': np.min(mins, axis=0),
        'max': np.max(maxs, axis=0)
    }
    
    return results

def save_predictions(predictions: torch.Tensor, filenames: List[str], output_dir: str, threshold: float = 0.5, 
                    save_probabilities: bool = True, is_probabilities: bool = False, 
                    postprocessing_config: dict = None, reference_dir: Optional[str] = None) -> None:
    """
    保存模型预测结果到文件。
    
    Args:
        predictions: 预测结果张量（概率值或logits）
        filenames: 文件名列表
        output_dir: 输出目录
        threshold: 二值化阈值
        save_probabilities: 是否保存概率图
        is_probabilities: 输入是否已经是概率值（如果是False，则认为是logits，需要应用sigmoid）
        postprocessing_config: 后处理配置字典，包含后处理参数
        reference_dir: 参考影像目录（用于保留GeoTIFF的空间参考）
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    reference_dir = os.path.abspath(reference_dir) if reference_dir else None

    def _load_reference_profile(filename: str) -> Optional[dict]:
        if not reference_dir:
            return None
        ref_path = os.path.join(reference_dir, filename)
        if not os.path.exists(ref_path):
            return None
        try:
            with rasterio.open(ref_path) as src:
                return src.profile.copy()
        except Exception:
            return None

    # 导入后处理函数
    from utils.postprocessing_utils import apply_postprocessing_pipeline
    
    # 保存概率图（如果需要）
    if save_probabilities:
        prob_dir = os.path.join(output_dir, 'probabilities')
        os.makedirs(prob_dir, exist_ok=True)
        
        for i, pred in enumerate(predictions):
            # 获取概率值
            if pred.shape[0] == 1:
                # 二值分割
                if is_probabilities:
                    # 输入已经是概率值，直接使用
                    prob = pred.cpu().numpy()
                else:
                    # 输入是logits，需要应用sigmoid
                    prob = torch.sigmoid(pred).cpu().numpy()
            else:
                # 多类别分割
                if is_probabilities:
                    # 输入已经是概率值，直接使用
                    prob = pred.cpu().numpy()
                else:
                    # 输入是logits，需要应用softmax
                    prob = torch.softmax(pred, dim=0).cpu().numpy()
            
            # 保存概率图
            filename = filenames[i]
            output_path = os.path.join(prob_dir, f"{filename}_prob.tif")
            
            try:
                # 保存为16位浮点数以保留概率值
                save_image = (prob[0] * 65535).astype(np.uint16)
                profile = _load_reference_profile(filename)
                if profile:
                    profile.update(count=1, dtype=rasterio.uint16)
                    with rasterio.open(output_path, 'w', **profile) as dst:
                        dst.write(save_image, 1)
                else:
                    cv2.imwrite(output_path, save_image)
            except Exception as e:
                print(f"保存概率图失败: {output_path}. 错误: {str(e)}")
                # 尝试使用更简单的方式保存
                try:
                    cv2.imwrite(os.path.join(prob_dir, f"prob_{i}.png"), (prob[0] * 255).astype(np.uint8))
                except:
                    pass
    
    # 保存二值化预测结果
    pred_dir = os.path.join(output_dir, 'predictions')
    os.makedirs(pred_dir, exist_ok=True)
    
    for i, pred in enumerate(predictions):
        # 将预测转换为二值掩码
        if pred.shape[0] == 1:
            # 二值分割
            if is_probabilities:
                # 输入已经是概率值，直接应用阈值
                binary_pred = (pred > threshold).float().cpu().numpy()
            else:
                # 输入是logits，需要先应用sigmoid再应用阈值
                binary_pred = (torch.sigmoid(pred) > threshold).float().cpu().numpy()
        else:
            # 多类别分割
            if is_probabilities:
                # 输入已经是概率值，直接取argmax
                binary_pred = torch.argmax(pred, dim=0, keepdim=True).float().cpu().numpy()
            else:
                # 输入是logits，需要先应用softmax再取argmax
                binary_pred = torch.argmax(torch.softmax(pred, dim=0), dim=0, keepdim=True).float().cpu().numpy()
        
        # 应用后处理（仅对二值分割有效）
        if postprocessing_config and pred.shape[0] == 1:
            try:
                # 获取概率值
                if is_probabilities:
                    prob_tensor = pred
                else:
                    prob_tensor = torch.sigmoid(pred)
                
                # 从概率值计算二值预测结果
                binary_pred_tensor = (prob_tensor > threshold).float()
                
                # 应用后处理
                processed_pred = apply_postprocessing_pipeline(
                    prob_tensor, 
                    binary_pred_tensor,
                    gaussian_sigma=postprocessing_config.get('gaussian_sigma', 1.0),
                    median_kernel_size=postprocessing_config.get('median_kernel_size', 3),
                    morph_close_kernel_size=postprocessing_config.get('morph_close_kernel_size', 3),
                    morph_open_kernel_size=postprocessing_config.get('morph_open_kernel_size', 0),
                    min_object_size=postprocessing_config.get('min_object_size', 50),
                    hole_area_threshold=postprocessing_config.get('hole_area_threshold', 30),
                    adaptive_threshold=postprocessing_config.get('adaptive_threshold', False)
                )
                # 确保后处理结果类型和形状正确
                binary_pred = processed_pred.astype(np.float32, copy=False)
                # 处理[1,1,H,W]维度，确保保存的是2D图像
                if binary_pred.ndim == 4:
                    binary_pred = binary_pred[0,0]  # 从[1,1,H,W]到[H,W]
                elif binary_pred.ndim == 3:
                    binary_pred = binary_pred[0]    # 从[1,H,W]到[H,W]
            except Exception as e:
                print(f"应用后处理失败: {str(e)}")
                # 如果后处理失败，使用原始二值预测
                if is_probabilities:
                    binary_pred = (pred > threshold).float().cpu().numpy()
                else:
                    binary_pred = (torch.sigmoid(pred) > threshold).float().cpu().numpy()
        
        # 保存为图像
        filename = filenames[i]
        output_path = os.path.join(pred_dir, filename)
        
        # 使用rasterio保存为GeoTIFF格式
        try:
            # 获取原始图像的元数据（如果可用）
            # 这里简化处理，直接保存为8位图像
            # 确保binary_pred是2D的[H,W]格式
            if binary_pred.ndim == 3:
                save_image = (binary_pred[0] * 255).astype(np.uint8)
            else:
                save_image = (binary_pred * 255).astype(np.uint8)
            profile = _load_reference_profile(filename)
            if profile:
                profile.update(count=1, dtype=rasterio.uint8)
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(save_image, 1)
            else:
                cv2.imwrite(output_path, save_image)  # 使用OpenCV保存
        except Exception as e:
            print(f"保存预测结果失败: {output_path}. 错误: {str(e)}")
            # 尝试使用更简单的方式保存
            try:
                # 确保binary_pred是2D的[H,W]格式
                if binary_pred.ndim == 3:
                    save_image = (binary_pred[0] * 255).astype(np.uint8)
                else:
                    save_image = (binary_pred * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(pred_dir, f"pred_{i}.png"), save_image)
            except:
                pass
