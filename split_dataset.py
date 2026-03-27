import os
import argparse
import random
import shutil
from pathlib import Path

def create_splits(images_dir, masks_dir, output_dir, val_ratio=0.12, test_ratio=0.0, seed=42):
    """
    将数据集按比例划分为训练集、验证集和测试集
    
    参数:
        images_dir: 影像文件目录
        masks_dir: 掩膜文件目录
        output_dir: 输出目录
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
    """
    # 设置随机种子
    random.seed(seed)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有影像文件
    image_files = []
    for ext in ['*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg']:
        image_files.extend(Path(images_dir).glob(ext))
    
    # 提取文件名（不含扩展名）
    file_names = [f.stem for f in image_files]
    
    # 检查对应的掩膜文件是否存在
    valid_files = []
    for name in file_names:
        # 尝试多种可能的掩膜文件名格式
        possible_mask_names = []
        
        # 如果文件名中包含'img'，尝试替换为'msk'或'mask'
        if '_img_' in name:
            possible_mask_names.append(name.replace('_img_', '_msk_'))
            possible_mask_names.append(name.replace('_img_', '_mask_'))
        
        # 尝试在文件名末尾添加'_msk'或'_mask'
        possible_mask_names.append(f"{name}_msk")
        possible_mask_names.append(f"{name}_mask")
        
        # 尝试替换扩展名前的部分
        base_name = name
        possible_mask_names.append(f"{base_name}.msk")
        possible_mask_names.append(f"{base_name}.mask")
        
        # 尝试所有可能的扩展名
        mask_found = False
        for mask_name in possible_mask_names:
            for ext in ['.tif', '.tiff', '.png', '.jpg', '.jpeg']:
                mask_path = Path(masks_dir) / f"{mask_name}{ext}"
                if mask_path.exists():
                    valid_files.append(name)
                    mask_found = True
                    break
            if mask_found:
                break
        
        if not mask_found:
            print(f"警告: 找不到文件 {name} 对应的掩膜")
    
    if not valid_files:
        raise ValueError("没有找到有效的影像-掩膜对")
    
    # 随机打乱文件列表
    random.shuffle(valid_files)
    
    # 计算划分点
    val_size = int(len(valid_files) * val_ratio)
    test_size = int(len(valid_files) * test_ratio)
    train_size = len(valid_files) - val_size - test_size
    
    val_files = valid_files[:val_size]
    test_files = valid_files[val_size:val_size+test_size]
    train_files = valid_files[val_size+test_size:]
    
    # 创建splits目录
    splits_dir = Path(output_dir) / "splits"
    os.makedirs(splits_dir, exist_ok=True)
    
    # 写入训练集文件列表
    with open(splits_dir / "train.txt", "w") as f:
        for name in train_files:
            f.write(f"{name}\n")
    
    # 写入验证集文件列表
    with open(splits_dir / "val.txt", "w") as f:
        for name in val_files:
            f.write(f"{name}\n")
    
    # 写入测试集文件列表
    with open(splits_dir / "test.txt", "w") as f:
        for name in test_files:
            f.write(f"{name}\n")
    
    # 创建数据集目录结构
    datasets_dir = Path(output_dir) / "datasets"
    os.makedirs(datasets_dir / "train" / "images", exist_ok=True)
    os.makedirs(datasets_dir / "train" / "masks", exist_ok=True)
    os.makedirs(datasets_dir / "val" / "images", exist_ok=True)
    os.makedirs(datasets_dir / "val" / "masks", exist_ok=True)
    os.makedirs(datasets_dir / "test" / "images", exist_ok=True)
    os.makedirs(datasets_dir / "test" / "masks", exist_ok=True)
    
    # 复制训练集文件
    print(f"复制训练集文件 ({len(train_files)} 个)...")
    for name in train_files:
        # 查找影像文件
        for ext in ['.tif', '.tiff', '.png', '.jpg', '.jpeg']:
            src_path = Path(images_dir) / f"{name}{ext}"
            if src_path.exists():
                shutil.copy2(src_path, datasets_dir / "train" / "images" / f"{name}{ext}")
                break
        
        # 查找掩膜文件
        mask_found = False
        # 尝试多种可能的掩膜文件名格式
        possible_mask_names = []
        
        # 如果文件名中包含'img'，尝试替换为'msk'或'mask'
        if '_img_' in name:
            possible_mask_names.append(name.replace('_img_', '_msk_'))
            possible_mask_names.append(name.replace('_img_', '_mask_'))
        
        # 尝试在文件名末尾添加'_msk'或'_mask'
        possible_mask_names.append(f"{name}_msk")
        possible_mask_names.append(f"{name}_mask")
        
        # 尝试替换扩展名前的部分
        base_name = name
        possible_mask_names.append(f"{base_name}.msk")
        possible_mask_names.append(f"{base_name}.mask")
        
        # 尝试所有可能的扩展名
        for mask_name in possible_mask_names:
            for ext in ['.tif', '.tiff', '.png', '.jpg', '.jpeg']:
                src_path = Path(masks_dir) / f"{mask_name}{ext}"
                if src_path.exists():
                    shutil.copy2(src_path, datasets_dir / "train" / "masks" / f"{name}{ext}")
                    mask_found = True
                    break
            if mask_found:
                break
    
    # 复制验证集文件
    print(f"复制验证集文件 ({len(val_files)} 个)...")
    for name in val_files:
        # 查找影像文件
        for ext in ['.tif', '.tiff', '.png', '.jpg', '.jpeg']:
            src_path = Path(images_dir) / f"{name}{ext}"
            if src_path.exists():
                shutil.copy2(src_path, datasets_dir / "val" / "images" / f"{name}{ext}")
                break
        
        # 查找掩膜文件
        mask_found = False
        # 尝试多种可能的掩膜文件名格式
        possible_mask_names = []
        
        # 如果文件名中包含'img'，尝试替换为'msk'或'mask'
        if '_img_' in name:
            possible_mask_names.append(name.replace('_img_', '_msk_'))
            possible_mask_names.append(name.replace('_img_', '_mask_'))
        
        # 尝试在文件名末尾添加'_msk'或'_mask'
        possible_mask_names.append(f"{name}_msk")
        possible_mask_names.append(f"{name}_mask")
        
        # 尝试替换扩展名前的部分
        base_name = name
        possible_mask_names.append(f"{base_name}.msk")
        possible_mask_names.append(f"{base_name}.mask")
        
        # 尝试所有可能的扩展名
        for mask_name in possible_mask_names:
            for ext in ['.tif', '.tiff', '.png', '.jpg', '.jpeg']:
                src_path = Path(masks_dir) / f"{mask_name}{ext}"
                if src_path.exists():
                    shutil.copy2(src_path, datasets_dir / "val" / "masks" / f"{name}{ext}")
                    mask_found = True
                    break
            if mask_found:
                break
    
    # 复制测试集文件
    print(f"复制测试集文件 ({len(test_files)} 个)...")
    for name in test_files:
        # 查找影像文件
        for ext in ['.tif', '.tiff', '.png', '.jpg', '.jpeg']:
            src_path = Path(images_dir) / f"{name}{ext}"
            if src_path.exists():
                shutil.copy2(src_path, datasets_dir / "test" / "images" / f"{name}{ext}")
                break
        
        # 查找掩膜文件
        mask_found = False
        # 尝试多种可能的掩膜文件名格式
        possible_mask_names = []
        
        # 如果文件名中包含'img'，尝试替换为'msk'或'mask'
        if '_img_' in name:
            possible_mask_names.append(name.replace('_img_', '_msk_'))
            possible_mask_names.append(name.replace('_img_', '_mask_'))
        
        # 尝试在文件名末尾添加'_msk'或'_mask'
        possible_mask_names.append(f"{name}_msk")
        possible_mask_names.append(f"{name}_mask")
        
        # 尝试替换扩展名前的部分
        base_name = name
        possible_mask_names.append(f"{base_name}.msk")
        possible_mask_names.append(f"{base_name}.mask")
        
        # 尝试所有可能的扩展名
        for mask_name in possible_mask_names:
            for ext in ['.tif', '.tiff', '.png', '.jpg', '.jpeg']:
                src_path = Path(masks_dir) / f"{mask_name}{ext}"
                if src_path.exists():
                    shutil.copy2(src_path, datasets_dir / "test" / "masks" / f"{name}{ext}")
                    mask_found = True
                    break
            if mask_found:
                break
    
    print(f"数据集划分完成!")
    print(f"训练集: {len(train_files)} 个样本")
    print(f"验证集: {len(val_files)} 个样本")
    print(f"测试集: {len(test_files)} 个样本")
    print(f"划分比例: 训练集 {100*train_size/len(valid_files):.1f}%, 验证集 {100*val_ratio:.1f}%, 测试集 {100*test_ratio:.1f}%")
    print(f"划分文件保存在: {splits_dir}")
    print(f"数据集保存在: {datasets_dir}")
    
    # 更新配置文件
    update_config(output_dir)

def update_config(output_dir):
    """更新配置文件以反映新的数据路径"""
    config_path = Path(output_dir) / "config.yaml"
    
    if not config_path.exists():
        print("警告: 未找到配置文件，跳过更新")
        return
    
    # 读取配置文件
    with open(config_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # 更新数据路径
    for i, line in enumerate(lines):
        if "images:" in line and "data/Images" in line:
            lines[i] = f"  images: datasets/train/images        # 存放 Sentinel-2 六通道 GeoTIFF（B2,B3,B4,B8,B11,B12）\n"
        elif "masks:" in line and "data/Masks" in line:
            lines[i] = f"  masks:  datasets/train/masks         # 存放与影像同名的二值掩膜 (0/1)\n"
        elif "train:" in line and "splits/train.txt" in line:
            lines[i] = f"    train: splits/train.txt  # 训练集划分文件\n"
        elif "val:" in line and "splits/val.txt" in line:
            lines[i] = f"    val: splits/val.txt      # 验证集划分文件\n"
    
    # 写入更新后的配置文件
    with open(config_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    
    print(f"配置文件已更新: {config_path}")

def main():
    parser = argparse.ArgumentParser(description="数据集划分工具")
    parser.add_argument("--images_dir", type=str, default="data/Images", help="影像文件目录 (默认: data/Images)")
    parser.add_argument("--masks_dir", type=str, default="data/Masks", help="掩膜文件目录 (默认: data/Masks)")
    parser.add_argument("--output_dir", type=str, default=".", help="输出目录")
    parser.add_argument("--val_ratio", type=float, default=0.12, help="验证集比例 (默认: 0.12)")
    parser.add_argument("--test_ratio", type=float, default=0.0, help="测试集比例 (默认: 0.0)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子 (默认: 42)")
    
    args = parser.parse_args()
    
    create_splits(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

if __name__ == "__main__":
    main()