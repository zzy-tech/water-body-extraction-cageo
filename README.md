# Sentinel-2 Water Segmentation

该仓库包含用于 Sentinel-2 遥感影像水体分割的模型实现与训练/评估/推理脚本（AER U-Net、U-Net、DeepLabV3+ 及轻量化变体），以及集成与后处理工具。

## 环境
- Python 3.8+
- 主要依赖：PyTorch、torchvision、numpy、opencv、rasterio（详见 `requirements.txt`）

安装依赖：
```bash
pip install -r requirements.txt
```

## 数据准备
默认数据根目录为 `datasets/`，目录结构建议如下（可在配置或命令行参数中修改）：
```
datasets/
  train/
    images/
    masks/
  val/
    images/
    masks/
  test/
    images/
    masks/
```

如需划分数据集，可使用：
```bash
python split_dataset.py --data_dir datasets --splits_dir splits
```

## 训练
使用 YAML 配置文件进行训练：
```bash
python train.py --config config/aer_unet.yaml --model aer_unet
```

也可直接通过命令行覆盖参数：
```bash
python train.py --model aer_unet --data_dir datasets --batch_size 8 --epochs 50
```

## 评估
```bash
python evaluate.py --config config/evaluate_config.yaml --model aer_unet
```

## 推理
对单张或批量影像预测：
```bash
python predict.py --model aer_unet --data_dir datasets --weights checkpoints/aer_unet_best.pth
```

集成预测：
```bash
python predict_ensemble.py --config config/ensemble_config.yaml
```

## 配置说明
- 训练/评估/集成配置在 `config/` 下
- 全局默认配置在 `config.py`

## 主要脚本
- `train.py` 训练
- `evaluate.py` 评估
- `predict.py` 推理
- `predict_ensemble.py` 集成推理
- `utils/` 数据处理、损失函数、指标、集成策略
- `models/` 模型结构

## 项目结构

部分文件夹被拆分为单独文件。你可以通过以下两种方式组织文件：

**方式一：使用压缩包（推荐）**
- 下载 `models.zip`、`config.zip`、`utils.zip` 压缩包
- 直接在项目根目录解压这些压缩包，会自动创建正确的目录结构

**方式二：手动组织文件**
请按照以下结构组织文件：

```
sentinel2_water_segmentation/
├── 训练脚本
│   ├── train.py             # 主训练脚本
│   ├── evaluate.py          # 评估脚本
│   ├── predict.py           # 推理脚本
│   ├── predict_ensemble.py  # 集成推理脚本
│   └── split_dataset.py     # 数据集划分脚本
├── 配置文件
│   ├── config.py            # 全局配置
│   ├── aer_unet.yaml        # AER-UNet 配置
│   ├── unet.yaml            # U-Net 配置
│   ├── deeplabv3_plus.yaml  # DeepLabV3+ 配置
│   ├── ultra_lightweight_deeplabv3_plus.yaml  # 超轻量级 DeepLabV3+配置
│   └── improved_ensemble_config.yaml  # 集成预测配置
├── models/                  # 模型结构目录（或解压 models.zip）
│   ├── __init__.py
│   ├── aer_unet.py          # AER-UNet 模型
│   ├── unet_model.py        # U-Net 模型主体
│   ├── unet_parts.py        # U-Net 组件
│   ├── deeplabv3_plus.py    # DeepLabV3+ 模型
│   └── ultra_lightweight_deeplabv3_plus.py  # 超轻量级 DeepLabV3+模型
├── utils/                   # 工具函数目录（或解压 utils.zip）
│   ├── __init__.py
│   ├── data_utils.py        # 数据处理工具
│   ├── losses.py            # 损失函数
│   ├── metrics.py           # 评估指标
│   ├── ensemble_utils.py    # 集成策略
│   ├── postprocessing_utils.py  # 后处理工具
│   ├── crf_utils.py         # CRF 工具
│   ├── augmentation_utils.py # 数据增强工具
│   └── attention_modules.py # 注意力模块
├── 集成学习脚本
│   ├── performance_weighted_ensemble.py      # 性能加权集成
│   └── improved_performance_weighted_ensemble.py  # 改进的性能加权集成
├── 其他文件
│   ├── requirements.txt     # 依赖项
│   ├── README.md            # 项目说明
│   ├── .gitignore           # Git 忽略文件
│   └── LICENSE              # 许可证
├── 压缩包（快速设置）
│   ├── models.zip           # models 目录压缩包
│   ├── config.zip           # config 目录压缩包
│   └── utils.zip            # utils 目录压缩包
├── checkpoints/             # 模型检查点目录（自动创建）
├── datasets/                # 数据集目录（需自行准备）
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   ├── val/
│   │   ├── images/
│   │   └── masks/
│   └── test/
│       ├── images/
│       └── masks/
└── splits/                  # 数据集划分文件目录（自动创建）
    ├── train.txt
    ├── val.txt
    └── test.txt
```

**文件组织说明：**

1. **模型文件**：
   - 将所有模型相关的 `.py` 文件放入 `models/` 目录
   - 确保 `models/__init__.py` 存在

2. **工具文件**：
   - 将所有工具函数相关的 `.py` 文件放入 `utils/` 目录
   - 确保 `utils/__init__.py` 存在

3. **配置文件**：
   - 将所有 YAML 配置文件放入 `config/` 目录

4. **数据集**：
   - 按照上述结构创建 `datasets/` 目录
   - 确保每个子目录包含 `images/` 和 `masks/` 文件夹

5. **检查点**：
   - `checkpoints/` 目录会在训练时自动创建
   - 每个模型会在其中创建独立的子目录

## 结果与日志
训练日志默认输出 `training.log`，评估/预测日志与结果输出目录可在配置文件中修改。

## 引用
若使用本代码，请在论文中引用对应工作（可添加 `CITATION.cff` 进一步规范化）。
