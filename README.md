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

## 结果与日志
训练日志默认输出 `training.log`，评估/预测日志与结果输出目录可在配置文件中修改。

## 引用
若使用本代码，请在论文中引用对应工作（可添加 `CITATION.cff` 进一步规范化）。
