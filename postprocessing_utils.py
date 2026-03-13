#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
后处理工具函数
"""

import numpy as np
import cv2
from skimage import morphology
from skimage.filters import threshold_otsu


def apply_postprocessing_pipeline(probability_prediction, binary_prediction,
                                median_kernel_size=0, gaussian_sigma=0.0,
                                morph_close_kernel_size=0, morph_open_kernel_size=0,
                                min_object_size=0, hole_area_threshold=0,
                                adaptive_threshold=False, threshold=0.5):
    """
    应用后处理管道到预测结果
    
    Args:
        probability_prediction: 概率预测图 [H, W] 或 [1, 1, H, W]
        binary_prediction: 二值预测结果 [H, W] 或 [1, 1, H, W]
        median_kernel_size: 中值滤波核大小，0表示不应用
        gaussian_sigma: 高斯滤波标准差，0表示不应用
        morph_close_kernel_size: 形态学闭运算核大小，0表示不应用
        morph_open_kernel_size: 形态学开运算核大小，0表示不应用
        min_object_size: 最小对象大小（像素数），0表示不应用
        hole_area_threshold: 小孔面积阈值，0表示不应用
        adaptive_threshold: 是否使用自适应阈值（Otsu）
        threshold: 二值化阈值
    
    Returns:
        processed_prediction: 处理后的二值预测结果 [1, 1, H, W] numpy数组
    """
    # 确保输入是numpy数组并去除多余的维度
    # 处理PyTorch张量
    if hasattr(probability_prediction, 'detach') and hasattr(probability_prediction, 'cpu'):
        probability_prediction = probability_prediction.detach().cpu().numpy()
    if hasattr(binary_prediction, 'detach') and hasattr(binary_prediction, 'cpu'):
        binary_prediction = binary_prediction.detach().cpu().numpy()
    
    # 确保是numpy数组
    if not isinstance(probability_prediction, np.ndarray):
        probability_prediction = np.array(probability_prediction)
    if not isinstance(binary_prediction, np.ndarray):
        binary_prediction = np.array(binary_prediction)
    
    # 统一处理输入维度，确保是[H, W]格式
    probability_prediction = np.squeeze(probability_prediction)
    binary_prediction = np.squeeze(binary_prediction)
    
    # 应用自适应阈值（如果需要）
    if adaptive_threshold:
        threshold = threshold_otsu(probability_prediction)
        binary_prediction = (probability_prediction > threshold).astype(np.float32)
    
    # 应用高斯滤波（如果需要）
    if gaussian_sigma > 0:
        probability_prediction = cv2.GaussianBlur(probability_prediction, (0, 0), gaussian_sigma)
    
    # 应用中值滤波（如果需要）
    if median_kernel_size > 0:
        binary_prediction = cv2.medianBlur((binary_prediction * 255).astype(np.uint8), median_kernel_size).astype(np.float32) / 255.0
    
    # 应用形态学闭运算（先膨胀后腐蚀，填充小孔）
    if morph_close_kernel_size > 0:
        kernel = morphology.disk(morph_close_kernel_size)
        binary_prediction = morphology.binary_closing(binary_prediction, kernel).astype(np.float32)
    
    # 应用形态学开运算（先腐蚀后膨胀，去除小物体）
    if morph_open_kernel_size > 0:
        kernel = morphology.disk(morph_open_kernel_size)
        binary_prediction = morphology.binary_opening(binary_prediction, kernel).astype(np.float32)
    
    # 移除小物体（如果需要）
    if min_object_size > 0:
        # 标记连通区域
        labeled = morphology.label(binary_prediction > 0)
        # 计算每个区域的大小
        regions = morphology.regionprops(labeled)
        # 创建掩码，只保留大于最小尺寸的区域
        mask = np.zeros_like(binary_prediction, dtype=bool)
        for region in regions:
            if region.area >= min_object_size:
                mask = mask | (labeled == region.label)
        binary_prediction = mask.astype(np.float32)
    
    # 填充小孔（如果需要）
    if hole_area_threshold > 0:
        # 反转图像，将孔变成物体
        inverted = 1 - binary_prediction
        # 标记连通区域
        labeled = morphology.label(inverted > 0)
        # 计算每个区域的大小
        regions = morphology.regionprops(labeled)
        # 创建掩码，只保留小于阈值的区域（即孔）
        hole_mask = np.zeros_like(inverted, dtype=bool)
        for region in regions:
            if region.area < hole_area_threshold:
                hole_mask = hole_mask | (labeled == region.label)
        # 填充这些孔
        binary_prediction[hole_mask] = 1.0
    
    # 恢复维度为[1, 1, H, W]以便与模型输出格式一致
    processed_prediction = np.expand_dims(np.expand_dims(binary_prediction, 0), 0)
    
    return processed_prediction