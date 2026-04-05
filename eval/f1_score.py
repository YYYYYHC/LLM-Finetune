#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F1-Score 计算脚本

用于计算生成场景（pred）与真实场景（gt）之间的 F1-score。

功能：
1. 读取 pred 和 gt 两个文件夹中的同名 JSON 文件
2. 对 pred 进行 0°、90°、180°、270° 四个旋转 + 镜像翻转（水平/垂直）的组合
3. 在 [-1, 1] 区间内以指定分辨率进行网格化均匀采样平移偏移，搜索最佳对齐
4. 逐物体计算匹配：在距离阈值内找到相同 class 的物体算成功
5. 计算 F1-score（取所有变换组合中最佳的结果）

使用方法：
    python f1_score.py --pred_dir <pred文件夹> --gt_dir <gt文件夹> [--threshold 0.5] [--output results.json]
"""

import os
import json
import copy
import math
import argparse
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # 如果没有安装 tqdm，提供一个简单的替代
    def tqdm(iterable, **kwargs):
        return iterable

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# 导入 params 准确性评估模块
from params_accuracy import evaluate_matched_objects_params, print_params_evaluation


@dataclass
class ObjectInfo:
    """物体信息"""
    obj_id: str
    obj_class: str
    location: List[float]  # [x, y, z]


def extract_objects_from_scene(scene_data: Dict) -> List[ObjectInfo]:
    """
    从场景数据中提取所有物体信息
    
    Args:
        scene_data: JSON 场景数据
        
    Returns:
        物体信息列表
    """
    objects = []
    
    for key, value in scene_data.items():
        # 跳过非物体键
        if key in ['blueprint', 'portal']:
            continue
            
        # 检查是否是物体（以 object_ 开头或包含 class 和 spatial）
        if isinstance(value, dict) and 'class' in value and 'spatial' in value:
            obj_class = value['class']
            location = value['spatial'].get('location', [0, 0, 0])
            
            objects.append(ObjectInfo(
                obj_id=key,
                obj_class=obj_class,
                location=list(location)
            ))
    
    return objects


def _get_full_object_from_scene(scene_data: Dict, obj_id: str) -> Optional[Dict]:
    """
    从场景中获取完整的物体数据（包括 class, spatial, params 等）
    
    Args:
        scene_data: 场景数据
        obj_id: 物体 ID
        
    Returns:
        完整的物体字典，如果未找到则返回 None
    """
    if obj_id in scene_data:
        return scene_data[obj_id]
    return None


def get_all_coordinates(scene_data: Dict) -> List[Tuple[float, float]]:
    """
    获取场景中所有坐标点（用于计算 min_x, min_y）
    
    Args:
        scene_data: JSON 场景数据
        
    Returns:
        所有 (x, y) 坐标点列表
    """
    coords = []
    
    # 从 blueprint 中获取房间坐标
    if 'blueprint' in scene_data and 'rooms' in scene_data['blueprint']:
        for room_name, room_data in scene_data['blueprint']['rooms'].items():
            if 'shape' in room_data and 'coordinates' in room_data['shape']:
                for polygon in room_data['shape']['coordinates']:
                    for point in polygon:
                        if len(point) >= 2:
                            coords.append((point[0], point[1]))
    
    # 从物体中获取坐标
    for key, value in scene_data.items():
        if key in ['blueprint', 'portal']:
            continue
        if isinstance(value, dict) and 'spatial' in value:
            loc = value['spatial'].get('location', [0, 0, 0])
            if len(loc) >= 2:
                coords.append((loc[0], loc[1]))
    
    # 从 portal 中获取坐标
    if 'portal' in scene_data:
        for portal in scene_data['portal']:
            if 'seg' in portal:
                for point in portal['seg']:
                    if len(point) >= 2:
                        coords.append((point[0], point[1]))
    
    return coords


def rotate_point_2d(x: float, y: float, angle_rad: float, cx: float = 0, cy: float = 0) -> Tuple[float, float]:
    """
    绕指定中心点旋转2D点
    
    Args:
        x, y: 原始坐标
        angle_rad: 旋转角度（弧度）
        cx, cy: 旋转中心
        
    Returns:
        旋转后的坐标 (new_x, new_y)
    """
    # 平移到原点
    x -= cx
    y -= cy
    
    # 旋转
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    new_x = x * cos_a - y * sin_a
    new_y = x * sin_a + y * cos_a
    
    # 平移回去
    new_x += cx
    new_y += cy
    
    return new_x, new_y


def rotate_scene(scene_data: Dict, angle_deg: float) -> Dict:
    """
    旋转整个场景
    
    Args:
        scene_data: 原始场景数据
        angle_deg: 旋转角度（度数：0, 90, 180, 270）
        
    Returns:
        旋转后的场景数据副本
    """
    if angle_deg == 0:
        return copy.deepcopy(scene_data)
    
    rotated = copy.deepcopy(scene_data)
    angle_rad = math.radians(angle_deg)
    
    # 首先获取原始场景的中心点（用于绕中心旋转）
    coords = get_all_coordinates(scene_data)
    if not coords:
        return rotated
    
    # 计算场景中心
    min_x = min(c[0] for c in coords)
    max_x = max(c[0] for c in coords)
    min_y = min(c[1] for c in coords)
    max_y = max(c[1] for c in coords)
    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2
    
    # 旋转 blueprint 中的房间坐标
    if 'blueprint' in rotated and 'rooms' in rotated['blueprint']:
        for room_name, room_data in rotated['blueprint']['rooms'].items():
            if 'shape' in room_data and 'coordinates' in room_data['shape']:
                new_coords = []
                for polygon in room_data['shape']['coordinates']:
                    new_polygon = []
                    for point in polygon:
                        if len(point) >= 2:
                            new_x, new_y = rotate_point_2d(point[0], point[1], angle_rad, cx, cy)
                            new_polygon.append([new_x, new_y])
                        else:
                            new_polygon.append(point)
                    new_coords.append(new_polygon)
                room_data['shape']['coordinates'] = new_coords
    
    # 旋转物体坐标
    for key, value in rotated.items():
        if key in ['blueprint', 'portal']:
            continue
        if isinstance(value, dict) and 'spatial' in value:
            loc = value['spatial'].get('location', [0, 0, 0])
            if len(loc) >= 2:
                new_x, new_y = rotate_point_2d(loc[0], loc[1], angle_rad, cx, cy)
                value['spatial']['location'] = [new_x, new_y] + loc[2:]
            
            # 旋转物体自身的 rotation
            rot = value['spatial'].get('rotation', [0, 0, 0])
            if len(rot) >= 3:
                # rotation 的第三个值是绕 z 轴的旋转
                value['spatial']['rotation'] = [rot[0], rot[1], rot[2] + angle_rad]
    
    # 旋转 portal 坐标
    if 'portal' in rotated:
        for portal in rotated['portal']:
            if 'seg' in portal:
                new_seg = []
                for point in portal['seg']:
                    if len(point) >= 2:
                        new_x, new_y = rotate_point_2d(point[0], point[1], angle_rad, cx, cy)
                        new_seg.append([new_x, new_y])
                    else:
                        new_seg.append(point)
                portal['seg'] = new_seg
            
            # 旋转 portal 的 rot 属性
            if 'rot' in portal:
                portal['rot'] = portal['rot'] + angle_rad
    
    return rotated


def translate_scene_to_origin(scene_data: Dict) -> Dict:
    """
    将场景平移到原点（基于 min_x, min_y）
    
    Args:
        scene_data: 场景数据
        
    Returns:
        平移后的场景数据副本
    """
    translated = copy.deepcopy(scene_data)
    
    # 获取所有坐标
    coords = get_all_coordinates(scene_data)
    if not coords:
        return translated
    
    # 计算最小坐标
    min_x = min(c[0] for c in coords)
    min_y = min(c[1] for c in coords)
    
    # 平移量
    dx = -min_x
    dy = -min_y
    
    # 平移 blueprint 中的房间坐标
    if 'blueprint' in translated and 'rooms' in translated['blueprint']:
        for room_name, room_data in translated['blueprint']['rooms'].items():
            if 'shape' in room_data and 'coordinates' in room_data['shape']:
                new_coords = []
                for polygon in room_data['shape']['coordinates']:
                    new_polygon = []
                    for point in polygon:
                        if len(point) >= 2:
                            new_polygon.append([point[0] + dx, point[1] + dy])
                        else:
                            new_polygon.append(point)
                    new_coords.append(new_polygon)
                room_data['shape']['coordinates'] = new_coords
    
    # 平移物体坐标
    for key, value in translated.items():
        if key in ['blueprint', 'portal']:
            continue
        if isinstance(value, dict) and 'spatial' in value:
            loc = value['spatial'].get('location', [0, 0, 0])
            if len(loc) >= 2:
                value['spatial']['location'] = [loc[0] + dx, loc[1] + dy] + loc[2:]
    
    # 平移 portal 坐标
    if 'portal' in translated:
        for portal in translated['portal']:
            if 'seg' in portal:
                new_seg = []
                for point in portal['seg']:
                    if len(point) >= 2:
                        new_seg.append([point[0] + dx, point[1] + dy])
                    else:
                        new_seg.append(point)
                portal['seg'] = new_seg
    
    return translated


def mirror_scene(scene_data: Dict, mirror_type: str) -> Dict:
    """
    对场景进行镜像翻转
    
    Args:
        scene_data: 原始场景数据
        mirror_type: 镜像类型 - 'none', 'horizontal'(水平翻转，沿y轴), 'vertical'(垂直翻转，沿x轴)
        
    Returns:
        镜像后的场景数据副本
    """
    if mirror_type == 'none':
        return copy.deepcopy(scene_data)
    
    mirrored = copy.deepcopy(scene_data)
    
    # 获取场景中心用于镜像
    coords = get_all_coordinates(scene_data)
    if not coords:
        return mirrored
    
    min_x = min(c[0] for c in coords)
    max_x = max(c[0] for c in coords)
    min_y = min(c[1] for c in coords)
    max_y = max(c[1] for c in coords)
    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2
    
    def mirror_point(x: float, y: float) -> Tuple[float, float]:
        """对单个点进行镜像"""
        if mirror_type == 'horizontal':  # 沿 y 轴翻转（x 坐标取反）
            return 2 * cx - x, y
        elif mirror_type == 'vertical':  # 沿 x 轴翻转（y 坐标取反）
            return x, 2 * cy - y
        return x, y
    
    # 镜像 blueprint 中的房间坐标
    if 'blueprint' in mirrored and 'rooms' in mirrored['blueprint']:
        for room_name, room_data in mirrored['blueprint']['rooms'].items():
            if 'shape' in room_data and 'coordinates' in room_data['shape']:
                new_coords = []
                for polygon in room_data['shape']['coordinates']:
                    new_polygon = []
                    for point in polygon:
                        if len(point) >= 2:
                            new_x, new_y = mirror_point(point[0], point[1])
                            new_polygon.append([new_x, new_y])
                        else:
                            new_polygon.append(point)
                    new_coords.append(new_polygon)
                room_data['shape']['coordinates'] = new_coords
    
    # 镜像物体坐标
    for key, value in mirrored.items():
        if key in ['blueprint', 'portal']:
            continue
        if isinstance(value, dict) and 'spatial' in value:
            loc = value['spatial'].get('location', [0, 0, 0])
            if len(loc) >= 2:
                new_x, new_y = mirror_point(loc[0], loc[1])
                value['spatial']['location'] = [new_x, new_y] + loc[2:]
            
            # 镜像物体自身的 rotation
            rot = value['spatial'].get('rotation', [0, 0, 0])
            if len(rot) >= 3:
                # 镜像时旋转角度也需要调整
                if mirror_type == 'horizontal':
                    value['spatial']['rotation'] = [rot[0], rot[1], math.pi - rot[2]]
                elif mirror_type == 'vertical':
                    value['spatial']['rotation'] = [rot[0], rot[1], -rot[2]]
    
    # 镜像 portal 坐标
    if 'portal' in mirrored:
        for portal in mirrored['portal']:
            if 'seg' in portal:
                new_seg = []
                for point in portal['seg']:
                    if len(point) >= 2:
                        new_x, new_y = mirror_point(point[0], point[1])
                        new_seg.append([new_x, new_y])
                    else:
                        new_seg.append(point)
                portal['seg'] = new_seg
            
            # 镜像 portal 的 rot 属性
            if 'rot' in portal:
                if mirror_type == 'horizontal':
                    portal['rot'] = math.pi - portal['rot']
                elif mirror_type == 'vertical':
                    portal['rot'] = -portal['rot']
    
    return mirrored


def translate_scene_by_offset(scene_data: Dict, dx: float, dy: float) -> Dict:
    """
    将场景按指定偏移量平移
    
    Args:
        scene_data: 场景数据
        dx: x 方向平移量
        dy: y 方向平移量
        
    Returns:
        平移后的场景数据副本
    """
    translated = copy.deepcopy(scene_data)
    
    # 平移 blueprint 中的房间坐标
    if 'blueprint' in translated and 'rooms' in translated['blueprint']:
        for room_name, room_data in translated['blueprint']['rooms'].items():
            if 'shape' in room_data and 'coordinates' in room_data['shape']:
                new_coords = []
                for polygon in room_data['shape']['coordinates']:
                    new_polygon = []
                    for point in polygon:
                        if len(point) >= 2:
                            new_polygon.append([point[0] + dx, point[1] + dy])
                        else:
                            new_polygon.append(point)
                    new_coords.append(new_polygon)
                room_data['shape']['coordinates'] = new_coords
    
    # 平移物体坐标
    for key, value in translated.items():
        if key in ['blueprint', 'portal']:
            continue
        if isinstance(value, dict) and 'spatial' in value:
            loc = value['spatial'].get('location', [0, 0, 0])
            if len(loc) >= 2:
                value['spatial']['location'] = [loc[0] + dx, loc[1] + dy] + loc[2:]
    
    # 平移 portal 坐标
    if 'portal' in translated:
        for portal in translated['portal']:
            if 'seg' in portal:
                new_seg = []
                for point in portal['seg']:
                    if len(point) >= 2:
                        new_seg.append([point[0] + dx, point[1] + dy])
                    else:
                        new_seg.append(point)
                portal['seg'] = new_seg
    
    return translated


def extract_object_locations(scene_data: Dict) -> List[List[float]]:
    """
    快速提取场景中所有物体的位置（用于网格搜索优化）
    
    Args:
        scene_data: 场景数据
        
    Returns:
        物体位置列表 [[x, y, z], ...]
    """
    locations = []
    for key, value in scene_data.items():
        if key in ['blueprint', 'portal']:
            continue
        if isinstance(value, dict) and 'class' in value and 'spatial' in value:
            loc = value['spatial'].get('location', [0, 0, 0])
            locations.append(list(loc))
    return locations


def match_objects_fast(pred_objects: List[ObjectInfo], 
                       gt_objects: List[ObjectInfo], 
                       threshold: float,
                       offset_x: float = 0,
                       offset_y: float = 0) -> Tuple[int, int, int]:
    """
    快速匹配预测物体和真实物体（支持额外偏移量）
    
    Args:
        pred_objects: 预测物体列表
        gt_objects: 真实物体列表
        threshold: 距离阈值
        offset_x: x 方向额外偏移量
        offset_y: y 方向额外偏移量
        
    Returns:
        (TP, FP, FN) - 真阳性、假阳性、假阴性数量
    """
    # 按类别分组
    pred_by_class = defaultdict(list)
    for obj in pred_objects:
        # 应用偏移量
        adjusted_loc = [obj.location[0] + offset_x, obj.location[1] + offset_y] + obj.location[2:]
        pred_by_class[obj.obj_class].append((obj.obj_id, adjusted_loc))
    
    # 记录已匹配的预测物体
    matched_pred = set()
    matched_gt = set()
    
    # 对每个 GT 物体，寻找最近的同类 pred 物体
    for gt_obj in gt_objects:
        best_match = None
        best_distance = float('inf')
        
        for pred_id, pred_loc in pred_by_class[gt_obj.obj_class]:
            if pred_id in matched_pred:
                continue
            
            dx = pred_loc[0] - gt_obj.location[0]
            dy = pred_loc[1] - gt_obj.location[1]
            distance = math.sqrt(dx * dx + dy * dy)
            
            if distance < threshold and distance < best_distance:
                best_distance = distance
                best_match = pred_id
        
        if best_match is not None:
            matched_pred.add(best_match)
            matched_gt.add(gt_obj.obj_id)
    
    # 计算 TP, FP, FN
    TP = len(matched_gt)
    FP = len(pred_objects) - len(matched_pred)
    FN = len(gt_objects) - len(matched_gt)
    
    return TP, FP, FN


def match_objects_multi_threshold(pred_objects: List[ObjectInfo], 
                                   gt_objects: List[ObjectInfo], 
                                   thresholds: List[float],
                                   offset_x: float = 0,
                                   offset_y: float = 0,
                                   return_matches: bool = False) -> Dict[float, Tuple[int, int, int]]:
    """
    多阈值匹配预测物体和真实物体（优化版：只计算一次距离矩阵）
    
    Args:
        pred_objects: 预测物体列表
        gt_objects: 真实物体列表
        thresholds: 多个距离阈值列表
        offset_x: x 方向额外偏移量
        offset_y: y 方向额外偏移量
        return_matches: 是否返回匹配对（用于 params 评估）
        
    Returns:
        Dict[threshold -> (TP, FP, FN)] 或 
        Dict[threshold -> (TP, FP, FN, matched_pairs)] 当 return_matches=True
    """
    # 按类别分组，并预先计算调整后的位置
    pred_by_class = defaultdict(list)
    pred_id_to_obj = {obj.obj_id: obj for obj in pred_objects}
    gt_id_to_obj = {obj.obj_id: obj for obj in gt_objects}
    
    for obj in pred_objects:
        adjusted_loc = [obj.location[0] + offset_x, obj.location[1] + offset_y] + obj.location[2:]
        pred_by_class[obj.obj_class].append((obj.obj_id, adjusted_loc))
    
    # 预先计算所有同类物体对之间的距离
    # distance_info: {gt_obj_id: [(pred_id, distance), ...]}
    distance_info = {}
    for gt_obj in gt_objects:
        distances = []
        for pred_id, pred_loc in pred_by_class[gt_obj.obj_class]:
            dx = pred_loc[0] - gt_obj.location[0]
            dy = pred_loc[1] - gt_obj.location[1]
            distance = math.sqrt(dx * dx + dy * dy)
            distances.append((pred_id, distance))
        # 按距离排序
        distances.sort(key=lambda x: x[1])
        distance_info[gt_obj.obj_id] = distances
    
    # 对每个阈值计算匹配结果
    results = {}
    max_threshold = max(thresholds)
    
    for threshold in thresholds:
        matched_pred = set()
        matched_gt = set()
        matched_pairs_ids = []  # [(pred_id, gt_id), ...]
        
        # 对每个 GT 物体，寻找最近的未匹配的同类 pred 物体
        for gt_obj in gt_objects:
            for pred_id, distance in distance_info[gt_obj.obj_id]:
                if distance >= threshold:
                    break  # 由于已排序，后面的距离只会更大
                if pred_id not in matched_pred:
                    matched_pred.add(pred_id)
                    matched_gt.add(gt_obj.obj_id)
                    matched_pairs_ids.append((pred_id, gt_obj.obj_id))
                    break
        
        TP = len(matched_gt)
        FP = len(pred_objects) - len(matched_pred)
        FN = len(gt_objects) - len(matched_gt)
        
        if return_matches:
            results[threshold] = (TP, FP, FN, matched_pairs_ids)
        else:
            results[threshold] = (TP, FP, FN)
    
    return results, pred_id_to_obj, gt_id_to_obj


def calculate_distance(loc1: List[float], loc2: List[float]) -> float:
    """
    计算两个位置之间的欧氏距离（只考虑 x, y 坐标）
    
    Args:
        loc1, loc2: 位置坐标 [x, y, z]
        
    Returns:
        欧氏距离
    """
    dx = loc1[0] - loc2[0]
    dy = loc1[1] - loc2[1]
    return math.sqrt(dx * dx + dy * dy)


def match_objects(pred_objects: List[ObjectInfo], 
                  gt_objects: List[ObjectInfo], 
                  threshold: float) -> Tuple[int, int, int]:
    """
    匹配预测物体和真实物体
    
    Args:
        pred_objects: 预测物体列表
        gt_objects: 真实物体列表
        threshold: 距离阈值
        
    Returns:
        (TP, FP, FN) - 真阳性、假阳性、假阴性数量
    """
    # 按类别分组
    pred_by_class = defaultdict(list)
    for obj in pred_objects:
        pred_by_class[obj.obj_class].append(obj)
    
    gt_by_class = defaultdict(list)
    for obj in gt_objects:
        gt_by_class[obj.obj_class].append(obj)
    
    # 记录已匹配的预测物体
    matched_pred = set()
    matched_gt = set()
    
    # 对每个 GT 物体，寻找最近的同类 pred 物体
    for gt_obj in gt_objects:
        best_match = None
        best_distance = float('inf')
        
        for pred_obj in pred_by_class[gt_obj.obj_class]:
            if pred_obj.obj_id in matched_pred:
                continue
            
            distance = calculate_distance(pred_obj.location, gt_obj.location)
            if distance < threshold and distance < best_distance:
                best_distance = distance
                best_match = pred_obj
        
        if best_match is not None:
            matched_pred.add(best_match.obj_id)
            matched_gt.add(gt_obj.obj_id)
    
    # 计算 TP, FP, FN
    TP = len(matched_gt)  # 成功匹配的 GT 物体数
    FP = len(pred_objects) - len(matched_pred)  # 未匹配的 pred 物体数
    FN = len(gt_objects) - len(matched_gt)  # 未匹配的 GT 物体数
    
    return TP, FP, FN


# 默认阈值列表：0.1 到 1.0 共 10 个值
DEFAULT_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def calculate_f1_for_scene(pred_data: Dict, 
                           gt_data: Dict, 
                           thresholds: List[float] = None,
                           return_transformed: bool = False,
                           num_samples: int = 1000,
                           search_range: float = 1.0) -> Tuple[float, float, float, int, Dict, Optional[Dict], Optional[Dict]]:
    """
    计算单个场景的 F1-score（尝试四个旋转角度 + 镜像翻转 + 网格搜索平移，取最佳结果）
    
    Args:
        pred_data: 预测场景数据
        gt_data: 真实场景数据
        thresholds: 距离阈值列表，默认 [0.1, 0.2, ..., 1.0]
        return_transformed: 是否返回变换后的场景数据（用于 debug）
        num_samples: 网格搜索每个维度的分辨率（默认1000，即 x 和 y 方向各采样 1000 次，总共 1000×1000=1,000,000 次）
        search_range: 平移搜索范围 [-search_range, search_range]（默认1.0米）
        
    Returns:
        (best_f1, best_precision, best_recall, best_angle, best_metrics, 
         best_pred_transformed, gt_transformed)
        best_metrics 中包含 'multi_threshold_results' 字段，存储各阈值的结果
    """
    # 首先将 GT 平移到原点
    gt_translated = translate_scene_to_origin(gt_data)
    gt_objects = extract_objects_from_scene(gt_translated)
    
    best_f1 = 0
    best_precision = 0
    best_recall = 0
    best_angle = 0
    best_mirror = 'none'
    best_offset = (0, 0)
    best_metrics = {}
    best_pred_transformed = None
    
    # 保存所有变换后的 pred（用于 debug）
    all_transformed_preds = {}
    
    # 网格化采样偏移量：在 [-search_range, search_range] 区间均匀划分
    # num_samples 表示每个维度的分辨率（x 和 y 方向各采样 num_samples 次）
    # 总共采样 num_samples * num_samples 个点
    grid_resolution = num_samples  # 每个维度的采样点数
    if grid_resolution < 1:
        grid_resolution = 1
    
    # 预计算 x 和 y 方向的采样点
    grid_x = []
    grid_y = []
    for i in range(grid_resolution):
        # 均匀划分 [-search_range, search_range]
        val = -search_range + (2 * search_range * i) / (grid_resolution - 1) if grid_resolution > 1 else 0
        grid_x.append(val)
        grid_y.append(val)
    
    # 定义所有变换组合：4个旋转角度 × 3种镜像（无、水平、垂直）
    rotations = [0, 90, 180, 270]
    mirrors = ['none', 'horizontal', 'vertical']
    
    # 使用默认阈值列表
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS
    
    # 初始化每个阈值的最佳结果
    best_results_by_threshold = {t: {'f1': 0, 'precision': 0, 'recall': 0, 
                                      'TP': 0, 'FP': 0, 'FN': 0,
                                      'angle': 0, 'mirror': 'none', 
                                      'offset_x': 0, 'offset_y': 0} 
                                 for t in thresholds}
    
    # 使用中间阈值（0.5）作为主阈值来确定最佳变换
    main_threshold = 0.5 if 0.5 in thresholds else thresholds[len(thresholds)//2]
    
    # 遍历所有旋转和镜像组合
    for angle in rotations:
        for mirror in mirrors:
            # 旋转 pred
            pred_rotated = rotate_scene(pred_data, angle)
            # 镜像翻转
            pred_mirrored = mirror_scene(pred_rotated, mirror)
            # 平移到原点
            pred_translated = translate_scene_to_origin(pred_mirrored)
            # 提取物体
            pred_objects = extract_objects_from_scene(pred_translated)
            
            # 保存变换后的 pred（用于 debug）
            transform_key = f"angle_{angle}_mirror_{mirror}"
            if return_transformed:
                all_transformed_preds[transform_key] = pred_translated
            
            # 网格搜索最佳平移偏移（x 和 y 方向各 grid_resolution 次）
            for offset_x in grid_x:
                for offset_y in grid_y:
                    # 一次计算所有阈值的结果
                    multi_results, _, _ = match_objects_multi_threshold(
                        pred_objects, gt_objects, thresholds, offset_x, offset_y)
                    
                    for t, (TP, FP, FN) in multi_results.items():
                        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
                        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
                        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                        
                        if f1 > best_results_by_threshold[t]['f1']:
                            best_results_by_threshold[t] = {
                                'f1': f1, 'precision': precision, 'recall': recall,
                                'TP': TP, 'FP': FP, 'FN': FN,
                                'angle': angle, 'mirror': mirror,
                                'offset_x': offset_x, 'offset_y': offset_y
                            }
                    
                    # 使用主阈值的结果来确定最佳变换
                    TP, FP, FN = multi_results[main_threshold]
                    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
                    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                    if f1 > best_f1:
                        best_f1 = f1
                        best_precision = precision
                        best_recall = recall
                        best_angle = angle
                        best_mirror = mirror
                        best_offset = (offset_x, offset_y)
                        best_metrics = {
                            'TP': TP,
                            'FP': FP,
                            'FN': FN,
                            'precision': precision,
                            'recall': recall,
                            'f1': f1,
                            'angle': angle,
                            'mirror': mirror,
                            'offset_x': offset_x,
                            'offset_y': offset_y,
                            'num_pred_objects': len(pred_objects),
                            'num_gt_objects': len(gt_objects)
                        }
                        if return_transformed:
                            # 应用最佳偏移量生成最终的 pred
                            best_pred_transformed = translate_scene_by_offset(pred_translated, offset_x, offset_y)
    
    # 将多阈值结果添加到 best_metrics
    best_metrics['multi_threshold_results'] = best_results_by_threshold
    best_metrics['num_pred_objects'] = len(extract_objects_from_scene(translate_scene_to_origin(pred_data)))
    best_metrics['num_gt_objects'] = len(gt_objects)
    
    # === 在最佳变换下计算 params 准确性 ===
    # 重新应用最佳变换，获取匹配对
    pred_rotated = rotate_scene(pred_data, best_angle)
    pred_mirrored = mirror_scene(pred_rotated, best_mirror)
    pred_translated = translate_scene_to_origin(pred_mirrored)
    pred_objects_final = extract_objects_from_scene(pred_translated)
    
    # 使用主阈值获取匹配对
    multi_results_with_matches, pred_id_to_obj, gt_id_to_obj = match_objects_multi_threshold(
        pred_objects_final, gt_objects, [main_threshold], 
        best_offset[0], best_offset[1], return_matches=True)
    
    _, _, _, matched_pairs_ids = multi_results_with_matches[main_threshold]
    
    # 构建匹配的物体对（完整物体数据，用于 params 评估）
    matched_pairs = []
    for pred_id, gt_id in matched_pairs_ids:
        pred_obj_dict = _get_full_object_from_scene(pred_translated, pred_id)
        gt_obj_dict = _get_full_object_from_scene(gt_translated, gt_id)
        if pred_obj_dict and gt_obj_dict:
            matched_pairs.append((pred_obj_dict, gt_obj_dict))
    
    # 计算 params 准确性
    if matched_pairs:
        params_result = evaluate_matched_objects_params(matched_pairs)
        best_metrics['params_accuracy'] = params_result
    else:
        best_metrics['params_accuracy'] = None
    
    if return_transformed:
        return best_f1, best_precision, best_recall, best_angle, best_metrics, best_pred_transformed, gt_translated, all_transformed_preds
    else:
        return best_f1, best_precision, best_recall, best_angle, best_metrics, None, None, None


def _process_single_file(gt_file: str, pred_dir: str, gt_dir: str, 
                         thresholds: List[float], debug: bool, 
                         num_samples: int, search_range: float) -> Dict:
    """
    处理单个文件的辅助函数（用于并行执行）
    """
    gt_path = os.path.join(gt_dir, gt_file)
    pred_path = os.path.join(pred_dir, gt_file)
    
    # 检查 pred 文件是否存在
    if not os.path.exists(pred_path):
        return {
            'status': 'missing_pred',
            'scene_result': {
                'status': 'missing_pred',
                'error': f'找不到对应的 pred 文件'
            }
        }
    
    try:
        # 读取文件
        with open(gt_path, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        with open(pred_path, 'r', encoding='utf-8') as f:
            pred_data = json.load(f)
        
        # 计算 F1-score
        f1, precision, recall, best_angle, metrics, pred_transformed, gt_transformed, all_transformed_preds = calculate_f1_for_scene(
            pred_data, gt_data, thresholds=thresholds, return_transformed=debug,
            num_samples=num_samples, search_range=search_range
        )
        
        return {
            'status': 'success',
            'scene_result': {'status': 'success', **metrics},
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'best_angle': best_angle,
            'metrics': metrics,
            'pred_transformed': pred_transformed,
            'gt_transformed': gt_transformed,
            'all_transformed_preds': all_transformed_preds
        }
    except Exception as e:
        return {
            'status': 'error',
            'scene_result': {'status': 'error', 'error': str(e)}
        }


def evaluate_folders(pred_dir: str, 
                     gt_dir: str, 
                     thresholds: List[float] = None,
                     verbose: bool = True,
                     debug: bool = False,
                     debug_output_dir: str = None,
                     show_progress: bool = True,
                     num_samples: int = 1000,
                     search_range: float = 1.0,
                     num_workers: int = None,
                     save_result_to_parent: bool = False) -> Dict:
    """
    评估两个文件夹中的所有场景
    
    Args:
        pred_dir: 预测文件夹路径
        gt_dir: 真实文件夹路径
        thresholds: 距离阈值列表，默认 [0.1, 0.2, ..., 1.0]
        verbose: 是否打印详细信息
        debug: 是否开启 debug 模式（输出旋转后的 JSON 文件）
        debug_output_dir: debug 模式下输出文件的目录
        show_progress: 是否显示进度条
        num_samples: 网格搜索的采样次数
        search_range: 平移搜索范围 [-range, range]
        num_workers: 并行进程数，默认为 CPU 核心数
        save_result_to_parent: 是否将结果保存到 pred_dir 的父目录下的 result.txt
        
    Returns:
        评估结果字典
    """
    # 如果开启 debug 模式，创建输出目录
    if debug:
        if debug_output_dir is None:
            debug_output_dir = os.path.join(os.path.dirname(pred_dir), 'debug_output')
        
        # 创建子目录
        debug_pred_dir = os.path.join(debug_output_dir, 'pred_transformed')
        debug_gt_dir = os.path.join(debug_output_dir, 'gt_transformed')
        debug_all_rotations_dir = os.path.join(debug_output_dir, 'all_rotations')
        
        os.makedirs(debug_pred_dir, exist_ok=True)
        os.makedirs(debug_gt_dir, exist_ok=True)
        os.makedirs(debug_all_rotations_dir, exist_ok=True)
        
        if verbose:
            print(f"Debug 模式已开启，输出目录: {debug_output_dir}")
    
    # 使用默认阈值列表
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS
    
    results = {
        'thresholds': thresholds,
        'pred_dir': pred_dir,
        'gt_dir': gt_dir,
        'scenes': {},
        'summary': {}
    }
    
    # 获取所有 GT 文件
    gt_files = [f for f in os.listdir(gt_dir) if f.endswith('.json')]
    
    if verbose:
        print(f"找到 {len(gt_files)} 个 GT 文件")
        print(f"阈值列表: {thresholds}")
        print("-" * 60)
    
    total_TP = 0
    total_FP = 0
    total_FN = 0
    f1_scores = []
    precisions = []
    recalls = []
    matched_count = 0
    
    # params 准确性汇总统计
    from params_accuracy import ParamsStats
    total_params_stats = ParamsStats()
    per_class_params_stats = defaultdict(ParamsStats)
    
    # 多阈值统计
    multi_threshold_stats = {t: {'total_TP': 0, 'total_FP': 0, 'total_FN': 0,
                                  'f1_scores': [], 'precisions': [], 'recalls': []}
                             for t in thresholds}
    
    # 并行处理所有文件
    gt_files_sorted = sorted(gt_files)
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    num_workers = min(num_workers, len(gt_files_sorted))
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有任务
        future_to_file = {
            executor.submit(
                _process_single_file,
                gt_file, pred_dir, gt_dir, thresholds, debug, num_samples, search_range
            ): gt_file for gt_file in gt_files_sorted
        }
        
        # 设置进度条
        if show_progress and HAS_TQDM:
            futures_iter = tqdm(as_completed(future_to_file), total=len(gt_files_sorted),
                               desc="处理场景", unit="个", ncols=100, leave=True,
                               bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        else:
            futures_iter = as_completed(future_to_file)
        
        # 收集结果
        for future in futures_iter:
            gt_file = future_to_file[future]
            try:
                result = future.result()
                results['scenes'][gt_file] = result['scene_result']
                
                if result['status'] == 'success':
                    metrics = result['metrics']
                    total_TP += metrics['TP']
                    total_FP += metrics['FP']
                    total_FN += metrics['FN']
                    f1_scores.append(result['f1'])
                    precisions.append(result['precision'])
                    recalls.append(result['recall'])
                    matched_count += 1
                    
                    # 收集多阈值结果
                    if 'multi_threshold_results' in metrics:
                        for t, t_result in metrics['multi_threshold_results'].items():
                            multi_threshold_stats[t]['total_TP'] += t_result['TP']
                            multi_threshold_stats[t]['total_FP'] += t_result['FP']
                            multi_threshold_stats[t]['total_FN'] += t_result['FN']
                            multi_threshold_stats[t]['f1_scores'].append(t_result['f1'])
                            multi_threshold_stats[t]['precisions'].append(t_result['precision'])
                            multi_threshold_stats[t]['recalls'].append(t_result['recall'])
                    
                    # 收集 params 准确性结果
                    if 'params_accuracy' in metrics and metrics['params_accuracy']:
                        params_result = metrics['params_accuracy']
                        total_summary = params_result['total']
                        key_stats = total_summary['key_stats']
                        float_stats = total_summary['float_stats']
                        int_stats = total_summary['int_stats']
                        bool_stats = total_summary['bool_stats']
                        str_stats = total_summary['str_stats']
                        object_perfect_stats = total_summary.get('object_perfect_stats', {})
                        
                        # 累加原始统计值
                        total_params_stats.total_gt_keys += key_stats['total_gt_keys']
                        total_params_stats.matched_keys += key_stats['matched_keys']
                        total_params_stats.extra_pred_keys += key_stats['extra_pred_keys']
                        total_params_stats.float_count += float_stats['count']
                        total_params_stats.float_abs_error_sum += float_stats['mean_abs_error'] * float_stats['count']
                        total_params_stats.float_rel_error_sum += float_stats['mean_rel_error'] * float_stats['count']
                        total_params_stats.int_count += int_stats['count']
                        total_params_stats.int_abs_error_sum += int_stats['mean_abs_error'] * int_stats['count']
                        # 累加整数 GT 最大最小值
                        if int_stats.get('gt_min') is not None:
                            total_params_stats.int_gt_min = min(total_params_stats.int_gt_min, int_stats['gt_min'])
                        if int_stats.get('gt_max') is not None:
                            total_params_stats.int_gt_max = max(total_params_stats.int_gt_max, int_stats['gt_max'])
                        total_params_stats.bool_count += bool_stats['count']
                        total_params_stats.bool_correct += int(bool_stats['accuracy'] * bool_stats['count'])
                        total_params_stats.str_count += str_stats['count']
                        total_params_stats.str_correct += int(str_stats['accuracy'] * str_stats['count'])
                        # 累加物体 params 全对统计
                        total_params_stats.total_objects += object_perfect_stats.get('total_objects', 0)
                        total_params_stats.perfect_objects += object_perfect_stats.get('perfect_objects', 0)
                        # 累加 thickness 统计
                        thickness_stats = total_summary.get('thickness_stats', {})
                        total_params_stats.thickness_count += thickness_stats.get('count', 0)
                        total_params_stats.thickness_abs_error_sum += thickness_stats.get('mean_abs_error', 0) * thickness_stats.get('count', 0)
                        total_params_stats.thickness_rel_error_sum += thickness_stats.get('mean_rel_error', 0) * thickness_stats.get('count', 0)
                        # 累加 thickness GT 最大最小值
                        if thickness_stats.get('gt_min') is not None:
                            total_params_stats.thickness_gt_min = min(total_params_stats.thickness_gt_min, thickness_stats['gt_min'])
                        if thickness_stats.get('gt_max') is not None:
                            total_params_stats.thickness_gt_max = max(total_params_stats.thickness_gt_max, thickness_stats['gt_max'])
                    
                    if verbose:
                        mirror_str = metrics.get('mirror', 'none')
                        offset_x = metrics.get('offset_x', 0)
                        offset_y = metrics.get('offset_y', 0)
                        print(f"{gt_file}: F1={result['f1']:.4f}, P={result['precision']:.4f}, R={result['recall']:.4f}, "
                              f"角度={result['best_angle']}度, 镜像={mirror_str}, 偏移=({offset_x:.3f}, {offset_y:.3f}), "
                              f"GT物体={metrics['num_gt_objects']}, Pred物体={metrics['num_pred_objects']}")
                    
                    # Debug 模式：保存变换后的 JSON 文件
                    if debug and result.get('pred_transformed') is not None:
                        best_mirror = metrics.get('mirror', 'none')
                        best_offset_x = metrics.get('offset_x', 0)
                        best_offset_y = metrics.get('offset_y', 0)
                        best_pred_filename = gt_file.replace('.json', f'_best_angle_{result["best_angle"]}_mirror_{best_mirror}_offset_{best_offset_x:.3f}_{best_offset_y:.3f}.json')
                        with open(os.path.join(debug_pred_dir, best_pred_filename), 'w', encoding='utf-8') as f:
                            json.dump(result['pred_transformed'], f, indent=2, ensure_ascii=False)
                        with open(os.path.join(debug_gt_dir, gt_file), 'w', encoding='utf-8') as f:
                            json.dump(result['gt_transformed'], f, indent=2, ensure_ascii=False)
                        if result.get('all_transformed_preds'):
                            scene_transforms_dir = os.path.join(debug_all_rotations_dir, gt_file.replace('.json', ''))
                            os.makedirs(scene_transforms_dir, exist_ok=True)
                            for transform_key, transformed_pred in result['all_transformed_preds'].items():
                                transform_filename = f'pred_{transform_key}.json'
                                with open(os.path.join(scene_transforms_dir, transform_filename), 'w', encoding='utf-8') as f:
                                    json.dump(transformed_pred, f, indent=2, ensure_ascii=False)
                elif verbose and result['status'] == 'missing_pred':
                    print(f"警告: 找不到对应的 pred 文件: {gt_file}")
                    
            except Exception as e:
                if verbose:
                    print(f"错误处理 {gt_file}: {str(e)}")
                results['scenes'][gt_file] = {
                    'status': 'error',
                    'error': str(e)
                }
    
    # 计算总体指标
    if matched_count > 0:
        # 宏平均（每个场景的平均）
        macro_f1 = sum(f1_scores) / len(f1_scores)
        macro_precision = sum(precisions) / len(precisions)
        macro_recall = sum(recalls) / len(recalls)
        
        # 微平均（基于总体 TP, FP, FN）
        micro_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
        micro_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) \
                   if (micro_precision + micro_recall) > 0 else 0
        
        results['summary'] = {
            'total_scenes': len(gt_files),
            'matched_scenes': matched_count,
            'total_TP': total_TP,
            'total_FP': total_FP,
            'total_FN': total_FN,
            'macro_f1': macro_f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'micro_f1': micro_f1,
            'micro_precision': micro_precision,
            'micro_recall': micro_recall
        }
        
        # 计算多阈值汇总结果
        multi_threshold_summary = {}
        for t in thresholds:
            stats = multi_threshold_stats[t]
            t_macro_f1 = sum(stats['f1_scores']) / len(stats['f1_scores']) if stats['f1_scores'] else 0
            t_macro_precision = sum(stats['precisions']) / len(stats['precisions']) if stats['precisions'] else 0
            t_macro_recall = sum(stats['recalls']) / len(stats['recalls']) if stats['recalls'] else 0
            
            t_micro_precision = stats['total_TP'] / (stats['total_TP'] + stats['total_FP']) if (stats['total_TP'] + stats['total_FP']) > 0 else 0
            t_micro_recall = stats['total_TP'] / (stats['total_TP'] + stats['total_FN']) if (stats['total_TP'] + stats['total_FN']) > 0 else 0
            t_micro_f1 = 2 * t_micro_precision * t_micro_recall / (t_micro_precision + t_micro_recall) if (t_micro_precision + t_micro_recall) > 0 else 0
            
            multi_threshold_summary[t] = {
                'macro_f1': t_macro_f1,
                'macro_precision': t_macro_precision,
                'macro_recall': t_macro_recall,
                'micro_f1': t_micro_f1,
                'micro_precision': t_micro_precision,
                'micro_recall': t_micro_recall,
                'total_TP': stats['total_TP'],
                'total_FP': stats['total_FP'],
                'total_FN': stats['total_FN']
            }
        results['summary']['multi_threshold'] = multi_threshold_summary
        
        # 添加 params 准确性汇总到 results
        results['summary']['params_accuracy'] = total_params_stats.to_summary()
        
        if verbose:
            print("-" * 60)
            print("总体结果:")
            print(f"  匹配场景数: {matched_count}/{len(gt_files)}")
            print(f"  总 TP: {total_TP}, FP: {total_FP}, FN: {total_FN}")
            print(f"  宏平均 - F1: {macro_f1:.4f}, Precision: {macro_precision:.4f}, Recall: {macro_recall:.4f}")
            print(f"  微平均 - F1: {micro_f1:.4f}, Precision: {micro_precision:.4f}, Recall: {micro_recall:.4f}")
            
            # 输出 params 准确性汇总
            print("\n" + "-" * 60)
            print("Params 准确性汇总:")
            params_summary = results['summary']['params_accuracy']
            print(f"  Key 匹配率: {params_summary['key_accuracy']:.4f}")
            if 'object_perfect_stats' in params_summary:
                obj_stats = params_summary['object_perfect_stats']
                print(f"  物体 Params 全对率: {obj_stats['perfect_rate']:.4f} "
                      f"({obj_stats['perfect_objects']}/{obj_stats['total_objects']})")
            print(f"  浮点数: 平均绝对误差={params_summary['float_stats']['mean_abs_error']:.6f}, "
                  f"平均相对误差={params_summary['float_stats']['mean_rel_error']*100:.2f}% (n={params_summary['float_stats']['count']})")
            print(f"  整数: 平均绝对误差={params_summary['int_stats']['mean_abs_error']:.4f} (n={params_summary['int_stats']['count']})")
            if params_summary['int_stats'].get('gt_min') is not None:
                print(f"         GT最小值={params_summary['int_stats']['gt_min']}, "
                      f"GT最大值={params_summary['int_stats']['gt_max']}")
            print(f"  布尔值: 正确率={params_summary['bool_stats']['accuracy']:.4f} (n={params_summary['bool_stats']['count']})")
            print(f"  字符串: 正确率={params_summary['str_stats']['accuracy']:.4f} (n={params_summary['str_stats']['count']})")
            print(f"  Thickness: 平均绝对误差={params_summary['thickness_stats']['mean_abs_error']:.6f}, "
                  f"平均相对误差={params_summary['thickness_stats']['mean_rel_error']*100:.2f}% (n={params_summary['thickness_stats']['count']})")
            if params_summary['thickness_stats'].get('gt_min') is not None:
                print(f"           GT最小值={params_summary['thickness_stats']['gt_min']:.6f}, "
                      f"GT最大值={params_summary['thickness_stats']['gt_max']:.6f}")
    else:
        results['summary'] = {
            'total_scenes': len(gt_files),
            'matched_scenes': 0,
            'error': '没有成功匹配的场景'
        }
        if verbose:
            print("没有成功匹配的场景")
    
    # 将结果写入 pred_dir 的父目录下的 result.txt
    if save_result_to_parent:
        parent_dir = os.path.dirname(os.path.abspath(pred_dir))
        result_file_path = os.path.join(parent_dir, 'result.txt')
        
        with open(result_file_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("F1-Score 评估结果\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"预测目录: {pred_dir}\n")
            f.write(f"真实目录: {gt_dir}\n")
            f.write(f"网格搜索分辨率: {num_samples}\n")
            f.write(f"平移搜索范围: [-{search_range}, {search_range}]\n\n")
            
            if 'summary' in results and 'error' not in results['summary']:
                summary = results['summary']
                
                f.write(f"场景数: {summary['matched_scenes']}/{summary['total_scenes']}\n")
                f.write("-" * 40 + "\n")
                f.write("阈值\tF1\tP\tR\n")
                f.write("-" * 40 + "\n")
                
                avg_f1 = 0
                avg_p = 0
                avg_r = 0
                count = 0
                
                for t in sorted(summary['multi_threshold'].keys()):
                    t_summary = summary['multi_threshold'][t]
                    f.write(f"{t:.1f}\t{t_summary['micro_f1']:.4f}\t{t_summary['micro_precision']:.4f}\t{t_summary['micro_recall']:.4f}\n")
                    avg_f1 += t_summary['micro_f1']
                    avg_p += t_summary['micro_precision']
                    avg_r += t_summary['micro_recall']
                    count += 1
                
                if count > 0:
                    f.write("-" * 40 + "\n")
                    f.write(f"AVG\t{avg_f1/count:.4f}\t{avg_p/count:.4f}\t{avg_r/count:.4f}\n")
                
                # 写入 params 准确性结果
                if 'params_accuracy' in summary:
                    params = summary['params_accuracy']
                    f.write("\n" + "=" * 40 + "\n")
                    f.write("Params 准确性评估\n")
                    f.write("=" * 40 + "\n")
                    f.write(f"Key 匹配率: {params['key_accuracy']:.4f}\n")
                    f.write(f"  GT Keys: {params['key_stats']['total_gt_keys']}, 匹配: {params['key_stats']['matched_keys']}\n")
                    # 写入物体 params 全对率
                    if 'object_perfect_stats' in params:
                        obj_stats = params['object_perfect_stats']
                        f.write(f"\n物体 Params 全对率: {obj_stats['perfect_rate']:.4f}\n")
                        f.write(f"  总物体数: {obj_stats['total_objects']}, 全对物体数: {obj_stats['perfect_objects']}\n")
                    f.write(f"\n浮点数 (n={params['float_stats']['count']}):")
                    f.write(f"\n  平均绝对误差: {params['float_stats']['mean_abs_error']:.6f}\n")
                    f.write(f"  平均相对误差: {params['float_stats']['mean_rel_error']*100:.2f}%\n")
                    f.write(f"\n整数 (n={params['int_stats']['count']}):")
                    f.write(f"\n  平均绝对误差: {params['int_stats']['mean_abs_error']:.4f}\n")
                    if params['int_stats'].get('gt_min') is not None:
                        f.write(f"  GT 最小值: {params['int_stats']['gt_min']}\n")
                        f.write(f"  GT 最大值: {params['int_stats']['gt_max']}\n")
                    f.write(f"\n布尔值 (n={params['bool_stats']['count']}):")
                    f.write(f"\n  正确率: {params['bool_stats']['accuracy']:.4f}\n")
                    f.write(f"\n字符串 (n={params['str_stats']['count']}):")
                    f.write(f"\n  正确率: {params['str_stats']['accuracy']:.4f}\n")
                    f.write(f"\nThickness (n={params['thickness_stats']['count']}):")
                    f.write(f"\n  平均绝对误差: {params['thickness_stats']['mean_abs_error']:.6f}\n")
                    f.write(f"  平均相对误差: {params['thickness_stats']['mean_rel_error']*100:.2f}%\n")
                    if params['thickness_stats'].get('gt_min') is not None:
                        f.write(f"  GT 最小值: {params['thickness_stats']['gt_min']:.6f}\n")
                        f.write(f"  GT 最大值: {params['thickness_stats']['gt_max']:.6f}\n")
            else:
                f.write("没有成功匹配的场景\n")
            
            f.write("\n" + "=" * 60 + "\n")
        
        if verbose:
            print(f"\n结果已保存到: {result_file_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='计算生成场景与真实场景之间的 F1-score',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python f1_score.py --pred_dir ./predictions --gt_dir ./ground_truth
    python f1_score.py --pred_dir ./pred --gt_dir ./gt --threshold 0.3 --output results.json
        """
    )
    
    parser.add_argument('--pred_dir', type=str, required=True,
                        help='预测结果文件夹路径')
    parser.add_argument('--gt_dir', type=str, default=None,
                        help='真实标注文件夹路径（可选，默认为 pred_dir 父目录下的 ground_truth）')
    parser.add_argument('--thresholds', type=str, default=None,
                        help='阈值列表，逗号分隔（默认 0.1,0.2,...,1.0）')
    parser.add_argument('--output', type=str, default=None,
                        help='输出结果的 JSON 文件路径（可选）')
    parser.add_argument('--quiet', action='store_true',
                        help='静默模式，不打印详细信息')
    parser.add_argument('--debug', action='store_true',
                        help='Debug 模式，输出旋转后的 JSON 文件用于检查物体是否错位')
    parser.add_argument('--debug_output_dir', type=str, default=None,
                        help='Debug 模式下输出文件的目录（默认在 pred_dir 同级目录下创建 debug_output）')
    parser.add_argument('--no_progress', action='store_true',
                        help='禁用进度条显示')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='网格搜索每个维度的分辨率（默认 100，即 x 和 y 方向各采样 100 次，总共 100×100=10,000 次）')
    parser.add_argument('--search_range', type=float, default=0.5,
                        help='平移搜索范围 [-range, range]（单位：米，默认 0.5）')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='并行进程数（默认为 CPU 核心数）')
    parser.add_argument('--save_result', action='store_true', default=True,
                        help='将结果保存到 pred_dir 父目录下的 result.txt（默认开启）')
    parser.add_argument('--no_save_result', action='store_true',
                        help='禁用将结果保存到 pred_dir 父目录下的 result.txt')
    
    args = parser.parse_args()
    
    # 如果未指定 gt_dir，自动设置为 pred_dir 父目录下的 ground_truth
    if args.gt_dir is None:
        parent_dir = os.path.dirname(os.path.abspath(args.pred_dir))
        args.gt_dir = os.path.join(parent_dir, 'ground_truth')
        print(f"未指定 --gt_dir，自动设置为: {args.gt_dir}")
    
    # 检查目录是否存在
    if not os.path.isdir(args.pred_dir):
        print(f"错误: pred 目录不存在: {args.pred_dir}")
        return 1
    if not os.path.isdir(args.gt_dir):
        print(f"错误: gt 目录不存在: {args.gt_dir}")
        return 1
    
    # 解析阈值列表
    thresholds = None
    if args.thresholds:
        thresholds = [float(t.strip()) for t in args.thresholds.split(',')]
    # 默认使用 DEFAULT_THRESHOLDS
    
    # 评估
    results = evaluate_folders(
        args.pred_dir,
        args.gt_dir,
        thresholds=thresholds,
        verbose=not args.quiet,
        debug=args.debug,
        debug_output_dir=args.debug_output_dir,
        show_progress=not args.no_progress,
        num_samples=args.num_samples,
        search_range=args.search_range,
        num_workers=args.num_workers,
        save_result_to_parent=not args.no_save_result
    )
    
    # 保存结果
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {args.output}")
    
    return 0


if __name__ == '__main__':
    exit(main())
