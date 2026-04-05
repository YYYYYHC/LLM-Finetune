#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Params 准确性评估模块

评估生成场景中物体 params 字段的准确性，包括：
1. Key 匹配率：params 中的字段名是否一致
2. 浮点数误差：连续数值的平均相对/绝对误差
3. 整数误差：整型数值的平均误差
4. 布尔正确率：布尔值的匹配率
5. 字符串正确率：字符串值的匹配率
"""

from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class ParamsStats:
    """Params 评估统计"""
    # Key 匹配统计
    total_gt_keys: int = 0
    matched_keys: int = 0
    extra_pred_keys: int = 0  # pred 中多出的 key
    
    # 浮点数统计
    float_count: int = 0
    float_abs_error_sum: float = 0.0
    float_rel_error_sum: float = 0.0
    
    # 整数统计
    int_count: int = 0
    int_abs_error_sum: float = 0.0
    int_gt_min: int = 2147483647  # GT 中整数的最小值
    int_gt_max: int = -2147483648  # GT 中整数的最大值
    
    # 布尔统计
    bool_count: int = 0
    bool_correct: int = 0
    
    # 字符串统计
    str_count: int = 0
    str_correct: int = 0
    
    # 列表统计（按元素类型分别统计）
    list_count: int = 0
    
    # thickness 浮点数统计（key 中包含 'thickness' 的浮点数）
    thickness_count: int = 0
    thickness_abs_error_sum: float = 0.0
    thickness_rel_error_sum: float = 0.0
    thickness_gt_min: float = float('inf')  # GT 中 thickness 的最小值
    thickness_gt_max: float = float('-inf')  # GT 中 thickness 的最大值
    
    # 物体 params 全对统计
    total_objects: int = 0
    perfect_objects: int = 0  # params 完全正确的物体数量
    
    def merge(self, other: 'ParamsStats') -> 'ParamsStats':
        """合并两个统计结果"""
        return ParamsStats(
            total_gt_keys=self.total_gt_keys + other.total_gt_keys,
            matched_keys=self.matched_keys + other.matched_keys,
            extra_pred_keys=self.extra_pred_keys + other.extra_pred_keys,
            float_count=self.float_count + other.float_count,
            float_abs_error_sum=self.float_abs_error_sum + other.float_abs_error_sum,
            float_rel_error_sum=self.float_rel_error_sum + other.float_rel_error_sum,
            int_count=self.int_count + other.int_count,
            int_abs_error_sum=self.int_abs_error_sum + other.int_abs_error_sum,
            int_gt_min=min(self.int_gt_min, other.int_gt_min),
            int_gt_max=max(self.int_gt_max, other.int_gt_max),
            bool_count=self.bool_count + other.bool_count,
            bool_correct=self.bool_correct + other.bool_correct,
            str_count=self.str_count + other.str_count,
            str_correct=self.str_correct + other.str_correct,
            list_count=self.list_count + other.list_count,
            thickness_count=self.thickness_count + other.thickness_count,
            thickness_abs_error_sum=self.thickness_abs_error_sum + other.thickness_abs_error_sum,
            thickness_rel_error_sum=self.thickness_rel_error_sum + other.thickness_rel_error_sum,
            thickness_gt_min=min(self.thickness_gt_min, other.thickness_gt_min),
            thickness_gt_max=max(self.thickness_gt_max, other.thickness_gt_max),
            total_objects=self.total_objects + other.total_objects,
            perfect_objects=self.perfect_objects + other.perfect_objects,
        )
    
    def to_summary(self) -> Dict[str, Any]:
        """生成汇总结果"""
        return {
            'key_accuracy': self.matched_keys / self.total_gt_keys if self.total_gt_keys > 0 else 0,
            'key_stats': {
                'total_gt_keys': self.total_gt_keys,
                'matched_keys': self.matched_keys,
                'extra_pred_keys': self.extra_pred_keys,
            },
            'float_stats': {
                'count': self.float_count,
                'mean_abs_error': self.float_abs_error_sum / self.float_count if self.float_count > 0 else 0,
                'mean_rel_error': self.float_rel_error_sum / self.float_count if self.float_count > 0 else 0,
            },
            'int_stats': {
                'count': self.int_count,
                'mean_abs_error': self.int_abs_error_sum / self.int_count if self.int_count > 0 else 0,
                'gt_min': self.int_gt_min if self.int_count > 0 else None,
                'gt_max': self.int_gt_max if self.int_count > 0 else None,
            },
            'bool_stats': {
                'count': self.bool_count,
                'accuracy': self.bool_correct / self.bool_count if self.bool_count > 0 else 0,
            },
            'str_stats': {
                'count': self.str_count,
                'accuracy': self.str_correct / self.str_count if self.str_count > 0 else 0,
            },
            'thickness_stats': {
                'count': self.thickness_count,
                'mean_abs_error': self.thickness_abs_error_sum / self.thickness_count if self.thickness_count > 0 else 0,
                'mean_rel_error': self.thickness_rel_error_sum / self.thickness_count if self.thickness_count > 0 else 0,
                'gt_min': self.thickness_gt_min if self.thickness_count > 0 else None,
                'gt_max': self.thickness_gt_max if self.thickness_count > 0 else None,
            },
            'object_perfect_stats': {
                'total_objects': self.total_objects,
                'perfect_objects': self.perfect_objects,
                'perfect_rate': self.perfect_objects / self.total_objects if self.total_objects > 0 else 0,
            },
        }


def is_float(value: Any) -> bool:
    """判断是否为浮点数（包括可转换的数值，但排除整数和布尔）"""
    if isinstance(value, bool):
        return False
    if isinstance(value, float):
        return True
    if isinstance(value, int):
        return False
    return False


def is_int(value: Any) -> bool:
    """判断是否为整数（排除布尔）"""
    if isinstance(value, bool):
        return False
    return isinstance(value, int)


def compare_values(pred_val: Any, gt_val: Any, stats: ParamsStats, key_name: str = None) -> None:
    """
    比较单个值并更新统计
    
    递归处理嵌套结构（dict、list）
    
    Args:
        pred_val: 预测值
        gt_val: 真实值
        stats: 统计对象
        key_name: 当前值对应的 key 名称（用于特殊 key 的统计，如 thickness）
    """
    # 1. 处理 None
    if gt_val is None:
        if pred_val is None:
            stats.str_count += 1
            stats.str_correct += 1
        else:
            stats.str_count += 1
        return
    
    # 2. 处理布尔值（必须在 int 之前判断，因为 bool 是 int 的子类）
    if isinstance(gt_val, bool):
        stats.bool_count += 1
        if isinstance(pred_val, bool) and pred_val == gt_val:
            stats.bool_correct += 1
        return
    
    # 3. 处理浮点数（只统计 -1 到 1 范围内的浮点数）
    if isinstance(gt_val, float):
        # 检查是否是 thickness 相关的 key（不受值域限制）
        is_thickness_key = key_name and 'thickness' in key_name.lower()
        
        if is_thickness_key and isinstance(pred_val, (int, float)) and not isinstance(pred_val, bool):
            # thickness 类型的浮点数：单独统计，不受值域限制
            stats.thickness_count += 1
            abs_error = abs(float(pred_val) - gt_val)
            rel_error = abs_error / abs(gt_val) if abs(gt_val) > 1e-9 else abs_error
            stats.thickness_abs_error_sum += abs_error
            stats.thickness_rel_error_sum += rel_error
            # 统计 GT 中 thickness 的最大最小值
            stats.thickness_gt_min = min(stats.thickness_gt_min, gt_val)
            stats.thickness_gt_max = max(stats.thickness_gt_max, gt_val)
        elif -1 <= gt_val <= 1 and isinstance(pred_val, (int, float)) and not isinstance(pred_val, bool):
            # 只比较值域在 [-1, 1] 范围内的浮点数，且预测值必须是数值类型
            stats.float_count += 1
            abs_error = abs(float(pred_val) - gt_val)
            rel_error = abs_error / abs(gt_val) if abs(gt_val) > 1e-9 else abs_error
            stats.float_abs_error_sum += abs_error
            stats.float_rel_error_sum += rel_error
        # 超出范围或类型不匹配的浮点数直接跳过，不计入统计
        return
    
    # 4. 处理整数（只统计 GT 值小于 32 的整数）
    if isinstance(gt_val, int):
        if gt_val < 32:
            stats.int_count += 1
            if isinstance(pred_val, (int, float)) and not isinstance(pred_val, bool):
                stats.int_abs_error_sum += abs(int(round(pred_val)) - gt_val)
            else:
                stats.int_abs_error_sum += abs(gt_val)
            # 统计 GT 中整数的最大最小值
            stats.int_gt_min = min(stats.int_gt_min, gt_val)
            stats.int_gt_max = max(stats.int_gt_max, gt_val)
        return
    
    # 5. 处理字符串
    if isinstance(gt_val, str):
        stats.str_count += 1
        if isinstance(pred_val, str) and pred_val == gt_val:
            stats.str_correct += 1
        return
    
    # 6. 处理列表（递归比较每个元素）
    if isinstance(gt_val, list):
        stats.list_count += 1
        if isinstance(pred_val, list):
            # 按位置比较，取较短的长度
            for i in range(min(len(gt_val), len(pred_val))):
                compare_values(pred_val[i], gt_val[i], stats, key_name)
            # 多出的 gt 元素也要统计（作为未匹配）
            for i in range(len(pred_val), len(gt_val)):
                compare_values(None, gt_val[i], stats, key_name)
        else:
            # pred 不是列表，所有 gt 元素视为未匹配
            for v in gt_val:
                compare_values(None, v, stats, key_name)
        return
    
    # 7. 处理嵌套字典（递归）
    if isinstance(gt_val, dict):
        if isinstance(pred_val, dict):
            compare_params(pred_val, gt_val, stats)
        else:
            # pred 不是字典，递归统计 gt 中的所有值为未匹配
            compare_params({}, gt_val, stats)
        return


def compare_params(pred_params: Dict, gt_params: Dict, stats: ParamsStats = None) -> ParamsStats:
    """
    比较 pred 和 gt 的 params 字段
    
    Args:
        pred_params: 预测的 params
        gt_params: 真实的 params
        stats: 可选的现有统计对象（用于递归）
    
    Returns:
        更新后的统计对象
    """
    if stats is None:
        stats = ParamsStats()
    
    gt_keys = set(gt_params.keys())
    pred_keys = set(pred_params.keys())
    
    # 统计 key 匹配
    stats.total_gt_keys += len(gt_keys)
    stats.matched_keys += len(gt_keys & pred_keys)
    stats.extra_pred_keys += len(pred_keys - gt_keys)
    
    # 比较共同 key 的值
    for key in gt_keys:
        gt_val = gt_params[key]
        pred_val = pred_params.get(key)
        
        if key in pred_keys:
            compare_values(pred_val, gt_val, stats, key_name=key)
        else:
            # key 不存在于 pred 中，仍需统计 gt 值的类型
            compare_values(None, gt_val, stats, key_name=key)
    
    return stats


def is_params_perfect(pred_params: Dict, gt_params: Dict, float_tol: float = 1e-6) -> bool:
    """
    判断单个物体的 params 是否完全正确（只检查 key 是否齐全）
    
    判断标准：GT 中的所有 key 是否都存在于 pred 中（递归检查嵌套字典）
    
    Args:
        pred_params: 预测的 params
        gt_params: 真实的 params
        float_tol: 浮点数比较的容差（当前未使用，保留参数兼容性）
    
    Returns:
        bool: GT 中的所有 key 是否都存在于 pred 中
    """
    return _all_gt_keys_exist(pred_params, gt_params)


def _all_gt_keys_exist(pred_val: Any, gt_val: Any) -> bool:
    """
    递归检查 GT 中的所有 key 是否都存在于 pred 中
    
    Args:
        pred_val: 预测值
        gt_val: 真实值
    
    Returns:
        bool: GT 中的所有 key 是否都存在于 pred 中
    """
    # 如果 gt 是字典，检查所有 key 是否存在于 pred 中
    if isinstance(gt_val, dict):
        if not isinstance(pred_val, dict):
            return False
        
        gt_keys = set(gt_val.keys())
        pred_keys = set(pred_val.keys())
        
        # GT 中的所有 key 必须都存在于 pred 中
        if not gt_keys.issubset(pred_keys):
            return False
        
        # 递归检查嵌套字典
        for key in gt_keys:
            if not _all_gt_keys_exist(pred_val.get(key), gt_val[key]):
                return False
        
        return True
    
    # 如果 gt 是列表，递归检查列表中的字典元素
    if isinstance(gt_val, list):
        if not isinstance(pred_val, list):
            return False
        
        # 列表长度必须一致才能逐个检查
        if len(pred_val) != len(gt_val):
            return False
        
        # 递归检查列表中的每个元素
        for p, g in zip(pred_val, gt_val):
            if not _all_gt_keys_exist(p, g):
                return False
        
        return True
    
    # 其他类型（非字典、非列表）不需要检查 key
    return True


def _values_equal(pred_val: Any, gt_val: Any, float_tol: float = 1e-6) -> bool:
    """
    递归比较两个值是否相等
    
    Args:
        pred_val: 预测值
        gt_val: 真实值
        float_tol: 浮点数比较的容差
    
    Returns:
        bool: 值是否相等
    """
    # 1. 处理 None
    if gt_val is None:
        return pred_val is None
    
    # 2. 处理布尔值
    if isinstance(gt_val, bool):
        return isinstance(pred_val, bool) and pred_val == gt_val
    
    # 3. 处理浮点数
    if isinstance(gt_val, float):
        if not isinstance(pred_val, (int, float)) or isinstance(pred_val, bool):
            return False
        return abs(float(pred_val) - gt_val) <= float_tol
    
    # 4. 处理整数
    if isinstance(gt_val, int):
        if not isinstance(pred_val, (int, float)) or isinstance(pred_val, bool):
            return False
        return int(round(pred_val)) == gt_val
    
    # 5. 处理字符串
    if isinstance(gt_val, str):
        return isinstance(pred_val, str) and pred_val == gt_val
    
    # 6. 处理列表
    if isinstance(gt_val, list):
        if not isinstance(pred_val, list) or len(pred_val) != len(gt_val):
            return False
        return all(_values_equal(p, g, float_tol) for p, g in zip(pred_val, gt_val))
    
    # 7. 处理字典
    if isinstance(gt_val, dict):
        if not isinstance(pred_val, dict):
            return False
        return is_params_perfect(pred_val, gt_val, float_tol)
    
    # 其他类型直接比较
    return pred_val == gt_val


def evaluate_matched_objects_params(
    matched_pairs: List[Tuple[Dict, Dict]],
    float_tol: float = 1e-6
) -> Dict[str, Any]:
    """
    评估所有匹配物体对的 params 准确性
    
    Args:
        matched_pairs: [(pred_obj, gt_obj), ...] 匹配的物体对列表
                       每个 obj 是完整的物体字典，包含 'class', 'spatial', 'params'
        float_tol: 判断物体 params 全对时，浮点数比较的容差
    
    Returns:
        汇总的评估结果
    """
    total_stats = ParamsStats()
    per_class_stats = defaultdict(ParamsStats)
    
    for pred_obj, gt_obj in matched_pairs:
        pred_params = pred_obj.get('params', {})
        gt_params = gt_obj.get('params', {})
        obj_class = gt_obj.get('class', 'unknown')
        
        # 计算这对物体的 params 统计
        obj_stats = compare_params(pred_params, gt_params)
        
        # 统计物体 params 全对
        obj_stats.total_objects = 1
        if is_params_perfect(pred_params, gt_params, float_tol):
            obj_stats.perfect_objects = 1
        
        # 合并到总统计
        total_stats = total_stats.merge(obj_stats)
        
        # 合并到类别统计
        per_class_stats[obj_class] = per_class_stats[obj_class].merge(obj_stats)
    
    # 生成结果
    result = {
        'total': total_stats.to_summary(),
        'per_class': {cls: stats.to_summary() for cls, stats in per_class_stats.items()},
        'num_matched_pairs': len(matched_pairs),
    }
    
    return result


def print_params_evaluation(result: Dict[str, Any], verbose: bool = True) -> str:
    """
    格式化输出 params 评估结果
    
    Args:
        result: evaluate_matched_objects_params 的返回结果
        verbose: 是否输出每个类别的详细信息
    
    Returns:
        格式化的字符串
    """
    lines = []
    lines.append("=" * 60)
    lines.append("Params 准确性评估结果")
    lines.append("=" * 60)
    lines.append(f"匹配物体对数: {result['num_matched_pairs']}")
    lines.append("")
    
    total = result['total']
    
    # Key 匹配率
    key_stats = total['key_stats']
    lines.append(f"1. Key 匹配率: {total['key_accuracy']:.4f}")
    lines.append(f"   GT Keys: {key_stats['total_gt_keys']}, 匹配: {key_stats['matched_keys']}, Pred 多余: {key_stats['extra_pred_keys']}")
    lines.append("")
    
    # 物体 params 全对率（只检查 key 是否齐全）
    obj_perfect_stats = total['object_perfect_stats']
    lines.append(f"2. 物体 Params Key 全对率: {obj_perfect_stats['perfect_rate']:.4f} ({obj_perfect_stats['perfect_rate']*100:.2f}%)")
    lines.append(f"   （判断标准：GT 中的所有 key 是否都存在于 pred 中）")
    lines.append(f"   总物体数: {obj_perfect_stats['total_objects']}, Key 全对物体数: {obj_perfect_stats['perfect_objects']}")
    lines.append("")
    
    # 浮点数
    float_stats = total['float_stats']
    lines.append(f"3. 浮点数 (共 {float_stats['count']} 个)")
    lines.append(f"   平均绝对误差: {float_stats['mean_abs_error']:.6f}")
    lines.append(f"   平均相对误差: {float_stats['mean_rel_error']:.4f} ({float_stats['mean_rel_error']*100:.2f}%)")
    lines.append("")
    
    # 整数
    int_stats = total['int_stats']
    lines.append(f"4. 整数 (共 {int_stats['count']} 个)")
    lines.append(f"   平均绝对误差: {int_stats['mean_abs_error']:.4f}")
    if int_stats['gt_min'] is not None:
        lines.append(f"   GT 最小值: {int_stats['gt_min']}")
        lines.append(f"   GT 最大值: {int_stats['gt_max']}")
    lines.append("")
    
    # 布尔
    bool_stats = total['bool_stats']
    lines.append(f"5. 布尔值 (共 {bool_stats['count']} 个)")
    lines.append(f"   正确率: {bool_stats['accuracy']:.4f} ({bool_stats['accuracy']*100:.2f}%)")
    lines.append("")
    
    # 字符串
    str_stats = total['str_stats']
    lines.append(f"6. 字符串 (共 {str_stats['count']} 个)")
    lines.append(f"   正确率: {str_stats['accuracy']:.4f} ({str_stats['accuracy']*100:.2f}%)")
    lines.append("")
    
    # thickness 浮点数
    thickness_stats = total['thickness_stats']
    lines.append(f"7. Thickness 浮点数 (共 {thickness_stats['count']} 个)")
    lines.append(f"   平均绝对误差: {thickness_stats['mean_abs_error']:.6f}")
    lines.append(f"   平均相对误差: {thickness_stats['mean_rel_error']:.4f} ({thickness_stats['mean_rel_error']*100:.2f}%)")
    if thickness_stats['gt_min'] is not None:
        lines.append(f"   GT 最小值: {thickness_stats['gt_min']:.6f}")
        lines.append(f"   GT 最大值: {thickness_stats['gt_max']:.6f}")
    lines.append("")
    
    # 按类别的详细信息
    if verbose and result.get('per_class'):
        lines.append("-" * 60)
        lines.append("按物体类别统计:")
        lines.append("-" * 60)
        
        for cls, cls_result in sorted(result['per_class'].items()):
            lines.append(f"\n[{cls}]")
            lines.append(f"  Key 匹配率: {cls_result['key_accuracy']:.4f}")
            lines.append(f"  Params 全对率: {cls_result['object_perfect_stats']['perfect_rate']:.4f} (n={cls_result['object_perfect_stats']['total_objects']})")
            lines.append(f"  浮点数误差: {cls_result['float_stats']['mean_abs_error']:.6f} (n={cls_result['float_stats']['count']})")
            lines.append(f"  整数误差: {cls_result['int_stats']['mean_abs_error']:.4f} (n={cls_result['int_stats']['count']})")
            lines.append(f"  布尔正确率: {cls_result['bool_stats']['accuracy']:.4f} (n={cls_result['bool_stats']['count']})")
            lines.append(f"  字符串正确率: {cls_result['str_stats']['accuracy']:.4f} (n={cls_result['str_stats']['count']})")
            lines.append(f"  Thickness误差: {cls_result['thickness_stats']['mean_abs_error']:.6f} (n={cls_result['thickness_stats']['count']})")
    
    lines.append("")
    lines.append("=" * 60)
    
    output = "\n".join(lines)
    print(output)
    return output


# ============================================================
# 测试代码
# ============================================================
if __name__ == '__main__':
    # 简单测试
    gt_params = {
        "width": 0.661,
        "depth": 0.246,
        "has_extrude": True,
        "bathtub_type": "freestanding",
        "levels": 5,
        "disp_x": [0.017, 0.017],
        "nested": {
            "inner_value": 0.5,
            "inner_bool": False
        }
    }
    
    pred_params = {
        "width": 0.65,  # 浮点数误差
        "depth": 0.25,
        "has_extrude": True,  # 布尔正确
        "bathtub_type": "freestanding",  # 字符串正确
        "levels": 4,  # 整数误差 1
        "disp_x": [0.02, 0.015],  # 列表元素误差
        "nested": {
            "inner_value": 0.6,
            "inner_bool": True  # 布尔错误
        },
        "extra_key": 123  # 多余的 key
    }
    
    stats = compare_params(pred_params, gt_params)
    print("测试结果:")
    summary = stats.to_summary()
    for k, v in summary.items():
        print(f"  {k}: {v}")
