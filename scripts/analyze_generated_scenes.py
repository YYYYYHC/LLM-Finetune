"""
分析生成的场景JSON文件

这个脚本分析生成的场景文件，提供统计信息和质量评估。

使用方式:
    python scripts/analyze_generated_scenes.py ./generated_scenes
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter
import statistics


def load_json_file(file_path: Path) -> Dict[str, Any]:
    """加载JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"警告: 无法加载文件 {file_path}: {e}")
        return None


def analyze_json_structure(scene: Dict[str, Any], depth: int = 0) -> Dict[str, Any]:
    """分析JSON结构"""
    if not isinstance(scene, dict):
        return {"type": type(scene).__name__, "depth": depth}
    
    structure = {
        "type": "dict",
        "depth": depth,
        "keys": list(scene.keys()),
        "num_keys": len(scene.keys()),
        "children": {}
    }
    
    for key, value in scene.items():
        if isinstance(value, dict):
            structure["children"][key] = analyze_json_structure(value, depth + 1)
        elif isinstance(value, list):
            structure["children"][key] = {
                "type": "list",
                "length": len(value),
                "depth": depth + 1
            }
        else:
            structure["children"][key] = {
                "type": type(value).__name__,
                "depth": depth + 1
            }
    
    return structure


def analyze_scenes(scenes_dir: Path) -> Dict[str, Any]:
    """分析所有场景文件"""
    
    # 查找所有场景文件
    scene_files = list(scenes_dir.glob("scene_*.json"))
    
    if not scene_files:
        print(f"警告: 在 {scenes_dir} 中未找到场景文件")
        return None
    
    print(f"找到 {len(scene_files)} 个场景文件")
    print("正在分析...\n")
    
    # 统计信息
    stats = {
        "total_files": len(scene_files),
        "valid_files": 0,
        "invalid_files": 0,
        "file_sizes": [],
        "json_lengths": [],
        "top_level_keys": [],
        "structures": [],
        "errors": []
    }
    
    # 分析每个文件
    for i, file_path in enumerate(scene_files, 1):
        # 文件大小
        file_size = file_path.stat().st_size
        stats["file_sizes"].append(file_size)
        
        # 加载JSON
        scene = load_json_file(file_path)
        
        if scene is None:
            stats["invalid_files"] += 1
            stats["errors"].append(str(file_path.name))
            continue
        
        stats["valid_files"] += 1
        
        # JSON长度
        json_str = json.dumps(scene, ensure_ascii=False)
        stats["json_lengths"].append(len(json_str))
        
        # 顶层键
        if isinstance(scene, dict):
            stats["top_level_keys"].extend(scene.keys())
            
            # 结构分析（只分析前100个，避免太慢）
            if i <= 100:
                structure = analyze_json_structure(scene)
                stats["structures"].append(structure)
    
    # 计算统计
    if stats["file_sizes"]:
        stats["avg_file_size"] = statistics.mean(stats["file_sizes"])
        stats["min_file_size"] = min(stats["file_sizes"])
        stats["max_file_size"] = max(stats["file_sizes"])
    
    if stats["json_lengths"]:
        stats["avg_json_length"] = statistics.mean(stats["json_lengths"])
        stats["min_json_length"] = min(stats["json_lengths"])
        stats["max_json_length"] = max(stats["json_lengths"])
    
    # 统计最常见的键
    if stats["top_level_keys"]:
        key_counts = Counter(stats["top_level_keys"])
        stats["common_keys"] = key_counts.most_common(20)
    
    # 分析结构一致性
    if stats["structures"]:
        key_sets = [set(s["keys"]) for s in stats["structures"]]
        if len(key_sets) > 0:
            common_keys = set.intersection(*key_sets) if len(key_sets) > 1 else key_sets[0]
            stats["consistent_keys"] = list(common_keys)
            
            # 计算结构一致性得分
            all_keys = set.union(*key_sets) if len(key_sets) > 1 else key_sets[0]
            stats["structure_consistency"] = len(common_keys) / len(all_keys) if all_keys else 0
    
    return stats


def print_analysis(stats: Dict[str, Any]) -> None:
    """打印分析结果"""
    
    print("=" * 60)
    print("场景文件分析报告")
    print("=" * 60)
    print()
    
    # 基本统计
    print("📊 基本统计")
    print("-" * 60)
    print(f"总文件数: {stats['total_files']}")
    print(f"有效文件: {stats['valid_files']} ({stats['valid_files']/stats['total_files']*100:.1f}%)")
    print(f"无效文件: {stats['invalid_files']} ({stats['invalid_files']/stats['total_files']*100:.1f}%)")
    print()
    
    # 文件大小
    if "avg_file_size" in stats:
        print("📏 文件大小")
        print("-" * 60)
        print(f"平均大小: {stats['avg_file_size']/1024:.2f} KB")
        print(f"最小大小: {stats['min_file_size']/1024:.2f} KB")
        print(f"最大大小: {stats['max_file_size']/1024:.2f} KB")
        print()
    
    # JSON长度
    if "avg_json_length" in stats:
        print("📝 JSON长度")
        print("-" * 60)
        print(f"平均长度: {stats['avg_json_length']:.0f} 字符")
        print(f"最小长度: {stats['min_json_length']:.0f} 字符")
        print(f"最大长度: {stats['max_json_length']:.0f} 字符")
        print()
    
    # 常见键
    if "common_keys" in stats:
        print("🔑 最常见的顶层键")
        print("-" * 60)
        for key, count in stats["common_keys"][:10]:
            percentage = count / stats["valid_files"] * 100
            print(f"  {key:30s}: {count:5d} 次 ({percentage:5.1f}%)")
        print()
    
    # 结构一致性
    if "structure_consistency" in stats:
        print("🔍 结构一致性")
        print("-" * 60)
        consistency_score = stats["structure_consistency"] * 100
        print(f"一致性得分: {consistency_score:.1f}%")
        
        if "consistent_keys" in stats:
            print(f"所有文件共有的键 ({len(stats['consistent_keys'])} 个):")
            for key in sorted(stats["consistent_keys"]):
                print(f"  - {key}")
        print()
    
    # 错误
    if stats["errors"]:
        print("⚠️  无效文件")
        print("-" * 60)
        for error in stats["errors"][:10]:
            print(f"  - {error}")
        if len(stats["errors"]) > 10:
            print(f"  ... 还有 {len(stats['errors']) - 10} 个")
        print()
    
    # 质量评估
    print("✅ 质量评估")
    print("-" * 60)
    
    quality_score = 0
    max_score = 0
    
    # 有效率
    max_score += 30
    validity_rate = stats['valid_files'] / stats['total_files']
    quality_score += validity_rate * 30
    print(f"有效率: {validity_rate*100:.1f}% (权重: 30)")
    
    # 结构一致性
    if "structure_consistency" in stats:
        max_score += 30
        consistency = stats["structure_consistency"]
        quality_score += consistency * 30
        print(f"结构一致性: {consistency*100:.1f}% (权重: 30)")
    
    # 文件大小合理性（假设合理范围是1KB-100KB）
    if "avg_file_size" in stats:
        max_score += 20
        avg_size_kb = stats['avg_file_size'] / 1024
        if 1 <= avg_size_kb <= 100:
            size_score = 20
        elif avg_size_kb < 1:
            size_score = avg_size_kb * 20  # 太小
        else:
            size_score = 20 * (100 / avg_size_kb)  # 太大
        quality_score += size_score
        print(f"文件大小合理性: {size_score/20*100:.1f}% (权重: 20)")
    
    # 内容丰富度（基于键的数量）
    if "common_keys" in stats:
        max_score += 20
        num_unique_keys = len(stats["common_keys"])
        richness = min(num_unique_keys / 10, 1.0)  # 假设10个键是丰富的
        richness_score = richness * 20
        quality_score += richness_score
        print(f"内容丰富度: {richness*100:.1f}% (权重: 20)")
    
    print()
    print(f"总体质量得分: {quality_score:.1f}/{max_score:.1f} ({quality_score/max_score*100:.1f}%)")
    print()
    
    # 评级
    score_percentage = quality_score / max_score * 100
    if score_percentage >= 90:
        rating = "优秀 ⭐⭐⭐⭐⭐"
    elif score_percentage >= 80:
        rating = "良好 ⭐⭐⭐⭐"
    elif score_percentage >= 70:
        rating = "中等 ⭐⭐⭐"
    elif score_percentage >= 60:
        rating = "及格 ⭐⭐"
    else:
        rating = "需要改进 ⭐"
    
    print(f"质量评级: {rating}")
    print()
    
    print("=" * 60)


def save_analysis(stats: Dict[str, Any], output_path: Path) -> None:
    """保存分析结果到JSON"""
    # 准备可序列化的数据
    save_data = {
        "total_files": stats["total_files"],
        "valid_files": stats["valid_files"],
        "invalid_files": stats["invalid_files"],
        "avg_file_size": stats.get("avg_file_size"),
        "min_file_size": stats.get("min_file_size"),
        "max_file_size": stats.get("max_file_size"),
        "avg_json_length": stats.get("avg_json_length"),
        "min_json_length": stats.get("min_json_length"),
        "max_json_length": stats.get("max_json_length"),
        "common_keys": stats.get("common_keys"),
        "consistent_keys": stats.get("consistent_keys"),
        "structure_consistency": stats.get("structure_consistency"),
        "errors": stats.get("errors")
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    print(f"分析结果已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="分析生成的场景JSON文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 分析生成的场景目录
  python scripts/analyze_generated_scenes.py ./generated_scenes

  # 保存分析结果
  python scripts/analyze_generated_scenes.py ./generated_scenes --output analysis.json
        """
    )
    
    parser.add_argument(
        "scenes_dir",
        type=str,
        help="包含生成场景的目录"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="保存分析结果的JSON文件路径"
    )
    
    args = parser.parse_args()
    
    # 检查目录
    scenes_dir = Path(args.scenes_dir)
    if not scenes_dir.exists():
        print(f"错误: 目录不存在: {scenes_dir}")
        return
    
    if not scenes_dir.is_dir():
        print(f"错误: 不是一个目录: {scenes_dir}")
        return
    
    # 分析
    stats = analyze_scenes(scenes_dir)
    
    if stats is None:
        return
    
    # 打印结果
    print_analysis(stats)
    
    # 保存结果
    if args.output:
        output_path = Path(args.output)
        save_analysis(stats, output_path)
    
    # 建议
    print("\n💡 建议:")
    
    if stats["invalid_files"] > 0:
        print(f"  - 有 {stats['invalid_files']} 个无效文件，建议检查生成参数")
    
    if "structure_consistency" in stats and stats["structure_consistency"] < 0.8:
        print(f"  - 结构一致性较低 ({stats['structure_consistency']*100:.1f}%)，可能需要调整prompt")
    
    if "avg_file_size" in stats:
        avg_size_kb = stats["avg_file_size"] / 1024
        if avg_size_kb < 1:
            print(f"  - 平均文件大小较小 ({avg_size_kb:.2f} KB)，可能内容不够丰富")
        elif avg_size_kb > 100:
            print(f"  - 平均文件大小较大 ({avg_size_kb:.2f} KB)，可能包含冗余信息")
    
    print()


if __name__ == "__main__":
    main()

