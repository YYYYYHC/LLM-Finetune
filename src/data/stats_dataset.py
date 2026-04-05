"""
统计tokenized数据集的详细信息，包括样本数量、token长度分布等。
"""

import argparse
import json
from pathlib import Path
import numpy as np
from datasets import load_from_disk
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import setup_logger

logger = setup_logger("stats_dataset")


def get_batch_dirs(data_dir: str) -> list:
    """
    获取所有batch目录。
    
    Args:
        data_dir: 数据目录路径
    
    Returns:
        batch目录列表，按序号排序
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")
    
    # 找到所有batch_*目录
    batch_dirs = sorted([d for d in data_path.iterdir() if d.is_dir() and d.name.startswith("batch_")])
    
    if not batch_dirs:
        raise ValueError(f"在 {data_dir} 中未找到batch目录")
    
    return batch_dirs


def analyze_batch(batch_dir: Path) -> dict:
    """
    分析单个batch的统计信息。
    
    Args:
        batch_dir: batch目录路径
    
    Returns:
        包含统计信息的字典
    """
    try:
        # 加载数据集
        dataset = load_from_disk(str(batch_dir))
        
        # 获取所有input_ids的长度
        lengths = [len(example) for example in dataset["input_ids"]]
        
        stats = {
            "batch_name": batch_dir.name,
            "num_samples": len(dataset),
            "token_stats": {
                "min_length": int(np.min(lengths)),
                "max_length": int(np.max(lengths)),
                "mean_length": float(np.mean(lengths)),
                "median_length": float(np.median(lengths)),
                "std_length": float(np.std(lengths)),
                "total_tokens": int(np.sum(lengths))
            },
            "percentiles": {
                "p25": float(np.percentile(lengths, 25)),
                "p50": float(np.percentile(lengths, 50)),
                "p75": float(np.percentile(lengths, 75)),
                "p90": float(np.percentile(lengths, 90)),
                "p95": float(np.percentile(lengths, 95)),
                "p99": float(np.percentile(lengths, 99))
            }
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"处理 {batch_dir.name} 时出错: {e}")
        return None


def print_batch_stats(stats: dict):
    """
    打印单个batch的统计信息。
    
    Args:
        stats: 统计信息字典
    """
    print(f"\n{'='*60}")
    print(f"Batch: {stats['batch_name']}")
    print(f"{'='*60}")
    print(f"样本数量: {stats['num_samples']:,}")
    print(f"总Token数: {stats['token_stats']['total_tokens']:,}")
    print(f"\nToken长度统计:")
    print(f"  最小值: {stats['token_stats']['min_length']:,}")
    print(f"  最大值: {stats['token_stats']['max_length']:,}")
    print(f"  平均值: {stats['token_stats']['mean_length']:,.2f}")
    print(f"  中位数: {stats['token_stats']['median_length']:,.2f}")
    print(f"  标准差: {stats['token_stats']['std_length']:,.2f}")
    print(f"\n百分位数:")
    print(f"  P25: {stats['percentiles']['p25']:,.0f}")
    print(f"  P50: {stats['percentiles']['p50']:,.0f}")
    print(f"  P75: {stats['percentiles']['p75']:,.0f}")
    print(f"  P90: {stats['percentiles']['p90']:,.0f}")
    print(f"  P95: {stats['percentiles']['p95']:,.0f}")
    print(f"  P99: {stats['percentiles']['p99']:,.0f}")


def print_summary_stats(all_stats: list):
    """
    打印所有batch的汇总统计信息。
    
    Args:
        all_stats: 所有batch的统计信息列表
    """
    total_samples = sum(s['num_samples'] for s in all_stats)
    total_tokens = sum(s['token_stats']['total_tokens'] for s in all_stats)
    
    all_min = min(s['token_stats']['min_length'] for s in all_stats)
    all_max = max(s['token_stats']['max_length'] for s in all_stats)
    
    # 加权平均
    weighted_mean = total_tokens / total_samples
    
    print(f"\n{'='*60}")
    print(f"总体统计信息")
    print(f"{'='*60}")
    print(f"Batch总数: {len(all_stats)}")
    print(f"样本总数: {total_samples:,}")
    print(f"Token总数: {total_tokens:,}")
    print(f"\n全局Token长度统计:")
    print(f"  全局最小值: {all_min:,}")
    print(f"  全局最大值: {all_max:,}")
    print(f"  加权平均值: {weighted_mean:,.2f}")
    
    print(f"\n各Batch样本数量分布:")
    sample_counts = [s['num_samples'] for s in all_stats]
    print(f"  最小值: {min(sample_counts):,}")
    print(f"  最大值: {max(sample_counts):,}")
    print(f"  平均值: {np.mean(sample_counts):,.2f}")
    print(f"  标准差: {np.std(sample_counts):,.2f}")


def save_stats_to_json(all_stats: list, output_file: str):
    """
    将统计信息保存到JSON文件。
    
    Args:
        all_stats: 所有batch的统计信息列表
        output_file: 输出文件路径
    """
    total_samples = sum(s['num_samples'] for s in all_stats)
    total_tokens = sum(s['token_stats']['total_tokens'] for s in all_stats)
    
    summary = {
        "total_batches": len(all_stats),
        "total_samples": total_samples,
        "total_tokens": total_tokens,
        "weighted_mean_length": total_tokens / total_samples,
        "global_min_length": min(s['token_stats']['min_length'] for s in all_stats),
        "global_max_length": max(s['token_stats']['max_length'] for s in all_stats),
        "batches": all_stats
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"统计信息已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="统计tokenized数据集的详细信息"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="数据集目录路径（包含batch_*子目录）"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="输出JSON文件路径（可选）"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="显示每个batch的详细统计信息"
    )
    
    args = parser.parse_args()
    
    # 获取所有batch目录
    logger.info(f"扫描数据目录: {args.data_dir}")
    batch_dirs = get_batch_dirs(args.data_dir)
    logger.info(f"找到 {len(batch_dirs)} 个batch")
    
    # 分析每个batch
    all_stats = []
    logger.info("开始分析各个batch...")
    
    for batch_dir in tqdm(batch_dirs, desc="分析batch"):
        stats = analyze_batch(batch_dir)
        if stats:
            all_stats.append(stats)
            if args.detailed:
                print_batch_stats(stats)
    
    if not all_stats:
        logger.error("未能成功分析任何batch")
        return
    
    # 打印汇总统计
    print_summary_stats(all_stats)
    
    # 保存到JSON文件
    if args.output_json:
        save_stats_to_json(all_stats, args.output_json)
    else:
        # 默认保存到数据目录下
        default_output = Path(args.data_dir) / "dataset_statistics.json"
        save_stats_to_json(all_stats, str(default_output))


if __name__ == "__main__":
    main()

