#!/usr/bin/env python3
"""
脚本：统计 Arrow 数据集 tokenize 后的序列长度
考虑 packing 情况：每条数据可能是多个 sequence 拼接而成
通过 sequence_lengths 字段获取每个原始 sequence 的长度

用法：python stats_arrow_length.py [data_dir]
"""

import os
import sys
import argparse
import glob
import numpy as np
from tqdm import tqdm
from datasets import Dataset


def stats_arrow_lengths(data_dir: str, verbose: bool = False):
    """统计目录下所有 Arrow 文件的序列长度"""
    
    # 查找所有 arrow 文件
    arrow_files = sorted(glob.glob(os.path.join(data_dir, "batch_*/data-*.arrow")))
    
    if not arrow_files:
        print(f"错误：在 {data_dir} 下没有找到 arrow 文件")
        return
    
    print(f"找到 {len(arrow_files)} 个 arrow 文件", flush=True)
    print("=" * 60, flush=True)
    
    # 统计数据
    all_sequence_lengths = []  # 所有原始 sequence 的长度
    total_packed_rows = 0      # packed 后的总行数
    sequences_per_row = []     # 每行包含的 sequence 数量
    
    for arrow_file in tqdm(arrow_files, desc="处理文件"):
        try:
            # 使用 datasets.Dataset.from_file 读取
            ds = Dataset.from_file(arrow_file)
            
            if verbose:
                print(f"\n处理: {arrow_file}", flush=True)
                print(f"  行数: {len(ds)}", flush=True)
                print(f"  字段: {ds.column_names}", flush=True)
            
            total_packed_rows += len(ds)
            
            # 使用向量化操作，直接获取整列数据（比逐行遍历快很多）
            if "sequence_lengths" in ds.column_names:
                # 获取整列 sequence_lengths
                seq_lengths_col = ds["sequence_lengths"]
                for seq_lengths in seq_lengths_col:
                    if seq_lengths:
                        all_sequence_lengths.extend(seq_lengths)
                        sequences_per_row.append(len(seq_lengths))
                    else:
                        sequences_per_row.append(0)
            else:
                # 没有 sequence_lengths 字段，按整行统计
                if "input_ids" in ds.column_names:
                    input_ids_col = ds["input_ids"]
                    for input_ids in input_ids_col:
                        length = len(input_ids)
                        all_sequence_lengths.append(length)
                        sequences_per_row.append(1)
                        
        except Exception as e:
            print(f"处理 {arrow_file} 时出错: {e}", flush=True)
            import traceback
            traceback.print_exc()
            continue
    
    # 计算统计结果
    if not all_sequence_lengths:
        print("没有找到有效的序列数据")
        return
    
    seq_lengths_array = np.array(all_sequence_lengths)
    sequences_per_row_array = np.array(sequences_per_row)
    total_sequences = len(all_sequence_lengths)
    total_tokens = int(np.sum(seq_lengths_array))
    
    print("\n" + "=" * 60, flush=True)
    print("统计结果", flush=True)
    print("=" * 60, flush=True)
    
    print("\n【基本信息】", flush=True)
    print(f"  数据目录: {data_dir}", flush=True)
    print(f"  Arrow 文件数: {len(arrow_files)}", flush=True)
    print(f"  Packed 行数 (训练样本数): {total_packed_rows:,}", flush=True)
    print(f"  原始 Sequence 总数: {total_sequences:,}", flush=True)
    print(f"  总 Token 数: {total_tokens:,}", flush=True)
    
    print("\n【原始 Sequence 长度统计】", flush=True)
    print(f"  平均长度: {np.mean(seq_lengths_array):,.2f}", flush=True)
    print(f"  中位数: {np.median(seq_lengths_array):,.2f}", flush=True)
    print(f"  标准差: {np.std(seq_lengths_array):,.2f}", flush=True)
    print(f"  最小值: {np.min(seq_lengths_array):,}", flush=True)
    print(f"  最大值: {np.max(seq_lengths_array):,}", flush=True)
    
    # 分位数
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print(f"\n  分位数分布:", flush=True)
    for p in percentiles:
        print(f"    P{p}: {np.percentile(seq_lengths_array, p):,.0f}", flush=True)
    
    print("\n【Packing 统计】", flush=True)
    print(f"  平均每行 Sequence 数: {np.mean(sequences_per_row_array):.2f}", flush=True)
    print(f"  每行 Sequence 数范围: {np.min(sequences_per_row_array)} - {np.max(sequences_per_row_array)}", flush=True)
    if total_packed_rows > 0:
        print(f"  Packing 压缩比: {total_sequences / total_packed_rows:.2f}x", flush=True)
    
    # 长度分布直方图
    print("\n【长度分布】", flush=True)
    bins = [0, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, float('inf')]
    bin_labels = ['0-256', '256-512', '512-1K', '1K-2K', '2K-4K', '4K-8K', '8K-16K', '16K-32K', '>32K']
    
    hist, _ = np.histogram(seq_lengths_array, bins=bins)
    total = len(seq_lengths_array)
    
    print(f"  {'长度区间':<12} {'数量':>10} {'占比':>10}", flush=True)
    print(f"  {'-'*12} {'-'*10} {'-'*10}", flush=True)
    for label, count in zip(bin_labels, hist):
        if count > 0:
            print(f"  {label:<12} {count:>10,} {count/total*100:>9.2f}%", flush=True)
    
    return {
        "total_packed_rows": total_packed_rows,
        "total_sequences": total_sequences,
        "total_tokens": total_tokens,
        "mean_length": float(np.mean(seq_lengths_array)),
        "median_length": float(np.median(seq_lengths_array)),
        "std_length": float(np.std(seq_lengths_array)),
        "min_length": int(np.min(seq_lengths_array)),
        "max_length": int(np.max(seq_lengths_array)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="统计 Arrow 数据集 tokenize 后的序列长度")
    parser.add_argument("data_dir", nargs="?",
                        default="/data/workspace/lutaojiang/data/tokenized/blueprint_condition_high_density_part1-5_85k",
                        help="包含 Arrow 文件的数据目录")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="显示详细信息")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"错误：目录不存在: {args.data_dir}")
        sys.exit(1)
    
    stats_arrow_lengths(args.data_dir, args.verbose)