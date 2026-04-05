#!/usr/bin/env python3
"""
根据JSON文件内容匹配找到原始文件

这个脚本将ground_truth目录下的JSON文件与源数据目录进行内容匹配，
找到对应的原始文件名，并复制/链接到输出目录。

使用方式:
    python scripts/find_original_json.py \
        --input_dir /path/to/ground_truth \
        --source_dir /path/to/source_data \
        --output_dir ./outputs/temp

示例:
    python scripts/find_original_json.py \
        --input_dir /root/lutaojiang/code/LLM_Finetuning/outputs/condition_m/generated_scenes/8BInstruct_part1-8hd_full_vision_bs256Ktokens_lr5e-5/checkpoint-3000/20260103_153736/ground_truth \
        --source_dir /root/lutaojiang/data/filtered/part6_high_density_27k \
        --output_dir ./outputs/temp
"""

import argparse
import json
import os
import sys
import shutil
import hashlib
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def compute_content_hash(file_path: Path) -> str:
    """计算JSON文件内容的哈希值（规范化后）"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 规范化JSON后计算哈希
        normalized = json.dumps(data, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(normalized.encode()).hexdigest()
    except Exception as e:
        return None


def process_source_file(file_path: str) -> Tuple[str, str]:
    """处理单个源文件，返回 (hash, filename)"""
    path = Path(file_path)
    hash_val = compute_content_hash(path)
    return (hash_val, path.name) if hash_val else (None, None)


def get_cache_path(source_dir: Path) -> Path:
    """获取缓存文件路径"""
    # 基于源目录名生成缓存文件名
    cache_name = f".index_cache_{hashlib.md5(str(source_dir).encode()).hexdigest()[:8]}.pkl"
    return source_dir / cache_name


def load_cached_index(source_dir: Path) -> Optional[Dict[str, str]]:
    """加载缓存的索引"""
    cache_path = get_cache_path(source_dir)
    if cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            # 验证缓存有效性（检查文件数量）
            json_files = list(source_dir.glob("*.json"))
            if cache_data.get('file_count') == len(json_files):
                print(f"使用缓存索引 (共 {len(cache_data['index'])} 个文件)")
                return cache_data['index']
            else:
                print("缓存已过期，重新构建索引...")
        except Exception as e:
            print(f"无法加载缓存: {e}")
    return None


def save_cached_index(source_dir: Path, index: Dict[str, str], file_count: int):
    """保存索引到缓存"""
    cache_path = get_cache_path(source_dir)
    try:
        cache_data = {
            'index': index,
            'file_count': file_count
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"索引已缓存到: {cache_path}")
    except Exception as e:
        print(f"警告: 无法保存缓存: {e}")


def build_source_index(source_dir: Path, num_workers: int = 8, use_cache: bool = True) -> Dict[str, str]:
    """
    构建源目录的内容哈希索引
    返回: {content_hash: original_filename}
    """
    # 尝试从缓存加载
    if use_cache:
        cached_index = load_cached_index(source_dir)
        if cached_index:
            return cached_index
    
    print(f"正在扫描源目录: {source_dir}")
    sys.stdout.flush()
    
    source_files = list(source_dir.glob("*.json"))
    print(f"找到 {len(source_files)} 个JSON文件")
    sys.stdout.flush()
    
    if not source_files:
        return {}
    
    # 使用多进程加速
    index = {}
    
    print("正在构建内容索引 (首次运行较慢，后续使用缓存)...")
    sys.stdout.flush()
    
    # 转换为字符串列表以便传递给子进程
    source_file_paths = [str(f) for f in source_files]
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_source_file, f): f for f in source_file_paths}
        
        with tqdm(total=len(futures), desc="索引源文件", unit="文件", file=sys.stdout) as pbar:
            for future in as_completed(futures):
                hash_val, filename = future.result()
                if hash_val and filename:
                    if hash_val in index:
                        # 静默处理重复内容
                        pass
                    index[hash_val] = filename
                pbar.update(1)
                pbar.set_postfix({"已索引": len(index)})
    
    print(f"索引构建完成，共 {len(index)} 个唯一文件")
    sys.stdout.flush()
    
    # 保存缓存
    if use_cache:
        save_cached_index(source_dir, index, len(source_files))
    
    return index


def find_and_copy_files(
    input_dir: Path,
    source_dir: Path,
    output_dir: Path,
    mode: str = "copy",
    num_workers: int = 8,
    use_cache: bool = True
) -> Dict[str, str]:
    """
    查找并复制/链接匹配的文件
    
    Args:
        input_dir: ground_truth目录
        source_dir: 源数据目录
        output_dir: 输出目录
        mode: "copy" 复制文件, "link" 创建符号链接, "hardlink" 创建硬链接
        num_workers: 并行工作进程数
        use_cache: 是否使用缓存
    
    Returns:
        映射关系: {input_filename: original_filename}
    """
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 构建源文件索引
    source_index = build_source_index(source_dir, num_workers, use_cache)
    
    if not source_index:
        print("错误: 源目录索引为空")
        return {}
    
    # 查找输入文件
    input_files = list(input_dir.glob("*.json"))
    print(f"\n找到 {len(input_files)} 个待匹配的JSON文件")
    sys.stdout.flush()
    
    if not input_files:
        print("警告: 输入目录中没有JSON文件")
        return {}
    
    # 匹配并复制文件
    mapping = {}
    matched = 0
    not_matched = 0
    not_matched_files = []
    
    print("\n正在匹配文件...")
    sys.stdout.flush()
    
    # 第一步：计算哈希并匹配
    hash_results = []  # [(input_file, input_hash), ...]
    
    print("步骤 1/2: 计算输入文件哈希...")
    sys.stdout.flush()
    
    with tqdm(total=len(input_files), desc="计算哈希", unit="文件", file=sys.stdout) as pbar:
        for input_file in input_files:
            input_hash = compute_content_hash(input_file)
            hash_results.append((input_file, input_hash))
            pbar.update(1)
    
    # 第二步：匹配并复制/链接文件
    print("步骤 2/2: 匹配并处理文件...")
    sys.stdout.flush()
    
    with tqdm(total=len(hash_results), desc="匹配并复制", unit="文件", file=sys.stdout) as pbar:
        for input_file, input_hash in hash_results:
            if input_hash and input_hash in source_index:
                original_name = source_index[input_hash]
                mapping[input_file.name] = original_name
                
                # 复制或链接文件
                source_path = source_dir / original_name
                output_path = output_dir / original_name
                
                try:
                    if output_path.exists():
                        output_path.unlink()
                    
                    if mode == "copy":
                        shutil.copy2(source_path, output_path)
                    elif mode == "link":
                        output_path.symlink_to(source_path.resolve())
                    elif mode == "hardlink":
                        os.link(source_path, output_path)
                    
                    matched += 1
                except Exception as e:
                    print(f"\n错误: 无法处理文件 {original_name}: {e}")
            else:
                not_matched += 1
                not_matched_files.append(input_file.name)
            
            pbar.update(1)
            pbar.set_postfix({"匹配": matched, "未匹配": not_matched})
    
    # 保存映射关系
    mapping_file = output_dir / "filename_mapping.json"
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    
    # 打印统计
    print(f"\n{'=' * 60}")
    print(f"统计结果:")
    print(f"  匹配成功: {matched}")
    print(f"  匹配失败: {not_matched}")
    print(f"  映射文件保存到: {mapping_file}")
    
    if not_matched_files and not_matched <= 10:
        print(f"\n未匹配的文件:")
        for f in not_matched_files:
            print(f"  - {f}")
    elif not_matched > 10:
        print(f"\n未匹配的文件 (显示前10个):")
        for f in not_matched_files[:10]:
            print(f"  - {f}")
        print(f"  ... 还有 {not_matched - 10} 个")
    
    print(f"{'=' * 60}")
    
    return mapping


def main():
    parser = argparse.ArgumentParser(
        description="根据JSON文件内容匹配找到原始文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 基本用法
    python scripts/find_original_json.py \\
        --input_dir /path/to/ground_truth \\
        --source_dir /path/to/source_data

    # 指定输出目录
    python scripts/find_original_json.py \\
        --input_dir /path/to/ground_truth \\
        --source_dir /path/to/source_data \\
        --output_dir ./my_output

    # 使用符号链接而非复制
    python scripts/find_original_json.py \\
        --input_dir /path/to/ground_truth \\
        --source_dir /path/to/source_data \\
        --mode link
    
    # 强制重新构建索引
    python scripts/find_original_json.py \\
        --input_dir /path/to/ground_truth \\
        --source_dir /path/to/source_data \\
        --no-cache
        """
    )
    
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="包含待匹配JSON文件的目录 (ground_truth目录)"
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="源数据目录 (包含原始命名的JSON文件)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/temp",
        help="输出目录 (默认: ./outputs/temp)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["copy", "link", "hardlink"],
        default="copy",
        help="文件处理模式: copy(复制), link(符号链接), hardlink(硬链接) (默认: copy)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="并行工作进程数 (默认: 8)"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="不使用缓存，强制重新构建索引"
    )
    
    args = parser.parse_args()
    
    # 验证路径
    input_dir = Path(args.input_dir)
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"错误: 输入目录不存在: {input_dir}")
        return 1
    
    if not source_dir.exists():
        print(f"错误: 源目录不存在: {source_dir}")
        return 1
    
    # 执行匹配
    print(f"{'=' * 60}")
    print(f"输入目录: {input_dir}")
    print(f"源数据目录: {source_dir}")
    print(f"输出目录: {output_dir}")
    print(f"处理模式: {args.mode}")
    print(f"工作进程数: {args.workers}")
    print(f"使用缓存: {not args.no_cache}")
    print(f"{'=' * 60}")
    sys.stdout.flush()
    
    mapping = find_and_copy_files(
        input_dir=input_dir,
        source_dir=source_dir,
        output_dir=output_dir,
        mode=args.mode,
        num_workers=args.workers,
        use_cache=not args.no_cache
    )
    
    if mapping:
        print(f"\n✅ 完成! 共匹配 {len(mapping)} 个文件")
        print(f"📁 输出目录: {output_dir}")
    else:
        print("\n❌ 没有找到任何匹配的文件")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
