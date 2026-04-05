#!/usr/bin/env python3
"""
统计JSON文件中房间数量的脚本
"""
import json
import os
from collections import Counter
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def count_rooms_in_file(file_path):
    """统计单个JSON文件中的房间数量"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 获取房间信息
        if 'blueprint' in data and 'rooms' in data['blueprint']:
            rooms = data['blueprint']['rooms']
            if isinstance(rooms, dict):
                return len(rooms)
            return 0
        return 0
    except Exception:
        return None


def process_file(file_path):
    """任务函数，返回文件路径及房间数量"""
    room_count = count_rooms_in_file(file_path)
    return str(file_path), room_count

def main():
    data_dir = Path("/root/data/json/yaml_9_processed")
    
    if not data_dir.exists():
        print(f"错误: 目录 {data_dir} 不存在")
        return
    
    print("正在扫描JSON文件...")
    json_files = list(data_dir.glob("*.json"))
    total_json_files = len(json_files)
    print(f"找到 {total_json_files} 个JSON文件")
    print(f"使用 {cpu_count()} 个CPU核心进行并行处理...\n")
    
    # 使用多进程并行处理，并显示进度条
    processed_counts = []
    error_files = 0
    deleted_files = 0
    deletion_errors = 0

    with Pool(processes=cpu_count()) as pool:
        with tqdm(total=total_json_files, desc="处理文件", unit="文件", ncols=100) as pbar:
            for file_path, room_count in pool.imap(process_file, json_files):
                pbar.update(1)

                if room_count is None:
                    error_files += 1
                    continue

                processed_counts.append(room_count)

                if room_count > 1:
                    try:
                        Path(file_path).unlink()
                        deleted_files += 1
                    except Exception:
                        deletion_errors += 1
    
    # 统计结果
    counter = Counter(processed_counts)
    
    print("\n" + "="*60)
    print("统计结果")
    print("="*60)
    print(f"总文件数: {total_json_files}")
    print(f"成功处理: {len(processed_counts)}")
    print(f"处理失败: {error_files}")
    print(f"删除含多房间文件: {deleted_files}")
    if deletion_errors:
        print(f"删除失败: {deletion_errors}")
    print("\n房间数量分布:")
    print("-"*60)
    
    # 按房间数量排序输出
    for room_num in sorted(counter.keys()):
        file_count = counter[room_num]
        percentage = (file_count / len(processed_counts) * 100) if processed_counts else 0
        print(f"{room_num} 个房间: {file_count:6d} 个文件 ({percentage:5.2f}%)")
    
    print("-"*60)
    print(f"总计: {len(processed_counts)} 个文件")
    print("="*60)

if __name__ == "__main__":
    main()

