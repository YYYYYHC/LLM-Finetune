#!/usr/bin/env python3
"""为已有的tar文件创建索引，加速随机访问。"""
import argparse
import pickle
import tarfile
from pathlib import Path
from tqdm import tqdm


def create_index(tar_path: Path) -> Path:
    """为单个tar文件创建索引。"""
    idx_path = tar_path.with_suffix(".tar.idx")
    index = {}
    with tarfile.open(tar_path, "r") as tar:
        for member in tar.getmembers():
            index[member.name] = (member.offset_data, member.size)
    with open(idx_path, "wb") as f:
        pickle.dump(index, f)
    return idx_path


def main():
    parser = argparse.ArgumentParser(description="为tar文件创建索引")
    parser.add_argument("path", help="tar文件或包含tar文件的目录")
    parser.add_argument("--recursive", "-r", action="store_true", help="递归处理子目录")
    args = parser.parse_args()

    path = Path(args.path)
    if path.is_file() and path.suffix == ".tar":
        tar_files = [path]
    elif path.is_dir():
        pattern = "**/*.tar" if args.recursive else "*.tar"
        tar_files = list(path.glob(pattern))
    else:
        print(f"错误: {path} 不是有效的tar文件或目录")
        return

    # 过滤已有索引的文件
    tar_files = [t for t in tar_files if not t.with_suffix(".tar.idx").exists()]
    
    if not tar_files:
        print("没有需要创建索引的tar文件")
        return

    print(f"找到 {len(tar_files)} 个需要创建索引的tar文件")
    for tar_path in tqdm(tar_files, desc="创建索引"):
        create_index(tar_path)
    print("索引创建完成")


if __name__ == "__main__":
    main()
