#!/usr/bin/env python3
"""
脚本：读取 Arrow 文件并打印第一条数据的内容
用法：python inspect_arrow.py [arrow_file_path] [--tokenizer tokenizer_path] [--output output_file]
"""

from datasets import Dataset
from transformers import AutoTokenizer
import sys
import argparse
import os

def inspect_arrow(arrow_file: str, tokenizer_path: str = None, output_file: str = None):
    """读取并打印 Arrow 文件的第一条数据"""
    
    # 加载 tokenizer（如果提供了路径）
    tokenizer = None
    if tokenizer_path:
        print(f"加载 tokenizer: {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        print(f"Tokenizer 词表大小: {len(tokenizer)}\n")
    
    # 使用 datasets 读取
    ds = Dataset.from_file(arrow_file)

    # 打印基本信息
    print("=" * 60)
    print("Dataset 信息:")
    print("=" * 60)
    print(ds)
    print(f"\n字段列表: {ds.column_names}")

    # 打印第一条数据
    print("\n" + "=" * 60)
    print("第一条数据的内容:")
    print("=" * 60)

    first_row = ds[0]
    
    # 用于保存到文件的内容
    output_content = []

    for key, value in first_row.items():
        print(f"\n【{key}】:")
        if isinstance(value, (list, tuple)):
            print(f"  类型: {type(value).__name__}")
            print(f"  长度: {len(value)}")
            if len(value) > 100:
                print(f"  前50个元素: {value[:50]}")
                print(f"  后50个元素: {value[-50:]}")
            else:
                print(f"  全部元素: {value}")
            
            # 如果是 input_ids 或 labels，且提供了 tokenizer，则 decode 显示
            if tokenizer and key in ("input_ids", "labels"):
                print(f"\n  === Decoded {key} ===")
                # 对于 labels，-100 是忽略的 token，替换为特殊标记
                if key == "labels":
                    # 过滤掉 -100，只 decode 有效的 token
                    valid_tokens = [t for t in value if t != -100]
                    decoded_text = tokenizer.decode(valid_tokens, skip_special_tokens=False)
                    print(f"  (已过滤 {len(value) - len(valid_tokens)} 个 -100 token)")
                else:
                    decoded_text = tokenizer.decode(value, skip_special_tokens=False)
                
                # 保存完整的 decoded 内容到列表
                output_content.append(f"=== {key} (Decoded) ===\n")
                output_content.append(decoded_text)
                output_content.append("\n\n")
                
                # 打印 decoded 文本
                if len(decoded_text) > 2000:
                    print(f"  前1000字符:\n  {decoded_text[:1000]}")
                    print(f"\n  ...(省略中间部分)...\n")
                    print(f"  后1000字符:\n  {decoded_text[-1000:]}")
                else:
                    print(f"  {decoded_text}")
        else:
            print(f"  类型: {type(value).__name__}")
            print(f"  值: {value}")
    
    # 如果指定了输出文件，保存完整的 decoded 结果
    if output_file and output_content:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(output_content)
        print(f"\n完整的 decoded 结果已保存到: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="读取 Arrow 文件并打印第一条数据的内容")
    parser.add_argument("arrow_file", nargs="?", 
                        default="/data/workspace/lutaojiang/data/tokenized/blueprint_condition_high_density_part1-5_85k/batch_0000/data-00000-of-00001.arrow",
                        help="Arrow 文件路径")
    parser.add_argument("--tokenizer", "-t", type=str, default="Qwen/Qwen3-VL-8B-Instruct",
                        help="Tokenizer 路径，用于将 token id 转换为 string")
    parser.add_argument("--output", "-o", type=str, default="outputs/temp/decoded_output.txt",
                        help="输出文件路径，用于保存完整的 decoded 结果")
    
    args = parser.parse_args()
    
    print(f"读取文件: {args.arrow_file}\n")
    inspect_arrow(args.arrow_file, args.tokenizer, args.output)
