#!/usr/bin/env python3
"""
Simple script to test token lengths of random JSON files.
随机采样JSON文件测试token长度
"""

import json
import random
import argparse
from pathlib import Path
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Test token lengths")
    parser.add_argument("--json_dir", type=str, required=True, help="JSON directory")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B", help="Model name")
    parser.add_argument("--n_samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--max_length", type=int, default=32768, help="Max length to test")
    args = parser.parse_args()
    
    # Get all JSON files
    json_dir = Path(args.json_dir)
    json_files = list(json_dir.glob("*.json"))
    print(f"Found {len(json_files)} JSON files")
    
    # Random sample
    sample_files = random.sample(json_files, min(args.n_samples, len(json_files)))
    print(f"Testing {len(sample_files)} random samples...")
    
    # Load tokenizer
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    # Test each file
    lengths = []
    for i, json_file in enumerate(sample_files, 1):
        # Load JSON
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Convert to compact JSON string (same as training)
        json_str = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
        
        # Create conversation (same as training)
        messages = [
            {"role": "user", "content": "Generate a 3D scene in JSON format:"},
            {"role": "assistant", "content": json_str}
        ]
        
        # Apply chat template
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        
        # Tokenize
        tokens = tokenizer(text, return_tensors=None)
        length = len(tokens['input_ids'])
        lengths.append(length)
        
        # Progress
        if i % 10 == 0:
            print(f"  Processed {i}/{len(sample_files)}")
    
    # Statistics
    lengths.sort()
    print(f"\n{'='*60}")
    print("Token Length Statistics")
    print(f"{'='*60}")
    print(f"Samples:     {len(lengths)}")
    print(f"Min:         {min(lengths):,}")
    print(f"Max:         {max(lengths):,}")
    print(f"Mean:        {sum(lengths)/len(lengths):,.0f}")
    print(f"Median:      {lengths[len(lengths)//2]:,}")
    print(f"P95:         {lengths[int(len(lengths)*0.95)]:,}")
    print(f"P99:         {lengths[int(len(lengths)*0.99)]:,}")
    
    # Check fit
    print(f"\n{'='*60}")
    print(f"Fit Check (max_length={args.max_length:,})")
    print(f"{'='*60}")
    over = sum(1 for l in lengths if l > args.max_length)
    print(f"Within limit: {len(lengths) - over} ({(len(lengths)-over)/len(lengths)*100:.1f}%)")
    print(f"Over limit:   {over} ({over/len(lengths)*100:.1f}%)")
    
    if over > 0:
        print(f"\n⚠️  {over} samples exceed max_length={args.max_length:,}")
        print(f"   Consider increasing max_length or they will be truncated")
    else:
        print(f"\n✓ All samples fit within max_length={args.max_length:,}")
    
    # Distribution
    print(f"\n{'='*60}")
    print("Distribution")
    print(f"{'='*60}")
    bins = [0, 8192, 16384, 24576, 32768, 40960, 49152]
    for i in range(len(bins)-1):
        count = sum(1 for l in lengths if bins[i] < l <= bins[i+1])
        if count > 0:
            bar = '█' * int(count / len(lengths) * 50)
            print(f"{bins[i]:>6,}-{bins[i+1]:>6,}: {count:>4} {bar}")


if __name__ == "__main__":
    main()

