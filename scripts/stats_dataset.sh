#!/bin/bash
# Script to collect statistics from tokenized dataset

# Configuration
DATA_DIR="./data/json_304k_qwen3_8b_tokenization"
OUTPUT_JSON="./data/json_304k_qwen3_8b_tokenization/summary.json"  # Optional output JSON file path
DETAILED=""     # Show detailed stats for each batch

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output_json)
            OUTPUT_JSON="$2"
            shift 2
            ;;
        --detailed)
            DETAILED="--detailed"
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --data_dir PATH       数据集目录路径 (默认: ./data/json_304k_qwen3_8b_tokenization)"
            echo "  --output_json PATH    输出JSON文件路径 (可选)"
            echo "  --detailed            显示每个batch的详细统计信息"
            echo "  --help                显示此帮助信息"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Dataset Statistics Collection"
echo "=========================================="
echo "Dataset directory: $DATA_DIR"
if [ -n "$OUTPUT_JSON" ]; then
    echo "Output JSON: $OUTPUT_JSON"
fi
if [ -n "$DETAILED" ]; then
    echo "Mode: DETAILED (showing individual batch statistics)"
else
    echo "Mode: SUMMARY (showing only overall statistics)"
fi
echo ""

# Check if the directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Directory $DATA_DIR does not exist!"
    exit 1
fi

echo "Collecting statistics from batches..."
echo ""

# Build command with optional arguments
CMD="python -m src.data.stats_dataset --data_dir \"$DATA_DIR\""

if [ -n "$OUTPUT_JSON" ]; then
    CMD="$CMD --output_json \"$OUTPUT_JSON\""
fi

if [ -n "$DETAILED" ]; then
    CMD="$CMD --detailed"
fi

# Execute the command
eval $CMD

echo ""
echo "=========================================="
echo "Statistics collection completed!"
echo "=========================================="

