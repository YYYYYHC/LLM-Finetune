#!/bin/bash
# Script to filter dataset by sequence length

# Configuration
INPUT_DIR="./data/json_304k_qwen3_8b_tokenization"
OUTPUT_DIR="./data/json_304k_qwen3_8b_filtered_6K"
MIN_LENGTH=0  # Minimum sequence length (inclusive)
MAX_LENGTH=6144  # Maximum sequence length (inclusive)
SEED=42
NUM_PROC=8  # Number of processes for parallel processing

# Batch mode options (for large datasets or systems with limited memory)
# In batch mode, each batch is saved independently.
# Uncomment the following lines to enable batch mode:
BATCH_MODE="--batch_mode"
BATCH_SIZE=5000  # Number of samples per batch

echo "Filtering dataset by sequence length..."
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Min length: $MIN_LENGTH"
echo "Max length: $MAX_LENGTH"
echo "Number of processes: $NUM_PROC"

if [ -n "$BATCH_MODE" ]; then
    echo "Mode: BATCH MODE (memory efficient)"
    echo "Batch size: $BATCH_SIZE samples per batch"
    python -m src.data.filter_by_length \
        --input_dir "$INPUT_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --min_length $MIN_LENGTH \
        --max_length $MAX_LENGTH \
        --seed $SEED \
        --num_proc $NUM_PROC \
        --batch_mode \
        --batch_size $BATCH_SIZE
else
    echo "Mode: SIMPLE MODE (one-time processing, faster but requires more memory)"
    python -m src.data.filter_by_length \
        --input_dir "$INPUT_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --min_length $MIN_LENGTH \
        --max_length $MAX_LENGTH \
        --seed $SEED \
        --num_proc $NUM_PROC
fi

echo "Dataset filtering completed!"
echo "Filtered dataset saved to: $OUTPUT_DIR"

