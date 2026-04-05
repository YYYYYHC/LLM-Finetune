#!/bin/bash
# Script to prepare dataset with tokenization

# Configuration
JSON_DIR="/root/zhenyang/datasets/GC_vlm_final/json/"
OUTPUT_DIR="/root/lutaojiang/data_clothes/GC_vlm_final_front"
MODEL_NAME="Qwen/Qwen3-VL-8B-Instruct"
MAX_LENGTH=16384  # Increased for long JSON unconditional generation
TEST_SPLIT=0.05  # 5% for evaluation
SEED=42
NUM_PROC=64  # Number of processes for parallel tokenization within each batch

# Batch mode options (for large datasets or systems with limited memory)
# In batch mode, each batch is saved independently. Use training config to specify which batches for train/eval.
# Uncomment the following lines to enable batch mode:
BATCH_MODE="--batch_mode"
BATCH_SIZE=5000  # Number of files per batch

# Parallel batch processing options (for multi-core servers)
# Set PARALLEL_BATCHES > 1 to process multiple batches simultaneously
# Total CPU usage ≈ PARALLEL_BATCHES × NUM_PROC cores
# Example: 48 parallel batches × 8 processes = 384 cores
PARALLEL_BATCHES=1  # Number of batches to process in parallel (set to 1 for sequential)

# Merge mode option (to append new data to existing dataset)
# Uncomment the following line to enable merge mode:
# MERGE_MODE="--merge_mode"

# Generation mode option
# Options: 
#   - "unconditional" (default, full scene generation)
#   - "blueprint" (blueprint -> objects)
#   - "caption" (text description -> JSON, requires CAPTION_DIR)
#   - "panorama" (panorama image -> JSON, requires IMAGE_DIR, for VL models)
#   - "multi_view" (orbit-based multi-view images -> JSON, requires IMAGE_DIR/orbit, for VL models)
#   - "multi_view2" (flat multi-view images -> JSON, requires IMAGE_DIR with PNGs directly under each scene folder)
#   - "multi_view3" (single front view -> JSON, requires IMAGE_DIR/<scene>/front.png)
# MODE="unconditional"
# To use blueprint mode:
# MODE="blueprint"
# To use caption mode (requires CAPTION_DIR):
# MODE="caption"
# CAPTION_DIR="/root/data/json/part8_145k_caption"
# To use panorama mode (requires IMAGE_DIR, for VL models like Qwen-VL):
# MODE="panorama"
# IMAGE_DIR="/root/lutaojiang/data/rendered_data_part6-8"
# To use multi_view mode (orbit layout, requires IMAGE_DIR/<scene>/orbit/orbit_*.png):
# MODE="multi_view"
# IMAGE_DIR="/root/data/rendered_data_part6-8"
# To use multi_view2 mode (flat layout, requires IMAGE_DIR/<scene>/*.png):
# MODE="multi_view2"
# IMAGE_DIR="/root/zhenyang/datasets/GC_vlm_final/image"
MODE="multi_view3"
IMAGE_DIR="/root/zhenyang/datasets/GC_vlm_final/image"

# Caption directory (required only for caption mode)
# Each .txt file should have the same basename as the corresponding .json file
# CAPTION_DIR="/Users/lutaojiang/Desktop/project/InfiniGen-NPR/data/demo_caption"

# Image directory (required only for panorama mode)
# Each subfolder should match JSON filename and contain panorama/panorama_rgb.png
# IMAGE_DIR="./data/demo_images"

# Image resolution for panorama mode (VL models)
# Format: WIDTHxHEIGHT, default is 1024x512
# This controls how images are processed by the VL model
IMAGE_RESOLUTION="1024x512"

# Multi-view mode options
# Number of views range (min_views, max_views)
NUM_VIEWS_MIN=1
NUM_VIEWS_MAX=10
# Image resolution for multi_view mode (fixed resolution)
# Format: WIDTHxHEIGHT, default is 512x512
# Set to empty string to use Python script's default (None)
MULTIVIEW_RESOLUTION="512x512"

# Dynamic resolution augmentation for multi_view mode
# When both MIN and MAX are set, enables random resolution sampling per view
# Each view will be independently sampled within [MIN, MAX] range, then center-cropped
# Format: WIDTHxHEIGHT (e.g., "512x512" and "1024x1024")
# Leave empty to disable dynamic resolution (use MULTIVIEW_RESOLUTION instead)
# MULTIVIEW_RESOLUTION_MIN="384x384"
# MULTIVIEW_RESOLUTION_MAX="768x768"

# Image format and quality options
# IMAGE_FORMAT: "png" (lossless, larger) or "jpeg" (lossy, smaller ~70-90% reduction)
# IMAGE_QUALITY: 1-100, only used for jpeg format (recommended: 85)
IMAGE_FORMAT="png"
IMAGE_QUALITY=95

# Packing option
# Packing option (reduces padding waste by packing multiple sequences together)
# Packed data includes 'sequence_lengths' field for attention isolation during training
PACKING="--packing"
# To disable packing, comment out the line above

echo "Preparing dataset for fine-tuning..."
echo "JSON directory: $JSON_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Model: $MODEL_NAME"
echo "Max length: $MAX_LENGTH"
echo "Test split: $TEST_SPLIT"
echo "Number of processes per batch: $NUM_PROC"
if [ -n "$BATCH_MODE" ] && [ "$PARALLEL_BATCHES" -gt 1 ]; then
    echo "Parallel batches: $PARALLEL_BATCHES"
    echo "Estimated total cores: $((PARALLEL_BATCHES * NUM_PROC))"
fi
if [ -n "$MERGE_MODE" ]; then
    echo "Merge mode: ENABLED (will append to existing dataset)"
else
    echo "Merge mode: DISABLED (will create new dataset)"
fi
echo "Generation mode: $MODE"
if [ "$MODE" == "caption" ]; then
    echo "Caption directory: $CAPTION_DIR"
fi
if [ "$MODE" == "panorama" ]; then
    echo "Image directory: $IMAGE_DIR"
    echo "Image resolution: $IMAGE_RESOLUTION"
fi
if [[ "$MODE" == "multi_view" || "$MODE" == "multi_view2" || "$MODE" == "multi_view3" ]]; then
    echo "Image directory: $IMAGE_DIR"
    if [ "$MODE" == "multi_view3" ]; then
        echo "Multi-view layout: FRONT-ONLY ({scene}/front.png)"
    elif [ "$MODE" == "multi_view2" ]; then
        echo "Multi-view layout: FLAT ({scene}/*.png)"
    else
        echo "Multi-view layout: ORBIT ({scene}/orbit/orbit_*.png)"
    fi
    echo "Number of views range: $NUM_VIEWS_MIN - $NUM_VIEWS_MAX"
    if [ -n "$MULTIVIEW_RESOLUTION_MIN" ] && [ -n "$MULTIVIEW_RESOLUTION_MAX" ]; then
        echo "Dynamic resolution: $MULTIVIEW_RESOLUTION_MIN ~ $MULTIVIEW_RESOLUTION_MAX (per-view random sampling)"
    elif [ -n "$MULTIVIEW_RESOLUTION" ]; then
        echo "Fixed resolution: $MULTIVIEW_RESOLUTION"
    else
        echo "Resolution: original (no resize)"
    fi
    echo "Image format: $IMAGE_FORMAT (quality: $IMAGE_QUALITY)"
fi
if [ -n "$PACKING" ]; then
    echo "Packing: ENABLED (sequences will be packed for efficiency)"
else
    echo "Packing: DISABLED"
fi

# Build caption_dir argument if in caption mode
CAPTION_ARG=""
if [ "$MODE" == "caption" ]; then
    if [ -z "$CAPTION_DIR" ]; then
        echo "Error: CAPTION_DIR is required for caption mode"
        exit 1
    fi
    CAPTION_ARG="--caption_dir $CAPTION_DIR"
fi

# Build image_dir argument if in panorama mode
IMAGE_ARG=""
RESOLUTION_ARG=""
if [ "$MODE" == "panorama" ]; then
    if [ -z "$IMAGE_DIR" ]; then
        echo "Error: IMAGE_DIR is required for panorama mode"
        exit 1
    fi
    IMAGE_ARG="--image_dir $IMAGE_DIR"
    RESOLUTION_ARG="--image_resolution $IMAGE_RESOLUTION"
fi

# Build arguments for multi_view mode
MULTIVIEW_ARG=""
if [[ "$MODE" == "multi_view" || "$MODE" == "multi_view2" || "$MODE" == "multi_view3" ]]; then
    if [ -z "$IMAGE_DIR" ]; then
        echo "Error: IMAGE_DIR is required for multi-view modes"
        exit 1
    fi
    IMAGE_ARG="--image_dir $IMAGE_DIR"
    if [ "$MODE" != "multi_view3" ]; then
        MULTIVIEW_ARG="--num_views_min $NUM_VIEWS_MIN --num_views_max $NUM_VIEWS_MAX"
    fi
    # Dynamic resolution (per-view random sampling) takes priority over fixed resolution
    if [ -n "$MULTIVIEW_RESOLUTION_MIN" ] && [ -n "$MULTIVIEW_RESOLUTION_MAX" ]; then
        MULTIVIEW_ARG="$MULTIVIEW_ARG --multiview_resolution_min $MULTIVIEW_RESOLUTION_MIN --multiview_resolution_max $MULTIVIEW_RESOLUTION_MAX"
    elif [ -n "$MULTIVIEW_RESOLUTION" ]; then
        MULTIVIEW_ARG="$MULTIVIEW_ARG --multiview_resolution $MULTIVIEW_RESOLUTION"
    fi
    # Image format and quality
    MULTIVIEW_ARG="$MULTIVIEW_ARG --image_format $IMAGE_FORMAT --image_quality $IMAGE_QUALITY"
fi

if [ -n "$BATCH_MODE" ]; then
    echo "Mode: BATCH MODE (memory efficient)"
    echo "Batch size: $BATCH_SIZE files per batch"
    
    if [ "$PARALLEL_BATCHES" -gt 1 ]; then
        echo "Parallel batches: $PARALLEL_BATCHES (total cores ≈ $((PARALLEL_BATCHES * NUM_PROC)))"
        
        # Calculate total number of batches
        TOTAL_FILES=$(find "$JSON_DIR" -name "*.json" | wc -l | tr -d ' ')
        TOTAL_BATCHES=$(( (TOTAL_FILES + BATCH_SIZE - 1) / BATCH_SIZE ))
        echo "Total files: $TOTAL_FILES"
        echo "Total batches: $TOTAL_BATCHES"
        
        # Create output directory
        mkdir -p "$OUTPUT_DIR"
        
        # Build the base command (without batch_index)
        BASE_CMD="python -m src.data.prepare_dataset \
            --json_dir \"$JSON_DIR\" \
            --output_dir \"$OUTPUT_DIR\" \
            --model_name \"$MODEL_NAME\" \
            --max_length $MAX_LENGTH \
            --test_split $TEST_SPLIT \
            --seed $SEED \
            --num_proc $NUM_PROC \
            --batch_mode \
            --batch_size $BATCH_SIZE \
            --total_batches $TOTAL_BATCHES \
            --mode \"$MODE\" \\
            $MERGE_MODE \\
            $PACKING \\
            $CAPTION_ARG \\
            $IMAGE_ARG \\
            $RESOLUTION_ARG \\
            $MULTIVIEW_ARG"
        # Run batches in parallel using GNU parallel or xargs
        if command -v parallel &> /dev/null; then
            echo "Using GNU parallel for batch-level parallelism"
            seq 0 $((TOTAL_BATCHES - 1)) | parallel -j $PARALLEL_BATCHES \
                "echo '[Batch {}] Starting...'; $BASE_CMD --batch_index {}; echo '[Batch {}] Completed'"
        else
            echo "GNU parallel not found, using xargs"
            seq 0 $((TOTAL_BATCHES - 1)) | xargs -P $PARALLEL_BATCHES -I {} \
                sh -c "echo '[Batch {}] Starting...'; $BASE_CMD --batch_index {}; echo '[Batch {}] Completed'"
        fi
    else
        # Sequential batch processing
        python -m src.data.prepare_dataset \
            --json_dir "$JSON_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --model_name "$MODEL_NAME" \
            --max_length $MAX_LENGTH \
            --test_split $TEST_SPLIT \
            --seed $SEED \
            --num_proc $NUM_PROC \
            --batch_mode \
            --batch_size $BATCH_SIZE \
            --mode "$MODE" \
            $MERGE_MODE \
            $PACKING \
            $CAPTION_ARG \
            $IMAGE_ARG \
            $RESOLUTION_ARG \
            $MULTIVIEW_ARG
    fi
else
    echo "Mode: SIMPLE MODE (one-time processing, faster but requires more memory)"
    python -m src.data.prepare_dataset \
        --json_dir "$JSON_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --model_name "$MODEL_NAME" \
        --max_length $MAX_LENGTH \
        --test_split $TEST_SPLIT \
        --seed $SEED \
        --num_proc $NUM_PROC \
        --mode "$MODE" \
        $MERGE_MODE \
        $PACKING \
        $CAPTION_ARG \
        $IMAGE_ARG \
        $RESOLUTION_ARG \
        $MULTIVIEW_ARG
fi

echo "Dataset preparation completed!"
echo "Dataset saved to: $OUTPUT_DIR"

