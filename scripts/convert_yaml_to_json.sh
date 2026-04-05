#!/bin/bash
# Script to convert YAML files to JSON format with optional float truncation and object renaming

# Configuration
INPUT_DIR="/root/data/original_yaml/yaml_9_processed"
OUTPUT_DIR="/root/data/json/yaml_9_processed"
NUM_WORKERS=8      # Adjust based on CPU cores
PRECISION=3        # Number of decimal places to keep (set to empty for no truncation)
                   # Example: 4 converts 0.01983328167140487 to 0.0198
RENAME_OBJECTS=true  # Set to true to rename object keys (e.g., "696816_BedFactory" -> "object_1")

echo "Converting YAML files to JSON..."
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Number of workers: $NUM_WORKERS"

# Build command arguments
CMD="python -m src.data.yaml_to_json --input_dir \"$INPUT_DIR\" --output_dir \"$OUTPUT_DIR\" --num_workers $NUM_WORKERS"

if [ -n "$PRECISION" ]; then
    echo "Float precision: $PRECISION decimal places"
    CMD="$CMD --precision $PRECISION"
else
    echo "Float precision: No truncation"
fi

if [ "$RENAME_OBJECTS" = true ]; then
    echo "Object renaming: Enabled"
    CMD="$CMD --rename_objects"
else
    echo "Object renaming: Disabled"
fi

# Execute the command
eval $CMD

echo "Conversion completed!"

