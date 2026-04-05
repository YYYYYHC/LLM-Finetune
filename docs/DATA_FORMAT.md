# Data Format Guide

This document explains the data format and how to customize it for your needs.

## Input Data Format

### YAML Files

Your original data is in YAML format. Each YAML file contains scene description data.

Example structure:
```yaml
scene_id: "scene_001"
objects:
  - name: "cube"
    position: [0, 0, 0]
    rotation: [0, 0, 0]
    scale: [1, 1, 1]
  - name: "sphere"
    position: [2, 0, 0]
    rotation: [0, 0, 0]
    scale: [1, 1, 1]
lighting:
  type: "directional"
  intensity: 1.0
camera:
  position: [5, 5, 5]
  target: [0, 0, 0]
```

## Conversion Pipeline

### Stage 1: YAML → JSON

The first step converts YAML to JSON for better compatibility:

```bash
python -m src.data.yaml_to_json \
    --input_dir /path/to/yaml \
    --output_dir ./data/json
```

Output: JSON files with identical structure to YAML.

### Stage 2: JSON → Qwen Conversation Format

The second step converts JSON to Qwen's conversation format:

```python
{
    "messages": [
        {
            "role": "system",
            "content": "System prompt"
        },
        {
            "role": "user",
            "content": "User input"
        },
        {
            "role": "assistant",
            "content": "Assistant response"
        }
    ]
}
```

### Stage 3: Tokenization

Conversations are tokenized using Qwen's tokenizer:

```python
text = tokenizer.apply_chat_template(messages, tokenize=False)
encoded = tokenizer(text, max_length=2048, truncation=True)
```

Output: HuggingFace Dataset with tokenized data.

## Customizing Data Format

### Modify Conversation Format

Edit `src/data/prepare_dataset.py`, function `convert_to_qwen_format()`:

```python
def convert_to_qwen_format(
    data: List[Dict[str, Any]],
    instruction_key: str = "instruction",
    output_key: str = "output"
) -> List[Dict[str, str]]:
    """
    Customize this function to match your data structure.
    """
    formatted_data = []
    
    for item in data:
        # CUSTOMIZE THIS PART
        # Example: Extract specific fields from your YAML
        
        # Option 1: Instruction-following format
        instruction = f"Generate a scene with {len(item.get('objects', []))} objects"
        response = json.dumps(item, ensure_ascii=False, indent=2)
        
        conversation = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a 3D scene generation assistant."
                },
                {
                    "role": "user",
                    "content": instruction
                },
                {
                    "role": "assistant",
                    "content": response
                }
            ]
        }
        
        # Option 2: Description-to-structure format
        # description = item.get("description", "")
        # structure = json.dumps(item.get("structure", {}))
        # conversation = {...}
        
        # Option 3: Multi-turn conversation
        # Create multiple user-assistant pairs
        
        formatted_data.append(conversation)
    
    return formatted_data
```

### Example Customizations

#### 1. Scene Generation Task

Train the model to generate scene YAML from text descriptions:

```python
conversation = {
    "messages": [
        {
            "role": "system",
            "content": "You generate 3D scene descriptions in YAML format."
        },
        {
            "role": "user",
            "content": "Create a scene with a cube and a sphere"
        },
        {
            "role": "assistant",
            "content": yaml.dump(scene_data)  # Your YAML data
        }
    ]
}
```

#### 2. Scene Understanding Task

Train the model to understand and answer questions about scenes:

```python
scene_json = json.dumps(item, indent=2)

conversation = {
    "messages": [
        {
            "role": "system",
            "content": "You understand 3D scene descriptions."
        },
        {
            "role": "user",
            "content": f"Analyze this scene:\n{scene_json}\n\nHow many objects are there?"
        },
        {
            "role": "assistant",
            "content": f"There are {len(item['objects'])} objects in this scene."
        }
    ]
}
```

#### 3. Scene Modification Task

Train the model to modify scenes based on instructions:

```python
conversation = {
    "messages": [
        {
            "role": "user",
            "content": f"Original scene:\n{original_scene}\n\nAdd a light source"
        },
        {
            "role": "assistant",
            "content": modified_scene  # Scene with light added
        }
    ]
}
```

#### 4. Multi-turn Conversation

Create conversational data about scenes:

```python
conversation = {
    "messages": [
        {"role": "user", "content": "Show me a simple scene"},
        {"role": "assistant", "content": scene_yaml},
        {"role": "user", "content": "Add a camera"},
        {"role": "assistant", "content": modified_scene},
        {"role": "user", "content": "Describe the lighting"},
        {"role": "assistant", "content": "The scene uses ambient lighting..."}
    ]
}
```

## Dataset Statistics

After preparation, check your dataset:

```python
from datasets import load_from_disk

dataset = load_from_disk("./data/processed")

print(f"Train samples: {len(dataset['train'])}")
print(f"Test samples: {len(dataset['test'])}")

# Check token length distribution
import numpy as np

lengths = [len(x['input_ids']) for x in dataset['train']]
print(f"Average length: {np.mean(lengths):.0f}")
print(f"Max length: {np.max(lengths)}")
print(f"Min length: {np.min(lengths)}")
```

## Data Augmentation

You can augment your data in `prepare_dataset.py`:

```python
def augment_scene(scene_data):
    """Apply data augmentation to scene data."""
    # Random rotation
    # Random scaling
    # Random object permutation
    # etc.
    return augmented_scene
```

## Quality Control

### Validate Data

Add validation in `yaml_to_json.py`:

```python
def validate_scene(scene_data):
    """Validate scene data structure."""
    required_keys = ["scene_id", "objects"]
    
    for key in required_keys:
        if key not in scene_data:
            return False
    
    return True
```

### Filter Data

Filter out invalid or low-quality samples:

```python
def should_include_sample(scene_data):
    """Decide whether to include a sample."""
    # Example: Filter scenes with too few objects
    if len(scene_data.get('objects', [])) < 2:
        return False
    
    # Example: Filter incomplete scenes
    if 'camera' not in scene_data:
        return False
    
    return True
```

## Tips

1. **Consistent Format**: Ensure all conversations follow the same format
2. **System Prompts**: Use clear, consistent system prompts
3. **Token Length**: Keep sequences under max_length (default 2048)
4. **Balance**: Ensure balanced distribution of different scene types
5. **Validation**: Always validate a few samples manually

## Example: Complete Custom Pipeline

```python
def custom_convert_to_qwen_format(data: List[Dict[str, Any]]):
    formatted_data = []
    
    for item in data:
        # Skip invalid data
        if not validate_scene(item):
            continue
        
        # Augment data (optional)
        if random.random() < 0.3:  # 30% augmentation
            item = augment_scene(item)
        
        # Create task-specific format
        task = random.choice(['generate', 'understand', 'modify'])
        
        if task == 'generate':
            conversation = create_generation_task(item)
        elif task == 'understand':
            conversation = create_understanding_task(item)
        else:
            conversation = create_modification_task(item)
        
        formatted_data.append(conversation)
    
    return formatted_data
```

## Testing Your Format

Before training, test your data format:

```python
# Load a few samples
dataset = load_from_disk("./data/processed")
sample = dataset['train'][0]

# Decode and inspect
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./data/processed/tokenizer_config")
decoded = tokenizer.decode(sample['input_ids'])

print(decoded)
# Verify it looks correct!
```

## Need Help?

If you're unsure about the format:

1. Check existing samples in `./data/processed`
2. Run `verify_setup.py` to test the pipeline
3. Start with a small subset of data (e.g., 100 files)
4. Validate results before scaling up

