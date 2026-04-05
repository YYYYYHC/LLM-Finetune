"""
Test script for data processing pipeline.
"""

import os
import sys
import json
import yaml
import tempfile
import shutil
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.yaml_to_json import convert_yaml_file
from src.utils.logger import setup_logger

logger = setup_logger("test")


def create_sample_yaml(output_dir: Path, num_files: int = 5) -> None:
    """Create sample YAML files for testing."""
    logger.info(f"Creating {num_files} sample YAML files")
    
    for i in range(num_files):
        sample_data = {
            "scene_id": f"scene_{i:03d}",
            "objects": [
                {
                    "name": f"object_{j}",
                    "type": "cube" if j % 2 == 0 else "sphere",
                    "position": [j, 0, 0],
                    "rotation": [0, 0, 0],
                    "scale": [1, 1, 1]
                }
                for j in range(3)
            ],
            "lighting": {
                "type": "directional",
                "intensity": 1.0,
                "color": [1.0, 1.0, 1.0]
            },
            "camera": {
                "position": [5, 5, 5],
                "target": [0, 0, 0],
                "fov": 60
            }
        }
        
        yaml_path = output_dir / f"scene_{i:03d}.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(sample_data, f)
    
    logger.info(f"Created {num_files} YAML files in {output_dir}")


def test_yaml_to_json() -> bool:
    """Test YAML to JSON conversion."""
    logger.info("Testing YAML to JSON conversion")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        yaml_dir = tmpdir / "yaml"
        json_dir = tmpdir / "json"
        
        yaml_dir.mkdir()
        json_dir.mkdir()
        
        # Create sample YAML files
        create_sample_yaml(yaml_dir, num_files=5)
        
        # Convert to JSON
        yaml_files = list(yaml_dir.glob("*.yaml"))
        logger.info(f"Converting {len(yaml_files)} files")
        
        success_count = 0
        for yaml_file in yaml_files:
            if convert_yaml_file(yaml_file, json_dir):
                success_count += 1
        
        # Verify JSON files
        json_files = list(json_dir.glob("*.json"))
        
        if len(json_files) != len(yaml_files):
            logger.error(f"Expected {len(yaml_files)} JSON files, got {len(json_files)}")
            return False
        
        # Verify content
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
                if "scene_id" not in data:
                    logger.error(f"Missing scene_id in {json_file}")
                    return False
                
                if "objects" not in data:
                    logger.error(f"Missing objects in {json_file}")
                    return False
        
        logger.info(f"✓ YAML to JSON conversion: {success_count}/{len(yaml_files)} successful")
        return success_count == len(yaml_files)


def test_conversation_format() -> bool:
    """Test conversation format conversion."""
    logger.info("Testing conversation format")
    
    from src.data.prepare_dataset import convert_to_qwen_format
    
    # Sample data
    sample_data = [
        {
            "scene_id": "test_scene",
            "objects": [
                {"name": "cube", "type": "cube"}
            ]
        }
    ]
    
    # Convert
    conversations = convert_to_qwen_format(sample_data)
    
    # Verify
    if len(conversations) != len(sample_data):
        logger.error("Conversation count mismatch")
        return False
    
    conv = conversations[0]
    
    if "messages" not in conv:
        logger.error("Missing 'messages' key")
        return False
    
    messages = conv["messages"]
    
    if len(messages) < 2:
        logger.error("Too few messages")
        return False
    
    # Check message structure
    for msg in messages:
        if "role" not in msg or "content" not in msg:
            logger.error("Invalid message structure")
            return False
        
        if msg["role"] not in ["system", "user", "assistant"]:
            logger.error(f"Invalid role: {msg['role']}")
            return False
    
    logger.info("✓ Conversation format test passed")
    return True


def test_imports() -> bool:
    """Test that all modules can be imported."""
    logger.info("Testing module imports")
    
    try:
        from src.data.yaml_to_json import convert_yaml_to_json
        from src.data.prepare_dataset import prepare_dataset
        from src.models.model_factory import create_model
        from src.training.trainer import Trainer
        from src.utils.config_loader import load_config
        from src.utils.logger import setup_logger
        
        logger.info("✓ All modules imported successfully")
        return True
    except Exception as e:
        logger.error(f"Import failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("Running Data Pipeline Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Module Imports", test_imports),
        ("YAML to JSON Conversion", test_yaml_to_json),
        ("Conversation Format", test_conversation_format)
    ]
    
    results = []
    
    for name, test_func in tests:
        logger.info(f"\nRunning: {name}")
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            logger.error(f"Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        logger.info("\n✓ All tests passed!")
        return 0
    else:
        logger.error("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

