"""Test environment setup and dependencies."""

import pytest
import sys
from pathlib import Path

# Add project root to sys.path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_python_version():
    """Verify Python version is 3.9+"""
    assert sys.version_info >= (3, 9), "Python 3.9+ required"

def test_directory_structure():
    """Verify required directories exist"""
    base_path = Path(__file__).parent.parent.parent
    
    required_dirs = [
        "code/src/ingestion",
        "code/src/preprocessing",
        "code/src/indexing",
        "code/src/retrieval",
        "code/src/generation",
        "code/src/utils",
        "data/raw",
        "data/processed",
        "models/llm",
        "logs"
    ]
    
    for dir_path in required_dirs:
        full_path = base_path / dir_path
        assert full_path.exists(), f"Missing directory: {dir_path}"

def test_dependencies_import():
    """Test critical dependencies can be imported"""
    try:
        import torch
        import sentence_transformers
        import faiss
        import indic_transliteration
        from llama_cpp import Llama
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import dependency: {e}")

def test_config_loads():
    """Test configuration file loads correctly"""
    from code.src.utils.config_loader import load_config, validate_config
    
    config = load_config()
    assert validate_config(config)
    assert 'data' in config
    assert 'models' in config

if __name__ == "__main__":
    pytest.main([__file__, "-v"])