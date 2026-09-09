#!/usr/bin/env python3
"""
Test script to verify Docker setup and CUDA compatibility.
Run with: python test_docker.py
"""

import sys
import torch
import numpy as np
from pathlib import Path

def test_pytorch_cuda():
    """Test PyTorch and CUDA installation."""
    print("=" * 60)
    print("PyTorch/CUDA Compatibility Test")
    print("=" * 60)
    
    # Test PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    
    # Test CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        # Test GPU memory
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"  Compute capability: {props.major}.{props.minor}")
            
        # Test tensor operations on GPU
        try:
            x = torch.randn(100, 100).cuda()
            y = torch.randn(100, 100).cuda()
            z = torch.matmul(x, y)
            print("✓ GPU tensor operations working")
        except Exception as e:
            print(f"✗ GPU tensor operations failed: {e}")
    else:
        print("⚠️ CUDA not available - running in CPU mode")
    
    return cuda_available

def test_imports():
    """Test import of all required packages."""
    print("\n" + "=" * 60)
    print("Package Import Test")
    print("=" * 60)
    
    packages = [
        ("fastapi", "fastapi"),
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("sentence_transformers", "sentence_transformers"),
        ("accelerate", "accelerate"),
        ("huggingface_hub", "huggingface_hub"),
        ("open_clip", "open_clip"),
        ("PIL", "PIL"),
        ("cv2", "cv2"),
        ("numpy", "numpy"),
        ("rasterio", "rasterio"),
        ("geopandas", "geopandas"),
        ("pydantic", "pydantic"),
        ("pydantic_settings", "pydantic_settings"),
        ("uvicorn", "uvicorn"),
        ("schedule", "schedule"),
        ("python_magic", "magic"),
    ]
    
    all_imported = True
    for module_name, import_name in packages:
        try:
            __import__(import_name if import_name != module_name else module_name)
            print(f"✓ {module_name}")
        except ImportError as e:
            print(f"✗ {module_name}: {e}")
            all_imported = False
        except Exception as e:
            print(f"⚠️ {module_name}: {e}")
    
    return all_imported

def test_directories():
    """Test that required directories exist and are writable."""
    print("\n" + "=" * 60)
    print("Directory Permissions Test")
    print("=" * 60)
    
    directories = [
        Path("uploads"),
        Path("model_cache"),
        Path("reports"),
        Path("logs"),
    ]
    
    all_ok = True
    for dir_path in directories:
        try:
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"✓ Created {dir_path}")
            
            # Test write permission
            test_file = dir_path / ".write_test"
            try:
                test_file.write_text("test")
                test_file.unlink()
                print(f"✓ {dir_path} is writable")
            except Exception as e:
                print(f"✗ {dir_path} not writable: {e}")
                all_ok = False
                
        except Exception as e:
            print(f"✗ Failed to create/check {dir_path}: {e}")
            all_ok = False
    
    return all_ok

def test_config():
    """Test configuration loading."""
    print("\n" + "=" * 60)
    print("Configuration Test")
    print("=" * 60)
    
    try:
        from config import Settings, get_settings
        
        # Test default settings
        settings = Settings()
        print(f"✓ Settings loaded")
        print(f"  Device: {settings.resolved_device}")
        print(f"  Upload dir: {settings.upload_dir}")
        print(f"  Model cache dir: {settings.model_cache_dir}")
        print(f"  Max concurrent inference: {settings.max_concurrent_inference}")
        
        return True
    except Exception as e:
        print(f"✗ Configuration failed: {e}")
        return False

def main():
    """Run all tests."""
    print("SatQuery AI Docker Setup Test")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("PyTorch/CUDA", test_pytorch_cuda()))
    results.append(("Package Imports", test_imports()))
    results.append(("Directory Permissions", test_directories()))
    results.append(("Configuration", test_config()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{test_name:25} {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All tests passed! Docker setup is ready.")
    else:
        print("❌ Some tests failed. Check the output above.")
        sys.exit(1)

if __name__ == "__main__":
    main()