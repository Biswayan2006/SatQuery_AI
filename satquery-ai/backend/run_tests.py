#!/usr/bin/env python3
"""
SatQuery AI â€” Test Runner
Run deployment-focused tests for production readiness.
"""
import sys
import subprocess
import os

def run_tests():
    """Run all deployment-focused tests."""
    print("=" * 60)
    print("SatQuery AI â€” Deployment Tests")
    print("=" * 60)
    
    # Set test environment variables
    env = os.environ.copy()
    env["ENVIRONMENT"] = "testing"
    env["API_KEY_ENABLED"] = "false"
    env["LOG_LEVEL"] = "WARNING"
    
    # Run pytest with deployment tests
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_deployment.py",
        "-v",
        "--tb=short",
        "--disable-warnings",
        "--color=yes"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, env=env, cwd=os.path.dirname(__file__))
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        return False
    except Exception as e:
        print(f"\n\nError running tests: {e}")
        return False

def run_specific_test_category(category: str):
    """Run tests for a specific category."""
    categories = {
        "api": "TestAPIEndpoints",
        "upload": "TestUploadFunctionality", 
        "validation": "TestValidation",
        "model": "TestModelFunctionality",
        "cleanup": "TestCleanupFunctionality",
        "health": "TestHealthAndMonitoring",
        "integration": "TestIntegration",
        "performance": "TestPerformance",
        "security": "TestSecurity",
        "all": None
    }
    
    if category not in categories:
        print(f"Unknown category: {category}")
        print(f"Available categories: {', '.join(categories.keys())}")
        return False
    
    env = os.environ.copy()
    env["ENVIRONMENT"] = "testing"
    env["API_KEY_ENABLED"] = "false"
    env["LOG_LEVEL"] = "WARNING"
    
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_deployment.py",
        "-v",
        "--tb=short",
        "--disable-warnings",
        "--color=yes"
    ]
    
    if category != "all":
        test_class = categories[category]
        cmd.extend(["-k", test_class])
    
    print(f"Running {category} tests: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, env=env, cwd=os.path.dirname(__file__))
        return result.returncode == 0
    except KeyboardInterrupt:
        print(f"\n\n{category} tests interrupted by user")
        return False
    except Exception as e:
        print(f"\n\nError running {category} tests: {e}")
        return False

def main():
    """Main test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run SatQuery AI deployment tests")
    parser.add_argument(
        "category",
        nargs="?",
        default="all",
        help="Test category to run (api, upload, validation, model, cleanup, health, integration, performance, security, all)"
    )
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="List available test categories"
    )
    
    args = parser.parse_args()
    
    if args.list_categories:
        print("Available test categories:")
        print("  api: Basic API functionality and routing")
        print("  upload: File upload functionality and validation")
        print("  validation: Request validation and error handling")
        print("  model: Model management and functionality")
        print("  cleanup: Cleanup service functionality")
        print("  health: Health monitoring and degradation endpoints")
        print("  integration: Integrated workflows")
        print("  performance: Performance characteristics")
        print("  security: Security features")
        print("  all: All deployment tests (default)")
        return 0
    
    if args.category == "all":
        success = run_tests()
    else:
        success = run_specific_test_category(args.category)
    
    print("\n" + "=" * 60)
    if success:
        print("âœ… All tests passed!")
        return 0
    else:
        print("âŒ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
