#!/usr/bin/env python3
"""
Installation and Testing Script
==============================

This script installs dependencies and runs comprehensive testing.
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path


def run_command(command, description):
    """Run a command and return success status."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
    except Exception as e:
        print(f"💥 {description} failed with exception: {e}")
        return False


def check_python_version():
    """Check Python version compatibility."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible (requires 3.8+)")
        return False


def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing Dependencies")
    print("=" * 40)
    
    # Check if we can install packages
    if not run_command("python3 -m pip --version", "Checking pip availability"):
        print("❌ pip is not available. Please install pip first.")
        return False
    
    # Try to install packages
    packages = [
        "pandas>=1.5.0",
        "pyyaml>=6.0",
        "httpx>=0.24.0",
        "pydantic>=2.0.0"
    ]
    
    for package in packages:
        if not run_command(f"python3 -m pip install {package}", f"Installing {package}"):
            print(f"⚠️  Warning: Could not install {package}")
    
    # Test imports
    print("\n🧪 Testing Package Imports")
    print("-" * 30)
    
    test_imports = [
        ("pandas", "import pandas as pd"),
        ("yaml", "import yaml"),
        ("httpx", "import httpx"),
        ("pydantic", "from pydantic import BaseModel")
    ]
    
    successful_imports = 0
    for package, import_cmd in test_imports:
        try:
            exec(import_cmd)
            print(f"✅ {package} imported successfully")
            successful_imports += 1
        except ImportError as e:
            print(f"❌ {package} import failed: {e}")
    
    print(f"\n📊 Import Results: {successful_imports}/{len(test_imports)} packages imported")
    return successful_imports == len(test_imports)


def run_system_tests():
    """Run system tests."""
    print("\n🧪 Running System Tests")
    print("=" * 40)
    
    # Run simplified test first
    print("1. Running simplified test (no dependencies)...")
    if run_command("python3 simplified_test.py", "Simplified test"):
        print("✅ Simplified test passed")
    else:
        print("❌ Simplified test failed")
    
    # Run comprehensive test if dependencies are available
    print("\n2. Running comprehensive test (with dependencies)...")
    if run_command("python3 comprehensive_test_suite.py", "Comprehensive test"):
        print("✅ Comprehensive test passed")
    else:
        print("❌ Comprehensive test failed (may be due to missing dependencies)")


def create_example_files():
    """Create example files for testing."""
    print("\n📝 Creating Example Files")
    print("=" * 40)
    
    # Create example dataset
    example_data = """prompt,response,ground_truth,user_features
"What is the capital of France?","The capital of France is Paris.","Paris","{\"location\": \"New York\"}"
"How do I cook pasta?","Boil water, add pasta, cook for 8-10 minutes.","Boil water, add pasta, cook","{\"cooking_level\": \"beginner\"}"
"What is 2+2?","2+2 equals 4.","4","{\"math_level\": \"basic\"}"
"""
    
    with open("data/example_dataset.csv", "w") as f:
        f.write(example_data)
    print("✅ Created example dataset: data/example_dataset.csv")
    
    # Create example configuration
    example_config = """dataset:
  path: data/example_dataset.csv
  prompt_column: prompt
  response_column: response
  ground_truth_column: ground_truth
  user_features_column: user_features

models:
  response_model: GOLDEN_RESPONSE
  judge_models: [gpt-4o]

api_keys:
  openai_api_key: your-openai-api-key
  openai_base_url: https://api.openai.com/v1

evaluation:
  experiment_name: example_evaluation
  run_name: example_run_1
  metrics:
    accuracy:
      prompt_template: |
        You are an impartial evaluator.
        Evaluate the accuracy of the response compared to the ground truth.
        
        User Query: {prompt}
        Model Response: {response}
        Ground Truth: {ground_truth}
        
        Rate accuracy from 1-5 where:
        1 = Completely inaccurate
        5 = Completely accurate
        
        Return JSON: {{"accuracy_score": <1-5>, "explanation": "<reasoning>"}}
      threshold: 3.0
      output_schema:
        fields:
          accuracy_score: int
          explanation: str
    
    relevance:
      prompt_template: |
        You are an impartial evaluator.
        Evaluate how relevant the response is to the user query.
        
        User Query: {prompt}
        Model Response: {response}
        
        Rate relevance from 1-5 where:
        1 = Completely irrelevant
        5 = Completely relevant
        
        Return JSON: {{"relevance_score": <1-5>, "explanation": "<reasoning>"}}
      threshold: 3.0
      output_schema:
        fields:
          relevance_score: int
          explanation: str
"""
    
    with open("configs/example_config.yaml", "w") as f:
        f.write(example_config)
    print("✅ Created example configuration: configs/example_config.yaml")


def show_usage_instructions():
    """Show usage instructions."""
    print("\n🚀 Usage Instructions")
    print("=" * 40)
    
    print("1. Set your OpenAI API key:")
    print("   export OPENAI_API_KEY='your-openai-api-key'")
    
    print("\n2. Run the interactive setup:")
    print("   python3 quick_start.py")
    
    print("\n3. Or run with the example configuration:")
    print("   python3 template_evaluation_system.py --config configs/example_config.yaml")
    
    print("\n4. View results in MLflow:")
    print("   mlflow ui")
    print("   # Open http://localhost:5000 in your browser")
    
    print("\n📁 File Structure:")
    print("   data/           # Your datasets")
    print("   configs/        # Your configurations")
    print("   results/        # Evaluation results")
    print("   examples/       # Example files")


def main():
    """Main installation and testing function."""
    print("🚀 Universal Evaluation Template System - Installation & Testing")
    print("=" * 70)
    
    # Check Python version
    if not check_python_version():
        print("\n❌ Installation aborted due to Python version incompatibility")
        return False
    
    # Create necessary directories
    print("\n📁 Creating Directories")
    print("-" * 25)
    directories = ["data", "configs", "results", "examples"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created: {directory}/")
    
    # Install dependencies
    deps_installed = install_dependencies()
    
    # Run tests
    run_system_tests()
    
    # Create example files
    create_example_files()
    
    # Show usage instructions
    show_usage_instructions()
    
    print("\n🎉 Installation and Testing Complete!")
    print("=" * 50)
    
    if deps_installed:
        print("✅ All dependencies installed successfully")
        print("✅ System is ready for full evaluation")
    else:
        print("⚠️  Some dependencies could not be installed")
        print("⚠️  System will work with limited functionality")
        print("💡 Try installing dependencies manually: pip install -r requirements.txt")
    
    print("\n🚀 Next Steps:")
    print("1. Set your OpenAI API key")
    print("2. Run: python3 quick_start.py")
    print("3. Follow the interactive prompts")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)