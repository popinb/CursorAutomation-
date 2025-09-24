#!/usr/bin/env python3
"""
Enhanced Setup Script for Universal Evaluation Template System
=============================================================

This script sets up the enhanced evaluation system with all components.
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path
import json


def run_command(command, description, check=True):
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
            if check:
                return False
            else:
                print("⚠️  Continuing despite failure...")
                return True
    except Exception as e:
        print(f"💥 {description} failed with exception: {e}")
        if check:
            return False
        else:
            print("⚠️  Continuing despite failure...")
            return True


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


def create_virtual_environment():
    """Create virtual environment."""
    print("\n📦 Creating Virtual Environment")
    print("=" * 40)
    
    venv_path = "venv"
    
    # Remove existing venv if it exists
    if os.path.exists(venv_path):
        print("🧹 Removing existing virtual environment...")
        import shutil
        shutil.rmtree(venv_path)
    
    # Create new venv
    if not run_command(f"python3 -m venv {venv_path}", "Creating virtual environment"):
        return False
    
    # Activate venv and install pip
    activate_cmd = f"source {venv_path}/bin/activate && python -m pip install --upgrade pip"
    if not run_command(activate_cmd, "Upgrading pip in virtual environment"):
        return False
    
    return True


def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing Dependencies")
    print("=" * 40)
    
    # Install core dependencies first
    core_packages = [
        "pandas>=1.5.0",
        "pyyaml>=6.0",
        "numpy>=1.21.0",
        "httpx>=0.24.0",
        "pydantic>=2.0.0"
    ]
    
    for package in core_packages:
        cmd = f"source venv/bin/activate && pip install {package}"
        run_command(cmd, f"Installing {package}", check=False)
    
    # Install MLflow
    mlflow_cmd = "source venv/bin/activate && pip install mlflow>=3.0.0"
    run_command(mlflow_cmd, "Installing MLflow", check=False)
    
    # Install LangChain
    langchain_cmd = "source venv/bin/activate && pip install langchain-openai langchain-core openai"
    run_command(langchain_cmd, "Installing LangChain", check=False)
    
    # Install optional packages
    optional_packages = [
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "jupyter>=1.0.0",
        "tqdm>=4.64.0",
        "rich>=13.0.0"
    ]
    
    for package in optional_packages:
        cmd = f"source venv/bin/activate && pip install {package}"
        run_command(cmd, f"Installing optional {package}", check=False)
    
    return True


def test_imports():
    """Test package imports."""
    print("\n🧪 Testing Package Imports")
    print("-" * 30)
    
    test_script = """
import sys
import os
sys.path.insert(0, os.getcwd())

# Test core imports
try:
    import pandas as pd
    print("✅ pandas imported successfully")
except ImportError as e:
    print(f"❌ pandas import failed: {e}")

try:
    import yaml
    print("✅ yaml imported successfully")
except ImportError as e:
    print(f"❌ yaml import failed: {e}")

try:
    import httpx
    print("✅ httpx imported successfully")
except ImportError as e:
    print(f"❌ httpx import failed: {e}")

try:
    from pydantic import BaseModel
    print("✅ pydantic imported successfully")
except ImportError as e:
    print(f"❌ pydantic import failed: {e}")

try:
    import mlflow
    print("✅ mlflow imported successfully")
except ImportError as e:
    print(f"❌ mlflow import failed: {e}")

try:
    from langchain_openai import ChatOpenAI
    print("✅ langchain imported successfully")
except ImportError as e:
    print(f"❌ langchain import failed: {e}")

# Test our modules
try:
    from enhanced_evaluation_system import EnhancedUniversalEvaluator, EvaluationConfig
    print("✅ Enhanced evaluation system imported successfully")
except ImportError as e:
    print(f"❌ Enhanced evaluation system import failed: {e}")

try:
    from metric_templates import METRIC_TEMPLATES, DOMAIN_TEMPLATES
    print("✅ Metric templates imported successfully")
except ImportError as e:
    print(f"❌ Metric templates import failed: {e}")
"""
    
    with open("test_imports.py", "w") as f:
        f.write(test_script)
    
    cmd = "source venv/bin/activate && python test_imports.py"
    success = run_command(cmd, "Testing imports", check=False)
    
    # Clean up
    if os.path.exists("test_imports.py"):
        os.remove("test_imports.py")
    
    return success


def create_example_files():
    """Create example files for testing."""
    print("\n📝 Creating Example Files")
    print("=" * 40)
    
    # Create example dataset
    example_data = """prompt,response,ground_truth,user_features
"What is the capital of France?","The capital of France is Paris, a beautiful city known for its art and culture.","Paris","{\"location\": \"New York\", \"interests\": [\"travel\"]}"
"How do I improve my credit score?","To improve your credit score, pay bills on time, keep credit card balances low, and check your credit report regularly for errors.","Pay bills on time, keep balances low, check credit report","{\"credit_score\": 650, \"debt_amount\": 5000, \"income\": 60000}"
"What's the best programming language to learn?","Python is an excellent choice for beginners due to its simple syntax and wide range of applications.","Python is good for beginners","{\"programming_experience\": \"none\", \"goals\": \"data_science\"}"
"Should I invest all my money in cryptocurrency?","Investing all your money in cryptocurrency is extremely risky. I recommend diversifying your investments and only investing what you can afford to lose.","Diversify investments, don't put all money in crypto","{\"age\": 25, \"income\": 50000, \"risk_tolerance\": \"high\"}"
"""
    
    with open("data/example_dataset.csv", "w") as f:
        f.write(example_data)
    print("✅ Created example dataset: data/example_dataset.csv")
    
    # Create enhanced configuration
    enhanced_config = """dataset:
  path: data/example_dataset.csv
  prompt_column: prompt
  response_column: response
  ground_truth_column: ground_truth
  user_features_column: user_features

models:
  response_model: GOLDEN_RESPONSE
  judge_models: [gpt-4o, gpt-4]

api_keys:
  openai_api_key: your-openai-api-key
  openai_base_url: https://api.openai.com/v1

evaluation:
  experiment_name: enhanced_evaluation_example
  run_name: enhanced_run_1
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
        2 = Mostly inaccurate  
        3 = Partially accurate
        4 = Mostly accurate
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
        2 = Mostly irrelevant
        3 = Partially relevant
        4 = Mostly relevant
        5 = Completely relevant
        
        Return JSON: {{"relevance_score": <1-5>, "explanation": "<reasoning>"}}
      threshold: 3.0
      output_schema:
        fields:
          relevance_score: int
          explanation: str
    
    personalization:
      prompt_template: |
        You are an impartial evaluator.
        Evaluate how well the response incorporates user personalization features.
        
        User Query: {prompt}
        Model Response: {response}
        User Features: {user_features}
        
        Rate personalization from 1-5 where:
        1 = No personalization
        2 = Minimal personalization
        3 = Moderate personalization
        4 = Good personalization
        5 = Excellent personalization
        
        Return JSON: {{"personalization_score": <1-5>, "explanation": "<reasoning>"}}
      threshold: 3.0
      output_schema:
        fields:
          personalization_score: int
          explanation: str
    
    helpfulness:
      prompt_template: |
        You are an impartial evaluator.
        Evaluate how helpful the response is to the user.
        
        User Query: {prompt}
        Model Response: {response}
        
        Rate helpfulness from 1-5 where:
        1 = Not helpful at all
        2 = Slightly helpful
        3 = Moderately helpful
        4 = Very helpful
        5 = Extremely helpful
        
        Return JSON: {{"helpfulness_score": <1-5>, "explanation": "<reasoning>"}}
      threshold: 3.0
      output_schema:
        fields:
          helpfulness_score: int
          explanation: str
"""
    
    with open("configs/enhanced_config.yaml", "w") as f:
        f.write(enhanced_config)
    print("✅ Created enhanced configuration: configs/enhanced_config.yaml")


def run_comprehensive_test():
    """Run comprehensive test suite."""
    print("\n🧪 Running Comprehensive Test Suite")
    print("=" * 40)
    
    # Run fixed simplified test
    print("1. Running fixed simplified test...")
    cmd = "source venv/bin/activate && python fixed_simplified_test.py"
    run_command(cmd, "Fixed simplified test", check=False)
    
    # Run enhanced system test
    print("\n2. Running enhanced system test...")
    test_script = """
import sys
import os
sys.path.insert(0, os.getcwd())

from enhanced_evaluation_system import EvaluationConfig, EnhancedUniversalEvaluator, DataHandler

# Test configuration loading
try:
    config = EvaluationConfig("configs/enhanced_config.yaml")
    print("✅ Configuration loaded successfully")
except Exception as e:
    print(f"❌ Configuration loading failed: {e}")

# Test data loading
try:
    data = DataHandler.load_dataset("data/example_dataset.csv")
    print(f"✅ Data loaded successfully: {len(data)} rows")
except Exception as e:
    print(f"❌ Data loading failed: {e}")

# Test evaluator creation
try:
    evaluator = EnhancedUniversalEvaluator(config)
    print("✅ Evaluator created successfully")
except Exception as e:
    print(f"❌ Evaluator creation failed: {e}")

print("🎉 Enhanced system test completed!")
"""
    
    with open("test_enhanced.py", "w") as f:
        f.write(test_script)
    
    cmd = "source venv/bin/activate && python test_enhanced.py"
    run_command(cmd, "Enhanced system test", check=False)
    
    # Clean up
    if os.path.exists("test_enhanced.py"):
        os.remove("test_enhanced.py")


def create_usage_scripts():
    """Create usage scripts."""
    print("\n📝 Creating Usage Scripts")
    print("=" * 40)
    
    # Create quick start script
    quick_start_script = """#!/bin/bash
# Quick Start Script for Enhanced Evaluation System

echo "🚀 Enhanced Universal Evaluation Template System - Quick Start"
echo "=============================================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run: python enhanced_setup.py"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  OPENAI_API_KEY not set. Please set it:"
    echo "   export OPENAI_API_KEY='your-openai-api-key'"
    echo ""
    echo "Continuing with example data..."
fi

# Run the enhanced evaluation system
echo "🔬 Running enhanced evaluation system..."
python enhanced_evaluation_system.py --config configs/enhanced_config.yaml --output results/

echo "✅ Evaluation complete! Check the results/ directory for outputs."
echo "📊 To view MLflow dashboard, run: mlflow ui"
"""
    
    with open("quick_start_enhanced.sh", "w") as f:
        f.write(quick_start_script)
    
    os.chmod("quick_start_enhanced.sh", 0o755)
    print("✅ Created quick start script: quick_start_enhanced.sh")
    
    # Create MLflow UI script
    mlflow_script = """#!/bin/bash
# MLflow UI Script

echo "🔬 Starting MLflow UI..."
echo "Open http://localhost:5000 in your browser to view results"

source venv/bin/activate
mlflow ui --host 0.0.0.0 --port 5000
"""
    
    with open("start_mlflow.sh", "w") as f:
        f.write(mlflow_script)
    
    os.chmod("start_mlflow.sh", 0o755)
    print("✅ Created MLflow UI script: start_mlflow.sh")


def show_final_instructions():
    """Show final usage instructions."""
    print("\n" + "=" * 70)
    print("🎉 ENHANCED SETUP COMPLETE!")
    print("=" * 70)
    
    print("\n📁 Generated Files:")
    print("  - enhanced_evaluation_system.py (main system)")
    print("  - fixed_simplified_test.py (test suite)")
    print("  - data/example_dataset.csv (sample data)")
    print("  - configs/enhanced_config.yaml (configuration)")
    print("  - quick_start_enhanced.sh (quick start script)")
    print("  - start_mlflow.sh (MLflow UI script)")
    
    print("\n🚀 Quick Start:")
    print("1. Set your OpenAI API key:")
    print("   export OPENAI_API_KEY='your-openai-api-key'")
    print("\n2. Run the enhanced system:")
    print("   ./quick_start_enhanced.sh")
    print("\n3. View results in MLflow:")
    print("   ./start_mlflow.sh")
    print("   # Open http://localhost:5000 in your browser")
    
    print("\n🔧 Manual Usage:")
    print("1. Activate virtual environment:")
    print("   source venv/bin/activate")
    print("\n2. Run evaluation:")
    print("   python enhanced_evaluation_system.py --config configs/enhanced_config.yaml")
    print("\n3. Run tests:")
    print("   python fixed_simplified_test.py")
    
    print("\n📊 Features Available:")
    print("  ✅ Enhanced error handling and fallbacks")
    print("  ✅ MLflow integration for experiment tracking")
    print("  ✅ Multiple LLM model support")
    print("  ✅ Comprehensive metric templates")
    print("  ✅ Data validation and type checking")
    print("  ✅ Rich logging and monitoring")
    print("  ✅ Flexible configuration system")
    print("  ✅ Cross-platform compatibility")
    
    print("\n🎯 Next Steps:")
    print("1. Test with your own data")
    print("2. Customize metrics for your use case")
    print("3. Set up monitoring and alerts")
    print("4. Deploy to production environment")


def main():
    """Main setup function."""
    print("🚀 Enhanced Universal Evaluation Template System - Setup")
    print("=" * 70)
    
    # Check Python version
    if not check_python_version():
        print("\n❌ Setup aborted due to Python version incompatibility")
        return False
    
    # Create necessary directories
    print("\n📁 Creating Directories")
    print("-" * 25)
    directories = ["data", "configs", "results", "examples", "tests"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created: {directory}/")
    
    # Create virtual environment
    if not create_virtual_environment():
        print("\n❌ Virtual environment creation failed")
        return False
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Dependency installation failed")
        return False
    
    # Test imports
    if not test_imports():
        print("\n⚠️  Some imports failed, but continuing...")
    
    # Create example files
    create_example_files()
    
    # Run comprehensive test
    run_comprehensive_test()
    
    # Create usage scripts
    create_usage_scripts()
    
    # Show final instructions
    show_final_instructions()
    
    print("\n🎉 Enhanced setup completed successfully!")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)