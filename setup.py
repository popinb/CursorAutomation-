#!/usr/bin/env python3
"""
Setup Script for Universal Evaluation Template System
====================================================

This script sets up the evaluation system and provides installation instructions.
"""

import os
import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3.8, 0):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True


def install_requirements():
    """Install required packages."""
    print("\n📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False


def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    directories = ["data", "results", "configs", "examples"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")


def create_example_files():
    """Create example files for users."""
    print("\n📝 Creating example files...")
    
    # Create example dataset
    example_data = """prompt,response,ground_truth,user_features
"What is the capital of France?","Paris is the capital of France.","Paris","{}"
"How do I bake a chocolate cake?","Mix flour, sugar, eggs, and cocoa powder. Bake at 350°F for 30 minutes.","Mix ingredients and bake","{\"cooking_experience\": \"beginner\"}"
"What are the benefits of exercise?","Exercise improves cardiovascular health, builds muscle, and boosts mental well-being.","Improved health and fitness","{\"fitness_level\": \"intermediate\"}"
"""
    
    with open("data/example_dataset.csv", "w") as f:
        f.write(example_data)
    print("✅ Created example dataset: data/example_dataset.csv")
    
    # Create example configuration
    example_config = """dataset:
  path: "data/example_dataset.csv"
  prompt_column: "prompt"
  response_column: "response"
  ground_truth_column: "ground_truth"
  user_features_column: "user_features"

models:
  response_model: "GOLDEN_RESPONSE"
  judge_models: ["gpt-4o"]

api_keys:
  openai_api_key: "your-openai-api-key"
  openai_base_url: "https://api.openai.com/v1"

evaluation:
  experiment_name: "example_evaluation"
  run_name: "example_run_1"
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
"""
    
    with open("configs/example_config.yaml", "w") as f:
        f.write(example_config)
    print("✅ Created example configuration: configs/example_config.yaml")


def run_tests():
    """Run the test suite."""
    print("\n🧪 Running tests...")
    try:
        result = subprocess.run([sys.executable, "test_template.py"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ All tests passed")
            return True
        else:
            print("❌ Some tests failed")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
    except Exception as e:
        print(f"❌ Failed to run tests: {e}")
        return False


def print_usage_instructions():
    """Print usage instructions."""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE!")
    print("="*60)
    print("\n📚 Quick Start:")
    print("1. Set your OpenAI API key:")
    print("   export OPENAI_API_KEY='your-api-key'")
    print("\n2. Run the interactive setup:")
    print("   python quick_start.py")
    print("\n3. Or run with example data:")
    print("   python template_evaluation_system.py --config configs/example_config.yaml")
    print("\n4. View results in MLflow:")
    print("   mlflow ui")
    print("\n📖 For more information, see README.md")
    print("\n🔧 Available commands:")
    print("   python quick_start.py              # Interactive setup")
    print("   python generate_config.py --help   # Configuration generator")
    print("   python test_template.py            # Run tests")
    print("   python example_usage.py            # See examples")


def main():
    """Main setup function."""
    print("🚀 Universal Evaluation Template System - Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Create directories
    create_directories()
    
    # Create example files
    create_example_files()
    
    # Run tests
    if not run_tests():
        print("⚠️  Some tests failed, but setup can continue")
    
    # Print usage instructions
    print_usage_instructions()
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)