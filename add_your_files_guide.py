#!/usr/bin/env python3
"""
Guide: How to Add Your Files to the Evaluation System
====================================================

This script shows you exactly how to add your own files and run evaluations.
"""

import os
import json
from pathlib import Path


def create_directory_structure():
    """Create the recommended directory structure."""
    print("📁 Creating Directory Structure")
    print("=" * 40)
    
    directories = [
        "data",           # Your datasets go here
        "configs",        # Your configurations go here  
        "results",        # Results will be saved here
        "examples"        # Example files
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created: {directory}/")
    
    print("\n📋 Directory Structure Created:")
    print("your_project/")
    print("├── data/           # ← Put your CSV datasets here")
    print("├── configs/        # ← Put your YAML configs here")
    print("├── results/        # ← Results saved here automatically")
    print("└── template_evaluation_system.py")


def show_dataset_requirements():
    """Show what your dataset needs to look like."""
    print("\n📊 Dataset Requirements")
    print("=" * 40)
    
    print("Your CSV file needs these columns:")
    print("✅ REQUIRED:")
    print("  - prompt: User questions/queries")
    print("  - response: Model responses to evaluate")
    
    print("\n✅ OPTIONAL (but recommended):")
    print("  - ground_truth: Correct answers for accuracy evaluation")
    print("  - user_features: User personalization data (JSON string)")
    print("  - any_other_columns: You can add any additional data")
    
    print("\n📝 Example CSV structure:")
    print("prompt,response,ground_truth,user_features")
    print('"What is 2+2?","2+2 equals 4.","4","{}"')
    print('"How do I cook pasta?","Boil water, add pasta, cook 8-10 minutes.","Boil and cook","{\"cooking_level\": \"beginner\"}"')


def create_example_dataset():
    """Create an example dataset to show the format."""
    example_data = [
        {
            "prompt": "What is the capital of France?",
            "response": "The capital of France is Paris, a beautiful city known for its art and culture.",
            "ground_truth": "Paris",
            "user_features": '{"location": "New York", "interests": ["travel"]}',
            "category": "geography"
        },
        {
            "prompt": "How do I improve my credit score?",
            "response": "To improve your credit score, pay bills on time, keep credit card balances low, and check your credit report regularly.",
            "ground_truth": "Pay bills on time, keep balances low, check credit report",
            "user_features": '{"credit_score": 650, "debt_amount": 5000}',
            "category": "finance"
        },
        {
            "prompt": "What's the best programming language to learn?",
            "response": "Python is an excellent choice for beginners due to its simple syntax and wide range of applications.",
            "ground_truth": "Python is good for beginners",
            "user_features": '{"programming_experience": "none", "goals": "data_science"}',
            "category": "technology"
        }
    ]
    
    # Save as CSV
    import csv
    with open("data/example_dataset.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=example_data[0].keys())
        writer.writeheader()
        writer.writerows(example_data)
    
    print("✅ Created example dataset: data/example_dataset.csv")
    return example_data


def show_configuration_options():
    """Show how to create configurations."""
    print("\n⚙️ Configuration Options")
    print("=" * 40)
    
    print("You have 3 ways to create configurations:")
    
    print("\n1️⃣ INTERACTIVE SETUP (Easiest):")
    print("   python quick_start.py")
    print("   # This will ask you questions and create the config")
    
    print("\n2️⃣ CONFIGURATION GENERATOR:")
    print("   python generate_config.py \\")
    print("       --dataset data/your_file.csv \\")
    print("       --prompt-col prompt \\")
    print("       --response-col response \\")
    print("       --experiment my_evaluation \\")
    print("       --metrics accuracy relevance helpfulness \\")
    print("       --output configs/my_config.yaml")
    
    print("\n3️⃣ MANUAL CONFIGURATION:")
    print("   # Copy config_template.yaml and edit it")
    print("   cp config_template.yaml configs/my_config.yaml")
    print("   # Then edit the file with your settings")


def show_step_by_step_guide():
    """Show step-by-step instructions."""
    print("\n🚀 Step-by-Step Guide")
    print("=" * 40)
    
    print("STEP 1: Prepare Your Data")
    print("-" * 25)
    print("1. Put your CSV file in the data/ folder")
    print("2. Make sure it has 'prompt' and 'response' columns")
    print("3. Add optional columns like 'ground_truth', 'user_features'")
    
    print("\nSTEP 2: Create Configuration")
    print("-" * 30)
    print("1. Run: python quick_start.py")
    print("2. Answer the questions about your dataset")
    print("3. Select which metrics you want to use")
    print("4. The system will create a config file for you")
    
    print("\nSTEP 3: Run Evaluation")
    print("-" * 25)
    print("1. Set your API key: export OPENAI_API_KEY='your-key'")
    print("2. Run: python template_evaluation_system.py --config configs/your_config.yaml")
    print("3. Results will be saved in results/ folder")
    
    print("\nSTEP 4: View Results")
    print("-" * 20)
    print("1. Check the CSV file in results/ folder")
    print("2. Run: mlflow ui")
    print("3. Open http://localhost:5000 in your browser")


def show_file_examples():
    """Show examples of different file types."""
    print("\n📄 File Examples")
    print("=" * 40)
    
    print("CSV Dataset Example:")
    print("-" * 20)
    print("prompt,response,ground_truth,user_features")
    print('"What is AI?","AI is artificial intelligence.","Artificial intelligence","{\"tech_level\": \"beginner\"}"')
    print('"How to cook rice?","Rinse rice, add water, boil 15 minutes.","Rinse, add water, boil","{\"cooking_level\": \"intermediate\"}"')
    
    print("\nYAML Configuration Example:")
    print("-" * 30)
    print("dataset:")
    print("  path: data/my_dataset.csv")
    print("  prompt_column: prompt")
    print("  response_column: response")
    print("  ground_truth_column: ground_truth")
    print("models:")
    print("  response_model: GOLDEN_RESPONSE")
    print("  judge_models: [gpt-4o]")
    print("evaluation:")
    print("  experiment_name: my_evaluation")
    print("  metrics:")
    print("    accuracy:")
    print("      prompt_template: 'Evaluate accuracy...'")
    print("      threshold: 3.0")


def show_common_scenarios():
    """Show common usage scenarios."""
    print("\n🎯 Common Scenarios")
    print("=" * 40)
    
    print("SCENARIO 1: Basic Evaluation")
    print("-" * 30)
    print("Files needed:")
    print("  - data/my_data.csv (with prompt, response columns)")
    print("  - configs/basic_config.yaml")
    print("Command: python template_evaluation_system.py --config configs/basic_config.yaml")
    
    print("\nSCENARIO 2: Evaluation with Ground Truth")
    print("-" * 40)
    print("Files needed:")
    print("  - data/my_data.csv (with prompt, response, ground_truth columns)")
    print("  - configs/accuracy_config.yaml")
    print("Command: python template_evaluation_system.py --config configs/accuracy_config.yaml")
    
    print("\nSCENARIO 3: Personalized Evaluation")
    print("-" * 35)
    print("Files needed:")
    print("  - data/my_data.csv (with prompt, response, user_features columns)")
    print("  - configs/personalized_config.yaml")
    print("Command: python template_evaluation_system.py --config configs/personalized_config.yaml")


def main():
    """Main guide function."""
    print("📚 How to Add Your Files to the Evaluation System")
    print("=" * 60)
    
    # Create directory structure
    create_directory_structure()
    
    # Show dataset requirements
    show_dataset_requirements()
    
    # Create example dataset
    create_example_dataset()
    
    # Show configuration options
    show_configuration_options()
    
    # Show step-by-step guide
    show_step_by_step_guide()
    
    # Show file examples
    show_file_examples()
    
    # Show common scenarios
    show_common_scenarios()
    
    print("\n🎉 You're Ready to Add Your Files!")
    print("=" * 40)
    print("1. Put your CSV in data/ folder")
    print("2. Run: python quick_start.py")
    print("3. Follow the prompts")
    print("4. Run your evaluation!")


if __name__ == "__main__":
    main()