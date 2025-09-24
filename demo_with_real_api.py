#!/usr/bin/env python3
"""
Demo with Real API Integration
=============================

This script demonstrates how to use the Universal Evaluation Template System
with real API calls (when OpenAI API key is available).
"""

import os
import json
from pathlib import Path


def create_demo_dataset():
    """Create a demo dataset for testing."""
    demo_data = [
        {
            "prompt": "What is the capital of France?",
            "response": "The capital of France is Paris, a beautiful city known for its art, culture, and cuisine.",
            "ground_truth": "Paris",
            "user_features": '{"location": "New York", "interests": ["travel", "culture"]}',
            "expected_tone": "informative"
        },
        {
            "prompt": "How do I improve my credit score?",
            "response": "To improve your credit score, pay bills on time, keep credit card balances low, avoid opening too many new accounts, and check your credit report regularly for errors.",
            "ground_truth": "Pay bills on time, keep balances low, avoid new accounts",
            "user_features": '{"credit_score": 650, "debt_amount": 5000, "income": 60000}',
            "expected_tone": "practical"
        },
        {
            "prompt": "Should I invest all my money in cryptocurrency?",
            "response": "Investing all your money in cryptocurrency is extremely risky. I recommend diversifying your investments and only investing what you can afford to lose. Consider consulting a financial advisor.",
            "ground_truth": "Diversify investments, don't put all money in crypto",
            "user_features": '{"age": 25, "income": 50000, "risk_tolerance": "high", "investment_experience": "beginner"}',
            "expected_tone": "cautious"
        }
    ]
    
    with open("demo_dataset.json", "w") as f:
        json.dump(demo_data, f, indent=2)
    
    print("✅ Created demo dataset: demo_dataset.json")
    return demo_data


def create_demo_configuration():
    """Create a demo configuration with 4 comprehensive metrics."""
    config = {
        "dataset": {
            "path": "demo_dataset.json",
            "prompt_column": "prompt",
            "response_column": "response",
            "ground_truth_column": "ground_truth",
            "user_features_column": "user_features"
        },
        "models": {
            "response_model": "GOLDEN_RESPONSE",
            "judge_models": ["gpt-4o"]
        },
        "api_keys": {
            "openai_api_key": os.getenv("OPENAI_API_KEY", "your-openai-api-key"),
            "openai_base_url": "https://api.openai.com/v1"
        },
        "evaluation": {
            "experiment_name": "demo_4_metrics_experiment",
            "run_name": "demo_run_1",
            "metrics": {
                "accuracy": {
                    "prompt_template": """
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
""",
                    "threshold": 3.0,
                    "output_schema": {
                        "fields": {
                            "accuracy_score": "int",
                            "explanation": "str"
                        }
                    }
                },
                
                "relevance": {
                    "prompt_template": """
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
""",
                    "threshold": 3.0,
                    "output_schema": {
                        "fields": {
                            "relevance_score": "int",
                            "explanation": "str"
                        }
                    }
                },
                
                "personalization": {
                    "prompt_template": """
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
""",
                    "threshold": 3.0,
                    "output_schema": {
                        "fields": {
                            "personalization_score": "int",
                            "explanation": "str"
                        }
                    }
                },
                
                "helpfulness": {
                    "prompt_template": """
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
""",
                    "threshold": 3.0,
                    "output_schema": {
                        "fields": {
                            "helpfulness_score": "int",
                            "explanation": "str"
                        }
                    }
                }
            }
        }
    }
    
    with open("demo_config.yaml", "w") as f:
        import yaml
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print("✅ Created demo configuration: demo_config.yaml")
    return config


def show_usage_instructions():
    """Show how to use the system with real API."""
    print("\n🚀 How to Use with Real API")
    print("=" * 50)
    
    print("\n1. Set your OpenAI API key:")
    print("   export OPENAI_API_KEY='your-actual-openai-key'")
    
    print("\n2. Install dependencies:")
    print("   pip install -r requirements.txt")
    
    print("\n3. Run the evaluation:")
    print("   python template_evaluation_system.py --config demo_config.yaml --output results/")
    
    print("\n4. View results in MLflow:")
    print("   mlflow ui")
    print("   # Open http://localhost:5000 in your browser")
    
    print("\n5. Alternative: Use the interactive setup:")
    print("   python quick_start.py")


def show_configuration_details():
    """Show the configuration structure."""
    print("\n⚙️  Configuration Structure")
    print("=" * 50)
    
    print("\nDataset Configuration:")
    print("- path: Path to your dataset file")
    print("- prompt_column: Column containing user questions")
    print("- response_column: Column containing model responses")
    print("- ground_truth_column: Column with correct answers (optional)")
    print("- user_features_column: Column with user data (optional)")
    
    print("\nModel Configuration:")
    print("- response_model: How to generate responses")
    print("  * GOLDEN_RESPONSE: Use existing responses")
    print("  * LLM_gpt-4o: Generate with GPT-4o")
    print("  * LLM_gpt-4: Generate with GPT-4")
    print("- judge_models: Models to use for evaluation")
    
    print("\nEvaluation Configuration:")
    print("- experiment_name: MLflow experiment name")
    print("- run_name: Specific run name")
    print("- metrics: Dictionary of evaluation metrics")
    print("  * prompt_template: Evaluation prompt")
    print("  * threshold: Pass/fail threshold")
    print("  * output_schema: Expected output format")


def show_metric_examples():
    """Show examples of different metric types."""
    print("\n📏 Metric Examples")
    print("=" * 50)
    
    print("\n1. Accuracy Metric:")
    print("   - Compares response to ground truth")
    print("   - Scale: 1-5 (inaccurate to accurate)")
    print("   - Use case: Factual correctness")
    
    print("\n2. Relevance Metric:")
    print("   - Measures how well response addresses query")
    print("   - Scale: 1-5 (irrelevant to completely relevant)")
    print("   - Use case: Query-response alignment")
    
    print("\n3. Personalization Metric:")
    print("   - Evaluates use of user features")
    print("   - Scale: 1-5 (no personalization to excellent)")
    print("   - Use case: Customized responses")
    
    print("\n4. Helpfulness Metric:")
    print("   - Assesses practical value of response")
    print("   - Scale: 1-5 (not helpful to extremely helpful)")
    print("   - Use case: User satisfaction")


def show_advanced_features():
    """Show advanced features of the system."""
    print("\n🔧 Advanced Features")
    print("=" * 50)
    
    print("\n1. Ensemble Evaluation:")
    print("   - Use multiple judge models")
    print("   - Majority voting for final scores")
    print("   - More robust evaluation")
    
    print("\n2. Custom Metrics:")
    print("   - Define your own evaluation criteria")
    print("   - Natural language prompt templates")
    print("   - Structured output schemas")
    
    print("\n3. Domain-Specific Templates:")
    print("   - Pre-built metrics for different domains")
    print("   - Financial advice, medical advice, etc.")
    print("   - Easy to extend and customize")
    
    print("\n4. MLflow Integration:")
    print("   - Automatic experiment tracking")
    print("   - Metric visualization")
    print("   - Artifact storage")
    print("   - Run comparison")
    
    print("\n5. Response Generation:")
    print("   - Multiple response generation methods")
    print("   - API integration support")
    print("   - Batch processing")


def main():
    """Main demo function."""
    print("🎯 Universal Evaluation Template System - 4 Metrics Demo")
    print("=" * 70)
    
    # Create demo files
    dataset = create_demo_dataset()
    config = create_demo_configuration()
    
    # Show usage instructions
    show_usage_instructions()
    
    # Show configuration details
    show_configuration_details()
    
    # Show metric examples
    show_metric_examples()
    
    # Show advanced features
    show_advanced_features()
    
    print("\n📁 Generated Demo Files:")
    print("  - demo_dataset.json (sample dataset)")
    print("  - demo_config.yaml (evaluation configuration)")
    
    print("\n🎉 Demo Complete!")
    print("\nThe Universal Evaluation Template System is ready to use with:")
    print("  ✅ 4 comprehensive evaluation metrics")
    print("  ✅ Flexible configuration system")
    print("  ✅ MLflow integration")
    print("  ✅ Custom metric support")
    print("  ✅ Domain-specific templates")
    print("  ✅ Response generation options")
    
    print("\n🚀 Get started by running:")
    print("   python quick_start.py")


if __name__ == "__main__":
    main()