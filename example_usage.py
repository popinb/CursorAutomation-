"""
Example Usage of the Universal Evaluation Template System
========================================================

This file demonstrates how to use the evaluation system with different types of datasets
and evaluation scenarios.
"""

import asyncio
import pandas as pd
from pathlib import Path
from template_evaluation_system import UniversalEvaluator, EvaluationConfig, ResponseGenerator
from generate_config import generate_config
import yaml


def example_1_basic_evaluation():
    """Example 1: Basic evaluation with a simple dataset."""
    print("=== Example 1: Basic Evaluation ===")
    
    # Create a sample dataset
    sample_data = {
        "prompt": [
            "What is the capital of France?",
            "How do I bake a chocolate cake?",
            "What are the benefits of exercise?"
        ],
        "response": [
            "The capital of France is Paris.",
            "To bake a chocolate cake, you need flour, sugar, eggs, cocoa powder, and butter. Mix dry ingredients, add wet ingredients, and bake at 350°F for 30 minutes.",
            "Exercise provides numerous benefits including improved cardiovascular health, stronger muscles, better mental health, and increased energy levels."
        ],
        "ground_truth": [
            "Paris",
            "Mix ingredients and bake at 350°F",
            "Improved health, strength, and energy"
        ]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv("sample_dataset.csv", index=False)
    print("✅ Created sample dataset: sample_dataset.csv")
    
    # Generate configuration
    config_dict = generate_config(
        dataset_path="sample_dataset.csv",
        prompt_column="prompt",
        response_column="response",
        experiment_name="basic_evaluation_example",
        metrics=["accuracy", "relevance", "helpfulness"],
        ground_truth_column="ground_truth"
    )
    
    # Save configuration
    with open("example_config.yaml", "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    print("✅ Created configuration: example_config.yaml")
    
    # Run evaluation
    config = EvaluationConfig("example_config.yaml")
    evaluator = UniversalEvaluator(config)
    results = evaluator.evaluate_dataset(df)
    
    print("\n📊 Evaluation Results:")
    print(results[["prompt", "accuracy", "relevance", "helpfulness"]].head())
    
    return results


def example_2_personalized_evaluation():
    """Example 2: Evaluation with personalization features."""
    print("\n=== Example 2: Personalized Evaluation ===")
    
    # Create a sample dataset with personalization
    sample_data = {
        "prompt": [
            "What mortgage rate can I get?",
            "Should I invest in stocks?",
            "How much should I save for retirement?"
        ],
        "response": [
            "Based on your credit score of 750 and income of $80,000, you could qualify for a mortgage rate around 3.5-4.0%.",
            "Given your age of 28 and risk tolerance, a diversified stock portfolio could be appropriate for long-term growth.",
            "With your current income of $80,000, aim to save 15-20% of your income, which would be $12,000-$16,000 annually."
        ],
        "user_features": [
            '{"credit_score": 750, "income": 80000, "age": 28, "location": "California"}',
            '{"age": 28, "risk_tolerance": "moderate", "investment_experience": "beginner"}',
            '{"income": 80000, "age": 28, "current_savings": 5000, "retirement_goal": 1000000}'
        ]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv("personalized_dataset.csv", index=False)
    print("✅ Created personalized dataset: personalized_dataset.csv")
    
    # Generate configuration with personalization metrics
    config_dict = generate_config(
        dataset_path="personalized_dataset.csv",
        prompt_column="prompt",
        response_column="response",
        experiment_name="personalized_evaluation_example",
        metrics=["personalization", "helpfulness", "accuracy"],
        user_features_column="user_features"
    )
    
    # Save configuration
    with open("personalized_config.yaml", "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    print("✅ Created personalized configuration: personalized_config.yaml")
    
    # Run evaluation
    config = EvaluationConfig("personalized_config.yaml")
    evaluator = UniversalEvaluator(config)
    results = evaluator.evaluate_dataset(df)
    
    print("\n📊 Personalized Evaluation Results:")
    print(results[["prompt", "personalization", "helpfulness", "accuracy"]].head())
    
    return results


def example_3_domain_specific_evaluation():
    """Example 3: Domain-specific evaluation for financial advice."""
    print("\n=== Example 3: Domain-Specific Financial Evaluation ===")
    
    # Create a financial advice dataset
    sample_data = {
        "prompt": [
            "Should I invest all my money in cryptocurrency?",
            "What's the best way to pay off my credit card debt?",
            "Is it better to rent or buy a home?"
        ],
        "response": [
            "Cryptocurrency is highly volatile and risky. I recommend diversifying your investments and only investing what you can afford to lose. Consider consulting a financial advisor.",
            "Start by paying off the highest interest rate cards first while making minimum payments on others. Consider a balance transfer card or personal loan for lower rates.",
            "The decision depends on your financial situation, location, and long-term plans. Generally, buying makes sense if you plan to stay 5+ years and can afford the down payment."
        ],
        "user_features": [
            '{"age": 25, "income": 50000, "risk_tolerance": "high", "investment_experience": "beginner"}',
            '{"debt_amount": 15000, "credit_score": 680, "monthly_income": 4000}',
            '{"age": 30, "income": 75000, "savings": 25000, "location": "San Francisco"}'
        ]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv("financial_dataset.csv", index=False)
    print("✅ Created financial dataset: financial_dataset.csv")
    
    # Generate configuration with financial domain metrics
    config_dict = generate_config(
        dataset_path="financial_dataset.csv",
        prompt_column="prompt",
        response_column="response",
        experiment_name="financial_evaluation_example",
        metrics=["risk_assessment", "regulatory_compliance", "helpfulness"],
        domain="financial_advice",
        user_features_column="user_features"
    )
    
    # Save configuration
    with open("financial_config.yaml", "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    print("✅ Created financial configuration: financial_config.yaml")
    
    # Run evaluation
    config = EvaluationConfig("financial_config.yaml")
    evaluator = UniversalEvaluator(config)
    results = evaluator.evaluate_dataset(df)
    
    print("\n📊 Financial Evaluation Results:")
    print(results[["prompt", "risk_assessment", "regulatory_compliance", "helpfulness"]].head())
    
    return results


def example_4_custom_metrics():
    """Example 4: Using custom metrics not in the template."""
    print("\n=== Example 4: Custom Metrics ===")
    
    # Create a dataset for testing custom metrics
    sample_data = {
        "prompt": [
            "Tell me a joke",
            "What's the weather like?",
            "Help me with my homework"
        ],
        "response": [
            "Why don't scientists trust atoms? Because they make up everything!",
            "I don't have access to real-time weather data, but you can check your local weather app or website.",
            "I'd be happy to help! What subject and specific question do you need assistance with?"
        ],
        "tone": ["humorous", "informative", "helpful"],
        "length": ["short", "medium", "medium"]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv("custom_dataset.csv", index=False)
    print("✅ Created custom dataset: custom_dataset.csv")
    
    # Create custom configuration with custom metrics
    custom_config = {
        "dataset": {
            "path": "custom_dataset.csv",
            "prompt_column": "prompt",
            "response_column": "response"
        },
        "models": {
            "response_model": "GOLDEN_RESPONSE",
            "judge_models": ["gpt-4o"]
        },
        "api_keys": {
            "openai_api_key": "your-openai-api-key",
            "openai_base_url": "https://api.openai.com/v1"
        },
        "evaluation": {
            "experiment_name": "custom_metrics_example",
            "run_name": "custom_run_1",
            "metrics": {
                "tone_appropriateness": {
                    "prompt_template": """
You are an impartial evaluator.
Evaluate if the response tone matches the expected tone.

User Query: {prompt}
Model Response: {response}
Expected Tone: {tone}

Rate tone appropriateness from 1-5 where:
1 = Completely inappropriate tone
2 = Mostly inappropriate tone
3 = Neutral tone
4 = Mostly appropriate tone
5 = Perfectly appropriate tone

Return JSON: {{"tone_appropriateness_score": <1-5>, "explanation": "<reasoning>"}}
""",
                    "threshold": 3.0,
                    "output_schema": {
                        "fields": {
                            "tone_appropriateness_score": "int",
                            "explanation": "str"
                        }
                    }
                },
                "response_length": {
                    "prompt_template": """
You are an impartial evaluator.
Evaluate if the response length is appropriate for the query.

User Query: {prompt}
Model Response: {response}
Expected Length: {length}

Rate length appropriateness from 1-5 where:
1 = Completely inappropriate length
2 = Mostly inappropriate length
3 = Neutral length
4 = Mostly appropriate length
5 = Perfectly appropriate length

Return JSON: {{"response_length_score": <1-5>, "explanation": "<reasoning>"}}
""",
                    "threshold": 3.0,
                    "output_schema": {
                        "fields": {
                            "response_length_score": "int",
                            "explanation": "str"
                        }
                    }
                }
            }
        }
    }
    
    # Save custom configuration
    with open("custom_config.yaml", "w") as f:
        yaml.dump(custom_config, f, default_flow_style=False, indent=2)
    print("✅ Created custom configuration: custom_config.yaml")
    
    # Run evaluation
    config = EvaluationConfig("custom_config.yaml")
    evaluator = UniversalEvaluator(config)
    results = evaluator.evaluate_dataset(df)
    
    print("\n📊 Custom Metrics Evaluation Results:")
    print(results[["prompt", "tone_appropriateness", "response_length"]].head())
    
    return results


def main():
    """Run all examples."""
    print("🚀 Universal Evaluation Template System - Examples")
    print("=" * 60)
    
    try:
        # Run examples
        example_1_basic_evaluation()
        example_2_personalized_evaluation()
        example_3_domain_specific_evaluation()
        example_4_custom_metrics()
        
        print("\n✅ All examples completed successfully!")
        print("\n📁 Generated files:")
        print("  - sample_dataset.csv")
        print("  - personalized_dataset.csv")
        print("  - financial_dataset.csv")
        print("  - custom_dataset.csv")
        print("  - example_config.yaml")
        print("  - personalized_config.yaml")
        print("  - financial_config.yaml")
        print("  - custom_config.yaml")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()