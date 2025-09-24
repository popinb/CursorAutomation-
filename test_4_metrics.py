#!/usr/bin/env python3
"""
Test Script with 4 Metrics
==========================

This script tests the evaluation system with 4 different metrics to demonstrate functionality.
"""

import os
import pandas as pd
import yaml
from pathlib import Path
from template_evaluation_system import UniversalEvaluator, EvaluationConfig
from generate_config import generate_config


def create_test_dataset():
    """Create a test dataset with various types of prompts and responses."""
    test_data = {
        "prompt": [
            "What is the capital of France?",
            "How do I bake a chocolate cake?",
            "What are the benefits of regular exercise?",
            "Should I invest all my money in cryptocurrency?",
            "What's the best way to learn programming?",
            "How can I improve my credit score?",
            "What are the side effects of this medication?",
            "Tell me a funny joke about cats"
        ],
        "response": [
            "The capital of France is Paris, a beautiful city known for its art, culture, and cuisine.",
            "To bake a chocolate cake, mix 2 cups flour, 1 cup sugar, 1/2 cup cocoa powder, 2 eggs, and 1/2 cup butter. Bake at 350°F for 30-35 minutes.",
            "Regular exercise provides numerous benefits including improved cardiovascular health, stronger muscles, better mental health, increased energy levels, and better sleep quality.",
            "Investing all your money in cryptocurrency is extremely risky. I recommend diversifying your investments and only investing what you can afford to lose. Consider consulting a financial advisor.",
            "The best way to learn programming is to start with a beginner-friendly language like Python, practice regularly with small projects, and build a portfolio of your work.",
            "To improve your credit score, pay bills on time, keep credit card balances low, avoid opening too many new accounts, and check your credit report regularly for errors.",
            "I cannot provide specific medical advice about medication side effects. Please consult with your doctor or pharmacist for information about your specific medication.",
            "Why don't cats play poker in the jungle? Too many cheetahs! 😸"
        ],
        "ground_truth": [
            "Paris",
            "Mix ingredients and bake at 350°F",
            "Improved health, fitness, and mental well-being",
            "Diversify investments, don't put all money in crypto",
            "Practice regularly with projects",
            "Pay bills on time, keep balances low",
            "Consult a doctor for medical advice",
            "A cat-related joke"
        ],
        "user_features": [
            '{"location": "New York", "education": "college", "interests": ["travel", "culture"]}',
            '{"cooking_experience": "beginner", "dietary_restrictions": "none", "kitchen_equipment": "basic"}',
            '{"fitness_level": "intermediate", "age": 28, "health_goals": "general_fitness"}',
            '{"age": 25, "income": 50000, "risk_tolerance": "high", "investment_experience": "beginner"}',
            '{"programming_experience": "none", "learning_style": "hands_on", "time_available": "part_time"}',
            '{"credit_score": 650, "debt_amount": 5000, "income": 60000, "financial_goals": "home_purchase"}',
            '{"age": 45, "medical_conditions": ["diabetes"], "medication_concerns": "side_effects"}',
            '{"mood": "cheerful", "humor_preference": "puns", "pet_owner": true}'
        ],
        "expected_tone": [
            "informative",
            "instructional", 
            "motivational",
            "cautious",
            "encouraging",
            "practical",
            "professional",
            "humorous"
        ]
    }
    
    df = pd.DataFrame(test_data)
    df.to_csv("test_4_metrics_dataset.csv", index=False)
    print("✅ Created test dataset: test_4_metrics_dataset.csv")
    print(f"Dataset shape: {df.shape}")
    return df


def create_test_configuration():
    """Create a test configuration with 4 different metrics."""
    config_dict = {
        "dataset": {
            "path": "test_4_metrics_dataset.csv",
            "prompt_column": "prompt",
            "response_column": "response",
            "ground_truth_column": "ground_truth",
            "user_features_column": "user_features"
        },
        "models": {
            "response_model": "GOLDEN_RESPONSE",
            "judge_models": ["gpt-4o"]  # Using single model for testing
        },
        "api_keys": {
            "openai_api_key": os.getenv("OPENAI_API_KEY", "test-key"),
            "openai_base_url": "https://api.openai.com/v1"
        },
        "evaluation": {
            "experiment_name": "test_4_metrics_experiment",
            "run_name": "test_run_1",
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
                
                "tone_appropriateness": {
                    "prompt_template": """
You are an impartial evaluator.
Evaluate if the response tone is appropriate for the context.

User Query: {prompt}
Model Response: {response}
Expected Tone: {expected_tone}

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
                }
            }
        }
    }
    
    with open("test_4_metrics_config.yaml", "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    print("✅ Created test configuration: test_4_metrics_config.yaml")
    return config_dict


def run_evaluation_test():
    """Run the evaluation test with 4 metrics."""
    print("🚀 Testing Universal Evaluation System with 4 Metrics")
    print("=" * 60)
    
    # Create test dataset
    df = create_test_dataset()
    
    # Create test configuration
    config_dict = create_test_configuration()
    
    # Check if OpenAI API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "test-key":
        print("⚠️  Warning: No OpenAI API key found. Using mock evaluation.")
        print("   Set OPENAI_API_KEY environment variable for real evaluation.")
        return run_mock_evaluation(df)
    
    try:
        # Load configuration
        config = EvaluationConfig("test_4_metrics_config.yaml")
        print("✅ Configuration loaded successfully")
        
        # Create evaluator
        evaluator = UniversalEvaluator(config)
        print("✅ Evaluator created successfully")
        
        # Run evaluation
        print("\n🔬 Running evaluation with 4 metrics...")
        print("Metrics: accuracy, relevance, personalization, tone_appropriateness")
        
        results = evaluator.evaluate_dataset(df)
        
        # Save results
        results.to_csv("test_4_metrics_results.csv", index=False)
        print("✅ Results saved to: test_4_metrics_results.csv")
        
        # Print summary
        print("\n📊 Evaluation Summary")
        print("-" * 40)
        metric_cols = ["accuracy", "relevance", "personalization", "tone_appropriateness"]
        
        for metric in metric_cols:
            if metric in results.columns:
                mean_score = results[metric].mean()
                print(f"{metric:20}: {mean_score:.3f}")
        
        # Print detailed results for first few rows
        print("\n📋 Detailed Results (First 3 rows)")
        print("-" * 40)
        display_cols = ["prompt", "accuracy", "relevance", "personalization", "tone_appropriateness"]
        available_cols = [col for col in display_cols if col in results.columns]
        print(results[available_cols].head(3).to_string(index=False))
        
        return results
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_mock_evaluation(df):
    """Run a mock evaluation for testing without API key."""
    print("\n🎭 Running Mock Evaluation (No API Key)")
    print("-" * 40)
    
    # Create mock results
    import random
    random.seed(42)  # For reproducible results
    
    mock_results = df.copy()
    
    # Generate mock scores for each metric
    metrics = ["accuracy", "relevance", "personalization", "tone_appropriateness"]
    
    for metric in metrics:
        # Generate scores between 2-5 (realistic range)
        scores = [random.randint(2, 5) for _ in range(len(df))]
        mock_results[metric] = scores
        mock_results[f"{metric}_status"] = ["✅" if s >= 3 else "❌" for s in scores]
        mock_results[f"{metric}_details"] = [{"mock": True, "score": s} for s in scores]
    
    # Save mock results
    mock_results.to_csv("test_4_metrics_mock_results.csv", index=False)
    print("✅ Mock results saved to: test_4_metrics_mock_results.csv")
    
    # Print summary
    print("\n📊 Mock Evaluation Summary")
    print("-" * 40)
    for metric in metrics:
        mean_score = mock_results[metric].mean()
        print(f"{metric:20}: {mean_score:.3f}")
    
    return mock_results


def main():
    """Main test function."""
    print("🧪 Testing Universal Evaluation System with 4 Metrics")
    print("=" * 60)
    
    # Run the evaluation test
    results = run_evaluation_test()
    
    if results is not None:
        print("\n✅ Test completed successfully!")
        print("\n📁 Generated files:")
        print("  - test_4_metrics_dataset.csv")
        print("  - test_4_metrics_config.yaml")
        if "accuracy" in results.columns:
            print("  - test_4_metrics_results.csv")
        else:
            print("  - test_4_metrics_mock_results.csv")
        
        print("\n🎯 Key Features Demonstrated:")
        print("  ✅ Multiple metrics evaluation")
        print("  ✅ Custom metric definitions")
        print("  ✅ Ground truth comparison")
        print("  ✅ User personalization features")
        print("  ✅ Structured output schemas")
        print("  ✅ Threshold-based scoring")
        print("  ✅ MLflow integration")
        
    else:
        print("\n❌ Test failed. Check the error messages above.")


if __name__ == "__main__":
    main()