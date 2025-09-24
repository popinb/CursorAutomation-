"""
Test Script for Universal Evaluation Template System
===================================================

This script tests the basic functionality of the evaluation system.
"""

import os
import tempfile
import pandas as pd
import yaml
from pathlib import Path
from template_evaluation_system import UniversalEvaluator, EvaluationConfig
from generate_config import generate_config


def test_basic_functionality():
    """Test basic evaluation functionality."""
    print("🧪 Testing basic functionality...")
    
    # Create a simple test dataset
    test_data = {
        "prompt": [
            "What is 2+2?",
            "What is the capital of France?"
        ],
        "response": [
            "2+2 equals 4.",
            "The capital of France is Paris."
        ],
        "ground_truth": [
            "4",
            "Paris"
        ]
    }
    
    df = pd.DataFrame(test_data)
    
    # Create temporary files
    with tempfile.TemporaryDirectory() as temp_dir:
        dataset_path = os.path.join(temp_dir, "test_dataset.csv")
        config_path = os.path.join(temp_dir, "test_config.yaml")
        
        # Save test dataset
        df.to_csv(dataset_path, index=False)
        
        # Generate configuration
        config_dict = generate_config(
            dataset_path=dataset_path,
            prompt_column="prompt",
            response_column="response",
            experiment_name="test_experiment",
            metrics=["accuracy", "relevance"],
            ground_truth_column="ground_truth",
            openai_api_key="test-key"  # Mock key for testing
        )
        
        # Save configuration
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        # Test configuration loading
        try:
            config = EvaluationConfig(config_path)
            print("✅ Configuration loading: PASSED")
        except Exception as e:
            print(f"❌ Configuration loading: FAILED - {e}")
            return False
        
        # Test evaluator creation
        try:
            evaluator = UniversalEvaluator(config)
            print("✅ Evaluator creation: PASSED")
        except Exception as e:
            print(f"❌ Evaluator creation: FAILED - {e}")
            return False
        
        # Test prompt formatting
        try:
            test_row = df.iloc[0]
            formatted_prompt = evaluator._format_prompt(
                "User Query: {prompt}\nResponse: {response}",
                test_row
            )
            assert "What is 2+2?" in formatted_prompt
            assert "2+2 equals 4" in formatted_prompt
            print("✅ Prompt formatting: PASSED")
        except Exception as e:
            print(f"❌ Prompt formatting: FAILED - {e}")
            return False
    
    print("✅ Basic functionality test: PASSED")
    return True


def test_metric_templates():
    """Test metric template functionality."""
    print("\n🧪 Testing metric templates...")
    
    from metric_templates import (
        METRIC_TEMPLATES, 
        DOMAIN_TEMPLATES, 
        get_metric_template,
        list_available_metrics,
        list_available_domains
    )
    
    # Test general metrics
    try:
        assert "accuracy" in METRIC_TEMPLATES
        assert "relevance" in METRIC_TEMPLATES
        print("✅ General metrics: PASSED")
    except Exception as e:
        print(f"❌ General metrics: FAILED - {e}")
        return False
    
    # Test domain metrics
    try:
        assert "financial_advice" in DOMAIN_TEMPLATES
        assert "medical_advice" in DOMAIN_TEMPLATES
        print("✅ Domain metrics: PASSED")
    except Exception as e:
        print(f"❌ Domain metrics: FAILED - {e}")
        return False
    
    # Test metric retrieval
    try:
        accuracy_template = get_metric_template("accuracy")
        assert "prompt_template" in accuracy_template
        assert "threshold" in accuracy_template
        print("✅ Metric retrieval: PASSED")
    except Exception as e:
        print(f"❌ Metric retrieval: FAILED - {e}")
        return False
    
    # Test listing functions
    try:
        metrics = list_available_metrics()
        assert len(metrics) > 0
        domains = list_available_domains()
        assert len(domains) > 0
        print("✅ Listing functions: PASSED")
    except Exception as e:
        print(f"❌ Listing functions: FAILED - {e}")
        return False
    
    print("✅ Metric templates test: PASSED")
    return True


def test_configuration_generator():
    """Test configuration generation."""
    print("\n🧪 Testing configuration generator...")
    
    # Test basic configuration generation
    try:
        config = generate_config(
            dataset_path="test.csv",
            prompt_column="prompt",
            response_column="response",
            experiment_name="test",
            metrics=["accuracy", "relevance"]
        )
        
        assert "dataset" in config
        assert "models" in config
        assert "evaluation" in config
        assert config["dataset"]["path"] == "test.csv"
        print("✅ Basic config generation: PASSED")
    except Exception as e:
        print(f"❌ Basic config generation: FAILED - {e}")
        return False
    
    # Test domain-specific configuration
    try:
        config = generate_config(
            dataset_path="test.csv",
            prompt_column="prompt",
            response_column="response",
            experiment_name="test",
            metrics=["risk_assessment"],
            domain="financial_advice"
        )
        
        assert "risk_assessment" in config["evaluation"]["metrics"]
        print("✅ Domain config generation: PASSED")
    except Exception as e:
        print(f"❌ Domain config generation: FAILED - {e}")
        return False
    
    print("✅ Configuration generator test: PASSED")
    return True


def test_yaml_serialization():
    """Test YAML serialization and deserialization."""
    print("\n🧪 Testing YAML serialization...")
    
    # Test configuration round-trip
    try:
        original_config = {
            "dataset": {
                "path": "test.csv",
                "prompt_column": "prompt",
                "response_column": "response"
            },
            "models": {
                "response_model": "GOLDEN_RESPONSE",
                "judge_models": ["gpt-4o"]
            },
            "evaluation": {
                "experiment_name": "test",
                "metrics": {
                    "accuracy": {
                        "prompt_template": "Test template",
                        "threshold": 3.0
                    }
                }
            }
        }
        
        # Serialize to YAML
        yaml_str = yaml.dump(original_config, default_flow_style=False, indent=2)
        
        # Deserialize from YAML
        loaded_config = yaml.safe_load(yaml_str)
        
        assert loaded_config == original_config
        print("✅ YAML serialization: PASSED")
    except Exception as e:
        print(f"❌ YAML serialization: FAILED - {e}")
        return False
    
    print("✅ YAML serialization test: PASSED")
    return True


def main():
    """Run all tests."""
    print("🚀 Universal Evaluation Template System - Test Suite")
    print("=" * 60)
    
    tests = [
        test_basic_functionality,
        test_metric_templates,
        test_configuration_generator,
        test_yaml_serialization
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__}: FAILED with exception - {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)