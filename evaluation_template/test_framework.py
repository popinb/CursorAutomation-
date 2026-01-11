#!/usr/bin/env python3
"""
Comprehensive test suite for the LLM Evaluation Template

This test bypasses actual LLM calls and tests the framework functionality
with mock responses to validate the complete pipeline.
"""

import sys
import os
import json
import tempfile
from unittest.mock import Mock, patch
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

def test_configuration_loading():
    """Test configuration loading and validation."""
    print("🔧 Testing configuration loading...")
    
    from evaluation_template import EvaluationConfig
    
    try:
        config = EvaluationConfig('config/zillow_copilot_evaluation.yaml')
        
        # Test dataset config
        dataset_config = config.dataset_config
        assert 'file_path' in dataset_config
        assert 'columns' in dataset_config
        
        # Test metrics config
        metrics_config = config.metrics_config
        expected_metrics = ['structured_presentation', 'personalization_accuracy', 
                          'actionability_guidance', 'coherence']
        for metric in expected_metrics:
            assert metric in metrics_config
            assert 'prompt_template' in metrics_config[metric]
            assert 'score_key' in metrics_config[metric]
        
        # Test composite config
        composite_config = config.composite_config
        assert 'copilot_readiness' in composite_config
        assert 'overall_quality' in composite_config
        
        print("✅ Configuration loading test passed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration loading test failed: {e}")
        return False


def test_data_loading():
    """Test data loading and validation."""
    print("🗂️  Testing data loading...")
    
    from evaluation_template import EvaluationConfig, DataLoader
    
    try:
        config = EvaluationConfig('config/zillow_copilot_evaluation.yaml')
        data_loader = DataLoader(config)
        
        # Load dataset
        df = data_loader.load_dataset()
        
        # Validate dataset structure
        assert len(df) > 0, "Dataset should not be empty"
        assert 'user_query' in df.columns, "Original column should exist"
        assert 'prompt' in df.columns, "Mapped column should exist"
        assert 'response' in df.columns, "Response column should exist"
        
        # Test validation
        assert data_loader.validate_dataset(df), "Dataset validation should pass"
        
        print(f"✅ Data loading test passed - {len(df)} rows loaded")
        return True
        
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False


def test_mock_evaluation():
    """Test evaluation pipeline with mocked LLM responses."""
    print("🤖 Testing evaluation pipeline with mock responses...")
    
    from evaluation_template import EvaluationConfig, DataLoader, LLMJudge
    
    try:
        config = EvaluationConfig('config/zillow_copilot_evaluation.yaml')
        data_loader = DataLoader(config)
        df = data_loader.load_dataset()
        
        # Create mock LLM judge with fake models
        llm_judge = LLMJudge.__new__(LLMJudge)  # Create without calling __init__
        llm_judge.config = config
        llm_judge.judges_config = config.judges_config
        
        # Create mock models
        mock_model1 = Mock()
        mock_model2 = Mock()
        mock_model1.model_name = "mock-gpt-4o"
        mock_model2.model_name = "mock-gpt-4o-mini"
        llm_judge.models = [mock_model1, mock_model2]
        
        # Test each metric with mock responses
        test_results = {}
        
        # Mock responses for different metrics
        mock_responses = {
            'structured_presentation': {
                'structured_presentation_score': 4,
                'explanation': 'Well organized with clear headings and bullet points'
            },
            'personalization_accuracy': {
                'personalization_accuracy_score': 1,
                'explanation': 'All financial facts are accurate'
            },
            'actionability_guidance': {
                'actionability_guidance_score': 1,
                'explanation': 'Clear next steps provided'
            },
            'coherence': {
                'coherence_score': 1,
                'explanation': 'Logically consistent throughout'
            }
        }
        
        # Test each metric
        for metric_name, metric_config in config.metrics_config.items():
            if not metric_config.get('enabled', True):
                continue
                
            print(f"   Testing {metric_name}...")
            
            # Mock the LLM responses
            mock_response = Mock()
            mock_response.content = json.dumps(mock_responses[metric_name])
            mock_model1.invoke.return_value = mock_response
            mock_model2.invoke.return_value = mock_response
            
            # Run evaluation for this metric
            eval_results = llm_judge.evaluate_single_metric(metric_name, metric_config, df.head(2))
            
            # Validate results
            assert 'scores' in eval_results
            assert 'details' in eval_results
            assert 'mean_score' in eval_results
            assert len(eval_results['scores']) == 2  # Test with 2 rows
            
            test_results[metric_name] = eval_results['mean_score']
            print(f"     Mean score: {eval_results['mean_score']}")
        
        print(f"✅ Mock evaluation test passed - {len(test_results)} metrics evaluated")
        return True
        
    except Exception as e:
        print(f"❌ Mock evaluation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_composite_scoring():
    """Test composite scoring functionality."""
    print("📊 Testing composite scoring...")
    
    from evaluation_template import EvaluationConfig, CompositeScorer
    import pandas as pd
    
    try:
        config = EvaluationConfig('config/zillow_copilot_evaluation.yaml')
        composite_scorer = CompositeScorer(config)
        
        # Create mock data with metric scores
        test_data = {
            'structured_presentation': [4, 3, 5, 2],
            'personalization_accuracy': [1, 1, 0, 1],
            'actionability_guidance': [1, 0, 1, 1],
            'coherence': [1, 1, 1, 0]
        }
        df = pd.DataFrame(test_data)
        
        # Compute composite scores
        df_result = composite_scorer.compute_composite_scores(df)
        
        # Validate composite scores exist
        expected_composites = ['copilot_readiness', 'delivery_effectiveness', 'overall_quality']
        for composite in expected_composites:
            assert composite in df_result.columns, f"Composite score {composite} missing"
            assert len(df_result[composite]) == 4, "Composite score should have same length as input"
        
        # Test specific composite logic
        # copilot_readiness should be minimum of binary metrics
        expected_readiness = [1, 0, 0, 0]  # min of [1,0,1,1], [1,1,0,1], [1,1,1,0]
        assert list(df_result['copilot_readiness']) == expected_readiness
        
        print("✅ Composite scoring test passed")
        print(f"   Sample overall_quality scores: {list(df_result['overall_quality'])}")
        return True
        
    except Exception as e:
        print(f"❌ Composite scoring test failed: {e}")
        return False


def test_mlflow_integration():
    """Test MLflow integration without actually logging."""
    print("📈 Testing MLflow integration...")
    
    try:
        # Test that MLflow imports and basic functionality works
        import mlflow
        import mlflow.metrics.genai
        
        # Test experiment setup (without actually creating)
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set temporary tracking URI
            temp_tracking_uri = f"file://{temp_dir}/mlruns"
            original_uri = mlflow.get_tracking_uri()
            
            try:
                mlflow.set_tracking_uri(temp_tracking_uri)
                
                # Test experiment creation
                experiment_name = "test_evaluation_experiment"
                experiment = mlflow.set_experiment(experiment_name)
                
                # Test run creation
                with mlflow.start_run(run_name="test_run") as run:
                    # Test metric logging
                    mlflow.log_metric("test_metric", 0.85)
                    mlflow.log_param("test_param", "test_value")
                    
                    # Test artifact logging
                    test_dict = {"test": "data"}
                    mlflow.log_dict(test_dict, "test_config.yaml")
                
                print("✅ MLflow integration test passed")
                return True
                
            finally:
                # Restore original tracking URI
                mlflow.set_tracking_uri(original_uri)
        
    except Exception as e:
        print(f"❌ MLflow integration test failed: {e}")
        return False


def test_error_handling():
    """Test error handling and edge cases."""
    print("⚠️  Testing error handling...")
    
    from evaluation_template import EvaluationConfig, DataLoader
    
    try:
        # Test with non-existent config file
        try:
            config = EvaluationConfig('non_existent_config.yaml')
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            pass  # Expected
        
        # Test with non-existent dataset
        config = EvaluationConfig('config/zillow_copilot_evaluation.yaml')
        config.config['dataset']['file_path'] = 'non_existent_file.csv'
        data_loader = DataLoader(config)
        
        try:
            df = data_loader.load_dataset()
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            pass  # Expected
        
        print("✅ Error handling test passed")
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False


def test_metric_templates():
    """Test that metric templates are properly formatted."""
    print("📝 Testing metric templates...")
    
    from evaluation_template import EvaluationConfig
    
    try:
        config = EvaluationConfig('config/zillow_copilot_evaluation.yaml')
        
        for metric_name, metric_config in config.metrics_config.items():
            # Check required fields
            assert 'prompt_template' in metric_config
            assert 'score_key' in metric_config
            assert 'threshold' in metric_config
            assert 'scale' in metric_config
            
            # Check prompt template has required placeholders
            template = metric_config['prompt_template']
            assert '{prompt}' in template
            assert '{response}' in template
            
            # Check JSON format is mentioned
            assert 'json' in template.lower()
            assert metric_config['score_key'] in template
        
        print("✅ Metric templates test passed")
        return True
        
    except Exception as e:
        print(f"❌ Metric templates test failed: {e}")
        return False


def test_example_configurations():
    """Test that example configurations are valid."""
    print("📋 Testing example configurations...")
    
    from evaluation_template import EvaluationConfig
    
    example_configs = [
        'examples/chatbot_evaluation.yaml',
        'examples/qa_evaluation.yaml', 
        'examples/creative_writing_evaluation.yaml'
    ]
    
    try:
        for config_file in example_configs:
            if os.path.exists(config_file):
                config = EvaluationConfig(config_file)
                
                # Basic validation
                assert config.dataset_config
                assert config.metrics_config
                assert config.experiment_config
                
                print(f"   ✅ {config_file} is valid")
            else:
                print(f"   ⚠️  {config_file} not found, skipping")
        
        print("✅ Example configurations test passed")
        return True
        
    except Exception as e:
        print(f"❌ Example configurations test failed: {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("🚀 Running comprehensive evaluation template tests...\n")
    
    tests = [
        test_configuration_loading,
        test_data_loading,
        test_mock_evaluation,
        test_composite_scoring,
        test_mlflow_integration,
        test_error_handling,
        test_metric_templates,
        test_example_configurations
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print()
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
            print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("="*60)
    print("🎯 TEST SUMMARY")
    print("="*60)
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! The evaluation template is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the output above.")
    
    return passed == total


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)