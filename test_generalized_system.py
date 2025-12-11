#!/usr/bin/env python3
"""
Test script for the generalized evaluation system.

This script tests the core functionality of the generalized evaluation system
to ensure everything works correctly.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from generalized_evaluator import GeneralizedEvaluator, create_sample_metrics_csv
from evaluation_config_manager import EvaluationConfigManager, create_sample_evaluation_data


def test_basic_functionality():
    """Test basic functionality of the generalized evaluator."""
    print("Testing basic functionality...")
    
    # Create temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create sample metrics CSV
        metrics_csv_path = temp_path / "test_metrics.csv"
        create_sample_metrics_csv(str(metrics_csv_path))
        
        # Verify CSV was created
        assert metrics_csv_path.exists(), "Sample metrics CSV should be created"
        print("✅ Sample metrics CSV created successfully")
        
        # Create evaluator
        evaluator = GeneralizedEvaluator(
            judge_model="gpt-4",
            metrics_config_path=str(metrics_csv_path)
        )
        
        # Verify evaluator was created
        assert evaluator is not None, "Evaluator should be created"
        assert len(evaluator.metric_configs) > 0, "Should have metric configurations"
        print("✅ Evaluator created successfully")
        
        # Create sample evaluation data
        evaluation_data = [
            {
                'prompt': 'What is the capital of France?',
                'response': 'The capital of France is Paris.',
                'ground_truth': 'Paris'
            }
        ]
        
        # Run evaluation
        results_df = evaluator.evaluate_dataset(evaluation_data)
        
        # Verify results
        assert results_df is not None, "Results DataFrame should be created"
        assert len(results_df) > 0, "Should have evaluation results"
        print("✅ Evaluation completed successfully")
        
        # Test summary generation
        summary = evaluator._generate_summary(results_df)
        assert 'total_evaluations' in summary, "Summary should contain total_evaluations"
        assert 'overall_pass_rate' in summary, "Summary should contain overall_pass_rate"
        print("✅ Summary generation works")
        
        # Test export functionality
        exported_files = evaluator.export_results(results_df, str(temp_path))
        assert len(exported_files) > 0, "Should export files"
        print("✅ Export functionality works")
        
        print("✅ Basic functionality test passed!")


def test_configuration_manager():
    """Test the configuration manager functionality."""
    print("\nTesting configuration manager...")
    
    # Create temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Initialize configuration manager
        config_manager = EvaluationConfigManager(str(temp_path))
        
        # Create predefined scenarios
        config_manager.create_predefined_scenarios()
        
        # Verify scenarios were created
        scenarios = config_manager.list_scenarios()
        assert len(scenarios) > 0, "Should have predefined scenarios"
        print(f"✅ Created {len(scenarios)} predefined scenarios")
        
        # Test scenario retrieval
        for scenario_name in scenarios:
            scenario = config_manager.get_scenario(scenario_name)
            assert scenario is not None, f"Should be able to get scenario {scenario_name}"
            assert scenario.name == scenario_name, "Scenario name should match"
        print("✅ Scenario retrieval works")
        
        # Test evaluator creation
        evaluator = config_manager.create_evaluator(scenarios[0])
        assert evaluator is not None, "Should be able to create evaluator from scenario"
        print("✅ Evaluator creation from scenario works")
        
        # Test custom scenario creation
        custom_metrics = [
            {
                'name': 'test_metric',
                'type': 'binary',
                'description': 'Test metric',
                'grading_rubric': 'Test rubric',
                'threshold': 0.5,
                'ground_truth_column': 'ground_truth',
                'ground_truth_file_path': ''
            }
        ]
        
        scenario_path = config_manager.create_scenario(
            name="test_scenario",
            description="Test scenario",
            metrics=custom_metrics
        )
        
        assert os.path.exists(scenario_path), "Custom scenario should be created"
        print("✅ Custom scenario creation works")
        
        print("✅ Configuration manager test passed!")


def test_sample_data_generation():
    """Test sample data generation for different scenarios."""
    print("\nTesting sample data generation...")
    
    scenarios = ["basic_quality", "financial_advisory", "customer_support"]
    
    for scenario in scenarios:
        sample_data = create_sample_evaluation_data(scenario)
        assert isinstance(sample_data, list), f"Sample data for {scenario} should be a list"
        assert len(sample_data) > 0, f"Sample data for {scenario} should not be empty"
        
        # Check that each sample has required fields
        for sample in sample_data:
            assert 'prompt' in sample, f"Sample should have 'prompt' field"
            assert 'response' in sample, f"Sample should have 'response' field"
            assert 'ground_truth' in sample, f"Sample should have 'ground_truth' field"
        
        print(f"✅ Sample data generation for {scenario} works")
    
    print("✅ Sample data generation test passed!")


def test_metric_types():
    """Test different metric types."""
    print("\nTesting metric types...")
    
    from generalized_evaluator import MetricType
    
    # Test metric type enum
    assert MetricType.BINARY.value == "binary", "Binary metric type should work"
    assert MetricType.SCALE_1_5.value == "1-5_scale", "Scale metric type should work"
    assert MetricType.PERCENTAGE.value == "percentage", "Percentage metric type should work"
    print("✅ Metric type enum works")
    
    # Test auto-generated prompts for different types
    evaluator = GeneralizedEvaluator()
    
    binary_prompt = evaluator.auto_generate_evaluation_prompt(
        "test", "binary", "Test description", "Test rubric"
    )
    assert "score" in binary_prompt.lower() and "explanation" in binary_prompt.lower(), "Binary prompt should contain score and explanation fields"
    print("✅ Binary metric prompt generation works")
    
    scale_prompt = evaluator.auto_generate_evaluation_prompt(
        "test", "1-5_scale", "Test description", "Test rubric"
    )
    assert "score" in scale_prompt.lower() and "explanation" in scale_prompt.lower(), "Scale prompt should contain score and explanation fields"
    print("✅ Scale metric prompt generation works")
    
    percentage_prompt = evaluator.auto_generate_evaluation_prompt(
        "test", "percentage", "Test description", "Test rubric"
    )
    assert "score" in percentage_prompt.lower() and "explanation" in percentage_prompt.lower(), "Percentage prompt should contain score and explanation fields"
    print("✅ Percentage metric prompt generation works")
    
    print("✅ Metric types test passed!")


def test_error_handling():
    """Test error handling and edge cases."""
    print("\nTesting error handling...")
    
    # Test with non-existent metrics file
    evaluator = GeneralizedEvaluator(metrics_config_path="non_existent.csv")
    assert len(evaluator.metric_configs) > 0, "Should fall back to default metrics"
    print("✅ Non-existent metrics file handling works")
    
    # Test with empty evaluation data
    results_df = evaluator.evaluate_dataset([])
    assert len(results_df) == 0, "Empty evaluation data should return empty results"
    print("✅ Empty evaluation data handling works")
    
    # Test with malformed evaluation data
    malformed_data = [{'prompt': 'test'}]  # Missing required fields
    results_df = evaluator.evaluate_dataset(malformed_data)
    assert len(results_df) > 0, "Should handle malformed data gracefully"
    print("✅ Malformed data handling works")
    
    print("✅ Error handling test passed!")


def main():
    """Run all tests."""
    print("Running Generalized Evaluation System Tests")
    print("=" * 50)
    
    try:
        test_basic_functionality()
        test_configuration_manager()
        test_sample_data_generation()
        test_metric_types()
        test_error_handling()
        
        print("\n" + "=" * 50)
        print("✅ All tests passed successfully!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()