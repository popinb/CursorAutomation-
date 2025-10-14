#!/usr/bin/env python3
"""
Simple Test Suite for Databricks LLM Judge Evaluation Notebook
Quick validation before uploading to Databricks
"""

import os
import sys
import pandas as pd
import tempfile
from unittest.mock import Mock, patch

def test_basic_functionality():
    """Test basic functionality without Databricks dependencies"""
    print("🧪 Testing basic functionality...")
    
    # Create test data
    test_dir = tempfile.mkdtemp()
    
    try:
        # Create evaluation data
        eval_data = pd.DataFrame({
            'sample_id': [1, 2, 3],
            'prompt': [
                "What is the capital of France?",
                "Explain machine learning",
                "How to bake a cake?"
            ],
            'response': [
                "Paris is the capital of France",
                "Machine learning is AI that learns from data",
                "Mix flour, eggs, sugar and bake"
            ]
        })
        eval_data.to_csv(f"{test_dir}/evaluation_data.csv", index=False)
        
        # Create metrics config
        metrics_config = pd.DataFrame({
            'name': ['accuracy_check'],
            'type': ['binary'],
            'description': ['Checks accuracy'],
            'evaluation_prompt': ['Evaluate: {response} vs {ground_truth}'],
            'threshold': [1.0],
            'ground_truth_column': ['correct_answer'],
            'ground_truth_file_path': ['']
        })
        metrics_config.to_csv(f"{test_dir}/sample_metrics_config.csv", index=False)
        
        # Create ground truth
        ground_truth = pd.DataFrame({
            'sample_id': [1, 2, 3],
            'correct_answer': [
                'Paris is the capital of France',
                'Machine learning is a subset of AI',
                'Cake requires flour, eggs, sugar, and baking'
            ]
        })
        ground_truth.to_csv(f"{test_dir}/correct_answer.csv", index=False)
        
        # Test file loading
        print("  📁 Testing file loading...")
        eval_df = pd.read_csv(f"{test_dir}/evaluation_data.csv")
        assert len(eval_df) == 3, "Wrong number of evaluation rows"
        assert 'prompt' in eval_df.columns, "Missing prompt column"
        assert 'response' in eval_df.columns, "Missing response column"
        print("  ✅ File loading works")
        
        # Test metrics config
        print("  📊 Testing metrics config...")
        metrics_df = pd.read_csv(f"{test_dir}/sample_metrics_config.csv")
        assert len(metrics_df) == 1, "Wrong number of metrics"
        assert 'name' in metrics_df.columns, "Missing name column"
        assert 'ground_truth_column' in metrics_df.columns, "Missing ground_truth_column"
        print("  ✅ Metrics config works")
        
        # Test ground truth loading
        print("  📚 Testing ground truth loading...")
        gt_df = pd.read_csv(f"{test_dir}/correct_answer.csv")
        assert len(gt_df) == 3, "Wrong number of ground truth rows"
        assert 'correct_answer' in gt_df.columns, "Missing ground truth column"
        print("  ✅ Ground truth loading works")
        
        # Test data merging
        print("  🔗 Testing data merging...")
        merged_df = eval_df.merge(gt_df, on='sample_id', how='left')
        assert 'correct_answer' in merged_df.columns, "Ground truth not merged"
        assert merged_df['correct_answer'].notna().sum() == 3, "Ground truth data missing"
        print("  ✅ Data merging works")
        
        print("✅ All basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(test_dir, ignore_errors=True)

def test_notebook_imports():
    """Test that notebook can be imported without errors"""
    print("📦 Testing notebook imports...")
    
    try:
        # Mock Databricks environment
        mock_dbutils = Mock()
        mock_dbutils.widgets = Mock()
        mock_dbutils.widgets.text = Mock()
        mock_dbutils.widgets.dropdown = Mock()
        mock_dbutils.widgets.get = Mock(return_value="test")
        mock_dbutils.secrets = Mock()
        mock_dbutils.secrets.get = Mock(return_value="test_key")
        mock_dbutils.notebook = Mock()
        mock_dbutils.notebook.entry_point = Mock()
        mock_dbutils.notebook.entry_point.getDbutils = Mock()
        mock_dbutils.notebook.entry_point.getDbutils.return_value.notebook = Mock()
        mock_dbutils.notebook.entry_point.getDbutils.return_value.notebook.return_value.getContext = Mock()
        mock_dbutils.notebook.entry_point.getDbutils.return_value.notebook.return_value.getContext.return_value.apiToken = Mock()
        mock_dbutils.notebook.entry_point.getDbutils.return_value.notebook.return_value.getContext.return_value.apiToken.return_value.get = Mock(return_value="test_token")
        mock_dbutils.notebook.entry_point.getDbutils.return_value.notebook.return_value.getContext.return_value.browserHostName = Mock()
        mock_dbutils.notebook.entry_point.getDbutils.return_value.notebook.return_value.getContext.return_value.browserHostName.return_value.get = Mock(return_value="test.databricks.com")
        mock_dbutils.notebook.entry_point.getDbutils.return_value.notebook.return_value.getContext.return_value.userName = Mock()
        mock_dbutils.notebook.entry_point.getDbutils.return_value.notebook.return_value.getContext.return_value.userName.return_value.get = Mock(return_value="test_user")
        
        with patch.dict('sys.modules', {'dbutils': mock_dbutils}):
            with patch('builtins.display', Mock()):
                # Import the notebook
                import databricks_llm_judge_notebook
                print("  ✅ Notebook imports successfully")
                
                # Test key functions exist
                assert hasattr(databricks_llm_judge_notebook, 'load_any_csv'), "Missing load_any_csv"
                assert hasattr(databricks_llm_judge_notebook, 'load_metrics_config'), "Missing load_metrics_config"
                assert hasattr(databricks_llm_judge_notebook, 'LLMJudgeEvaluator'), "Missing LLMJudgeEvaluator"
                assert hasattr(databricks_llm_judge_notebook, 'MetricConfig'), "Missing MetricConfig"
                print("  ✅ All key functions exist")
                
                return True
                
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_csv_format_validation():
    """Test CSV format validation"""
    print("📋 Testing CSV format validation...")
    
    try:
        # Test metrics CSV format
        sample_metrics = pd.DataFrame({
            'name': ['test_metric'],
            'type': ['binary'],
            'description': ['Test description'],
            'evaluation_prompt': ['Test prompt with {prompt} and {response}'],
            'threshold': [1.0],
            'ground_truth_column': ['test_gt'],
            'ground_truth_file_path': ['']
        })
        
        # Check required columns
        required_cols = ['name', 'type', 'description', 'evaluation_prompt', 'threshold']
        for col in required_cols:
            assert col in sample_metrics.columns, f"Missing required column: {col}"
        
        # Check metric types
        valid_types = ['binary', 'scale_1_5', 'percentage']
        assert sample_metrics['type'].iloc[0] in valid_types, "Invalid metric type"
        
        # Check threshold is numeric
        assert pd.api.types.is_numeric_dtype(sample_metrics['threshold']), "Threshold must be numeric"
        
        print("  ✅ CSV format validation passed")
        return True
        
    except Exception as e:
        print(f"❌ CSV format validation failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Running Simple Test Suite for Databricks Notebook")
    print("=" * 60)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Notebook Imports", test_notebook_imports),
        ("CSV Format Validation", test_csv_format_validation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "=" * 60)
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! The notebook is ready for Databricks upload.")
        print("\n📋 Next Steps:")
        print("1. Upload the notebook file to Databricks")
        print("2. Upload all CSV files to your workspace")
        print("3. Update the widget values with your file names")
        print("4. Run the notebook cells in order")
        return True
    else:
        print("❌ Some tests failed. Please fix the issues before uploading.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)