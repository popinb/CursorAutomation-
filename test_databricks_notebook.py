#!/usr/bin/env python3
"""
Comprehensive Test Suite for Databricks LLM Judge Evaluation Notebook
Tests all functionality before uploading to Databricks workspace
"""

import os
import sys
import pandas as pd
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import traceback

# Add current directory to path for imports
sys.path.append('.')

class DatabricksNotebookTester:
    """Comprehensive test suite for the Databricks notebook"""
    
    def __init__(self):
        self.test_dir = tempfile.mkdtemp(prefix="databricks_test_")
        self.test_results = []
        self.setup_test_environment()
        
    def setup_test_environment(self):
        """Set up test environment with sample files"""
        print("🔧 Setting up test environment...")
        
        # Create test files
        self.create_test_files()
        
        # Mock Databricks environment
        self.mock_databricks_environment()
        
    def create_test_files(self):
        """Create all necessary test files"""
        print("📁 Creating test files...")
        
        # Evaluation data
        eval_data = pd.DataFrame({
            'sample_id': [1, 2, 3],
            'prompt': [
                "What is the capital of France?",
                "Explain machine learning in simple terms",
                "How do I bake a chocolate cake?"
            ],
            'response': [
                "The capital of France is Paris, a beautiful city known for its culture and history.",
                "Machine learning is a type of AI where computers learn patterns from data to make predictions.",
                "To bake a chocolate cake, mix flour, cocoa, eggs, and sugar, then bake at 350°F for 30 minutes."
            ]
        })
        eval_data.to_csv(f"{self.test_dir}/evaluation_data.csv", index=False)
        
        # Metrics configuration
        metrics_config = pd.DataFrame({
            'name': ['accuracy_check', 'helpfulness_rating', 'safety_check'],
            'type': ['binary', 'scale_1_5', 'binary'],
            'description': [
                'Checks if response contains accurate information',
                'Rate how helpful the response is',
                'Checks if the response is safe and appropriate'
            ],
            'evaluation_prompt': [
                'Evaluate accuracy: {response} vs {ground_truth}',
                'Rate helpfulness: {response} vs {ground_truth}',
                'Check safety: {response} vs {ground_truth}'
            ],
            'threshold': [1.0, 3.0, 1.0],
            'ground_truth_column': ['correct_answer', 'helpful_answer', 'safe_response'],
            'ground_truth_file_path': ['', '', '']
        })
        metrics_config.to_csv(f"{self.test_dir}/sample_metrics_config.csv", index=False)
        
        # Ground truth files
        correct_answer = pd.DataFrame({
            'sample_id': [1, 2, 3],
            'correct_answer': [
                'Paris is the capital of France',
                'Machine learning is a subset of artificial intelligence',
                'Chocolate cake requires flour, sugar, eggs, and cocoa powder'
            ]
        })
        correct_answer.to_csv(f"{self.test_dir}/correct_answer.csv", index=False)
        
        helpful_answer = pd.DataFrame({
            'sample_id': [1, 2, 3],
            'helpful_answer': [
                'This response provides clear, accurate information',
                'The explanation is simple and easy to understand',
                'The recipe includes all necessary ingredients'
            ]
        })
        helpful_answer.to_csv(f"{self.test_dir}/helpful_answer.csv", index=False)
        
        safe_response = pd.DataFrame({
            'sample_id': [1, 2, 3],
            'safe_response': [
                'This response is safe and appropriate',
                'No harmful content detected',
                'The response contains only safe cooking instructions'
            ]
        })
        safe_response.to_csv(f"{self.test_dir}/safe_response.csv", index=False)
        
        print(f"✅ Test files created in: {self.test_dir}")
        
    def mock_databricks_environment(self):
        """Mock Databricks environment components"""
        print("🏢 Mocking Databricks environment...")
        
        # Mock dbutils
        self.mock_dbutils = Mock()
        self.mock_dbutils.widgets = Mock()
        self.mock_dbutils.widgets.text = Mock()
        self.mock_dbutils.widgets.dropdown = Mock()
        self.mock_dbutils.widgets.get = Mock(side_effect=self.mock_widget_get)
        self.mock_dbutils.secrets = Mock()
        self.mock_dbutils.secrets.get = Mock(return_value="mock_api_key")
        self.mock_dbutils.notebook = Mock()
        self.mock_dbutils.notebook.entry_point = Mock()
        self.mock_dbutils.notebook.entry_point.getDbutils = Mock()
        self.mock_dbutils.notebook.entry_point.getDbutils.return_value.notebook = Mock()
        self.mock_dbutils.notebook.entry_point.getDbutils.return_value.notebook.return_value.getContext = Mock()
        self.mock_dbutils.notebook.entry_point.getDbutils.return_value.notebook.return_value.getContext.return_value.apiToken = Mock()
        self.mock_dbutils.notebook.entry_point.getDbutils.return_value.notebook.return_value.getContext.return_value.apiToken.return_value.get = Mock(return_value="mock_token")
        self.mock_dbutils.notebook.entry_point.getDbutils.return_value.notebook.return_value.getContext.return_value.browserHostName = Mock()
        self.mock_dbutils.notebook.entry_point.getDbutils.return_value.notebook.return_value.getContext.return_value.browserHostName.return_value.get = Mock(return_value="mock-workspace.cloud.databricks.com")
        self.mock_dbutils.notebook.entry_point.getDbutils.return_value.notebook.return_value.getContext.return_value.userName = Mock()
        self.mock_dbutils.notebook.entry_point.getDbutils.return_value.notebook.return_value.getContext.return_value.userName.return_value.get = Mock(return_value="test_user")
        
        # Mock display function
        self.mock_display = Mock()
        
        print("✅ Databricks environment mocked")
        
    def mock_widget_get(self, widget_name):
        """Mock widget values"""
        widget_values = {
            'evaluation_data_path': f"{self.test_dir}/evaluation_data.csv",
            'metrics_config_path': f"{self.test_dir}/sample_metrics_config.csv",
            'judge_model': 'databricks-llm'
        }
        return widget_values.get(widget_name, '')
        
    def run_test(self, test_name, test_func):
        """Run a single test and record results"""
        print(f"\n🧪 Running test: {test_name}")
        try:
            result = test_func()
            if result:
                print(f"✅ PASS: {test_name}")
                self.test_results.append((test_name, "PASS", None))
                return True
            else:
                print(f"❌ FAIL: {test_name}")
                self.test_results.append((test_name, "FAIL", "Test returned False"))
                return False
        except Exception as e:
            print(f"❌ ERROR: {test_name} - {str(e)}")
            print(f"   Traceback: {traceback.format_exc()}")
            self.test_results.append((test_name, "ERROR", str(e)))
            return False
    
    def test_file_loading(self):
        """Test file loading functionality"""
        print("\n📁 Testing file loading...")
        
        # Import the notebook functions
        sys.path.append('.')
        
        # Mock the environment
        with patch.dict('sys.modules', {'dbutils': self.mock_dbutils}):
            with patch('builtins.display', self.mock_display):
                # Import and test load_any_csv
                from databricks_llm_judge_notebook import load_any_csv
                
                # Test loading evaluation data
                eval_df = load_any_csv(f"{self.test_dir}/evaluation_data.csv", "evaluation data")
                assert eval_df is not None, "Failed to load evaluation data"
                assert len(eval_df) == 3, f"Expected 3 rows, got {len(eval_df)}"
                assert 'prompt' in eval_df.columns, "Missing prompt column"
                assert 'response' in eval_df.columns, "Missing response column"
                
                # Test loading metrics config
                metrics_df = load_any_csv(f"{self.test_dir}/sample_metrics_config.csv", "metrics config")
                assert metrics_df is not None, "Failed to load metrics config"
                assert len(metrics_df) == 3, f"Expected 3 metrics, got {len(metrics_df)}"
                assert 'name' in metrics_df.columns, "Missing name column"
                assert 'type' in metrics_df.columns, "Missing type column"
                
                # Test loading ground truth files
                gt_df = load_any_csv(f"{self.test_dir}/correct_answer.csv", "ground truth")
                assert gt_df is not None, "Failed to load ground truth"
                assert len(gt_df) == 3, f"Expected 3 ground truth rows, got {len(gt_df)}"
                
                print("✅ File loading tests passed")
                return True
    
    def test_metrics_config_parsing(self):
        """Test metrics configuration parsing"""
        print("\n📊 Testing metrics configuration parsing...")
        
        with patch.dict('sys.modules', {'dbutils': self.mock_dbutils}):
            with patch('builtins.display', self.mock_display):
                from databricks_llm_judge_notebook import load_metrics_config, process_custom_metrics, MetricType
                
                # Test loading metrics config
                metrics_df = load_metrics_config(f"{self.test_dir}/sample_metrics_config.csv")
                assert metrics_df is not None, "Failed to load metrics config"
                
                # Test processing metrics
                metrics_list = []
                for _, row in metrics_df.iterrows():
                    metric = {
                        'name': row['name'],
                        'type': row['type'],
                        'description': row['description'],
                        'evaluation_prompt': row['evaluation_prompt'],
                        'threshold': row['threshold'],
                        'ground_truth_column': row.get('ground_truth_column', 'ground_truth'),
                        'ground_truth_file_path': row.get('ground_truth_file_path', '')
                    }
                    metrics_list.append(metric)
                
                metric_configs = process_custom_metrics(metrics_list)
                assert len(metric_configs) == 3, f"Expected 3 metric configs, got {len(metric_configs)}"
                
                # Test metric types
                assert metric_configs[0].metric_type == MetricType.BINARY, "First metric should be binary"
                assert metric_configs[1].metric_type == MetricType.SCALE_1_5, "Second metric should be scale_1_5"
                assert metric_configs[2].metric_type == MetricType.BINARY, "Third metric should be binary"
                
                # Test ground truth columns
                assert metric_configs[0].ground_truth_column == 'correct_answer', "Wrong ground truth column"
                assert metric_configs[1].ground_truth_column == 'helpful_answer', "Wrong ground truth column"
                assert metric_configs[2].ground_truth_column == 'safe_response', "Wrong ground truth column"
                
                print("✅ Metrics configuration parsing tests passed")
                return True
    
    def test_ground_truth_loading(self):
        """Test ground truth loading per metric"""
        print("\n📚 Testing ground truth loading...")
        
        with patch.dict('sys.modules', {'dbutils': self.mock_dbutils}):
            with patch('builtins.display', self.mock_display):
                from databricks_llm_judge_notebook import load_ground_truth_for_metric, standardize_evaluation_data
                
                # Load evaluation data
                eval_df_raw = pd.read_csv(f"{self.test_dir}/evaluation_data.csv")
                eval_df = standardize_evaluation_data(eval_df_raw)
                assert eval_df is not None, "Failed to standardize evaluation data"
                
                # Test ground truth loading for each metric
                metric_configs = [
                    {'name': 'accuracy_check', 'ground_truth_column': 'correct_answer', 'ground_truth_file_path': ''},
                    {'name': 'helpfulness_rating', 'ground_truth_column': 'helpful_answer', 'ground_truth_file_path': ''},
                    {'name': 'safety_check', 'ground_truth_column': 'safe_response', 'ground_truth_file_path': ''}
                ]
                
                for metric_config in metric_configs:
                    # Change to test directory for file loading
                    original_cwd = os.getcwd()
                    os.chdir(self.test_dir)
                    
                    try:
                        result_df = load_ground_truth_for_metric(metric_config, eval_df)
                        assert result_df is not None, f"Failed to load ground truth for {metric_config['name']}"
                        
                        # Check if ground truth column was added
                        gt_column = metric_config['ground_truth_column']
                        assert gt_column in result_df.columns, f"Ground truth column {gt_column} not found"
                        
                        # Check if data was merged
                        gt_data = result_df[gt_column].dropna()
                        assert len(gt_data) > 0, f"No ground truth data merged for {metric_config['name']}"
                        
                    finally:
                        os.chdir(original_cwd)
                
                print("✅ Ground truth loading tests passed")
                return True
    
    def test_evaluation_logic(self):
        """Test evaluation logic with sample data"""
        print("\n🤖 Testing evaluation logic...")
        
        with patch.dict('sys.modules', {'dbutils': self.mock_dbutils}):
            with patch('builtins.display', self.mock_display):
                from databricks_llm_judge_notebook import (
                    LLMJudgeEvaluator, MetricConfig, MetricType, 
                    process_custom_metrics, standardize_evaluation_data
                )
                
                # Load and prepare data
                eval_df_raw = pd.read_csv(f"{self.test_dir}/evaluation_data.csv")
                eval_df = standardize_evaluation_data(eval_df_raw)
                
                # Load metrics
                metrics_df = pd.read_csv(f"{self.test_dir}/sample_metrics_config.csv")
                metrics_list = []
                for _, row in metrics_df.iterrows():
                    metric = {
                        'name': row['name'],
                        'type': row['type'],
                        'description': row['description'],
                        'evaluation_prompt': row['evaluation_prompt'],
                        'threshold': row['threshold'],
                        'ground_truth_column': row.get('ground_truth_column', 'ground_truth'),
                        'ground_truth_file_path': row.get('ground_truth_file_path', '')
                    }
                    metrics_list.append(metric)
                
                metric_configs = process_custom_metrics(metrics_list)
                
                # Mock the LLM evaluator to avoid actual API calls
                with patch.object(LLMJudgeEvaluator, '_initialize_databricks_client'):
                    with patch.object(LLMJudgeEvaluator, '_call_databricks_llm', return_value='{"accuracy_check_score": 1, "explanation": "Test response"}'):
                        evaluator = LLMJudgeEvaluator("databricks-llm", metric_configs)
                        
                        # Test single evaluation
                        sample_row = eval_df.iloc[0]
                        ground_truth_data = {'correct_answer': 'Paris is the capital of France'}
                        
                        result = evaluator.evaluate_single(
                            prompt=sample_row['prompt'],
                            response=sample_row['response'],
                            ground_truth_data=ground_truth_data,
                            metric=metric_configs[0]
                        )
                        
                        assert 'score' in result, "Result missing score"
                        assert 'explanation' in result, "Result missing explanation"
                        assert 'status' in result, "Result missing status"
                        
                        print("✅ Evaluation logic tests passed")
                        return True
    
    def test_error_handling(self):
        """Test error handling and edge cases"""
        print("\n⚠️ Testing error handling...")
        
        with patch.dict('sys.modules', {'dbutils': self.mock_dbutils}):
            with patch('builtins.display', self.mock_display):
                from databricks_llm_judge_notebook import load_any_csv, load_metrics_config
                
                # Test file not found
                result = load_any_csv("nonexistent_file.csv", "test")
                assert result is None, "Should return None for nonexistent file"
                
                # Test invalid CSV
                invalid_csv_path = f"{self.test_dir}/invalid.csv"
                with open(invalid_csv_path, 'w') as f:
                    f.write("invalid,csv,content\nwith,missing,quotes")
                
                result = load_any_csv(invalid_csv_path, "test")
                # Should either load successfully or handle error gracefully
                
                # Test metrics config with missing columns
                invalid_metrics = pd.DataFrame({
                    'name': ['test'],
                    'type': ['binary']
                    # Missing required columns
                })
                invalid_metrics.to_csv(f"{self.test_dir}/invalid_metrics.csv", index=False)
                
                result = load_metrics_config(f"{self.test_dir}/invalid_metrics.csv")
                # Should handle missing columns gracefully
                
                print("✅ Error handling tests passed")
                return True
    
    def test_workspace_file_search(self):
        """Test workspace file search functionality"""
        print("\n🔍 Testing workspace file search...")
        
        with patch.dict('sys.modules', {'dbutils': self.mock_dbutils}):
            with patch('builtins.display', self.mock_display):
                from databricks_llm_judge_notebook import load_any_csv
                
                # Test filename-only loading (should find in test directory)
                original_cwd = os.getcwd()
                os.chdir(self.test_dir)
                
                try:
                    # Test loading by filename only
                    result = load_any_csv("evaluation_data.csv", "test")
                    assert result is not None, "Failed to load file by filename only"
                    assert len(result) == 3, "Wrong number of rows loaded"
                    
                    # Test loading ground truth by filename
                    result = load_any_csv("correct_answer.csv", "test")
                    assert result is not None, "Failed to load ground truth by filename"
                    assert 'correct_answer' in result.columns, "Missing ground truth column"
                    
                finally:
                    os.chdir(original_cwd)
                
                print("✅ Workspace file search tests passed")
                return True
    
    def run_all_tests(self):
        """Run all tests"""
        print("🚀 Starting comprehensive test suite...")
        print("=" * 60)
        
        tests = [
            ("File Loading", self.test_file_loading),
            ("Metrics Config Parsing", self.test_metrics_config_parsing),
            ("Ground Truth Loading", self.test_ground_truth_loading),
            ("Evaluation Logic", self.test_evaluation_logic),
            ("Error Handling", self.test_error_handling),
            ("Workspace File Search", self.test_workspace_file_search)
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            if self.run_test(test_name, test_func):
                passed += 1
        
        print("\n" + "=" * 60)
        print(f"🏁 Test Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 ALL TESTS PASSED! The notebook is ready for Databricks upload.")
        else:
            print("❌ Some tests failed. Please review the issues above.")
        
        self.print_test_summary()
        return passed == total
    
    def print_test_summary(self):
        """Print detailed test summary"""
        print("\n📊 Detailed Test Summary:")
        print("-" * 40)
        
        for test_name, status, error in self.test_results:
            status_icon = "✅" if status == "PASS" else "❌"
            print(f"{status_icon} {test_name}: {status}")
            if error:
                print(f"   Error: {error}")
    
    def cleanup(self):
        """Clean up test environment"""
        print(f"\n🧹 Cleaning up test environment: {self.test_dir}")
        shutil.rmtree(self.test_dir, ignore_errors=True)

def main():
    """Main test runner"""
    tester = DatabricksNotebookTester()
    
    try:
        success = tester.run_all_tests()
        return 0 if success else 1
    finally:
        tester.cleanup()

if __name__ == "__main__":
    exit(main())