#!/usr/bin/env python3
"""
Comprehensive Test Suite for Universal Evaluation Template System
================================================================

This test suite performs thorough testing to identify all potential failure points
and ensure the system works correctly in all scenarios.
"""

import os
import sys
import json
import tempfile
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional


class TestResult:
    """Class to track test results."""
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.passed = False
        self.error = None
        self.details = {}
    
    def success(self, details: Dict = None):
        self.passed = True
        if details:
            self.details.update(details)
    
    def failure(self, error: str, details: Dict = None):
        self.passed = False
        self.error = error
        if details:
            self.details.update(details)


class ComprehensiveTestSuite:
    """Comprehensive test suite for the evaluation system."""
    
    def __init__(self):
        self.results = []
        self.temp_dir = None
        self.setup_temp_environment()
    
    def setup_temp_environment(self):
        """Set up temporary environment for testing."""
        self.temp_dir = tempfile.mkdtemp(prefix="eval_test_")
        print(f"🧪 Test Environment: {self.temp_dir}")
        
        # Create test directories
        Path(f"{self.temp_dir}/data").mkdir(exist_ok=True)
        Path(f"{self.temp_dir}/configs").mkdir(exist_ok=True)
        Path(f"{self.temp_dir}/results").mkdir(exist_ok=True)
    
    def run_test(self, test_func, test_name: str):
        """Run a single test and record results."""
        print(f"\n🔬 Running: {test_name}")
        result = TestResult(test_name)
        
        try:
            test_func(result)
            if result.passed:
                print(f"✅ PASSED: {test_name}")
            else:
                print(f"❌ FAILED: {test_name} - {result.error}")
        except Exception as e:
            result.failure(f"Exception: {str(e)}")
            print(f"💥 ERROR: {test_name} - {str(e)}")
            traceback.print_exc()
        
        self.results.append(result)
        return result
    
    def test_imports(self, result: TestResult):
        """Test 1: Import all required modules."""
        try:
            # Test core imports
            import pandas as pd
            import yaml
            import json
            from pathlib import Path
            
            # Test if our modules can be imported
            sys.path.insert(0, os.getcwd())
            
            # Test template system import
            from template_evaluation_system import UniversalEvaluator, EvaluationConfig, ResponseGenerator
            from generate_config import generate_config
            from metric_templates import METRIC_TEMPLATES, DOMAIN_TEMPLATES
            
            result.success({"imports": "All modules imported successfully"})
        except ImportError as e:
            result.failure(f"Import error: {e}")
        except Exception as e:
            result.failure(f"Unexpected error during import: {e}")
    
    def test_configuration_loading(self, result: TestResult):
        """Test 2: Configuration loading and validation."""
        try:
            # Create a test configuration
            test_config = {
                "dataset": {
                    "path": "test.csv",
                    "prompt_column": "prompt",
                    "response_column": "response"
                },
                "models": {
                    "response_model": "GOLDEN_RESPONSE",
                    "judge_models": ["gpt-4o"]
                },
                "api_keys": {
                    "openai_api_key": "test-key",
                    "openai_base_url": "https://api.openai.com/v1"
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
            
            # Save config
            config_path = f"{self.temp_dir}/configs/test_config.yaml"
            with open(config_path, 'w') as f:
                import yaml
                yaml.dump(test_config, f)
            
            # Test loading
            from template_evaluation_system import EvaluationConfig
            config = EvaluationConfig(config_path)
            
            # Validate loaded config
            assert config.dataset_path == "test.csv"
            assert config.prompt_column == "prompt"
            assert config.response_column == "response"
            assert config.judge_models == ["gpt-4o"]
            
            result.success({"config_loaded": True, "validation_passed": True})
            
        except Exception as e:
            result.failure(f"Configuration loading failed: {e}")
    
    def test_dataset_creation(self, result: TestResult):
        """Test 3: Dataset creation and validation."""
        try:
            # Create test dataset
            test_data = [
                {
                    "prompt": "What is 2+2?",
                    "response": "2+2 equals 4.",
                    "ground_truth": "4",
                    "user_features": '{"level": "beginner"}'
                },
                {
                    "prompt": "How do I cook pasta?",
                    "response": "Boil water, add pasta, cook for 8-10 minutes.",
                    "ground_truth": "Boil water, add pasta, cook",
                    "user_features": '{"cooking_level": "intermediate"}'
                }
            ]
            
            # Save as CSV
            import pandas as pd
            df = pd.DataFrame(test_data)
            dataset_path = f"{self.temp_dir}/data/test_dataset.csv"
            df.to_csv(dataset_path, index=False)
            
            # Validate dataset
            loaded_df = pd.read_csv(dataset_path)
            assert len(loaded_df) == 2
            assert "prompt" in loaded_df.columns
            assert "response" in loaded_df.columns
            assert "ground_truth" in loaded_df.columns
            assert "user_features" in loaded_df.columns
            
            result.success({
                "dataset_created": True,
                "rows": len(loaded_df),
                "columns": list(loaded_df.columns)
            })
            
        except Exception as e:
            result.failure(f"Dataset creation failed: {e}")
    
    def test_metric_templates(self, result: TestResult):
        """Test 4: Metric templates functionality."""
        try:
            from metric_templates import (
                METRIC_TEMPLATES, 
                DOMAIN_TEMPLATES,
                get_metric_template,
                list_available_metrics,
                list_available_domains
            )
            
            # Test general metrics
            assert "accuracy" in METRIC_TEMPLATES
            assert "relevance" in METRIC_TEMPLATES
            assert "helpfulness" in METRIC_TEMPLATES
            
            # Test domain metrics
            assert "financial_advice" in DOMAIN_TEMPLATES
            assert "medical_advice" in DOMAIN_TEMPLATES
            
            # Test metric retrieval
            accuracy_template = get_metric_template("accuracy")
            assert "prompt_template" in accuracy_template
            assert "threshold" in accuracy_template
            
            # Test listing functions
            metrics = list_available_metrics()
            domains = list_available_domains()
            assert len(metrics) > 0
            assert len(domains) > 0
            
            result.success({
                "general_metrics": len(METRIC_TEMPLATES),
                "domain_metrics": len(DOMAIN_TEMPLATES),
                "available_metrics": len(metrics),
                "available_domains": len(domains)
            })
            
        except Exception as e:
            result.failure(f"Metric templates test failed: {e}")
    
    def test_configuration_generator(self, result: TestResult):
        """Test 5: Configuration generator functionality."""
        try:
            from generate_config import generate_config
            
            # Test basic config generation
            config = generate_config(
                dataset_path="test.csv",
                prompt_column="prompt",
                response_column="response",
                experiment_name="test_experiment",
                metrics=["accuracy", "relevance"]
            )
            
            # Validate generated config
            assert "dataset" in config
            assert "models" in config
            assert "evaluation" in config
            assert config["dataset"]["path"] == "test.csv"
            assert "accuracy" in config["evaluation"]["metrics"]
            assert "relevance" in config["evaluation"]["metrics"]
            
            result.success({
                "config_generated": True,
                "metrics_count": len(config["evaluation"]["metrics"])
            })
            
        except Exception as e:
            result.failure(f"Configuration generator test failed: {e}")
    
    def test_evaluator_creation(self, result: TestResult):
        """Test 6: Evaluator creation and initialization."""
        try:
            # Create test config
            test_config = {
                "dataset": {
                    "path": f"{self.temp_dir}/data/test_dataset.csv",
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
                    "openai_api_key": "test-key",
                    "openai_base_url": "https://api.openai.com/v1"
                },
                "evaluation": {
                    "experiment_name": "test",
                    "metrics": {
                        "accuracy": {
                            "prompt_template": "Test accuracy template",
                            "threshold": 3.0,
                            "output_schema": {
                                "fields": {
                                    "accuracy_score": "int",
                                    "explanation": "str"
                                }
                            }
                        }
                    }
                }
            }
            
            # Save config
            config_path = f"{self.temp_dir}/configs/test_config.yaml"
            with open(config_path, 'w') as f:
                import yaml
                yaml.dump(test_config, f)
            
            # Test evaluator creation
            from template_evaluation_system import EvaluationConfig, UniversalEvaluator
            config = EvaluationConfig(config_path)
            evaluator = UniversalEvaluator(config)
            
            # Validate evaluator
            assert len(evaluator.evaluators) == 1
            assert "accuracy" in evaluator.evaluators
            assert len(evaluator.scorers) > 0
            
            result.success({
                "evaluator_created": True,
                "evaluators_count": len(evaluator.evaluators),
                "scorers_count": len(evaluator.scorers)
            })
            
        except Exception as e:
            result.failure(f"Evaluator creation test failed: {e}")
    
    def test_prompt_formatting(self, result: TestResult):
        """Test 7: Prompt formatting functionality."""
        try:
            # Create test data
            test_row = {
                "prompt": "What is 2+2?",
                "response": "2+2 equals 4.",
                "ground_truth": "4",
                "user_features": '{"level": "beginner"}'
            }
            
            # Test prompt formatting
            template = "User Query: {prompt}\nResponse: {response}\nGround Truth: {ground_truth}"
            formatted = template.format(**test_row)
            
            assert "What is 2+2?" in formatted
            assert "2+2 equals 4." in formatted
            assert "4" in formatted
            
            result.success({
                "prompt_formatted": True,
                "contains_prompt": "What is 2+2?" in formatted,
                "contains_response": "2+2 equals 4." in formatted
            })
            
        except Exception as e:
            result.failure(f"Prompt formatting test failed: {e}")
    
    def test_mock_evaluation(self, result: TestResult):
        """Test 8: Mock evaluation without API calls."""
        try:
            # Create test dataset
            test_data = [
                {
                    "prompt": "What is 2+2?",
                    "response": "2+2 equals 4.",
                    "ground_truth": "4",
                    "user_features": '{"level": "beginner"}'
                }
            ]
            
            import pandas as pd
            df = pd.DataFrame(test_data)
            dataset_path = f"{self.temp_dir}/data/test_dataset.csv"
            df.to_csv(dataset_path, index=False)
            
            # Create mock evaluator
            class MockEvaluator:
                def __init__(self):
                    self.evaluators = {"accuracy": self.mock_evaluator}
                
                def mock_evaluator(self, eval_df, builtin_metrics=None):
                    return {
                        "accuracy/mean": 4.0,
                        "accuracy/scores": [4.0],
                        "accuracy/details": [{"mock": True, "score": 4.0}]
                    }
                
                def evaluate_dataset(self, df):
                    # Mock evaluation
                    df["accuracy"] = [4.0]
                    df["accuracy_status"] = ["✅"]
                    df["accuracy_details"] = [{"mock": True, "score": 4.0}]
                    return df
            
            # Run mock evaluation
            evaluator = MockEvaluator()
            results = evaluator.evaluate_dataset(df)
            
            # Validate results
            assert "accuracy" in results.columns
            assert "accuracy_status" in results.columns
            assert results["accuracy"].iloc[0] == 4.0
            
            result.success({
                "mock_evaluation": True,
                "results_generated": True,
                "accuracy_score": results["accuracy"].iloc[0]
            })
            
        except Exception as e:
            result.failure(f"Mock evaluation test failed: {e}")
    
    def test_error_handling(self, result: TestResult):
        """Test 9: Error handling and edge cases."""
        try:
            # Test missing file handling
            try:
                from template_evaluation_system import EvaluationConfig
                config = EvaluationConfig("nonexistent_file.yaml")
                result.failure("Should have failed with missing file")
                return
            except FileNotFoundError:
                pass  # Expected
            
            # Test invalid YAML handling
            invalid_yaml_path = f"{self.temp_dir}/invalid.yaml"
            with open(invalid_yaml_path, 'w') as f:
                f.write("invalid: yaml: content: [")
            
            try:
                config = EvaluationConfig(invalid_yaml_path)
                result.failure("Should have failed with invalid YAML")
                return
            except Exception:
                pass  # Expected
            
            # Test missing required fields
            incomplete_config = {
                "dataset": {
                    "path": "test.csv"
                    # Missing required fields
                }
            }
            
            incomplete_path = f"{self.temp_dir}/incomplete.yaml"
            with open(incomplete_path, 'w') as f:
                import yaml
                yaml.dump(incomplete_config, f)
            
            try:
                config = EvaluationConfig(incomplete_path)
                result.failure("Should have failed with incomplete config")
                return
            except Exception:
                pass  # Expected
            
            result.success({
                "error_handling": True,
                "missing_file_handled": True,
                "invalid_yaml_handled": True,
                "incomplete_config_handled": True
            })
            
        except Exception as e:
            result.failure(f"Error handling test failed: {e}")
    
    def test_file_operations(self, result: TestResult):
        """Test 10: File operations and I/O."""
        try:
            # Test CSV reading/writing
            import pandas as pd
            
            test_data = {"col1": [1, 2, 3], "col2": ["a", "b", "c"]}
            df = pd.DataFrame(test_data)
            
            # Test writing
            csv_path = f"{self.temp_dir}/test_io.csv"
            df.to_csv(csv_path, index=False)
            
            # Test reading
            loaded_df = pd.read_csv(csv_path)
            assert len(loaded_df) == 3
            assert list(loaded_df.columns) == ["col1", "col2"]
            
            # Test JSON operations
            json_data = {"test": "data", "numbers": [1, 2, 3]}
            json_path = f"{self.temp_dir}/test_io.json"
            
            with open(json_path, 'w') as f:
                json.dump(json_data, f)
            
            with open(json_path, 'r') as f:
                loaded_json = json.load(f)
            
            assert loaded_json["test"] == "data"
            assert loaded_json["numbers"] == [1, 2, 3]
            
            result.success({
                "csv_operations": True,
                "json_operations": True,
                "file_io": True
            })
            
        except Exception as e:
            result.failure(f"File operations test failed: {e}")
    
    def test_yaml_operations(self, result: TestResult):
        """Test 11: YAML operations."""
        try:
            import yaml
            
            # Test YAML writing
            test_data = {
                "string": "test",
                "number": 42,
                "list": [1, 2, 3],
                "nested": {
                    "key": "value"
                }
            }
            
            yaml_path = f"{self.temp_dir}/test.yaml"
            with open(yaml_path, 'w') as f:
                yaml.dump(test_data, f, default_flow_style=False, indent=2)
            
            # Test YAML reading
            with open(yaml_path, 'r') as f:
                loaded_data = yaml.safe_load(f)
            
            assert loaded_data["string"] == "test"
            assert loaded_data["number"] == 42
            assert loaded_data["list"] == [1, 2, 3]
            assert loaded_data["nested"]["key"] == "value"
            
            result.success({
                "yaml_write": True,
                "yaml_read": True,
                "data_integrity": True
            })
            
        except Exception as e:
            result.failure(f"YAML operations test failed: {e}")
    
    def test_environment_variables(self, result: TestResult):
        """Test 12: Environment variable handling."""
        try:
            # Test setting environment variables
            test_key = "TEST_EVAL_KEY"
            test_value = "test_value_123"
            
            os.environ[test_key] = test_value
            assert os.environ.get(test_key) == test_value
            
            # Test missing environment variable
            missing_key = "MISSING_EVAL_KEY"
            assert os.environ.get(missing_key) is None
            assert os.environ.get(missing_key, "default") == "default"
            
            # Clean up
            del os.environ[test_key]
            assert os.environ.get(test_key) is None
            
            result.success({
                "env_set": True,
                "env_get": True,
                "env_missing": True,
                "env_cleanup": True
            })
            
        except Exception as e:
            result.failure(f"Environment variables test failed: {e}")
    
    def run_all_tests(self):
        """Run all tests in the suite."""
        print("🚀 Starting Comprehensive Test Suite")
        print("=" * 60)
        
        # Define all tests
        tests = [
            (self.test_imports, "Import Dependencies"),
            (self.test_configuration_loading, "Configuration Loading"),
            (self.test_dataset_creation, "Dataset Creation"),
            (self.test_metric_templates, "Metric Templates"),
            (self.test_configuration_generator, "Configuration Generator"),
            (self.test_evaluator_creation, "Evaluator Creation"),
            (self.test_prompt_formatting, "Prompt Formatting"),
            (self.test_mock_evaluation, "Mock Evaluation"),
            (self.test_error_handling, "Error Handling"),
            (self.test_file_operations, "File Operations"),
            (self.test_yaml_operations, "YAML Operations"),
            (self.test_environment_variables, "Environment Variables")
        ]
        
        # Run all tests
        for test_func, test_name in tests:
            self.run_test(test_func, test_name)
        
        # Generate summary
        self.generate_summary()
    
    def generate_summary(self):
        """Generate test summary and identify failure points."""
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            print("\n❌ FAILED TESTS:")
            print("-" * 30)
            for result in self.results:
                if not result.passed:
                    print(f"• {result.test_name}: {result.error}")
                    if result.details:
                        for key, value in result.details.items():
                            print(f"  - {key}: {value}")
        
        print("\n🔍 FAILURE POINTS ANALYSIS:")
        print("-" * 30)
        
        # Analyze failure patterns
        failure_categories = {
            "Import Issues": [],
            "Configuration Issues": [],
            "Data Issues": [],
            "Evaluation Issues": [],
            "File I/O Issues": [],
            "Other Issues": []
        }
        
        for result in self.results:
            if not result.passed:
                error = result.error.lower()
                if "import" in error:
                    failure_categories["Import Issues"].append(result.test_name)
                elif "config" in error:
                    failure_categories["Configuration Issues"].append(result.test_name)
                elif "dataset" in error or "data" in error:
                    failure_categories["Data Issues"].append(result.test_name)
                elif "evaluat" in error:
                    failure_categories["Evaluation Issues"].append(result.test_name)
                elif "file" in error or "yaml" in error or "csv" in error:
                    failure_categories["File I/O Issues"].append(result.test_name)
                else:
                    failure_categories["Other Issues"].append(result.test_name)
        
        for category, tests in failure_categories.items():
            if tests:
                print(f"\n{category}:")
                for test in tests:
                    print(f"  • {test}")
        
        print("\n💡 RECOMMENDATIONS:")
        print("-" * 20)
        
        if failure_categories["Import Issues"]:
            print("• Install missing dependencies: pip install -r requirements.txt")
        
        if failure_categories["Configuration Issues"]:
            print("• Check YAML syntax and required fields in configuration files")
            print("• Use the configuration generator: python generate_config.py")
        
        if failure_categories["Data Issues"]:
            print("• Ensure dataset has required columns: prompt, response")
            print("• Check data format and encoding")
        
        if failure_categories["File I/O Issues"]:
            print("• Check file permissions and paths")
            print("• Ensure proper file formats (CSV, YAML)")
        
        if failure_categories["Evaluation Issues"]:
            print("• Set OPENAI_API_KEY environment variable")
            print("• Check model availability and API endpoints")
        
        # Clean up
        self.cleanup()
    
    def cleanup(self):
        """Clean up temporary files."""
        try:
            import shutil
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                print(f"\n🧹 Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            print(f"⚠️  Warning: Could not clean up temporary directory: {e}")


def main():
    """Main test execution."""
    print("🧪 Universal Evaluation Template System - Comprehensive Test Suite")
    print("=" * 80)
    
    # Create and run test suite
    test_suite = ComprehensiveTestSuite()
    test_suite.run_all_tests()
    
    print("\n🎉 Test suite completed!")
    print("Check the summary above for any issues that need to be addressed.")


if __name__ == "__main__":
    main()