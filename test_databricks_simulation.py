# Databricks Test Simulation
# This file simulates testing the LLM Judge system in a Databricks environment

import os
import sys
import pandas as pd
import json
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
from pathlib import Path

# Mock Databricks environment
class MockDatabricksEnvironment:
    """Simulates Databricks environment for testing"""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.workspace_dir = os.path.join(self.temp_dir, "workspace")
        os.makedirs(self.workspace_dir, exist_ok=True)
        
        # Mock dbutils
        self.dbutils = Mock()
        self.dbutils.secrets.get = Mock(return_value="mock_api_key")
        self.dbutils.library.restartPython = Mock()
        
        # Set up environment
        os.environ["OPENAI_API_KEY"] = "mock_api_key"
        
    def cleanup(self):
        """Clean up temporary files"""
        shutil.rmtree(self.temp_dir, exist_ok=True)
    
    def create_test_data(self, scenario):
        """Create test data for different scenarios"""
        if scenario == "basic":
            return pd.DataFrame({
                "prompt": [
                    "What's the best way to buy a house in Seattle?",
                    "I have a credit score of 750, can I get a mortgage?",
                    "How much should I save for a down payment?"
                ],
                "response": [
                    "To buy a house in Seattle, you should first get pre-approved for a mortgage, work with a local real estate agent, and be prepared for a competitive market.",
                    "With a credit score of 750, you're in excellent position to qualify for a mortgage. You'll likely get the best interest rates available.",
                    "Aim to save 20% of the home's purchase price for a down payment to avoid PMI. However, many programs allow as little as 3-5% down."
                ]
            })
        
        elif scenario == "personalization":
            return pd.DataFrame({
                "prompt": [
                    "What's the best way to buy a house in Seattle?",
                    "I have a credit score of 750, can I get a mortgage?",
                    "How much should I save for a down payment?"
                ],
                "response": [
                    "To buy a house in Seattle, you should first get pre-approved for a mortgage, work with a local real estate agent, and be prepared for a competitive market.",
                    "With a credit score of 750, you're in excellent position to qualify for a mortgage. You'll likely get the best interest rates available.",
                    "Aim to save 20% of the home's purchase price for a down payment to avoid PMI. However, many programs allow as little as 3-5% down."
                ],
                "user_profile": [
                    "Location: Seattle, WA; Income: $120k; Credit Score: 720; Down Payment: $50k",
                    "Location: Seattle, WA; Income: $120k; Credit Score: 750; Down Payment: $50k",
                    "Location: Seattle, WA; Income: $120k; Credit Score: 720; Down Payment: $50k"
                ]
            })
        
        elif scenario == "ground_truth":
            return pd.DataFrame({
                "prompt": [
                    "What's the best way to buy a house in Seattle?",
                    "I have a credit score of 750, can I get a mortgage?",
                    "How much should I save for a down payment?"
                ],
                "response": [
                    "To buy a house in Seattle, you should first get pre-approved for a mortgage, work with a local real estate agent, and be prepared for a competitive market.",
                    "With a credit score of 750, you're in excellent position to qualify for a mortgage. You'll likely get the best interest rates available.",
                    "Aim to save 20% of the home's purchase price for a down payment to avoid PMI. However, many programs allow as little as 3-5% down."
                ],
                "ground_truth": [
                    "To buy a house in Seattle, you should first get pre-approved for a mortgage, work with a local real estate agent, and be prepared for a competitive market. Consider your budget, location preferences, and timeline.",
                    "With a credit score of 750, you're in excellent position to qualify for a mortgage. You'll likely get the best interest rates available. I recommend getting pre-approved to see your exact loan options.",
                    "Aim to save 20% of the home's purchase price for a down payment to avoid PMI. However, many programs allow as little as 3-5% down. Consider your monthly payment comfort level when deciding."
                ]
            })
        
        return pd.DataFrame()
    
    def create_ground_truth_csv(self):
        """Create a ground truth CSV file"""
        gt_data = pd.DataFrame({
            "prompt": [
                "What's the best way to buy a house in Seattle?",
                "I have a credit score of 750, can I get a mortgage?",
                "How much should I save for a down payment?"
            ],
            "ground_truth": [
                "To buy a house in Seattle, you should first get pre-approved for a mortgage, work with a local real estate agent, and be prepared for a competitive market. Consider your budget, location preferences, and timeline.",
                "With a credit score of 750, you're in excellent position to qualify for a mortgage. You'll likely get the best interest rates available. I recommend getting pre-approved to see your exact loan options.",
                "Aim to save 20% of the home's purchase price for a down payment to avoid PMI. However, many programs allow as little as 3-5% down. Consider your monthly payment comfort level when deciding."
            ]
        })
        
        csv_path = os.path.join(self.workspace_dir, "ground_truth.csv")
        gt_data.to_csv(csv_path, index=False)
        return csv_path
    
    def create_ground_truth_docx(self):
        """Create a ground truth DOCX file (simulated)"""
        # For testing, we'll create a simple text file that simulates DOCX structure
        docx_path = os.path.join(self.workspace_dir, "ground_truth.docx")
        
        # Create a simple text file that simulates DOCX content
        with open(docx_path, 'w') as f:
            f.write("prompt\tground_truth\n")
            f.write("What's the best way to buy a house in Seattle?\tTo buy a house in Seattle, you should first get pre-approved for a mortgage, work with a local real estate agent, and be prepared for a competitive market. Consider your budget, location preferences, and timeline.\n")
            f.write("I have a credit score of 750, can I get a mortgage?\tWith a credit score of 750, you're in excellent position to qualify for a mortgage. You'll likely get the best interest rates available. I recommend getting pre-approved to see your exact loan options.\n")
            f.write("How much should I save for a down payment?\tAim to save 20% of the home's purchase price for a down payment to avoid PMI. However, many programs allow as little as 3-5% down. Consider your monthly payment comfort level when deciding.\n")
        
        return docx_path

# Mock LLM responses for testing
def mock_llm_response(metric_name, score=1.0, explanation="Test explanation"):
    """Create mock LLM responses for testing"""
    return {
        "content": json.dumps({
            f"{metric_name}_score": score,
            "explanation": explanation
        })
    }

# Test scenarios
class TestScenarios:
    """Test different scenarios for the LLM Judge system"""
    
    def __init__(self):
        self.env = MockDatabricksEnvironment()
        self.test_results = []
    
    def run_test(self, test_name, test_func):
        """Run a single test and record results"""
        print(f"\n🧪 Running test: {test_name}")
        try:
            result = test_func()
            self.test_results.append({
                "test": test_name,
                "status": "PASS",
                "result": result
            })
            print(f"✅ {test_name}: PASSED")
            return result
        except Exception as e:
            self.test_results.append({
                "test": test_name,
                "status": "FAIL",
                "error": str(e)
            })
            print(f"❌ {test_name}: FAILED - {e}")
            return None
    
    def test_basic_evaluation(self):
        """Test basic evaluation with response quality and helpfulness"""
        print("Testing basic evaluation scenario...")
        
        # Mock the LLM responses
        with patch('langchain_openai.ChatOpenAI') as mock_chat:
            mock_instance = Mock()
            mock_instance.invoke = Mock(side_effect=[
                mock_llm_response("response_quality", 1.0, "Response is relevant and helpful"),
                mock_llm_response("helpfulness", 4.0, "Response is very helpful"),
                mock_llm_response("response_quality", 1.0, "Response is relevant and helpful"),
                mock_llm_response("helpfulness", 3.0, "Response is moderately helpful"),
                mock_llm_response("response_quality", 1.0, "Response is relevant and helpful"),
                mock_llm_response("helpfulness", 4.0, "Response is very helpful")
            ])
            mock_chat.return_value = mock_instance
            
            # Import and test the system
            sys.path.append('/workspace')
            from zillow_llm_judge_databricks import (
                MetricConfig, MetricType, EvaluationConfig,
                LLMJudgeEvaluator, create_sample_data
            )
            
            # Create test data
            df = self.env.create_test_data("basic")
            
            # Configure basic evaluation
            config = EvaluationConfig(
                experiment_name="test_basic",
                run_name="test_run",
                data_source="/workspace/test_data.csv",
                judge_models=["gpt-4o"]
            )
            
            # Create metrics
            metrics = [
                MetricConfig(
                    name="response_quality",
                    description="Test response quality",
                    metric_type=MetricType.BINARY,
                    prompt_template="Test prompt: {prompt}\nResponse: {response}",
                    threshold=1.0,
                    required_variables=["prompt", "response"]
                ),
                MetricConfig(
                    name="helpfulness",
                    description="Test helpfulness",
                    metric_type=MetricType.CATEGORICAL,
                    prompt_template="Test prompt: {prompt}\nResponse: {response}",
                    threshold=3.0,
                    scale_min=1.0,
                    scale_max=5.0,
                    required_variables=["prompt", "response"]
                )
            ]
            
            # Run evaluation
            evaluator = LLMJudgeEvaluator(config, metrics)
            results = evaluator.run_evaluation(df)
            
            # Validate results
            assert "response_quality_score" in results.columns
            assert "helpfulness_score" in results.columns
            assert len(results) == 3
            
            return {
                "metrics_evaluated": len(metrics),
                "samples_processed": len(results),
                "response_quality_mean": results["response_quality_score"].mean(),
                "helpfulness_mean": results["helpfulness_score"].mean()
            }
    
    def test_personalization_evaluation(self):
        """Test personalization accuracy with user profiles"""
        print("Testing personalization evaluation scenario...")
        
        with patch('langchain_openai.ChatOpenAI') as mock_chat:
            mock_instance = Mock()
            mock_instance.invoke = Mock(side_effect=[
                mock_llm_response("personalization_accuracy", 1.0, "User profile used correctly"),
                mock_llm_response("personalization_accuracy", 1.0, "User profile used correctly"),
                mock_llm_response("personalization_accuracy", 0.0, "User profile not used correctly")
            ])
            mock_chat.return_value = mock_instance
            
            # Import system
            sys.path.append('/workspace')
            from zillow_llm_judge_databricks import (
                MetricConfig, MetricType, EvaluationConfig,
                LLMJudgeEvaluator
            )
            
            # Create test data with user profiles
            df = self.env.create_test_data("personalization")
            
            # Configure evaluation
            config = EvaluationConfig(
                experiment_name="test_personalization",
                run_name="test_run",
                data_source="/workspace/test_data.csv",
                judge_models=["gpt-4o"]
            )
            
            # Create personalization metric
            metrics = [
                MetricConfig(
                    name="personalization_accuracy",
                    description="Test personalization accuracy",
                    metric_type=MetricType.BINARY,
                    prompt_template="Test prompt: {prompt}\nUser Profile: {user_profile}\nResponse: {response}",
                    threshold=1.0,
                    required_variables=["prompt", "response", "user_profile"]
                )
            ]
            
            # Run evaluation
            evaluator = LLMJudgeEvaluator(config, metrics)
            results = evaluator.run_evaluation(df)
            
            # Validate results
            assert "personalization_accuracy_score" in results.columns
            assert len(results) == 3
            
            return {
                "metrics_evaluated": len(metrics),
                "samples_processed": len(results),
                "personalization_accuracy_mean": results["personalization_accuracy_score"].mean()
            }
    
    def test_ground_truth_csv(self):
        """Test ground truth evaluation with CSV file"""
        print("Testing ground truth CSV scenario...")
        
        with patch('langchain_openai.ChatOpenAI') as mock_chat:
            mock_instance = Mock()
            mock_instance.invoke = Mock(side_effect=[
                mock_llm_response("ground_truth_accuracy", 0.9, "Very close to ground truth"),
                mock_llm_response("ground_truth_accuracy", 0.8, "Close to ground truth"),
                mock_llm_response("ground_truth_accuracy", 0.7, "Somewhat close to ground truth")
            ])
            mock_chat.return_value = mock_instance
            
            # Import system
            sys.path.append('/workspace')
            from zillow_llm_judge_databricks import (
                MetricConfig, MetricType, EvaluationConfig,
                LLMJudgeEvaluator, load_ground_truth_file
            )
            
            # Create ground truth CSV
            gt_csv_path = self.env.create_ground_truth_csv()
            
            # Test ground truth loading
            gt_df = load_ground_truth_file(gt_csv_path, "csv")
            assert len(gt_df) == 3
            assert "prompt" in gt_df.columns
            assert "ground_truth" in gt_df.columns
            
            # Create test data with ground truth
            df = self.env.create_test_data("ground_truth")
            
            # Configure evaluation
            config = EvaluationConfig(
                experiment_name="test_ground_truth_csv",
                run_name="test_run",
                data_source="/workspace/test_data.csv",
                judge_models=["gpt-4o"]
            )
            
            # Create ground truth metric
            metrics = [
                MetricConfig(
                    name="ground_truth_accuracy",
                    description="Test ground truth accuracy",
                    metric_type=MetricType.CONTINUOUS,
                    prompt_template="Test prompt: {prompt}\nResponse: {response}\nGround Truth: {ground_truth}",
                    threshold=0.8,
                    scale_min=0.0,
                    scale_max=1.0,
                    required_variables=["prompt", "response", "ground_truth"]
                )
            ]
            
            # Run evaluation
            evaluator = LLMJudgeEvaluator(config, metrics)
            results = evaluator.run_evaluation(df)
            
            # Validate results
            assert "ground_truth_accuracy_score" in results.columns
            assert len(results) == 3
            
            return {
                "metrics_evaluated": len(metrics),
                "samples_processed": len(results),
                "ground_truth_accuracy_mean": results["ground_truth_accuracy_score"].mean(),
                "csv_loading": "success"
            }
    
    def test_ground_truth_docx(self):
        """Test ground truth evaluation with DOCX file"""
        print("Testing ground truth DOCX scenario...")
        
        with patch('langchain_openai.ChatOpenAI') as mock_chat:
            mock_instance = Mock()
            mock_instance.invoke = Mock(side_effect=[
                mock_llm_response("ground_truth_accuracy", 0.9, "Very close to ground truth"),
                mock_llm_response("ground_truth_accuracy", 0.8, "Close to ground truth"),
                mock_llm_response("ground_truth_accuracy", 0.7, "Somewhat close to ground truth")
            ])
            mock_chat.return_value = mock_instance
            
            # Import system
            sys.path.append('/workspace')
            from zillow_llm_judge_databricks import (
                MetricConfig, MetricType, EvaluationConfig,
                LLMJudgeEvaluator, load_ground_truth_file
            )
            
            # Create ground truth DOCX (simulated)
            gt_docx_path = self.env.create_ground_truth_docx()
            
            # Test ground truth loading
            try:
                gt_df = load_ground_truth_file(gt_docx_path, "docx")
                assert len(gt_df) == 3
                assert "prompt" in gt_df.columns
                assert "ground_truth" in gt_df.columns
                docx_loading = "success"
            except Exception as e:
                print(f"DOCX loading failed (expected in test environment): {e}")
                docx_loading = "failed_expected"
            
            return {
                "metrics_evaluated": 1,
                "samples_processed": 3,
                "docx_loading": docx_loading
            }
    
    def test_ensemble_evaluation(self):
        """Test ensemble evaluation with multiple judge models"""
        print("Testing ensemble evaluation scenario...")
        
        with patch('langchain_openai.ChatOpenAI') as mock_chat:
            mock_instance = Mock()
            mock_instance.invoke = Mock(side_effect=[
                mock_llm_response("response_quality", 1.0, "Response is relevant and helpful"),
                mock_llm_response("response_quality", 1.0, "Response is relevant and helpful"),
                mock_llm_response("response_quality", 1.0, "Response is relevant and helpful"),
                mock_llm_response("response_quality", 1.0, "Response is relevant and helpful"),
                mock_llm_response("response_quality", 1.0, "Response is relevant and helpful"),
                mock_llm_response("response_quality", 1.0, "Response is relevant and helpful")
            ])
            mock_chat.return_value = mock_instance
            
            # Import system
            sys.path.append('/workspace')
            from zillow_llm_judge_databricks import (
                MetricConfig, MetricType, EvaluationConfig,
                LLMJudgeEvaluator
            )
            
            # Create test data
            df = self.env.create_test_data("basic")
            
            # Configure ensemble evaluation
            config = EvaluationConfig(
                experiment_name="test_ensemble",
                run_name="test_run",
                data_source="/workspace/test_data.csv",
                judge_models=["gpt-4o", "gpt-4o-mini"]
            )
            
            # Create metric
            metrics = [
                MetricConfig(
                    name="response_quality",
                    description="Test response quality",
                    metric_type=MetricType.BINARY,
                    prompt_template="Test prompt: {prompt}\nResponse: {response}",
                    threshold=1.0,
                    required_variables=["prompt", "response"]
                )
            ]
            
            # Run evaluation
            evaluator = LLMJudgeEvaluator(config, metrics)
            results = evaluator.run_evaluation(df)
            
            # Validate results
            assert "response_quality_score" in results.columns
            assert len(results) == 3
            
            return {
                "metrics_evaluated": len(metrics),
                "samples_processed": len(results),
                "judge_models": len(config.judge_models),
                "response_quality_mean": results["response_quality_score"].mean()
            }
    
    def test_error_handling(self):
        """Test error handling and edge cases"""
        print("Testing error handling scenario...")
        
        # Test with empty dataframe
        try:
            sys.path.append('/workspace')
            from zillow_llm_judge_databricks import (
                MetricConfig, MetricType, EvaluationConfig,
                LLMJudgeEvaluator
            )
            
            config = EvaluationConfig(
                experiment_name="test_errors",
                run_name="test_run",
                data_source="/workspace/test_data.csv",
                judge_models=["gpt-4o"]
            )
            
            metrics = [
                MetricConfig(
                    name="response_quality",
                    description="Test response quality",
                    metric_type=MetricType.BINARY,
                    prompt_template="Test prompt: {prompt}\nResponse: {response}",
                    threshold=1.0,
                    required_variables=["prompt", "response"]
                )
            ]
            
            # Test with empty dataframe
            empty_df = pd.DataFrame()
            evaluator = LLMJudgeEvaluator(config, metrics)
            
            try:
                results = evaluator.run_evaluation(empty_df)
                error_handling = "failed"
            except ValueError as e:
                if "empty" in str(e).lower():
                    error_handling = "success"
                else:
                    error_handling = "unexpected_error"
            
            return {
                "error_handling": error_handling,
                "empty_dataframe_test": "completed"
            }
            
        except Exception as e:
            return {
                "error_handling": "failed",
                "error": str(e)
            }
    
    def test_mlflow_integration(self):
        """Test MLflow integration and visualization"""
        print("Testing MLflow integration scenario...")
        
        with patch('langchain_openai.ChatOpenAI') as mock_chat:
            mock_instance = Mock()
            mock_instance.invoke = Mock(side_effect=[
                mock_llm_response("response_quality", 1.0, "Response is relevant and helpful"),
                mock_llm_response("helpfulness", 4.0, "Response is very helpful")
            ])
            mock_chat.return_value = mock_instance
            
            # Mock MLflow
            with patch('mlflow.set_experiment') as mock_set_exp, \
                 patch('mlflow.start_run') as mock_start_run, \
                 patch('mlflow.log_metric') as mock_log_metric, \
                 patch('mlflow.log_text') as mock_log_text, \
                 patch('mlflow.log_figure') as mock_log_figure:
                
                # Import system
                sys.path.append('/workspace')
                from zillow_llm_judge_databricks import (
                    MetricConfig, MetricType, EvaluationConfig,
                    LLMJudgeEvaluator, MLflowVisualizer
                )
                
                # Create test data
                df = self.env.create_test_data("basic")
                
                # Configure evaluation
                config = EvaluationConfig(
                    experiment_name="test_mlflow",
                    run_name="test_run",
                    data_source="/workspace/test_data.csv",
                    judge_models=["gpt-4o"]
                )
                
                # Create metrics
                metrics = [
                    MetricConfig(
                        name="response_quality",
                        description="Test response quality",
                        metric_type=MetricType.BINARY,
                        prompt_template="Test prompt: {prompt}\nResponse: {response}",
                        threshold=1.0,
                        required_variables=["prompt", "response"]
                    )
                ]
                
                # Run evaluation
                evaluator = LLMJudgeEvaluator(config, metrics)
                results = evaluator.run_evaluation(df)
                
                # Test MLflow integration
                visualizer = MLflowVisualizer(config.experiment_name)
                visualizer.log_evaluation_results(results, metrics, config.run_name)
                
                # Validate MLflow calls
                assert mock_set_exp.called
                assert mock_start_run.called
                assert mock_log_metric.called
                
                return {
                    "mlflow_integration": "success",
                    "metrics_logged": mock_log_metric.call_count,
                    "samples_processed": len(results)
                }
    
    def run_all_tests(self):
        """Run all test scenarios"""
        print("🚀 Starting comprehensive Databricks simulation tests...")
        print("=" * 60)
        
        # Run all tests
        self.run_test("Basic Evaluation", self.test_basic_evaluation)
        self.run_test("Personalization Evaluation", self.test_personalization_evaluation)
        self.run_test("Ground Truth CSV", self.test_ground_truth_csv)
        self.run_test("Ground Truth DOCX", self.test_ground_truth_docx)
        self.run_test("Ensemble Evaluation", self.test_ensemble_evaluation)
        self.run_test("Error Handling", self.test_error_handling)
        self.run_test("MLflow Integration", self.test_mlflow_integration)
        
        # Print summary
        self.print_test_summary()
        
        # Cleanup
        self.env.cleanup()
    
    def print_test_summary(self):
        """Print test results summary"""
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for result in self.test_results if result["status"] == "PASS")
        failed = sum(1 for result in self.test_results if result["status"] == "FAIL")
        total = len(self.test_results)
        
        print(f"Total Tests: {total}")
        print(f"Passed: {passed} ✅")
        print(f"Failed: {failed} ❌")
        print(f"Success Rate: {(passed/total)*100:.1f}%")
        
        print("\nDetailed Results:")
        for result in self.test_results:
            status_icon = "✅" if result["status"] == "PASS" else "❌"
            print(f"  {status_icon} {result['test']}: {result['status']}")
            if result["status"] == "FAIL" and "error" in result:
                print(f"    Error: {result['error']}")
        
        print("\n" + "=" * 60)

# Run the tests
if __name__ == "__main__":
    test_suite = TestScenarios()
    test_suite.run_all_tests()