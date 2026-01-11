"""
Generalized LLM Evaluation System with Flexible Directory Management

This module provides a generalized evaluation system that can work across
different environments and directory structures, replacing hardcoded paths
with a flexible directory management system.
"""

import os
import time
import pandas as pd
import mlflow
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Import our directory management system
from directory_manager import DirectoryManager, DirectoryConfig, create_directory_manager


class MetricType(Enum):
    """Types of evaluation metrics."""
    BINARY = "binary"
    SCALE_1_5 = "scale_1_5" 
    PERCENTAGE = "percentage"


@dataclass
class MetricConfig:
    """Configuration for a single evaluation metric."""
    name: str
    metric_type: MetricType
    description: str
    prompt_template: str
    threshold: float
    ground_truth_column: str
    ground_truth_file_path: str = ""


class LLMJudgeEvaluator:
    """
    Generalized LLM Judge Evaluator with flexible directory management.
    
    This evaluator can work in different environments (local, Databricks, cloud)
    and handles directory paths, file loading, and output management automatically.
    """
    
    def __init__(
        self,
        judge_model: str,
        metrics: List[MetricConfig],
        ground_truth_data: Optional[pd.DataFrame] = None,
        directory_manager: Optional[DirectoryManager] = None
    ):
        """
        Initialize the evaluator.
        
        Args:
            judge_model: Name of the judge model to use
            metrics: List of metric configurations
            ground_truth_data: Optional ground truth DataFrame
            directory_manager: Optional DirectoryManager instance
        """
        self.judge_model = judge_model
        self.metrics = metrics
        self.ground_truth_data = ground_truth_data
        
        # Initialize directory manager
        if directory_manager is None:
            self.dir_manager = create_directory_manager()
        else:
            self.dir_manager = directory_manager
        
        print(f"🏗️  Environment: {self.dir_manager.environment}")
        print(f"📁 Workspace: {self.dir_manager.workspace_root}")
        print(f"📊 Output: {self.dir_manager.output_dir}")
    
    def evaluate_dataset(self, evaluation_data: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluate a dataset using the configured metrics.
        
        Args:
            evaluation_data: DataFrame with evaluation samples
            
        Returns:
            DataFrame with evaluation results
        """
        results = []
        
        for idx, row in evaluation_data.iterrows():
            for metric in self.metrics:
                try:
                    # Simulate evaluation (replace with actual LLM judge call)
                    score = self._evaluate_sample(row, metric)
                    
                    result = {
                        'sample_id': idx,
                        'metric_name': metric.name,
                        'score': score,
                        'threshold': metric.threshold,
                        'status': '✅' if score >= metric.threshold else '❌',
                        'prompt': row.get('prompt', ''),
                        'response': row.get('response', ''),
                        'ground_truth': row.get('ground_truth', '')
                    }
                    results.append(result)
                    
                except Exception as e:
                    print(f"❌ Error evaluating {metric.name} for sample {idx}: {e}")
                    result = {
                        'sample_id': idx,
                        'metric_name': metric.name,
                        'score': 0.0,
                        'threshold': metric.threshold,
                        'status': '❌',
                        'error': str(e),
                        'prompt': row.get('prompt', ''),
                        'response': row.get('response', ''),
                        'ground_truth': row.get('ground_truth', '')
                    }
                    results.append(result)
        
        return pd.DataFrame(results)
    
    def _evaluate_sample(self, sample: pd.Series, metric: MetricConfig) -> float:
        """
        Evaluate a single sample against a metric.
        
        Args:
            sample: Sample data
            metric: Metric configuration
            
        Returns:
            Evaluation score
        """
        # This is a placeholder - replace with actual LLM judge evaluation
        # For demonstration, return a random score based on metric type
        import random
        
        if metric.metric_type == MetricType.BINARY:
            return random.choice([0.0, 1.0])
        elif metric.metric_type == MetricType.SCALE_1_5:
            return random.randint(1, 5)
        elif metric.metric_type == MetricType.PERCENTAGE:
            return random.randint(0, 100)
        else:
            return random.random()


def auto_generate_evaluation_prompt(
    metric_name: str, 
    metric_type: str, 
    description: str, 
    grading_rubric: str
) -> str:
    """
    Auto-generate evaluation prompt from grading rubric.
    PM just provides grading rubric, code handles the rest!
    """
    
    # Add grading rubric section
    rubric_section = f"\n**Grading Rubric:**\n{grading_rubric}\n" if grading_rubric else ""
    
    # Build complete evaluation prompt
    prompt = f"""You are an expert evaluator. Your task: {description}

{rubric_section}
**Evaluation Details:**
- User Query: {{prompt}}
- AI Response: {{response}}
- Ground Truth Reference: {{ground_truth}}

**Instructions:**
Carefully evaluate the AI response using the grading rubric above.

**Required Output Format:**
Return ONLY a valid JSON object with these two fields:
{{
  "score": <your_score>,
  "explanation": "Brief explanation of your score"
}}

Do not include any other text outside the JSON object."""
    
    return prompt


def load_metrics_from_csv(
    metrics_config_data: pd.DataFrame,
    directory_manager: Optional[DirectoryManager] = None
) -> List[MetricConfig]:
    """
    Load metrics from CSV and auto-generate evaluation prompts.
    
    Args:
        metrics_config_data: DataFrame with metric configurations
        directory_manager: Optional DirectoryManager for path resolution
        
    Returns:
        List of MetricConfig objects
    """
    if metrics_config_data is None or metrics_config_data.empty:
        return []
    
    if directory_manager is None:
        directory_manager = create_directory_manager()
    
    metric_configs = []
    
    for _, row in metrics_config_data.iterrows():
        # Parse metric type
        metric_type_str = row['type'].strip().lower()
        if metric_type_str == 'binary':
            metric_type = MetricType.BINARY
        elif metric_type_str in ['1-5_scale', 'scale_1_5']:
            metric_type = MetricType.SCALE_1_5
        elif metric_type_str == 'percentage':
            metric_type = MetricType.PERCENTAGE
        else:
            metric_type = MetricType.BINARY
        
        # Check if using new format (grading_rubric) or old format (evaluation_prompt)
        if 'grading_rubric' in row and pd.notna(row.get('grading_rubric')):
            # NEW FORMAT: Auto-generate evaluation prompt from grading rubric
            grading_rubric = str(row['grading_rubric']).strip()
            prompt_template = auto_generate_evaluation_prompt(
                metric_name=row['name'].strip(),
                metric_type=metric_type_str,
                description=row.get('description', '').strip(),
                grading_rubric=grading_rubric
            )
            print(f"✅ {row['name'].strip()} - Auto-generated prompt from grading rubric")
        elif 'evaluation_prompt' in row and pd.notna(row.get('evaluation_prompt')):
            # OLD FORMAT: Use evaluation_prompt directly (backward compatibility)
            prompt_template = row['evaluation_prompt'].strip()
            print(f"✅ {row['name'].strip()} - Using provided evaluation_prompt")
        else:
            # Fallback: Generate basic prompt from description
            prompt_template = auto_generate_evaluation_prompt(
                metric_name=row['name'].strip(),
                metric_type=metric_type_str,
                description=row.get('description', '').strip(),
                grading_rubric=""
            )
            print(f"⚠️ {row['name'].strip()} - No rubric or prompt provided, using basic prompt")
        
        # Ground truth file matching (PM provides filename, code matches to uploaded files)
        gt_file_value = row.get('ground_truth_file_path', '').strip()
        
        metric_config = MetricConfig(
            name=row['name'].strip(),
            metric_type=metric_type,
            description=row.get('description', '').strip(),
            prompt_template=prompt_template,
            threshold=float(row['threshold']),
            ground_truth_column=row['ground_truth_column'].strip(),
            ground_truth_file_path=gt_file_value
        )
        
        # Show which GT file this metric will use
        if gt_file_value:
            gt_filename = os.path.basename(gt_file_value)
            print(f"   📚 Ground truth: {gt_filename} → column '{row['ground_truth_column'].strip()}'")
        
        metric_configs.append(metric_config)
    
    return metric_configs


def run_evaluation_pipeline(
    evaluation_data: pd.DataFrame,
    metrics_config_data: pd.DataFrame,
    ground_truth_data: Optional[pd.DataFrame] = None,
    judge_model: str = "gpt-4",
    directory_manager: Optional[DirectoryManager] = None,
    save_results: bool = True,
    log_to_mlflow: bool = True
) -> pd.DataFrame:
    """
    Run the complete evaluation pipeline with generalized directory handling.
    
    Args:
        evaluation_data: DataFrame with samples to evaluate
        metrics_config_data: DataFrame with metric configurations
        ground_truth_data: Optional ground truth data
        judge_model: Name of the judge model
        directory_manager: Optional DirectoryManager instance
        save_results: Whether to save results to file
        log_to_mlflow: Whether to log results to MLflow
        
    Returns:
        DataFrame with evaluation results
    """
    
    # Initialize directory manager if not provided
    if directory_manager is None:
        directory_manager = create_directory_manager()
    
    print("="*60)
    print("🚀 STARTING GENERALIZED EVALUATION PIPELINE")
    print("="*60)
    print(f"📊 Environment: {directory_manager.environment}")
    print(f"📁 Workspace: {directory_manager.workspace_root}")
    print(f"💾 Output Directory: {directory_manager.output_dir}")
    
    # Load metrics configuration
    metric_configs = load_metrics_from_csv(metrics_config_data, directory_manager)
    
    if not metric_configs:
        print("❌ No valid metrics found!")
        return pd.DataFrame()
    
    # Create evaluator
    evaluator = LLMJudgeEvaluator(
        judge_model=judge_model,
        metrics=metric_configs,
        ground_truth_data=ground_truth_data,
        directory_manager=directory_manager
    )
    
    # Run evaluation
    print("="*60)
    print("🔄 RUNNING EVALUATION")
    print("="*60)
    
    start_time = time.time()
    results_df = evaluator.evaluate_dataset(evaluation_data)
    eval_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"✅ EVALUATION COMPLETE in {eval_time:.1f}s")
    print(f"{'='*60}")
    
    # Display summary
    total = len(results_df)
    passed = len(results_df[results_df['status'] == '✅'])
    pass_rate = (passed / total) * 100 if total > 0 else 0
    
    print(f"\n📊 RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total Evaluations: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Pass Rate: {pass_rate:.1f}%")
    print(f"{'='*60}")
    
    # Per-metric results
    for metric_name in results_df['metric_name'].unique():
        metric_results = results_df[results_df['metric_name'] == metric_name]
        metric_passed = len(metric_results[metric_results['status'] == '✅'])
        metric_total = len(metric_results)
        metric_pass_rate = (metric_passed / metric_total) * 100 if metric_total > 0 else 0
        avg_score = metric_results['score'].mean()
        
        print(f"\n🔹 {metric_name}:")
        print(f"   Pass Rate: {metric_pass_rate:.1f}% ({metric_passed}/{metric_total})")
        print(f"   Avg Score: {avg_score:.2f}")
        print(f"   Threshold: {metric_results['threshold'].iloc[0]}")
    
    # Save results to file
    if save_results:
        try:
            csv_path = directory_manager.get_output_path("evaluation_results.csv")
            results_df.to_csv(csv_path, index=False)
            print(f"\n💾 Results saved to: {csv_path}")
        except Exception as e:
            print(f"⚠️ Failed to save results: {e}")
    
    # Log to MLflow
    if log_to_mlflow:
        try:
            _log_to_mlflow(
                results_df=results_df,
                metric_configs=metric_configs,
                judge_model=judge_model,
                evaluation_data=evaluation_data,
                pass_rate=pass_rate,
                directory_manager=directory_manager
            )
            print(f"✅ Results logged to MLflow")
        except Exception as e:
            print(f"⚠️ Failed to log to MLflow: {e}")
    
    return results_df


def _log_to_mlflow(
    results_df: pd.DataFrame,
    metric_configs: List[MetricConfig],
    judge_model: str,
    evaluation_data: pd.DataFrame,
    pass_rate: float,
    directory_manager: DirectoryManager
):
    """Log evaluation results to MLflow with environment-appropriate settings."""
    
    # Set experiment path based on environment
    experiment_path = directory_manager.get_mlflow_experiment_path()
    mlflow.set_experiment(experiment_path)
    
    run_name = f"eval_{judge_model}_{time.strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_param("judge_model", judge_model)
        mlflow.log_param("num_samples", len(evaluation_data))
        mlflow.log_param("metrics", ",".join([cfg.name for cfg in metric_configs]))
        mlflow.log_param("environment", directory_manager.environment)
        
        # Log overall metrics
        mlflow.log_metric("overall_pass_rate", pass_rate)
        mlflow.log_metric("total_evaluations", len(results_df))
        mlflow.log_metric("passed_evaluations", len(results_df[results_df['status'] == '✅']))
        
        # Log per-metric results
        for metric_name in results_df['metric_name'].unique():
            metric_results = results_df[results_df['metric_name'] == metric_name]
            scores = metric_results['score'].tolist()
            numeric_scores = [s for s in scores if isinstance(s, (int, float))]
            
            if numeric_scores:
                mean_score = sum(numeric_scores) / len(numeric_scores)
                threshold = metric_results['threshold'].iloc[0]
                pass_count = sum(1 for s in numeric_scores if s >= threshold)
                pass_rate_metric = (pass_count / len(numeric_scores)) * 100
                
                mlflow.log_metric(f"{metric_name}_mean", float(mean_score))
                mlflow.log_metric(f"{metric_name}_pass_rate", float(pass_rate_metric))
        
        # Save and log results file
        try:
            csv_path = directory_manager.get_temp_path("mlflow_results", ".csv")
            results_df.to_csv(csv_path, index=False)
            mlflow.log_artifact(str(csv_path), artifact_path=directory_manager.config.mlflow_artifact_path)
        except Exception as e:
            print(f"Warning: Failed to log results artifact: {e}")


# Example usage function
def example_usage():
    """Demonstrate the generalized evaluation system."""
    
    print("=== Generalized Evaluation System Demo ===\n")
    
    # Create sample data
    evaluation_data = pd.DataFrame({
        'prompt': ['What is 2+2?', 'Explain photosynthesis'],
        'response': ['2+2 equals 4', 'Photosynthesis is the process by which plants make food'],
        'ground_truth': ['4', 'Plants convert sunlight to energy']
    })
    
    metrics_config_data = pd.DataFrame({
        'name': ['Accuracy', 'Completeness'],
        'type': ['binary', 'scale_1_5'],
        'description': ['Is the answer correct?', 'How complete is the response?'],
        'grading_rubric': [
            'Score 1 if answer is mathematically correct, 0 otherwise',
            'Score 1-5 based on completeness: 1=minimal, 5=comprehensive'
        ],
        'threshold': [1.0, 3.0],
        'ground_truth_column': ['ground_truth', 'ground_truth'],
        'ground_truth_file_path': ['', '']
    })
    
    # Run evaluation with different directory configurations
    print("1. Using auto-detected environment:")
    results1 = run_evaluation_pipeline(
        evaluation_data=evaluation_data,
        metrics_config_data=metrics_config_data,
        save_results=False,
        log_to_mlflow=False
    )
    
    print("\n2. Using local environment configuration:")
    local_dm = create_directory_manager(
        workspace_root="/tmp/eval_test",
        environment="local"
    )
    results2 = run_evaluation_pipeline(
        evaluation_data=evaluation_data,
        metrics_config_data=metrics_config_data,
        directory_manager=local_dm,
        save_results=False,
        log_to_mlflow=False
    )
    
    print(f"\nResults shape: {results1.shape}")
    print(f"Sample results:\n{results1.head()}")


if __name__ == "__main__":
    example_usage()