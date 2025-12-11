"""
Generalized LLM Judge Evaluator with Flexible Directory Configuration

This module provides a flexible evaluation system that can:
1. Load metrics from CSV files with auto-generated prompts
2. Support multiple metric types (binary, scale, percentage)
3. Handle different ground truth data sources
4. Auto-generate evaluation prompts from grading rubrics
5. Export results to various formats
"""

import json
import os
import time
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import mlflow
import mlflow.sklearn


class MetricType(Enum):
    """Supported metric types for evaluation."""
    BINARY = "binary"
    SCALE_1_5 = "1-5_scale"
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


class GeneralizedEvaluator:
    """
    Generalized evaluator that can handle different metric configurations
    and evaluation scenarios.
    """
    
    def __init__(
        self,
        judge_model: str = "gpt-4",
        metrics_config_path: Optional[str] = None,
        ground_truth_data: Optional[Dict[str, Any]] = None,
        assets_directory: Optional[str] = None
    ):
        """
        Initialize the generalized evaluator.
        
        Args:
            judge_model: The LLM model to use for evaluation
            metrics_config_path: Path to CSV file containing metrics configuration
            ground_truth_data: Dictionary containing ground truth data
            assets_directory: Directory containing asset files (JSON, CSV, etc.)
        """
        self.judge_model = judge_model
        self.metrics_config_path = metrics_config_path
        self.ground_truth_data = ground_truth_data or {}
        self.assets_directory = Path(assets_directory) if assets_directory else Path(__file__).parent / "assets"
        
        # Load metrics configuration
        self.metric_configs = self._load_metrics_config()
        
        # Initialize MLflow if available
        self._setup_mlflow()
    
    def _setup_mlflow(self):
        """Setup MLflow for experiment tracking."""
        try:
            # Try to get current user for experiment naming
            import dbutils
            username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
            experiment_name = f"/Users/{username}/llm_evaluation_experiment"
        except:
            experiment_name = "/llm_evaluation_experiment"
        
        mlflow.set_experiment(experiment_name)
    
    def auto_generate_evaluation_prompt(
        self, 
        metric_name: str, 
        metric_type: str, 
        description: str, 
        grading_rubric: str
    ) -> str:
        """
        Auto-generate evaluation prompt from grading rubric.
        
        Args:
            metric_name: Name of the metric
            metric_type: Type of metric (binary, scale, percentage)
            description: Description of what the metric evaluates
            grading_rubric: Detailed grading criteria
            
        Returns:
            Formatted evaluation prompt
        """
        # Add grading rubric section
        rubric_section = f"\n**Grading Rubric:**\n{grading_rubric}\n" if grading_rubric else ""
        
        # Determine output format based on metric type
        if metric_type.lower() == "binary":
            output_format = '{"score": true/false, "explanation": "Brief explanation"}'
        elif metric_type.lower() in ["1-5_scale", "scale_1_5"]:
            output_format = '{"score": 1-5, "explanation": "Brief explanation"}'
        elif metric_type.lower() == "percentage":
            output_format = '{"score": 0-100, "explanation": "Brief explanation"}'
        else:
            output_format = '{"score": <your_score>, "explanation": "Brief explanation"}'
        
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
    
    def _load_metrics_config(self) -> List[MetricConfig]:
        """Load metrics configuration from CSV file."""
        if not self.metrics_config_path or not os.path.exists(self.metrics_config_path):
            return self._get_default_metrics()
        
        try:
            df = pd.read_csv(self.metrics_config_path)
            metric_configs = []
            
            for _, row in df.iterrows():
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
                
                # Generate or use provided prompt template
                if 'grading_rubric' in row and pd.notna(row.get('grading_rubric')):
                    # NEW FORMAT: Auto-generate evaluation prompt from grading rubric
                    grading_rubric = str(row['grading_rubric']).strip()
                    prompt_template = self.auto_generate_evaluation_prompt(
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
                    prompt_template = self.auto_generate_evaluation_prompt(
                        metric_name=row['name'].strip(),
                        metric_type=metric_type_str,
                        description=row.get('description', '').strip(),
                        grading_rubric=""
                    )
                    print(f"⚠️ {row['name'].strip()} - No rubric or prompt provided, using basic prompt")
                
                # Ground truth file matching
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
            
        except Exception as e:
            print(f"❌ Error loading metrics config: {e}")
            return self._get_default_metrics()
    
    def _get_default_metrics(self) -> List[MetricConfig]:
        """Get default metrics configuration if CSV loading fails."""
        print("⚠️ Using default metrics configuration")
        
        default_metrics = [
            MetricConfig(
                name="accuracy",
                metric_type=MetricType.BINARY,
                description="Evaluates if the response is factually accurate",
                prompt_template=self.auto_generate_evaluation_prompt(
                    "accuracy", "binary", 
                    "Evaluates if the response is factually accurate",
                    "Check if the response contains correct information and facts."
                ),
                threshold=0.5,
                ground_truth_column="ground_truth",
                ground_truth_file_path=""
            ),
            MetricConfig(
                name="completeness",
                metric_type=MetricType.SCALE_1_5,
                description="Evaluates how complete the response is",
                prompt_template=self.auto_generate_evaluation_prompt(
                    "completeness", "1-5_scale",
                    "Evaluates how complete the response is",
                    "Rate from 1-5: 1=very incomplete, 5=completely addresses all aspects"
                ),
                threshold=3.0,
                ground_truth_column="ground_truth",
                ground_truth_file_path=""
            )
        ]
        
        return default_metrics
    
    def evaluate_dataset(self, evaluation_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Evaluate a dataset using the configured metrics.
        
        Args:
            evaluation_data: List of dictionaries containing evaluation samples
            
        Returns:
            DataFrame with evaluation results
        """
        results = []
        
        print(f"🚀 Starting evaluation of {len(evaluation_data)} samples...")
        
        for i, sample in enumerate(evaluation_data):
            print(f"Processing sample {i+1}/{len(evaluation_data)}...")
            
            # Extract sample data
            prompt = sample.get('prompt', '')
            response = sample.get('response', '')
            ground_truth = sample.get('ground_truth', '')
            
            # Evaluate each metric
            for metric_config in self.metric_configs:
                try:
                    # Format the prompt template with actual data
                    formatted_prompt = metric_config.prompt_template.format(
                        prompt=prompt,
                        response=response,
                        ground_truth=ground_truth
                    )
                    
                    # Here you would call your LLM judge
                    # For now, we'll simulate the evaluation
                    score, explanation = self._simulate_llm_evaluation(
                        formatted_prompt, metric_config
                    )
                    
                    # Determine if the metric passed
                    passed = self._evaluate_metric_pass(score, metric_config)
                    
                    results.append({
                        'sample_id': i,
                        'metric_name': metric_config.name,
                        'score': score,
                        'explanation': explanation,
                        'threshold': metric_config.threshold,
                        'status': '✅' if passed else '❌',
                        'prompt': prompt[:100] + '...' if len(prompt) > 100 else prompt
                    })
                    
                except Exception as e:
                    print(f"❌ Error evaluating {metric_config.name} for sample {i}: {e}")
                    results.append({
                        'sample_id': i,
                        'metric_name': metric_config.name,
                        'score': None,
                        'explanation': f"Error: {str(e)}",
                        'threshold': metric_config.threshold,
                        'status': '❌',
                        'prompt': prompt[:100] + '...' if len(prompt) > 100 else prompt
                    })
        
        return pd.DataFrame(results)
    
    def _simulate_llm_evaluation(self, prompt: str, metric_config: MetricConfig) -> tuple:
        """
        Simulate LLM evaluation (replace with actual LLM call).
        
        Args:
            prompt: Formatted evaluation prompt
            metric_config: Metric configuration
            
        Returns:
            Tuple of (score, explanation)
        """
        # This is a placeholder - replace with actual LLM API call
        import random
        
        if metric_config.metric_type == MetricType.BINARY:
            score = random.choice([True, False])
            explanation = "Simulated binary evaluation"
        elif metric_config.metric_type == MetricType.SCALE_1_5:
            score = random.randint(1, 5)
            explanation = "Simulated 1-5 scale evaluation"
        elif metric_config.metric_type == MetricType.PERCENTAGE:
            score = random.randint(0, 100)
            explanation = "Simulated percentage evaluation"
        else:
            score = random.random()
            explanation = "Simulated evaluation"
        
        return score, explanation
    
    def _evaluate_metric_pass(self, score: Any, metric_config: MetricConfig) -> bool:
        """Determine if a metric score passes the threshold."""
        if score is None:
            return False
        
        try:
            if metric_config.metric_type == MetricType.BINARY:
                return bool(score)
            else:
                return float(score) >= metric_config.threshold
        except (ValueError, TypeError):
            return False
    
    def export_results(self, results_df: pd.DataFrame, output_dir: str = "/tmp/llm_eval_artifacts") -> Dict[str, str]:
        """
        Export evaluation results to various formats.
        
        Args:
            results_df: DataFrame containing evaluation results
            output_dir: Directory to save exported files
            
        Returns:
            Dictionary mapping format names to file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = int(time.time())
        
        exported_files = {}
        
        # Export to CSV
        csv_path = os.path.join(output_dir, f"results_{timestamp}.csv")
        results_df.to_csv(csv_path, index=False)
        exported_files['csv'] = csv_path
        
        # Export to JSON
        json_path = os.path.join(output_dir, f"results_{timestamp}.json")
        results_df.to_json(json_path, orient='records', indent=2)
        exported_files['json'] = json_path
        
        # Export summary statistics
        summary = self._generate_summary(results_df)
        summary_path = os.path.join(output_dir, f"summary_{timestamp}.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        exported_files['summary'] = summary_path
        
        return exported_files
    
    def _generate_summary(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics from evaluation results."""
        total = len(results_df)
        passed = len(results_df[results_df['status'] == '✅'])
        pass_rate = (passed / total) * 100 if total > 0 else 0
        
        summary = {
            'total_evaluations': total,
            'passed_evaluations': passed,
            'failed_evaluations': total - passed,
            'overall_pass_rate': pass_rate,
            'metrics_summary': {}
        }
        
        # Per-metric summary
        for metric_name in results_df['metric_name'].unique():
            metric_results = results_df[results_df['metric_name'] == metric_name]
            metric_passed = len(metric_results[metric_results['status'] == '✅'])
            metric_total = len(metric_results)
            metric_pass_rate = (metric_passed / metric_total) * 100 if metric_total > 0 else 0
            
            # Calculate average score for numeric metrics
            numeric_scores = []
            for score in metric_results['score']:
                try:
                    numeric_scores.append(float(score))
                except (ValueError, TypeError):
                    continue
            
            avg_score = sum(numeric_scores) / len(numeric_scores) if numeric_scores else None
            
            summary['metrics_summary'][metric_name] = {
                'pass_rate': metric_pass_rate,
                'passed': metric_passed,
                'total': metric_total,
                'average_score': avg_score,
                'threshold': metric_results['threshold'].iloc[0] if len(metric_results) > 0 else None
            }
        
        return summary
    
    def log_to_mlflow(self, results_df: pd.DataFrame, run_name: Optional[str] = None):
        """
        Log evaluation results to MLflow.
        
        Args:
            results_df: DataFrame containing evaluation results
            run_name: Optional name for the MLflow run
        """
        if not run_name:
            run_name = f"eval_{self.judge_model}_{time.strftime('%Y%m%d_%H%M%S')}"
        
        with mlflow.start_run(run_name=run_name):
            # Log parameters
            mlflow.log_param("judge_model", self.judge_model)
            mlflow.log_param("num_samples", len(results_df['sample_id'].unique()))
            mlflow.log_param("metrics", ",".join([cfg.name for cfg in self.metric_configs]))
            
            # Log overall metrics
            summary = self._generate_summary(results_df)
            mlflow.log_metric("overall_pass_rate", summary['overall_pass_rate'])
            mlflow.log_metric("total_evaluations", summary['total_evaluations'])
            mlflow.log_metric("passed_evaluations", summary['passed_evaluations'])
            
            # Log per-metric results
            for metric_name, metric_summary in summary['metrics_summary'].items():
                mlflow.log_metric(f"{metric_name}_pass_rate", metric_summary['pass_rate'])
                if metric_summary['average_score'] is not None:
                    mlflow.log_metric(f"{metric_name}_mean", metric_summary['average_score'])
            
            # Export and log artifacts
            exported_files = self.export_results(results_df)
            for format_name, file_path in exported_files.items():
                mlflow.log_artifact(file_path, artifact_path="tables")
            
            print(f"✅ Results logged to MLflow run: {run_name}")


def create_sample_metrics_csv(output_path: str = "sample_metrics.csv"):
    """
    Create a sample metrics configuration CSV file.
    
    Args:
        output_path: Path where to save the CSV file
    """
    sample_data = [
        {
            'name': 'accuracy',
            'type': 'binary',
            'description': 'Evaluates if the response is factually accurate',
            'grading_rubric': 'Check if the response contains correct information and facts. Look for any false statements or misleading information.',
            'threshold': 0.5,
            'ground_truth_column': 'ground_truth',
            'ground_truth_file_path': 'ground_truth_data.csv'
        },
        {
            'name': 'completeness',
            'type': '1-5_scale',
            'description': 'Evaluates how complete the response is',
            'grading_rubric': 'Rate from 1-5: 1=very incomplete, missing key information; 2=mostly incomplete; 3=partially complete; 4=mostly complete; 5=completely addresses all aspects of the question.',
            'threshold': 3.0,
            'ground_truth_column': 'ground_truth',
            'ground_truth_file_path': 'ground_truth_data.csv'
        },
        {
            'name': 'helpfulness',
            'type': 'percentage',
            'description': 'Evaluates how helpful the response is to the user',
            'grading_rubric': 'Rate from 0-100%: Consider how actionable, clear, and useful the response is for the user\'s specific needs.',
            'threshold': 70.0,
            'ground_truth_column': 'ground_truth',
            'ground_truth_file_path': 'ground_truth_data.csv'
        }
    ]
    
    df = pd.DataFrame(sample_data)
    df.to_csv(output_path, index=False)
    print(f"✅ Sample metrics CSV created: {output_path}")


def main():
    """Example usage of the generalized evaluator."""
    print("=== Generalized LLM Judge Evaluator ===\n")
    
    # Create sample metrics CSV
    create_sample_metrics_csv("sample_metrics.csv")
    
    # Initialize evaluator
    evaluator = GeneralizedEvaluator(
        judge_model="gpt-4",
        metrics_config_path="sample_metrics.csv"
    )
    
    # Create sample evaluation data
    sample_data = [
        {
            'prompt': 'What is the capital of France?',
            'response': 'The capital of France is Paris.',
            'ground_truth': 'Paris'
        },
        {
            'prompt': 'How do I calculate mortgage payments?',
            'response': 'You can calculate mortgage payments using the formula: P = L[c(1 + c)^n]/[(1 + c)^n - 1] where P is payment, L is loan amount, c is monthly interest rate, and n is number of payments.',
            'ground_truth': 'Mortgage payment calculation formula'
        }
    ]
    
    # Run evaluation
    print("Running evaluation...")
    results_df = evaluator.evaluate_dataset(sample_data)
    
    # Display results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(results_df.to_string(index=False))
    
    # Generate summary
    summary = evaluator._generate_summary(results_df)
    print(f"\n📊 SUMMARY")
    print(f"Total Evaluations: {summary['total_evaluations']}")
    print(f"Passed: {summary['passed_evaluations']}")
    print(f"Pass Rate: {summary['overall_pass_rate']:.1f}%")
    
    # Export results
    exported_files = evaluator.export_results(results_df)
    print(f"\n📁 Exported files:")
    for format_name, file_path in exported_files.items():
        print(f"  {format_name}: {file_path}")
    
    # Log to MLflow
    evaluator.log_to_mlflow(results_df)


if __name__ == "__main__":
    main()