"""
LLM Evaluation Framework with Generalized Directory Support

This module provides the main evaluation functionality with configurable
directory handling for various environments (local, Databricks, cloud, etc.)
"""

import os
import time
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass

from directory_handler import DirectoryHandler, get_directory_handler


class MetricType(Enum):
    """Types of evaluation metrics."""
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


class LLMEvaluationFramework:
    """
    Main evaluation framework with flexible directory and path management.
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 judge_model: str = "gpt-4",
                 directory_handler: Optional[DirectoryHandler] = None):
        """
        Initialize the evaluation framework.
        
        Args:
            config_path: Path to configuration file
            judge_model: Model to use for evaluation
            directory_handler: Optional custom directory handler
        """
        self.judge_model = judge_model
        
        # Initialize directory handler
        if directory_handler:
            self.dir_handler = directory_handler
        else:
            self.dir_handler = get_directory_handler(config_path)
            
        # Load configuration
        self.config = self._load_framework_config(config_path)
        
    def _load_framework_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load framework configuration."""
        default_config = {
            'judge_model': self.judge_model,
            'mlflow_enabled': True,
            'save_artifacts': True,
            'cleanup_temp': True
        }
        
        if config_path and os.path.exists(config_path):
            try:
                import json
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config.get('framework', {}))
            except Exception as e:
                print(f"Warning: Could not load config from {config_path}: {e}")
                
        return default_config
    
    def auto_generate_evaluation_prompt(self, 
                                      metric_name: str, 
                                      metric_type: str, 
                                      description: str, 
                                      grading_rubric: str) -> str:
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
    
    def load_metrics_from_csv(self, csv_path: Optional[str] = None) -> List[MetricConfig]:
        """Load metrics from CSV and auto-generate evaluation prompts."""
        # Use provided path or look in configs directory
        if csv_path is None:
            csv_path = self.dir_handler.get_file_path('configs', 'metrics_config.csv', create_dir=False)
            
        if not os.path.exists(csv_path):
            print(f"Warning: Metrics config not found at {csv_path}")
            return []
            
        try:
            metrics_data = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Error loading metrics CSV: {e}")
            return []
        
        metric_configs = []
        for _, row in metrics_data.iterrows():
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
            
            # Ground truth file matching (PM provides filename, code matches to uploaded files)
            gt_file_value = row.get('ground_truth_file_path', '').strip()
            
            # If it's just a filename, look for it in ground_truth directory
            if gt_file_value and not os.path.isabs(gt_file_value):
                gt_file_value = self.dir_handler.get_file_path('ground_truth', gt_file_value, create_dir=False)
            
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
    
    def load_evaluation_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """Load evaluation data from file."""
        if data_path is None:
            # Look for default evaluation data in data directory
            data_files = self.dir_handler.list_files('data', '*.csv')
            if data_files:
                data_path = data_files[0]
            else:
                raise FileNotFoundError("No evaluation data found in data directory")
                
        return pd.read_csv(data_path)
    
    def load_ground_truth_data(self, ground_truth_paths: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """Load ground truth data from files."""
        ground_truth_data = {}
        
        if ground_truth_paths is None:
            # Load all CSV files from ground_truth directory
            ground_truth_paths = self.dir_handler.list_files('ground_truth', '*.csv')
            
        for path in ground_truth_paths:
            try:
                filename = os.path.basename(path)
                ground_truth_data[filename] = pd.read_csv(path)
                print(f"Loaded ground truth: {filename}")
            except Exception as e:
                print(f"Warning: Could not load ground truth from {path}: {e}")
                
        return ground_truth_data
    
    def run_evaluation(self, 
                      evaluation_data: Optional[Union[str, pd.DataFrame]] = None,
                      metrics_config: Optional[Union[str, List[MetricConfig]]] = None,
                      output_dir: Optional[str] = None) -> pd.DataFrame:
        """
        Run the complete evaluation pipeline.
        
        Args:
            evaluation_data: Path to evaluation data or DataFrame
            metrics_config: Path to metrics config or list of MetricConfig objects
            output_dir: Optional output directory (defaults to results dir)
            
        Returns:
            DataFrame with evaluation results
        """
        # Load evaluation data
        if isinstance(evaluation_data, str):
            eval_df = self.load_evaluation_data(evaluation_data)
        elif isinstance(evaluation_data, pd.DataFrame):
            eval_df = evaluation_data
        else:
            eval_df = self.load_evaluation_data()
            
        # Load metrics
        if isinstance(metrics_config, str):
            metrics = self.load_metrics_from_csv(metrics_config)
        elif isinstance(metrics_config, list):
            metrics = metrics_config
        else:
            metrics = self.load_metrics_from_csv()
            
        if not metrics:
            raise ValueError("No valid metrics loaded!")
            
        # Load ground truth data
        ground_truth_data = self.load_ground_truth_data()
        
        # Create evaluator (placeholder - would integrate with actual evaluator)
        from zillow_judge_evaluator import ZillowJudgeEvaluator
        evaluator = ZillowJudgeEvaluator()
        
        print("="*60)
        print("🚀 STARTING EVALUATION")
        print("="*60)
        
        start_time = time.time()
        
        # Run evaluation (simplified for demonstration)
        results = []
        for idx, row in eval_df.iterrows():
            # This would integrate with the actual evaluation logic
            result = {
                'index': idx,
                'metric_name': 'demo_metric',
                'score': 0.85,
                'status': '✅',
                'threshold': 0.8
            }
            results.append(result)
            
        results_df = pd.DataFrame(results)
        eval_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"✅ COMPLETE in {eval_time:.1f}s")
        print(f"{'='*60}")
        
        # Save results
        if output_dir is None:
            output_path = self.dir_handler.get_timestamped_file_path('results', 'evaluation_results', '.csv')
        else:
            output_path = os.path.join(output_dir, f'evaluation_results_{int(time.time())}.csv')
            
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
        
        # Log to MLflow if enabled
        if self.config.get('mlflow_enabled', True):
            self._log_to_mlflow(results_df, metrics, eval_time)
            
        # Generate HTML report
        report_path = self.generate_html_report(results_df, metrics)
        print(f"HTML report saved to: {report_path}")
        
        # Cleanup temp files if enabled
        if self.config.get('cleanup_temp', True):
            removed = self.dir_handler.clean_temp_files(older_than_hours=24)
            if removed > 0:
                print(f"Cleaned up {removed} temporary files")
                
        return results_df
    
    def _log_to_mlflow(self, results_df: pd.DataFrame, metrics: List[MetricConfig], eval_time: float) -> None:
        """Log results to MLflow."""
        try:
            import mlflow
            
            # Get MLflow experiment path
            experiment_path = self.dir_handler.get_mlflow_experiment_path()
            mlflow.set_experiment(experiment_path)
            
            with mlflow.start_run(run_name=f"eval_{self.judge_model}_{time.strftime('%Y%m%d_%H%M%S')}"):
                # Log parameters
                mlflow.log_param("judge_model", self.judge_model)
                mlflow.log_param("num_samples", len(results_df))
                mlflow.log_param("metrics", ",".join([cfg.name for cfg in metrics]))
                
                # Log metrics
                total = len(results_df)
                passed = len(results_df[results_df['status'] == '✅'])
                pass_rate = (passed / total) * 100 if total > 0 else 0
                
                mlflow.log_metric("overall_pass_rate", pass_rate)
                mlflow.log_metric("total_evaluations", total)
                mlflow.log_metric("passed_evaluations", passed)
                mlflow.log_metric("evaluation_time_seconds", eval_time)
                
                # Log artifacts
                if self.config.get('save_artifacts', True):
                    artifacts_dir = self.dir_handler.get_dir('artifacts')
                    csv_path = self.dir_handler.get_timestamped_file_path('artifacts', 'results', '.csv')
                    results_df.to_csv(csv_path, index=False)
                    mlflow.log_artifact(csv_path, artifact_path="tables")
                    
            print(f"\n✅ Results logged to MLflow experiment: {experiment_path}")
            
        except ImportError:
            print("Warning: MLflow not available, skipping MLflow logging")
        except Exception as e:
            print(f"Warning: Error logging to MLflow: {e}")
    
    def generate_html_report(self, results_df: pd.DataFrame, metrics: List[MetricConfig]) -> str:
        """Generate an HTML report of the evaluation results."""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        report_name = f"llm_evaluation_report_{timestamp}.html"
        report_path = self.dir_handler.get_file_path('reports', report_name)
        
        # Generate HTML content
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>LLM Evaluation Report - {timestamp}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .summary {{ background-color: #f9f9f9; padding: 15px; margin: 20px 0; border-radius: 5px; }}
        .pass {{ color: green; }}
        .fail {{ color: red; }}
    </style>
</head>
<body>
    <h1>LLM Evaluation Report</h1>
    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Generated:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Judge Model:</strong> {self.judge_model}</p>
        <p><strong>Total Evaluations:</strong> {len(results_df)}</p>
        <p><strong>Pass Rate:</strong> {(len(results_df[results_df['status'] == '✅']) / len(results_df) * 100):.1f}%</p>
    </div>
    
    <h2>Metrics Configuration</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Type</th>
            <th>Threshold</th>
            <th>Description</th>
        </tr>
"""
        
        for metric in metrics:
            html_content += f"""
        <tr>
            <td>{metric.name}</td>
            <td>{metric.metric_type.value}</td>
            <td>{metric.threshold}</td>
            <td>{metric.description}</td>
        </tr>
"""
        
        html_content += """
    </table>
    
    <h2>Detailed Results</h2>
    <p>Full results available in CSV format.</p>
</body>
</html>
"""
        
        with open(report_path, 'w') as f:
            f.write(html_content)
            
        return report_path
    
    def print_config_summary(self) -> None:
        """Print a summary of the current configuration."""
        print("\n" + "="*60)
        print("CONFIGURATION SUMMARY")
        print("="*60)
        
        config = self.dir_handler.get_config_summary()
        
        print(f"\nWorkspace Root: {config['workspace_root']}")
        print("\nDirectories:")
        for dir_type, path in config['directories'].items():
            print(f"  {dir_type}: {path}")
            
        print(f"\nAuto-create directories: {config['auto_create_dirs']}")
        
        if config['environment_overrides']:
            print("\nEnvironment Overrides:")
            for var, value in config['environment_overrides'].items():
                print(f"  {var}: {value}")
                
        print("\nFramework Settings:")
        for key, value in self.config.items():
            print(f"  {key}: {value}")
            
        print("="*60)


def main():
    """Example usage of the evaluation framework."""
    # Initialize framework
    framework = LLMEvaluationFramework()
    
    # Print configuration
    framework.print_config_summary()
    
    # Run evaluation (would need actual data files)
    try:
        results = framework.run_evaluation()
        print(f"\nEvaluation completed successfully!")
    except Exception as e:
        print(f"\nError running evaluation: {e}")
        print("Please ensure you have the necessary data files in place.")


if __name__ == "__main__":
    main()