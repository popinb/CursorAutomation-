"""
Generic Evaluation Template for LLM Responses

This module provides a flexible framework for evaluating LLM responses using
customizable metrics and MLflow tracking.
"""

import asyncio
import collections
import json
import os
import time
from typing import Any, Dict, List, Optional, Type, Union
import yaml

import httpx
import pandas as pd
from langchain_openai import ChatOpenAI
import mlflow
import mlflow.metrics.genai
from pydantic import BaseModel

class EvaluationConfig:
    """Configuration class for the evaluation template."""
    
    def __init__(self, config_path: str):
        """Initialize configuration from YAML file."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
    
    @property
    def dataset_config(self) -> Dict[str, Any]:
        return self.config.get('dataset', {})
    
    @property
    def experiment_config(self) -> Dict[str, Any]:
        return self.config.get('experiment', {})
    
    @property
    def judges_config(self) -> Dict[str, Any]:
        return self.config.get('judges', {})
    
    @property
    def metrics_config(self) -> Dict[str, Any]:
        return self.config.get('metrics', {})
    
    @property
    def composite_config(self) -> Dict[str, Any]:
        return self.config.get('composite_metrics', {})
    
    @property
    def mlflow_config(self) -> Dict[str, Any]:
        return self.config.get('mlflow', {})


class MetricResult(BaseModel):
    """Base class for metric evaluation results."""
    explanation: str


class DataLoader:
    """Handles loading and preprocessing of evaluation datasets."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.dataset_config = config.dataset_config
        self.column_mapping = self.dataset_config.get('columns', {})
    
    def load_dataset(self) -> pd.DataFrame:
        """Load dataset from configured path and apply column mapping."""
        file_path = self.dataset_config.get('file_path')
        if not file_path or not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        
        # Validate required columns exist
        required_cols = ['prompt', 'response']
        for col in required_cols:
            mapped_col = self.column_mapping.get(col, col)
            if mapped_col not in df.columns:
                raise ValueError(f"Required column '{mapped_col}' not found in dataset")
        
        # Apply column mapping
        df_mapped = df.copy()
        for standard_name, actual_name in self.column_mapping.items():
            if actual_name in df.columns:
                df_mapped[standard_name] = df[actual_name]
        
        return df_mapped
    
    def validate_dataset(self, df: pd.DataFrame) -> bool:
        """Validate that the dataset has required structure."""
        required_columns = ['prompt', 'response']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if df.empty:
            raise ValueError("Dataset is empty")
        
        return True


class LLMJudge:
    """Handles LLM-based evaluation of responses."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.judges_config = config.judges_config
        self.models = self._initialize_models()
    
    def _initialize_models(self) -> List[ChatOpenAI]:
        """Initialize ChatOpenAI models for judging."""
        models = []
        api_config = self.judges_config.get('api', {})
        
        # Set up API configuration from environment
        openai_key_env = api_config.get('openai_api_key_env', 'OPENAI_API_KEY')
        openai_base_env = api_config.get('openai_base_url_env', 'OPENAI_API_BASE')
        
        if openai_key_env in os.environ:
            openai_key = os.environ[openai_key_env]
        else:
            raise ValueError(f"Environment variable {openai_key_env} not set")
        
        base_url = os.environ.get(openai_base_env)
        
        for model_name in self.judges_config.get('models', ['gpt-4o']):
            model_kwargs = {"response_format": {"type": "json_object"}}
            
            model = ChatOpenAI(
                model_name=model_name,
                temperature=api_config.get('temperature', 0),
                default_headers={"apikey": openai_key} if openai_key else None,
                base_url=base_url,
                model_kwargs=model_kwargs,
            )
            models.append(model)
        
        return models
    
    def evaluate_single_metric(self, 
                             metric_name: str, 
                             metric_config: Dict[str, Any], 
                             eval_data: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate a single metric across all data points."""
        results = []
        details = []
        
        prompt_template = metric_config['prompt_template']
        score_key = metric_config['score_key']
        
        for _, row in eval_data.iterrows():
            # Format the evaluation prompt with available data
            format_kwargs = {
                'prompt': row.get('prompt', ''),
                'response': row.get('response', ''),
                'ground_truth': row.get('ground_truth', ''),
                'metadata': row.get('metadata', ''),
            }
            
            eval_prompt = prompt_template.format(**format_kwargs)
            
            # Get judgments from all models
            model_scores = []
            model_details = []
            
            for model in self.models:
                try:
                    llm_response = model.invoke(eval_prompt)
                    result_json = json.loads(llm_response.content)
                    
                    score = result_json.get(score_key, 0)
                    model_scores.append(float(score))
                    model_details.append({
                        'model': getattr(model, 'model_name', 'unknown'),
                        'result': result_json
                    })
                    
                except Exception as e:
                    print(f"Error evaluating {metric_name} with model {getattr(model, 'model_name', 'unknown')}: {e}")
                    model_scores.append(0.0)
                    model_details.append({
                        'model': getattr(model, 'model_name', 'unknown'),
                        'error': str(e)
                    })
            
            # Ensemble scoring (majority vote for discrete, average for continuous)
            if len(set(model_scores)) <= metric_config.get('scale', 5):
                # Discrete scores - use majority vote
                counter = collections.Counter(model_scores)
                final_score = max(counter.items(), key=lambda x: (x[1], -x[0]))[0]
            else:
                # Continuous scores - use average
                final_score = sum(model_scores) / len(model_scores) if model_scores else 0.0
            
            results.append(final_score)
            details.append({
                'model_scores': model_scores,
                'model_details': model_details,
                'final_score': final_score
            })
        
        return {
            'scores': results,
            'details': details,
            'mean_score': sum(results) / len(results) if results else 0.0
        }


class CompositeScorer:
    """Handles computation of composite scores from individual metrics."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.composite_config = config.composite_config
    
    def compute_composite_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all configured composite scores."""
        df_result = df.copy()
        
        for score_name, score_config in self.composite_config.items():
            method = score_config.get('method', 'average')
            metrics = score_config.get('metrics', [])
            weights = score_config.get('weights', {})
            
            # Validate that all required metrics exist
            missing_metrics = [m for m in metrics if m not in df_result.columns]
            if missing_metrics:
                print(f"Warning: Missing metrics for composite score {score_name}: {missing_metrics}")
                continue
            
            if method == 'average':
                df_result[score_name] = df_result[metrics].mean(axis=1)
            elif method == 'weighted_average':
                weighted_sum = sum(df_result[metric] * weights.get(metric, 1.0) for metric in metrics)
                total_weight = sum(weights.get(metric, 1.0) for metric in metrics)
                df_result[score_name] = weighted_sum / total_weight
            elif method == 'minimum':
                df_result[score_name] = df_result[metrics].min(axis=1)
            else:
                print(f"Unknown composite method: {method}")
        
        return df_result


class EvaluationPipeline:
    """Main evaluation pipeline that orchestrates the entire evaluation process."""
    
    def __init__(self, config_path: str):
        """Initialize the evaluation pipeline with configuration."""
        self.config = EvaluationConfig(config_path)
        self.data_loader = DataLoader(self.config)
        self.llm_judge = LLMJudge(self.config)
        self.composite_scorer = CompositeScorer(self.config)
    
    def run_evaluation(self) -> pd.DataFrame:
        """Run the complete evaluation pipeline."""
        print("Loading dataset...")
        df = self.data_loader.load_dataset()
        self.data_loader.validate_dataset(df)
        print(f"Loaded {len(df)} samples for evaluation")
        
        # Set up MLflow experiment
        if self.config.mlflow_config.get('enabled', True):
            self._setup_mlflow()
        
        # Run individual metric evaluations
        print("Running metric evaluations...")
        for metric_name, metric_config in self.config.metrics_config.items():
            if not metric_config.get('enabled', True):
                print(f"Skipping disabled metric: {metric_name}")
                continue
            
            print(f"Evaluating {metric_name}...")
            eval_results = self.llm_judge.evaluate_single_metric(
                metric_name, metric_config, df
            )
            
            # Add results to dataframe
            df[metric_name] = eval_results['scores']
            df[f"{metric_name}_details"] = eval_results['details']
            
            # Add threshold-based status
            threshold = metric_config.get('threshold', 0)
            df[f"{metric_name}_status"] = ["✅" if s >= threshold else "❌" for s in eval_results['scores']]
            
            print(f"  Mean {metric_name}: {eval_results['mean_score']:.3f}")
        
        # Compute composite scores
        print("Computing composite scores...")
        df = self.composite_scorer.compute_composite_scores(df)
        
        # Log to MLflow
        if self.config.mlflow_config.get('enabled', True):
            self._log_to_mlflow(df)
        
        # Save results
        self._save_results(df)
        
        return df
    
    def _setup_mlflow(self):
        """Set up MLflow experiment and tracking."""
        tracking_uri = self.config.mlflow_config.get('tracking_uri')
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        experiment_name = self.config.experiment_config.get('name', 'evaluation_experiment')
        mlflow.set_experiment(experiment_name)
        
        # Set experiment tags
        tags = self.config.mlflow_config.get('experiment_tags', {})
        for key, value in tags.items():
            mlflow.set_experiment_tag(key, value)
    
    def _log_to_mlflow(self, df: pd.DataFrame):
        """Log evaluation results to MLflow."""
        run_name = f"{self.config.experiment_config.get('run_name_prefix', 'eval_run')}_{int(time.time())}"
        
        with mlflow.start_run(run_name=run_name):
            # Log configuration
            mlflow.log_dict(self.config.config, "evaluation_config.yaml")
            
            # Log dataset info
            mlflow.log_param("dataset_size", len(df))
            mlflow.log_param("dataset_path", self.config.dataset_config.get('file_path'))
            
            # Log metric means
            for metric_name, metric_config in self.config.metrics_config.items():
                if metric_config.get('enabled', True) and metric_name in df.columns:
                    mean_score = df[metric_name].mean()
                    mlflow.log_metric(f"{metric_name}_mean", mean_score)
                    
                    # Log threshold performance
                    threshold = metric_config.get('threshold', 0)
                    pass_rate = (df[metric_name] >= threshold).mean()
                    mlflow.log_metric(f"{metric_name}_pass_rate", pass_rate)
            
            # Log composite scores
            for score_name in self.config.composite_config.keys():
                if score_name in df.columns:
                    mlflow.log_metric(f"{score_name}_mean", df[score_name].mean())
    
    def _save_results(self, df: pd.DataFrame):
        """Save evaluation results to files."""
        output_dir = self.config.experiment_config.get('output_directory', './results/')
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = int(time.time())
        
        # Save detailed results
        detailed_path = os.path.join(output_dir, f"evaluation_results_{timestamp}.csv")
        df.to_csv(detailed_path, index=False)
        print(f"Detailed results saved to: {detailed_path}")
        
        # Save summary statistics
        metric_columns = [name for name in self.config.metrics_config.keys() 
                         if self.config.metrics_config[name].get('enabled', True)]
        composite_columns = list(self.config.composite_config.keys())
        all_score_columns = metric_columns + composite_columns
        
        summary_stats = df[all_score_columns].describe()
        summary_path = os.path.join(output_dir, f"evaluation_summary_{timestamp}.csv")
        summary_stats.to_csv(summary_path)
        print(f"Summary statistics saved to: {summary_path}")


def main(config_path: str = "config/evaluation_config.yaml"):
    """Main entry point for running the evaluation pipeline."""
    try:
        pipeline = EvaluationPipeline(config_path)
        results_df = pipeline.run_evaluation()
        
        print("\n" + "="*50)
        print("EVALUATION COMPLETE")
        print("="*50)
        
        # Print summary of results
        metric_columns = [name for name in pipeline.config.metrics_config.keys() 
                         if pipeline.config.metrics_config[name].get('enabled', True)]
        
        print("\nMetric Summary:")
        for metric in metric_columns:
            if metric in results_df.columns:
                mean_score = results_df[metric].mean()
                threshold = pipeline.config.metrics_config[metric].get('threshold', 0)
                pass_rate = (results_df[metric] >= threshold).mean() * 100
                print(f"  {metric}: {mean_score:.3f} (pass rate: {pass_rate:.1f}%)")
        
        return results_df
        
    except Exception as e:
        print(f"Error running evaluation: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/evaluation_config.yaml"
    main(config_path)