"""
Universal Evaluation Template System
====================================

A flexible template for evaluating any dataset with custom metrics using MLflow.
This system allows users to define their own datasets, evaluation metrics, and scoring systems.

Usage:
    python template_evaluation_system.py --config config.yaml
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml

import httpx
import pandas as pd
from langchain_openai import ChatOpenAI
import mlflow
import mlflow.metrics
import mlflow.metrics.genai
from mlflow.genai.scorers import scorer
from pydantic import BaseModel
import collections


class EvaluationConfig:
    """Configuration class for evaluation parameters."""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Dataset configuration
        self.dataset_path = self.config['dataset']['path']
        self.prompt_column = self.config['dataset']['prompt_column']
        self.response_column = self.config['dataset']['response_column']
        self.ground_truth_column = self.config['dataset'].get('ground_truth_column')
        self.user_features_column = self.config['dataset'].get('user_features_column')
        
        # Model configuration
        self.response_model = self.config['models']['response_model']
        self.judge_models = self.config['models']['judge_models']
        self.openai_api_key = self.config['api_keys']['openai_api_key']
        self.openai_base_url = self.config['api_keys'].get('openai_base_url')
        
        # Evaluation configuration
        self.metrics = self.config['evaluation']['metrics']
        self.experiment_name = self.config['evaluation']['experiment_name']
        self.run_name = self.config['evaluation'].get('run_name')
        
        # API configuration (optional)
        self.api_config = self.config.get('api', {})
        
        # Set environment variables
        os.environ["OPENAI_API_KEY"] = self.openai_api_key
        if self.openai_base_url:
            os.environ["OPENAI_API_BASE"] = self.openai_base_url


class MetricTemplate:
    """Template for defining evaluation metrics."""
    
    def __init__(self, name: str, prompt_template: str, threshold: float, 
                 output_schema: Optional[BaseModel] = None, 
                 score_key: Optional[str] = None):
        self.name = name
        self.prompt_template = prompt_template
        self.threshold = threshold
        self.output_schema = output_schema
        self.score_key = score_key or f"{name.lower()}_score"


class UniversalEvaluator:
    """Universal evaluator that can work with any dataset and metrics."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.evaluators = {}
        self.scorers = []
        self._setup_evaluators()
        self._setup_scorers()
    
    def _setup_evaluators(self):
        """Setup evaluator functions for each metric."""
        for metric_name, metric_config in self.config.metrics.items():
            prompt_template = metric_config['prompt_template']
            threshold = metric_config.get('threshold', 0.5)
            output_schema = self._get_output_schema(metric_config.get('output_schema'))
            
            metric_template = MetricTemplate(
                name=metric_name,
                prompt_template=prompt_template,
                threshold=threshold,
                output_schema=output_schema
            )
            
            evaluator_func = self._create_ensemble_evaluator(metric_template)
            self.evaluators[metric_name] = evaluator_func
    
    def _get_output_schema(self, schema_config: Optional[Dict]) -> Optional[BaseModel]:
        """Create Pydantic model from schema configuration."""
        if not schema_config:
            return None
        
        # Create a dynamic Pydantic model
        fields = {}
        for field_name, field_type in schema_config['fields'].items():
            if field_type == 'int':
                fields[field_name] = (int, ...)
            elif field_type == 'str':
                fields[field_name] = (str, ...)
            elif field_type == 'float':
                fields[field_name] = (float, ...)
            else:
                fields[field_name] = (str, ...)
        
        return type('DynamicModel', (BaseModel,), fields)
    
    def _create_ensemble_evaluator(self, metric_template: MetricTemplate):
        """Create an ensemble evaluator for a metric."""
        models = [self._create_openai_model(model_name) for model_name in self.config.judge_models]
        
        def evaluator(eval_df, builtin_metrics=None):
            results = []
            details = []
            
            for _, row in eval_df.iterrows():
                # Format the evaluation prompt with available data
                eval_prompt = self._format_prompt(metric_template.prompt_template, row)
                
                model_outputs = []
                scores = []
                
                for model in models:
                    try:
                        if metric_template.output_schema:
                            structured = model.with_structured_output(metric_template.output_schema)
                            llm_response = structured.invoke(eval_prompt)
                            result_json = (
                                llm_response.model_dump()
                                if isinstance(llm_response, BaseModel)
                                else dict(llm_response)
                            )
                        else:
                            llm_response = model.invoke(eval_prompt)
                            result_json = json.loads(llm_response.content)
                        
                        model_outputs.append((getattr(model, "model_name", "model"), result_json))
                        
                        if metric_template.score_key not in result_json:
                            # Try to find the score key in nested structure
                            for key, value in result_json.items():
                                if isinstance(value, dict) and metric_template.score_key in value:
                                    result_json = value
                                    break
                        
                        scores.append(float(result_json[metric_template.score_key]))
                        
                    except Exception as exc:
                        model_outputs.append({"error": str(exc)})
                        scores.append(0)
                
                # Majority vote
                counter = collections.Counter(scores)
                most_common_score, _ = max(counter.items(), key=lambda x: (x[1], -x[0]))
                
                results.append(most_common_score)
                details.append({
                    "per_model": model_outputs,
                    "votes": scores,
                    "final_score": most_common_score,
                })
            
            mean_score = sum(results) / len(results) if results else 0.0
            return {
                f"{metric_template.name.lower()}/mean": mean_score,
                f"{metric_template.name.lower()}/scores": results,
                f"{metric_template.name.lower()}/details": details,
            }
        
        return evaluator
    
    def _format_prompt(self, template: str, row: pd.Series) -> str:
        """Format prompt template with row data."""
        # Get all available columns for formatting
        format_dict = {}
        for col in row.index:
            format_dict[col] = row[col]
        
        # Add common aliases
        format_dict['prompt'] = row.get(self.config.prompt_column, '')
        format_dict['response'] = row.get(self.config.response_column, '')
        format_dict['user_personalization_features'] = row.get(self.config.user_features_column, '')
        format_dict['ground_truth'] = row.get(self.config.ground_truth_column, '')
        
        return template.format(**format_dict)
    
    def _create_openai_model(self, model_name: str) -> ChatOpenAI:
        """Create OpenAI-compatible model."""
        return ChatOpenAI(
            model_name=model_name,
            default_headers={"apikey": self.config.openai_api_key},
            base_url=self.config.openai_base_url,
            temperature=0,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
    
    def _setup_scorers(self):
        """Setup MLflow scorers."""
        for metric_name, metric_config in self.config.metrics.items():
            threshold = metric_config.get('threshold', 0.5)
            score_key = f"{metric_name.lower()}_score"
            
            # Raw score scorer
            @scorer(name=metric_name)
            def scorer_fn(outputs, score_key=score_key):
                return outputs[score_key]
            
            # Status scorer (threshold-based)
            @scorer(name=f"{metric_name}_status")
            def status_scorer_fn(outputs, score_key=score_key, threshold=threshold):
                return outputs[score_key] >= threshold
            
            self.scorers.extend([scorer_fn, status_scorer_fn])
    
    def evaluate_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Evaluate the entire dataset."""
        print("Running evaluations...")
        
        # Prepare data for evaluation
        eval_data = df.copy()
        eval_data = eval_data.rename(columns={
            self.config.prompt_column: "inputs",
            self.config.response_column: "predictions"
        })
        
        # Run evaluations
        all_scores = {}
        
        for metric_name, evaluator_func in self.evaluators.items():
            print(f"Running {metric_name} evaluation...")
            
            eval_results = evaluator_func(eval_data)
            
            individual_scores = eval_results.get(f"{metric_name.lower()}/scores", [])
            mean_score = eval_results.get(f"{metric_name.lower()}/mean", 0.0)
            details = eval_results.get(f"{metric_name.lower()}/details", [])
            
            df[f"{metric_name}_details"] = details
            
            if len(individual_scores) == len(df):
                scores = individual_scores
            else:
                scores = [mean_score] * len(df)
            
            all_scores[metric_name] = scores
            
            print(f"✅ {metric_name}: {mean_score:.3f}")
        
        # Add scores and status to dataframe
        for metric_name, scores in all_scores.items():
            df[metric_name] = scores
            threshold = self.config.metrics[metric_name].get('threshold', 0.5)
            df[f"{metric_name}_status"] = ["✅" if s >= threshold else "❌" for s in scores]
        
        # Log to MLflow
        self._log_to_mlflow(df, all_scores)
        
        return df
    
    def _log_to_mlflow(self, df: pd.DataFrame, scores: Dict[str, List]):
        """Log evaluation results to MLflow."""
        mlflow.set_experiment(self.config.experiment_name)
        
        # Prepare data for MLflow evaluation
        db_data = []
        for _, row in df.iterrows():
            item = {
                "inputs": {
                    'question': row[self.config.prompt_column],
                    'answer': row[self.config.response_column],
                },
                "outputs": {}
            }
            
            # Add user features if available
            if self.config.user_features_column and self.config.user_features_column in row:
                item["inputs"]['user_features'] = row[self.config.user_features_column]
            
            # Add ground truth if available
            if self.config.ground_truth_column and self.config.ground_truth_column in row:
                item["inputs"]['ground_truth'] = row[self.config.ground_truth_column]
            
            # Add scores
            for metric_name in self.config.metrics.keys():
                item["outputs"][f"{metric_name.lower()}_score"] = row[metric_name]
                item["outputs"][f"{metric_name.lower()}_details"] = row[f"{metric_name}_details"]
            
            db_data.append(item)
        
        # Run MLflow evaluation
        with mlflow.start_run(run_name=self.config.run_name):
            mlflow.genai.evaluate(
                data=db_data,
                scorers=self.scorers
            )


class ResponseGenerator:
    """Generate responses using various models or APIs."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
    
    async def generate_responses(self, prompts: List[str]) -> List[str]:
        """Generate responses for a list of prompts."""
        if self.config.response_model == "GOLDEN_RESPONSE":
            return []  # Use existing responses
        
        elif self.config.response_model.startswith("LLM_"):
            return await self._generate_with_llm(prompts)
        
        elif self.config.response_model == "FIRST_CALL":
            return await self._generate_with_api(prompts)
        
        else:
            raise ValueError(f"Unknown response model: {self.config.response_model}")
    
    async def _generate_with_llm(self, prompts: List[str]) -> List[str]:
        """Generate responses using LLM."""
        model = ChatOpenAI(
            model=self.config.response_model.replace("LLM_", ""),
            temperature=0,
            max_tokens=2000
        )
        
        responses = []
        for prompt in prompts:
            response = model.invoke(prompt)
            responses.append(response.content)
        
        return responses
    
    async def _generate_with_api(self, prompts: List[str]) -> List[str]:
        """Generate responses using external API."""
        if not self.config.api_config:
            raise ValueError("API configuration required for FIRST_CALL model")
        
        # Implement API call logic here
        # This would be similar to the original call_endpoint function
        pass


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Universal Evaluation Template System")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--output", help="Output directory for results")
    args = parser.parse_args()
    
    # Load configuration
    config = EvaluationConfig(args.config)
    
    # Load dataset
    print(f"Loading dataset from {config.dataset_path}")
    df = pd.read_csv(config.dataset_path)
    print(f"Loaded {len(df)} rows")
    
    # Generate responses if needed
    if config.response_model != "GOLDEN_RESPONSE":
        print(f"Generating responses with {config.response_model}")
        response_generator = ResponseGenerator(config)
        responses = asyncio.run(response_generator.generate_responses(df[config.prompt_column].tolist()))
        df[config.response_column] = responses
    
    # Run evaluation
    evaluator = UniversalEvaluator(config)
    results_df = evaluator.evaluate_dataset(df)
    
    # Save results
    output_dir = Path(args.output) if args.output else Path("results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"evaluation_results_{timestamp}.csv"
    results_df.to_csv(output_file, index=False)
    
    print(f"✅ Evaluation complete - results saved to {output_file}")
    
    # Print summary
    metric_cols = list(config.metrics.keys())
    print("\nMetric Summary:")
    print(results_df[metric_cols].mean())


if __name__ == "__main__":
    main()