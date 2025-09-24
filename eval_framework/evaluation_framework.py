"""
Modular Evaluation Framework
A flexible system for evaluating model responses against ground truth data
"""

import os
import json
import yaml
import asyncio
from typing import Dict, List, Any, Optional, Union, Callable
from pathlib import Path
import pandas as pd
import mlflow
import mlflow.metrics
from datetime import datetime
from abc import ABC, abstractmethod
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricDefinition:
    """Represents a single evaluation metric"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.description = config.get('description', '')
        self.prompt_template = config['prompt_template']
        self.score_range = config.get('score_range', [0, 1])
        self.threshold = config.get('threshold', 0.5)
        self.output_format = config.get('output_format', 'json')
        self.required_columns = self._extract_required_columns()
        
    def _extract_required_columns(self) -> List[str]:
        """Extract column names referenced in the prompt template"""
        import re
        # Find all {column_name} patterns
        pattern = r'\{(\w+)\}'
        columns = re.findall(pattern, self.prompt_template)
        return list(set(columns))
        
    def format_prompt(self, **kwargs) -> str:
        """Format the evaluation prompt with provided data"""
        return self.prompt_template.format(**kwargs)
        
    @property
    def score_key(self) -> str:
        """The key name for the score in the output"""
        return f"{self.name}_score"


class DataLoader:
    """Handles loading and preprocessing of ground truth data"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ground_truth_dir = Path(config['data']['ground_truth_dir'])
        self.column_mappings = config['data']['columns']
        self.specific_files = config['data'].get('files', [])
        
    def load_all_data(self) -> pd.DataFrame:
        """Load all CSV files from the ground truth directory"""
        all_data = []
        
        # Get list of files to process
        if self.specific_files:
            csv_files = [self.ground_truth_dir / f for f in self.specific_files]
        else:
            csv_files = list(self.ground_truth_dir.glob('*.csv'))
            
        if not csv_files:
            raise ValueError(f"No CSV files found in {self.ground_truth_dir}")
            
        logger.info(f"Found {len(csv_files)} CSV files to process")
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                df['source_file'] = csv_file.name
                all_data.append(df)
                logger.info(f"Loaded {len(df)} rows from {csv_file.name}")
            except Exception as e:
                logger.error(f"Error loading {csv_file}: {e}")
                
        if not all_data:
            raise ValueError("No data loaded from CSV files")
            
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Validate required columns exist
        self._validate_columns(combined_df)
        
        return combined_df
        
    def _validate_columns(self, df: pd.DataFrame):
        """Ensure required columns exist in the dataframe"""
        required_cols = [self.column_mappings['prompt'], self.column_mappings['response']]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")


class BaseEvaluator(ABC):
    """Base class for different types of evaluators"""
    
    @abstractmethod
    async def evaluate(self, prompt: str, response: str, **kwargs) -> Dict[str, Any]:
        """Evaluate a single prompt-response pair"""
        pass


class LLMEvaluator(BaseEvaluator):
    """Evaluator using LLM as a judge"""
    
    def __init__(self, model_name: str, metric: MetricDefinition):
        self.model_name = model_name
        self.metric = metric
        self._model = None
        
    def _get_model(self):
        """Lazy load the model"""
        if self._model is None:
            from langchain_openai import ChatOpenAI
            self._model = ChatOpenAI(
                model_name=self.model_name,
                temperature=0,
                model_kwargs={"response_format": {"type": "json_object"}}
            )
        return self._model
        
    async def evaluate(self, prompt: str, response: str, **kwargs) -> Dict[str, Any]:
        """Evaluate using LLM judge"""
        eval_prompt = self.metric.format_prompt(
            prompt=prompt,
            response=response,
            **kwargs
        )
        
        try:
            model = self._get_model()
            llm_response = await model.ainvoke(eval_prompt)
            
            if self.metric.output_format == 'json':
                result = json.loads(llm_response.content)
            else:
                result = {'content': llm_response.content}
                
            return result
            
        except Exception as e:
            logger.error(f"Error in LLM evaluation: {e}")
            return {
                self.metric.score_key: self.metric.score_range[0],
                'error': str(e)
            }


class EnsembleEvaluator(BaseEvaluator):
    """Evaluator that combines multiple judge models"""
    
    def __init__(self, evaluators: List[BaseEvaluator], metric: MetricDefinition):
        self.evaluators = evaluators
        self.metric = metric
        
    async def evaluate(self, prompt: str, response: str, **kwargs) -> Dict[str, Any]:
        """Evaluate using multiple judges and aggregate results"""
        # Run all evaluators in parallel
        tasks = [
            evaluator.evaluate(prompt, response, **kwargs)
            for evaluator in self.evaluators
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Extract scores and handle errors
        scores = []
        all_results = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Evaluator {i} failed: {result}")
                scores.append(self.metric.score_range[0])
                all_results.append({'error': str(result)})
            else:
                score = result.get(self.metric.score_key, self.metric.score_range[0])
                scores.append(score)
                all_results.append(result)
                
        # Aggregate scores (majority vote for discrete, mean for continuous)
        if len(set(scores)) <= 5:  # Likely discrete scores
            from collections import Counter
            score_counts = Counter(scores)
            final_score = score_counts.most_common(1)[0][0]
        else:
            final_score = sum(scores) / len(scores)
            
        return {
            self.metric.score_key: final_score,
            'individual_scores': scores,
            'individual_results': all_results,
            'aggregation_method': 'majority_vote' if len(set(scores)) <= 5 else 'mean'
        }


class EvaluationPipeline:
    """Main pipeline for running evaluations"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.data_loader = DataLoader(self.config)
        self.metrics = self._load_metrics()
        self.evaluators = self._create_evaluators()
        self.results_dir = Path(self.config['output']['results_dir'])
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
    def _load_metrics(self) -> List[MetricDefinition]:
        """Load metric definitions from config"""
        metrics = []
        for metric_config in self.config['metrics']:
            metric = MetricDefinition(
                name=metric_config['name'],
                config=metric_config
            )
            metrics.append(metric)
        return metrics
        
    def _create_evaluators(self) -> Dict[str, BaseEvaluator]:
        """Create evaluators for each metric"""
        evaluators = {}
        judge_models = self.config['models']['judge_models']
        
        for metric in self.metrics:
            if len(judge_models) == 1:
                # Single judge
                evaluator = LLMEvaluator(judge_models[0], metric)
            else:
                # Ensemble of judges
                llm_evaluators = [
                    LLMEvaluator(model, metric) for model in judge_models
                ]
                evaluator = EnsembleEvaluator(llm_evaluators, metric)
                
            evaluators[metric.name] = evaluator
            
        return evaluators
        
    async def run_evaluation(self) -> pd.DataFrame:
        """Run the complete evaluation pipeline"""
        # Load data
        logger.info("Loading ground truth data...")
        df = self.data_loader.load_all_data()
        
        # Initialize MLflow if configured
        if self.config['mlflow']['enable_logging']:
            self._setup_mlflow()
            
        # Run evaluations
        logger.info(f"Running evaluations on {len(df)} samples...")
        results = await self._evaluate_all(df)
        
        # Save results
        self._save_results(results)
        
        # Generate summary
        if self.config['output']['generate_summary']:
            self._generate_summary(results)
            
        return results
        
    async def _evaluate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Evaluate all samples in the dataframe"""
        results_df = df.copy()
        
        # Get column mappings
        prompt_col = self.config['data']['columns']['prompt']
        response_col = self.config['data']['columns']['response']
        
        for metric_name, evaluator in self.evaluators.items():
            logger.info(f"Evaluating metric: {metric_name}")
            
            scores = []
            details = []
            
            for idx, row in df.iterrows():
                # Prepare kwargs for evaluation
                eval_kwargs = {
                    col: row[col] for col in df.columns 
                    if col in evaluator.metric.required_columns
                }
                
                # Run evaluation
                result = await evaluator.evaluate(
                    prompt=row[prompt_col],
                    response=row[response_col],
                    **eval_kwargs
                )
                
                # Extract score and details
                score_key = evaluator.metric.score_key
                score = result.get(score_key, evaluator.metric.score_range[0])
                
                scores.append(score)
                details.append(result)
                
            # Add to results dataframe
            results_df[metric_name] = scores
            results_df[f"{metric_name}_details"] = details
            
            # Add pass/fail status
            threshold = evaluator.metric.threshold
            results_df[f"{metric_name}_status"] = [
                "✅" if s >= threshold else "❌" for s in scores
            ]
            
        return results_df
        
    def _setup_mlflow(self):
        """Configure MLflow tracking"""
        mlflow.set_experiment(self.config['experiment']['name'])
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{self.config['experiment']['run_name_prefix']}_{timestamp}"
        
        mlflow.start_run(run_name=run_name)
        
        # Log configuration
        mlflow.log_dict(self.config, "config.yaml")
        
        # Log metrics configuration
        for metric in self.metrics:
            mlflow.log_param(f"metric_{metric.name}_threshold", metric.threshold)
            
    def _save_results(self, results_df: pd.DataFrame):
        """Save evaluation results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save full results
        results_path = self.results_dir / f"evaluation_results_{timestamp}.csv"
        results_df.to_csv(results_path, index=False)
        logger.info(f"Saved results to {results_path}")
        
        # Save to MLflow
        if self.config['mlflow']['enable_logging']:
            mlflow.log_artifact(str(results_path))
            
            # Log aggregate metrics
            for metric in self.metrics:
                scores = results_df[metric.name]
                mlflow.log_metric(f"{metric.name}_mean", scores.mean())
                mlflow.log_metric(f"{metric.name}_min", scores.min())
                mlflow.log_metric(f"{metric.name}_max", scores.max())
                
    def _generate_summary(self, results_df: pd.DataFrame):
        """Generate and save summary statistics"""
        summary = {}
        
        for metric in self.metrics:
            scores = results_df[metric.name]
            summary[metric.name] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'min': scores.min(),
                'max': scores.max(),
                'pass_rate': (scores >= metric.threshold).mean()
            }
            
        # Save summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = self.results_dir / f"evaluation_summary_{timestamp}.json"
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Saved summary to {summary_path}")
        
        # Print summary
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        
        for metric_name, stats in summary.items():
            print(f"\n{metric_name}:")
            for stat_name, value in stats.items():
                print(f"  {stat_name}: {value:.3f}")


# Convenience function to run evaluation
def run_evaluation(config_path: str):
    """Run evaluation pipeline with a config file"""
    pipeline = EvaluationPipeline(config_path)
    results = asyncio.run(pipeline.run_evaluation())
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run evaluation pipeline")
    parser.add_argument("config", help="Path to configuration YAML file")
    args = parser.parse_args()
    
    results = run_evaluation(args.config)