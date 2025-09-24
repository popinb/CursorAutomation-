"""
Enhanced Universal Evaluation Template System
============================================

A comprehensive evaluation system with MLflow integration, advanced metrics,
and easy-to-add components.
"""

import argparse
import asyncio
import json
import os
import sys
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import yaml

# Core dependencies (with fallbacks)
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("⚠️  pandas not available, using basic data handling")

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    print("⚠️  httpx not available, using basic HTTP")

try:
    from pydantic import BaseModel
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    print("⚠️  pydantic not available, using basic validation")

try:
    from langchain_openai import ChatOpenAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("⚠️  langchain not available, using basic LLM interface")

try:
    import mlflow
    import mlflow.metrics
    import mlflow.metrics.genai
    from mlflow.genai.scorers import scorer
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("⚠️  mlflow not available, using basic logging")

import collections


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BaseModel:
    """Fallback BaseModel if Pydantic is not available."""
    def __init__(self, **data):
        for key, value in data.items():
            setattr(self, key, value)
    
    def model_dump(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


class EvaluationConfig:
    """Enhanced configuration class with validation."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self._load_config()
        self._validate_config()
        self._set_environment()
    
    def _load_config(self):
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")
    
    def _validate_config(self):
        """Validate configuration structure."""
        required_sections = ['dataset', 'models', 'api_keys', 'evaluation']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required section: {section}")
        
        # Dataset validation
        dataset = self.config['dataset']
        required_dataset_fields = ['path', 'prompt_column', 'response_column']
        for field in required_dataset_fields:
            if field not in dataset:
                raise ValueError(f"Missing required dataset field: {field}")
        
        # Models validation
        models = self.config['models']
        if 'judge_models' not in models or not models['judge_models']:
            raise ValueError("judge_models must be specified and non-empty")
        
        # API keys validation
        api_keys = self.config['api_keys']
        if 'openai_api_key' not in api_keys:
            raise ValueError("openai_api_key must be specified")
    
    def _set_environment(self):
        """Set environment variables."""
        os.environ["OPENAI_API_KEY"] = self.config['api_keys']['openai_api_key']
        if 'openai_base_url' in self.config['api_keys']:
            os.environ["OPENAI_API_BASE"] = self.config['api_keys']['openai_base_url']
    
    @property
    def dataset_path(self):
        return self.config['dataset']['path']
    
    @property
    def prompt_column(self):
        return self.config['dataset']['prompt_column']
    
    @property
    def response_column(self):
        return self.config['dataset']['response_column']
    
    @property
    def ground_truth_column(self):
        return self.config['dataset'].get('ground_truth_column')
    
    @property
    def user_features_column(self):
        return self.config['dataset'].get('user_features_column')
    
    @property
    def response_model(self):
        return self.config['models']['response_model']
    
    @property
    def judge_models(self):
        return self.config['models']['judge_models']
    
    @property
    def openai_api_key(self):
        return self.config['api_keys']['openai_api_key']
    
    @property
    def openai_base_url(self):
        return self.config['api_keys'].get('openai_base_url', 'https://api.openai.com/v1')
    
    @property
    def metrics(self):
        return self.config['evaluation']['metrics']
    
    @property
    def experiment_name(self):
        return self.config['evaluation']['experiment_name']
    
    @property
    def run_name(self):
        return self.config['evaluation'].get('run_name', f"run_{int(time.time())}")


class DataHandler:
    """Enhanced data handling with fallbacks."""
    
    @staticmethod
    def load_dataset(file_path: str) -> Union[pd.DataFrame, List[Dict]]:
        """Load dataset with fallback to basic CSV parsing."""
        if PANDAS_AVAILABLE:
            return pd.read_csv(file_path)
        else:
            return DataHandler._load_csv_basic(file_path)
    
    @staticmethod
    def _load_csv_basic(file_path: str) -> List[Dict]:
        """Basic CSV loading without pandas."""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if not lines:
            return data
        
        # Parse header
        header = [col.strip() for col in lines[0].strip().split(',')]
        
        # Parse data rows
        for line in lines[1:]:
            if line.strip():
                values = [val.strip().strip('"') for val in line.strip().split(',')]
                if len(values) == len(header):
                    data.append(dict(zip(header, values)))
        
        return data
    
    @staticmethod
    def save_results(data: Union[pd.DataFrame, List[Dict]], file_path: str):
        """Save results with fallback to basic CSV writing."""
        if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
            data.to_csv(file_path, index=False)
        else:
            DataHandler._save_csv_basic(data, file_path)
    
    @staticmethod
    def _save_csv_basic(data: List[Dict], file_path: str):
        """Basic CSV writing without pandas."""
        if not data:
            return
        
        with open(file_path, 'w', encoding='utf-8') as f:
            # Write header
            header = list(data[0].keys())
            f.write(','.join(f'"{col}"' for col in header) + '\n')
            
            # Write data rows
            for row in data:
                values = [str(row.get(col, '')) for col in header]
                f.write(','.join(f'"{val}"' for val in values) + '\n')


class LLMInterface:
    """Enhanced LLM interface with fallbacks."""
    
    def __init__(self, model_name: str, api_key: str, base_url: str = None):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url or "https://api.openai.com/v1"
        
        if LANGCHAIN_AVAILABLE:
            self.client = ChatOpenAI(
                model_name=model_name,
                default_headers={"apikey": api_key},
                base_url=self.base_url,
                temperature=0,
                model_kwargs={"response_format": {"type": "json_object"}},
            )
        else:
            self.client = None
            logger.warning("LangChain not available, using basic HTTP interface")
    
    def invoke(self, prompt: str) -> str:
        """Invoke LLM with fallback to basic HTTP."""
        if self.client:
            try:
                response = self.client.invoke(prompt)
                return response.content
            except Exception as e:
                logger.error(f"LangChain invocation failed: {e}")
                return self._basic_http_call(prompt)
        else:
            return self._basic_http_call(prompt)
    
    def _basic_http_call(self, prompt: str) -> str:
        """Basic HTTP call as fallback."""
        if not HTTPX_AVAILABLE:
            # Return mock response for testing
            return '{"score": 3, "explanation": "Mock response - HTTP client not available"}'
        
        try:
            import httpx
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "response_format": {"type": "json_object"}
                },
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"HTTP call failed: {e}")
            return '{"score": 3, "explanation": "Error in HTTP call"}'
    
    def with_structured_output(self, schema):
        """Structured output wrapper."""
        if self.client and hasattr(self.client, 'with_structured_output'):
            return self.client.with_structured_output(schema)
        else:
            return self  # Return self as fallback


class MLflowLogger:
    """Enhanced MLflow logging with fallbacks."""
    
    def __init__(self, experiment_name: str, run_name: str = None):
        self.experiment_name = experiment_name
        self.run_name = run_name or f"run_{int(time.time())}"
        self.mlflow_available = MLFLOW_AVAILABLE
        
        if self.mlflow_available:
            try:
                mlflow.set_experiment(experiment_name)
                self.run = mlflow.start_run(run_name=self.run_name)
            except Exception as e:
                logger.warning(f"MLflow initialization failed: {e}")
                self.mlflow_available = False
        else:
            logger.info("MLflow not available, using basic logging")
    
    def log_metric(self, key: str, value: float):
        """Log metric with fallback."""
        if self.mlflow_available:
            try:
                mlflow.log_metric(key, value)
            except Exception as e:
                logger.error(f"MLflow metric logging failed: {e}")
        
        # Always log to console
        logger.info(f"Metric: {key} = {value}")
    
    def log_metrics(self, metrics: Dict[str, float]):
        """Log multiple metrics."""
        for key, value in metrics.items():
            self.log_metric(key, value)
    
    def log_param(self, key: str, value: str):
        """Log parameter with fallback."""
        if self.mlflow_available:
            try:
                mlflow.log_param(key, value)
            except Exception as e:
                logger.error(f"MLflow param logging failed: {e}")
        
        logger.info(f"Parameter: {key} = {value}")
    
    def log_artifact(self, file_path: str):
        """Log artifact with fallback."""
        if self.mlflow_available:
            try:
                mlflow.log_artifact(file_path)
            except Exception as e:
                logger.error(f"MLflow artifact logging failed: {e}")
        
        logger.info(f"Artifact logged: {file_path}")
    
    def close(self):
        """Close MLflow run."""
        if self.mlflow_available and hasattr(self, 'run'):
            try:
                mlflow.end_run()
            except Exception as e:
                logger.error(f"MLflow run close failed: {e}")


class EnhancedUniversalEvaluator:
    """Enhanced universal evaluator with all components."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.evaluators = {}
        self.scorers = []
        self.mlflow_logger = MLflowLogger(config.experiment_name, config.run_name)
        self._setup_evaluators()
        self._setup_scorers()
        self._log_configuration()
    
    def _setup_evaluators(self):
        """Setup evaluator functions for each metric."""
        for metric_name, metric_config in self.config.metrics.items():
            prompt_template = metric_config['prompt_template']
            threshold = metric_config.get('threshold', 0.5)
            output_schema = self._get_output_schema(metric_config.get('output_schema'))
            
            evaluator_func = self._create_ensemble_evaluator(
                metric_name, prompt_template, threshold, output_schema
            )
            self.evaluators[metric_name] = evaluator_func
    
    def _get_output_schema(self, schema_config: Optional[Dict]) -> Optional[BaseModel]:
        """Create output schema with fallback."""
        if not schema_config or not PYDANTIC_AVAILABLE:
            return None
        
        try:
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
        except Exception as e:
            logger.warning(f"Schema creation failed: {e}")
            return None
    
    def _create_ensemble_evaluator(self, metric_name: str, prompt_template: str, 
                                 threshold: float, output_schema: Optional[BaseModel]):
        """Create ensemble evaluator."""
        models = [self._create_llm_model(model_name) for model_name in self.config.judge_models]
        
        def evaluator(eval_data, builtin_metrics=None):
            results = []
            details = []
            
            # Handle both DataFrame and list of dicts
            if PANDAS_AVAILABLE and hasattr(eval_data, 'iterrows'):
                data_iter = eval_data.iterrows()
            else:
                data_iter = enumerate(eval_data)
            
            for idx, row in data_iter:
                if PANDAS_AVAILABLE and hasattr(eval_data, 'iterrows'):
                    row_data = row[1]  # pandas Series
                else:
                    row_data = row  # dict
                
                eval_prompt = self._format_prompt(prompt_template, row_data)
                
                model_outputs = []
                scores = []
                
                for model in models:
                    try:
                        if output_schema:
                            structured_model = model.with_structured_output(output_schema)
                            llm_response = structured_model.invoke(eval_prompt)
                            
                            if hasattr(llm_response, 'model_dump'):
                                result_json = llm_response.model_dump()
                            elif hasattr(llm_response, '__dict__'):
                                result_json = llm_response.__dict__
                            else:
                                result_json = json.loads(str(llm_response))
                        else:
                            response_text = model.invoke(eval_prompt)
                            result_json = json.loads(response_text)
                        
                        model_outputs.append((getattr(model, "model_name", "model"), result_json))
                        
                        score_key = f"{metric_name.lower()}_score"
                        if score_key not in result_json:
                            # Try to find the score key in nested structure
                            for key, value in result_json.items():
                                if isinstance(value, dict) and score_key in value:
                                    result_json = value
                                    break
                        
                        scores.append(float(result_json.get(score_key, 0)))
                        
                    except Exception as exc:
                        logger.error(f"Model evaluation failed: {exc}")
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
                f"{metric_name.lower()}/mean": mean_score,
                f"{metric_name.lower()}/scores": results,
                f"{metric_name.lower()}/details": details,
            }
        
        return evaluator
    
    def _create_llm_model(self, model_name: str) -> LLMInterface:
        """Create LLM model interface."""
        return LLMInterface(
            model_name=model_name,
            api_key=self.config.openai_api_key,
            base_url=self.config.openai_base_url
        )
    
    def _format_prompt(self, template: str, row_data) -> str:
        """Format prompt template with row data."""
        if PANDAS_AVAILABLE and hasattr(row_data, 'to_dict'):
            format_dict = row_data.to_dict()
        else:
            format_dict = dict(row_data)
        
        # Add common aliases
        format_dict['prompt'] = format_dict.get(self.config.prompt_column, '')
        format_dict['response'] = format_dict.get(self.config.response_column, '')
        format_dict['user_personalization_features'] = format_dict.get(self.config.user_features_column, '')
        format_dict['ground_truth'] = format_dict.get(self.config.ground_truth_column, '')
        
        return template.format(**format_dict)
    
    def _setup_scorers(self):
        """Setup MLflow scorers with fallback."""
        if not MLFLOW_AVAILABLE:
            return
        
        try:
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
        except Exception as e:
            logger.warning(f"Scorer setup failed: {e}")
    
    def _log_configuration(self):
        """Log configuration parameters."""
        self.mlflow_logger.log_param("dataset_path", self.config.dataset_path)
        self.mlflow_logger.log_param("prompt_column", self.config.prompt_column)
        self.mlflow_logger.log_param("response_column", self.config.response_column)
        self.mlflow_logger.log_param("judge_models", str(self.config.judge_models))
        self.mlflow_logger.log_param("metrics_count", str(len(self.config.metrics)))
    
    def evaluate_dataset(self, data) -> Union[pd.DataFrame, List[Dict]]:
        """Evaluate the entire dataset."""
        logger.info("Starting evaluation...")
        
        # Prepare data for evaluation
        if PANDAS_AVAILABLE and hasattr(data, 'copy'):
            eval_data = data.copy()
            eval_data = eval_data.rename(columns={
                self.config.prompt_column: "inputs",
                self.config.response_column: "predictions"
            })
        else:
            eval_data = data  # Use as-is for list of dicts
        
        # Run evaluations
        all_scores = {}
        
        for metric_name, evaluator_func in self.evaluators.items():
            logger.info(f"Running {metric_name} evaluation...")
            
            eval_results = evaluator_func(eval_data)
            
            individual_scores = eval_results.get(f"{metric_name.lower()}/scores", [])
            mean_score = eval_results.get(f"{metric_name.lower()}/mean", 0.0)
            details = eval_results.get(f"{metric_name.lower()}/details", [])
            
            # Add scores to data
            if PANDAS_AVAILABLE and hasattr(data, '__setitem__'):
                data[f"{metric_name}_details"] = details
                if len(individual_scores) == len(data):
                    scores = individual_scores
                else:
                    scores = [mean_score] * len(data)
                data[metric_name] = scores
            else:
                # Handle list of dicts
                for i, detail in enumerate(details):
                    if i < len(data):
                        data[i][f"{metric_name}_details"] = detail
                        if i < len(individual_scores):
                            data[i][metric_name] = individual_scores[i]
                        else:
                            data[i][metric_name] = mean_score
            
            all_scores[metric_name] = individual_scores if len(individual_scores) == len(data) else [mean_score] * len(data)
            
            # Log metrics
            self.mlflow_logger.log_metric(f"{metric_name}_mean", mean_score)
            logger.info(f"✅ {metric_name}: {mean_score:.3f}")
        
        # Add status indicators
        for metric_name, scores in all_scores.items():
            threshold = self.config.metrics[metric_name].get('threshold', 0.5)
            status_values = ["✅" if s >= threshold else "❌" for s in scores]
            
            if PANDAS_AVAILABLE and hasattr(data, '__setitem__'):
                data[f"{metric_name}_status"] = status_values
            else:
                for i, status in enumerate(status_values):
                    if i < len(data):
                        data[i][f"{metric_name}_status"] = status
        
        # Log summary metrics
        summary_metrics = {}
        for metric_name, scores in all_scores.items():
            if scores:
                summary_metrics[f"{metric_name}_mean"] = sum(scores) / len(scores)
                summary_metrics[f"{metric_name}_max"] = max(scores)
                summary_metrics[f"{metric_name}_min"] = min(scores)
        
        self.mlflow_logger.log_metrics(summary_metrics)
        
        return data
    
    def close(self):
        """Close MLflow logger."""
        self.mlflow_logger.close()


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Enhanced Universal Evaluation Template System")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--output", help="Output directory for results")
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = EvaluationConfig(args.config)
        
        # Load dataset
        logger.info(f"Loading dataset from {config.dataset_path}")
        data = DataHandler.load_dataset(config.dataset_path)
        logger.info(f"Loaded {len(data)} rows")
        
        # Run evaluation
        evaluator = EnhancedUniversalEvaluator(config)
        results = evaluator.evaluate_dataset(data)
        
        # Save results
        output_dir = Path(args.output) if args.output else Path("results")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"evaluation_results_{timestamp}.csv"
        DataHandler.save_results(results, str(output_file))
        
        logger.info(f"✅ Evaluation complete - results saved to {output_file}")
        
        # Print summary
        if PANDAS_AVAILABLE and hasattr(results, 'columns'):
            metric_cols = [col for col in results.columns if col.endswith('_score') or col in config.metrics.keys()]
            if metric_cols:
                logger.info("\nMetric Summary:")
                for col in metric_cols:
                    if col in results.columns:
                        mean_val = results[col].mean()
                        logger.info(f"{col}: {mean_val:.3f}")
        
        evaluator.close()
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()