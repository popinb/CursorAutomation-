"""
LLM as a Judge Application for OpenAI Evals Framework

This application provides a flexible framework to convert Custom GPT evaluations
into automated LLM-based judgments using the OpenAI Evals framework.
"""

import json
import os
import re
import yaml
from typing import Dict, Any, List, Optional, Union, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import openai
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Enumeration of supported metric types."""
    BINARY = "binary"  # True/False
    SCALE = "scale"  # 1-5 rating
    CATEGORICAL = "categorical"  # predefined categories
    NUMERIC = "numeric"  # numerical score
    TEXT = "text"  # text-based evaluation


@dataclass
class EvaluationMetric:
    """Configuration for a single evaluation metric."""
    name: str
    type: MetricType
    description: str
    scoring_criteria: Dict[str, Any]
    weight: float = 1.0
    required: bool = True


@dataclass
class JudgeConfig:
    """Configuration for the LLM Judge."""
    name: str
    description: str
    metrics: List[EvaluationMetric]
    knowledge_sources: List[str]
    evaluation_prompt_template: str
    model: str = "gpt-4"
    temperature: float = 0.2
    max_tokens: int = 2000


class LLMJudgeApp:
    """
    Main application class for LLM-as-a-Judge evaluation system.
    
    This class provides a flexible framework to create automated evaluations
    based on custom GPT logic and integrate with OpenAI Evals.
    """
    
    def __init__(self, config_path: Optional[str] = None, config: Optional[JudgeConfig] = None):
        """
        Initialize the LLM Judge application.
        
        Args:
            config_path: Path to YAML configuration file
            config: Direct configuration object
        """
        if config_path:
            self.config = self._load_config(config_path)
        elif config:
            self.config = config
        else:
            raise ValueError("Either config_path or config must be provided")
        
        # Initialize OpenAI client
        self.client = openai.OpenAI()
        
        # Load knowledge sources
        self.knowledge_base = self._load_knowledge_sources()
        
        # Initialize evaluation history
        self.evaluation_history = []
    
    def _load_config(self, config_path: str) -> JudgeConfig:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Parse metrics
            metrics = []
            for metric_data in config_data.get('metrics', []):
                metric = EvaluationMetric(
                    name=metric_data['name'],
                    type=MetricType(metric_data['type']),
                    description=metric_data['description'],
                    scoring_criteria=metric_data['scoring_criteria'],
                    weight=metric_data.get('weight', 1.0),
                    required=metric_data.get('required', True)
                )
                metrics.append(metric)
            
            return JudgeConfig(
                name=config_data['name'],
                description=config_data['description'],
                metrics=metrics,
                knowledge_sources=config_data.get('knowledge_sources', []),
                evaluation_prompt_template=config_data['evaluation_prompt_template'],
                model=config_data.get('model', 'gpt-4'),
                temperature=config_data.get('temperature', 0.2),
                max_tokens=config_data.get('max_tokens', 2000)
            )
        except Exception as e:
            raise ValueError(f"Error loading configuration: {e}")
    
    def _load_knowledge_sources(self) -> Dict[str, Any]:
        """Load knowledge sources specified in configuration."""
        knowledge_base = {}
        
        for source in self.config.knowledge_sources:
            try:
                if source.endswith('.json'):
                    with open(source, 'r') as f:
                        knowledge_base[source] = json.load(f)
                elif source.endswith('.txt'):
                    with open(source, 'r') as f:
                        knowledge_base[source] = f.read()
                elif source.endswith('.yaml') or source.endswith('.yml'):
                    with open(source, 'r') as f:
                        knowledge_base[source] = yaml.safe_load(f)
                else:
                    logger.warning(f"Unsupported knowledge source format: {source}")
            except FileNotFoundError:
                logger.error(f"Knowledge source not found: {source}")
            except Exception as e:
                logger.error(f"Error loading knowledge source {source}: {e}")
        
        return knowledge_base
    
    def evaluate(
        self,
        candidate_answer: str,
        question: str = "",
        context: Optional[Dict[str, Any]] = None,
        reference_answer: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a candidate answer using the configured LLM judge.
        
        Args:
            candidate_answer: The response to evaluate
            question: The original question or prompt
            context: Additional context for evaluation
            reference_answer: Optional reference/ground truth answer
            
        Returns:
            Dictionary containing evaluation results
        """
        # Prepare evaluation prompt
        evaluation_prompt = self._prepare_evaluation_prompt(
            candidate_answer, question, context, reference_answer
        )
        
        # Get LLM judgment
        llm_judgment = self._get_llm_judgment(evaluation_prompt)
        
        # Parse and structure the results
        evaluation_results = self._parse_llm_judgment(llm_judgment)
        
        # Add metadata
        evaluation_results['metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'candidate_answer_length': len(candidate_answer),
            'model': self.config.model,
            'context_provided': context is not None,
            'reference_provided': reference_answer is not None
        }
        
        # Store in history
        self.evaluation_history.append(evaluation_results)
        
        return evaluation_results
    
    def _prepare_evaluation_prompt(
        self,
        candidate_answer: str,
        question: str,
        context: Optional[Dict[str, Any]],
        reference_answer: Optional[str]
    ) -> str:
        """Prepare the evaluation prompt for the LLM judge."""
        
        # Build knowledge context
        knowledge_context = ""
        for source_name, source_content in self.knowledge_base.items():
            knowledge_context += f"\n=== {source_name} ===\n"
            if isinstance(source_content, dict):
                knowledge_context += json.dumps(source_content, indent=2)
            else:
                knowledge_context += str(source_content)
            knowledge_context += "\n"
        
        # Build metrics description
        metrics_description = ""
        for metric in self.config.metrics:
            metrics_description += f"\n{metric.name} ({metric.type.value}): {metric.description}\n"
            metrics_description += f"Scoring Criteria: {json.dumps(metric.scoring_criteria, indent=2)}\n"
        
        # Format the main evaluation prompt
        prompt = self.config.evaluation_prompt_template.format(
            judge_name=self.config.name,
            judge_description=self.config.description,
            knowledge_context=knowledge_context,
            metrics_description=metrics_description,
            question=question,
            candidate_answer=candidate_answer,
            reference_answer=reference_answer or "Not provided",
            context=json.dumps(context or {}, indent=2)
        )
        
        return prompt
    
    def _get_llm_judgment(self, prompt: str) -> str:
        """Get judgment from the LLM."""
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert evaluator. Provide detailed, objective evaluations based on the given criteria."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error getting LLM judgment: {e}")
            raise
    
    def _parse_llm_judgment(self, judgment: str) -> Dict[str, Any]:
        """Parse the LLM judgment into structured results."""
        results = {
            'overall_score': 0.0,
            'metric_scores': {},
            'detailed_evaluation': judgment,
            'justifications': {}
        }
        
        # Extract metric scores using patterns
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for metric in self.config.metrics:
            # Try to extract score for this metric
            score, justification = self._extract_metric_score(judgment, metric)
            
            # Normalize score to 0-1 range
            normalized_score = self._normalize_score(score, metric)
            
            results['metric_scores'][metric.name] = {
                'raw_score': score,
                'normalized_score': normalized_score,
                'weight': metric.weight,
                'type': metric.type.value
            }
            results['justifications'][metric.name] = justification
            
            # Calculate weighted contribution
            total_weighted_score += normalized_score * metric.weight
            total_weight += metric.weight
        
        # Calculate overall score
        if total_weight > 0:
            results['overall_score'] = total_weighted_score / total_weight
        
        return results
    
    def _extract_metric_score(self, judgment: str, metric: EvaluationMetric) -> tuple:
        """Extract score and justification for a specific metric."""
        # Look for metric name in judgment
        pattern = rf"{metric.name}.*?:.*?([^\n]+)"
        match = re.search(pattern, judgment, re.IGNORECASE | re.DOTALL)
        
        if match:
            score_text = match.group(1).strip()
            
            # Extract score based on metric type
            if metric.type == MetricType.BINARY:
                score = self._extract_binary_score(score_text)
            elif metric.type == MetricType.SCALE:
                score = self._extract_scale_score(score_text)
            elif metric.type == MetricType.CATEGORICAL:
                score = self._extract_categorical_score(score_text, metric)
            elif metric.type == MetricType.NUMERIC:
                score = self._extract_numeric_score(score_text)
            else:
                score = score_text
            
            # Extract justification (look for next few lines)
            justification_pattern = rf"{metric.name}.*?:.*?([^\n]+(?:\n[^\n]*){0,3})"
            justification_match = re.search(justification_pattern, judgment, re.IGNORECASE | re.DOTALL)
            justification = justification_match.group(1).strip() if justification_match else score_text
            
            return score, justification
        
        return None, f"No evaluation found for {metric.name}"
    
    def _extract_binary_score(self, text: str) -> bool:
        """Extract binary (True/False) score."""
        text_lower = text.lower()
        if any(word in text_lower for word in ['true', 'yes', 'correct', 'accurate', 'present']):
            return True
        elif any(word in text_lower for word in ['false', 'no', 'incorrect', 'inaccurate', 'absent']):
            return False
        return False  # Default to False if unclear
    
    def _extract_scale_score(self, text: str) -> int:
        """Extract scale score (1-5)."""
        numbers = re.findall(r'\b([1-5])\b', text)
        if numbers:
            return int(numbers[0])
        return 1  # Default to 1 if no score found
    
    def _extract_categorical_score(self, text: str, metric: EvaluationMetric) -> str:
        """Extract categorical score."""
        categories = metric.scoring_criteria.get('categories', [])
        text_lower = text.lower()
        
        for category in categories:
            if category.lower() in text_lower:
                return category
        
        return categories[0] if categories else "Unknown"
    
    def _extract_numeric_score(self, text: str) -> float:
        """Extract numeric score."""
        numbers = re.findall(r'\b(\d+\.?\d*)\b', text)
        if numbers:
            return float(numbers[0])
        return 0.0
    
    def _normalize_score(self, score: Any, metric: EvaluationMetric) -> float:
        """Normalize score to 0-1 range."""
        if metric.type == MetricType.BINARY:
            return 1.0 if score else 0.0
        elif metric.type == MetricType.SCALE:
            scale_max = metric.scoring_criteria.get('max_scale', 5)
            scale_min = metric.scoring_criteria.get('min_scale', 1)
            return (score - scale_min) / (scale_max - scale_min)
        elif metric.type == MetricType.CATEGORICAL:
            categories = metric.scoring_criteria.get('categories', [])
            if score in categories:
                return categories.index(score) / (len(categories) - 1) if len(categories) > 1 else 1.0
            return 0.0
        elif metric.type == MetricType.NUMERIC:
            numeric_max = metric.scoring_criteria.get('max_value', 100)
            numeric_min = metric.scoring_criteria.get('min_value', 0)
            return min(max((score - numeric_min) / (numeric_max - numeric_min), 0.0), 1.0)
        else:
            return 1.0  # Default for text metrics
    
    def batch_evaluate(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate multiple samples in batch.
        
        Args:
            samples: List of samples to evaluate
            
        Returns:
            Batch evaluation results
        """
        results = []
        failed_evaluations = 0
        
        for i, sample in enumerate(samples):
            try:
                result = self.evaluate(
                    candidate_answer=sample.get('candidate_answer', ''),
                    question=sample.get('question', ''),
                    context=sample.get('context'),
                    reference_answer=sample.get('reference_answer')
                )
                results.append(result)
                logger.info(f"Evaluated sample {i+1}/{len(samples)}")
            except Exception as e:
                logger.error(f"Failed to evaluate sample {i+1}: {e}")
                failed_evaluations += 1
        
        # Calculate batch statistics
        if results:
            avg_score = sum(r['overall_score'] for r in results) / len(results)
            metric_averages = {}
            
            for metric in self.config.metrics:
                metric_scores = [r['metric_scores'][metric.name]['normalized_score'] 
                               for r in results if metric.name in r['metric_scores']]
                if metric_scores:
                    metric_averages[metric.name] = sum(metric_scores) / len(metric_scores)
        else:
            avg_score = 0.0
            metric_averages = {}
        
        return {
            'results': results,
            'batch_statistics': {
                'total_samples': len(samples),
                'successful_evaluations': len(results),
                'failed_evaluations': failed_evaluations,
                'average_score': avg_score,
                'metric_averages': metric_averages
            }
        }
    
    def export_results(self, filepath: str, format: str = 'json') -> None:
        """Export evaluation results to file."""
        if format.lower() == 'json':
            with open(filepath, 'w') as f:
                json.dump(self.evaluation_history, f, indent=2)
        elif format.lower() == 'yaml':
            with open(filepath, 'w') as f:
                yaml.dump(self.evaluation_history, f, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Results exported to {filepath}")
    
    def generate_report(self) -> str:
        """Generate a summary report of evaluations."""
        if not self.evaluation_history:
            return "No evaluations performed yet."
        
        total_evaluations = len(self.evaluation_history)
        avg_score = sum(e['overall_score'] for e in self.evaluation_history) / total_evaluations
        
        report = f"""
# LLM Judge Evaluation Report

**Judge:** {self.config.name}
**Total Evaluations:** {total_evaluations}
**Average Score:** {avg_score:.3f}

## Metric Performance

"""
        
        # Calculate average scores per metric
        metric_stats = {}
        for metric in self.config.metrics:
            scores = []
            for evaluation in self.evaluation_history:
                if metric.name in evaluation['metric_scores']:
                    scores.append(evaluation['metric_scores'][metric.name]['normalized_score'])
            
            if scores:
                metric_stats[metric.name] = {
                    'average': sum(scores) / len(scores),
                    'count': len(scores)
                }
        
        for metric_name, stats in metric_stats.items():
            report += f"- **{metric_name}:** {stats['average']:.3f} (n={stats['count']})\n"
        
        return report


# OpenAI Evals Framework Compatible Interface
class LLMJudgeOAIEval:
    """OpenAI Evals framework compatible wrapper for LLM Judge."""
    
    def __init__(self, config_path: str):
        """Initialize with configuration path."""
        self.judge_app = LLMJudgeApp(config_path=config_path)
    
    def eval_sample(self, sample: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Evaluate a single sample."""
        input_data = sample.get('input', {})
        
        result = self.judge_app.evaluate(
            candidate_answer=input_data.get('candidate_answer', ''),
            question=input_data.get('question', ''),
            context=input_data.get('context'),
            reference_answer=input_data.get('reference_answer')
        )
        
        return {
            'score': result['overall_score'],
            'evaluation_details': result
        }
    
    def run_eval(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run evaluation on multiple samples."""
        batch_results = self.judge_app.batch_evaluate([
            {
                'candidate_answer': s.get('input', {}).get('candidate_answer', ''),
                'question': s.get('input', {}).get('question', ''),
                'context': s.get('input', {}).get('context'),
                'reference_answer': s.get('input', {}).get('reference_answer')
            }
            for s in samples
        ])
        
        return batch_results


if __name__ == "__main__":
    # Example usage
    print("LLM Judge Application initialized. Use with configuration file.")
    print("Example: judge = LLMJudgeApp(config_path='config.yaml')")