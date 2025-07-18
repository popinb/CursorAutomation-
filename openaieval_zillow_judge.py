#!/usr/bin/env python3
"""
OpenAI Evals Framework Implementation: Zillow Home Affordability Judge

This is a completely separate implementation using the OpenAI Evals framework
to evaluate home affordability responses with the same 12-metric criteria.
"""

import json
import os
import time
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import re

# OpenAI Evals imports
try:
    import evals
    from evals.api import CompletionFn
    from evals.eval import Eval
    from evals.record import RecorderBase
    from evals.registry import Registry
    EVALS_AVAILABLE = True
except ImportError:
    EVALS_AVAILABLE = False
    print("OpenAI Evals framework not installed. Install with: pip install evals")


class ZillowAffordabilityEval(Eval):
    """
    OpenAI Evals framework implementation for Zillow Home Affordability evaluation.
    
    This evaluator uses the same 12-metric criteria as the custom implementation
    but integrates with the OpenAI Evals framework for standardized evaluation.
    """
    
    def __init__(
        self,
        completion_fns: List[CompletionFn],
        samples_jsonl: str,
        eval_registry_path: Optional[str] = None,
        seed: int = 42,
        name: str = "zillow_affordability_eval",
        registry: Optional[Registry] = None,
    ):
        """
        Initialize the Zillow Affordability Evaluator.
        
        Args:
            completion_fns: List of completion functions to evaluate
            samples_jsonl: Path to the JSONL file containing evaluation samples
            eval_registry_path: Path to the eval registry
            seed: Random seed for reproducibility
            name: Name of the evaluation
            registry: Optional registry instance
        """
        super().__init__(
            completion_fns=completion_fns,
            seed=seed,
            name=name,
            registry=registry,
        )
        
        self.samples_jsonl = samples_jsonl
        
        # Load knowledge base files
        script_dir = Path(__file__).parent
        self.golden_responses = self._load_json(script_dir / "assets" / "golden_responses.json")
        self.buyability_profiles = self._load_json(script_dir / "assets" / "buyability_profiles.json") 
        self.fair_housing_guide = self._load_json(script_dir / "assets" / "fair_housing_guide.json")
        
        # Define evaluation metrics and scoring
        self.metrics = self._define_metrics()
        
        # OpenAI Evals configuration
        self.judge_model = "gpt-4"  # Model for LLM-as-a-judge evaluation
        self.judge_params = {
            "temperature": 0,
            "max_tokens": 4000,
            "top_p": 1.0,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        }
    
    def _load_json(self, filepath: Path) -> Dict[str, Any]:
        """Load JSON data from file."""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: {filepath} not found, continuing without it")
            return {}
        except json.JSONDecodeError as e:
            print(f"Warning: Invalid JSON in {filepath}: {e}")
            return {}
    
    def _define_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Define the 12 evaluation metrics with their scoring criteria."""
        return {
            "personalization_accuracy": {
                "type": "boolean",
                "description": "Does the response use the user's specific financial data correctly?",
                "scoring": {"accurate": 10, "inaccurate": 0}
            },
            "context_personalization": {
                "type": "scale",
                "description": "How well does the response tailor advice to the user's situation?",
                "scoring": {1: 2, 2: 4, 3: 6, 4: 8, 5: 10}
            },
            "next_step_identification": {
                "type": "boolean", 
                "description": "Does the response provide clear, actionable next steps?",
                "scoring": {"present": 10, "not_present": 0}
            },
            "assumption_listing": {
                "type": "boolean",
                "description": "Are assumptions clearly stated when data is incomplete?",
                "scoring": {"true": 10, "false": 0}
            },
            "assumption_trust": {
                "type": "scale",
                "description": "How reasonable and trustworthy are the stated assumptions?",
                "scoring": {1: 2, 2: 4, 3: 6, 4: 8, 5: 10}
            },
            "calculation_accuracy": {
                "type": "boolean",
                "description": "Are all financial calculations mathematically correct?",
                "scoring": {"true": 10, "false": 0}
            },
            "faithfulness_ground_truth": {
                "type": "boolean",
                "description": "Does the response align with provided reference information?",
                "scoring": {"true": 10, "false": 0}
            },
            "overall_accuracy": {
                "type": "boolean",
                "description": "Is the response factually correct and reliable?",
                "scoring": {"true": 10, "false": 0}
            },
            "structured_presentation": {
                "type": "scale",
                "description": "How well-organized and readable is the response?",
                "scoring": {1: 2, 2: 4, 3: 6, 4: 8, 5: 10}
            },
            "coherence": {
                "type": "boolean",
                "description": "Is the response logically consistent throughout?",
                "scoring": {"true": 10, "false": 0}
            },
            "completeness": {
                "type": "scale",
                "description": "How thoroughly does the response address the question?",
                "scoring": {1: 2, 2: 4, 3: 6, 4: 8, 5: 10}
            },
            "fair_housing_classifier": {
                "type": "boolean",
                "description": "Does the response comply with Fair Housing Act requirements?",
                "scoring": {"true": 10, "false": 0}
            }
        }
    
    def eval_sample(self, sample: Dict[str, Any], *_) -> Dict[str, Any]:
        """
        Evaluate a single sample using OpenAI Evals framework.
        
        Args:
            sample: Dictionary containing question, answer, and user_profile
            
        Returns:
            Dictionary with evaluation results and metrics
        """
        question = sample.get("question", "")
        candidate_answer = sample.get("candidate_answer", "")
        user_profile = sample.get("user_profile", {})
        
        # Create evaluation prompt for LLM-as-a-judge
        evaluation_prompt = self._create_evaluation_prompt(
            question, candidate_answer, user_profile
        )
        
        # Get LLM evaluation using the first completion function
        completion_fn = self.completion_fns[0]
        
        try:
            response = completion_fn(
                prompt=[{"role": "user", "content": evaluation_prompt}],
                **self.judge_params
            )
            
            llm_evaluation = response.get_completions()[0]
            
            # Parse evaluation results
            parsed_results = self._parse_llm_evaluation(llm_evaluation)
            
            # Calculate scores
            scores = self._calculate_scores(parsed_results)
            
            return {
                "llm_evaluation": llm_evaluation,
                "parsed_results": parsed_results,
                "individual_scores": scores["individual"],
                "alpha_score": scores["alpha_total"],
                "alpha_max": scores["alpha_max"],
                "full_score": scores["full_total"],
                "full_max": scores["full_max"],
                "alpha_percentage": scores["alpha_percentage"],
                "full_percentage": scores["full_percentage"],
            }
            
        except Exception as e:
            return {
                "error": f"Evaluation failed: {str(e)}",
                "llm_evaluation": None,
                "individual_scores": {},
                "alpha_score": 0,
                "full_score": 0,
                "alpha_percentage": 0,
                "full_percentage": 0,
            }
    
    def _create_evaluation_prompt(self, question: str, candidate_answer: str, user_profile: Dict[str, Any]) -> str:
        """Create the evaluation prompt for LLM-as-a-judge."""
        
        profile_str = json.dumps(user_profile, indent=2) if user_profile else "No profile provided"
        knowledge_base = self._format_knowledge_base()
        
        prompt = f"""
You are an expert evaluator for Zillow Home Affordability responses. Evaluate the candidate answer using the following 12 metrics.

QUESTION: {question}
CANDIDATE ANSWER: {candidate_answer}
USER PROFILE: {profile_str}

KNOWLEDGE BASE:
{knowledge_base}

EVALUATION METRICS:

1. **Personalization Accuracy** (Accurate/Inaccurate):
   - Does the response use the user's specific financial data correctly?

2. **Context-based Personalization** (1-5 scale):
   - How well does the response tailor advice to the user's situation?

3. **Next Step Identification** (Present/Not Present):
   - Does the response provide clear, actionable next steps?

4. **Assumption Listing** (True/False):
   - Are assumptions clearly stated when data is incomplete?

5. **Assumption Trust** (1-5 scale):
   - How reasonable and trustworthy are the stated assumptions?

6. **Calculation Accuracy** (True/False):
   - Are all financial calculations mathematically correct?

7. **Faithfulness to Ground Truth** (True/False):
   - Does the response align with provided reference information?

8. **Overall Accuracy** (True/False):
   - Is the response factually correct and reliable?

9. **Structured Presentation** (1-5 scale):
   - How well-organized and readable is the response?

10. **Coherence** (True/False):
    - Is the response logically consistent throughout?

11. **Completeness** (1-5 scale):
    - How thoroughly does the response address the question?

12. **Fair Housing Classifier** (True/False):
    - Does the response comply with Fair Housing Act requirements?

INSTRUCTIONS:
- Provide a score for each metric
- Include a 150+ word justification for each metric
- Use the exact metric names in your response
- Format your response as a structured evaluation

OUTPUT FORMAT:
| Metric | Score | Justification |
|--------|-------|---------------|
[Complete table with all 12 metrics]

Calculate Alpha evaluation score (excluding Completeness and Structured Presentation).
        """
        
        return prompt.strip()
    
    def _format_knowledge_base(self) -> str:
        """Format knowledge base for prompt inclusion."""
        knowledge_parts = []
        
        if self.golden_responses:
            knowledge_parts.append(f"Golden Responses:\n{json.dumps(self.golden_responses, indent=2)}")
        
        if self.buyability_profiles:
            knowledge_parts.append(f"Buyability Profiles:\n{json.dumps(self.buyability_profiles, indent=2)}")
        
        if self.fair_housing_guide:
            knowledge_parts.append(f"Fair Housing Guide:\n{json.dumps(self.fair_housing_guide, indent=2)}")
        
        return "\n\n".join(knowledge_parts) if knowledge_parts else "No additional knowledge base available."
    
    def _parse_llm_evaluation(self, llm_response: str) -> Dict[str, Dict[str, Any]]:
        """Parse LLM evaluation response into structured data."""
        metrics_data = {}
        
        # Extract table data using regex
        table_pattern = r'\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|'
        matches = re.findall(table_pattern, llm_response)
        
        for match in matches:
            if len(match) >= 3 and "Metric" not in match[0]:
                metric_name = match[0].strip()
                score_value = match[1].strip()
                justification = match[2].strip()
                
                # Clean up metric name to match our internal keys
                metric_key = self._normalize_metric_name(metric_name)
                
                if metric_key in self.metrics:
                    metrics_data[metric_key] = {
                        "score": score_value,
                        "justification": justification,
                        "raw_metric_name": metric_name
                    }
        
        return metrics_data
    
    def _normalize_metric_name(self, metric_name: str) -> str:
        """Normalize metric names to match internal keys."""
        name_mapping = {
            "personalization accuracy": "personalization_accuracy",
            "context-based personalization": "context_personalization",
            "next step identification": "next_step_identification",
            "assumption listing": "assumption_listing",
            "assumption trust": "assumption_trust",
            "calculation accuracy": "calculation_accuracy",
            "faithfulness to ground truth": "faithfulness_ground_truth",
            "overall accuracy": "overall_accuracy",
            "structured presentation": "structured_presentation",
            "coherence": "coherence",
            "completeness": "completeness",
            "fair housing classifier": "fair_housing_classifier"
        }
        
        normalized = metric_name.lower().strip()
        return name_mapping.get(normalized, normalized.replace(" ", "_").replace("-", "_"))
    
    def _calculate_scores(self, parsed_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate individual and aggregate scores."""
        individual_scores = {}
        
        for metric_key, metric_data in parsed_results.items():
            if metric_key in self.metrics:
                score_value = metric_data["score"]
                metric_config = self.metrics[metric_key]
                
                # Convert score to numeric value
                if metric_config["type"] == "boolean":
                    score_lower = score_value.lower()
                    if score_lower in ["true", "accurate", "present"]:
                        individual_scores[metric_key] = 10
                    else:
                        individual_scores[metric_key] = 0
                elif metric_config["type"] == "scale":
                    try:
                        scale_value = int(score_value)
                        individual_scores[metric_key] = scale_value * 2  # Convert 1-5 to 2,4,6,8,10
                    except ValueError:
                        individual_scores[metric_key] = 0
        
        # Calculate aggregate scores
        alpha_metrics = [k for k in individual_scores.keys() 
                        if k not in ["structured_presentation", "completeness"]]
        
        alpha_total = sum(individual_scores.get(k, 0) for k in alpha_metrics)
        alpha_max = len(alpha_metrics) * 10
        
        full_total = sum(individual_scores.values())
        full_max = len(individual_scores) * 10
        
        return {
            "individual": individual_scores,
            "alpha_total": alpha_total,
            "alpha_max": alpha_max,
            "full_total": full_total,
            "full_max": full_max,
            "alpha_percentage": round((alpha_total / alpha_max) * 100, 1) if alpha_max > 0 else 0,
            "full_percentage": round((full_total / full_max) * 100, 1) if full_max > 0 else 0,
        }
    
    def run(self, recorder: RecorderBase) -> Dict[str, Any]:
        """
        Run the evaluation using OpenAI Evals framework.
        
        Args:
            recorder: Recorder instance for logging results
            
        Returns:
            Dictionary with evaluation summary and results
        """
        samples = self._load_samples()
        
        total_samples = len(samples)
        results = []
        
        print(f"🔄 Running OpenAI Evals evaluation on {total_samples} samples...")
        
        for i, sample in enumerate(samples, 1):
            print(f"📊 Evaluating sample {i}/{total_samples}...")
            
            start_time = time.time()
            result = self.eval_sample(sample)
            evaluation_time = time.time() - start_time
            
            # Log to recorder
            recorder.record_event(
                "evaluation_result",
                sample_id=i,
                result=result,
                evaluation_time=evaluation_time,
                **sample
            )
            
            results.append(result)
        
        # Calculate summary statistics
        summary = self._calculate_summary(results)
        
        print(f"\n✅ OpenAI Evals evaluation completed!")
        print(f"📊 Average Alpha Score: {summary['avg_alpha_percentage']:.1f}%")
        print(f"📈 Average Full Score: {summary['avg_full_percentage']:.1f}%")
        
        return {
            "summary": summary,
            "results": results,
            "total_samples": total_samples,
            "evaluation_framework": "OpenAI Evals"
        }
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load evaluation samples from JSONL file."""
        samples = []
        
        if not os.path.exists(self.samples_jsonl):
            # Create default samples if file doesn't exist
            print(f"Warning: {self.samples_jsonl} not found, using default samples")
            return [
                {
                    "question": "what is my buyability?",
                    "candidate_answer": "Your buyability is $400,000.",
                    "user_profile": {
                        "annual_income": None,
                        "monthly_debts": None,
                        "down_payment": None,
                        "credit_score": None
                    }
                }
            ]
        
        try:
            with open(self.samples_jsonl, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        samples.append(json.loads(line))
        except Exception as e:
            print(f"Error loading samples: {e}")
            return []
        
        return samples
    
    def _calculate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics from evaluation results."""
        valid_results = [r for r in results if "error" not in r]
        
        if not valid_results:
            return {
                "avg_alpha_score": 0,
                "avg_alpha_percentage": 0,
                "avg_full_score": 0,
                "avg_full_percentage": 0,
                "total_evaluations": 0,
                "successful_evaluations": 0,
                "error_rate": 100.0,
            }
        
        avg_alpha_score = sum(r["alpha_score"] for r in valid_results) / len(valid_results)
        avg_alpha_percentage = sum(r["alpha_percentage"] for r in valid_results) / len(valid_results)
        avg_full_score = sum(r["full_score"] for r in valid_results) / len(valid_results)
        avg_full_percentage = sum(r["full_percentage"] for r in valid_results) / len(valid_results)
        
        return {
            "avg_alpha_score": round(avg_alpha_score, 2),
            "avg_alpha_percentage": round(avg_alpha_percentage, 1),
            "avg_full_score": round(avg_full_score, 2),
            "avg_full_percentage": round(avg_full_percentage, 1),
            "total_evaluations": len(results),
            "successful_evaluations": len(valid_results),
            "error_rate": round((len(results) - len(valid_results)) / len(results) * 100, 1),
        }


def main():
    """Main function to run the OpenAI Evals evaluation."""
    if not EVALS_AVAILABLE:
        print("❌ OpenAI Evals framework not available. Install with: pip install evals")
        return
    
    print("🤖 OPENAI EVALS ZILLOW HOME AFFORDABILITY JUDGE")
    print("=" * 70)
    print("Using OpenAI Evals framework for LLM-as-a-judge evaluation")
    print("12-metric criteria with deterministic scoring")
    print("=" * 70)
    
    # Create mock completion function for demonstration
    # In real usage, this would be provided by the evals framework
    class MockCompletionFn:
        def __init__(self, api_key: str, base_url: str = None):
            import openai
            self.client = openai.OpenAI(
                api_key=api_key,
                base_url=base_url
            )
        
        def __call__(self, prompt, **kwargs):
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=prompt,
                **kwargs
            )
            return MockCompletion(response.choices[0].message.content)
    
    class MockCompletion:
        def __init__(self, content):
            self.content = content
        
        def get_completions(self):
            return [self.content]
    
    # Initialize evaluator
    api_key = os.getenv("OPENAI_API_KEY", "your-api-key")
    base_url = os.getenv("OPENAI_BASE_URL")
    
    completion_fn = MockCompletionFn(api_key, base_url)
    
    evaluator = ZillowAffordabilityEval(
        completion_fns=[completion_fn],
        samples_jsonl="zillow_eval_samples.jsonl",
        name="zillow_affordability_openai_evals"
    )
    
    # Create mock recorder
    class MockRecorder:
        def __init__(self):
            self.events = []
        
        def record_event(self, event_type, **kwargs):
            self.events.append({"type": event_type, **kwargs})
    
    recorder = MockRecorder()
    
    # Run evaluation
    results = evaluator.run(recorder)
    
    print(f"\n📋 EVALUATION SUMMARY:")
    print("=" * 50)
    for key, value in results["summary"].items():
        print(f"• {key}: {value}")
    
    print(f"\n🔍 FRAMEWORK COMPARISON:")
    print("=" * 50)
    print("✅ OpenAI Evals Framework:")
    print("  - Standardized evaluation structure")
    print("  - Built-in recording and logging")
    print("  - Registry-based configuration")
    print("  - Compatible with oaieval CLI")
    print("  - Framework-level reproducibility")
    
    print(f"\n📊 DETAILED RESULTS:")
    print("=" * 50)
    for i, result in enumerate(results["results"], 1):
        if "error" not in result:
            print(f"Sample {i}:")
            print(f"  Alpha Score: {result['alpha_score']}/{result['alpha_max']} ({result['alpha_percentage']}%)")
            print(f"  Full Score: {result['full_score']}/{result['full_max']} ({result['full_percentage']}%)")
        else:
            print(f"Sample {i}: {result['error']}")


if __name__ == "__main__":
    main()