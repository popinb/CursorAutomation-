#!/usr/bin/env python3
"""
Standalone OpenAI Evals-Compatible Zillow Judge

This implementation follows OpenAI Evals framework patterns and structure
without requiring the full evals package installation. It demonstrates
how the evaluation would work within the OpenAI Evals ecosystem.
"""

import json
import os
import time
from typing import Dict, Any, List, Optional
from pathlib import Path
import re
import openai


class EvalResult:
    """Simple evaluation result container compatible with OpenAI Evals format."""
    
    def __init__(self, data: Dict[str, Any]):
        self.data = data
        
    def __getitem__(self, key):
        return self.data[key]
        
    def get(self, key, default=None):
        return self.data.get(key, default)


class EvalRecorder:
    """Simple recorder that mimics OpenAI Evals recording functionality."""
    
    def __init__(self, output_file: str = None):
        self.events = []
        self.output_file = output_file or f"evals_log_{int(time.time())}.jsonl"
        
    def record_event(self, event_type: str, **kwargs):
        """Record an evaluation event."""
        event = {
            "type": event_type,
            "timestamp": time.time(),
            **kwargs
        }
        self.events.append(event)
        
        # Write to file
        with open(self.output_file, "a") as f:
            f.write(json.dumps(event) + "\n")
    
    def record_sample(self, sample_id: int, input_data: Dict, output: Any, scores: Dict):
        """Record a sample evaluation."""
        self.record_event(
            "sample_evaluation",
            sample_id=sample_id,
            input=input_data,
            output=output,
            scores=scores
        )
    
    def record_final_report(self, summary: Dict[str, Any]):
        """Record the final evaluation summary."""
        self.record_event("final_report", **summary)


class ZillowAffordabilityEval:
    """
    OpenAI Evals-compatible implementation for Zillow Home Affordability evaluation.
    
    This follows the OpenAI Evals framework structure and patterns while being
    completely standalone and not requiring the evals package.
    """
    
    def __init__(
        self,
        model: str = "gpt-4",
        api_key: str = None,
        base_url: str = None,
        samples_jsonl: str = "zillow_eval_samples.jsonl",
        seed: int = 42,
        max_tokens: int = 4000,
        temperature: float = 0,
    ):
        """Initialize the evaluator with OpenAI Evals-style configuration."""
        
        # OpenAI client setup
        self.model = model
        self.client = openai.OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL")
        )
        
        # Evaluation configuration
        self.samples_jsonl = samples_jsonl
        self.seed = seed
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Load knowledge base
        script_dir = Path(__file__).parent
        self.golden_responses = self._load_json(script_dir / "assets" / "golden_responses.json")
        self.buyability_profiles = self._load_json(script_dir / "assets" / "buyability_profiles.json")
        self.fair_housing_guide = self._load_json(script_dir / "assets" / "fair_housing_guide.json")
        
        # Define metrics
        self.metrics = self._define_metrics()
    
    def _load_json(self, filepath: Path) -> Dict[str, Any]:
        """Load JSON data from file."""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError:
            return {}
    
    def _define_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Define the 12 evaluation metrics compatible with OpenAI Evals format."""
        return {
            "personalization_accuracy": {"type": "boolean", "max_score": 10},
            "context_personalization": {"type": "scale", "max_score": 10},
            "next_step_identification": {"type": "boolean", "max_score": 10},
            "assumption_listing": {"type": "boolean", "max_score": 10},
            "assumption_trust": {"type": "scale", "max_score": 10},
            "calculation_accuracy": {"type": "boolean", "max_score": 10},
            "faithfulness_ground_truth": {"type": "boolean", "max_score": 10},
            "overall_accuracy": {"type": "boolean", "max_score": 10},
            "structured_presentation": {"type": "scale", "max_score": 10},
            "coherence": {"type": "boolean", "max_score": 10},
            "completeness": {"type": "scale", "max_score": 10},
            "fair_housing_classifier": {"type": "boolean", "max_score": 10}
        }
    
    def eval_sample(self, sample: Dict[str, Any]) -> EvalResult:
        """
        Evaluate a single sample following OpenAI Evals patterns.
        
        Args:
            sample: Dictionary containing question, candidate_answer, and user_profile
            
        Returns:
            EvalResult with evaluation metrics and scores
        """
        question = sample.get("question", "")
        candidate_answer = sample.get("candidate_answer", "")
        user_profile = sample.get("user_profile", {})
        
        # Create evaluation prompt
        evaluation_prompt = self._create_evaluation_prompt(question, candidate_answer, user_profile)
        
        try:
            # Get LLM evaluation
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": evaluation_prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                seed=self.seed
            )
            
            llm_evaluation = response.choices[0].message.content
            
            # Parse evaluation results
            parsed_results = self._parse_evaluation(llm_evaluation)
            
            # Calculate scores
            scores = self._calculate_scores(parsed_results)
            
            return EvalResult({
                "status": "success",
                "llm_evaluation": llm_evaluation,
                "individual_scores": scores["individual"],
                "alpha_score": scores["alpha_total"],
                "alpha_max": scores["alpha_max"],
                "full_score": scores["full_total"],
                "full_max": scores["full_max"],
                "alpha_percentage": scores["alpha_percentage"],
                "full_percentage": scores["full_percentage"],
                "tokens_used": response.usage.total_tokens,
                "model": self.model
            })
            
        except Exception as e:
            return EvalResult({
                "status": "error",
                "error": str(e),
                "alpha_score": 0,
                "full_score": 0,
                "alpha_percentage": 0,
                "full_percentage": 0
            })
    
    def _create_evaluation_prompt(self, question: str, candidate_answer: str, user_profile: Dict[str, Any]) -> str:
        """Create the evaluation prompt for LLM-as-a-judge."""
        
        profile_str = json.dumps(user_profile, indent=2) if user_profile else "No profile provided"
        knowledge_base = self._format_knowledge_base()
        
        prompt = f"""
You are an expert evaluator for Zillow Home Affordability responses following OpenAI Evals framework standards.

EVALUATION TASK:
Question: {question}
Candidate Answer: {candidate_answer}
User Profile: {profile_str}

KNOWLEDGE BASE:
{knowledge_base}

EVALUATION METRICS (12 total):

1. **Personalization Accuracy** (Accurate/Inaccurate)
2. **Context-based Personalization** (1-5 scale)
3. **Next Step Identification** (Present/Not Present)
4. **Assumption Listing** (True/False)
5. **Assumption Trust** (1-5 scale)
6. **Calculation Accuracy** (True/False)
7. **Faithfulness to Ground Truth** (True/False)
8. **Overall Accuracy** (True/False)
9. **Structured Presentation** (1-5 scale)
10. **Coherence** (True/False)
11. **Completeness** (1-5 scale)
12. **Fair Housing Classifier** (True/False)

INSTRUCTIONS:
- Evaluate each metric with 150+ word justification
- Use exact metric names and specified scoring formats
- Provide structured output for automated parsing

OUTPUT FORMAT:
| Metric | Score | Justification |
|--------|-------|---------------|
[Complete table with all 12 metrics]

Alpha Score: [Calculated excluding Completeness and Structured Presentation]
        """
        
        return prompt.strip()
    
    def _format_knowledge_base(self) -> str:
        """Format knowledge base for prompt inclusion."""
        parts = []
        if self.golden_responses:
            parts.append(f"Golden Responses: {json.dumps(self.golden_responses, indent=2)}")
        if self.buyability_profiles:
            parts.append(f"Buyability Profiles: {json.dumps(self.buyability_profiles, indent=2)}")
        if self.fair_housing_guide:
            parts.append(f"Fair Housing Guide: {json.dumps(self.fair_housing_guide, indent=2)}")
        return "\n\n".join(parts) if parts else "No knowledge base available"
    
    def _parse_evaluation(self, llm_response: str) -> Dict[str, Dict[str, Any]]:
        """Parse LLM evaluation response into structured data."""
        metrics_data = {}
        
        # Extract table data
        table_pattern = r'\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|'
        matches = re.findall(table_pattern, llm_response)
        
        for match in matches:
            if len(match) >= 3 and "Metric" not in match[0]:
                metric_name = self._normalize_metric_name(match[0].strip())
                score_value = match[1].strip()
                justification = match[2].strip()
                
                if metric_name in self.metrics:
                    metrics_data[metric_name] = {
                        "score": score_value,
                        "justification": justification
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
        """Calculate scores following OpenAI Evals aggregation patterns."""
        individual_scores = {}
        
        for metric_key, metric_data in parsed_results.items():
            if metric_key in self.metrics:
                score_value = metric_data["score"]
                metric_config = self.metrics[metric_key]
                
                if metric_config["type"] == "boolean":
                    score_lower = score_value.lower()
                    individual_scores[metric_key] = 10 if score_lower in ["true", "accurate", "present"] else 0
                elif metric_config["type"] == "scale":
                    try:
                        scale_value = int(score_value)
                        individual_scores[metric_key] = scale_value * 2  # Convert 1-5 to 2,4,6,8,10
                    except ValueError:
                        individual_scores[metric_key] = 0
        
        # Calculate aggregate scores (OpenAI Evals style)
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
    
    def run_eval(self, recorder: EvalRecorder = None) -> Dict[str, Any]:
        """
        Run the complete evaluation following OpenAI Evals patterns.
        
        Args:
            recorder: Optional recorder for logging (OpenAI Evals style)
            
        Returns:
            Complete evaluation results with summary statistics
        """
        if recorder is None:
            recorder = EvalRecorder()
        
        # Load samples
        samples = self._load_samples()
        
        print(f"🔄 Running OpenAI Evals-compatible evaluation...")
        print(f"📊 Model: {self.model}")
        print(f"📝 Samples: {len(samples)}")
        print(f"🔧 Config: temp={self.temperature}, seed={self.seed}, max_tokens={self.max_tokens}")
        print("=" * 60)
        
        results = []
        total_tokens = 0
        
        for i, sample in enumerate(samples, 1):
            print(f"📊 Evaluating sample {i}/{len(samples)}...")
            
            start_time = time.time()
            result = self.eval_sample(sample)
            evaluation_time = time.time() - start_time
            
            # Record sample (OpenAI Evals style)
            recorder.record_sample(
                sample_id=i,
                input_data=sample,
                output=result.get("llm_evaluation"),
                scores={
                    "alpha_score": result.get("alpha_score", 0),
                    "full_score": result.get("full_score", 0),
                    "alpha_percentage": result.get("alpha_percentage", 0),
                    "full_percentage": result.get("full_percentage", 0),
                }
            )
            
            if result.get("tokens_used"):
                total_tokens += result["tokens_used"]
            
            results.append(result)
        
        # Calculate summary (OpenAI Evals format)
        summary = self._calculate_summary(results, total_tokens)
        
        # Record final report
        recorder.record_final_report(summary)
        
        print(f"\n✅ Evaluation completed!")
        print(f"📊 Average Alpha Score: {summary['avg_alpha_percentage']:.1f}%")
        print(f"📈 Average Full Score: {summary['avg_full_percentage']:.1f}%")
        print(f"🔢 Total Tokens: {summary['total_tokens']}")
        print(f"📝 Log file: {recorder.output_file}")
        
        return {
            "summary": summary,
            "results": results,
            "recorder": recorder,
            "eval_config": {
                "model": self.model,
                "temperature": self.temperature,
                "seed": self.seed,
                "max_tokens": self.max_tokens,
                "framework": "OpenAI Evals Compatible"
            }
        }
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load evaluation samples from JSONL file."""
        samples = []
        
        if not os.path.exists(self.samples_jsonl):
            return [{
                "question": "what is my buyability?",
                "candidate_answer": "Your buyability is $400,000.",
                "user_profile": {"annual_income": None, "monthly_debts": None, "down_payment": None, "credit_score": None}
            }]
        
        try:
            with open(self.samples_jsonl, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        samples.append(json.loads(line))
        except Exception as e:
            print(f"Warning: Error loading samples: {e}")
            return []
        
        return samples
    
    def _calculate_summary(self, results: List[EvalResult], total_tokens: int) -> Dict[str, Any]:
        """Calculate summary statistics in OpenAI Evals format."""
        valid_results = [r for r in results if r.get("status") == "success"]
        
        if not valid_results:
            return {
                "total_samples": len(results),
                "successful_samples": 0,
                "error_rate": 100.0,
                "avg_alpha_score": 0,
                "avg_alpha_percentage": 0,
                "avg_full_score": 0,
                "avg_full_percentage": 0,
                "total_tokens": total_tokens
            }
        
        # OpenAI Evals style aggregation
        avg_alpha_score = sum(r["alpha_score"] for r in valid_results) / len(valid_results)
        avg_alpha_percentage = sum(r["alpha_percentage"] for r in valid_results) / len(valid_results)
        avg_full_score = sum(r["full_score"] for r in valid_results) / len(valid_results)
        avg_full_percentage = sum(r["full_percentage"] for r in valid_results) / len(valid_results)
        
        return {
            "total_samples": len(results),
            "successful_samples": len(valid_results),
            "error_rate": round((len(results) - len(valid_results)) / len(results) * 100, 1),
            "avg_alpha_score": round(avg_alpha_score, 2),
            "avg_alpha_percentage": round(avg_alpha_percentage, 1),
            "avg_full_score": round(avg_full_score, 2),
            "avg_full_percentage": round(avg_full_percentage, 1),
            "total_tokens": total_tokens,
            "evaluation_framework": "OpenAI Evals Compatible"
        }


def main():
    """Main function demonstrating OpenAI Evals-compatible execution."""
    print("🤖 STANDALONE OPENAI EVALS ZILLOW JUDGE")
    print("=" * 70)
    print("OpenAI Evals framework patterns without requiring evals package")
    print("Uses your Zillow Labs API credentials with deterministic evaluation")
    print("=" * 70)
    
    # Initialize evaluator
    evaluator = ZillowAffordabilityEval(
        model="gpt-4",
        api_key=os.getenv("OPENAI_API_KEY", "popinb_zillowlabs__hs7x0vTjbLwjKhNStdgL1Dd"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.zillowlabs.com/openai/v1"),
        samples_jsonl="zillow_eval_samples.jsonl",
        seed=42,
        temperature=0,
        max_tokens=4000
    )
    
    # Run evaluation
    results = evaluator.run_eval()
    
    # Display detailed results
    print(f"\n📋 DETAILED RESULTS:")
    print("=" * 50)
    for i, result in enumerate(results["results"], 1):
        if result.get("status") == "success":
            print(f"Sample {i}:")
            print(f"  🎯 Alpha: {result['alpha_score']}/{result['alpha_max']} ({result['alpha_percentage']:.1f}%)")
            print(f"  📊 Full: {result['full_score']}/{result['full_max']} ({result['full_percentage']:.1f}%)")
            print(f"  🔢 Tokens: {result.get('tokens_used', 'N/A')}")
        else:
            print(f"Sample {i}: ❌ {result.get('error', 'Unknown error')}")
    
    print(f"\n🔍 FRAMEWORK FEATURES:")
    print("=" * 50)
    print("✅ OpenAI Evals-compatible structure")
    print("✅ Structured logging with timestamps")
    print("✅ Sample-by-sample recording")
    print("✅ Standardized result format")
    print("✅ Framework-style configuration")
    print("✅ Deterministic evaluation (seed=42)")
    print("✅ Token usage tracking")
    print("✅ Error handling and reporting")
    
    # Save formatted results
    output_file = "openai_evals_compatible_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n💾 Results saved to: {output_file}")


if __name__ == "__main__":
    main()