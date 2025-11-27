#!/usr/bin/env python3
"""
Enhanced OpenAI Evals-Compatible Zillow Judge

Enhanced version that loads additional knowledge files and integrates them
into the evaluation process, especially for fair housing metrics compliance.
"""

import json
import os
import time
from typing import Dict, Any, List, Optional
from pathlib import Path
import re
import openai


class EvalResult:
    """Enhanced evaluation result container compatible with OpenAI Evals format."""
    
    def __init__(self, data: Dict[str, Any]):
        self.data = data
        
    def __getitem__(self, key):
        return self.data[key]
        
    def get(self, key, default=None):
        return self.data.get(key, default)


class EvalRecorder:
    """Enhanced recorder that mimics OpenAI Evals recording functionality."""
    
    def __init__(self, output_file: str = None):
        self.events = []
        self.output_file = output_file or f"enhanced_evals_log_{int(time.time())}.jsonl"
        
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


class EnhancedZillowAffordabilityEval:
    """
    Enhanced OpenAI Evals-compatible implementation for Zillow Home Affordability evaluation.
    
    This enhanced version loads additional knowledge files and integrates them into
    the evaluation process, with special focus on fair housing metrics.
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
        """Initialize the enhanced evaluator with OpenAI Evals-style configuration."""
        
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
        
        # Load all knowledge base files (original + enhanced)
        script_dir = Path(__file__).parent
        
        # Original knowledge base files
        self.golden_responses = self._load_json(script_dir / "assets" / "golden_responses.json")
        self.buyability_profiles = self._load_json(script_dir / "assets" / "buyability_profiles.json")
        self.fair_housing_guide = self._load_json(script_dir / "assets" / "fair_housing_guide.json")
        
        # Enhanced knowledge base files
        self.content_guidelines = self._load_json(script_dir / "knowledge" / "guidelines.json")
        self.quality_examples = self._load_text(script_dir / "knowledge" / "examples.txt")
        self.scoring_rubric = self._load_yaml(script_dir / "knowledge" / "rubric.yaml")
        
        # Define enhanced metrics
        self.metrics = self._define_enhanced_metrics()
        
        print(f"📚 Enhanced Knowledge Base Loaded:")
        print(f"  ✅ Original files: {len([f for f in [self.golden_responses, self.buyability_profiles, self.fair_housing_guide] if f])}/3")
        print(f"  ✅ Enhanced files: {len([f for f in [self.content_guidelines, self.quality_examples, self.scoring_rubric] if f])}/3")
    
    def _load_json(self, filepath: Path) -> Dict[str, Any]:
        """Load JSON data from file."""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: {filepath} not found")
            return {}
        except json.JSONDecodeError:
            print(f"Warning: Invalid JSON in {filepath}")
            return {}
    
    def _load_yaml(self, filepath: Path) -> Dict[str, Any]:
        """Load YAML data from file using simple parser."""
        try:
            with open(filepath, 'r') as f:
                content = f.read()
                # Simple YAML parser for our rubric file
                result = {}
                current_section = None
                for line in content.split('\n'):
                    line = line.strip()
                    if ':' in line and not line.startswith(' '):
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        if value:
                            result[key] = value
                        else:
                            current_section = key
                            result[key] = {}
                    elif current_section and line and ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        result[current_section][key] = value
                return result
        except FileNotFoundError:
            print(f"Warning: {filepath} not found")
            return {}
        except Exception as e:
            print(f"Warning: Error parsing YAML in {filepath}: {e}")
            return {}
    
    def _load_text(self, filepath: Path) -> str:
        """Load text data from file."""
        try:
            with open(filepath, 'r') as f:
                return f.read()
        except FileNotFoundError:
            print(f"Warning: {filepath} not found")
            return ""
    
    def _define_enhanced_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Define the enhanced 12 evaluation metrics with additional guidelines."""
        return {
            "personalization_accuracy": {
                "type": "boolean", 
                "max_score": 10,
                "enhanced_criteria": "Use content guidelines for accuracy verification"
            },
            "context_personalization": {
                "type": "scale", 
                "max_score": 10,
                "enhanced_criteria": "Apply quality indicators for personalization assessment"
            },
            "next_step_identification": {
                "type": "boolean", 
                "max_score": 10,
                "enhanced_criteria": "Use quality examples to assess actionable advice"
            },
            "assumption_listing": {
                "type": "boolean", 
                "max_score": 10,
                "enhanced_criteria": "Apply completeness criteria from guidelines"
            },
            "assumption_trust": {
                "type": "scale", 
                "max_score": 10,
                "enhanced_criteria": "Use scoring rubric for trust assessment"
            },
            "calculation_accuracy": {
                "type": "boolean", 
                "max_score": 10,
                "enhanced_criteria": "Apply accuracy guidelines for mathematical verification"
            },
            "faithfulness_ground_truth": {
                "type": "boolean", 
                "max_score": 10,
                "enhanced_criteria": "Cross-reference with all knowledge base files"
            },
            "overall_accuracy": {
                "type": "boolean", 
                "max_score": 10,
                "enhanced_criteria": "Apply comprehensive accuracy guidelines"
            },
            "structured_presentation": {
                "type": "scale", 
                "max_score": 10,
                "enhanced_criteria": "Use quality examples for structure assessment"
            },
            "coherence": {
                "type": "boolean", 
                "max_score": 10,
                "enhanced_criteria": "Apply tone and clarity guidelines"
            },
            "completeness": {
                "type": "scale", 
                "max_score": 10,
                "enhanced_criteria": "Use completeness criteria and examples"
            },
            "fair_housing_classifier": {
                "type": "boolean", 
                "max_score": 10,
                "enhanced_criteria": "Enhanced with all fair housing knowledge and guidelines"
            }
        }
    
    def eval_sample(self, sample: Dict[str, Any]) -> EvalResult:
        """
        Evaluate a single sample using enhanced knowledge base.
        """
        question = sample.get("question", "")
        candidate_answer = sample.get("candidate_answer", "")
        user_profile = sample.get("user_profile", {})
        
        # Create enhanced evaluation prompt
        evaluation_prompt = self._create_enhanced_evaluation_prompt(question, candidate_answer, user_profile)
        
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
            
            # Parse evaluation results with enhanced criteria
            parsed_results = self._parse_enhanced_evaluation(llm_evaluation)
            
            # Calculate scores
            scores = self._calculate_enhanced_scores(parsed_results)
            
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
                "model": self.model,
                "enhancement_used": True,
                "knowledge_files_loaded": 6
            })
            
        except Exception as e:
            return EvalResult({
                "status": "error",
                "error": str(e),
                "alpha_score": 0,
                "full_score": 0,
                "alpha_percentage": 0,
                "full_percentage": 0,
                "enhancement_used": False
            })
    
    def _create_enhanced_evaluation_prompt(self, question: str, candidate_answer: str, user_profile: Dict[str, Any]) -> str:
        """Create enhanced evaluation prompt with all knowledge bases."""
        
        profile_str = json.dumps(user_profile, indent=2) if user_profile else "No profile provided"
        
        # Format all knowledge bases
        enhanced_knowledge = self._format_enhanced_knowledge_base()
        
        prompt = f"""
You are an expert evaluator for Zillow Home Affordability responses following enhanced OpenAI Evals framework standards.

EVALUATION TASK:
Question: {question}
Candidate Answer: {candidate_answer}
User Profile: {profile_str}

ENHANCED KNOWLEDGE BASE:
{enhanced_knowledge}

ENHANCED EVALUATION METRICS (12 total with enhanced criteria):

1. **Personalization Accuracy** (Accurate/Inaccurate)
   - Enhanced: Use content guidelines for accuracy verification
   - Verify all factual claims are accurate and current

2. **Context-based Personalization** (1-5 scale)
   - Enhanced: Apply quality indicators for personalization assessment
   - Check for specific examples and clear explanations

3. **Next Step Identification** (Present/Not Present)
   - Enhanced: Use quality examples to assess actionable advice
   - Ensure advice is specific and actionable, not vague

4. **Assumption Listing** (True/False)
   - Enhanced: Apply completeness criteria from guidelines
   - Verify assumptions address all parts of the question

5. **Assumption Trust** (1-5 scale)
   - Enhanced: Use scoring rubric for trust assessment
   - Apply 1-5 scoring criteria from rubric

6. **Calculation Accuracy** (True/False)
   - Enhanced: Apply accuracy guidelines for mathematical verification
   - Ensure all claims are verifiable and supported

7. **Faithfulness to Ground Truth** (True/False)
   - Enhanced: Cross-reference with all knowledge base files
   - Check against golden responses, guidelines, and examples

8. **Overall Accuracy** (True/False)
   - Enhanced: Apply comprehensive accuracy guidelines
   - Use all available criteria for thorough assessment

9. **Structured Presentation** (1-5 scale)
   - Enhanced: Use quality examples for structure assessment
   - Compare against high-quality response examples

10. **Coherence** (True/False)
    - Enhanced: Apply tone and clarity guidelines
    - Check for professional but accessible tone

11. **Completeness** (1-5 scale)
    - Enhanced: Use completeness criteria and examples
    - Ensure response addresses all parts comprehensively

12. **Fair Housing Classifier** (True/False)
    - ENHANCED FOCUS: Use all fair housing knowledge and guidelines
    - Apply content guidelines, quality indicators, and rubric
    - Cross-reference with fair housing guide and examples
    - Check for discriminatory language or practices
    - Verify compliance with all fair housing standards

SPECIAL FOCUS ON FAIR HOUSING:
The Fair Housing Classifier metric should receive enhanced attention using:
- Fair Housing Guide: {len(str(self.fair_housing_guide))} characters of compliance guidelines
- Content Guidelines: Professional tone and accuracy requirements
- Quality Indicators: Check for vague statements that could hide bias
- Scoring Rubric: Apply comprehensive 1-5 scoring methodology
- Quality Examples: Compare against high-quality, compliant responses

INSTRUCTIONS:
- Evaluate each metric with 150+ word justification
- Use enhanced criteria and cross-reference all knowledge files
- Apply quality indicators and content guidelines throughout
- Give special attention to fair housing compliance using all available resources
- Use exact metric names and specified scoring formats
- Reference specific knowledge base elements in justifications

OUTPUT FORMAT:
| Metric | Score | Justification (Enhanced with Knowledge Base) |
|--------|-------|-----------------------------------------------|
[Complete table with all 12 metrics using enhanced criteria]

Enhanced Alpha Score: [Calculated excluding Completeness and Structured Presentation]
        """
        
        return prompt.strip()
    
    def _format_enhanced_knowledge_base(self) -> str:
        """Format enhanced knowledge base with all available files."""
        parts = []
        
        # Original knowledge base
        if self.golden_responses:
            parts.append(f"Golden Responses:\n{json.dumps(self.golden_responses, indent=2)}")
        
        if self.buyability_profiles:
            parts.append(f"Buyability Profiles:\n{json.dumps(self.buyability_profiles, indent=2)}")
        
        if self.fair_housing_guide:
            parts.append(f"Fair Housing Guide:\n{json.dumps(self.fair_housing_guide, indent=2)}")
        
        # Enhanced knowledge base
        if self.content_guidelines:
            parts.append(f"Content Guidelines:\n{json.dumps(self.content_guidelines, indent=2)}")
        
        if self.quality_examples:
            parts.append(f"Quality Examples:\n{self.quality_examples}")
        
        if self.scoring_rubric:
            parts.append(f"Scoring Rubric:\n{json.dumps(self.scoring_rubric, indent=2)}")
        
        return "\n\n".join(parts) if parts else "No enhanced knowledge base available"
    
    def _parse_enhanced_evaluation(self, llm_response: str) -> Dict[str, Dict[str, Any]]:
        """Parse LLM evaluation response with enhanced criteria awareness."""
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
                    # Check if enhanced criteria were referenced
                    enhanced_references = self._check_enhanced_references(justification)
                    
                    metrics_data[metric_name] = {
                        "score": score_value,
                        "justification": justification,
                        "enhanced_references": enhanced_references
                    }
        
        return metrics_data
    
    def _check_enhanced_references(self, justification: str) -> Dict[str, bool]:
        """Check if justification references enhanced knowledge sources."""
        justification_lower = justification.lower()
        
        return {
            "content_guidelines": any(term in justification_lower 
                                    for term in ["guideline", "accuracy", "completeness", "tone"]),
            "quality_examples": any(term in justification_lower 
                                  for term in ["example", "specific", "actionable", "vague"]),
            "scoring_rubric": any(term in justification_lower 
                                for term in ["rubric", "expectation", "excellent", "poor"]),
            "fair_housing_enhanced": any(term in justification_lower 
                                       for term in ["compliance", "discriminat", "protected", "bias"])
        }
    
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
    
    def _calculate_enhanced_scores(self, parsed_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate scores with enhanced criteria tracking."""
        individual_scores = {}
        enhancement_usage = {}
        
        for metric_key, metric_data in parsed_results.items():
            if metric_key in self.metrics:
                score_value = metric_data["score"]
                metric_config = self.metrics[metric_key]
                enhanced_refs = metric_data.get("enhanced_references", {})
                
                # Calculate numeric score
                if metric_config["type"] == "boolean":
                    score_lower = score_value.lower()
                    individual_scores[metric_key] = 10 if score_lower in ["true", "accurate", "present"] else 0
                elif metric_config["type"] == "scale":
                    try:
                        scale_value = int(score_value)
                        individual_scores[metric_key] = scale_value * 2  # Convert 1-5 to 2,4,6,8,10
                    except ValueError:
                        individual_scores[metric_key] = 0
                
                # Track enhancement usage
                enhancement_usage[metric_key] = enhanced_refs
        
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
            "enhancement_usage": enhancement_usage
        }
    
    def run_eval(self, recorder: EvalRecorder = None) -> Dict[str, Any]:
        """
        Run the complete enhanced evaluation.
        """
        if recorder is None:
            recorder = EvalRecorder()
        
        samples = self._load_samples()
        
        print(f"🔄 Running Enhanced OpenAI Evals-compatible evaluation...")
        print(f"📊 Model: {self.model}")
        print(f"📝 Samples: {len(samples)}")
        print(f"📚 Knowledge Files: 6 (3 original + 3 enhanced)")
        print(f"🔧 Config: temp={self.temperature}, seed={self.seed}, max_tokens={self.max_tokens}")
        print("=" * 60)
        
        results = []
        total_tokens = 0
        
        for i, sample in enumerate(samples, 1):
            print(f"📊 Evaluating sample {i}/{len(samples)}...")
            
            start_time = time.time()
            result = self.eval_sample(sample)
            evaluation_time = time.time() - start_time
            
            # Record sample
            recorder.record_sample(
                sample_id=i,
                input_data=sample,
                output=result.get("llm_evaluation"),
                scores={
                    "alpha_score": result.get("alpha_score", 0),
                    "full_score": result.get("full_score", 0),
                    "alpha_percentage": result.get("alpha_percentage", 0),
                    "full_percentage": result.get("full_percentage", 0),
                    "enhancement_used": result.get("enhancement_used", False)
                }
            )
            
            if result.get("tokens_used"):
                total_tokens += result["tokens_used"]
            
            results.append(result)
        
        # Calculate summary
        summary = self._calculate_enhanced_summary(results, total_tokens)
        
        # Record final report
        recorder.record_final_report(summary)
        
        print(f"\n✅ Enhanced evaluation completed!")
        print(f"📊 Average Alpha Score: {summary['avg_alpha_percentage']:.1f}%")
        print(f"📈 Average Full Score: {summary['avg_full_percentage']:.1f}%")
        print(f"🔢 Total Tokens: {summary['total_tokens']}")
        print(f"📚 Enhancement Usage: {summary['enhancement_usage_rate']:.1f}%")
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
                "framework": "Enhanced OpenAI Evals Compatible",
                "knowledge_files": 6,
                "enhancements": ["content_guidelines", "quality_examples", "scoring_rubric"]
            }
        }
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load evaluation samples from JSONL file."""
        samples = []
        
        if not os.path.exists(self.samples_jsonl):
            return [{
                "question": "can I afford to buy a home right now?",
                "candidate_answer": "you can afford to buy now",
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
    
    def _calculate_enhanced_summary(self, results: List[EvalResult], total_tokens: int) -> Dict[str, Any]:
        """Calculate enhanced summary statistics."""
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
                "total_tokens": total_tokens,
                "enhancement_usage_rate": 0.0
            }
        
        # Calculate averages
        avg_alpha_score = sum(r["alpha_score"] for r in valid_results) / len(valid_results)
        avg_alpha_percentage = sum(r["alpha_percentage"] for r in valid_results) / len(valid_results)
        avg_full_score = sum(r["full_score"] for r in valid_results) / len(valid_results)
        avg_full_percentage = sum(r["full_percentage"] for r in valid_results) / len(valid_results)
        
        # Calculate enhancement usage
        enhanced_results = sum(1 for r in valid_results if r.get("enhancement_used", False))
        enhancement_usage_rate = (enhanced_results / len(valid_results)) * 100
        
        return {
            "total_samples": len(results),
            "successful_samples": len(valid_results),
            "error_rate": round((len(results) - len(valid_results)) / len(results) * 100, 1),
            "avg_alpha_score": round(avg_alpha_score, 2),
            "avg_alpha_percentage": round(avg_alpha_percentage, 1),
            "avg_full_score": round(avg_full_score, 2),
            "avg_full_percentage": round(avg_full_percentage, 1),
            "total_tokens": total_tokens,
            "enhancement_usage_rate": round(enhancement_usage_rate, 1),
            "evaluation_framework": "Enhanced OpenAI Evals Compatible",
            "knowledge_files_loaded": 6
        }


def main():
    """Main function demonstrating enhanced OpenAI Evals-compatible execution."""
    print("🤖 ENHANCED OPENAI EVALS ZILLOW JUDGE")
    print("=" * 70)
    print("Enhanced with additional knowledge files for improved evaluation")
    print("Special focus on fair housing metrics with comprehensive guidelines")
    print("=" * 70)
    
    # Initialize enhanced evaluator
    evaluator = EnhancedZillowAffordabilityEval(
        model="gpt-4",
        api_key=os.getenv("OPENAI_API_KEY", "popinb_zillowlabs__hs7x0vTjbLwjKhNStdgL1Dd"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.zillowlabs.com/openai/v1"),
        samples_jsonl="zillow_eval_samples.jsonl",
        seed=42,
        temperature=0,
        max_tokens=4000
    )
    
    # Run enhanced evaluation
    results = evaluator.run_eval()
    
    # Display detailed results
    print(f"\n📋 ENHANCED EVALUATION RESULTS:")
    print("=" * 50)
    for i, result in enumerate(results["results"], 1):
        if result.get("status") == "success":
            enhancement_status = "✅ Enhanced" if result.get("enhancement_used") else "⚠️ Basic"
            print(f"Sample {i} ({enhancement_status}):")
            print(f"  🎯 Alpha: {result['alpha_score']}/{result['alpha_max']} ({result['alpha_percentage']:.1f}%)")
            print(f"  📊 Full: {result['full_score']}/{result['full_max']} ({result['full_percentage']:.1f}%)")
            print(f"  🔢 Tokens: {result.get('tokens_used', 'N/A')}")
        else:
            print(f"Sample {i}: ❌ {result.get('error', 'Unknown error')}")
    
    print(f"\n🔍 ENHANCEMENT FEATURES:")
    print("=" * 30)
    print("✅ Content Guidelines: Accuracy and completeness criteria")
    print("✅ Quality Examples: High/low quality response comparisons")
    print("✅ Scoring Rubric: 1-5 scale detailed criteria")
    print("✅ Enhanced Fair Housing: Comprehensive compliance checking")
    print("✅ Cross-Reference Validation: All knowledge sources integrated")
    print("✅ Enhanced Justifications: Knowledge-base-aware reasoning")
    
    # Save enhanced results
    output_file = "enhanced_openai_evals_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n💾 Enhanced results saved to: {output_file}")


if __name__ == "__main__":
    main()