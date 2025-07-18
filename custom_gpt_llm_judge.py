#!/usr/bin/env python3
"""
Custom GPT LLM-as-a-Judge using OpenAI Responses API

This evaluator implements the recommended parameters for LLM-as-a-Judge systems
to ensure deterministic, reliable evaluation with minimal stochasticity.
"""

import json
import os
import time
from typing import Dict, Any, List, Optional
from pathlib import Path
import openai


class CustomGPTLLMJudge:
    """
    Custom GPT LLM-as-a-Judge with optimized parameters for evaluation tasks.
    
    Implements recommended settings:
    - temperature=0 for deterministic output
    - seed=42 for reproducible results
    - max_tokens=750 for comprehensive responses
    - All randomness parameters disabled
    """
    
    def __init__(self, 
                 api_key: str = None, 
                 base_url: str = None, 
                 model: str = "gpt-4",
                 custom_instructions: str = None):
        """
        Initialize the Custom GPT LLM Judge.
        
        Args:
            api_key: OpenAI API key
            base_url: OpenAI base URL (for custom endpoints)
            model: Model to use (gpt-4, o1-preview, etc.)
            custom_instructions: Custom GPT instructions for the judge
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.model = model
        
        # Optimal LLM-as-a-Judge parameters (as recommended)
        self.judge_params = {
            # Sampling / randomness - Forces greedy decoding
            "temperature": 0,
            "top_p": 1.0,  # Irrelevant with temp=0, but prevents truncation
            "seed": 42,    # Fixed seed for reproducible results
            
            # Output length - Prevents cutoff while controlling costs
            "max_tokens": 750,  # Balanced for comprehensive responses
            
            # Repetition shaping - Judges rarely need diversity
            "frequency_penalty": 0,
            "presence_penalty": 0,
            
            # Response count - Sample once for consistency
            "n": 1,
            
            # Stop conditions - Clean evaluation endings
            "stop": ["\n###", "###"]
        }
        
        # Initialize OpenAI client
        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
            
        self.client = openai.OpenAI(**client_kwargs)
        
        # Set custom GPT instructions
        self.custom_instructions = custom_instructions or self._default_custom_gpt_instructions()
        
        # Load ground truth data
        script_dir = Path(__file__).parent
        self.golden_responses = self._load_json(script_dir / "assets" / "golden_responses.json")
        self.buyability_profiles = self._load_json(script_dir / "assets" / "buyability_profiles.json")
        self.fair_housing_guide = self._load_json(script_dir / "assets" / "fair_housing_guide.json")
    
    def _default_custom_gpt_instructions(self) -> str:
        """Default Custom GPT instructions for Zillow home affordability evaluation."""
        return """
You are a Custom GPT specialized as a Zillow Home Affordability Judge. Your role is to evaluate responses about home buying and affordability with expert-level precision.

CORE IDENTITY:
- Expert in real estate, mortgage lending, and home affordability
- Specialized in Zillow's BuyAbility methodology and tools
- Trained on fair housing compliance and best practices
- Focused on providing detailed, objective evaluations

EVALUATION EXPERTISE:
- Personalization accuracy assessment
- Financial calculation verification
- User experience and guidance quality
- Regulatory compliance (Fair Housing Act)
- Communication effectiveness analysis

RESPONSE STYLE:
- Provide detailed 150+ word justifications for each metric
- Use precise scoring based on provided criteria
- Reference ground truth documents when available
- Maintain objectivity and consistency
- Focus on actionable feedback for improvement

KNOWLEDGE BASE:
- Zillow BuyAbility calculation methods
- Mortgage industry standards and practices
- Fair housing regulations and compliance
- User experience best practices for financial guidance
- Home buying process workflows and requirements

Always follow the exact evaluation instructions provided and maintain the highest standards of accuracy and professionalism.
        """
    
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
    
    def evaluate(self, 
                 candidate_answer: str, 
                 question: str = "", 
                 user_profile: Dict[str, Any] = None,
                 custom_metrics: List[str] = None) -> Dict[str, Any]:
        """
        Evaluate using Custom GPT LLM-as-a-Judge with optimal parameters.
        
        Args:
            candidate_answer: The response to evaluate
            question: The original question
            user_profile: User's buyability profile data
            custom_metrics: Optional list of specific metrics to focus on
            
        Returns:
            Complete evaluation results with metadata
        """
        
        # Create evaluation prompt with custom GPT context
        evaluation_prompt = self._create_custom_gpt_prompt(
            candidate_answer, question, user_profile, custom_metrics
        )
        
        try:
            print(f"🤖 Custom GPT Judge using {self.model}")
            print(f"⚙️  Parameters: temp={self.judge_params['temperature']}, seed={self.judge_params['seed']}")
            
            start_time = time.time()
            
            # Make API call with optimal judge parameters
            if self.model.startswith("o1"):
                # o1 models don't support all parameters
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": evaluation_prompt}
                    ],
                    seed=self.judge_params["seed"]
                )
            else:
                # GPT-4 and other models with full parameter support
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.custom_instructions},
                        {"role": "user", "content": evaluation_prompt}
                    ],
                    **self.judge_params
                )
            
            evaluation_time = time.time() - start_time
            
            # Extract evaluation results
            llm_evaluation = response.choices[0].message.content
            
            # Parse and structure results
            structured_results = self._parse_evaluation_response(llm_evaluation)
            
            # Add metadata
            evaluation_metadata = {
                "model": self.model,
                "evaluation_method": "Custom GPT LLM-as-a-Judge",
                "parameters_used": self.judge_params,
                "evaluation_time_seconds": round(evaluation_time, 2),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "tokens_used": response.usage.total_tokens if hasattr(response, 'usage') else "N/A",
                "deterministic": True,  # Due to temperature=0 and seed=42
                "custom_instructions_applied": True
            }
            
            return {
                "evaluation_results": llm_evaluation,
                "structured_scores": structured_results,
                "metadata": evaluation_metadata,
                "input_data": {
                    "question": question,
                    "candidate_answer": candidate_answer,
                    "user_profile": user_profile
                }
            }
            
        except Exception as e:
            error_msg = f"❌ Custom GPT evaluation failed: {str(e)}"
            print(error_msg)
            return self._create_error_response(error_msg, candidate_answer, question)
    
    def _create_custom_gpt_prompt(self, 
                                  candidate_answer: str, 
                                  question: str, 
                                  user_profile: Dict[str, Any],
                                  custom_metrics: List[str] = None) -> str:
        """Create evaluation prompt for Custom GPT LLM Judge."""
        
        # Include knowledge base context
        knowledge_context = self._format_knowledge_base()
        
        # Determine which metrics to evaluate
        metrics_to_evaluate = custom_metrics or [
            "Personalization Accuracy", "Context based Personalization", 
            "Next Step Identification", "Assumption Listing", "Assumption Trust",
            "Calculation Accuracy", "Faithfulness to Ground Truth", "Overall Accuracy",
            "Structured Presentation", "Coherence", "Completeness", "Fair Housing Classifier"
        ]
        
        prompt = f"""
As a Custom GPT specialized in Zillow home affordability evaluation, assess the following response using your expert knowledge and the provided evaluation framework.

EVALUATION INPUT:
Question: "{question}"
Candidate Answer: "{candidate_answer}"
User Profile: {user_profile}

KNOWLEDGE BASE CONTEXT:
{knowledge_context}

EVALUATION TASK:
Evaluate the candidate answer on these specific metrics: {', '.join(metrics_to_evaluate)}

EVALUATION INSTRUCTIONS:
Follow these exact criteria for each metric:

1. **Personalization Accuracy** (Accurate/Inaccurate): 
   - Assess if personalization improves answer quality
   - Use chain-of-thought reasoning
   - Check against buyability profile data
   - Explain how elements improve or fail to improve the answer

2. **Context-based Personalization** (1-5):
   - Identify ALL relevant customizations possible
   - Count present vs. relevant elements
   - Score: <20%=1, 20-40%=2, 40-60%=3, 60-80%=4, >80%=5
   - Provide 150+ word justification

3. **Next Step Identification** (Present/Not Present):
   - Look for explicit/implicit action guidance
   - Check for terms like apply, schedule, consult, get pre-approved
   - Provide 150+ word justification

4. **Assumption Listing** (True/False):
   - True = clearly states assumptions OR notes "no assumptions"
   - Reward transparency
   - Provide 150+ word justification

5. **Assumption Trust** (1-5):
   - Evaluate transparency and candor
   - Check if shortcomings are called out
   - Focus on user prompt context, not hypothetical errors
   - Provide 150+ word justification

6. **Calculation Accuracy** (True/False):
   - Verify math against buyability profile
   - If absent → True, but note validation gaps
   - Provide 150+ word justification

7. **Faithfulness to Ground Truth** (True/False):
   - Align with Zillow tools and golden responses
   - Don't penalize omissions, only inaccuracies
   - Provide 150+ word justification with citations

8. **Overall Accuracy** (True/False):
   - Holistic verdict: does response correctly answer the question?
   - Provide 150+ word justification

9. **Structured Presentation** (1-5):
   - 1=wall of text, 2=minimal structure, 3=clear headings/lists for ≥50%, 
   - 4=logical hierarchy, 5=outline-quality organization
   - Provide 150+ word justification

10. **Coherence** (True/False):
    - Check logical consistency, no contradictions, minimal repetition
    - Provide detailed justification

11. **Completeness** (1-5):
    - Estimate percentage of sub-questions answered
    - <20%=1, 20-40%=2, 40-60%=3, 60-80%=4, >80%=5
    - Provide 150+ word justification

12. **Fair Housing Classifier** (True/False):
    - Check for fair housing violations
    - True = no violations
    - Provide 150+ word justification

OUTPUT FORMAT:
| Metric | Score | Justification |
|--------|-------|---------------|
[Complete table with all metrics]

Calculate Alpha evaluation score (excluding Completeness and Structured Presentation).

###
        """
        
        return prompt
    
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
    
    def _parse_evaluation_response(self, llm_response: str) -> Dict[str, Any]:
        """Parse LLM evaluation response into structured data."""
        # Extract Alpha score if present
        alpha_score = None
        if "Alpha evaluation" in llm_response:
            import re
            alpha_match = re.search(r"Alpha evaluation[:\s]*(\d+(?:\.\d+)?)", llm_response)
            if alpha_match:
                alpha_score = float(alpha_match.group(1))
        
        return {
            "alpha_score": alpha_score,
            "response_length": len(llm_response),
            "contains_table": "|" in llm_response,
            "parsed_successfully": True
        }
    
    def _create_error_response(self, error_msg: str, candidate_answer: str, question: str) -> Dict[str, Any]:
        """Create error response when evaluation fails."""
        return {
            "error": error_msg,
            "evaluation_results": None,
            "metadata": {
                "evaluation_method": "Custom GPT LLM-as-a-Judge",
                "status": "failed",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "input_data": {
                "question": question,
                "candidate_answer": candidate_answer
            }
        }
    
    def batch_evaluate(self, samples: List[Dict[str, Any]], save_results: bool = True) -> Dict[str, Any]:
        """
        Evaluate multiple samples using Custom GPT Judge.
        
        Args:
            samples: List of evaluation samples
            save_results: Whether to save results to file
            
        Returns:
            Batch evaluation results with statistics
        """
        print(f"🔄 Starting batch evaluation of {len(samples)} samples...")
        
        results = []
        total_time = 0
        
        for i, sample in enumerate(samples, 1):
            print(f"📊 Evaluating sample {i}/{len(samples)}...")
            
            result = self.evaluate(
                candidate_answer=sample.get('candidate_answer', ''),
                question=sample.get('question', ''),
                user_profile=sample.get('user_profile', {})
            )
            
            results.append(result)
            if 'metadata' in result and 'evaluation_time_seconds' in result['metadata']:
                total_time += result['metadata']['evaluation_time_seconds']
        
        # Calculate batch statistics
        successful_evals = [r for r in results if 'error' not in r]
        alpha_scores = [r['structured_scores']['alpha_score'] 
                       for r in successful_evals 
                       if r['structured_scores']['alpha_score'] is not None]
        
        batch_stats = {
            "total_samples": len(samples),
            "successful_evaluations": len(successful_evals),
            "failed_evaluations": len(samples) - len(successful_evals),
            "average_alpha_score": sum(alpha_scores) / len(alpha_scores) if alpha_scores else None,
            "total_evaluation_time": round(total_time, 2),
            "average_time_per_sample": round(total_time / len(samples), 2) if samples else 0
        }
        
        batch_results = {
            "results": results,
            "statistics": batch_stats,
            "metadata": {
                "batch_evaluation_method": "Custom GPT LLM-as-a-Judge",
                "parameters_used": self.judge_params,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": self.model
            }
        }
        
        # Save results if requested
        if save_results:
            output_file = f"custom_gpt_batch_results_{int(time.time())}.json"
            with open(output_file, 'w') as f:
                json.dump(batch_results, f, indent=2, default=str)
            print(f"💾 Batch results saved to: {output_file}")
        
        return batch_results


def main():
    """Demo the Custom GPT LLM Judge with optimal parameters."""
    
    print("🤖 CUSTOM GPT LLM-AS-A-JUDGE")
    print("=" * 70)
    print("Optimized parameters for deterministic evaluation:")
    print("• temperature=0 (greedy decoding)")
    print("• seed=42 (reproducible results)")
    print("• max_tokens=750 (comprehensive responses)")
    print("• All randomness parameters disabled")
    print("=" * 70)
    
    # Initialize Custom GPT Judge
    judge = CustomGPTLLMJudge(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        model="gpt-4"  # Use gpt-4 for full parameter support
    )
    
    print(f"✅ Custom GPT Judge initialized")
    print(f"🤖 Model: {judge.model}")
    print(f"⚙️  Parameters: {judge.judge_params}")
    
    # Test evaluation
    test_answer = "you can afford to buy now"
    test_question = "can I afford to buy a home right now?"
    test_profile = {
        "annual_income": None,
        "monthly_debts": None,
        "down_payment": None,
        "credit_score": None
    }
    
    print(f"\n🔍 Running Custom GPT evaluation...")
    result = judge.evaluate(test_answer, test_question, test_profile)
    
    if 'error' not in result:
        print(f"\n📊 EVALUATION COMPLETED")
        print(f"⏱️  Time: {result['metadata']['evaluation_time_seconds']}s")
        print(f"🔢 Tokens: {result['metadata']['tokens_used']}")
        print(f"🎯 Alpha Score: {result['structured_scores']['alpha_score']}")
        print(f"\n📋 EVALUATION RESULTS:")
        print("=" * 70)
        print(result['evaluation_results'])
    else:
        print(f"\n❌ Evaluation failed: {result['error']}")


if __name__ == "__main__":
    main()