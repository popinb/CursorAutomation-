"""
OpenAI Evals Framework Compatible Zillow Judge Evaluator

This module provides an evaluator that can be used directly with the OpenAI Evals framework.
"""

import json
from typing import Any, Dict, Optional
from zillow_judge_evaluator import ZillowJudgeEvaluator


class ZillowJudgeOAIEval:
    """
    OpenAI Evals compatible evaluator for Zillow Judge criteria.
    
    This class follows the OpenAI Evals interface pattern and can be used
    with the evals framework or standalone.
    """
    
    def __init__(self, eval_spec: Optional[Dict[str, Any]] = None):
        """
        Initialize the evaluator.
        
        Args:
            eval_spec: Optional evaluation specification from OpenAI Evals
        """
        self.eval_spec = eval_spec or {}
        self.judge_evaluator = ZillowJudgeEvaluator()
        
    def eval_sample(self, sample: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Evaluate a single sample using Zillow Judge criteria.
        
        Args:
            sample: Dictionary containing 'input' and expected 'answer' fields
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary with evaluation results
        """
        # Extract input components
        candidate_answer = sample.get('input', {}).get('candidate_answer', '')
        question = sample.get('input', {}).get('question', '')
        user_profile = sample.get('input', {}).get('user_profile', None)
        
        # Run evaluation
        evaluation_table = self.judge_evaluator.evaluate(
            candidate_answer=candidate_answer,
            question=question,
            user_profile=user_profile
        )
        
        # Parse results for structured output
        parsed_results = self._parse_evaluation_table(evaluation_table)
        
        # Calculate overall score (for OpenAI Evals compatibility)
        overall_score = self._calculate_overall_score(parsed_results)
        
        return {
            "score": overall_score,
            "evaluation_table": evaluation_table,
            "detailed_scores": parsed_results,
            "metadata": {
                "question": question,
                "user_profile_provided": user_profile is not None,
                "answer_length": len(candidate_answer.split())
            }
        }
    
    def _parse_evaluation_table(self, table: str) -> Dict[str, Any]:
        """Parse the evaluation table to extract individual metric scores."""
        lines = table.strip().split('\n')
        results = {}
        
        # Skip header and separator lines
        for line in lines[2:]:
            if '|' in line:
                parts = [part.strip() for part in line.split('|')]
                if len(parts) >= 4:  # metric, score, justification
                    metric = parts[1].lower().replace(' ', '_')
                    score = parts[2]
                    justification = parts[3]
                    
                    results[metric] = {
                        "score": score,
                        "justification": justification,
                        "numeric_score": self._convert_to_numeric(score)
                    }
        
        return results
    
    def _convert_to_numeric(self, score: str) -> float:
        """Convert various score types to numeric values for aggregation."""
        score_lower = score.lower().strip()
        
        # Handle binary scores
        if score_lower in ["accurate", "true", "present"]:
            return 1.0
        elif score_lower in ["inaccurate", "false", "not present"]:
            return 0.0
        
        # Handle numeric scores (1-5 scale)
        try:
            numeric = float(score)
            if 1 <= numeric <= 5:
                return numeric / 5.0  # Normalize to 0-1 scale
            return numeric
        except ValueError:
            return 0.5  # Default for unknown scores
    
    def _calculate_overall_score(self, parsed_results: Dict[str, Any]) -> float:
        """Calculate an overall score from individual metric scores."""
        if not parsed_results:
            return 0.0
        
        # Get all numeric scores
        scores = [data["numeric_score"] for data in parsed_results.values()]
        
        # Return average score
        return sum(scores) / len(scores) if scores else 0.0
    
    def run_eval(self, samples: list, **kwargs) -> Dict[str, Any]:
        """
        Run evaluation on multiple samples (for batch processing).
        
        Args:
            samples: List of sample dictionaries
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary with aggregated results
        """
        results = []
        total_score = 0.0
        
        for i, sample in enumerate(samples):
            try:
                result = self.eval_sample(sample, **kwargs)
                results.append({
                    "sample_id": i,
                    "result": result,
                    "status": "success"
                })
                total_score += result["score"]
                
            except Exception as e:
                results.append({
                    "sample_id": i,
                    "result": None,
                    "status": "error",
                    "error": str(e)
                })
        
        successful_evals = len([r for r in results if r["status"] == "success"])
        average_score = total_score / successful_evals if successful_evals > 0 else 0.0
        
        return {
            "average_score": average_score,
            "total_samples": len(samples),
            "successful_evaluations": successful_evals,
            "failed_evaluations": len(samples) - successful_evals,
            "individual_results": results
        }


def create_sample_data():
    """Create sample data for testing the evaluator."""
    return [
        {
            "input": {
                "candidate_answer": """
                Your personalized BuyAbility estimate is $318,431, based on your specific financial profile.
                
                This calculation uses your $90,000 annual income, $200 monthly debts, $18,000 down payment,
                and credit score range of 660-719. With your monthly income of $7,500, lenders typically
                recommend up to 36% DTI, giving you about $2,500 available for mortgage payments.
                
                Your monthly payment breaks down as:
                • Principal & Interest: $1,975  
                • Property Taxes: $212
                • Homeowners Insurance: $106
                • PMI: $207
                
                Based on this analysis, you should consider getting pre-approved to start your home search in Georgia.
                """,
                "question": "What factors were considered to calculate my Buyability?",
                "user_profile": {
                    "annual_income": 90000,
                    "monthly_debts": 200,
                    "down_payment": 18000,
                    "credit_score": "660-719"
                }
            }
        },
        {
            "input": {
                "candidate_answer": """
                Yes, your BuyAbility is personalized to you. We used your income and debts to calculate it.
                """,
                "question": "Is my Buyability personalized to me?",
                "user_profile": {
                    "annual_income": 100000,
                    "monthly_debts": 200,
                    "down_payment": 30000,
                    "credit_score": "good"
                }
            }
        }
    ]


def main():
    """Main function for testing the evaluator."""
    print("=== OpenAI Evals Compatible Zillow Judge Evaluator ===\n")
    
    # Create evaluator
    evaluator = ZillowJudgeOAIEval()
    
    # Create sample data
    samples = create_sample_data()
    
    print(f"Running evaluation on {len(samples)} samples...\n")
    
    # Run batch evaluation
    results = evaluator.run_eval(samples)
    
    # Print summary
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Average Score: {results['average_score']:.3f}")
    print(f"Total Samples: {results['total_samples']}")
    print(f"Successful Evaluations: {results['successful_evaluations']}")
    print(f"Failed Evaluations: {results['failed_evaluations']}")
    
    # Print detailed results for first sample
    if results['individual_results'] and results['individual_results'][0]['status'] == 'success':
        print("\nDETAILED RESULTS (Sample 1)")
        print("=" * 50)
        first_result = results['individual_results'][0]['result']
        print(first_result['evaluation_table'])


if __name__ == "__main__":
    main()