"""
OpenAI Evals Framework Wrapper for Zillow Judge Evaluator

This module provides a simple interface to run the Zillow Judge Evaluator
either standalone or as part of an evaluation pipeline.
"""

from zillow_judge_evaluator import ZillowJudgeEvaluator
from typing import Dict, Any, List, Optional
import json


class ZillowEvalsWrapper:
    """
    Wrapper class to make Zillow Judge Evaluator compatible with
    OpenAI Evals framework and provide standalone evaluation capabilities.
    """
    
    def __init__(self):
        """Initialize the wrapper with the core evaluator."""
        self.evaluator = ZillowJudgeEvaluator()
    
    def evaluate_response(
        self,
        candidate_answer: str,
        question: str = "",
        user_profile: Optional[Dict[str, Any]] = None,
        return_raw_scores: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate a candidate response using the Zillow Judge criteria.
        
        Args:
            candidate_answer: The LLM response to evaluate
            question: The original question (optional)
            user_profile: User's buyability profile data (optional)
            return_raw_scores: If True, return parsed scores; if False, return formatted table
            
        Returns:
            Dictionary containing evaluation results
        """
        # Get the evaluation table
        evaluation_table = self.evaluator.evaluate(candidate_answer, question, user_profile)
        
        if return_raw_scores:
            # Parse the table to extract individual scores
            return self._parse_evaluation_table(evaluation_table)
        else:
            # Return the formatted table
            return {
                "evaluation_table": evaluation_table,
                "summary": self._generate_summary(evaluation_table)
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
                        "justification": justification
                    }
        
        return results
    
    def _generate_summary(self, table: str) -> Dict[str, Any]:
        """Generate a summary of the evaluation results."""
        parsed = self._parse_evaluation_table(table)
        
        # Count different score types
        accurate_count = 0
        inaccurate_count = 0
        numeric_scores = []
        
        for metric, data in parsed.items():
            score = data["score"]
            
            if score.lower() in ["accurate", "true", "present"]:
                accurate_count += 1
            elif score.lower() in ["inaccurate", "false", "not present"]:
                inaccurate_count += 1
            elif score.isdigit():
                numeric_scores.append(int(score))
        
        summary = {
            "total_metrics": len(parsed),
            "accurate_metrics": accurate_count,
            "inaccurate_metrics": inaccurate_count,
            "average_numeric_score": sum(numeric_scores) / len(numeric_scores) if numeric_scores else None,
            "numeric_score_range": f"{min(numeric_scores)}-{max(numeric_scores)}" if numeric_scores else None
        }
        
        return summary
    
    def evaluate_batch(
        self,
        test_cases: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Evaluate multiple test cases in batch.
        
        Args:
            test_cases: List of dictionaries containing 'answer', 'question', and 'profile' keys
            
        Returns:
            List of evaluation results
        """
        results = []
        
        for i, test_case in enumerate(test_cases):
            try:
                answer = test_case.get('answer', '')
                question = test_case.get('question', '')
                profile = test_case.get('profile', None)
                
                evaluation = self.evaluate_response(
                    candidate_answer=answer,
                    question=question,
                    user_profile=profile,
                    return_raw_scores=True
                )
                
                results.append({
                    "test_case_id": i + 1,
                    "evaluation": evaluation,
                    "status": "success"
                })
                
            except Exception as e:
                results.append({
                    "test_case_id": i + 1,
                    "evaluation": None,
                    "status": "error",
                    "error": str(e)
                })
        
        return results
    
    def run_example_evaluation(self):
        """Run an example evaluation to demonstrate the system."""
        print("=== Zillow Judge Evaluator - Example Evaluation ===\n")
        
        # Example test case
        test_answer = """
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
        """
        
        test_question = "What factors were considered to calculate my Buyability?"
        test_profile = {
            "annual_income": 90000,
            "monthly_debts": 200,
            "down_payment": 18000,
            "credit_score": "660-719"
        }
        
        print("Question:", test_question)
        print("\nCandidate Answer:")
        print(test_answer)
        print("\nUser Profile:", json.dumps(test_profile, indent=2))
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)
        
        # Get evaluation
        result = self.evaluate_response(test_answer, test_question, test_profile)
        
        print("\n" + result["evaluation_table"])
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        summary = result["summary"]
        print(f"Total Metrics Evaluated: {summary['total_metrics']}")
        print(f"Accurate/True/Present: {summary['accurate_metrics']}")
        print(f"Inaccurate/False/Not Present: {summary['inaccurate_metrics']}")
        if summary['average_numeric_score']:
            print(f"Average Numeric Score: {summary['average_numeric_score']:.1f}")
            print(f"Numeric Score Range: {summary['numeric_score_range']}")


def main():
    """Main function for demonstration."""
    wrapper = ZillowEvalsWrapper()
    wrapper.run_example_evaluation()


if __name__ == "__main__":
    main()