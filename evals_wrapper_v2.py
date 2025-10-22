"""
OpenAI Evals Framework Wrapper for Zillow Judge Evaluator (V2)
With integrated directory management and configuration support

This module provides a simple interface to run the Zillow Judge Evaluator
either standalone or as part of an evaluation pipeline, with flexible
directory and path management.
"""

from zillow_judge_evaluator import ZillowJudgeEvaluator
from directory_handler import DirectoryHandler, get_directory_handler
from typing import Dict, Any, List, Optional, Union
import json
import os
import time
from pathlib import Path


class ZillowEvalsWrapperV2:
    """
    Enhanced wrapper class with directory management for Zillow Judge Evaluator.
    Compatible with OpenAI Evals framework and provides standalone evaluation capabilities.
    """
    
    def __init__(self, config_path: Optional[str] = None, directory_handler: Optional[DirectoryHandler] = None):
        """
        Initialize the wrapper with the core evaluator and directory handler.
        
        Args:
            config_path: Path to configuration file
            directory_handler: Optional custom directory handler
        """
        self.evaluator = ZillowJudgeEvaluator()
        
        # Initialize directory handler
        if directory_handler:
            self.dir_handler = directory_handler
        else:
            self.dir_handler = get_directory_handler(config_path)
    
    def evaluate_response(
        self,
        candidate_answer: str,
        question: str = "",
        user_profile: Optional[Dict[str, Any]] = None,
        return_raw_scores: bool = False,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate a candidate response using the Zillow Judge criteria.
        
        Args:
            candidate_answer: The LLM response to evaluate
            question: The original question (optional)
            user_profile: User's buyability profile data (optional)
            return_raw_scores: If True, return parsed scores; if False, return formatted table
            save_results: Whether to save results to file
            
        Returns:
            Dictionary containing evaluation results
        """
        # Get the evaluation table
        evaluation_table = self.evaluator.evaluate(candidate_answer, question, user_profile)
        
        # Parse results
        parsed_results = self._parse_evaluation_table(evaluation_table)
        
        # Prepare result dictionary
        result = {
            "evaluation_table": evaluation_table,
            "summary": self._generate_summary(evaluation_table),
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "metadata": {
                "question": question,
                "user_profile_provided": user_profile is not None,
                "answer_length": len(candidate_answer.split())
            }
        }
        
        if return_raw_scores:
            result["raw_scores"] = parsed_results
        
        # Save results if requested
        if save_results:
            self._save_evaluation_result(result)
        
        return result
    
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
    
    def _save_evaluation_result(self, result: Dict[str, Any]) -> str:
        """Save evaluation result to file."""
        filename = self.dir_handler.get_timestamped_file_path('results', 'evaluation', '.json')
        
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2)
        
        return filename
    
    def evaluate_batch(
        self,
        test_cases: Union[List[Dict[str, Any]], str],
        save_summary: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Evaluate multiple test cases in batch.
        
        Args:
            test_cases: List of test case dictionaries or path to JSON file
            save_summary: Whether to save a summary report
            
        Returns:
            List of evaluation results
        """
        # Load test cases if path provided
        if isinstance(test_cases, str):
            test_cases = self._load_test_cases(test_cases)
        
        results = []
        start_time = time.time()
        
        print(f"Evaluating {len(test_cases)} test cases...")
        
        for i, test_case in enumerate(test_cases):
            try:
                answer = test_case.get('answer', '')
                question = test_case.get('question', '')
                profile = test_case.get('profile', None)
                
                evaluation = self.evaluate_response(
                    candidate_answer=answer,
                    question=question,
                    user_profile=profile,
                    return_raw_scores=True,
                    save_results=False  # We'll save batch results separately
                )
                
                results.append({
                    "test_case_id": i + 1,
                    "evaluation": evaluation,
                    "status": "success"
                })
                
                print(f"  ✓ Test case {i + 1} completed")
                
            except Exception as e:
                results.append({
                    "test_case_id": i + 1,
                    "evaluation": None,
                    "status": "error",
                    "error": str(e)
                })
                print(f"  ✗ Test case {i + 1} failed: {str(e)}")
        
        elapsed_time = time.time() - start_time
        
        # Generate and save summary
        if save_summary:
            summary_path = self._save_batch_summary(results, elapsed_time)
            print(f"\nBatch evaluation completed in {elapsed_time:.2f}s")
            print(f"Summary saved to: {summary_path}")
        
        return results
    
    def _load_test_cases(self, path: str) -> List[Dict[str, Any]]:
        """Load test cases from JSON file."""
        # If relative path, look in data directory
        if not os.path.isabs(path):
            path = self.dir_handler.get_file_path('data', path, create_dir=False)
        
        with open(path, 'r') as f:
            return json.load(f)
    
    def _save_batch_summary(self, results: List[Dict[str, Any]], elapsed_time: float) -> str:
        """Save batch evaluation summary."""
        successful = [r for r in results if r["status"] == "success"]
        failed = [r for r in results if r["status"] == "error"]
        
        # Calculate aggregate metrics
        all_summaries = [r["evaluation"]["summary"] for r in successful if r["evaluation"]]
        
        avg_accurate = sum(s["accurate_metrics"] for s in all_summaries) / len(all_summaries) if all_summaries else 0
        avg_inaccurate = sum(s["inaccurate_metrics"] for s in all_summaries) / len(all_summaries) if all_summaries else 0
        
        summary = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "total_test_cases": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "elapsed_time_seconds": elapsed_time,
            "average_accurate_metrics": avg_accurate,
            "average_inaccurate_metrics": avg_inaccurate,
            "detailed_results": results
        }
        
        # Save to results directory
        summary_path = self.dir_handler.get_timestamped_file_path('results', 'batch_summary', '.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Also generate HTML report
        html_path = self._generate_html_report(summary)
        
        return summary_path
    
    def _generate_html_report(self, summary: Dict[str, Any]) -> str:
        """Generate an HTML report from batch summary."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Zillow Judge Evaluation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        h1, h2 {{ color: #333; }}
        .summary-box {{ background-color: #e8f4f8; padding: 15px; margin: 20px 0; border-radius: 5px; }}
        .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #0066cc; }}
        .metric-label {{ font-size: 14px; color: #666; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .success {{ color: green; }}
        .error {{ color: red; }}
        .details {{ font-size: 12px; color: #666; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Zillow Judge Evaluation Report</h1>
        <p class="details">Generated: {summary['timestamp']}</p>
        
        <div class="summary-box">
            <h2>Summary</h2>
            <div class="metric">
                <div class="metric-value">{summary['total_test_cases']}</div>
                <div class="metric-label">Total Test Cases</div>
            </div>
            <div class="metric">
                <div class="metric-value">{summary['successful']}</div>
                <div class="metric-label">Successful</div>
            </div>
            <div class="metric">
                <div class="metric-value">{summary['failed']}</div>
                <div class="metric-label">Failed</div>
            </div>
            <div class="metric">
                <div class="metric-value">{summary['elapsed_time_seconds']:.2f}s</div>
                <div class="metric-label">Evaluation Time</div>
            </div>
        </div>
        
        <h2>Test Case Results</h2>
        <table>
            <tr>
                <th>Test Case ID</th>
                <th>Status</th>
                <th>Accurate Metrics</th>
                <th>Inaccurate Metrics</th>
                <th>Average Score</th>
            </tr>
"""
        
        for result in summary['detailed_results']:
            status_class = 'success' if result['status'] == 'success' else 'error'
            status_text = '✓ Success' if result['status'] == 'success' else '✗ Error'
            
            if result['status'] == 'success' and result['evaluation']:
                eval_summary = result['evaluation']['summary']
                accurate = eval_summary['accurate_metrics']
                inaccurate = eval_summary['inaccurate_metrics']
                avg_score = eval_summary['average_numeric_score'] or 'N/A'
                if isinstance(avg_score, float):
                    avg_score = f"{avg_score:.2f}"
            else:
                accurate = inaccurate = avg_score = 'N/A'
            
            html_content += f"""
            <tr>
                <td>{result['test_case_id']}</td>
                <td class="{status_class}">{status_text}</td>
                <td>{accurate}</td>
                <td>{inaccurate}</td>
                <td>{avg_score}</td>
            </tr>
"""
        
        html_content += """
        </table>
    </div>
</body>
</html>
"""
        
        # Save HTML report
        html_path = self.dir_handler.get_timestamped_file_path('reports', 'evaluation_report', '.html')
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        return html_path
    
    def run_example_evaluation(self):
        """Run an example evaluation to demonstrate the system."""
        print("=== Zillow Judge Evaluator V2 - Example Evaluation ===\n")
        
        # Show directory configuration
        print("Directory Configuration:")
        config = self.dir_handler.get_config_summary()
        for dir_type, path in config['directories'].items():
            print(f"  {dir_type}: {path}")
        print()
        
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
        
        print(f"\nResults saved to: {self.dir_handler.get_dir('results')}")


def main():
    """Main function for demonstration."""
    # Check if config file exists
    config_path = "config_example.json"
    if not os.path.exists(config_path):
        print(f"Note: Using default configuration. Copy {config_path} to config.json and customize as needed.")
        config_path = None
    
    wrapper = ZillowEvalsWrapperV2(config_path)
    wrapper.run_example_evaluation()


if __name__ == "__main__":
    main()