#!/usr/bin/env python3
"""
Test the evaluation framework with 4 metrics
"""

import os
import sys
import asyncio
from pathlib import Path
import pandas as pd
import json

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# For testing, we'll mock the OpenAI API calls
from unittest.mock import Mock, patch, AsyncMock


def mock_llm_responses(metric_name, prompt_content):
    """Generate mock responses for testing without API calls"""
    
    # Simulate different scores for different samples
    sample_responses = {
        "personalization_accuracy": [
            {"personalization_accuracy_score": 1, "explanation": "All personal details (credit score 780, 20% down payment, California location) are accurately used."},
            {"personalization_accuracy_score": 1, "explanation": "Income $120,000 and monthly debts $500 are correctly referenced."},
            {"personalization_accuracy_score": 1, "explanation": "Current rate 7.5% is accurately mentioned."},
            {"personalization_accuracy_score": 1, "explanation": "New York location and $600,000 home price are correct."},
            {"personalization_accuracy_score": 0, "explanation": "Credit score mentioned as 620 but response talks about 650 score."}
        ],
        "relevance": [
            {"relevance_score": 5, "explanation": "Directly answers the mortgage rate question with specific numbers."},
            {"relevance_score": 5, "explanation": "Perfectly addresses affordability with specific price range."},
            {"relevance_score": 4, "explanation": "Answers refinancing question but could mention closing costs."},
            {"relevance_score": 5, "explanation": "Directly provides closing cost ranges for the area."},
            {"relevance_score": 5, "explanation": "Clearly explains loan options for the credit score."}
        ],
        "advice_quality": [
            {"advice_quality_score": 4, "explanation": "Good personalized rate estimate based on credit and down payment."},
            {"advice_quality_score": 5, "explanation": "Excellent calculation considering income and debts."},
            {"advice_quality_score": 4, "explanation": "Sound advice with specific monthly savings estimate."},
            {"advice_quality_score": 3, "explanation": "Accurate but could be more specific to user's situation."},
            {"advice_quality_score": 4, "explanation": "Good advice about FHA loans for lower credit scores."}
        ],
        "actionability": [
            {"actionability_score": 3, "explanation": "Provides rate but no next steps for getting it."},
            {"actionability_score": 4, "explanation": "Clear price range, could add lender recommendations."},
            {"actionability_score": 5, "explanation": "Specific savings amount and implies next step is to refinance."},
            {"actionability_score": 4, "explanation": "Specific cost range, could mention how to prepare for it."},
            {"actionability_score": 5, "explanation": "Clear loan type and down payment requirement."}
        ]
    }
    
    # Return responses in order for each metric
    if metric_name not in sample_responses:
        return {"error": "Unknown metric"}
    
    # Use a counter to return different responses for each call
    if not hasattr(mock_llm_responses, 'counters'):
        mock_llm_responses.counters = {}
    
    if metric_name not in mock_llm_responses.counters:
        mock_llm_responses.counters[metric_name] = 0
    
    idx = mock_llm_responses.counters[metric_name]
    mock_llm_responses.counters[metric_name] = (idx + 1) % len(sample_responses[metric_name])
    
    return sample_responses[metric_name][idx]


async def test_framework():
    """Test the evaluation framework"""
    
    print("="*60)
    print("TESTING EVALUATION FRAMEWORK WITH 4 METRICS")
    print("="*60)
    
    # Import after sys.path is set
    from evaluation_framework import EvaluationPipeline, LLMEvaluator
    
    # Mock the LLM calls
    async def mock_evaluate(self, prompt, response, **kwargs):
        # Extract metric name from self
        metric_name = self.metric.name
        
        # Get mock response
        mock_response = mock_llm_responses(metric_name, prompt)
        
        return mock_response
    
    # Patch the evaluate method
    with patch.object(LLMEvaluator, 'evaluate', mock_evaluate):
        
        try:
            # Create pipeline
            print("\n1. Loading configuration...")
            pipeline = EvaluationPipeline("test_config.yaml")
            print(f"   ✓ Loaded {len(pipeline.metrics)} metrics")
            
            # Load data
            print("\n2. Loading ground truth data...")
            df = pipeline.data_loader.load_all_data()
            print(f"   ✓ Loaded {len(df)} samples from {df['source_file'].iloc[0]}")
            
            # Show sample data
            print("\n3. Sample data:")
            print(df[['prompt', 'response']].head(2).to_string(index=False, max_colwidth=50))
            
            # Run evaluation
            print("\n4. Running evaluation with 4 metrics...")
            print("   - personalization_accuracy")
            print("   - relevance") 
            print("   - advice_quality")
            print("   - actionability")
            
            results = await pipeline._evaluate_all(df)
            
            # Display results
            print("\n5. EVALUATION RESULTS")
            print("="*60)
            
            # Get metric columns
            metric_cols = ['personalization_accuracy', 'relevance', 'advice_quality', 'actionability']
            
            # Summary statistics
            print("\nSummary Statistics:")
            print("-"*40)
            for metric in metric_cols:
                if metric in results.columns:
                    scores = results[metric]
                    threshold = next(m.threshold for m in pipeline.metrics if m.name == metric)
                    pass_rate = (scores >= threshold).mean() * 100
                    
                    print(f"\n{metric}:")
                    print(f"  Mean Score: {scores.mean():.2f}")
                    print(f"  Min/Max: {scores.min():.2f} / {scores.max():.2f}")
                    print(f"  Pass Rate: {pass_rate:.1f}% (threshold: {threshold})")
            
            # Detailed results table
            print("\n\nDetailed Results:")
            print("-"*40)
            display_cols = ['prompt'] + metric_cols + [f"{m}_status" for m in metric_cols]
            display_cols = [col for col in display_cols if col in results.columns]
            
            # Truncate prompt for display
            results_display = results[display_cols].copy()
            results_display['prompt'] = results_display['prompt'].str[:40] + '...'
            
            print(results_display.to_string(index=False))
            
            # Failed cases analysis
            print("\n\nFailed Cases Analysis:")
            print("-"*40)
            for metric in metric_cols:
                status_col = f"{metric}_status"
                if status_col in results.columns:
                    failed_count = (results[status_col] == "❌").sum()
                    if failed_count > 0:
                        print(f"\n{metric}: {failed_count} failed cases")
                        failed_df = results[results[status_col] == "❌"]
                        for idx, row in failed_df.iterrows():
                            print(f"  - Sample {idx}: {row[metric]:.1f} (threshold: {next(m.threshold for m in pipeline.metrics if m.name == metric)})")
                            if f"{metric}_details" in row:
                                print(f"    Reason: {row[f'{metric}_details'].get('explanation', 'N/A')}")
            
            # Save results
            print("\n6. Saving results...")
            output_dir = Path("test_results")
            output_dir.mkdir(exist_ok=True)
            
            results_file = output_dir / "test_evaluation_results.csv"
            results.to_csv(results_file, index=False)
            print(f"   ✓ Results saved to {results_file}")
            
            # Save summary
            summary = {
                "total_samples": len(results),
                "metrics": {}
            }
            
            for metric in metric_cols:
                if metric in results.columns:
                    scores = results[metric]
                    summary["metrics"][metric] = {
                        "mean": float(scores.mean()),
                        "std": float(scores.std()),
                        "min": float(scores.min()),
                        "max": float(scores.max()),
                        "pass_rate": float((scores >= next(m.threshold for m in pipeline.metrics if m.name == metric)).mean())
                    }
            
            summary_file = output_dir / "test_evaluation_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"   ✓ Summary saved to {summary_file}")
            
            print("\n" + "="*60)
            print("✅ TEST COMPLETED SUCCESSFULLY!")
            print("="*60)
            
            return results
            
        except Exception as e:
            print(f"\n❌ Error during test: {e}")
            import traceback
            traceback.print_exc()
            return None


if __name__ == "__main__":
    # Run the test
    results = asyncio.run(test_framework())