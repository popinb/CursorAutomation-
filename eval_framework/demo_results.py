#!/usr/bin/env python3
"""
Demonstration of the evaluation framework results with 4 metrics
"""

import json
from datetime import datetime

# Simulated evaluation results for demonstration
def create_demo_results():
    """Create demonstration results showing how the framework works with 4 metrics"""
    
    # Sample data
    samples = [
        {
            "prompt": "What's the best mortgage rate I can get?",
            "response": "Based on your excellent credit score of 780 and 20% down payment, you could qualify for rates around 6.5% for a 30-year fixed mortgage.",
            "user_info": "credit_score: 780, down_payment: 20%, location: California"
        },
        {
            "prompt": "How much house can I afford?",
            "response": "With your annual income of $120,000 and current monthly debts of $500, you could afford a home around $450,000-500,000.",
            "user_info": "income: $120,000, monthly_debts: $500, location: Texas"
        },
        {
            "prompt": "Should I refinance my mortgage?",
            "response": "Given that your current rate is 7.5% and market rates are now around 6.5%, refinancing could save you approximately $300 per month.",
            "user_info": "current_rate: 7.5%, loan_balance: $350,000"
        },
        {
            "prompt": "What are closing costs like in my area?",
            "response": "In New York, typical closing costs range from 2-5% of the home price, so for a $600,000 home, expect $12,000-30,000.",
            "user_info": "location: New York, home_price: $600,000"
        },
        {
            "prompt": "Can I get a loan with my credit score?",
            "response": "With a credit score of 620, you can still qualify for an FHA loan with as little as 3.5% down, though rates will be higher than conventional loans.",
            "user_info": "credit_score: 620"
        }
    ]
    
    # Evaluation results for 4 metrics
    results = {
        "personalization_accuracy": {
            "scores": [1, 1, 1, 1, 0],
            "threshold": 1,
            "explanations": [
                "All personal details (credit score 780, 20% down payment, California location) are accurately used.",
                "Income $120,000 and monthly debts $500 are correctly referenced.",
                "Current rate 7.5% is accurately mentioned.",
                "New York location and $600,000 home price are correct.",
                "Response mentions general info but doesn't use the specific 620 credit score incorrectly."
            ]
        },
        "relevance": {
            "scores": [5, 5, 4, 5, 5],
            "threshold": 4,
            "explanations": [
                "Directly answers the mortgage rate question with specific numbers.",
                "Perfectly addresses affordability with specific price range.",
                "Answers refinancing question but could mention closing costs.",
                "Directly provides closing cost ranges for the area.",
                "Clearly explains loan options for the credit score."
            ]
        },
        "advice_quality": {
            "scores": [4, 5, 4, 3, 4],
            "threshold": 3,
            "explanations": [
                "Good personalized rate estimate based on credit and down payment.",
                "Excellent calculation considering income and debts.",
                "Sound advice with specific monthly savings estimate.",
                "Accurate but could be more specific to user's situation.",
                "Good advice about FHA loans for lower credit scores."
            ]
        },
        "actionability": {
            "scores": [3, 4, 5, 4, 5],
            "threshold": 3,
            "explanations": [
                "Provides rate but no clear next steps for getting it.",
                "Clear price range, could add lender recommendations.",
                "Specific savings amount and implies next step is to refinance.",
                "Specific cost range, could mention how to prepare for it.",
                "Clear loan type and down payment requirement."
            ]
        }
    }
    
    return samples, results


def display_results():
    """Display the evaluation results in a formatted way"""
    
    samples, results = create_demo_results()
    
    print("="*80)
    print("EVALUATION FRAMEWORK DEMONSTRATION - 4 METRICS")
    print("="*80)
    
    print("\n📊 CONFIGURATION SUMMARY:")
    print("-"*40)
    print("• Ground Truth Data: 5 mortgage/finance Q&A samples")
    print("• Judge Model: gpt-4o (simulated for demo)")
    print("• Metrics: 4 custom evaluation criteria")
    
    print("\n📏 METRICS DEFINED:")
    print("-"*40)
    for i, (metric, data) in enumerate(results.items(), 1):
        print(f"\n{i}. {metric.replace('_', ' ').title()}")
        print(f"   - Threshold: {data['threshold']}")
        print(f"   - Type: {'Binary (0/1)' if data['threshold'] == 1 else f'Scale (1-5)'}")
    
    print("\n\n📈 EVALUATION RESULTS:")
    print("="*80)
    
    # Summary statistics
    print("\nSUMMARY STATISTICS:")
    print("-"*40)
    
    for metric, data in results.items():
        scores = data['scores']
        threshold = data['threshold']
        
        mean_score = sum(scores) / len(scores)
        passed = sum(1 for s in scores if s >= threshold)
        pass_rate = (passed / len(scores)) * 100
        
        print(f"\n{metric.replace('_', ' ').title()}:")
        print(f"  • Mean Score: {mean_score:.2f}")
        print(f"  • Range: {min(scores)} - {max(scores)}")
        print(f"  • Pass Rate: {pass_rate:.0f}% ({passed}/{len(scores)} passed)")
        print(f"  • Threshold: ≥{threshold}")
    
    # Detailed results table
    print("\n\nDETAILED RESULTS BY SAMPLE:")
    print("-"*80)
    print(f"{'Sample':<8} {'Question':<35} {'PA':<4} {'REL':<4} {'AQ':<4} {'ACT':<4} {'Status'}")
    print("-"*80)
    
    for i, sample in enumerate(samples):
        question = sample['prompt'][:32] + "..." if len(sample['prompt']) > 35 else sample['prompt']
        
        # Get scores for this sample
        pa_score = results['personalization_accuracy']['scores'][i]
        rel_score = results['relevance']['scores'][i]
        aq_score = results['advice_quality']['scores'][i]
        act_score = results['actionability']['scores'][i]
        
        # Check if all pass thresholds
        all_pass = (
            pa_score >= results['personalization_accuracy']['threshold'] and
            rel_score >= results['relevance']['threshold'] and
            aq_score >= results['advice_quality']['threshold'] and
            act_score >= results['actionability']['threshold']
        )
        
        status = "✅ Pass" if all_pass else "❌ Fail"
        
        print(f"{i+1:<8} {question:<35} {pa_score:<4} {rel_score:<4} {aq_score:<4} {act_score:<4} {status}")
    
    print("\nLegend: PA=Personalization Accuracy, REL=Relevance, AQ=Advice Quality, ACT=Actionability")
    
    # Failed cases analysis
    print("\n\n❌ FAILED CASES ANALYSIS:")
    print("-"*40)
    
    failed_found = False
    for metric, data in results.items():
        failed_indices = [i for i, score in enumerate(data['scores']) if score < data['threshold']]
        
        if failed_indices:
            failed_found = True
            print(f"\n{metric.replace('_', ' ').title()} (threshold: {data['threshold']}):")
            
            for idx in failed_indices:
                print(f"\n  Sample {idx+1}: {samples[idx]['prompt'][:50]}...")
                print(f"  Score: {data['scores'][idx]}")
                print(f"  Issue: {data['explanations'][idx]}")
    
    if not failed_found:
        print("\nNo failures found across all metrics!")
    
    # MLflow tracking info
    print("\n\n🔬 MLFLOW TRACKING:")
    print("-"*40)
    print("In a real run, the following would be tracked in MLflow:")
    print("  • All metric scores and aggregations")
    print("  • Configuration parameters")
    print("  • Detailed evaluation results")
    print("  • Visualizations and artifacts")
    print("\nRun `mlflow ui` to view the dashboard")
    
    # Save demo results
    output = {
        "timestamp": datetime.now().isoformat(),
        "total_samples": len(samples),
        "metrics_summary": {},
        "detailed_results": []
    }
    
    for metric, data in results.items():
        scores = data['scores']
        output["metrics_summary"][metric] = {
            "mean": sum(scores) / len(scores),
            "min": min(scores),
            "max": max(scores),
            "pass_rate": sum(1 for s in scores if s >= data['threshold']) / len(scores),
            "threshold": data['threshold']
        }
    
    for i, sample in enumerate(samples):
        result = {
            "sample_id": i + 1,
            "prompt": sample["prompt"],
            "scores": {
                metric: data["scores"][i] 
                for metric, data in results.items()
            }
        }
        output["detailed_results"].append(result)
    
    with open("demo_evaluation_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print("\n\n💾 Results saved to: demo_evaluation_results.json")
    print("="*80)


if __name__ == "__main__":
    display_results()