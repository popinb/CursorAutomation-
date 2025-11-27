#!/usr/bin/env python3
"""
Test OpenAI Evals Framework with Specific Example

This script tests the OpenAI Evals implementation with the same example
used in the custom LLM judge: "can I afford to buy a home right now?"
"""

import os
import json
from standalone_openai_evals import ZillowAffordabilityEval, EvalRecorder

def main():
    """Test OpenAI Evals implementation with specific example."""
    
    print("🤖 TESTING OPENAI EVALS FRAMEWORK")
    print("=" * 60)
    print("Testing with: 'can I afford to buy a home right now?'")
    print("Response: 'you can afford to buy now'")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = ZillowAffordabilityEval(
        model="gpt-4",
        api_key=os.getenv("OPENAI_API_KEY", "popinb_zillowlabs__hs7x0vTjbLwjKhNStdgL1Dd"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.zillowlabs.com/openai/v1"),
        seed=42,
        temperature=0,
        max_tokens=4000
    )
    
    # Create the specific test sample
    test_sample = {
        "question": "can I afford to buy a home right now?",
        "candidate_answer": "you can afford to buy now",
        "user_profile": {
            "annual_income": None,
            "monthly_debts": None,
            "down_payment": None,
            "credit_score": None
        }
    }
    
    print(f"🔍 Evaluating sample...")
    print(f"Question: {test_sample['question']}")
    print(f"Answer: {test_sample['candidate_answer']}")
    print(f"User Profile: {test_sample['user_profile']}")
    print("\n" + "=" * 60)
    
    # Run evaluation on single sample
    result = evaluator.eval_sample(test_sample)
    
    if result.get("status") == "success":
        print(f"✅ EVALUATION COMPLETED SUCCESSFULLY")
        print(f"⏱️  Model: {result['model']}")
        print(f"🔢 Tokens Used: {result['tokens_used']}")
        
        print(f"\n🎯 SCORING RESULTS:")
        print("=" * 50)
        print(f"📊 Alpha Score: {result['alpha_score']}/{result['alpha_max']} ({result['alpha_percentage']:.1f}%)")
        print(f"📈 Full Score: {result['full_score']}/{result['full_max']} ({result['full_percentage']:.1f}%)")
        
        print(f"\n📋 INDIVIDUAL METRIC SCORES:")
        print("=" * 50)
        individual_scores = result['individual_scores']
        for metric, score in individual_scores.items():
            metric_display = metric.replace('_', ' ').title()
            print(f"• {metric_display}: {score}/10")
        
        print(f"\n📋 DETAILED LLM EVALUATION:")
        print("=" * 70)
        print(result['llm_evaluation'])
        
        # Compare with expected custom LLM judge results
        print(f"\n🔍 COMPARISON WITH CUSTOM LLM JUDGE:")
        print("=" * 50)
        print(f"🎯 OpenAI Evals Alpha Score: {result['alpha_percentage']:.1f}%")
        print(f"🎯 Custom Judge Alpha Score: ~24.0% (from previous run)")
        print(f"📊 Framework: OpenAI Evals Compatible")
        print(f"📊 Evaluation Method: LLM-as-a-Judge (gpt-4)")
        print(f"📊 Deterministic: Yes (seed=42, temp=0)")
        
    else:
        print(f"❌ EVALUATION FAILED")
        print(f"Error: {result.get('error', 'Unknown error')}")
    
    # Save result for analysis
    output_file = "single_sample_openai_evals_result.json"
    with open(output_file, 'w') as f:
        json.dump(result.data, f, indent=2, default=str)
    
    print(f"\n💾 Result saved to: {output_file}")
    
    return result

if __name__ == "__main__":
    main()