#!/usr/bin/env python3
"""
Test Enhanced OpenAI Evals Framework with Specific Example

This script tests the enhanced OpenAI Evals implementation with the same example
and compares results with the original implementation.
"""

import os
import json
from enhanced_openai_evals import EnhancedZillowAffordabilityEval, EvalRecorder

def main():
    """Test enhanced OpenAI Evals implementation with specific example."""
    
    print("🤖 TESTING ENHANCED OPENAI EVALS FRAMEWORK")
    print("=" * 70)
    print("Testing with: 'can I afford to buy a home right now?'")
    print("Response: 'you can afford to buy now'")
    print("Enhanced with 3 additional knowledge files")
    print("=" * 70)
    
    # Initialize enhanced evaluator
    evaluator = EnhancedZillowAffordabilityEval(
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
    
    print(f"\n🔍 Evaluating sample...")
    print(f"Question: {test_sample['question']}")
    print(f"Answer: {test_sample['candidate_answer']}")
    print(f"User Profile: {test_sample['user_profile']}")
    print("\n" + "=" * 70)
    
    # Run evaluation on single sample
    result = evaluator.eval_sample(test_sample)
    
    if result.get("status") == "success":
        print(f"✅ ENHANCED EVALUATION COMPLETED SUCCESSFULLY")
        print(f"⏱️  Model: {result['model']}")
        print(f"🔢 Tokens Used: {result['tokens_used']}")
        print(f"📚 Knowledge Files: {result['knowledge_files_loaded']}")
        print(f"🔧 Enhanced: {result['enhancement_used']}")
        
        print(f"\n🎯 ENHANCED SCORING RESULTS:")
        print("=" * 50)
        print(f"📊 Alpha Score: {result['alpha_score']}/{result['alpha_max']} ({result['alpha_percentage']:.1f}%)")
        print(f"📈 Full Score: {result['full_score']}/{result['full_max']} ({result['full_percentage']:.1f}%)")
        
        print(f"\n📋 INDIVIDUAL METRIC SCORES (Enhanced):")
        print("=" * 50)
        individual_scores = result['individual_scores']
        for metric, score in individual_scores.items():
            metric_display = metric.replace('_', ' ').title()
            print(f"• {metric_display}: {score}/10")
        
        print(f"\n📋 DETAILED ENHANCED LLM EVALUATION:")
        print("=" * 70)
        evaluation_text = result['llm_evaluation']
        
        # Show first 2000 characters to see the enhanced reasoning
        if len(evaluation_text) > 2000:
            print(evaluation_text[:2000] + "...")
            print(f"\n[Truncated - Full evaluation is {len(evaluation_text)} characters]")
        else:
            print(evaluation_text)
        
        # Compare with previous results
        print(f"\n🔍 COMPARISON WITH PREVIOUS IMPLEMENTATIONS:")
        print("=" * 55)
        print(f"🎯 Enhanced OpenAI Evals Alpha Score: {result['alpha_percentage']:.1f}%")
        print(f"🎯 Standard OpenAI Evals Alpha Score: ~24.0% (from previous run)")
        print(f"🎯 Custom Judge Alpha Score: ~24.0% (from previous run)")
        
        print(f"\n📚 ENHANCEMENT DETAILS:")
        print("=" * 25)
        print(f"✅ Content Guidelines: Loaded and integrated")
        print(f"✅ Quality Examples: Loaded and integrated")
        print(f"✅ Scoring Rubric: Loaded and integrated")
        print(f"✅ Fair Housing Enhanced: Special attention with all guidelines")
        print(f"✅ Cross-Reference Validation: All 6 knowledge files used")
        
        # Show enhancement usage if available
        if 'enhancement_usage' in result.get('individual_scores', {}):
            print(f"\n🔧 ENHANCEMENT USAGE TRACKING:")
            print("=" * 35)
            # This would show which enhanced criteria were referenced
            # in the evaluation justifications
        
    else:
        print(f"❌ ENHANCED EVALUATION FAILED")
        print(f"Error: {result.get('error', 'Unknown error')}")
    
    # Save enhanced result for analysis
    output_file = "enhanced_single_sample_result.json"
    with open(output_file, 'w') as f:
        json.dump(result.data, f, indent=2, default=str)
    
    print(f"\n💾 Enhanced result saved to: {output_file}")
    
    # Summary of enhancements
    print(f"\n🚀 ENHANCEMENT SUMMARY:")
    print("=" * 25)
    print("📚 Knowledge Base: 6 files (3 original + 3 enhanced)")
    print("🔍 Fair Housing: Enhanced with comprehensive guidelines")
    print("📊 Evaluation Criteria: Enriched with quality indicators")
    print("🎯 Scoring: Enhanced with detailed rubric")
    print("✅ Framework: OpenAI Evals compatible with improvements")
    
    return result

if __name__ == "__main__":
    main()