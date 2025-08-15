#!/usr/bin/env python3
"""
Real-world test simulation for the Databricks LLM Evaluator system.
This shows how the system would work with actual user data and responses.
"""

from test_system import TestLLMEvaluator, TEST_GOLDEN_RESPONSES, TEST_BUYABILITY_PROFILES, TEST_FAIR_HOUSING_RULES

def test_real_world_scenario():
    """Test with a realistic real-world scenario."""
    print("🌍 Real-World LLM Evaluation Test")
    print("=" * 50)
    
    # Initialize evaluator with test data (simulating your uploaded files)
    evaluator = TestLLMEvaluator(
        golden_responses_data=TEST_GOLDEN_RESPONSES,
        buyability_profiles_data=TEST_BUYABILITY_PROFILES,
        fair_housing_rules_data=TEST_FAIR_HOUSING_RULES
    )
    
    # Simulate a real user scenario
    print("\n👤 User Profile:")
    user_profile = {
        "buyability_score": 750,
        "monthly_payment": 2500,
        "interest_rate": 6.5,
        "loan_amount": 400000,
        "credit_score": 720,
        "annual_income": 90000,
        "monthly_debts": 200,
        "location": "Georgia"
    }
    
    for key, value in user_profile.items():
        print(f"  • {key}: {value}")
    
    # Simulate the user's question
    question = "What's my monthly payment and how can I improve my BuyAbility score?"
    
    print(f"\n❓ User Question: {question}")
    
    # Simulate an LLM response (this is what you'd evaluate)
    llm_response = """
    Based on your BuyAbility score of 750 and financial profile, here's your personalized analysis:
    
    ## Monthly Payment Breakdown
    Your projected monthly payment: **$2,500**
    
    **Components:**
    - Principal & Interest: $2,100
    - Property Taxes: $250
    - Homeowners Insurance: $150
    
    **Your Profile Details:**
    - BuyAbility Score: 750 (Excellent tier)
    - Loan Amount: $400,000
    - Interest Rate: 6.5%
    - Credit Score: 720
    
    ## Improving Your BuyAbility Score
    Your current score of 750 is already excellent, but here are ways to reach 800+:
    
    1. **Reduce Credit Utilization** (currently at 25%)
    2. **Maintain Payment History** (you're doing great here)
    3. **Avoid New Credit Applications** for 6 months
    
    ## Next Steps
    1. **Apply for Pre-approval** - Lock in your 6.5% rate
    2. **Schedule Consultation** - Discuss loan options
    3. **Review Credit Report** - Ensure accuracy
    
    *Note: Calculations assume 30-year fixed-rate mortgage with 20% down payment. Your strong credit profile qualifies you for competitive rates.*
    """
    
    print(f"\n🤖 LLM Response Length: {len(llm_response)} characters")
    print("Response Preview:", llm_response[:200] + "...")
    
    # Run the evaluation
    print("\n🔍 Running Evaluation...")
    print("-" * 30)
    
    try:
        results = evaluator.evaluate_response(
            candidate_answer=llm_response,
            question=question,
            user_profile=user_profile
        )
        
        print("✅ Evaluation completed successfully!")
        
        # Display comprehensive results
        print("\n📊 COMPREHENSIVE EVALUATION RESULTS")
        print("=" * 60)
        
        # Show the formatted table
        table = evaluator.format_evaluation_table(results)
        print(table)
        
        # Show detailed analysis
        print("\n🔍 DETAILED ANALYSIS")
        print("-" * 30)
        
        # Group results by performance
        excellent_metrics = []
        good_metrics = []
        needs_improvement = []
        
        for result in results["evaluation_results"]:
            if isinstance(result.score, bool):
                if result.score:
                    excellent_metrics.append(f"{result.metric} (True)")
                else:
                    needs_improvement.append(f"{result.metric} (False)")
            elif isinstance(result.score, str):
                if result.score in ["Accurate", "Present", "True"]:
                    excellent_metrics.append(f"{result.metric} ({result.score})")
                else:
                    needs_improvement.append(f"{result.metric} ({result.score})")
            elif isinstance(result.score, int):
                if result.score >= 4:
                    excellent_metrics.append(f"{result.metric} ({result.score}/5)")
                elif result.score >= 3:
                    good_metrics.append(f"{result.metric} ({result.score}/5)")
                else:
                    needs_improvement.append(f"{result.metric} ({result.score}/5)")
        
        print("🏆 Excellent Performance:")
        for metric in excellent_metrics:
            print(f"  ✅ {metric}")
        
        print("\n👍 Good Performance:")
        for metric in good_metrics:
            print(f"  ⚠️  {metric}")
        
        print("\n📈 Areas for Improvement:")
        for metric in needs_improvement:
            print(f"  ❌ {metric}")
        
        # Show scratchpad (internal reasoning)
        print("\n📝 INTERNAL REASONING (Scratchpad)")
        print("-" * 40)
        scratchpad = results["scratchpad"]
        print(f"Generated {len(scratchpad)} characters of internal reasoning")
        print("First 300 characters:")
        print(scratchpad[:300] + "...")
        
        # Show score breakdown
        print("\n🎯 SCORE BREAKDOWN")
        print("-" * 20)
        final_score = results["final_score"]
        
        if final_score >= 80:
            grade = "A (Excellent)"
            color = "🟢"
        elif final_score >= 70:
            grade = "B (Good)"
            color = "🟡"
        elif final_score >= 60:
            grade = "C (Average)"
            color = "🟠"
        else:
            grade = "D (Needs Improvement)"
            color = "🔴"
        
        print(f"{color} Final Score: {final_score}/100")
        print(f"Grade: {grade}")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS")
        print("-" * 20)
        
        if final_score < 70:
            print("🔴 This response needs significant improvement:")
            print("  • Increase personalization using user profile data")
            print("  • Add more specific next steps")
            print("  • Improve structure and formatting")
        elif final_score < 80:
            print("🟡 This response is good but could be improved:")
            print("  • Enhance personalization further")
            print("  • Add more detailed assumptions")
            print("  • Improve ground truth alignment")
        else:
            print("🟢 This is an excellent response!")
            print("  • Maintain current quality standards")
            print("  • Consider minor refinements for perfection")
        
        # Show what the system detected
        print("\n🔍 WHAT THE SYSTEM DETECTED")
        print("-" * 30)
        
        # Extract variables found
        extracted_vars = evaluator._extract_variables(llm_response)
        print("Variables extracted from response:")
        for key, value in extracted_vars.items():
            print(f"  • {key}: {value}")
        
        # Profile matching
        profile_match = evaluator._analyze_profile_match(extracted_vars, user_profile)
        print(f"\nProfile matching analysis: {len(profile_match)} matches found")
        
        # Fair housing check
        violations = evaluator._check_fair_housing_violations(llm_response)
        if violations:
            print(f"\n⚠️  Fair housing concerns detected: {len(violations)}")
            for violation in violations:
                print(f"  • {violation}")
        else:
            print("\n✅ No fair housing violations detected")
        
        print(f"\n🎉 Real-world evaluation completed! Score: {final_score}/100")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_real_world_scenario()