#!/usr/bin/env python3
"""
Additional test scenarios for the Databricks LLM Evaluator system.
"""

from test_system import TestLLMEvaluator, TEST_GOLDEN_RESPONSES, TEST_BUYABILITY_PROFILES, TEST_FAIR_HOUSING_RULES

def test_different_scenarios():
    """Test the system with different types of responses."""
    print("🧪 Testing Different Response Scenarios")
    print("=" * 50)
    
    # Initialize evaluator
    evaluator = TestLLMEvaluator(
        golden_responses_data=TEST_GOLDEN_RESPONSES,
        buyability_profiles_data=TEST_BUYABILITY_PROFILES,
        fair_housing_rules_data=TEST_FAIR_HOUSING_RULES
    )
    
    # Scenario 1: Highly personalized response
    print("\n📊 Scenario 1: Highly Personalized Response")
    print("-" * 40)
    
    personalized_answer = """
    Based on your BuyAbility score of 750 and your current financial profile, your monthly payment would be $2,500.
    
    Your specific details:
    • BuyAbility Score: 750 (Excellent tier)
    • Monthly Payment Capacity: $2,500
    • Interest Rate: 6.5%
    • Loan Amount: $400,000
    
    Next steps tailored to your profile:
    1. Apply for pre-approval with your 750 score
    2. Schedule a consultation to discuss your $400,000 loan
    3. Contact our mortgage specialist for your 6.5% rate
    
    Note: These calculations assume current market conditions and your specific credit profile.
    """
    
    results1 = evaluator.evaluate_response(
        candidate_answer=personalized_answer,
        question="What's my monthly payment?",
        user_profile={"buyability_score": 750, "monthly_payment": 2500, "interest_rate": 6.5, "loan_amount": 400000}
    )
    
    print(f"✅ Score: {results1['final_score']}/100")
    print("Key metrics:")
    for result in results1["evaluation_results"]:
        if result.metric in ["Personalization Accuracy", "Context-Based Personalization", "Next-Step Identification"]:
            print(f"  • {result.metric}: {result.score}")
    
    # Scenario 2: Generic response
    print("\n📊 Scenario 2: Generic Response")
    print("-" * 40)
    
    generic_answer = """
    A typical monthly payment for a mortgage depends on various factors including the loan amount, interest rate, and loan term.
    
    Generally, you can expect to pay between $1,500 and $3,000 per month for most home loans.
    
    To get more specific information, you should contact a mortgage lender or use an online calculator.
    """
    
    results2 = evaluator.evaluate_response(
        candidate_answer=generic_answer,
        question="What's my monthly payment?",
        user_profile={"buyability_score": 750, "monthly_payment": 2500, "interest_rate": 6.5}
    )
    
    print(f"✅ Score: {results2['final_score']}/100")
    print("Key metrics:")
    for result in results2["evaluation_results"]:
        if result.metric in ["Personalization Accuracy", "Context-Based Personalization", "Next-Step Identification"]:
            print(f"  • {result.metric}: {result.score}")
    
    # Scenario 3: Response with fair housing violations
    print("\n📊 Scenario 3: Response with Fair Housing Violations")
    print("-" * 40)
    
    problematic_answer = """
    This property is perfect for families with children and young professionals.
    
    The neighborhood is ideal for:
    • Families with children
    • Young professionals
    • Married couples
    • Retired individuals
    
    We recommend this area for working families and empty nesters.
    """
    
    results3 = evaluator.evaluate_response(
        candidate_answer=problematic_answer,
        question="Tell me about this neighborhood",
        user_profile={"buyability_score": 750, "monthly_payment": 2500}
    )
    
    print(f"✅ Score: {results3['final_score']}/100")
    print("Key metrics:")
    for result in results3["evaluation_results"]:
        if result.metric in ["Fair Housing Classifier", "Personalization Accuracy"]:
            print(f"  • {result.metric}: {result.score}")
    
    # Scenario 4: Well-structured response
    print("\n📊 Scenario 4: Well-Structured Response")
    print("-" * 40)
    
    structured_answer = """
    # Monthly Payment Analysis
    
    ## Your Current Profile
    - BuyAbility Score: 750
    - Monthly Payment Capacity: $2,500
    - Interest Rate: 6.5%
    
    ## Payment Breakdown
    | Component | Amount |
    |-----------|---------|
    | Principal & Interest | $2,100 |
    | Property Taxes | $250 |
    | Insurance | $150 |
    | **Total Monthly** | **$2,500** |
    
    ## Next Steps
    1. **Apply for Pre-approval** - Lock in your rate
    2. **Schedule Consultation** - Discuss loan options
    3. **Review Documents** - Prepare financial statements
    
    *Note: Calculations assume 30-year fixed-rate mortgage with 20% down payment.*
    """
    
    results4 = evaluator.evaluate_response(
        candidate_answer=structured_answer,
        question="What's my monthly payment?",
        user_profile={"buyability_score": 750, "monthly_payment": 2500, "interest_rate": 6.5}
    )
    
    print(f"✅ Score: {results4['final_score']}/100")
    print("Key metrics:")
    for result in results4["evaluation_results"]:
        if result.metric in ["Structured Presentation", "Personalization Accuracy", "Next-Step Identification"]:
            print(f"  • {result.metric}: {result.score}")
    
    # Scenario 5: Response with calculations
    print("\n📊 Scenario 5: Response with Calculations")
    print("-" * 40)
    
    calculation_answer = """
    Let me calculate your monthly payment:
    
    Loan Amount: $400,000
    Interest Rate: 6.5%
    Term: 30 years
    
    Monthly Payment Calculation:
    Principal & Interest: $2,528.27
    Property Taxes (1.2%): $400
    Insurance: $150
    
    Total Monthly Payment: $3,078.27
    
    This is above your $2,500 target, so you may need to:
    - Increase your down payment by $50,000 to reduce the loan to $350,000
    - Or look for properties in the $350,000 range
    """
    
    results5 = evaluator.evaluate_response(
        candidate_answer=calculation_answer,
        question="What's my monthly payment?",
        user_profile={"buyability_score": 750, "monthly_payment": 2500, "interest_rate": 6.5, "loan_amount": 400000}
    )
    
    print(f"✅ Score: {results5['final_score']}/100")
    print("Key metrics:")
    for result in results5["evaluation_results"]:
        if result.metric in ["Calculation Accuracy", "Personalization Accuracy", "Next-Step Identification"]:
            print(f"  • {result.metric}: {result.score}")
    
    # Summary comparison
    print("\n📊 SCENARIO COMPARISON")
    print("=" * 50)
    scenarios = [
        ("Highly Personalized", results1),
        ("Generic", results2),
        ("Fair Housing Issues", results3),
        ("Well-Structured", results4),
        ("With Calculations", results5)
    ]
    
    print("| Scenario | Score | Key Strengths | Key Weaknesses |")
    print("|----------|-------|---------------|----------------|")
    
    for name, results in scenarios:
        score = results['final_score']
        
        # Find strengths and weaknesses
        strengths = []
        weaknesses = []
        
        for result in results['evaluation_results']:
            if isinstance(result.score, bool) and result.score:
                strengths.append(result.metric)
            elif isinstance(result.score, str) and result.score in ["Accurate", "Present", "True"]:
                strengths.append(result.metric)
            elif isinstance(result.score, int) and result.score >= 4:
                strengths.append(result.metric)
            elif isinstance(result.score, bool) and not result.score:
                weaknesses.append(result.metric)
            elif isinstance(result.score, str) and result.score in ["Inaccurate", "Not-Present", "False"]:
                weaknesses.append(result.metric)
            elif isinstance(result.score, int) and result.score <= 2:
                weaknesses.append(result.metric)
        
        # Limit to top 2 each
        strengths = strengths[:2]
        weaknesses = weaknesses[:2]
        
        print(f"| {name} | {score}/100 | {', '.join(strengths)} | {', '.join(weaknesses)} |")
    
    print("\n🎉 All scenarios tested successfully!")

if __name__ == "__main__":
    test_different_scenarios()