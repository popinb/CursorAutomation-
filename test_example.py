#!/usr/bin/env python3
"""
Test Example for Zillow Judge Evaluator

This script demonstrates how to use the evaluator in various scenarios:
1. Perfect response (high scores)
2. Poor response (low scores) 
3. Mixed quality response (medium scores)
"""

from zillow_judge_evaluator import ZillowJudgeEvaluator
from evals_wrapper import ZillowEvalsWrapper
from oai_evals_zillow_judge import ZillowJudgeOAIEval


def test_perfect_response():
    """Test a high-quality response that should score well."""
    print("=" * 80)
    print("TEST 1: High-Quality Response")
    print("=" * 80)
    
    evaluator = ZillowJudgeEvaluator()
    
    # Perfect response that matches the user profile exactly
    response = """
    Your personalized BuyAbility estimate is $318,431, calculated using your specific financial profile.
    
    Here's how we determined this based on your information:
    
    **Income and DTI Analysis:**
    Your annual income of $100,000 gives you a monthly income of $8,333. With your current monthly debts of $200, 
    lenders typically recommend keeping your total debt-to-income ratio below 36%, which means up to $3,000/month 
    could be allocated to housing expenses. After subtracting your existing debt, you have approximately $2,800 
    available for a mortgage payment.
    
    **Down Payment Impact:**
    Your $30,000 down payment represents a strong foundation. This amount allows you to:
    • Avoid PMI if purchasing a home under $150,000 (20% down)
    • Access conventional loans with as little as 3% down
    • Target homes up to $1,000,000 with a 3% down payment
    
    **Credit Score Benefits:**
    Your "good" credit score range should qualify you for competitive interest rates, typically between 6.5-7.5% 
    depending on current market conditions.
    
    **Next Steps:**
    1. Get pre-approved for a mortgage to confirm your buying power
    2. Schedule consultations with 2-3 lenders to compare rates
    3. Consider working with a real estate agent familiar with your target price range
    4. Start exploring neighborhoods within your budget
    
    **Important Assumptions:**
    - Interest rates are estimated based on current market averages
    - Property taxes and insurance costs vary by location
    - These estimates are for illustration purposes and may vary
    - Final loan approval depends on complete financial review
    
    This analysis is personalized to your specific financial situation and should give you confidence in your home-buying journey.
    """
    
    question = "What factors were considered to calculate my Buyability?"
    profile = {
        "annual_income": 100000,
        "monthly_debts": 200,
        "down_payment": 30000,
        "credit_score": "good",
        "preferred_monthly_payment": 1000,
        "comfortable_max_monthly_payment": 3967
    }
    
    result = evaluator.evaluate(response, question, profile)
    print(result)


def test_poor_response():
    """Test a low-quality response that should score poorly."""
    print("\n" + "=" * 80)
    print("TEST 2: Low-Quality Response")
    print("=" * 80)
    
    evaluator = ZillowJudgeEvaluator()
    
    # Poor response - generic, no personalization, no structure
    response = """
    People can generally afford houses. It depends on income and debts. Most people should get pre-approved. 
    You might be able to buy something. Check with a realtor.
    """
    
    question = "What factors were considered to calculate my Buyability?"
    profile = {
        "annual_income": 100000,
        "monthly_debts": 200,
        "down_payment": 30000,
        "credit_score": "good"
    }
    
    result = evaluator.evaluate(response, question, profile)
    print(result)


def test_batch_evaluation():
    """Test batch evaluation with multiple samples."""
    print("\n" + "=" * 80)
    print("TEST 3: Batch Evaluation")
    print("=" * 80)
    
    evaluator = ZillowJudgeOAIEval()
    
    samples = [
        {
            "input": {
                "candidate_answer": """
                Your BuyAbility of $318,431 is based on your $90,000 income, $200 monthly debts, 
                and $18,000 down payment. With a 36% DTI ratio, you can afford about $2,500/month 
                for housing. You should consider getting pre-approved.
                """,
                "question": "How was my Buyability calculated?",
                "user_profile": {"annual_income": 90000, "monthly_debts": 200, "down_payment": 18000}
            }
        },
        {
            "input": {
                "candidate_answer": """
                Your monthly payment includes several components:
                • Principal & Interest: $1,850
                • Property Taxes: $275
                • Homeowners Insurance: $125
                • PMI: $150 (since your down payment is below 20%)
                
                Total estimated monthly payment: $2,400
                """,
                "question": "What's included in my monthly payment?",
                "user_profile": {"annual_income": 75000, "monthly_debts": 300, "down_payment": 15000}
            }
        }
    ]
    
    results = evaluator.run_eval(samples)
    
    print(f"Batch Evaluation Results:")
    print(f"Average Score: {results['average_score']:.3f}")
    print(f"Successful Evaluations: {results['successful_evaluations']}/{results['total_samples']}")
    
    for i, result in enumerate(results['individual_results']):
        if result['status'] == 'success':
            score = result['result']['score']
            print(f"Sample {i+1}: {score:.3f}")


def main():
    """Run all tests."""
    print("Zillow Judge Evaluator - Comprehensive Test Suite")
    print("Testing different response qualities and scenarios...\n")
    
    # Test high-quality response
    test_perfect_response()
    
    # Test low-quality response  
    test_poor_response()
    
    # Test batch evaluation
    test_batch_evaluation()
    
    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()