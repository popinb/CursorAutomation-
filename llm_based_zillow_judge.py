#!/usr/bin/env python3
"""
LLM-Based Zillow Judge Evaluator

This evaluator uses OpenAI's LLM API to perform evaluations following
the exact evaluation instructions, rather than rule-based logic.
"""

import json
import os
import re
from typing import Dict, Any, List, Optional
from pathlib import Path
import openai


class LLMBasedZillowJudge:
    """
    LLM-based evaluator that uses OpenAI API to assess responses
    according to the exact evaluation instructions provided.
    """
    
    def __init__(self, api_key: str = None, base_url: str = None, model: str = "o1-preview"):
        """
        Initialize the LLM-based evaluator.
        
        Args:
            api_key: OpenAI API key
            base_url: OpenAI base URL (for custom endpoints)
            model: Model to use (o1-preview, gpt-4, etc.)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.model = model
        
        # Initialize OpenAI client
        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
            
        self.client = openai.OpenAI(**client_kwargs)
        
        # Load ground truth data
        script_dir = Path(__file__).parent
        self.golden_responses = self._load_json(script_dir / "assets" / "golden_responses.json")
        self.buyability_profiles = self._load_json(script_dir / "assets" / "buyability_profiles.json")
        self.fair_housing_guide = self._load_json(script_dir / "assets" / "fair_housing_guide.json")
    
    def _load_json(self, filepath: Path) -> Dict[str, Any]:
        """Load JSON data from file."""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: {filepath} not found, continuing without it")
            return {}
        except json.JSONDecodeError as e:
            print(f"Warning: Invalid JSON in {filepath}: {e}")
            return {}
    
    def evaluate(self, candidate_answer: str, question: str = "", user_profile: Dict[str, Any] = None) -> str:
        """
        Main evaluation method using LLM API.
        
        Args:
            candidate_answer: The LLM response to evaluate
            question: The original question
            user_profile: User's buyability profile data
            
        Returns:
            Formatted evaluation table with scores and justifications
        """
        
        # Construct the evaluation prompt with exact instructions
        evaluation_prompt = self._create_evaluation_prompt(candidate_answer, question, user_profile)
        
        try:
            # Make API call to LLM
            print(f"🤖 Using {self.model} for LLM-based evaluation...")
            
            # Use different message format based on model
            if self.model.startswith("o1"):
                # o1 models use simplified message format
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": evaluation_prompt}
                    ]
                )
            else:
                # GPT-4 and other models use system + user messages
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a professional Zillow home affordability evaluation expert. Follow the evaluation instructions exactly as provided."},
                        {"role": "user", "content": evaluation_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=4000
                )
            
            # Get LLM evaluation
            llm_evaluation = response.choices[0].message.content
            
            # Parse and format the results
            formatted_results = self._format_llm_results(llm_evaluation, candidate_answer, question, user_profile)
            
            return formatted_results
            
        except Exception as e:
            error_msg = f"❌ LLM evaluation failed: {str(e)}"
            print(error_msg)
            return self._create_error_response(error_msg, candidate_answer, question)
    
    def _create_evaluation_prompt(self, candidate_answer: str, question: str, user_profile: Dict[str, Any]) -> str:
        """Create the evaluation prompt with exact instructions."""
        
        # Include knowledge base context
        knowledge_context = ""
        if self.golden_responses:
            knowledge_context += f"Golden Responses: {json.dumps(self.golden_responses, indent=2)}\n\n"
        if self.buyability_profiles:
            knowledge_context += f"Buyability Profiles: {json.dumps(self.buyability_profiles, indent=2)}\n\n"
        if self.fair_housing_guide:
            knowledge_context += f"Fair Housing Guide: {json.dumps(self.fair_housing_guide, indent=2)}\n\n"
        
        prompt = f"""
You are a Zillow home affordability expert evaluator. You must evaluate the candidate response using EXACTLY the following instructions:

EVALUATION INPUT:
Question: "{question}"
Candidate Answer: "{candidate_answer}"
User Profile: {user_profile}

KNOWLEDGE BASE CONTEXT:
{knowledge_context}

EVALUATION INSTRUCTIONS (Follow EXACTLY):

For each response, evaluate:

1. Personalization Accuracy (Accurate/Inaccurate): Does the personalization help answer question better?
How to evaluate: The breakdown might use Zillow Buyability number or monthly payment or interest rate and info from user data to better answer the user question in a tailored fashion. Judge if the use of the personalization parameters makes sense in the context and any relevant personalization parameter is omitted and clearly state why that added parameter might have improved the answer
Compare the response against values in buyabilityprofile. Credit additional user-specific figures derived from that data (e.g., BuyAbility amount, projected monthly payment). Use chain-of-thought reasoning, then explain in detail how these elements improve—or fail to improve—the answer.

2. Context-Based Personalization — 1 – 5
Identify every relevant customization the answer could reasonably include.
Count how many of those appear in the response.
Score = floor((present ÷ relevant) × 5) with buckets:
<20% = 1 · 20-40% = 2 · 40-60% = 3 · 60-80% = 4 · >80% = 5.
Give detailed justification for your evaluation more than 150 words

3. Next-Step Identification — Present / Not-Present
Look for explicit or implicit action guidance (e.g., apply, schedule, consult, get pre-approved).
Give detailed justification for your evaluation more than 150 words

4. Assumption Listing — True / False
True = the response clearly states all assumptions (or explicitly notes "no assumptions").
Reward transparency; explain your judgment.

5. Assumption Trust: ambiguous or missing data are appropriately called out and co-pilot mentions any other deficiencies with the answer. How to evaluate: Evaluate the transparency and candor of the answer. Any shortcoming called out, assumptions made stated. Grade it on a scale of 1 to 5, with 1 being worst, 5 being best. Use golden response as reference but focus on the exact user prompt to make determination, but do not expect it call out potential inaccuracies on all responses. Have a detailed explanation for judgement. Anchor explanation to the user prompt and golden answer, not to hypothetical errors. Give detailed justification for your evaluation more than 150 words

6. Calculation Accuracy — True / False
Verify all math with inputs from buyabilityprofile.
If calculations are absent → True
If a figure can't be checked because an input is missing, mark True but explicitly call out the gap and any validation you could not perform.
Give detailed justification for your evaluation more than 150 words

7. Faithfulness to Ground Truth — True / False
Confirm that statements align with Zillow tools, goldenresponsealpha.docx, and any other provided ground-truth docs.
Do not penalize omissions—only inaccuracies.
Provide an in-depth rationale and supporting citations at least 150 words

8. Overall Accuracy — True / False
Holistic verdict: does the response, taken as a whole, correctly answer the user's question?
Give detailed justification for your evaluation more than 150 words

9. Structured Presentation — 1 – 5
1 = wall of text.
2 = minimal, inconsistent structure.
3 = clear headings and lists for ≥ 50% of content; simple tables where obvious.
4 = logical hierarchy, parallel lists, well-formatted tables.
5 = outline-quality headings, flawless lists, captioned tables (with alt text if apt).
Give detailed justification for your evaluation more than 150 words

10. Coherence — True / False
Check for logical consistency, absence of contradictions, and minimal repetition.
Flag any issues and justify your decision.

11. Completeness — 1 – 5
Estimate the percentage of the user's sub-questions answered:
<20% = 1 · 20-40% = 2 · 40-60% = 3 · 60-80% = 4 · >80% = 5.
Give detailed justification for your evaluation more than 150 words

12. Fair Housing Classifier (Binary) verifies if fair housing classifier is invoked How to evaluate: Check 'Zillow_Fair_Housing_Classifier.docx' for rules and rely on web sources to make binary evaluation whether any fair house violation. True means no violations. Give detailed justification for your evaluation more than 150 words

OUTPUT FORMAT REQUIRED:
| Metric | Score | Justification |
|--------|-------|---------------|
| Personalization Accuracy | Accurate/Inaccurate | Detailed Explanation |
| Context based Personalization | Score (1-5) | Detailed Explanation |
| Next Step Identification | Present/Not-Present | Detailed Explanation |
| Assumption Listing | True/False | Detailed Explanation |
| Assumption Trust | Score (1-5) | Detailed Explanation |
| Calculation Accuracy | True/False | Detailed Explanation |
| Faithfulness to Ground Truth | True/False | Detailed Explanation |
| Overall Accuracy | True/False | Detailed Explanation |
| Structured Presentation | Score (1-5) | Detailed Explanation |
| Coherence | True/False | Detailed Explanation |
| Completeness | Score (1-5) | Detailed Explanation |
| Fair Housing Classifier | True/False | Detailed Explanation |

Give a final score out of 100, weighing each metrics equally. Omit completeness and structured presentation in scoring. Label the weighted score as 'Alpha evaluation'

IMPORTANT: Provide ALL justifications with more than 150 words each. Follow the exact scoring criteria provided.
"""
        
        return prompt
    
    def _format_llm_results(self, llm_evaluation: str, candidate_answer: str, question: str, user_profile: Dict[str, Any]) -> str:
        """Format the LLM evaluation results."""
        
        # Add header information
        header = f"""
🤖 LLM-BASED ZILLOW JUDGE EVALUATION RESULTS
================================================================================
Model Used: {self.model}
Evaluation Method: LLM API-based (OpenAI)

📝 EVALUATION INPUT:
Question: '{question}'
Answer: '{candidate_answer}'
User Profile: {user_profile}

================================================================================
LLM EVALUATION RESULTS:
================================================================================
"""
        
        # Return formatted results
        return header + llm_evaluation
    
    def _create_error_response(self, error_msg: str, candidate_answer: str, question: str) -> str:
        """Create error response when LLM evaluation fails."""
        
        return f"""
❌ LLM EVALUATION ERROR
================================================================================
{error_msg}

📝 INPUT DETAILS:
Question: '{question}'
Answer: '{candidate_answer}'

💡 TROUBLESHOOTING:
1. Check if your OpenAI API key is valid
2. Verify if the model '{self.model}' is available
3. Ensure you have sufficient API credits
4. Check your internet connection

To use rule-based evaluation as backup, use ZillowJudgeEvaluatorUpdated instead.
================================================================================
"""


def main():
    """Main function for testing the LLM-based evaluator."""
    
    print("🤖 LLM-BASED ZILLOW JUDGE EVALUATOR")
    print("=" * 60)
    
    # Get API credentials from environment or user input
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    
    if not api_key:
        print("❌ No OpenAI API key found in environment variables")
        print("Set OPENAI_API_KEY before running this script")
        return
    
    # Choose model (o1-preview if available, otherwise gpt-4)
    model = "o1-preview"  # Will fall back to gpt-4 if o1 not available
    
    print(f"🔑 API Key: {'*' * 20}{api_key[-10:] if len(api_key) > 10 else '*' * len(api_key)}")
    print(f"🌐 Base URL: {base_url or 'Default OpenAI'}")
    print(f"🤖 Model: {model}")
    
    # Initialize LLM-based evaluator
    try:
        evaluator = LLMBasedZillowJudge(api_key=api_key, base_url=base_url, model=model)
        print("✅ LLM-based evaluator initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize evaluator: {e}")
        return
    
    # Test evaluation
    test_answer = "you can afford to buy now"
    test_question = "can I afford to buy a home right now?"
    test_profile = {
        "annual_income": None,
        "monthly_debts": None,
        "down_payment": None,
        "credit_score": None
    }
    
    print(f"\n🔍 Running LLM-based evaluation...")
    result = evaluator.evaluate(test_answer, test_question, test_profile)
    print(result)


if __name__ == "__main__":
    main()