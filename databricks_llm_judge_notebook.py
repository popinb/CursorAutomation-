# Databricks notebook source
# MAGIC %md
# MAGIC # LLM Response Evaluator - AI Judge System
# MAGIC 
# MAGIC This notebook implements a comprehensive LLM evaluation system that uses AI as a judge to assess responses across 11 metrics using ground truth documents and buyability profiles.

# COMMAND ----------

# MAGIC %pip install mlflow pandas numpy python-docx striprtf openai

# COMMAND ----------

# COMMAND ----------
# MAGIC %md
# MAGIC ## Import Libraries and Initialize

# COMMAND ----------

import json
import re
import os
from decimal import Decimal
from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path
from typing_extensions import deprecated
import mlflow
import mlflow.pyfunc
import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import logging
import openai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# COMMAND ----------

# MAGIC %md
# MAGIC ## OpenAI API Setup

# COMMAND ----------

# Set your OpenAI API key here
openai.api_key = "your-api-key-here"  # Replace with your actual OpenAI API key

# Alternative: Use environment variable (more secure)
# import os
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"
# openai.api_key = os.environ["OPENAI_API_KEY"]

print("OpenAI API configured successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Classes and Core Evaluator

# COMMAND ----------

@dataclass
class EvaluationResult:
    """Data class for evaluation results."""
    metric: str
    score: Union[str, int, bool]
    justification: str
    raw_score: Optional[float] = None

class DatabricksLLMEvaluator:
    """
    Databricks-based LLM evaluator that assesses responses across 11 metrics using AI as a judge.
    """
    
    def __init__(self, 
                 golden_responses_data: Dict[str, Any] = None,
                 buyability_profiles_data: Dict[str, Any] = None,
                 fair_housing_rules_data: Dict[str, Any] = None,
                 mlflow_tracking_uri: str = None):
        """
        Initialize the evaluator.
        """
        self.golden_responses = golden_responses_data or {}
        self.buyability_profiles = buyability_profiles_data or {}
        self.fair_housing_rules = fair_housing_rules_data or {}
        
        # Initialize MLflow if URI provided
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            logger.info(f"MLflow tracking initialized: {mlflow_tracking_uri}")
    
    def evaluate_response(self, 
                         candidate_answer: str, 
                         question: str = "", 
                         user_profile: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main evaluation method that assesses a candidate answer across all 11 metrics.
        """
        # Generate internal scratchpad (not shown to user)
        scratchpad = self._generate_scratchpad(candidate_answer, question, user_profile)
        
        # Evaluate each metric using LLM judge
        results = []
        
        # 1. Personalization Accuracy
        results.append(self._evaluate_personalization_accuracy(
            candidate_answer, user_profile, scratchpad))
        
        # 2. Context-Based Personalization
        results.append(self._evaluate_context_based_personalization(
            candidate_answer, user_profile, scratchpad))
        
        # 3. Next-Step Identification
        results.append(self._evaluate_next_step_identification(
            candidate_answer, scratchpad))
        
        # 4. Assumption Listing
        results.append(self._evaluate_assumption_listing(
            candidate_answer, scratchpad))
        
        # 5. Assumption Trust
        results.append(self._evaluate_assumption_trust(
            candidate_answer, scratchpad))
        
        # 6. Calculation Accuracy
        results.append(self._evaluate_calculation_accuracy(
            candidate_answer, user_profile, scratchpad))
        
        # 7. Faithfulness to Ground Truth
        results.append(self._evaluate_faithfulness_to_ground_truth(
            candidate_answer, scratchpad))
        
        # 8. Overall Accuracy
        results.append(self._evaluate_overall_accuracy(
            candidate_answer, question, scratchpad))
        
        # 9. Structured Presentation
        results.append(self._evaluate_structured_presentation(
            candidate_answer, scratchpad))
        
        # 10. Coherence
        results.append(self._evaluate_coherence(
            candidate_answer, scratchpad))
        
        # 11. Completeness
        results.append(self._evaluate_completeness(
            candidate_answer, question, scratchpad))
        
        # 12. Fair Housing Classifier
        results.append(self._evaluate_fair_housing_compliance(
            candidate_answer, scratchpad))
        
        # Calculate final score
        final_score = self._calculate_final_score(results)
        
        return {
            "evaluation_results": results,
            "final_score": final_score,
            "scratchpad": scratchpad,
            "timestamp": datetime.now().isoformat()
        }

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluation Methods Implementation

# COMMAND ----------

def _generate_scratchpad(self, candidate_answer: str, question: str, user_profile: Dict[str, Any]) -> str:
    """Generate internal reasoning scratchpad (not shown to user)."""
    scratchpad = "=== SCRATCHPAD (Internal Reasoning) ===\n\n"
    
    # Extract key variables from candidate answer
    extracted_vars = self._extract_variables(candidate_answer)
    scratchpad += f"Extracted variables: {extracted_vars}\n\n"
    
    # Find matching ground truth
    ground_truth = self._find_matching_ground_truth(question, extracted_vars)
    scratchpad += f"Selected ground truth: {ground_truth}\n\n"
    
    # Profile matching analysis
    if user_profile:
        profile_match = self._analyze_profile_match(extracted_vars, user_profile)
        scratchpad += f"Profile matching: {profile_match}\n\n"
    
    # Mathematical verification
    math_checks = self._verify_calculations(extracted_vars, user_profile)
    scratchpad += f"Math verification: {math_checks}\n\n"
    
    return scratchpad

def _extract_variables(self, text: str) -> Dict[str, str]:
    """Extract numeric and variable values from candidate text."""
    variables = {}
    
    # Extract dollar amounts
    dollar_pattern = r'\$[\d,]+(?:\.\d{2})?'
    dollar_matches = re.findall(dollar_pattern, text)
    if dollar_matches:
        variables['dollar_amounts'] = dollar_matches
    
    # Extract percentages
    percent_pattern = r'\d+(?:\.\d+)?%'
    percent_matches = re.findall(percent_pattern, text)
    if percent_matches:
        variables['percentages'] = percent_matches
    
    # Extract numbers
    number_pattern = r'\b\d+(?:,\d{3})*(?:\.\d+)?\b'
    number_matches = re.findall(number_pattern, text)
    if number_matches:
        variables['numbers'] = number_matches
    
    # Extract key terms
    key_terms = ['buyability', 'monthly payment', 'interest rate', 'loan amount', 'down payment']
    for term in key_terms:
        if term.lower() in text.lower():
            variables[term] = True
    
    return variables

def _find_matching_ground_truth(self, question: str, extracted_vars: Dict[str, str]) -> Dict[str, Any]:
    """Find the most relevant ground truth for the given question and variables."""
    if not self.golden_responses:
        return {}
    
    # Simple keyword matching
    for key, response in self.golden_responses.items():
        if any(term in question.lower() for term in key.lower().split('_')):
            return response
    
    return {}

def _analyze_profile_match(self, extracted_vars: Dict[str, str], user_profile: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze how well the extracted variables match the user profile."""
    if not user_profile:
        return {"status": "no_profile_provided"}
    
    matches = {}
    for key, value in extracted_vars.items():
        if key in user_profile:
            matches[key] = {
                "extracted": value,
                "profile": user_profile[key],
                "match": str(value) == str(user_profile[key])
            }
    
    return matches

def _verify_calculations(self, extracted_vars: Dict[str, str], user_profile: Dict[str, Any]) -> Dict[str, Any]:
    """Verify mathematical calculations in the response."""
    if not user_profile:
        return {"status": "no_profile_provided"}
    
    verifications = {}
    
    # Example: verify monthly payment calculation
    if 'monthly_payment' in extracted_vars and 'loan_amount' in user_profile:
        verifications['monthly_payment'] = "calculation_verified"
    
    return verifications

# COMMAND ----------

# MAGIC %md
# MAGIC ## Individual Metric Evaluations (LLM Judge Version)

# COMMAND ----------

def _evaluate_personalization_accuracy(self, candidate_answer: str, user_profile: Dict[str, Any], scratchpad: str) -> EvaluationResult:
    """Use LLM to judge personalization accuracy."""
    
    prompt = f"""
    You are an expert evaluator assessing LLM responses for a real estate company.
    
    TASK: Evaluate if this response effectively uses personalization to answer the user's question better.
    
    QUESTION: "What's my monthly payment?"
    USER PROFILE: {user_profile}
    CANDIDATE ANSWER: {candidate_answer}
    
    EVALUATION CRITERIA:
    - Does the response use profile-specific information (buyability score, monthly payment, interest rate)?
    - Does it make the answer more relevant to this specific user?
    - Does it incorporate user data appropriately?
    
    OUTPUT FORMAT:
    Score: "Accurate" or "Inaccurate"
    Justification: Provide detailed explanation (minimum 150 words) explaining your judgment, citing specific evidence from the response, and explaining why this matters for user experience.
    
    EVALUATE NOW:
    """
    
    llm_response = self._call_llm_judge(prompt)
    score, justification = self._parse_llm_response(llm_response)
    
    return EvaluationResult(
        metric="Personalization Accuracy",
        score=score,
        justification=justification
    )

def _evaluate_context_based_personalization(self, candidate_answer: str, user_profile: Dict[str, Any], scratchpad: str) -> EvaluationResult:
    """Use LLM to judge context-based personalization on 1-5 scale."""
    
    prompt = f"""
    You are an expert evaluator assessing LLM responses for a real estate company.
    
    TASK: Rate the context-based personalization on a scale of 1-5.
    
    QUESTION: "What's my monthly payment?"
    USER PROFILE: {user_profile}
    CANDIDATE ANSWER: {candidate_answer}
    
    SCORING GUIDE:
    1 = No personalization, generic response
    2 = Minimal personalization, barely relevant
    3 = Some personalization, moderately relevant
    4 = Good personalization, highly relevant
    5 = Excellent personalization, perfectly tailored
    
    EVALUATION CRITERIA:
    - How well does the response incorporate user-specific data?
    - Does it address the user's unique financial situation?
    - Is the advice tailored to their specific profile?
    
    OUTPUT FORMAT:
    Score: [1-5]
    Justification: Provide detailed explanation (minimum 150 words) explaining your rating, citing specific evidence, and explaining why this level of personalization matters.
    
    EVALUATE NOW:
    """
    
    llm_response = self._call_llm_judge(prompt)
    score, justification = self._parse_llm_response(llm_response)
    
    return EvaluationResult(
        metric="Context-Based Personalization",
        score=int(score) if score.isdigit() else 1,
        justification=justification
    )

def _evaluate_next_step_identification(self, candidate_answer: str, scratchpad: str) -> EvaluationResult:
    """Use LLM to judge if response provides clear next steps."""
    
    prompt = f"""
    You are an expert evaluator assessing LLM responses for a real estate company.
    
    TASK: Determine if this response provides clear, actionable next steps.
    
    QUESTION: "What's my monthly payment?"
    CANDIDATE ANSWER: {candidate_answer}
    
    EVALUATION CRITERIA:
    - Does the response identify specific actions the user should take?
    - Are the next steps clear and actionable?
    - Does it help the user understand how to proceed?
    
    OUTPUT FORMAT:
    Score: "Present" or "Not-Present"
    Justification: Provide detailed explanation (minimum 150 words) explaining your judgment, citing specific evidence from the response, and explaining why clear next steps matter for user experience.
    
    EVALUATE NOW:
    """
    
    llm_response = self._call_llm_judge(prompt)
    score, justification = self._parse_llm_response(llm_response)
    
    return EvaluationResult(
        metric="Next-Step Identification",
        score=score,
        justification=justification
    )

def _evaluate_assumption_listing(self, candidate_answer: str, scratchpad: str) -> EvaluationResult:
    """Use LLM to judge if response clearly states assumptions."""
    
    prompt = f"""
    You are an expert evaluator assessing LLM responses for a real estate company.
    
    TASK: Determine if this response clearly states its assumptions and conditions.
    
    QUESTION: "What's my monthly payment?"
    CANDIDATE ANSWER: {candidate_answer}
    
    EVALUATION CRITERIA:
    - Does the response explicitly state what it assumes?
    - Does it communicate limitations or conditions?
    - Is there transparency about what information is being assumed?
    
    OUTPUT FORMAT:
    Score: "True" or "False"
    Justification: Provide detailed explanation (minimum 150 words) explaining your judgment, citing specific evidence from the response, and explaining why assumption transparency matters for user trust.
    
    EVALUATE NOW:
    """
    
    llm_response = self._call_llm_judge(prompt)
    score, justification = self._parse_llm_response(llm_response)
    
    return EvaluationResult(
        metric="Assumption Listing",
        score=score == "True",
        justification=justification
    )

def _evaluate_assumption_trust(self, candidate_answer: str, scratchpad: str) -> EvaluationResult:
    """Use LLM to judge assumption trust on 1-5 scale."""
    
    prompt = f"""
    You are an expert evaluator assessing LLM responses for a real estate company.
    
    TASK: Rate the assumption trust and transparency on a scale of 1-5.
    
    QUESTION: "What's my monthly payment?"
    CANDIDATE ANSWER: {candidate_answer}
    
    SCORING GUIDE:
    1 = No transparency, hidden assumptions
    2 = Minimal transparency, vague assumptions
    3 = Some transparency, basic assumptions stated
    4 = Good transparency, clear assumptions and limitations
    5 = Excellent transparency, comprehensive assumption disclosure
    
    EVALUATION CRITERIA:
    - How clearly are assumptions stated?
    - Are limitations acknowledged?
    - Is there candor about potential gaps?
    
    OUTPUT FORMAT:
    Score: [1-5]
    Justification: Provide detailed explanation (minimum 150 words) explaining your rating, citing specific evidence, and explaining why assumption trust matters for user confidence.
    
    EVALUATE NOW:
    """
    
    llm_response = self._call_llm_judge(prompt)
    score, justification = self._parse_llm_response(llm_response)
    
    return EvaluationResult(
        metric="Assumption Trust",
        score=int(score) if score.isdigit() else 2,
        justification=justification
    )

def _evaluate_calculation_accuracy(self, candidate_answer: str, user_profile: Dict[str, Any], scratchpad: str) -> EvaluationResult:
    """Use LLM to judge mathematical calculation accuracy."""
    
    prompt = f"""
    You are an expert evaluator assessing LLM responses for a real estate company.
    
    TASK: Determine if mathematical calculations in the response are accurate.
    
    QUESTION: "What's my monthly payment?"
    USER PROFILE: {user_profile}
    CANDIDATE ANSWER: {candidate_answer}
    
    EVALUATION CRITERIA:
    - Are any calculations present in the response?
    - If calculations exist, are they mathematically correct?
    - Do the numbers align with the user profile data?
    
    OUTPUT FORMAT:
    Score: "True" or "False"
    Justification: Provide detailed explanation (minimum 150 words) explaining your judgment, citing specific evidence from the response, and explaining why calculation accuracy matters for user trust.
    
    EVALUATE NOW:
    """
    
    llm_response = self._call_llm_judge(prompt)
    score, justification = self._parse_llm_response(llm_response)
    
    return EvaluationResult(
        metric="Calculation Accuracy",
        score=score == "True",
        justification=justification
    )

def _extract_calculations(self, text: str) -> list[str]:
    """Extract mathematical calculations from text."""
    calculations = []
    
    # Look for mathematical expressions
    math_patterns = [
        r'\d+\s*[+\-*/]\s*\d+',  # Basic arithmetic
        r'\$\d+\s*[+\-*/]\s*\d+',  # Dollar amounts with arithmetic
        r'\d+%\s*of\s*\d+',  # Percentage calculations
    ]
    
    for pattern in math_patterns:
        matches = re.findall(pattern, text)
        calculations.extend(matches)
    
    return calculations

def _verify_single_calculation(self, calculation: str, user_profile: Dict[str, Any]) -> bool:
    """Verify a single mathematical calculation."""
    try:
        # Remove dollar signs and evaluate
        clean_calc = calculation.replace('$', '').replace(',', '')
        result = eval(clean_calc)
        return True
    except:
        return False

def _evaluate_faithfulness_to_ground_truth(self, candidate_answer: str, scratchpad: str) -> EvaluationResult:
    """Use LLM to judge faithfulness to ground truth documents."""
    
    prompt = f"""
    You are an expert evaluator assessing LLM responses for a real estate company.
    
    TASK: Determine if this response aligns with established ground truth and best practices.
    
    QUESTION: "What's my monthly payment?"
    CANDIDATE ANSWER: {candidate_answer}
    GROUND TRUTH DOCUMENTS: {self.golden_responses}
    
    EVALUATION CRITERIA:
    - Does the response align with established guidelines?
    - Is the information consistent with best practices?
    - Does it reflect accurate industry knowledge?
    
    OUTPUT FORMAT:
    Score: "True" or "False"
    Justification: Provide detailed explanation (minimum 150 words) explaining your judgment, citing specific evidence from the response, and explaining why faithfulness to ground truth matters for reliability.
    
    EVALUATE NOW:
    """
    
    llm_response = self._call_llm_judge(prompt)
    score, justification = self._parse_llm_response(llm_response)
    
    return EvaluationResult(
        metric="Faithfulness to Ground Truth",
        score=score == "True",
        justification=justification
    )

def _evaluate_overall_accuracy(self, candidate_answer: str, question: str, scratchpad: str) -> EvaluationResult:
    """Use LLM to judge overall accuracy of the response."""
    
    prompt = f"""
    You are an expert evaluator assessing LLM responses for a real estate company.
    
    TASK: Determine if this response, taken as a whole, correctly answers the user's question.
    
    QUESTION: "{question}"
    CANDIDATE ANSWER: {candidate_answer}
    
    EVALUATION CRITERIA:
    - Does the response address the question comprehensively?
    - Is the information accurate and helpful?
    - Does it provide value to the user?
    
    OUTPUT FORMAT:
    Score: "True" or "False"
    Justification: Provide detailed explanation (minimum 150 words) explaining your judgment, citing specific evidence from the response, and explaining why overall accuracy matters for user satisfaction.
    
    EVALUATE NOW:
    """
    
    llm_response = self._call_llm_judge(prompt)
    score, justification = self._parse_llm_response(llm_response)
    
    return EvaluationResult(
        metric="Overall Accuracy",
        score=score == "True",
        justification=justification
    )

def _evaluate_structured_presentation(self, candidate_answer: str, scratchpad: str) -> EvaluationResult:
    """Use LLM to judge structured presentation on 1-5 scale."""
    
    prompt = f"""
    You are an expert evaluator assessing LLM responses for a real estate company.
    
    TASK: Rate the structured presentation and organization on a scale of 1-5.
    
    QUESTION: "What's my monthly payment?"
    CANDIDATE ANSWER: {candidate_answer}
    
    SCORING GUIDE:
    1 = Wall of text, no structure
    2 = Minimal structure, hard to follow
    3 = Some structure, partially organized
    4 = Good structure, well organized
    5 = Excellent structure, professional presentation
    
    EVALUATION CRITERIA:
    - How well is the information organized?
    - Are there clear headings, lists, or sections?
    - Is it easy to scan and understand?
    
    OUTPUT FORMAT:
    Score: [1-5]
    Justification: Provide detailed explanation (minimum 150 words) explaining your rating, citing specific evidence, and explaining why structured presentation matters for user experience.
    
    EVALUATE NOW:
    """
    
    llm_response = self._call_llm_judge(prompt)
    score, justification = self._parse_llm_response(llm_response)
    
    return EvaluationResult(
        metric="Structured Presentation",
        score=int(score) if score.isdigit() else 1,
        justification=justification
    )

def _evaluate_coherence(self, candidate_answer: str, scratchpad: str) -> EvaluationResult:
    """Use LLM to judge logical coherence and consistency."""
    
    prompt = f"""
    You are an expert evaluator assessing LLM responses for a real estate company.
    
    TASK: Determine if this response demonstrates logical coherence and consistency.
    
    QUESTION: "What's my monthly payment?"
    CANDIDATE ANSWER: {candidate_answer}
    
    EVALUATION CRITERIA:
    - Is the information logically organized?
    - Are there contradictions or inconsistencies?
    - Does the response flow well from one point to the next?
    
    OUTPUT FORMAT:
    Score: "True" or "False"
    Justification: Provide detailed explanation (minimum 150 words) explaining your judgment, citing specific evidence from the response, and explaining why coherence matters for user understanding.
    
    EVALUATE NOW:
    """
    
    llm_response = self._call_llm_judge(prompt)
    score, justification = self._parse_llm_response(llm_response)
    
    return EvaluationResult(
        metric="Coherence",
        score=score == "True",
        justification=justification
    )

def _evaluate_completeness(self, candidate_answer: str, question: str, scratchpad: str) -> EvaluationResult:
    """Use LLM to judge completeness on 1-5 scale."""
    
    prompt = f"""
    You are an expert evaluator assessing LLM responses for a real estate company.
    
    TASK: Rate the completeness of this response on a scale of 1-5.
    
    QUESTION: "{question}"
    CANDIDATE ANSWER: {candidate_answer}
    
    SCORING GUIDE:
    1 = Very incomplete, misses key points
    2 = Incomplete, missing important information
    3 = Somewhat complete, covers basics
    4 = Mostly complete, covers most aspects
    5 = Very complete, comprehensive coverage
    
    EVALUATION CRITERIA:
    - How well does it address the question?
    - Are there important gaps in the information?
    - Does it provide sufficient detail?
    
    OUTPUT FORMAT:
    Score: [1-5]
    Justification: Provide detailed explanation (minimum 150 words) explaining your rating, citing specific evidence, and explaining why completeness matters for user satisfaction.
    
    EVALUATE NOW:
    """
    
    llm_response = self._call_llm_judge(prompt)
    score, justification = self._parse_llm_response(llm_response)
    
    return EvaluationResult(
        metric="Completeness",
        score=int(score) if score.isdigit() else 3,
        justification=justification
    )

def _evaluate_fair_housing_compliance(self, candidate_answer: str, scratchpad: str) -> EvaluationResult:
    """Use LLM to judge fair housing compliance."""
    
    prompt = f"""
    You are an expert evaluator assessing LLM responses for a real estate company.
    
    TASK: Determine if this response complies with fair housing regulations.
    
    QUESTION: "What's my monthly payment?"
    CANDIDATE ANSWER: {candidate_answer}
    FAIR HOUSING RULES: {self.fair_housing_rules}
    
    EVALUATION CRITERIA:
    - Does the response avoid discriminatory language?
    - Is it inclusive and welcoming to all?
    - Does it follow fair housing guidelines?
    
    OUTPUT FORMAT:
    Score: "True" or "False"
    Justification: Provide detailed explanation (minimum 150 words) explaining your judgment, citing specific evidence from the response, and explaining why fair housing compliance matters for legal and ethical reasons.
    
    EVALUATE NOW:
    """
    
    llm_response = self._call_llm_judge(prompt)
    score, justification = self._parse_llm_response(llm_response)
    
    return EvaluationResult(
        metric="Fair Housing Classifier",
        score=score == "True",
        justification=justification
    )

def _check_fair_housing_violations(self, text: str) -> list[str]:
    """Check for potential fair housing violations."""
    violations = []
    
    # Check for discriminatory language
    if hasattr(self, 'fair_housing_rules') and 'prohibited_language' in self.fair_housing_rules:
        for term in self.fair_housing_rules['prohibited_language']:
            if term.lower() in text.lower():
                violations.append(f"Potentially discriminatory language: {term}")
    
    return violations

# COMMAND ----------

# MAGIC %md
# MAGIC ## Score Calculation and Output Formatting

# COMMAND ----------

def _calculate_final_score(self, results: list[EvaluationResult]) -> float:
    """Calculate final score out of 100, excluding completeness and structured presentation."""
    # Filter out completeness and structured presentation
    scoring_metrics = [r for r in results if r.metric not in ["Completeness", "Structured Presentation"]]
    
    if not scoring_metrics:
        return 0.0
    
    total_score = 0
    max_possible = 0
    
    for result in scoring_metrics:
        if isinstance(result.score, bool):
            # Boolean metrics: True = 100%, False = 0%
            total_score += 100 if result.score else 0
            max_possible += 100
        elif isinstance(result.score, int):
            # Integer metrics (1-5): Convert to percentage
            total_score += (result.score / 5) * 100
            max_possible += 100
        elif isinstance(result.score, str):
            # String metrics: "Accurate" = 100%, "Present" = 100%, etc.
            if result.score in ["Accurate", "Present", "True"]:
                total_score += 100
            else:
                total_score += 0
            max_possible += 100
    
    final_score = (total_score / max_possible) * 100 if max_possible > 0 else 0
    return round(final_score, 2)

def format_evaluation_table(self, evaluation_results: Dict[str, Any]) -> str:
    """Format evaluation results as a markdown table."""
    results = evaluation_results["evaluation_results"]
    final_score = evaluation_results["final_score"]
    
    table = "| Metric | Score | Justification |\n"
    table += "|--------|-------|---------------|\n"
    
    for result in results:
        table += f"| {result.metric} | {result.score} | {result.justification} |\n"
    
    table += f"\n**Alpha evaluation: {final_score}/100**\n"
    
    return table

# COMMAND ----------

# MAGIC %md
# MAGIC ## LLM Integration Helper Methods

# COMMAND ----------

def _call_llm_judge(self, prompt: str) -> str:
    """Call OpenAI to get evaluation judgment."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Use GPT-4 for best evaluation quality
            messages=[
                {"role": "system", "content": "You are an expert evaluator of LLM responses. Be thorough, fair, and consistent in your evaluations. Always provide detailed justifications."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Low temperature for consistency
            max_tokens=800
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"LLM API error: {e}")
        return "Error: Could not evaluate"

def _parse_llm_response(self, llm_response: str) -> Tuple[str, str]:
    """Parse LLM response to extract score and justification."""
    try:
        # Look for score patterns
        if "Score:" in llm_response:
            score_line = [line for line in llm_response.split('\n') if line.startswith('Score:')][0]
            score = score_line.replace('Score:', '').strip()
        else:
            score = "Error"
        
        # Get justification (everything after score)
        if "Justification:" in llm_response:
            justification = llm_response.split('Justification:')[1].strip()
        else:
            justification = llm_response
        
        return score, justification
    except:
        return "Error", "Could not parse LLM response"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Main Evaluation Function

# COMMAND ----------

def evaluate_llm_response(candidate_answer: str, 
                         question: str, 
                         user_profile: Dict[str, Any] = None,
                         golden_responses_data: Dict[str, Any] = None,
                         buyability_profiles_data: Dict[str, Any] = None,
                         fair_housing_rules_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Main function to evaluate an LLM response using AI as a judge.
    """
    # Initialize evaluator
    evaluator = DatabricksLLMEvaluator(
        golden_responses_data=golden_responses_data,
        buyability_profiles_data=buyability_profiles_data,
        fair_housing_rules_data=fair_housing_rules_data
    )
    
    # Evaluate response
    results = evaluator.evaluate_response(candidate_answer, question, user_profile)
    
    # Format results
    table = evaluator.format_evaluation_table(results)
    
    return {
        "evaluation_results": results,
        "formatted_table": table,
        "evaluator": evaluator
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Ground Truth Files

# COMMAND ----------

def load_ground_truth_files():
    """Load the three ground truth files."""
    
    # Load Golden Responses (DOCX)
    try:
        # For now, create sample data since DOCX parsing requires additional setup
        golden_responses = {
            "mortgage_calculation": {
                "question": "What's my monthly payment?",
                "response": "Your monthly payment depends on your loan amount, interest rate, and term.",
                "variables": ["loan_amount", "interest_rate", "term"],
                "next_steps": ["get pre-approved", "consult specialist"],
                "assumptions": ["current market rates", "standard terms"],
                "fair_housing_compliance": True
            },
            "buyability_explanation": {
                "question": "What's my buyability score?",
                "response": "Your buyability score indicates your likelihood of mortgage approval.",
                "variables": ["credit_score", "income", "debt_ratio"],
                "next_steps": ["check credit report", "improve score"],
                "assumptions": ["accurate financial data"],
                "fair_housing_compliance": True
            }
        }
        print("✅ Golden responses loaded (sample data)")
    except Exception as e:
        print(f"❌ Error loading golden responses: {e}")
        golden_responses = {}
    
    # Load Buyability Profiles (RTF)
    try:
        buyability_profiles = {
            "profile_1": {
                "buyability_score": 750,
                "monthly_payment": 2500,
                "interest_rate": 6.5,
                "loan_amount": 400000,
                "down_payment": 80000,
                "credit_score": 720,
                "annual_income": 85000,
                "location": "California"
            },
            "profile_2": {
                "buyability_score": 680,
                "monthly_payment": 1800,
                "interest_rate": 7.2,
                "loan_amount": 300000,
                "down_payment": 60000,
                "credit_score": 680,
                "annual_income": 65000,
                "location": "Texas"
            }
        }
        print("✅ Buyability profiles loaded (sample data)")
    except Exception as e:
        print(f"❌ Error loading buyability profiles: {e}")
        buyability_profiles = {}
    
    # Load Fair Housing Rules (DOCX)
    try:
        fair_housing_rules = {
            "protected_classes": ["race", "color", "national origin", "religion", "sex", "familial status", "disability"],
            "prohibited_actions": ["discriminatory advertising", "steering", "redlining"],
            "prohibited_language": ["no children", "no pets", "adults only", "preferred race"],
            "prohibited_questions": ["What's your religion?", "Are you married?", "Do you have children?"],
            "red_flags": ["excessive fees", "unreasonable requirements", "delayed processing"],
            "safe_language": ["all welcome", "equal opportunity", "fair housing"],
            "compliance_guidelines": ["treat all applicants equally", "use objective criteria", "document decisions"],
            "examples": {
                "compliant": "We welcome all qualified applicants regardless of background.",
                "non_compliant": "This neighborhood is perfect for young professionals without children."
            }
        }
        print("✅ Fair housing rules loaded (sample data)")
    except Exception as e:
        print(f"❌ Error loading fair housing rules: {e}")
        fair_housing_rules = {}
    
    return golden_responses, buyability_profiles, fair_housing_rules

# Load the files
golden_responses, buyability_profiles, fair_housing_rules = load_ground_truth_files()

print("\n📊 Files loaded successfully!")
print(f"Golden responses: {len(golden_responses)} entries")
print(f"Buyability profiles: {len(buyability_profiles)} profiles")
print(f"Fair housing rules: {len(fair_housing_rules)} rule categories")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Example Usage

# COMMAND ----------

def run_example_evaluation():
    """Run an example evaluation with sample data."""
    
    # Sample candidate answer
    candidate_answer = """
    Based on your profile with buyability score 750, your monthly payment would be $2,500 at 6.5% interest rate.
    
    Here are your options:
    1. Apply for pre-approval
    2. Schedule a consultation
    3. Contact a mortgage specialist
    
    Note: These calculations assume current interest rates and your provided financial information.
    """
    
    question = "What's my monthly payment?"
    user_profile = {
        "buyability_score": 750,
        "monthly_payment": 2500,
        "interest_rate": 6.5
    }
    
    # Run evaluation
    results = evaluate_llm_response(
        candidate_answer=candidate_answer,
        question=question,
        user_profile=user_profile,
        golden_responses_data=golden_responses,
        buyability_profiles_data=buyability_profiles,
        fair_housing_rules_data=fair_housing_rules
    )
    
    # Display results
    print("=== EVALUATION RESULTS ===\n")
    print(results["formatted_table"])
    
    return results

# COMMAND ----------

# MAGIC %md
# MAGIC ## Usage Instructions

# COMMAND ----------

# MAGIC %md
# MAGIC ### How to Use This Notebook
# MAGIC 
# MAGIC 1. **Set Your OpenAI API Key**: Replace `"your-api-key-here"` with your actual OpenAI API key
# MAGIC 2. **Upload Your Ground Truth Files**: Upload your three files to Databricks and modify the `load_ground_truth_files()` function
# MAGIC 3. **Run Evaluations**: Use the `evaluate_llm_response()` function with your data
# MAGIC 
# MAGIC ### Example Usage:
# MAGIC 
# MAGIC ```python
# MAGIC # Test evaluation with your data
# MAGIC results = evaluate_llm_response(
# MAGIC     candidate_answer="Your LLM response here",
# MAGIC     question="Original question",
# MAGIC     user_profile={"buyability_score": 750, "monthly_payment": 2500},
# MAGIC     golden_responses_data=golden_responses,
# MAGIC     buyability_profiles_data=buyability_profiles,
# MAGIC     fair_housing_rules_data=fair_housing_rules
# MAGIC )
# MAGIC 
# MAGIC print(results["formatted_table"])
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initialize the System

# COMMAND ----------

print("LLM Response Evaluator (AI Judge) initialized successfully!")
print("Please set your OpenAI API key and upload your ground truth files to start evaluating.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the System

# COMMAND ----------

# Run example evaluation to test the system
example_results = run_example_evaluation()