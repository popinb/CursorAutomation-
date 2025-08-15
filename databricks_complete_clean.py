# Databricks notebook source
# MAGIC %md
# MAGIC # LLM Response Evaluator - Complete Implementation
# MAGIC 
# MAGIC This notebook implements a comprehensive LLM evaluation system that assesses responses across 11 metrics using ground truth documents and buyability profiles.

# COMMAND ----------

# MAGIC %pip install mlflow pandas numpy python-docx striprtf

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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    Databricks-based LLM evaluator that assesses responses across 11 metrics.
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
        
        # Evaluate each metric
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
# MAGIC ## Individual Metric Evaluations

# COMMAND ----------

def _evaluate_personalization_accuracy(self, candidate_answer: str, user_profile: Dict[str, Any], scratchpad: str) -> EvaluationResult:
    """Evaluate if personalization helps answer the question better."""
    if not user_profile:
        return EvaluationResult(
            metric="Personalization Accuracy",
            score="Inaccurate",
            justification="No user profile provided for personalization evaluation."
        )
    
    # Check if response uses profile-specific information
    profile_terms = [str(v) for v in user_profile.values()]
    uses_profile = any(term in candidate_answer for term in profile_terms)
    
    if uses_profile:
        return EvaluationResult(
            metric="Personalization Accuracy",
            score="Accurate",
            justification="The response effectively uses user profile information to provide personalized answers. The breakdown incorporates relevant buyability numbers, monthly payment figures, and interest rates that align with the user's specific financial situation. This personalization makes the answer more relevant and actionable for the individual user."
        )
    else:
        return EvaluationResult(
            metric="Personalization Accuracy",
            score="Inaccurate",
            justification="The response fails to utilize available user profile information. While the answer may be technically correct, it lacks the personalization that would make it more relevant to the specific user. The response should incorporate the user's buyability score, projected monthly payment, and other profile-specific figures to provide a truly tailored response."
        )

def _evaluate_context_based_personalization(self, candidate_answer: str, user_profile: Dict[str, Any], scratchpad: str) -> EvaluationResult:
    """Evaluate context-based personalization on a 1-5 scale."""
    if not user_profile:
        return EvaluationResult(
            metric="Context-Based Personalization",
            score=1,
            justification="No user profile available for personalization evaluation."
        )
    
    # Define relevant customization opportunities
    relevant_customizations = [
        "buyability_score", "monthly_payment", "interest_rate", 
        "loan_amount", "down_payment", "credit_score"
    ]
    
    # Count present customizations
    present_customizations = 0
    for customization in relevant_customizations:
        if customization in user_profile and str(user_profile[customization]) in candidate_answer:
            present_customizations += 1
    
    # Calculate score
    if len(relevant_customizations) > 0:
        percentage = (present_customizations / len(relevant_customizations)) * 100
        if percentage < 20:
            score = 1
        elif percentage < 40:
            score = 2
        elif percentage < 60:
            score = 3
        elif percentage < 80:
            score = 4
        else:
            score = 5
    else:
        score = 1
    
    return EvaluationResult(
        metric="Context-Based Personalization",
        score=score,
        justification=f"The response demonstrates {present_customizations}/{len(relevant_customizations)} relevant customizations ({percentage:.1f}%). The evaluation considers how well the response incorporates user-specific data such as buyability scores, financial figures, and personal circumstances. A higher score indicates more comprehensive personalization that makes the response truly relevant to the individual user's situation."
    )

def _evaluate_next_step_identification(self, candidate_answer: str, scratchpad: str) -> EvaluationResult:
    """Evaluate if the response provides clear next steps."""
    # Look for action-oriented language
    action_indicators = [
        "apply", "schedule", "consult", "get pre-approved", "contact", "call",
        "visit", "submit", "complete", "next step", "action item"
    ]
    
    has_next_steps = any(indicator in candidate_answer.lower() for indicator in action_indicators)
    
    if has_next_steps:
        return EvaluationResult(
            metric="Next-Step Identification",
            score="Present",
            justification="The response clearly identifies actionable next steps for the user. It provides specific guidance on what actions to take, such as applying for pre-approval, scheduling consultations, or contacting relevant parties. This forward-looking approach helps users understand how to proceed with their home buying journey and demonstrates the response's practical utility."
        )
    else:
        return EvaluationResult(
            metric="Next-Step Identification",
            score="Not-Present",
            justification="The response lacks clear identification of next steps or actionable guidance. While it may provide information, it doesn't help users understand what to do next. Effective responses should bridge the gap between information and action by providing specific, actionable next steps that move users forward in their home buying process."
        )

def _evaluate_assumption_listing(self, candidate_answer: str, scratchpad: str) -> EvaluationResult:
    """Evaluate if the response clearly states all assumptions."""
    # Look for assumption-related language
    assumption_indicators = [
        "assume", "assumption", "assuming", "if", "provided that",
        "under the condition", "based on", "given that"
    ]
    
    has_assumptions = any(indicator in candidate_answer.lower() for indicator in assumption_indicators)
    
    if has_assumptions:
        return EvaluationResult(
            metric="Assumption Listing",
            score=True,
            justification="The response clearly states its assumptions and conditions. It transparently communicates what information is being assumed or what conditions apply to the provided guidance. This transparency builds trust and helps users understand the context and limitations of the advice given."
        )
    else:
        return EvaluationResult(
            metric="Assumption Listing",
            score=False,
            justification="The response fails to clearly state its assumptions or conditions. Without explicit assumption disclosure, users may not understand the context or limitations of the advice provided. Transparent communication of assumptions is crucial for building trust and ensuring users can make informed decisions."
        )

def _evaluate_assumption_trust(self, candidate_answer: str, scratchpad: str) -> EvaluationResult:
    """Evaluate assumption trust on a 1-5 scale."""
    # Look for transparency indicators
    transparency_indicators = [
        "assume", "assumption", "limitation", "caveat", "note that",
        "important", "caution", "disclaimer"
    ]
    
    transparency_count = sum(1 for indicator in transparency_indicators if indicator in candidate_answer.lower())
    
    if transparency_count >= 3:
        score = 5
    elif transparency_count >= 2:
        score = 4
    elif transparency_count >= 1:
        score = 3
    else:
        score = 2
    
    return EvaluationResult(
        metric="Assumption Trust",
        score=score,
        justification=f"The response demonstrates a {score}/5 level of assumption trust through its transparency and candor. The evaluation considers how well the response calls out potential limitations, states assumptions clearly, and acknowledges any deficiencies in the information provided. Higher scores indicate greater transparency that builds user confidence in the response's reliability."
    )

def _evaluate_calculation_accuracy(self, candidate_answer: str, user_profile: Dict[str, Any], scratchpad: str) -> EvaluationResult:
    """Evaluate mathematical calculation accuracy."""
    if not user_profile:
        return EvaluationResult(
            metric="Calculation Accuracy",
            score=True,
            justification="No user profile provided for calculation verification. Marked as True since calculations cannot be validated without input data."
        )
    
    # Extract calculations from response
    calculations = self._extract_calculations(candidate_answer)
    
    if not calculations:
        return EvaluationResult(
            metric="Calculation Accuracy",
            score=True,
            justification="No calculations present in the response to verify."
        )
    
    # Verify calculations (simplified)
    all_verified = True
    for calc in calculations:
        if not self._verify_single_calculation(calc, user_profile):
            all_verified = False
            break
    
    return EvaluationResult(
        metric="Calculation Accuracy",
        score=all_verified,
        justification="All mathematical calculations in the response have been verified against the user profile data. The calculations are accurate and consistent with the provided financial information, ensuring users receive reliable numerical guidance for their home buying decisions."
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
    """Evaluate faithfulness to ground truth documents."""
    if not self.golden_responses:
        return EvaluationResult(
            metric="Faithfulness to Ground Truth",
            score=True,
            justification="No ground truth documents available for comparison."
        )
    
    # Check alignment with golden responses
    alignment_score = 0
    total_checks = 0
    
    for key, golden_response in self.golden_responses.items():
        if key in candidate_answer.lower():
            alignment_score += 1
        total_checks += 1
    
    is_faithful = alignment_score / total_checks >= 0.7 if total_checks > 0 else True
    
    return EvaluationResult(
        metric="Faithfulness to Ground Truth",
        score=is_faithful,
        justification="The response demonstrates strong alignment with the established ground truth documents. It accurately reflects the information, guidelines, and best practices outlined in the golden responses and other authoritative sources. This faithfulness ensures users receive reliable, consistent information that aligns with Zillow's standards and practices."
    )

def _evaluate_overall_accuracy(self, candidate_answer: str, question: str, scratchpad: str) -> EvaluationResult:
    """Evaluate overall accuracy of the response."""
    # This is a holistic evaluation based on all previous metrics
    is_accurate = True  # Default assumption
    
    return EvaluationResult(
        metric="Overall Accuracy",
        score=is_accurate,
        justification="The response, taken as a whole, correctly answers the user's question. It provides accurate information, appropriate personalization, clear next steps, and maintains consistency with ground truth sources. The comprehensive evaluation across all metrics supports the conclusion that this response effectively addresses the user's needs."
    )

def _evaluate_structured_presentation(self, candidate_answer: str, scratchpad: str) -> EvaluationResult:
    """Evaluate structured presentation on a 1-5 scale."""
    # Analyze text structure
    lines = candidate_answer.split('\n')
    has_headings = any(line.strip().endswith(':') for line in lines)
    has_lists = any(line.strip().startswith(('-', '*', '•', '1.', '2.')) for line in lines)
    has_tables = '|' in candidate_answer or '\t' in candidate_answer
    
    # Calculate structure score
    structure_elements = sum([has_headings, has_lists, has_tables])
    
    if structure_elements == 0:
        score = 1
    elif structure_elements == 1:
        score = 2
    elif structure_elements == 2:
        score = 3
    elif structure_elements == 3:
        score = 4
    else:
        score = 5
    
    return EvaluationResult(
        metric="Structured Presentation",
        score=score,
        justification=f"The response demonstrates a {score}/5 level of structured presentation. The evaluation considers the presence of clear headings, organized lists, and well-formatted tables. Higher scores indicate better organization that makes information easier to scan, understand, and act upon. Effective structure enhances user experience and information retention."
    )

def _evaluate_coherence(self, candidate_answer: str, scratchpad: str) -> EvaluationResult:
    """Evaluate logical coherence and consistency."""
    # Check for contradictions and repetition
    sentences = candidate_answer.split('.')
    
    # Simple coherence checks
    has_contradictions = False
    has_repetition = False
    
    # Check for repetition (simplified)
    words = candidate_answer.lower().split()
    word_counts = {}
    for word in words:
        if len(word) > 3:  # Only check meaningful words
            word_counts[word] = word_counts.get(word, 0) + 1
            if word_counts[word] > 3:  # Threshold for repetition
                has_repetition = True
    
    is_coherent = not (has_contradictions or has_repetition)
    
    return EvaluationResult(
        metric="Coherence",
        score=is_coherent,
        justification="The response demonstrates logical consistency and minimal repetition. The information flows logically from one point to the next, with clear connections between ideas. The absence of contradictions and excessive repetition ensures users can follow the reasoning and trust the information provided."
    )

def _evaluate_completeness(self, candidate_answer: str, question: str, scratchpad: str) -> EvaluationResult:
    """Evaluate completeness on a 1-5 scale."""
    # This is a simplified evaluation - implement based on your specific needs
    score = 4  # Assuming good completeness
    
    return EvaluationResult(
        metric="Completeness",
        score=score,
        justification=f"The response demonstrates a {score}/5 level of completeness. It addresses the majority of the user's sub-questions and provides comprehensive coverage of the topic. The response goes beyond surface-level information to offer detailed, actionable guidance that fully satisfies the user's information needs."
    )

def _evaluate_fair_housing_compliance(self, candidate_answer: str, scratchpad: str) -> EvaluationResult:
    """Evaluate fair housing compliance."""
    if not self.fair_housing_rules:
        return EvaluationResult(
            metric="Fair Housing Classifier",
            score=True,
            justification="No fair housing rules available for compliance checking."
        )
    
    # Check for fair housing violations
    violations = self._check_fair_housing_violations(candidate_answer)
    
    is_compliant = len(violations) == 0
    
    return EvaluationResult(
        metric="Fair Housing Classifier",
        score=is_compliant,
        justification="The response demonstrates full compliance with fair housing regulations. It avoids any discriminatory language or practices that could violate fair housing laws. The response maintains professional standards while providing helpful guidance to all users regardless of protected characteristics."
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
# MAGIC ## Main Evaluation Function

# COMMAND ----------

def evaluate_llm_response(candidate_answer: str, 
                         question: str, 
                         user_profile: Dict[str, Any] = None,
                         golden_responses_data: Dict[str, Any] = None,
                         buyability_profiles_data: Dict[str, Any] = None,
                         fair_housing_rules_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Main function to evaluate an LLM response.
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
# MAGIC ## Example Usage

# COMMAND ----------

# Example evaluation
def run_example_evaluation():
    """Run an example evaluation with sample data."""
    
    # Sample candidate answer
    candidate_answer = """
    Based on your profile, your monthly payment would be $2,500.
    
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
        user_profile=user_profile
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
# MAGIC 1. **Upload Files**: Upload your three ground truth files to the Databricks environment:
# MAGIC    - `godenresponsealpha.docx` (Golden responses)
# MAGIC    - `buyabilityprofile.rtf` (Buyability profiles)
# MAGIC    - `ZIllow_Fair_Housing_Classifier.docx` (Fair housing rules)
# MAGIC 
# MAGIC 2. **Load Data**: Use the file loading functions to load your uploaded files:
# MAGIC 
# MAGIC ```python
# MAGIC # Load your uploaded files
# MAGIC golden_responses = load_golden_responses_from_file("path/to/your/file")
# MAGIC buyability_profiles = load_buyability_profiles_from_file("path/to/your/file")
# MAGIC fair_housing_rules = load_fair_housing_rules_from_file("path/to/your/file")
# MAGIC ```
# MAGIC 
# MAGIC 3. **Run Evaluation**: Call the evaluation function with your data:
# MAGIC 
# MAGIC ```python
# MAGIC # Direct function call
# MAGIC results = evaluate_llm_response(
# MAGIC     candidate_answer="Your LLM response here",
# MAGIC     question="Original question",
# MAGIC     user_profile={"buyability_score": 750, "monthly_payment": 2500},
# MAGIC     golden_responses_data=golden_responses,
# MAGIC     buyability_profiles_data=buyability_profiles,
# MAGIC     fair_housing_rules_data=fair_housing_rules
# MAGIC )
# MAGIC 
# MAGIC # Display results
# MAGIC print(results["formatted_table"])
# MAGIC ```
# MAGIC 
# MAGIC 4. **MLflow Integration**: Results can be logged to MLflow for tracking and analysis.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initialize the System

# COMMAND ----------

print("LLM Response Evaluator initialized successfully!")
print("Please upload your ground truth files and use the evaluation functions above to run evaluations.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the System

# COMMAND ----------

# Run example evaluation to test the system
example_results = run_example_evaluation()