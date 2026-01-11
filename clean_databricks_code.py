# LLM Response Evaluator - Clean Python Code for Databricks
# Copy this code into a new Databricks notebook

# COMMAND ----------
# Install required packages
# MAGIC %pip install mlflow pandas numpy python-docx striprtf

# COMMAND ----------
# Import Libraries
import json
import re
import os
from decimal import Decimal
from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path
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
# Data Classes
@dataclass
class EvaluationResult:
    """Data class for evaluation results."""
    metric: str
    score: Union[str, int, bool]
    justification: str
    raw_score: Optional[float] = None

# COMMAND ----------
# Core Evaluator Class
class DatabricksLLMEvaluator:
    """
    Databricks-based LLM evaluator that assesses responses across 11 metrics.
    """
    
    def __init__(self, 
                 golden_responses_data: Dict[str, Any] = None,
                 buyability_profiles_data: Dict[str, Any] = None,
                 fair_housing_rules_data: Dict[str, Any] = None,
                 mlflow_tracking_uri: str = None):
        """Initialize the evaluator."""
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
        """Main evaluation method that assesses a candidate answer across all 11 metrics."""
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
# Helper Methods
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

# COMMAND ----------
# Main Evaluation Function
def evaluate_llm_response(candidate_answer: str, 
                         question: str, 
                         user_profile: Dict[str, Any] = None,
                         golden_responses_data: Dict[str, Any] = None,
                         buyability_profiles_data: Dict[str, Any] = None,
                         fair_housing_rules_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Main function to evaluate an LLM response."""
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
# Example Usage
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
# Initialize the System
print("LLM Response Evaluator initialized successfully!")
print("Please upload your ground truth files and use the evaluation functions above to run evaluations.")

# COMMAND ----------
# Test the System
example_results = run_example_evaluation()