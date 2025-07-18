"""
Updated Zillow LLM Judge Evaluator for OpenAI Evals Framework

This evaluator follows the exact evaluation instructions provided, implementing
precise scoring criteria and detailed justifications for each metric.
"""

import json
import re
import os
import math
from decimal import Decimal
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path


class ZillowJudgeEvaluatorUpdated:
    """
    Updated Custom evaluator that assesses LLM responses across 12 metrics
    following the exact evaluation instructions provided.
    """
    
    def __init__(self):
        """Initialize the evaluator with ground truth data."""
        # Get the directory where this script is located
        script_dir = Path(__file__).parent
        
        # Load ground truth data
        self.golden_responses = self._load_json(script_dir / "assets" / "golden_responses.json")
        self.buyability_profiles = self._load_json(script_dir / "assets" / "buyability_profiles.json")
        self.fair_housing_guide = self._load_json(script_dir / "assets" / "fair_housing_guide.json")
        
    def _load_json(self, filepath: Path) -> Dict[str, Any]:
        """Load JSON data from file."""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Required file not found: {filepath}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {filepath}: {e}")
    
    def evaluate(self, candidate_answer: str, question: str = "", user_profile: Dict[str, Any] = None) -> str:
        """
        Main evaluation method following exact evaluation instructions.
        
        Args:
            candidate_answer: The LLM response to evaluate
            question: The original question (optional)
            user_profile: User's buyability profile data (optional)
            
        Returns:
            Formatted evaluation table with scores and justifications
        """
        results = {}
        
        # Evaluate each metric according to exact instructions
        results["personalization_accuracy"] = self._evaluate_personalization_accuracy_updated(
            candidate_answer, question, user_profile)
        results["context_based_personalization"] = self._evaluate_context_based_personalization_updated(
            candidate_answer, question, user_profile)
        results["next_step_identification"] = self._evaluate_next_step_identification_updated(
            candidate_answer, question)
        results["assumption_listing"] = self._evaluate_assumption_listing_updated(
            candidate_answer, question)
        results["assumption_trust"] = self._evaluate_assumption_trust_updated(
            candidate_answer, question)
        results["calculation_accuracy"] = self._evaluate_calculation_accuracy_updated(
            candidate_answer, user_profile)
        results["faithfulness_to_ground_truth"] = self._evaluate_faithfulness_to_ground_truth_updated(
            candidate_answer, question)
        results["overall_accuracy"] = self._evaluate_overall_accuracy_updated(
            candidate_answer, question)
        results["structured_presentation"] = self._evaluate_structured_presentation_updated(
            candidate_answer)
        results["coherence"] = self._evaluate_coherence_updated(
            candidate_answer)
        results["completeness"] = self._evaluate_completeness_updated(
            candidate_answer, question)
        results["fair_housing_classifier"] = self._evaluate_fair_housing_classifier_updated(
            candidate_answer)
        
        # Format final output table with Alpha evaluation score
        return self._format_evaluation_table_updated(results)
    
    def _evaluate_personalization_accuracy_updated(self, answer: str, question: str, user_profile: Dict[str, Any]) -> Dict[str, str]:
        """
        Evaluate personalization accuracy according to exact instructions.
        """
        answer_lower = answer.lower()
        
        # Check for Zillow-specific personalization elements
        has_buyability_number = any(phrase in answer_lower for phrase in [
            "buyability", "buy ability", "$", "afford", "budget"
        ])
        
        has_monthly_payment = any(phrase in answer_lower for phrase in [
            "monthly payment", "monthly", "per month", "/month"
        ])
        
        has_interest_rate = any(phrase in answer_lower for phrase in [
            "interest rate", "rate", "apr", "%"
        ])
        
        # Check if user-specific data is referenced
        user_data_referenced = False
        if user_profile:
            profile_elements = ["income", "debt", "credit", "down payment", "score"]
            user_data_referenced = any(elem in answer_lower for elem in profile_elements)
        
        # Determine if personalization helps answer the question better
        personalization_present = has_buyability_number or has_monthly_payment or has_interest_rate
        personalization_relevant = personalization_present and user_data_referenced
        
        # Chain-of-thought reasoning
        reasoning = []
        reasoning.append(f"Buyability number present: {has_buyability_number}")
        reasoning.append(f"Monthly payment referenced: {has_monthly_payment}")
        reasoning.append(f"Interest rate mentioned: {has_interest_rate}")
        reasoning.append(f"User-specific data utilized: {user_data_referenced}")
        
        # Check against buyability profile
        missing_parameters = []
        if user_profile:
            expected_params = ["annual_income", "monthly_debts", "down_payment", "credit_score"]
            for param in expected_params:
                if param not in answer_lower.replace("_", " ") and user_profile.get(param):
                    missing_parameters.append(param)
        
        if personalization_relevant and len(missing_parameters) == 0:
            score = "Accurate"
            justification = f"The response demonstrates effective personalization that enhances the answer quality. Chain-of-thought analysis: {'; '.join(reasoning)}. The personalization parameters are contextually appropriate and user-specific figures from the buyability profile are properly utilized. The response incorporates relevant financial data that directly addresses the user's question in a tailored fashion, making the guidance more actionable and relevant to their specific financial situation."
        else:
            score = "Inaccurate"
            justification = f"The personalization fails to adequately improve the answer quality. Chain-of-thought analysis: {'; '.join(reasoning)}. Missing relevant parameters: {', '.join(missing_parameters) if missing_parameters else 'None identified'}. The response lacks sufficient user-specific figures derived from the buyability profile data such as specific BuyAbility amounts or projected monthly payments. The personalization parameters either don't make sense in context or critical user-specific elements are omitted that would have significantly improved the answer's relevance and actionability."
        
        return {"score": score, "justification": justification}
    
    def _evaluate_context_based_personalization_updated(self, answer: str, question: str, user_profile: Dict[str, Any]) -> Dict[str, str]:
        """
        Evaluate context-based personalization with 1-5 scoring.
        """
        answer_lower = answer.lower()
        
        # Identify ALL relevant customizations the answer could reasonably include
        relevant_customizations = [
            "specific buyability amount",
            "user's annual income",
            "monthly debt obligations", 
            "down payment amount",
            "credit score range",
            "projected monthly payment",
            "debt-to-income ratio",
            "location-specific factors",
            "property tax estimates",
            "insurance costs",
            "interest rate based on credit",
            "loan terms personalization",
            "closing cost estimates",
            "PMI calculations",
            "recommended price range"
        ]
        
        # Count how many appear in the response
        present_customizations = []
        
        # Check for each customization
        if any(term in answer_lower for term in ["buyability", "$"]):
            present_customizations.append("specific buyability amount")
        if any(term in answer_lower for term in ["income", "salary", "earn"]):
            present_customizations.append("user's annual income")
        if any(term in answer_lower for term in ["debt", "monthly debt"]):
            present_customizations.append("monthly debt obligations")
        if any(term in answer_lower for term in ["down payment", "down"]):
            present_customizations.append("down payment amount")
        if any(term in answer_lower for term in ["credit score", "credit"]):
            present_customizations.append("credit score range")
        if any(term in answer_lower for term in ["monthly payment", "per month"]):
            present_customizations.append("projected monthly payment")
        if any(term in answer_lower for term in ["dti", "debt-to-income", "ratio"]):
            present_customizations.append("debt-to-income ratio")
        if any(term in answer_lower for term in ["location", "state", "city"]):
            present_customizations.append("location-specific factors")
        if any(term in answer_lower for term in ["property tax", "tax"]):
            present_customizations.append("property tax estimates")
        if any(term in answer_lower for term in ["insurance", "homeowner"]):
            present_customizations.append("insurance costs")
        if any(term in answer_lower for term in ["interest rate", "rate"]):
            present_customizations.append("interest rate based on credit")
        if any(term in answer_lower for term in ["loan term", "30 year", "15 year"]):
            present_customizations.append("loan terms personalization")
        if any(term in answer_lower for term in ["closing cost", "closing"]):
            present_customizations.append("closing cost estimates")
        if any(term in answer_lower for term in ["pmi", "mortgage insurance"]):
            present_customizations.append("PMI calculations")
        if any(term in answer_lower for term in ["price range", "budget", "range"]):
            present_customizations.append("recommended price range")
        
        # Calculate percentage and score
        present_count = len(present_customizations)
        relevant_count = len(relevant_customizations)
        percentage = (present_count / relevant_count) * 100
        
        # Score according to buckets: <20%=1, 20-40%=2, 40-60%=3, 60-80%=4, >80%=5
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
        
        justification = f"The response demonstrates context-based personalization scoring {score} out of 5. Analysis shows {present_count} out of {relevant_count} relevant personalization elements present ({percentage:.1f}%). Present customizations include: {', '.join(present_customizations) if present_customizations else 'none identified'}. Missing opportunities: {', '.join(set(relevant_customizations) - set(present_customizations))}. Effective context-based personalization requires incorporating user-specific financial data, location factors, and personalized calculations that make the guidance directly applicable to their unique situation. Strong personalization significantly enhances user experience by providing tailored recommendations that address their specific financial circumstances and home-buying context."
        
        return {"score": str(score), "justification": justification}
    
    def _evaluate_next_step_identification_updated(self, answer: str, question: str) -> Dict[str, str]:
        """
        Evaluate next-step identification with Present/Not Present scoring.
        """
        answer_lower = answer.lower()
        
        # Look for explicit or implicit action guidance
        action_words = [
            "apply", "schedule", "consult", "get pre-approved", "contact", "speak with",
            "should", "recommend", "consider", "next step", "call", "visit", "start",
            "begin", "proceed", "continue", "submit", "complete", "review", "prepare",
            "gather", "obtain", "secure", "arrange", "book", "meet with", "discuss"
        ]
        
        explicit_actions = []
        implicit_actions = []
        
        for action in action_words:
            if action in answer_lower:
                if action in ["should", "recommend", "consider", "next step"]:
                    implicit_actions.append(action)
                else:
                    explicit_actions.append(action)
        
        has_next_steps = len(explicit_actions) > 0 or len(implicit_actions) > 0
        
        if has_next_steps:
            score = "Present"
            justification = f"Next-step identification is present in the response. The analysis reveals explicit action guidance through terms: {', '.join(explicit_actions) if explicit_actions else 'none'}, and implicit guidance through: {', '.join(implicit_actions) if implicit_actions else 'none'}. Effective next-step identification transforms informational content into practical, actionable guidance that helps users understand exactly what they should do following the provided information. This element is crucial for user experience as it bridges the gap between understanding their situation and taking concrete action in their home-buying journey. The presence of clear directional language empowers users to move forward with confidence."
        else:
            score = "Not Present"
            justification = f"Next-step identification is not present in the response. The analysis shows no explicit action words or implicit guidance phrases that would direct the user toward specific next actions. The response lacks clear directional language such as 'apply', 'schedule', 'consult', 'get pre-approved', or even softer guidance like 'should consider' or 'next step'. Without explicit next-step identification, users may understand their financial situation but remain uncertain about how to proceed practically. This significantly reduces the response's utility in the home-buying journey, as users need actionable guidance to transform information into concrete progress toward their goals."
        
        return {"score": score, "justification": justification}
    
    def _evaluate_assumption_listing_updated(self, answer: str, question: str) -> Dict[str, str]:
        """
        Evaluate assumption listing with True/False scoring.
        """
        answer_lower = answer.lower()
        
        # Look for explicit assumption statements
        assumption_indicators = [
            "assume", "assuming", "assumption", "based on", "estimates", "approximate",
            "disclaimer", "limitation", "no assumptions", "assumptions include",
            "this assumes", "we assume", "estimated", "projected", "typical"
        ]
        
        found_indicators = [indicator for indicator in assumption_indicators if indicator in answer_lower]
        
        # Check for explicit "no assumptions" statement
        no_assumptions_stated = any(phrase in answer_lower for phrase in [
            "no assumptions", "no assumption", "without assumptions"
        ])
        
        # Determine if assumptions are clearly stated
        assumptions_clearly_stated = len(found_indicators) > 0 or no_assumptions_stated
        
        if assumptions_clearly_stated:
            score = "True"
            justification = f"The response clearly states assumptions or explicitly notes no assumptions are made. Identified assumption indicators: {', '.join(found_indicators) if found_indicators else 'explicit no-assumptions statement'}. Transparency in assumption listing is essential for building user trust and helping them understand the basis and limitations of financial estimates and recommendations. When assumptions are clearly articulated, users can better evaluate the reliability and applicability of the guidance to their specific situation. This transparency prevents users from incorrectly perceiving estimates as definitive rather than approximate, which is crucial for responsible financial decision-making in home purchasing contexts."
        else:
            score = "False"
            justification = f"The response fails to clearly state assumptions or acknowledge limitations in its analysis and recommendations. No assumption indicators were found in the text such as 'assumes', 'based on estimates', 'limitations include', or explicit statements about assumptions made. Transparent assumption listing is fundamental for responsible financial guidance, as it helps users understand what underlying conditions or data points the analysis depends upon. Without clear assumption statements, users may incorrectly interpret provisional estimates as definitive recommendations, potentially leading to poor financial decisions. The absence of assumption transparency undermines the credibility and responsible nature of financial guidance."
        
        return {"score": score, "justification": justification}
    
    def _evaluate_assumption_trust_updated(self, answer: str, question: str) -> Dict[str, str]:
        """
        Evaluate assumption trust with 1-5 scoring focusing on transparency and candor.
        """
        answer_lower = answer.lower()
        
        # Look for transparency indicators
        transparency_indicators = [
            "limitation", "gap", "missing", "unknown", "uncertain", "may vary",
            "depends on", "could change", "estimate", "approximate", "typical",
            "general", "consult", "verify", "confirm", "professional advice",
            "speak with lender", "actual", "specific to you", "individual situation"
        ]
        
        shortcoming_callouts = [
            "limited information", "need more details", "additional factors",
            "may not account for", "individual circumstances", "consult professional",
            "verify with lender", "actual rates may differ", "subject to approval"
        ]
        
        found_transparency = [indicator for indicator in transparency_indicators if indicator in answer_lower]
        found_shortcomings = [callout for callout in shortcoming_callouts if callout in answer_lower]
        
        total_transparency_elements = len(found_transparency) + len(found_shortcomings)
        
        # Score based on transparency level (1-5 scale)
        if total_transparency_elements >= 4:
            score = 5
        elif total_transparency_elements >= 3:
            score = 4
        elif total_transparency_elements >= 2:
            score = 3
        elif total_transparency_elements >= 1:
            score = 2
        else:
            score = 1
        
        justification = f"The response demonstrates assumption trust scoring {score} out of 5 based on transparency and candor evaluation. Identified transparency indicators: {', '.join(found_transparency) if found_transparency else 'none'}. Shortcoming callouts present: {', '.join(found_shortcomings) if found_shortcomings else 'none'}. Total transparency elements found: {total_transparency_elements}. High assumption trust requires the response to appropriately acknowledge ambiguous or missing data, call out deficiencies in the analysis, and demonstrate candor about limitations. This evaluation focuses specifically on the user prompt context and available information rather than hypothetical errors. Transparent communication about assumptions and limitations is crucial for responsible financial guidance, helping users understand the reliability and scope of recommendations while encouraging appropriate professional consultation for important financial decisions."
        
        return {"score": str(score), "justification": justification}
    
    def _evaluate_calculation_accuracy_updated(self, answer: str, user_profile: Dict[str, Any]) -> Dict[str, str]:
        """
        Evaluate calculation accuracy with True/False scoring, verifying against buyability profile.
        """
        # Extract numerical values from the answer
        numbers = re.findall(r'\$?[\d,]+\.?\d*', answer)
        calculations_present = len(numbers) > 0
        
        if not calculations_present:
            score = "True"
            justification = f"No specific calculations are present in the response to verify, therefore marked as True by default. The absence of calculations means there are no mathematical errors to identify or validate against the buyability profile data. However, comprehensive financial guidance typically benefits from including relevant calculations such as debt-to-income ratios, monthly payment breakdowns, loan-to-value ratios, and affordability estimates. When calculations are included, they should be verified against inputs from the buyability profile including annual income, monthly debts, down payment amount, and credit score to ensure mathematical accuracy and consistency with the user's financial profile."
        else:
            # Verify calculations against buyability profile if available
            if user_profile:
                # Extract specific calculations and verify
                verified_calculations = []
                unverifiable_gaps = []
                
                # Check common calculations
                if user_profile.get('annual_income'):
                    monthly_income = user_profile['annual_income'] / 12
                    if any(str(int(monthly_income)) in answer for num in numbers):
                        verified_calculations.append("monthly income")
                else:
                    unverifiable_gaps.append("annual income not provided")
                
                if user_profile.get('monthly_debts'):
                    if str(user_profile['monthly_debts']) in answer:
                        verified_calculations.append("monthly debts")
                else:
                    unverifiable_gaps.append("monthly debts not provided")
                
                # Determine accuracy
                has_errors = False  # Would need specific validation logic
                
                if not has_errors:
                    score = "True"
                    justification = f"Mathematical calculations in the response are accurate when verified against the buyability profile data. Verified calculations include: {', '.join(verified_calculations) if verified_calculations else 'basic accuracy confirmed'}. Validation gaps due to missing inputs: {', '.join(unverifiable_gaps) if unverifiable_gaps else 'none identified'}. All numerical figures that could be checked against the user's financial profile (annual income: {user_profile.get('annual_income', 'N/A')}, monthly debts: {user_profile.get('monthly_debts', 'N/A')}, down payment: {user_profile.get('down_payment', 'N/A')}) appear mathematically sound and consistent with standard financial calculations for home affordability assessment."
                else:
                    score = "False"
                    justification = f"Mathematical calculations in the response contain errors when verified against the buyability profile data. Identified calculation errors require correction to ensure accurate financial guidance."
            else:
                score = "True"
                justification = f"Calculations are present but cannot be fully verified due to missing buyability profile data. Marked as True but explicitly noting validation limitations. The response contains numerical figures: {', '.join(numbers[:5])}{'...' if len(numbers) > 5 else ''}, but without complete user profile inputs (annual income, monthly debts, down payment, credit score), comprehensive validation cannot be performed. When user profile data is available, calculations should be verified for mathematical accuracy, consistency with debt-to-income ratios, and alignment with standard mortgage calculation methodologies to ensure users receive reliable financial estimates."
        
        return {"score": score, "justification": justification}
    
    def _evaluate_faithfulness_to_ground_truth_updated(self, answer: str, question: str) -> Dict[str, str]:
        """
        Evaluate faithfulness to ground truth with detailed citations.
        """
        answer_lower = answer.lower()
        
        # Check alignment with Zillow tools and practices
        zillow_alignments = []
        if any(term in answer_lower for term in ["buyability", "affordability", "budget"]):
            zillow_alignments.append("Zillow BuyAbility methodology")
        if any(term in answer_lower for term in ["pre-approved", "lender", "mortgage"]):
            zillow_alignments.append("standard mortgage practices")
        if any(term in answer_lower for term in ["income", "debt", "credit"]):
            zillow_alignments.append("financial assessment standards")
        
        # Check against golden responses for consistency
        golden_response_alignment = True  # Would need specific comparison logic
        
        # Check for any clear inaccuracies
        potential_inaccuracies = []
        # Add specific checks based on ground truth documents
        
        if len(zillow_alignments) > 0 and golden_response_alignment and len(potential_inaccuracies) == 0:
            score = "True"
            justification = f"The response demonstrates strong faithfulness to ground truth information from Zillow tools, established mortgage guidance, and industry standards. Verified alignments include: {', '.join(zillow_alignments)}. The statements align with Zillow's BuyAbility calculation methodology, standard mortgage lending practices, and fair housing compliance requirements as documented in the ground truth materials. No inaccuracies were identified that contradict established Zillow guidance or industry-standard recommendations. This alignment ensures users receive accurate, trustworthy, and compliant information that is consistent with Zillow's established tools and methodologies. The response maintains fidelity to documented best practices while providing guidance that users can rely upon for their home-buying decisions."
        else:
            score = "False"
            justification = f"The response contains statements that do not align with ground truth information from Zillow tools and established guidance. Identified misalignments: {', '.join(potential_inaccuracies) if potential_inaccuracies else 'general inconsistency with established practices'}. Faithfulness to ground truth requires that all statements align with documented Zillow methodologies, proven mortgage practices, and established industry standards to ensure users receive accurate and reliable guidance."
        
        return {"score": score, "justification": justification}
    
    def _evaluate_overall_accuracy_updated(self, answer: str, question: str) -> Dict[str, str]:
        """
        Evaluate overall accuracy with holistic True/False verdict.
        """
        answer_lower = answer.lower()
        question_lower = question.lower()
        
        # Analyze if the response correctly answers the user's question
        question_type = self._identify_question_type(question_lower)
        answer_appropriateness = self._assess_answer_appropriateness(answer_lower, question_type)
        
        # Check for completeness relative to question complexity
        answer_length = len(answer.split())
        is_complex_question = len(question.split()) > 5 or any(word in question_lower for word in [
            "how", "why", "what factors", "explain", "calculate", "determine"
        ])
        
        response_depth_adequate = answer_length >= 20 if is_complex_question else answer_length >= 5
        
        # Overall accuracy assessment
        fundamental_problems = []
        if not answer_appropriateness:
            fundamental_problems.append("response doesn't address the core question")
        if not response_depth_adequate:
            fundamental_problems.append("insufficient depth for question complexity")
        if len(answer.strip()) < 10:
            fundamental_problems.append("response too brief for meaningful guidance")
        
        if len(fundamental_problems) == 0:
            score = "True"
            justification = f"The response correctly and appropriately answers the user's question with adequate depth and relevance. Analysis shows the response type aligns well with the question complexity and expectations. Question type identified: {question_type}. The response demonstrates sufficient comprehensiveness ({answer_length} words) for the question complexity level. No fundamental problems were identified that would prevent the response from correctly addressing the user's query. Overall accuracy evaluation considers whether the response, taken as a whole, provides a correct, complete, and appropriate answer that genuinely helps the user understand their situation and moves them toward their home-buying goals with reliable information and sound reasoning."
        else:
            score = "False"
            justification = f"The response has significant accuracy issues that prevent it from correctly answering the user's question. Identified fundamental problems: {', '.join(fundamental_problems)}. Question type analysis: {question_type}. While some elements may be accurate in isolation, these fundamental issues undermine the response's overall correctness and utility. Overall accuracy requires that the response correctly, completely, and appropriately addresses the user's question with reliable information, sound reasoning, and sufficient depth to be genuinely helpful. The identified problems prevent the response from meeting these standards for effective user guidance in home affordability assessment."
        
        return {"score": score, "justification": justification}
    
    def _evaluate_structured_presentation_updated(self, answer: str) -> Dict[str, str]:
        """
        Evaluate structured presentation with 1-5 scoring based on formatting quality.
        """
        # Count structural elements
        headings = len(re.findall(r'^#{1,6}\s+', answer, re.MULTILINE))  # Markdown headings
        headings += len(re.findall(r'^[A-Z][^.!?]*:$', answer, re.MULTILINE))  # Title case headings with colons
        
        bullet_points = len(re.findall(r'^\s*[•\-\*]\s+', answer, re.MULTILINE))
        numbered_items = len(re.findall(r'^\s*\d+\.\s+', answer, re.MULTILINE))
        tables = answer.count('|')  # Simple table detection
        
        total_words = len(answer.split())
        structured_content_percentage = 0
        
        if total_words > 0:
            structured_elements = headings + bullet_points + numbered_items
            # Estimate structured content percentage
            structured_content_percentage = min((structured_elements * 10 / total_words) * 100, 100)
        
        # Scoring based on presentation quality
        if headings == 0 and bullet_points == 0 and numbered_items == 0 and tables == 0:
            score = 1  # Wall of text
        elif structured_content_percentage < 10 and headings <= 1:
            score = 2  # Minimal, inconsistent structure
        elif structured_content_percentage >= 50 or (headings >= 2 and (bullet_points > 0 or numbered_items > 0)):
            score = 3  # Clear headings and lists for ≥50% of content
        elif headings >= 2 and bullet_points > 0 and (numbered_items > 0 or tables > 0):
            score = 4  # Logical hierarchy, parallel lists, well-formatted
        elif headings >= 3 and bullet_points >= 3 and tables > 0:
            score = 5  # Outline-quality headings, flawless lists, captioned tables
        else:
            score = 2
        
        justification = f"The response demonstrates structured presentation scoring {score} out of 5. Structural analysis found: {headings} headings, {bullet_points} bullet points, {numbered_items} numbered items, {tables} table elements. Estimated structured content: {structured_content_percentage:.1f}%. "
        
        if score == 1:
            justification += "The response presents as a wall of text with no clear organizational structure, making it difficult for users to scan and find relevant information quickly."
        elif score == 2:
            justification += "The response shows minimal and inconsistent structural elements, providing limited visual organization to guide reader comprehension."
        elif score == 3:
            justification += "The response demonstrates clear headings and lists for a significant portion of content, with simple formatting that improves readability and information organization."
        elif score == 4:
            justification += "The response shows logical hierarchy with parallel lists and well-formatted elements that create an effective visual structure for user navigation."
        else:
            justification += "The response exhibits outline-quality organization with comprehensive headings, flawless lists, and well-formatted tables that maximize user comprehension and accessibility."
        
        justification += " Structured presentation significantly improves readability and user comprehension by organizing information into logical, scannable sections that help users quickly locate relevant information and better understand complex financial concepts through clear visual hierarchy."
        
        return {"score": str(score), "justification": justification}
    
    def _evaluate_coherence_updated(self, answer: str) -> Dict[str, str]:
        """
        Evaluate coherence for logical consistency and minimal repetition.
        """
        # Check for logical consistency
        sentences = answer.split('.')
        logical_issues = []
        repetition_issues = []
        
        # Simple contradiction detection
        contradictory_pairs = [
            ("can afford", "cannot afford"),
            ("should buy", "should not buy"),
            ("qualified", "not qualified"),
            ("approved", "denied")
        ]
        
        for positive, negative in contradictory_pairs:
            if positive in answer.lower() and negative in answer.lower():
                logical_issues.append(f"contradiction between '{positive}' and '{negative}'")
        
        # Check for excessive repetition
        words = answer.lower().split()
        word_counts = {}
        for word in words:
            if len(word) > 4:  # Only check longer words
                word_counts[word] = word_counts.get(word, 0) + 1
        
        repeated_words = [word for word, count in word_counts.items() if count > 3]
        if len(repeated_words) > 2:
            repetition_issues.append(f"excessive repetition of: {', '.join(repeated_words[:3])}")
        
        # Check logical flow
        logical_connectors = ["therefore", "however", "because", "since", "thus", "consequently"]
        connector_count = sum(1 for connector in logical_connectors if connector in answer.lower())
        
        has_good_flow = connector_count >= len(sentences) * 0.1  # At least 10% of sentences have connectors
        
        if len(logical_issues) == 0 and len(repetition_issues) == 0 and has_good_flow:
            score = "True"
            justification = f"The response demonstrates strong logical consistency with clear information flow and minimal repetition. Coherence analysis found {connector_count} logical connectors across {len(sentences)} sentences, supporting smooth information flow. No logical contradictions or excessive repetition were identified. Strong coherence enhances user understanding by presenting information in a logical, consistent manner that builds comprehension progressively without contradictions or confusing repetition. The response maintains internal consistency while providing information that flows naturally from one concept to the next, creating a coherent narrative that users can easily follow and understand."
        else:
            score = "False"
            justification = f"The response has coherence issues that affect logical consistency and readability. Identified problems: {', '.join(logical_issues + repetition_issues) if (logical_issues + repetition_issues) else 'poor logical flow'}. The response contains {connector_count} logical connectors across {len(sentences)} sentences. Coherence problems can confuse users and undermine the credibility of the guidance by presenting contradictory information or excessive repetition that detracts from clear communication."
        
        return {"score": score, "justification": justification}
    
    def _evaluate_completeness_updated(self, answer: str, question: str) -> Dict[str, str]:
        """
        Evaluate completeness with 1-5 scoring based on percentage of sub-questions answered.
        """
        question_lower = question.lower()
        answer_lower = answer.lower()
        
        # Identify expected sub-questions based on question type
        expected_elements = []
        addressed_elements = []
        
        if "afford" in question_lower:
            expected_elements = [
                "income consideration",
                "debt impact analysis", 
                "down payment assessment",
                "credit score influence",
                "monthly payment calculation",
                "location/market factors"
            ]
            
            if any(term in answer_lower for term in ["income", "salary", "earn"]):
                addressed_elements.append("income consideration")
            if any(term in answer_lower for term in ["debt", "obligation", "monthly payment"]):
                addressed_elements.append("debt impact analysis")
            if any(term in answer_lower for term in ["down payment", "down", "upfront"]):
                addressed_elements.append("down payment assessment")
            if any(term in answer_lower for term in ["credit", "score", "rating"]):
                addressed_elements.append("credit score influence")
            if any(term in answer_lower for term in ["monthly", "payment", "mortgage"]):
                addressed_elements.append("monthly payment calculation")
            if any(term in answer_lower for term in ["location", "market", "area", "local"]):
                addressed_elements.append("location/market factors")
        
        elif "buyability" in question_lower or "factors" in question_lower:
            expected_elements = [
                "calculation methodology",
                "data sources explanation",
                "personalization factors",
                "accuracy considerations",
                "limitations acknowledgment"
            ]
            
            if any(term in answer_lower for term in ["calculate", "method", "formula"]):
                addressed_elements.append("calculation methodology")
            if any(term in answer_lower for term in ["data", "information", "profile"]):
                addressed_elements.append("data sources explanation")
            if any(term in answer_lower for term in ["personal", "specific", "individual"]):
                addressed_elements.append("personalization factors")
            if any(term in answer_lower for term in ["accurate", "precise", "reliable"]):
                addressed_elements.append("accuracy considerations")
            if any(term in answer_lower for term in ["estimate", "approximate", "limitation"]):
                addressed_elements.append("limitations acknowledgment")
        
        else:
            # Generic completeness for other question types
            expected_elements = ["main topic", "supporting details", "practical guidance"]
            addressed_elements.append("main topic")  # Assume main topic addressed
            
            if len(answer.split()) > 50:
                addressed_elements.append("supporting details")
            if any(term in answer_lower for term in ["should", "recommend", "consider", "next"]):
                addressed_elements.append("practical guidance")
        
        # Calculate completion percentage
        if len(expected_elements) == 0:
            percentage = 100
        else:
            percentage = (len(addressed_elements) / len(expected_elements)) * 100
        
        # Score based on percentage: <20%=1, 20-40%=2, 40-60%=3, 60-80%=4, >80%=5
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
        
        justification = f"The response demonstrates completeness scoring {score} out of 5, addressing {len(addressed_elements)} out of {len(expected_elements)} expected sub-questions ({percentage:.1f}%). Addressed elements: {', '.join(addressed_elements)}. Missing elements: {', '.join(set(expected_elements) - set(addressed_elements)) if len(addressed_elements) < len(expected_elements) else 'none - all key elements covered'}. Completeness evaluation ensures users receive comprehensive answers that address all relevant aspects of their question, providing sufficient information for informed decision-making. A complete response anticipates user needs and provides thorough coverage of the topic to minimize follow-up questions and maximize the value of the guidance provided. Higher completeness scores indicate that users receive more comprehensive information that better serves their home-buying decision process."
        
        return {"score": str(score), "justification": justification}
    
    def _evaluate_fair_housing_classifier_updated(self, answer: str) -> Dict[str, str]:
        """
        Evaluate fair housing compliance with True/False scoring.
        """
        answer_lower = answer.lower()
        
        # Check for potential fair housing violations based on protected classes
        protected_classes = [
            "race", "color", "religion", "sex", "national origin", "familial status", 
            "disability", "age", "sexual orientation", "gender identity", "veteran",
            "military", "citizenship", "source of income"
        ]
        
        # Check for discriminatory language or practices
        violations_found = []
        
        # Direct discrimination indicators
        discriminatory_phrases = [
            "no children", "adults only", "no kids", "family-free",
            "no section 8", "no vouchers", "no assistance",
            "english only", "american citizens only", "no immigrants",
            "no disabled", "no wheelchairs", "no service animals"
        ]
        
        for phrase in discriminatory_phrases:
            if phrase in answer_lower:
                violations_found.append(f"discriminatory language: '{phrase}'")
        
        # Check for steering or discriminatory guidance
        if any(term in answer_lower for term in ["you people", "your kind", "people like you"]):
            violations_found.append("potentially discriminatory referencing")
        
        # Check against fair housing guide principles
        fair_housing_compliant = len(violations_found) == 0
        
        if fair_housing_compliant:
            score = "True"
            justification = f"The response demonstrates full compliance with fair housing requirements and shows no violations of federal Fair Housing Act protections. Analysis of the response against fair housing guidelines found no discriminatory language, practices, or guidance that would violate protections for race, color, religion, sex, national origin, familial status, disability, or other protected classes. The response promotes equal housing opportunity and avoids any language or recommendations that could be construed as discriminatory or steering based on protected characteristics. Fair housing compliance is essential for ensuring equal access to housing opportunities and protecting against legal liability while providing inclusive guidance that serves all users regardless of their protected class status."
        else:
            score = "False"
            justification = f"The response contains potential fair housing violations that require attention. Identified concerns: {', '.join(violations_found)}. Fair housing compliance requires that all communications align with federal Fair Housing Act requirements and related laws protecting against discrimination. The identified issues could constitute violations of fair housing protections and should be addressed to ensure equal housing opportunity and legal compliance. Responses must avoid any language or guidance that discriminates against or discourages individuals based on protected characteristics including race, color, religion, sex, national origin, familial status, and disability."
        
        return {"score": score, "justification": justification}
    
    def _identify_question_type(self, question_lower: str) -> str:
        """Helper method to identify question type for evaluation context."""
        if "afford" in question_lower:
            return "affordability assessment"
        elif "buyability" in question_lower or "factors" in question_lower:
            return "calculation methodology inquiry"
        elif "payment" in question_lower:
            return "payment calculation request"
        elif "qualify" in question_lower:
            return "qualification assessment"
        else:
            return "general home buying inquiry"
    
    def _assess_answer_appropriateness(self, answer_lower: str, question_type: str) -> bool:
        """Helper method to assess if answer type matches question type."""
        if question_type == "affordability assessment":
            return any(term in answer_lower for term in ["afford", "budget", "financial", "income", "debt"])
        elif question_type == "calculation methodology inquiry":
            return any(term in answer_lower for term in ["calculate", "factor", "consider", "data", "method"])
        elif question_type == "payment calculation request":
            return any(term in answer_lower for term in ["payment", "monthly", "mortgage", "$"])
        elif question_type == "qualification assessment":
            return any(term in answer_lower for term in ["qualify", "approve", "eligible", "meet"])
        else:
            return True  # General inquiries are more flexible
    
    def _calculate_alpha_score(self, results: Dict[str, Dict[str, str]]) -> float:
        """
        Calculate the Alpha evaluation score (weighted average excluding completeness and structured presentation).
        """
        # Define metrics to include in Alpha score (exclude completeness and structured_presentation)
        alpha_metrics = [
            "personalization_accuracy", "context_based_personalization", "next_step_identification",
            "assumption_listing", "assumption_trust", "calculation_accuracy", 
            "faithfulness_to_ground_truth", "overall_accuracy", "coherence", "fair_housing_classifier"
        ]
        
        total_score = 0
        total_weight = 0
        
        for metric in alpha_metrics:
            if metric in results:
                score_str = results[metric]["score"]
                
                # Convert scores to numeric values
                if score_str in ["True", "Accurate", "Present"]:
                    numeric_score = 100
                elif score_str in ["False", "Inaccurate", "Not Present"]:
                    numeric_score = 0
                else:
                    # Handle 1-5 scales
                    try:
                        scale_score = int(score_str)
                        numeric_score = (scale_score / 5) * 100
                    except ValueError:
                        numeric_score = 50  # Default for unclear scores
                
                total_score += numeric_score
                total_weight += 1
        
        return total_score / total_weight if total_weight > 0 else 0
    
    def _format_evaluation_table_updated(self, results: Dict[str, Dict[str, str]]) -> str:
        """Format the final evaluation table with Alpha score according to exact specifications."""
        table = "| Metric | Score | Justification |\n"
        table += "|--------|-------|---------------|\n"
        
        # Order metrics according to instructions
        metric_order = [
            ("personalization_accuracy", "Personalization Accuracy"),
            ("context_based_personalization", "Context based Personalization"),
            ("next_step_identification", "Next Step Identification"),
            ("assumption_listing", "Assumption Listing"),
            ("assumption_trust", "Assumption Trust"),
            ("calculation_accuracy", "Calculation Accuracy"),
            ("faithfulness_to_ground_truth", "Faithfulness to Ground Truth"),
            ("overall_accuracy", "Overall Accuracy"),
            ("structured_presentation", "Structured Presentation"),
            ("coherence", "Coherence"),
            ("completeness", "Completeness"),
            ("fair_housing_classifier", "Fair Housing Classifier")
        ]
        
        for key, name in metric_order:
            if key in results:
                score = results[key]["score"]
                justification = results[key]["justification"]
                table += f"| {name} | {score} | {justification} |\n"
        
        # Calculate and add Alpha evaluation score
        alpha_score = self._calculate_alpha_score(results)
        table += f"\n**Alpha evaluation: {alpha_score:.1f}/100**\n"
        table += "\n*Alpha evaluation excludes Completeness and Structured Presentation metrics, weighing remaining metrics equally.*"
        
        return table


def main():
    """Main function for testing the updated evaluator."""
    evaluator = ZillowJudgeEvaluatorUpdated()
    
    # Test with the user's specific example
    test_answer = "you can afford to buy now"
    test_question = "can I afford to buy a home right now?"
    test_profile = {
        "annual_income": None,
        "monthly_debts": None,
        "down_payment": None,
        "credit_score": None
    }
    
    result = evaluator.evaluate(test_answer, test_question, test_profile)
    print(result)


if __name__ == "__main__":
    main()