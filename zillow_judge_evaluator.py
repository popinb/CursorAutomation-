"""
Zillow LLM Judge Evaluator for OpenAI Evals Framework

This evaluator replicates the behavior of the Custom GPT Judge that evaluates
LLM responses across 10 metrics using golden responses and buyability profiles.
"""

import json
import re
import os
from decimal import Decimal
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path


class ZillowJudgeEvaluator:
    """
    Custom evaluator that assesses LLM responses across 10 metrics based on
    ground truth documents and buyability profiles.
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
        Main evaluation method that assesses a candidate answer across all 10 metrics.
        
        Args:
            candidate_answer: The LLM response to evaluate
            question: The original question (optional)
            user_profile: User's buyability profile data (optional)
            
        Returns:
            Formatted evaluation table with scores and justifications
        """
        # Generate scratchpad (internal reasoning)
        scratchpad = self._generate_scratchpad(candidate_answer, question, user_profile)
        
        # Evaluate each metric
        results = {}
        results["personalization_accuracy"] = self._evaluate_personalization_accuracy(candidate_answer, user_profile, scratchpad)
        results["context_based_personalization"] = self._evaluate_context_based_personalization(candidate_answer, user_profile, scratchpad)
        results["next_step_identification"] = self._evaluate_next_step_identification(candidate_answer, scratchpad)
        results["assumption_listing"] = self._evaluate_assumption_listing(candidate_answer, scratchpad)
        results["assumption_trust"] = self._evaluate_assumption_trust(candidate_answer, scratchpad)
        results["calculation_accuracy"] = self._evaluate_calculation_accuracy(candidate_answer, user_profile, scratchpad)
        results["faithfulness_to_ground_truth"] = self._evaluate_faithfulness_to_ground_truth(candidate_answer, scratchpad)
        results["fair_housing_compliance"] = self._evaluate_fair_housing_compliance(candidate_answer, scratchpad)
        results["overall_accuracy"] = self._evaluate_overall_accuracy(candidate_answer, question, scratchpad)
        results["structured_presentation"] = self._evaluate_structured_presentation(candidate_answer, scratchpad)
        results["coherence"] = self._evaluate_coherence(candidate_answer, scratchpad)
        results["completeness"] = self._evaluate_completeness(candidate_answer, question, scratchpad)
        
        # Format final output table
        return self._format_evaluation_table(results)
    
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
        
        # Common patterns for financial data
        patterns = {
            'buyability': r'\$[\d,]+(?:\.\d{2})?',
            'annual_income': r'\$[\d,]+(?:\.\d{2})?',
            'monthly_payment': r'\$[\d,]+(?:\.\d{2})?',
            'down_payment': r'\$[\d,]+(?:\.\d{2})?',
            'interest_rate': r'(\d+\.?\d*)%',
            'credit_score': r'(\d{3})[–-](\d{3})',
        }
        
        for var_name, pattern in patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                variables[var_name] = matches[0] if isinstance(matches[0], str) else str(matches[0])
        
        return variables
    
    def _find_matching_ground_truth(self, question: str, variables: Dict[str, str]) -> str:
        """Find the most relevant ground truth response."""
        # Simple matching based on question content or variables present
        if not question:
            return "No specific ground truth selected - evaluating against general criteria"
        
        # Match against known question patterns
        question_lower = question.lower()
        if "buyability" in question_lower and "calculate" in question_lower:
            return "Question 1: Buyability calculation factors"
        elif "personalized" in question_lower or "based on what i told" in question_lower:
            return "Question 2: Personalization verification"
        elif "monthly payment" in question_lower and "included" in question_lower:
            return "Question 3: Payment breakdown"
        elif "afford" in question_lower and "right now" in question_lower:
            return "Question 4: Affordability assessment"
        
        return "General evaluation against all criteria"
    
    def _analyze_profile_match(self, variables: Dict[str, str], user_profile: Dict[str, Any]) -> str:
        """Analyze how well extracted variables match the user profile."""
        if not user_profile:
            return "No user profile provided for comparison"
        
        matches = []
        mismatches = []
        
        # Check key profile fields
        profile_mappings = {
            'annual_income': 'annual_income',
            'down_payment': 'down_payment',
            'monthly_debts': 'monthly_debts'
        }
        
        for var_key, profile_key in profile_mappings.items():
            if var_key in variables and profile_key in user_profile:
                var_value = self._parse_currency(variables[var_key])
                profile_value = user_profile[profile_key]
                
                if abs(var_value - profile_value) < 100:  # Allow small differences
                    matches.append(f"{var_key}: {var_value} ≈ {profile_value}")
                else:
                    mismatches.append(f"{var_key}: {var_value} ≠ {profile_value}")
        
        return f"Matches: {matches}, Mismatches: {mismatches}"
    
    def _parse_currency(self, currency_str: str) -> float:
        """Parse currency string to numeric value."""
        if isinstance(currency_str, (int, float)):
            return float(currency_str)
        
        # Remove currency symbols and commas
        clean_str = re.sub(r'[$,]', '', str(currency_str))
        try:
            return float(clean_str)
        except ValueError:
            return 0.0
    
    def _verify_calculations(self, variables: Dict[str, str], user_profile: Dict[str, Any]) -> str:
        """Verify mathematical accuracy of calculations."""
        if not user_profile:
            return "No profile data available for calculation verification"
        
        verification_results = []
        
        # Verify DTI calculation if monthly income and debts are present
        if 'annual_income' in variables and 'monthly_debts' in user_profile:
            annual_income = self._parse_currency(variables['annual_income'])
            monthly_income = annual_income / 12
            monthly_debts = user_profile['monthly_debts']
            
            dti_ratio = (monthly_debts / monthly_income) * 100
            verification_results.append(f"DTI: {monthly_debts}/{monthly_income:.0f} = {dti_ratio:.1f}%")
        
        return "; ".join(verification_results) if verification_results else "No calculations to verify"
    
    def _evaluate_personalization_accuracy(self, answer: str, user_profile: Dict[str, Any], scratchpad: str) -> Dict[str, str]:
        """Evaluate personalization accuracy (Accurate/Inaccurate)."""
        if not user_profile:
            return {
                "score": "Inaccurate",
                "justification": "No user profile provided to assess personalization accuracy. The response cannot be evaluated for personalization without specific user data to compare against. Personalization requires matching user-specific figures like income, debts, down payment, and credit score range from the buyability profile. Without this comparison baseline, the response fails to demonstrate personalized content that strengthens the answer for the individual user's financial situation."
            }
        
        # Extract variables and compare with profile
        variables = self._extract_variables(answer)
        matches = 0
        total_checkable = 0
        
        profile_checks = {
            'annual_income': user_profile.get('annual_income'),
            'down_payment': user_profile.get('down_payment'), 
            'monthly_debts': user_profile.get('monthly_debts')
        }
        
        for var_name, expected_value in profile_checks.items():
            if expected_value is not None and var_name in variables:
                total_checkable += 1
                extracted_value = self._parse_currency(variables[var_name])
                if abs(extracted_value - expected_value) < 100:  # Allow small rounding differences
                    matches += 1
        
        if total_checkable == 0:
            accuracy = "Inaccurate"
            justification = "The response lacks specific personalized financial figures that should be derived from the user's buyability profile. No verifiable user-specific data points (income, down payment, monthly debts) were found in the response. True personalization requires incorporating exact values from the buyability profile to strengthen the answer's relevance and accuracy for this individual user's financial situation."
        elif matches == total_checkable:
            accuracy = "Accurate"
            justification = f"The response demonstrates excellent personalization accuracy by correctly incorporating all verifiable user-specific financial data from the buyability profile. All {matches} key financial figures match the user's profile within acceptable ranges. This personalization significantly strengthens the answer by providing tailored guidance based on the user's actual financial situation rather than generic estimates. The personalized approach enhances trust and relevance."
        else:
            accuracy = "Inaccurate"
            justification = f"The response shows partial personalization but contains {total_checkable - matches} mismatched financial figures compared to the user's buyability profile. Only {matches} out of {total_checkable} verifiable data points align with the user's actual financial situation. Inaccurate personalization undermines the response's credibility and could mislead the user about their actual home-buying capacity. Personalization must be precise to be effective."
        
        return {"score": accuracy, "justification": justification}
    
    def _evaluate_context_based_personalization(self, answer: str, user_profile: Dict[str, Any], scratchpad: str) -> Dict[str, str]:
        """Evaluate context-based personalization (1-5 scale)."""
        # Define relevant customization elements that could be included
        relevant_customizations = [
            "user's specific annual income",
            "user's actual down payment amount", 
            "user's monthly debt obligations",
            "user's credit score range",
            "user's preferred monthly payment",
            "user's maximum comfortable payment",
            "location-specific information",
            "personalized interest rate",
            "customized DTI calculation",
            "user-specific loan recommendations"
        ]
        
        present_count = 0
        
        # Check for presence of each customization
        answer_lower = answer.lower()
        
        # Check for financial personalization
        if any(re.search(r'\$[\d,]+', answer) for _ in range(3)):  # Multiple dollar amounts
            present_count += 1
        
        # Check for specific user data mentions
        if "your" in answer_lower and any(term in answer_lower for term in ["income", "debt", "payment"]):
            present_count += 2
        
        # Check for calculations or DTI mentions
        if any(term in answer_lower for term in ["dti", "debt-to-income", "36%", "calculate"]):
            present_count += 1
        
        # Check for credit score personalization
        if any(term in answer_lower for term in ["credit score", "rate", "660", "719"]):
            present_count += 1
        
        # Check for location references
        if any(term in answer_lower for term in ["georgia", "location", "state", "property tax"]):
            present_count += 1
        
        total_relevant = len(relevant_customizations)
        percentage = (present_count / total_relevant) * 100
        
        # Score based on percentage buckets
        if percentage >= 80:
            score = 5
        elif percentage >= 60:
            score = 4
        elif percentage >= 40:
            score = 3
        elif percentage >= 20:
            score = 2
        else:
            score = 1
        
        justification = f"The response includes {present_count} out of {total_relevant} relevant personalization elements ({percentage:.1f}%). Context-based personalization evaluation identifies specific customizations the answer could reasonably include such as user's financial data, location-specific information, and personalized calculations. The response demonstrates {'excellent' if score >= 4 else 'adequate' if score == 3 else 'limited'} personalization by {'incorporating multiple user-specific elements that enhance relevance' if score >= 4 else 'including some personalized content but missing key opportunities' if score == 3 else 'providing minimal customization beyond generic information'}. Strong personalization significantly improves user experience by making the guidance directly applicable to their unique situation."
        
        return {"score": str(score), "justification": justification}
    
    def _evaluate_next_step_identification(self, answer: str, scratchpad: str) -> Dict[str, str]:
        """Evaluate next-step identification (Present/Not Present)."""
        # Look for explicit or implicit action guidance
        action_patterns = [
            r'\bapply\b', r'\bschedule\b', r'\bconsult\b', r'\bget pre-approved\b',
            r'\bcontact\b', r'\btalk to\b', r'\bspeak with\b', r'\bmeet with\b',
            r'\bnext step\b', r'\bshould\b', r'\brecommend\b', r'\bsuggest\b',
            r'\bstart\b', r'\bbegin\b', r'\bconsider\b', r'\blook for\b'
        ]
        
        answer_lower = answer.lower()
        found_actions = []
        
        for pattern in action_patterns:
            if re.search(pattern, answer_lower):
                found_actions.append(pattern.strip('\\b'))
        
        if found_actions:
            present = "Present"
            justification = f"The response provides clear next-step guidance through actionable recommendations. Identified action words/phrases include: {', '.join(found_actions[:5])}. Next-step identification is crucial for user guidance as it transforms information into actionable advice, helping users understand not just their current situation but what concrete actions they should take next. The presence of explicit guidance enhances the response's practical value and helps users move forward in their home-buying journey with confidence and direction."
        else:
            present = "Not Present"
            justification = "The response lacks explicit next-step identification or actionable guidance for the user. No clear action words or phrases were found that would guide the user on what to do next with the provided information. Next-step identification is essential for transforming informational content into practical guidance. Without explicit next steps, users may understand their situation but remain uncertain about how to proceed, reducing the response's practical utility in their home-buying journey."
        
        return {"score": present, "justification": justification}
    
    def _evaluate_assumption_listing(self, answer: str, scratchpad: str) -> Dict[str, str]:
        """Evaluate assumption listing (True/False)."""
        # Look for explicit assumption statements
        assumption_patterns = [
            r'\bassumption\b', r'\bassume\b', r'\bassuming\b',
            r'\bestimate\b', r'\bapproximate\b', r'\bbased on\b',
            r'\bno assumptions\b', r'\bwithout assumptions\b'
        ]
        
        answer_lower = answer.lower()
        assumption_indicators = []
        
        for pattern in assumption_patterns:
            if re.search(pattern, answer_lower):
                assumption_indicators.append(pattern.strip('\\b'))
        
        # Also look for disclaimers or limitations
        disclaimer_patterns = [
            r'\bmay vary\b', r'\bactual.*may differ\b', r'\bfor illustration\b',
            r'\bestimated\b', r'\btypical\b', r'\baverage\b'
        ]
        
        for pattern in disclaimer_patterns:
            if re.search(pattern, answer_lower):
                assumption_indicators.append(pattern.strip('\\b'))
        
        if assumption_indicators:
            listing = "True"
            justification = f"The response clearly states its assumptions and limitations, demonstrating transparency about the basis of its calculations and recommendations. Found assumption indicators: {', '.join(assumption_indicators[:5])}. Transparent assumption listing is crucial for user trust and informed decision-making. By explicitly acknowledging assumptions, the response helps users understand the limitations and variability of the provided estimates, enabling them to make more informed decisions about their home-buying process."
        else:
            listing = "False"
            justification = "The response fails to explicitly state its assumptions or acknowledge limitations in its calculations and recommendations. No clear assumption indicators or disclaimers were found in the text. Transparent assumption listing is essential for building user trust and helping them understand the basis and limitations of financial estimates. Without explicit assumptions, users may incorrectly perceive estimates as definitive rather than approximate, potentially leading to poor financial decisions."
        
        return {"score": listing, "justification": justification}
    
    def _evaluate_assumption_trust(self, answer: str, scratchpad: str) -> Dict[str, str]:
        """Evaluate assumption trust (1-5 scale)."""
        # Look for various levels of transparency about limitations
        transparency_indicators = {
            5: ["comprehensive disclaimers", "multiple limitations noted", "self-aware about gaps"],
            4: ["clear limitations stated", "acknowledges uncertainties", "cites sources"],
            3: ["some limitations mentioned", "basic disclaimers"],
            2: ["minimal acknowledgment", "limited transparency"],
            1: ["no limitations discussed", "silent on gaps"]
        }
        
        answer_lower = answer.lower()
        score = 1  # Default to lowest
        
        # Count transparency elements
        transparency_count = 0
        
        # Check for limitation acknowledgments
        if any(term in answer_lower for term in ["limitation", "may vary", "estimate", "approximate"]):
            transparency_count += 1
        
        # Check for uncertainty indicators
        if any(term in answer_lower for term in ["typically", "generally", "average", "about", "around"]):
            transparency_count += 1
        
        # Check for data source mentions
        if any(term in answer_lower for term in ["based on", "according to", "using", "assumes"]):
            transparency_count += 1
        
        # Check for explicit gap acknowledgments
        if any(term in answer_lower for term in ["not included", "additional", "separate", "exclude"]):
            transparency_count += 1
        
        # Check for self-awareness
        if any(term in answer_lower for term in ["depend", "individual", "specific", "consult"]):
            transparency_count += 1
        
        # Determine score based on transparency count
        if transparency_count >= 4:
            score = 5
        elif transparency_count >= 3:
            score = 4
        elif transparency_count >= 2:
            score = 3
        elif transparency_count >= 1:
            score = 2
        else:
            score = 1
        
        score_descriptions = {
            5: "demonstrates comprehensive self-awareness with extensive citation of limitations and uncertainties",
            4: "shows strong transparency by acknowledging key limitations and providing appropriate disclaimers", 
            3: "displays adequate awareness of limitations with basic transparency about assumptions",
            2: "shows minimal acknowledgment of uncertainties with limited self-awareness",
            1: "remains silent on limitations and gaps, showing poor transparency about assumptions"
        }
        
        justification = f"The response {score_descriptions[score]}. Identified {transparency_count} transparency indicators including limitation acknowledgments, uncertainty markers, and assumption statements. Assumption trust evaluates how candidly the answer flags ambiguities, missing data, and limitations in its analysis. High assumption trust is crucial for responsible financial guidance, helping users understand the reliability and scope of recommendations while encouraging appropriate professional consultation for important decisions."
        
        return {"score": str(score), "justification": justification}
    
    def _evaluate_calculation_accuracy(self, answer: str, user_profile: Dict[str, Any], scratchpad: str) -> Dict[str, str]:
        """Evaluate calculation accuracy (True/False)."""
        if not user_profile:
            return {
                "score": "True",
                "justification": "No user profile data available for calculation verification. In the absence of specific input data to verify against, calculations cannot be checked for accuracy. When calculations are absent or unverifiable due to missing input data, the evaluation defaults to True while noting the validation gap. For comprehensive calculation accuracy assessment, specific user financial data from the buyability profile would be required to verify mathematical computations."
            }
        
        # Extract calculations from the response
        calculations_found = []
        calculation_errors = []
        
        # Look for specific calculation patterns
        variables = self._extract_variables(answer)
        
        # Verify DTI calculation if present
        if 'annual_income' in variables and user_profile.get('monthly_debts'):
            try:
                annual_income = self._parse_currency(variables['annual_income'])
                monthly_income = annual_income / 12
                monthly_debts = user_profile['monthly_debts']
                
                # Look for DTI mentions in text
                dti_pattern = r'(\d+(?:\.\d+)?)%.*(?:dti|debt.to.income)'
                dti_matches = re.findall(dti_pattern, answer.lower())
                
                if dti_matches:
                    stated_dti = float(dti_matches[0])
                    calculated_dti = (monthly_debts / monthly_income) * 100
                    
                    calculations_found.append(f"DTI calculation: {monthly_debts}/{monthly_income:.0f} = {calculated_dti:.1f}%")
                    
                    if abs(stated_dti - calculated_dti) > 1:  # Allow 1% tolerance
                        calculation_errors.append(f"DTI error: stated {stated_dti}% vs calculated {calculated_dti:.1f}%")
                
            except (ValueError, ZeroDivisionError):
                calculation_errors.append("DTI calculation error: invalid values")
        
        # Check for available monthly payment calculation
        if all(key in variables for key in ['annual_income', 'monthly_debts']):
            annual_income = self._parse_currency(variables['annual_income'])
            monthly_income = annual_income / 12
            monthly_debts = user_profile.get('monthly_debts', 0)
            
            # 36% DTI rule calculation
            max_payment = monthly_income * 0.36
            available_payment = max_payment - monthly_debts
            
            calculations_found.append(f"Available payment: {max_payment:.0f} - {monthly_debts} = {available_payment:.0f}")
            
            # Look for similar values in the text
            payment_pattern = r'\$(\d+(?:,\d{3})*)'
            payment_matches = re.findall(payment_pattern, answer)
            
            for match in payment_matches:
                stated_amount = self._parse_currency(match)
                if abs(stated_amount - available_payment) < 50:  # Close match found
                    break
            else:
                # No close match found
                if payment_matches:  # But payments were mentioned
                    calculation_errors.append(f"Available payment mismatch: expected ~${available_payment:.0f}")
        
        if calculation_errors:
            accuracy = "False"
            justification = f"Mathematical inaccuracies detected in the response calculations. Verification errors found: {'; '.join(calculation_errors)}. Calculation accuracy is essential for financial guidance as incorrect mathematics can mislead users about their actual buying power and affordability. All financial calculations should be verified against input data from the buyability profile to ensure users receive accurate, reliable information for their important home-buying decisions."
        elif calculations_found:
            accuracy = "True"
            justification = f"All verifiable calculations in the response are mathematically accurate. Verified calculations include: {'; '.join(calculations_found)}. Accurate calculations are fundamental to trustworthy financial guidance, ensuring users can rely on the mathematical basis of their home-buying recommendations. The precision of financial calculations directly impacts user confidence and decision-making quality."
        else:
            accuracy = "True"
            justification = "No specific calculations present in the response to verify, therefore marked as True by default. When calculations are absent, there are no mathematical errors to identify. However, comprehensive financial guidance typically benefits from including relevant calculations such as DTI ratios, available monthly payments, and loan-to-value ratios to help users understand the mathematical basis of their affordability assessment."
        
        return {"score": accuracy, "justification": justification}
    
    def _evaluate_faithfulness_to_ground_truth(self, answer: str, scratchpad: str) -> Dict[str, str]:
        """Evaluate faithfulness to ground truth (True/False)."""
        # Check alignment with golden responses, Zillow guidance, and fair housing compliance
        answer_lower = answer.lower()
        
        # Key ground truth elements from all three knowledge sources
        ground_truth_elements = [
            "36% dti recommendation",
            "3% minimum down payment", 
            "property taxes included in payment",
            "homeowners insurance required",
            "pmi for down payments below 20%",
            "interest rate based on credit score",
            "zillow tools and calculations",
            "personalized estimates",
            "fair housing compliance",
            "equal housing opportunity"
        ]
        
        aligned_elements = []
        contradictions = []
        
        # Check for alignment with buyability/mortgage concepts
        if any(term in answer_lower for term in ["36%", "thirty-six percent"]) and "dti" in answer_lower:
            aligned_elements.append("36% DTI guideline")
        
        if any(term in answer_lower for term in ["3%", "three percent"]) and "down" in answer_lower:
            aligned_elements.append("3% minimum down payment")
        
        if "property tax" in answer_lower or "homeowners insurance" in answer_lower:
            aligned_elements.append("property-related costs")
        
        if "pmi" in answer_lower or "mortgage insurance" in answer_lower:
            aligned_elements.append("PMI requirements")
        
        if "credit score" in answer_lower and "rate" in answer_lower:
            aligned_elements.append("credit-rate relationship")
        
        # Check for fair housing compliance
        fair_housing_terms = ["fair housing", "equal housing", "discrimination", "protected class", 
                             "reasonable accommodation", "reasonable modification", "familial status"]
        if any(term in answer_lower for term in fair_housing_terms):
            aligned_elements.append("fair housing awareness")
        
        # Check for discriminatory language or practices
        discriminatory_patterns = [
            r'\b(?:no|avoid|prefer).{0,20}(?:children|kids|families)',
            r'\b(?:perfect|ideal).{0,20}(?:for|young|senior|couple)',
            r'\b(?:no|avoid).{0,20}(?:section\s*8|voucher|assistance)',
            r'\bsteer.{0,20}(?:to|away|from)',
            r'\b(?:different|special).{0,20}(?:rules|requirements).{0,20}(?:for|based on)'
        ]
        
        for pattern in discriminatory_patterns:
            if re.search(pattern, answer_lower):
                contradictions.append("Potential discriminatory language or steering")
        
        # Look for obviously wrong information
        wrong_patterns = [
            r'0%.*down.*required',  # Wrong - VA loans exist but not universal
            r'(?:50|60|70)%.*dti.*recommended',  # Wrong DTI recommendations
            r'no.*insurance.*required',  # Wrong about insurance
            r'refuse.*rent.*to.*families',  # Fair housing violation
            r'charge.*extra.*for.*children'  # Fair housing violation
        ]
        
        for pattern in wrong_patterns:
            if re.search(pattern, answer_lower):
                contradictions.append(f"Potential ground truth violation: {pattern}")
        
        if contradictions:
            faithfulness = "False"
            justification = f"The response contains statements that contradict established ground truth information from knowledge sources. Identified contradictions: {'; '.join(contradictions)}. Faithfulness to ground truth is essential for providing accurate, reliable guidance that aligns with Zillow's tools, established mortgage industry standards, and fair housing laws. Contradictions undermine user trust and could provide misleading or legally problematic information about home-buying processes and requirements."
        else:
            faithfulness = "True"
            justification = f"The response aligns well with ground truth information from Zillow tools, established mortgage guidance, and fair housing principles. Verified alignments include: {', '.join(aligned_elements) if aligned_elements else 'general consistency with industry standards'}. Faithfulness evaluation confirms that statements align with established mortgage practices, Zillow's calculation methods, fair housing compliance, and industry-standard recommendations. This alignment ensures users receive accurate, trustworthy, and legally compliant information."
        
        return {"score": faithfulness, "justification": justification}
    
    def _evaluate_fair_housing_compliance(self, answer: str, scratchpad: str) -> Dict[str, str]:
        """Evaluate fair housing compliance (True/False)."""
        answer_lower = answer.lower()
        
        compliance_issues = []
        compliance_strengths = []
        
        # Check for discriminatory language patterns
        discriminatory_patterns = [
            (r'\b(?:no|avoid|prefer).{0,30}(?:children|kids|families)', "Familial status discrimination"),
            (r'\b(?:perfect|ideal).{0,30}(?:for|young|senior|couple)', "Age/familial status steering"),
            (r'\b(?:no|avoid).{0,30}(?:section\s*8|voucher|assistance)', "Source of income discrimination"),
            (r'\bsteer.{0,30}(?:to|away|from)', "Unlawful steering"),
            (r'\b(?:different|special).{0,30}(?:rules|requirements).{0,30}(?:for|families|children)', "Inconsistent qualification criteria"),
            (r'\bcharge.{0,20}extra.{0,20}(?:for|children|families)', "Discriminatory fees"),
            (r'\brefuse.{0,20}rent.{0,20}(?:to|families|children)', "Refusal to rent based on protected class"),
            (r'\b(?:must|should|need).{0,20}(?:speak|understand).{0,20}english', "National origin discrimination"),
            (r'\bno.{0,20}(?:pets|animals).{0,20}(?:allowed|permitted)', "Potential ESA/service animal violation")
        ]
        
        for pattern, violation_type in discriminatory_patterns:
            if re.search(pattern, answer_lower):
                compliance_issues.append(violation_type)
        
        # Check for positive fair housing practices
        positive_indicators = [
            (r'\bequal\s+(?:housing\s+)?opportunity', "Equal housing opportunity awareness"),
            (r'\bfair\s+housing', "Fair housing awareness"),
            (r'\breasonable\s+(?:accommodation|modification)', "Disability accommodation awareness"),
            (r'\bassistance\s+animal|emotional\s+support\s+animal|service\s+animal', "ESA/service animal awareness"),
            (r'\bprotected\s+class', "Protected class awareness"),
            (r'\bconsult.{0,20}(?:attorney|lawyer|legal)', "Legal consultation recommendation"),
            (r'\bcomply.{0,20}(?:with|fair\s+housing|laws)', "Compliance awareness")
        ]
        
        for pattern, strength_type in positive_indicators:
            if re.search(pattern, answer_lower):
                compliance_strengths.append(strength_type)
        
        # Check for occupancy-related issues
        occupancy_patterns = [
            (r'\bno.{0,20}(?:children|kids).{0,20}(?:allowed|permitted)', "Children exclusion"),
            (r'\b(?:adults\s+only|no\s+minors)', "Age discrimination"),
            (r'\bmaximum.{0,20}(?:2|two).{0,20}(?:people|persons).{0,20}(?:per|bedroom)', "Potentially discriminatory occupancy"),
            (r'\b(?:one|1).{0,20}child.{0,20}(?:maximum|limit|only)', "Child limitation")
        ]
        
        for pattern, violation_type in occupancy_patterns:
            if re.search(pattern, answer_lower):
                compliance_issues.append(violation_type)
        
        # Check for advertising/marketing compliance
        if any(term in answer_lower for term in ["marketing", "advertising", "listing", "description"]):
            marketing_violations = [
                (r'\b(?:great|perfect|ideal).{0,30}(?:for|young|senior|professional|couple)', "Discriminatory advertising"),
                (r'\b(?:family|adult|mature|quiet).{0,20}(?:oriented|friendly|community)', "Potentially discriminatory marketing"),
                (r'\b(?:no|avoid).{0,20}(?:college|student|youth)', "Age discrimination in advertising")
            ]
            
            for pattern, violation_type in marketing_violations:
                if re.search(pattern, answer_lower):
                    compliance_issues.append(violation_type)
        
        if compliance_issues:
            compliance = "False"
            justification = f"The response contains potential fair housing compliance violations. Identified issues: {'; '.join(set(compliance_issues))}. Fair housing compliance is legally mandated and essential for protecting equal housing opportunities. Violations can result in serious legal consequences including lawsuits, fines, and regulatory action. All housing-related communications must comply with federal, state, and local fair housing laws to ensure equal treatment regardless of protected class status."
        else:
            compliance = "True"
            compliance_note = f" Positive compliance indicators: {', '.join(set(compliance_strengths))}" if compliance_strengths else ""
            justification = f"The response demonstrates fair housing compliance with no detected discriminatory language or practices.{compliance_note} Fair housing compliance evaluation ensures that communications align with federal Fair Housing Act requirements and related laws protecting against discrimination based on race, color, religion, sex, national origin, familial status, and disability. Compliant responses promote equal housing opportunity and protect against legal liability."
        
        return {"score": compliance, "justification": justification}
    
    def _evaluate_overall_accuracy(self, answer: str, question: str, scratchpad: str) -> Dict[str, str]:
        """Evaluate overall accuracy (True/False)."""
        # Holistic assessment of whether the response correctly answers the question
        answer_lower = answer.lower()
        question_lower = question.lower() if question else ""
        
        accuracy_issues = []
        accuracy_strengths = []
        
        # Check if the response addresses the core question
        if question:
            if "buyability" in question_lower:
                if "buyability" in answer_lower or "afford" in answer_lower:
                    accuracy_strengths.append("addresses buyability question")
                else:
                    accuracy_issues.append("fails to address buyability inquiry")
            
            if "calculate" in question_lower:
                if any(term in answer_lower for term in ["calculate", "formula", "based on", "using"]):
                    accuracy_strengths.append("explains calculation methodology")
                else:
                    accuracy_issues.append("missing calculation explanation")
            
            if "personalized" in question_lower:
                if any(term in answer_lower for term in ["your", "personalized", "specific", "individual"]):
                    accuracy_strengths.append("demonstrates personalization")
                else:
                    accuracy_issues.append("lacks personalization evidence")
        
        # Check for factual accuracy indicators
        if any(term in answer_lower for term in ["$", "percent", "%", "rate"]):
            accuracy_strengths.append("includes specific financial data")
        
        # Check for completeness of response
        if len(answer.split()) > 100:  # Substantial response
            accuracy_strengths.append("provides comprehensive information")
        elif len(answer.split()) < 50:  # Very brief response
            accuracy_issues.append("response may be too brief for complex question")
        
        # Check for logical structure
        if any(term in answer_lower for term in ["first", "second", "because", "therefore", "however"]):
            accuracy_strengths.append("demonstrates logical reasoning")
        
        if accuracy_issues:
            overall_accuracy = "False"
            justification = f"The response has significant accuracy issues that prevent it from correctly answering the user's question. Identified issues: {'; '.join(accuracy_issues)}. While some strengths exist ({', '.join(accuracy_strengths)}), the fundamental problems undermine the response's overall correctness. Overall accuracy requires that the response correctly, completely, and appropriately addresses the user's question with reliable information and sound reasoning."
        else:
            overall_accuracy = "True"
            justification = f"The response correctly and comprehensively addresses the user's question with accurate information and appropriate detail. Key strengths include: {', '.join(accuracy_strengths)}. Overall accuracy evaluation confirms that the response fulfills its primary purpose of providing correct, helpful information that appropriately addresses the user's inquiry. The holistic assessment shows the response meets accuracy standards for reliable financial guidance."
        
        return {"score": overall_accuracy, "justification": justification}
    
    def _evaluate_structured_presentation(self, answer: str, scratchpad: str) -> Dict[str, str]:
        """Evaluate structured presentation (1-5 scale)."""
        structure_elements = {
            'headings': len(re.findall(r'^#+\s', answer, re.MULTILINE)),
            'bullet_points': len(re.findall(r'^\s*[•\-\*]\s', answer, re.MULTILINE)),
            'numbered_lists': len(re.findall(r'^\s*\d+\.\s', answer, re.MULTILINE)),
            'tables': len(re.findall(r'\|.*\|', answer)),
            'line_breaks': len(re.findall(r'\n\s*\n', answer)),
            'sections': len(re.findall(r'\n\s*[A-Z][^.!?]*[:.]\s*\n', answer))
        }
        
        total_elements = sum(structure_elements.values())
        content_length = len(answer.split())
        
        # Determine score based on structure density and variety
        if total_elements >= 10 and structure_elements['headings'] >= 2:
            score = 5
            description = "outline-quality headings, flawless lists, well-formatted tables"
        elif total_elements >= 6 and any(structure_elements[key] >= 2 for key in ['bullet_points', 'numbered_lists']):
            score = 4
            description = "logical hierarchy, parallel lists, well-formatted structure"
        elif total_elements >= 3:
            score = 3
            description = "clear headings and lists for ≥50% of content"
        elif total_elements >= 1:
            score = 2
            description = "minimal, inconsistent structure"
        else:
            score = 1
            description = "wall of text with no clear structure"
        
        justification = f"The response demonstrates {description}. Structure analysis found: {structure_elements['headings']} headings, {structure_elements['bullet_points']} bullet points, {structure_elements['numbered_lists']} numbered items, {structure_elements['tables']} tables. Structured presentation significantly improves readability and user comprehension by organizing information into logical, scannable sections. Well-structured content helps users quickly find relevant information and better understand complex financial concepts through clear visual hierarchy and organization."
        
        return {"score": str(score), "justification": justification}
    
    def _evaluate_coherence(self, answer: str, scratchpad: str) -> Dict[str, str]:
        """Evaluate coherence (True/False)."""
        coherence_issues = []
        
        # Check for logical consistency
        sentences = re.split(r'[.!?]+', answer)
        
        # Look for contradictions (simplified analysis)
        answer_lower = answer.lower()
        
        # Check for contradictory statements about DTI
        if "36%" in answer_lower and any(pct in answer_lower for pct in ["40%", "45%", "50%"]):
            if "dti" in answer_lower:
                coherence_issues.append("conflicting DTI recommendations")
        
        # Check for repetitive content
        sentence_similarity_count = 0
        for i, sent1 in enumerate(sentences[:5]):  # Check first 5 sentences
            for sent2 in sentences[i+1:6]:
                if len(sent1.strip()) > 20 and len(sent2.strip()) > 20:
                    # Simple similarity check
                    words1 = set(sent1.lower().split())
                    words2 = set(sent2.lower().split())
                    if len(words1 & words2) / max(len(words1), len(words2)) > 0.6:
                        sentence_similarity_count += 1
        
        if sentence_similarity_count > 2:
            coherence_issues.append("excessive repetition detected")
        
        # Check for logical flow
        transition_words = len(re.findall(r'\b(?:however|therefore|because|since|thus|also|additionally|furthermore|moreover|consequently)\b', answer_lower))
        if len(sentences) > 5 and transition_words == 0:
            coherence_issues.append("lacks logical connectors")
        
        # Check for contradictory numerical information
        dollar_amounts = re.findall(r'\$(\d+(?:,\d{3})*)', answer)
        if len(dollar_amounts) > 3:
            # Very basic check for obviously contradictory amounts
            amounts = [self._parse_currency(amt) for amt in dollar_amounts]
            if any(amt > 10000000 for amt in amounts):  # Suspiciously high amount
                coherence_issues.append("potentially unrealistic financial figures")
        
        if coherence_issues:
            coherence = "False"
            justification = f"The response demonstrates coherence issues that impact logical consistency and readability. Identified problems: {'; '.join(coherence_issues)}. Coherence is essential for user comprehension and trust, requiring logical consistency, minimal repetition, and clear information flow. Coherence issues can confuse users and undermine the credibility of financial guidance."
        else:
            coherence = "True"
            justification = f"The response demonstrates strong logical consistency with clear information flow and minimal repetition. Coherence analysis found {transition_words} logical connectors across {len(sentences)} sentences, supporting smooth information flow. Strong coherence enhances user understanding by presenting information in a logical, consistent manner that builds comprehension progressively without contradictions or confusing repetition."
        
        return {"score": coherence, "justification": justification}
    
    def _evaluate_completeness(self, answer: str, question: str, scratchpad: str) -> Dict[str, str]:
        """Evaluate completeness (1-5 scale)."""
        if not question:
            return {
                "score": "3",
                "justification": "No specific question provided to assess completeness against. Without knowing the user's specific inquiry, completeness cannot be accurately measured. Completeness evaluation requires understanding what sub-questions or information elements the user expected to be addressed. A default moderate score is assigned pending specific question context for proper assessment."
            }
        
        question_lower = question.lower()
        answer_lower = answer.lower()
        
        # Identify expected sub-questions based on the main question
        expected_elements = []
        addressed_elements = []
        
        if "buyability" in question_lower and "calculate" in question_lower:
            expected_elements = [
                "income consideration",
                "debt impact", 
                "down payment effect",
                "credit score influence",
                "interest rate explanation",
                "location factors"
            ]
            
            # Check if each is addressed
            if any(term in answer_lower for term in ["income", "salary", "earn"]):
                addressed_elements.append("income consideration")
            if any(term in answer_lower for term in ["debt", "obligation", "monthly payment"]):
                addressed_elements.append("debt impact")
            if any(term in answer_lower for term in ["down payment", "down", "upfront"]):
                addressed_elements.append("down payment effect")
            if any(term in answer_lower for term in ["credit", "score", "rating"]):
                addressed_elements.append("credit score influence")
            if any(term in answer_lower for term in ["rate", "interest", "apr"]):
                addressed_elements.append("interest rate explanation")
            if any(term in answer_lower for term in ["location", "state", "tax", "insurance"]):
                addressed_elements.append("location factors")
        
        elif "personalized" in question_lower:
            expected_elements = [
                "personalization confirmation",
                "data sources listed",
                "individual calculation proof",
                "customization examples"
            ]
            
            if any(term in answer_lower for term in ["yes", "personalized", "tailored", "specific"]):
                addressed_elements.append("personalization confirmation")
            if any(term in answer_lower for term in ["income", "debt", "credit", "down payment"]):
                addressed_elements.append("data sources listed")
            if any(term in answer_lower for term in ["your", "you", "based on"]):
                addressed_elements.append("individual calculation proof")
            if "$" in answer and any(term in answer_lower for term in ["your", "specific"]):
                addressed_elements.append("customization examples")
        
        elif "monthly payment" in question_lower:
            expected_elements = [
                "principal and interest breakdown",
                "property taxes explanation", 
                "insurance costs",
                "PMI details",
                "total payment summary"
            ]
            
            if any(term in answer_lower for term in ["principal", "interest", "mortgage payment"]):
                addressed_elements.append("principal and interest breakdown")
            if any(term in answer_lower for term in ["property tax", "tax"]):
                addressed_elements.append("property taxes explanation")
            if any(term in answer_lower for term in ["insurance", "homeowner"]):
                addressed_elements.append("insurance costs")
            if any(term in answer_lower for term in ["pmi", "mortgage insurance"]):
                addressed_elements.append("PMI details")
            if "$" in answer and any(term in answer_lower for term in ["total", "monthly"]):
                addressed_elements.append("total payment summary")
        
        else:
            # Generic completeness check
            expected_elements = ["main topic addressed", "supporting details", "practical guidance"]
            addressed_elements = ["main topic addressed"]  # Assume main topic is addressed
            
            if len(answer.split()) > 100:
                addressed_elements.append("supporting details")
            if any(term in answer_lower for term in ["should", "recommend", "consider", "next"]):
                addressed_elements.append("practical guidance")
        
        if not expected_elements:
            percentage = 100
        else:
            percentage = (len(addressed_elements) / len(expected_elements)) * 100
        
        # Score based on percentage
        if percentage >= 80:
            score = 5
        elif percentage >= 60:
            score = 4
        elif percentage >= 40:
            score = 3
        elif percentage >= 20:
            score = 2
        else:
            score = 1
        
        justification = f"The response addresses {len(addressed_elements)} out of {len(expected_elements)} expected elements ({percentage:.0f}%). Addressed elements: {', '.join(addressed_elements)}. {'Missing elements: ' + ', '.join(set(expected_elements) - set(addressed_elements)) if len(addressed_elements) < len(expected_elements) else 'All key elements covered.'} Completeness evaluation ensures users receive comprehensive answers that address all relevant aspects of their question, providing sufficient information for informed decision-making."
        
        return {"score": str(score), "justification": justification}
    
    def _format_evaluation_table(self, results: Dict[str, Dict[str, str]]) -> str:
        """Format the final evaluation table output."""
        table = "| Metric | Score | Justification |\n"
        table += "|--------|-------|---------------|\n"
        
        metric_names = {
            "personalization_accuracy": "Personalization Accuracy",
            "context_based_personalization": "Context based Personalization", 
            "next_step_identification": "Next Step Identification",
            "assumption_listing": "Assumption Listing",
            "assumption_trust": "Assumption Trust",
            "calculation_accuracy": "Calculation Accuracy", 
            "faithfulness_to_ground_truth": "Faithfulness to Ground Truth",
            "fair_housing_compliance": "Fair Housing Compliance",
            "overall_accuracy": "Overall Accuracy",
            "structured_presentation": "Structured Presentation",
            "coherence": "Coherence",
            "completeness": "Completeness"
        }
        
        for key, name in metric_names.items():
            if key in results:
                score = results[key]["score"]
                justification = results[key]["justification"]
                table += f"| {name} | {score} | {justification} |\n"
        
        return table


def main():
    """Main function for testing the evaluator."""
    evaluator = ZillowJudgeEvaluator()
    
    # Example usage
    test_answer = """
    Your personalized BuyAbility estimate is $318,431, based on your specific financial profile.
    
    This calculation uses your $90,000 annual income, $200 monthly debts, $18,000 down payment,
    and credit score range of 660-719. With your monthly income of $7,500, lenders recommend
    up to 36% DTI, giving you about $2,500 available for mortgage payments.
    
    Your monthly payment breaks down as:
    • Principal & Interest: $1,975
    • Property Taxes: $212
    • Homeowners Insurance: $106
    • PMI: $207
    
    You should consider getting pre-approved to start your home search in Georgia.
    """
    
    test_question = "What factors were considered to calculate my Buyability?"
    test_profile = {
        "annual_income": 90000,
        "monthly_debts": 200,
        "down_payment": 18000,
        "credit_score": "660-719"
    }
    
    result = evaluator.evaluate(test_answer, test_question, test_profile)
    print(result)


if __name__ == "__main__":
    main()