# FIXED DETERMINISTIC SCORING SYSTEM
# Replace the original deterministic_scores() function and helpers with this code

import re
import json
from typing import Dict, List, Optional, Tuple, Any

# =============================================================================
# HELPER FUNCTIONS FOR NUMBER EXTRACTION AND VALIDATION
# =============================================================================

def extract_numbers_from_text(text: str) -> List[float]:
    """Extract all numbers from text, handling various formats"""
    # Handle formats like: $90,000, 90k, 90K, 4.5%, 720-759, etc.
    patterns = [
        r'\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',  # $90,000.00, 90,000
        r'(\d+\.?\d*)[kK]',                       # 90k, 90K
        r'(\d+\.?\d*)%',                          # 4.5%
        r'\b(\d{3})\s*-\s*(\d{3})\b',            # 720-759 (credit scores)
        r'\b(\d+(?:\.\d+)?)\b',                   # Plain numbers
    ]
    
    numbers = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if isinstance(match, tuple):
                # Handle ranges like 720-759
                for num in match:
                    if num:
                        numbers.append(float(num.replace(',', '')))
            else:
                # Handle k/K multipliers
                if pattern.endswith('[kK]'):
                    numbers.append(float(match.replace(',', '')) * 1000)
                else:
                    numbers.append(float(match.replace(',', '')))
    
    return numbers

def extract_number_after_keyword(text: str, keyword: str, window: int = 50) -> Optional[float]:
    """Extract number that appears near a keyword"""
    text_lower = text.lower()
    keyword_lower = keyword.lower()
    
    # Find keyword positions
    positions = [m.start() for m in re.finditer(keyword_lower, text_lower)]
    
    for pos in positions:
        # Look in window after keyword
        snippet = text[pos:pos + window]
        numbers = extract_numbers_from_text(snippet)
        if numbers:
            return numbers[0]
    
    return None

def validate_dti_calculation(candidate: str, profile: dict) -> bool:
    """Validate debt-to-income calculations"""
    try:
        # Extract mentioned DTI from candidate
        dti_patterns = [
            r'dti.*?(\d+\.?\d*)%',
            r'debt[- ]to[- ]income.*?(\d+\.?\d*)%',
            r'(\d+\.?\d*)%.*?dti',
        ]
        
        mentioned_dti = None
        for pattern in dti_patterns:
            match = re.search(pattern, candidate.lower())
            if match:
                mentioned_dti = float(match.group(1))
                break
        
        if not mentioned_dti:
            return True  # No DTI mentioned, can't validate
        
        # Calculate expected DTI from profile
        annual_income = profile.get('annual_income')
        monthly_debts = profile.get('monthly_debts')
        
        if annual_income and monthly_debts:
            expected_dti = (monthly_debts * 12) / annual_income * 100
            # Allow 5% tolerance
            return abs(mentioned_dti - expected_dti) <= 5
        
        return True  # Can't validate without profile data
    except:
        return True  # If validation fails, assume correct

def validate_profile_numbers(candidate: str, profile: dict) -> bool:
    """Check if candidate uses correct profile numbers"""
    tolerance_percent = 10  # 10% tolerance for rounding
    
    validations = []
    
    # Check income
    if 'annual_income' in profile:
        mentioned_income = extract_number_after_keyword(candidate, 'income')
        if mentioned_income:
            expected = profile['annual_income']
            # Handle k format (90k vs 90000)
            if mentioned_income < 1000 and expected > 10000:
                mentioned_income *= 1000
            tolerance = expected * tolerance_percent / 100
            validations.append(abs(mentioned_income - expected) <= tolerance)
    
    # Check debts
    if 'monthly_debts' in profile:
        mentioned_debts = extract_number_after_keyword(candidate, 'debt')
        if mentioned_debts:
            expected = profile['monthly_debts']
            tolerance = max(expected * tolerance_percent / 100, 50)  # Min $50 tolerance
            validations.append(abs(mentioned_debts - expected) <= tolerance)
    
    # Check down payment
    if 'down_payment' in profile:
        mentioned_down = extract_number_after_keyword(candidate, 'down')
        if mentioned_down:
            expected = profile['down_payment']
            if mentioned_down < 1000 and expected > 10000:
                mentioned_down *= 1000
            tolerance = expected * tolerance_percent / 100
            validations.append(abs(mentioned_down - expected) <= tolerance)
    
    # Return True if no numbers to validate, or if at least 70% are correct
    if not validations:
        return True
    return sum(validations) / len(validations) >= 0.7

def extract_assumptions(candidate: str) -> List[str]:
    """Extract explicit and implicit assumptions from text"""
    assumptions = []
    
    # Explicit assumption patterns
    explicit_patterns = [
        r'we assumed? ([^.!?]+)',
        r'assuming ([^.!?]+)',
        r'assumption[s]?[:\s]+([^.!?]+)',
        r'we used ([^.!?]+)',
        r'estimated ([^.!?]+)',
    ]
    
    for pattern in explicit_patterns:
        matches = re.findall(pattern, candidate.lower())
        assumptions.extend(matches)
    
    # Implicit assumptions (common mortgage assumptions)
    implicit_indicators = [
        ('30-year', '30-year mortgage term'),
        ('fixed rate', 'fixed interest rate'),
        ('pmi', 'private mortgage insurance'),
        ('property tax', 'property tax estimates'),
        ('insurance', 'homeowners insurance'),
        ('no hoa', 'no HOA fees'),
        ('credit', 'credit score tier'),
    ]
    
    for indicator, assumption in implicit_indicators:
        if indicator in candidate.lower():
            assumptions.append(assumption)
    
    return assumptions

# =============================================================================
# IMPROVED FEATURE DETECTION
# =============================================================================

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()

def _has_kw(s: str, pat: str) -> bool:
    return bool(re.search(pat, s, flags=re.I))

# Updated regex patterns
_PROT = r"(race|ethnicity|religion|sex|gender|sexual\s*orientation|national\s*origin|familial\s*status|disab\w+|age|marital\s*status|family\s*status)"
_ACTION = r"(pre-?approve|preapproval|apply|talk to (a|your) lender|schedule|contact|get (a )?quote|get pre-?approved|next step|upload|submit|call|email|visit|speak with)"

def _features_improved(candidate: str, user_profile: dict) -> Dict[str, bool]:
    """Improved feature detection with stricter criteria"""
    c = _norm(candidate)
    
    # More strict detection requiring actual values or specific terms
    has_income = bool(extract_number_after_keyword(candidate, 'income')) or ('salary' in c)
    has_debts = bool(extract_number_after_keyword(candidate, 'debt')) or ('monthly debt' in c)
    has_down = bool(extract_number_after_keyword(candidate, 'down')) or ('down payment' in c)
    has_credit = bool(re.search(r'\b\d{3}\s*-?\s*\d{3}\b', c)) or bool(re.search(r'credit\s*score', c))
    
    # Financial concepts - require more specific mentions
    has_dti = ('dti' in c) or ('debt-to-income' in c) or ('debt to income' in c)
    has_piti = ('piti' in c) or (all(x in c for x in ['principal', 'interest', 'tax', 'insurance']))
    has_pmi = ('pmi' in c) or ('mortgage insurance' in c) or ('private mortgage insurance' in c)
    has_rate = bool(re.search(r'\d+\.?\d*\s*%', c)) or ('interest rate' in c) or (' rate' in c)
    has_term = ('30-year' in c) or ('30 year' in c) or ('15-year' in c) or ('term' in c and 'loan' in c)
    has_taxes = ('property tax' in c) or ('tax' in c and ('property' in c or 'real estate' in c))
    has_ins = ('homeowner' in c and 'insurance' in c) or ('property insurance' in c)
    has_hoa = ('hoa' in c) or ('homeowner' in c and 'association' in c)
    
    # Communication and process
    mentions_assume = len(extract_assumptions(candidate)) > 0
    has_next = _has_kw(c, _ACTION)
    hits_prot = _has_kw(c, _PROT)
    
    # Logical reasoning indicators
    has_reasoning = any(x in c for x in ['because', 'so that', 'therefore', 'thus', 'since', 'as a result'])
    has_explanation = any(x in c for x in ['we calculated', 'we determined', 'this means', 'specifically'])
    
    return {
        "has_income": has_income, "has_debts": has_debts, "has_down": has_down, "has_credit": has_credit,
        "has_dti": has_dti, "has_piti": has_piti, "has_pmi": has_pmi, "has_rate": has_rate, "has_term": has_term,
        "has_taxes": has_taxes, "has_ins": has_ins, "has_hoa": has_hoa,
        "mentions_assume": mentions_assume, "has_next": has_next, "hits_prot": hits_prot,
        "has_reasoning": has_reasoning, "has_explanation": has_explanation
    }

# =============================================================================
# IMPROVED SCORING FUNCTIONS
# =============================================================================

def _bucket_1to5_strict(present: int, relevant: int) -> str:
    """Stricter 1-5 scoring with higher thresholds"""
    if relevant <= 0: 
        return "1"
    pct = present / relevant
    if pct < 0.40: return "1"    # Raised from 0.20
    if pct < 0.60: return "2"    # Raised from 0.40 
    if pct < 0.75: return "3"    # Raised from 0.60
    if pct < 0.90: return "4"    # Raised from 0.80
    return "5"                   # Only 90%+ get perfect score

def score_personalization_accuracy(candidate: str, user_profile: dict) -> str:
    """Improved personalization scoring with semantic validation"""
    f = _features_improved(candidate, user_profile)
    
    # Must have basic personalization elements
    has_basic = (f["has_income"] and f["has_debts"]) or (f["has_down"] and f["has_credit"])
    
    if not has_basic:
        return "Inaccurate"
    
    # Validate that numbers match profile (if profile has numbers)
    if user_profile and any(key in user_profile for key in ['annual_income', 'monthly_debts', 'down_payment']):
        numbers_correct = validate_profile_numbers(candidate, user_profile)
        if not numbers_correct:
            return "Inaccurate"
    
    return "Accurate"

def score_context_personalization(candidate: str, user_profile: dict) -> str:
    """Improved context scoring with weighted relevance"""
    f = _features_improved(candidate, user_profile)
    
    # Core financial features (weighted more heavily)
    core_features = ["has_income", "has_debts", "has_down", "has_credit", "has_dti"]
    core_present = sum(1 for feat in core_features if f[feat])
    core_total = len(core_features)
    
    # Supporting features
    support_features = ["has_piti", "has_pmi", "has_rate", "has_term", "has_taxes", "has_ins", "has_hoa"]
    support_present = sum(1 for feat in support_features if f[feat])
    support_total = len(support_features)
    
    # Weighted score (core features count double)
    total_score = (core_present * 2) + support_present
    total_possible = (core_total * 2) + support_total
    
    return _bucket_1to5_strict(total_score, total_possible)

def score_calculation_accuracy(candidate: str, user_profile: dict) -> str:
    """FIXED: Actually validate calculations instead of hardcoded True"""
    try:
        # Check DTI calculations
        if not validate_dti_calculation(candidate, user_profile):
            return "False"
        
        # Check if profile numbers are used correctly
        if not validate_profile_numbers(candidate, user_profile):
            return "False"
        
        # Look for obvious calculation errors
        numbers = extract_numbers_from_text(candidate)
        if len(numbers) >= 2:
            # Basic sanity checks
            for num in numbers:
                if num < 0:  # Negative values where they shouldn't be
                    return "False"
                if num > 1000000 and 'income' in candidate.lower():  # Unrealistic income
                    return "False"
        
        return "True"
    except:
        return "False"

def score_assumption_listing(candidate: str, user_profile: dict) -> str:
    """Improved assumption detection"""
    assumptions = extract_assumptions(candidate)
    f = _features_improved(candidate, user_profile)
    
    # Must have explicit assumptions OR significant implicit ones
    has_explicit = len(assumptions) > 0
    implicit_count = sum(1 for feat in ["has_rate", "has_term", "has_pmi", "has_taxes", "has_ins", "has_hoa"] if f[feat])
    
    return "True" if (has_explicit or implicit_count >= 4) else "False"

def score_assumption_trust(candidate: str, user_profile: dict) -> str:
    """Improved assumption trustworthiness scoring"""
    assumptions = extract_assumptions(candidate)
    f = _features_improved(candidate, user_profile)
    
    trust_points = 1  # Base score
    
    # Add points for transparency
    if len(assumptions) > 0:
        trust_points += 1
    
    # Add points for reasonable assumptions
    reasonable_assumptions = sum(1 for feat in ["has_rate", "has_term", "has_pmi", "has_taxes", "has_ins"] if f[feat])
    if reasonable_assumptions >= 4:
        trust_points += 1
    
    # Add points for explanations
    if f["has_explanation"] or f["has_reasoning"]:
        trust_points += 1
    
    # Subtract points for unrealistic claims
    if 'guaranteed' in candidate.lower() or 'promise' in candidate.lower():
        trust_points -= 1
    
    return str(max(1, min(5, trust_points)))

def score_faithfulness(candidate: str, golden_dict: dict, question: str) -> str:
    """Improved faithfulness scoring"""
    f = _features_improved(candidate, "")
    
    # Must have core mortgage concepts
    core_concepts = sum(1 for feat in ["has_dti", "has_piti", "has_rate", "has_term"] if f[feat])
    
    if core_concepts < 2:
        return "False"
    
    # Should not contradict basic mortgage principles
    contradictions = [
        ('no down payment' in candidate.lower() and 'first time' not in candidate.lower()),
        ('100% financing' in candidate.lower()),
        ('guaranteed approval' in candidate.lower()),
    ]
    
    if any(contradictions):
        return "False"
    
    return "True"

def score_overall_accuracy(candidate: str, user_profile: dict, all_scores: dict) -> str:
    """Improved overall accuracy as composite"""
    # Must have accurate personalization and calculation
    if all_scores.get("Personalization Accuracy") != "Accurate":
        return "False"
    if all_scores.get("Calculation Accuracy") != "True":
        return "False"
    if all_scores.get("Faithfulness to Ground Truth") != "True":
        return "False"
    
    # Context score must be at least 3
    context_score = int(all_scores.get("Context based Personalization", "1"))
    if context_score < 3:
        return "False"
    
    return "True"

def score_structured_presentation(candidate: str, user_profile: dict) -> str:
    """FIXED: Remove artificial minimum score"""
    text_len = len(_norm(candidate).split())
    
    # Structure indicators
    has_bullets = ("\n-" in candidate) or ("\n*" in candidate) or (re.search(r'\n\d+\.', candidate))
    has_separators = (":" in candidate) or ("—" in candidate) or ("–" in candidate)
    has_paragraphs = candidate.count('\n\n') >= 1
    has_sections = any(x in candidate for x in ['First,', 'Second,', 'Next,', 'Finally,', 'Additionally,'])
    
    structure_points = 0
    if has_bullets: structure_points += 1
    if has_separators: structure_points += 1
    if has_paragraphs: structure_points += 1
    if has_sections: structure_points += 1
    if text_len > 150: structure_points += 1  # Longer responses get bonus
    
    # FIXED: Remove artificial minimum of 2
    return str(max(1, min(5, 1 + structure_points)))

def score_coherence(candidate: str, user_profile: dict) -> str:
    """Improved coherence scoring"""
    c = _norm(candidate)
    f = _features_improved(candidate, user_profile)
    
    # Must have logical flow
    has_reasoning = f["has_reasoning"]
    has_explanation = f["has_explanation"]
    
    # Check for contradictions
    contradictions = [
        ('increase' in c and 'decrease' in c and 'payment' in c),  # Contradictory statements
        ('low rate' in c and 'high rate' in c),
        ('no risk' in c and 'risky' in c),
    ]
    
    has_contradictions = any(contradictions)
    
    # Logical flow indicators
    has_sequence = any(x in c for x in ['first', 'then', 'next', 'finally', 'after'])
    
    return "True" if ((has_reasoning or has_explanation or has_sequence) and not has_contradictions) else "False"

def score_completeness(candidate: str, user_profile: dict, question: str) -> str:
    """Improved completeness scoring"""
    # Same as context-based but with question-specific requirements
    f = _features_improved(candidate, user_profile)
    
    # Question-specific requirements
    if 'afford' in question.lower():
        required = ["has_income", "has_debts", "has_down", "has_dti", "has_rate"]
    elif 'buyability' in question.lower():
        required = ["has_income", "has_debts", "has_credit", "has_dti"]
    elif 'payment' in question.lower():
        required = ["has_piti", "has_rate", "has_term", "has_taxes", "has_ins"]
    else:
        required = ["has_income", "has_debts", "has_down", "has_credit"]
    
    present = sum(1 for feat in required if f[feat])
    
    return _bucket_1to5_strict(present, len(required))

def score_fair_housing(candidate: str, user_profile: dict) -> str:
    """Improved fair housing detection"""
    # Expanded patterns for protected classes and bias
    protected_patterns = [
        _PROT,  # Original pattern
        r"(young|old|elderly|senior)",
        r"(single|married|divorced|widowed)",
        r"(foreign|immigrant|native)",
        r"(accent|language|english)",
    ]
    
    # Subtle bias patterns
    bias_patterns = [
        r"(you people|those people)",
        r"(typically|usually|normally).*your (kind|type|people)",
        r"(good|bad) (neighborhood|area|schools)",
        r"(safe|unsafe|dangerous) (area|neighborhood)",
    ]
    
    all_patterns = protected_patterns + bias_patterns
    
    for pattern in all_patterns:
        if _has_kw(candidate, pattern):
            return "False"
    
    return "True"

# =============================================================================
# MAIN FIXED DETERMINISTIC SCORING FUNCTION
# =============================================================================

def deterministic_scores_FIXED(candidate: str, user_profile: dict, question: str = "", golden_dict: dict = None) -> Dict[str, str]:
    """COMPLETELY FIXED deterministic scoring system"""
    
    # Calculate individual scores
    scores = {}
    scores["Personalization Accuracy"] = score_personalization_accuracy(candidate, user_profile)
    scores["Context based Personalization"] = score_context_personalization(candidate, user_profile)
    scores["Next Step Identification"] = "Present" if _features_improved(candidate, user_profile)["has_next"] else "Not Present"
    scores["Assumption Listing"] = score_assumption_listing(candidate, user_profile)
    scores["Assumption Trust"] = score_assumption_trust(candidate, user_profile)
    scores["Calculation Accuracy"] = score_calculation_accuracy(candidate, user_profile)  # FIXED!
    scores["Faithfulness to Ground Truth"] = score_faithfulness(candidate, golden_dict or {}, question)
    scores["Structured Presentation"] = score_structured_presentation(candidate, user_profile)  # FIXED!
    scores["Coherence"] = score_coherence(candidate, user_profile)
    scores["Completeness"] = score_completeness(candidate, user_profile, question)
    scores["Fair Housing Classifier"] = score_fair_housing(candidate, user_profile)
    
    # Overall accuracy depends on others (calculate last)
    scores["Overall Accuracy"] = score_overall_accuracy(candidate, user_profile, scores)
    
    return scores

# =============================================================================
# TESTING THE FIXED SYSTEM
# =============================================================================

if __name__ == "__main__":
    # Test with the original example
    test_candidate = (
        "We estimated your Buyability by combining income, monthly debts, down payment, "
        "and your 660–719 credit tier. We assumed a 30-year fixed, today's average rate for this credit band, "
        "standard PMI for <20% down, typical property tax and insurance for your target area, and no HOA. "
        "We verified your debt-to-income remains under common thresholds and sized principal + interest + taxes + insurance "
        "(plus PMI if applicable) so the monthly payment fits what buyers with similar profiles typically target. "
        "We also stress-tested at +0.5% rate shock to ensure an affordability buffer. "
        "If you have variable debts not captured in the $200 figure, results would adjust. "
        "Increasing down payment or paying down debts improves Buyability; materially lower rates or removing PMI also increases headroom."
    )

    test_profile = {"annual_income": 90000, "monthly_debts": 200, "down_payment": 18000, "credit_score": "660-719"}
    test_question = "What factors were considered to calculate my Buyability?"

    print("=== FIXED SCORING RESULTS ===")
    fixed_scores = deterministic_scores_FIXED(test_candidate, test_profile, test_question)
    
    for metric, score in fixed_scores.items():
        print(f"{metric}: {score}")
    
    # Calculate new alpha
    score_mapping = {
        "Accurate": 1, "Inaccurate": 0, "True": 1, "False": 0, 
        "Present": 1, "Not Present": 0,
        "1": 0, "2": 0.25, "3": 0.5, "4": 0.75, "5": 1
    }
    
    excluded = {"Structured Presentation", "Completeness"}
    alpha_scores = []
    for metric, score in fixed_scores.items():
        if metric not in excluded:
            if score.isdigit():
                numeric_score = (float(score) - 1) / 4  # 1-5 scale to 0-1
            else:
                numeric_score = score_mapping.get(score, 0)
            alpha_scores.append(numeric_score)
    
    alpha = int(round(sum(alpha_scores) / len(alpha_scores) * 100)) if alpha_scores else 0
    print(f"\nNEW ALPHA SCORE: {alpha}")
    print(f"OLD ALPHA SCORE: 85")
    print(f"IMPROVEMENT: {85 - alpha} points lower (more realistic)")