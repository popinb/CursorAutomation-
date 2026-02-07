# Extracted scoring logic from the LLM judge code for analysis

import re
from typing import Dict

def _bucket_1to5(present: int, relevant: int) -> str:
    """Convert ratio to 1-5 scale"""
    if relevant <= 0: 
        return "1"
    pct = present/relevant
    if pct < 0.20: return "1"
    if pct < 0.40: return "2" 
    if pct < 0.60: return "3"
    if pct < 0.80: return "4"
    return "5"

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()

def _has_kw(s: str, pat: str) -> bool:
    return bool(re.search(pat, s, flags=re.I))

_PROT = r"(race|ethnicity|religion|sex|gender|sexual\s*orientation|national\s*origin|familial\s*status|disab\w+)"
_ACTION = r"(pre-?approve|preapproval|apply|talk to (a|your) lender|schedule|contact|get (a )?quote|get pre-?approved|next step|upload|submit)"

def _features(candidate: str, user_profile: dict) -> Dict[str, bool]:
    c = _norm(candidate)
    has_income  = ("income" in c)
    has_debts   = ("debt" in c or "debts" in c)
    has_down    = ("down payment" in c or "downpayment" in c or "down-payment" in c)
    has_credit  = ("credit" in c and ("score" in c or re.search(r"\d{3}\s*-\s*\d{3}", c)))
    has_dti     = ("dti" in c or "debt-to-income" in c)
    has_piti    = ("piti" in c or ("principal" in c and "interest" in c and "tax" in c and "insurance" in c))
    has_pmi     = ("pmi" in c or "mortgage insurance" in c)
    has_rate    = ("rate" in c or "%" in c)
    has_term    = ("30-year" in c or "30 year" in c or "term" in c)
    has_taxes   = ("tax" in c or "property tax" in c)
    has_ins     = ("insurance" in c)
    has_hoa     = ("hoa" in c)
    mentions_assume = ("assum" in c or "assumption" in c or "we assumed" in c or "assume " in c)
    has_next = _has_kw(c, _ACTION)
    hits_prot = _has_kw(c, _PROT)
    return {
        "has_income":has_income, "has_debts":has_debts, "has_down":has_down, "has_credit":has_credit,
        "has_dti":has_dti, "has_piti":has_piti, "has_pmi":has_pmi, "has_rate":has_rate, "has_term":has_term,
        "has_taxes":has_taxes, "has_ins":has_ins, "has_hoa":has_hoa,
        "mentions_assume":mentions_assume, "has_next":has_next, "hits_prot":hits_prot
    }

def deterministic_scores(candidate: str, user_profile: dict) -> Dict[str,str]:
    f = _features(candidate, user_profile)
    relevant_list = [
        "has_income","has_debts","has_down","has_credit","has_rate","has_dti",
        "has_piti","has_pmi","has_taxes","has_ins","has_hoa","has_term"
    ]
    present = sum(1 for k in relevant_list if f[k])
    relevant = len(relevant_list)

    # ANALYSIS: This is where the inflation happens!
    personalization = "Accurate" if ((f["has_income"] and f["has_debts"]) or (f["has_down"] and f["has_credit"])) else "Inaccurate"
    context_score   = _bucket_1to5(present, relevant)
    next_step       = "Present" if f["has_next"] else "Not Present"
    named = sum(int(f[k]) for k in ["has_rate","has_term","has_pmi","has_taxes","has_ins","has_hoa"])
    assumption_listing = "True" if (f["mentions_assume"] or named >= 3) else "False"
    trust_pts = 1 + (1 if f["mentions_assume"] else 0) + (1 if named >= 3 else 0)
    assumption_trust = str(max(1, min(5, trust_pts)))
    calc_acc = "True"  # ALWAYS TRUE! 
    core_mech = sum(int(f[k]) for k in ["has_dti","has_piti","has_pmi","has_rate"])
    faithful = "True" if core_mech >= 1 else "False"
    overall  = "True" if (personalization=="Accurate" and faithful=="True" and int(context_score) >= 3) else "False"
    text_len = len(_norm(candidate).split())
    bullets  = ("\n-" in candidate) or ("\n*" in candidate) or ("1." in candidate and "\n" in candidate)
    separators = (":" in candidate) or ("—" in candidate) or ("–" in candidate)
    struc_pts = (1 if bullets else 0) + (1 if separators else 0) + (1 if text_len > 120 else 0)
    structured = str(min(5, max(1, 2 + struc_pts)))  # Minimum score of 2!
    coherence = "True" if (("because" in _norm(candidate)) or ("so that" in _norm(candidate)) or core_mech >= 2) else "False"
    completeness = _bucket_1to5(present, relevant)
    fh = "False" if f["hits_prot"] else "True"  # Usually True

    return {
        "Personalization Accuracy": personalization,
        "Context based Personalization": context_score,
        "Next Step Identification": next_step,
        "Assumption Listing": assumption_listing,
        "Assumption Trust": assumption_trust,
        "Calculation Accuracy": calc_acc,
        "Faithfulness to Ground Truth": faithful,
        "Overall Accuracy": overall,
        "Structured Presentation": structured,
        "Coherence": coherence,
        "Completeness": completeness,
        "Fair Housing Classifier": fh,
    }

# Let's test with the example from the code
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

if __name__ == "__main__":
    features = _features(test_candidate, test_profile)
    scores = deterministic_scores(test_candidate, test_profile)
    
    print("=== FEATURE DETECTION ===")
    for k, v in features.items():
        print(f"{k}: {v}")
    
    print("\n=== SCORES ===")
    for k, v in scores.items():
        print(f"{k}: {v}")
    
    # Calculate what the alpha would be
    score_mapping = {
        "Accurate": 1, "Inaccurate": 0, "True": 1, "False": 0, 
        "Present": 1, "Not Present": 0,
        "1": 0, "2": 0.25, "3": 0.5, "4": 0.75, "5": 1
    }
    
    excluded = {"Structured Presentation", "Completeness"}
    alpha_scores = []
    for metric, score in scores.items():
        if metric not in excluded:
            numeric_score = score_mapping.get(score, float(score) if score.isdigit() else 0)
            if score.isdigit():
                numeric_score = (float(score) - 1) / 4  # 1-5 scale to 0-1
            alpha_scores.append(numeric_score)
            print(f"{metric}: {score} -> {numeric_score}")
    
    alpha = int(round(sum(alpha_scores) / len(alpha_scores) * 100))
    print(f"\nALPHA SCORE: {alpha}")