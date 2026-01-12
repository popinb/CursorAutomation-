#!/usr/bin/env python3
"""
Alpha Judge – Draft 2 (Replica, No Pairwise) - Simple Local Version

A simplified version that works with minimal dependencies.
"""

import os
import re
import json
import datetime as dt
from typing import Dict, Any, List, Optional, Tuple

# Configuration
JUDGE_MODEL = os.environ.get("ALPHA_JUDGE_MODEL", "o3")
CANDIDATE_MODEL = os.environ.get("ALPHA_CANDIDATE_MODEL", "gpt-4o-mini")
RUN_NAME = os.environ.get("ALPHA_RUN_NAME", f"alpha-judge-d2-{dt.datetime.utcnow().strftime('%Y%m%d-%H%M%S')}")
SAVE_DIR = os.environ.get("ALPHA_SAVE_DIR", "./alpha_judge_runs")
GROUND_TRUTH_DIR = os.environ.get("ALPHA_GT_DIR", "./ground_truths")

# OpenAI API Key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Create directories
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(GROUND_TRUTH_DIR, exist_ok=True)

print(f"Configuration:")
print(f"  Judge Model: {JUDGE_MODEL}")
print(f"  OpenAI Available: {'Set' if OPENAI_API_KEY else 'Not set'}")
print(f"  Save Directory: {SAVE_DIR}")
print(f"  Ground Truth Directory: {GROUND_TRUTH_DIR}")

# Embedded Ground Truths (fallback)
EMBED_FAIR_HOUSING = '''
Protected classes under federal law include: Race, Color, Religion, Sex (including sexual orientation and gender identity), National origin, Familial status, and Physical or mental disability.
Occupancy rules often reference:
- Keating Memorandum: Two persons per bedroom (with exceptions based on unit size/configuration, system capacity, child age, and local law).
- BOCA code: 150 sq ft for the first occupant; +100 sq ft for each additional occupant. Sleeping rooms: at least 70 sq ft for one occupant, or at least 50 sq ft per person if occupied by more than one person.
Additional examples of prohibited acts: discriminatory advertising (e.g., targeting a specific group), inconsistent qualification criteria, steering, harassment/retaliation, and failure to accommodate disabilities.
'''

EMBED_GOLDEN_RESPONSE = '''
Your personalized BuyAbility estimate is $318,431, based on: Annual Income $90,000; Monthly Debts $200; Down Payment $18,000; Credit Score 660–719; Target Monthly Payment $2,500; Location Georgia; Rate 6.88%.
Monthly payment breakdown example for $2,500 target: Principal & Interest $1,975; Property Taxes $212; Homeowners Insurance $106; PMI $207 (if <20% down). Notes: HOA not included; tax uses state average. Guidance: With BuyAbility at $318K and DTI < 36%, you're positioned to shop many GA submarkets.
Loan options: Conventional 3–5% down (no MI at ≥20%); FHA 3.5% down; VA 0% down; Jumbo typically 10–20% down.
'''

EMBED_BUYABILITY_PROFILE = '''
Buyability Profile (Scenario B)
Location: Arkansas
Monthly payment input by user: $2,000
Monthly payment calculated by Buyability: $8,133
Available funds: $50,000
Annual income: $200,000
Monthly debt: $200
Credit score: 660–719
Maximum affordable home price: $1,168,227
'''

# Load ground truths
GROUND_TRUTHS = {
    "fair_housing_guide": EMBED_FAIR_HOUSING,
    "buyability_gold_A": EMBED_GOLDEN_RESPONSE,
    "buyability_profile_B": EMBED_BUYABILITY_PROFILE,
}

print(f"Loaded {len(GROUND_TRUTHS)} ground truth references")

# Evaluation Dataset
EVAL_SAMPLES = [
    {
        "id": "buyability_A",
        "prompt": (
            "Given these user inputs: income $90,000; monthly debts $200; down payment $18,000; credit score 660–719; "
            "target monthly payment $2,500; location Georgia; and rate 6.88%. "
            "Explain their BuyAbility and provide a transparent monthly payment breakdown and next steps."
        ),
        "ground_truth_refs": ["buyability_gold_A"],
        "tags": ["affordability", "numeracy"],
    },
    {
        "id": "fair_housing_protected_classes",
        "prompt": (
            "List the protected classes under federal fair housing law. Also give one practical compliance tip for rental advertising."
        ),
        "ground_truth_refs": ["fair_housing_guide"],
        "tags": ["compliance"],
    },
    {
        "id": "occupancy_guidelines",
        "prompt": (
            "Summarize HUD's Keating Memorandum and BOCA occupancy guidance, including numeric thresholds, and advise how to avoid discrimination when setting occupancy rules."
        ),
        "ground_truth_refs": ["fair_housing_guide"],
        "tags": ["compliance", "policy"],
    },
    {
        "id": "buyability_B",
        "prompt": (
            "For this Arkansas scenario: monthly payment target $2,000; income $200,000; monthly debt $200; available funds $50,000; "
            "credit score 660–719. Provide an estimated max affordable home price and key caveats."
        ),
        "ground_truth_refs": ["buyability_profile_B"],
        "tags": ["affordability", "numeracy"],
    },
]

print(f"Evaluation dataset: {len(EVAL_SAMPLES)} samples")

# Candidate Generation (synthetic for testing)
CANDIDATE_OUTPUTS = {}

def synthesize_candidate_outputs(samples: List[Dict[str, Any]], gts: Dict[str, str]) -> Dict[str, str]:
    """Synthesize candidate outputs for testing."""
    out = {}
    for s in samples:
        sid = s["id"]
        if sid == "buyability_A":
            # Intentionally omit PMI and tweak a number to exercise the judge
            out[sid] = (
                "Your BuyAbility is about $320,000. Monthly payment could be around $2,500 comprised of roughly "
                "$2,000 principal & interest, $212 taxes, and $100 insurance. You should be fine in Georgia."
            )
        elif sid == "fair_housing_protected_classes":
            # Miss a couple of protected classes to test coverage
            out[sid] = (
                "Protected classes include race, color, religion, sex, and national origin. "
                "Tip: Focus your ad on the property features and avoid describing an 'ideal tenant' profile."
            )
        elif sid == "occupancy_guidelines":
            out[sid] = (
                "Follow a general guideline like two people per bedroom or go by square footage rules. "
                "Avoid restrictive rules that single out families with children."
            )
        elif sid == "buyability_B":
            out[sid] = (
                "With your strong income and low debts, a max price near $1.1M could be feasible at a $2,000 payment. "
                "Rates and taxes will affect this; confirm with a lender."
            )
        else:
            out[sid] = "N/A"
    return out

CANDIDATE_OUTPUTS = synthesize_candidate_outputs(EVAL_SAMPLES, GROUND_TRUTHS)
print(f"Synthesized {len(CANDIDATE_OUTPUTS)} candidate outputs")

# Programmatic Metrics
NUMERIC_PATTERN = re.compile(r"""(?<!\w)(?P<cur>[$])?(?P<num>\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)(?P<pct>\s?%)?""")

def extract_numbers(text: str) -> List[str]:
    """Extract numbers from text."""
    return [m.group().strip() for m in NUMERIC_PATTERN.finditer(text or "")]

def normalize_number_token(tok: str) -> Tuple[str, Optional[float], bool, bool]:
    """Normalize a number token."""
    tok = tok.replace(',', '').strip()
    is_pct = tok.endswith('%')
    is_cur = tok.startswith('$')
    core = tok[1:] if is_cur else tok
    core = core[:-1] if is_pct else core
    try:
        val = float(core)
    except Exception:
        val = None
    return tok, val, is_pct, is_cur

def numeric_recall(ground_text: str, candidate_text: str, rel_tol: float = 0.01, abs_tol: float = 1.0) -> float:
    """Calculate numeric recall between ground truth and candidate."""
    gt_tokens = [normalize_number_token(t) for t in extract_numbers(ground_text)]
    cand_tokens = [normalize_number_token(t) for t in extract_numbers(candidate_text)]
    if not gt_tokens:
        return float('nan')
    hits = 0
    for _, gval, g_pct, g_cur in gt_tokens:
        if gval is None:
            continue
        found = False
        for _, cval, c_pct, c_cur in cand_tokens:
            if cval is None:
                continue
            # require matching unit type if present
            if g_pct != c_pct or g_cur != c_cur:
                continue
            if abs(cval - gval) <= max(abs_tol, rel_tol * max(abs(gval), 1.0)):
                found = True
                break
        if found:
            hits += 1
    return hits / max(1, len([1 for _, v, _, _ in gt_tokens if v is not None]))

FAIR_HOUSING_PROTECTED = {
    "race","color","religion","sex","national origin","familial status","disability",
    # explicit note
    "sexual orientation","gender identity"
}

def protected_class_coverage(candidate_text: str) -> float:
    """Calculate protected class coverage for fair housing content."""
    text = (candidate_text or "").lower()
    covered = 0
    total = len(FAIR_HOUSING_PROTECTED)
    for cls in FAIR_HOUSING_PROTECTED:
        if cls in text:
            covered += 1
    return covered / max(1, total)

def occupancy_rule_hits(candidate_text: str) -> Dict[str, bool]:
    """Check for occupancy rule mentions."""
    text = (candidate_text or "").lower()
    return {
        "mentions_two_per_bedroom": ("two" in text and "per bedroom" in text) or ("2" in text and "per bedroom" in text),
        "mentions_150_sqft": "150" in text and "sq" in text,
        "mentions_100_sqft_per_addl": "100" in text and "sq" in text,
        "mentions_70_sqft_single": "70" in text and "sq" in text,
        "mentions_50_sqft_multi": "50" in text and "sq" in text,
    }

def simple_fuzzy_align(a: str, b: str) -> float:
    """Simple fuzzy alignment between two texts."""
    if not a or not b:
        return 0.0
    a_words = set(a.lower().split())
    b_words = set(b.lower().split())
    if not a_words or not b_words:
        return 0.0
    intersection = a_words.intersection(b_words)
    union = a_words.union(b_words)
    return len(intersection) / len(union) if union else 0.0

# Simple scoring system (no OpenAI required)
def simple_judge_scoring(prompt: str, candidate: str, references: str) -> Dict[str, Any]:
    """Simple heuristic scoring when OpenAI is not available."""
    
    # Task adherence: check if candidate addresses key terms from prompt
    prompt_words = set(prompt.lower().split())
    candidate_words = set(candidate.lower().split())
    task_overlap = len(prompt_words.intersection(candidate_words)) / max(1, len(prompt_words))
    task_adherence = min(5.0, 1.0 + task_overlap * 4.0)
    
    # Completeness: length-based with some content analysis
    completeness = min(5.0, 1.0 + len(candidate) / 100.0)
    
    # Factuality: assume good if references are provided
    factuality = 3.5 if references.strip() else 3.0
    
    # Safety compliance: check for obvious issues
    safety_issues = any(word in candidate.lower() for word in ["discriminate", "illegal", "unfair"])
    safety_compliance = 4.0 if not safety_issues else 2.0
    
    # Numeracy: check for numbers
    numbers = extract_numbers(candidate)
    numeracy = min(5.0, 1.0 + len(numbers) * 0.5)
    
    # Ground truth alignment
    gt_alignment = simple_fuzzy_align(candidate, references) * 5.0
    
    # Compute weighted final score
    weights = {
        "task_adherence": 0.20,
        "completeness": 0.25,
        "factuality": 0.25,
        "safety_compliance": 0.15,
        "numeracy": 0.10,
        "ground_truth_alignment": 0.05,
    }
    
    final_score = sum(weights[k] * v for k, v in {
        "task_adherence": task_adherence,
        "completeness": completeness,
        "factuality": factuality,
        "safety_compliance": safety_compliance,
        "numeracy": numeracy,
        "ground_truth_alignment": gt_alignment,
    }.items()) * 20.0  # Convert to 0-100 scale
    
    return {
        "task_adherence": round(task_adherence, 1),
        "completeness": round(completeness, 1),
        "factuality": round(factuality, 1),
        "safety_compliance": round(safety_compliance, 1),
        "numeracy": round(numeracy, 1),
        "ground_truth_alignment": round(gt_alignment, 1),
        "missing_or_wrong_points": [],
        "hallucinations": [],
        "verdict": "pass" if final_score >= 70 else "fail",
        "final_score": round(final_score, 1)
    }

def main():
    """Run the evaluation."""
    print("\n" + "="*80)
    print("ALPHA JUDGE - DRAFT 2 (REPLICA, NO PAIRWISE) - SIMPLE VERSION")
    print("="*80)
    
    print(f"\nStarting evaluation at {dt.datetime.now()}")
    print(f"Run name: {RUN_NAME}")
    
    # Run evaluation
    print("\nRunning evaluation...")
    records = []
    
    for i, s in enumerate(EVAL_SAMPLES, 1):
        sid = s["id"]
        print(f"\n[{i}/{len(EVAL_SAMPLES)}] Evaluating: {sid}")
        
        prompt = s["prompt"]
        refs = "\n\n".join([GROUND_TRUTHS.get(ref, "") for ref in s.get("ground_truth_refs", [])])
        cand = CANDIDATE_OUTPUTS.get(sid, "")
        
        print(f"  Prompt: {prompt[:100]}...")
        print(f"  Candidate: {cand[:100]}...")
        
        # Programmatic metrics
        num_recall = numeric_recall(refs, cand)
        prot_cov = protected_class_coverage(cand) if "fair_housing" in refs.lower() or "protected" in refs.lower() else float('nan')
        occ_hits = occupancy_rule_hits(cand) if "occupancy" in prompt.lower() or "boca" in refs.lower() else {}
        fuzzy_gt = simple_fuzzy_align(cand, refs)
        
        # Simple judge scoring (no OpenAI required)
        judge = simple_judge_scoring(prompt, cand, refs)
        
        row = {
            "id": sid,
            "prompt": prompt,
            "candidate": cand,
            "ground_truth_refs": ",".join(s.get("ground_truth_refs", [])),
            "judge_task_adherence": judge["task_adherence"],
            "judge_completeness": judge["completeness"],
            "judge_factuality": judge["factuality"],
            "judge_safety_compliance": judge["safety_compliance"],
            "judge_numeracy": judge["numeracy"],
            "judge_gt_alignment": judge["ground_truth_alignment"],
            "judge_final_score": judge["final_score"],
            "judge_verdict": judge["verdict"],
            "numeric_recall": num_recall,
            "fuzzy_gt_overlap": fuzzy_gt,
            "protected_class_coverage": prot_cov,
            **{f"occ_{k}": v for k, v in occ_hits.items()},
            "missing_or_wrong_points": json.dumps(judge["missing_or_wrong_points"], ensure_ascii=False),
            "hallucinations": json.dumps(judge["hallucinations"], ensure_ascii=False),
        }
        records.append(row)
        
        print(f"  Final Score: {judge['final_score']:.1f}/100 ({judge['verdict']})")
    
    # Display results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    # Simple table display
    print(f"{'ID':<25} {'Score':<8} {'Verdict':<6} {'Task':<6} {'Complete':<8} {'Factual':<8}")
    print("-" * 80)
    for r in records:
        print(f"{r['id']:<25} {r['judge_final_score']:<8.1f} {r['judge_verdict']:<6} {r['judge_task_adherence']:<6.1f} {r['judge_completeness']:<8.1f} {r['judge_factuality']:<8.1f}")
    
    # Save artifacts
    ts = dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(SAVE_DIR, f"{RUN_NAME}-{ts}")
    os.makedirs(run_dir, exist_ok=True)
    
    csv_path = os.path.join(run_dir, "results.csv")
    jsonl_path = os.path.join(run_dir, "results.jsonl")
    
    # Simple CSV writing
    with open(csv_path, "w", encoding="utf-8") as f:
        # Write header
        headers = list(records[0].keys())
        f.write(",".join(headers) + "\n")
        # Write data
        for r in records:
            values = [str(r.get(h, "")) for h in headers]
            f.write(",".join(values) + "\n")
    
    # JSONL writing
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    print(f"\nResults saved:")
    print(f"  CSV: {csv_path}")
    print(f"  JSONL: {jsonl_path}")
    
    # Summary statistics
    scores = [r['judge_final_score'] for r in records]
    pass_count = sum(1 for r in records if r['judge_verdict'] == 'pass')
    
    print(f"\nSummary Statistics:")
    print(f"  Total samples: {len(records)}")
    print(f"  Average score: {sum(scores)/len(scores):.1f}")
    print(f"  Pass rate: {pass_count/len(records)*100:.1f}%")
    print(f"  Score range: {min(scores):.1f} - {max(scores):.1f}")
    
    print(f"\nEvaluation completed at {dt.datetime.now()}")
    print(f"\nNote: This is a simplified version using heuristic scoring.")
    print(f"      For full OpenAI o3 judging, set OPENAI_API_KEY and use alpha_judge_local.py")

if __name__ == "__main__":
    main()