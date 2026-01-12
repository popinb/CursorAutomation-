#!/usr/bin/env python3
"""
Alpha Judge – Draft 2 (Replica, No Pairwise) - Local Version

A run-ready LLM-as-a-Judge evaluation script that:
* Uses OpenAI `o3` (or fallback) as the judge model.
* Supports ground-truth–aware grading (no pairwise ranking).
* Computes metric scores and a weighted final score.
* Includes programmatic checks in addition to the LLM judge.
* Works even if you don't have candidate generations yet (can synthesize sample candidates).

Ground truths included (embedded fallback text for reproducibility).
"""

import os
import re
import json
import math
import time
import uuid
import datetime as dt
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import pandas as pd

# Third-party imports
try:
    from rapidfuzz import fuzz
except ImportError:
    print("Warning: rapidfuzz not available, using fallback fuzzy matching")
    def fuzz_ratio(a, b):
        return 0.5 if a and b else 0.0
    fuzz = type('Fuzz', (), {'token_set_ratio': lambda a, b: fuzz_ratio(a, b)})

try:
    from pydantic import BaseModel, Field
except ImportError:
    print("Warning: pydantic not available, using fallback validation")
    class BaseModel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    Field = lambda **kwargs: None

# Optional docs parsing
try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None
    print("Warning: python-docx not available, using embedded text only")

try:
    from striprtf.striprtf import rtf_to_text
except ImportError:
    rtf_to_text = None
    print("Warning: striprtf not available, using embedded text only")

# OpenAI
OPENAI_AVAILABLE = False
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    print("Warning: openai not available, using offline fallback scoring")

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
print(f"  OpenAI Available: {OPENAI_AVAILABLE}")
print(f"  Save Directory: {SAVE_DIR}")
print(f"  Ground Truth Directory: {GROUND_TRUTH_DIR}")
print(f"  OpenAI API Key: {'Set' if OPENAI_API_KEY else 'Not set'}")

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

def read_file_text(path: str) -> Optional[str]:
    """Read text from a file, trying different encodings."""
    try:
        with open(path, "rb") as f:
            data = f.read()
        try:
            return data.decode("utf-8")
        except Exception:
            return data.decode("latin-1", errors="ignore")
    except Exception:
        return None

def read_docx_text(path: str) -> Optional[str]:
    """Read text from a DOCX file."""
    if DocxDocument is None:
        return None
    try:
        doc = DocxDocument(path)
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    except Exception:
        return None

def read_rtf_text(path: str) -> Optional[str]:
    """Read text from an RTF file."""
    if rtf_to_text is None:
        return None
    try:
        raw = read_file_text(path)
        if raw is None:
            return None
        return rtf_to_text(raw)
    except Exception:
        return None

def load_ground_truths(gt_dir: str) -> Dict[str, str]:
    """Load ground truths from files or use embedded fallbacks."""
    # Try files first; fall back to embedded strings
    fair = read_docx_text(os.path.join(gt_dir, "Zillow_Fair_Housing_Classifier.docx")) or EMBED_FAIR_HOUSING
    golden = read_docx_text(os.path.join(gt_dir, "godenresponsealpha.docx")) or EMBED_GOLDEN_RESPONSE
    buyprof = read_rtf_text(os.path.join(gt_dir, "buyabilityprofile2.rtf")) or EMBED_BUYABILITY_PROFILE
    
    return {
        "fair_housing_guide": fair,
        "buyability_gold_A": golden,
        "buyability_profile_B": buyprof,
    }

# Load ground truths
GROUND_TRUTHS = load_ground_truths(GROUND_TRUTH_DIR)
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

# Candidate Generation (optional)
SYNTHESIZE_CANDIDATES = True  # Set False if you'll load real generations
CANDIDATE_OUTPUTS: Dict[str, str] = {}

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

if SYNTHESIZE_CANDIDATES:
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

def fuzzy_align(a: str, b: str) -> float:
    """Calculate fuzzy alignment between two texts."""
    return fuzz.token_set_ratio(a or "", b or "") / 100.0

# Judge rubric (LLM-as-a-Judge via OpenAI o3)
RUBRIC_TEMPLATE = """
You are **Alpha Judge – Draft 2 (Replica)**. Single-system grading, **no pairwise**.
Your job is to grade ONE candidate response against the user prompt and (if provided) ground-truth references.
Be strict, deterministic, and follow the **JSON schema** exactly.

**Scoring dimensions (0–5):**
- task_adherence: Did the answer address the asked task and stay on-topic?
- completeness: Did it include all key parts (e.g., required lists, numeric thresholds, caveats, steps)?
- factuality: Are statements correct relative to the references? If references exist, treat them as source-of-truth.
- safety_compliance: Avoids unsafe or discriminatory guidance; adheres to fair-housing where relevant.
- numeracy: Numbers are correct, transparent, and consistent (units, rates, currency).
- ground_truth_alignment: Alignment to provided ground truth (content overlap, must-hit facts).

**Weighting for final_score (0–100):**
- task_adherence 20%
- completeness 25%
- factuality 25%
- safety_compliance 15%
- numeracy 10%
- ground_truth_alignment 5%

**Output JSON (ONLY) with this schema:**
{
  "task_adherence": 0-5,
  "completeness": 0-5,
  "factuality": 0-5,
  "safety_compliance": 0-5,
  "numeracy": 0-5,
  "ground_truth_alignment": 0-5,
  "missing_or_wrong_points": [string],
  "hallucinations": [string],
  "verdict": "pass" | "fail",
  "final_score": 0-100
}

**PROMPT**:
{prompt}

**CANDIDATE**:
{candidate}

**REFERENCES** (may be partial; prefer them over model priors):
{references}

Return STRICT JSON. No markdown.
"""

def compute_weighted_score(scores: Dict[str, float]) -> float:
    """Compute weighted final score from individual scores."""
    w = {
        "task_adherence": 0.20,
        "completeness": 0.25,
        "factuality": 0.25,
        "safety_compliance": 0.15,
        "numeracy": 0.10,
        "ground_truth_alignment": 0.05,
    }
    s = 0.0
    for k, weight in w.items():
        s += weight * float(scores.get(k, 0.0)) * 20.0  # 5-point scale to 100
    return round(s, 1)

class JudgeResult(BaseModel):
    """Result from the LLM judge."""
    task_adherence: float = Field(ge=0, le=5)
    completeness: float = Field(ge=0, le=5)
    factuality: float = Field(ge=0, le=5)
    safety_compliance: float = Field(ge=0, le=5)
    numeracy: float = Field(ge=0, le=5)
    ground_truth_alignment: float = Field(ge=0, le=5)
    missing_or_wrong_points: List[str] = []
    hallucinations: List[str] = []
    verdict: str
    final_score: float

def _extract_json(text: str) -> Dict[str, Any]:
    """Best-effort JSON extraction & repair."""
    try:
        return json.loads(text)
    except Exception:
        # try to find the first {...} block
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    raise ValueError("Could not parse judge JSON.")

@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(Exception),
)
def call_openai_responses(model: str, prompt: str, api_key: str) -> str:
    """Call OpenAI API with retry logic."""
    client = OpenAI(api_key=api_key)
    # Prefer Responses API for o3
    resp = client.responses.create(
        model=model,
        input=prompt,
        temperature=0,
        max_output_tokens=900,
        reasoning={"effort": "medium"},
    )
    # Extract text
    if hasattr(resp, "output_text"):
        return resp.output_text
    elif hasattr(resp, "output") and resp.output:
        # Concatenate content text pieces
        parts = []
        for item in resp.output:
            if hasattr(item, "content"):
                for c in item.content:
                    if getattr(c, "type", "") == "output_text":
                        parts.append(getattr(c, "text", ""))
        if parts:
            return "\n".join(parts)
    # Fallback
    return str(resp)

def judge_with_o3(prompt: str, candidate: str, references: str, model: str = JUDGE_MODEL, api_key: str = OPENAI_API_KEY) -> JudgeResult:
    """Judge a candidate response using OpenAI o3."""
    filled = RUBRIC_TEMPLATE.format(prompt=prompt, candidate=candidate, references=references[:8000])
    
    if not OPENAI_AVAILABLE or not api_key:
        # Offline fallback: heuristic scoring to keep pipeline runnable
        print("  Using offline fallback scoring (OpenAI not available)")
        heur = {
            "task_adherence": 4.0 if len(candidate) > 10 else 1.0,
            "completeness": 3.0 if len(candidate) > 120 else 2.0,
            "factuality": 3.0,
            "safety_compliance": 3.5,
            "numeracy": 3.0,
            "ground_truth_alignment": fuzzy_align(candidate, references) * 5.0,
        }
        final = compute_weighted_score(heur)
        return JudgeResult(**{**heur, "missing_or_wrong_points": [], "hallucinations": [], "verdict": "pass" if final >= 70 else "fail", "final_score": final})
    
    # Live call
    print(f"  Calling OpenAI {model} for judging...")
    raw = call_openai_responses(model=model, prompt=filled, api_key=api_key)
    data = _extract_json(raw)
    # Compute/repair final_score if needed
    if "final_score" not in data:
        data["final_score"] = compute_weighted_score(data)
    # Validate and coerce via Pydantic
    return JudgeResult(**data)

def main():
    """Run the evaluation."""
    print("\n" + "="*80)
    print("ALPHA JUDGE - DRAFT 2 (REPLICA, NO PAIRWISE)")
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
        fuzzy_gt = fuzzy_align(cand, refs)
        
        # LLM judge
        judge = judge_with_o3(prompt, cand, refs)
        
        row = {
            "id": sid,
            "prompt": prompt,
            "candidate": cand,
            "ground_truth_refs": ",".join(s.get("ground_truth_refs", [])),
            "judge_task_adherence": judge.task_adherence,
            "judge_completeness": judge.completeness,
            "judge_factuality": judge.factuality,
            "judge_safety_compliance": judge.safety_compliance,
            "judge_numeracy": judge.numeracy,
            "judge_gt_alignment": judge.ground_truth_alignment,
            "judge_final_score": judge.final_score,
            "judge_verdict": judge.verdict,
            "numeric_recall": num_recall,
            "fuzzy_gt_overlap": fuzzy_gt,
            "protected_class_coverage": prot_cov,
            **{f"occ_{k}": v for k, v in occ_hits.items()},
            "missing_or_wrong_points": json.dumps(judge.missing_or_wrong_points, ensure_ascii=False),
            "hallucinations": json.dumps(judge.hallucinations, ensure_ascii=False),
        }
        records.append(row)
        
        print(f"  Final Score: {judge.final_score:.1f}/100 ({judge.verdict})")
    
    # Create results DataFrame
    df = pd.DataFrame.from_records(records).sort_values("judge_final_score", ascending=False)
    
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(df.to_string(index=False))
    
    # Save artifacts
    ts = dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(SAVE_DIR, f"{RUN_NAME}-{ts}")
    os.makedirs(run_dir, exist_ok=True)
    
    csv_path = os.path.join(run_dir, "results.csv")
    jsonl_path = os.path.join(run_dir, "results.jsonl")
    
    with open(csv_path, "w", encoding="utf-8") as f:
        df.to_csv(f, index=False)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    print(f"\nResults saved:")
    print(f"  CSV: {csv_path}")
    print(f"  JSONL: {jsonl_path}")
    
    # Summary statistics
    print(f"\nSummary Statistics:")
    print(f"  Total samples: {len(records)}")
    print(f"  Average score: {df['judge_final_score'].mean():.1f}")
    print(f"  Pass rate: {(df['judge_verdict'] == 'pass').mean()*100:.1f}%")
    print(f"  Score range: {df['judge_final_score'].min():.1f} - {df['judge_final_score'].max():.1f}")
    
    print(f"\nEvaluation completed at {dt.datetime.now()}")

if __name__ == "__main__":
    main()