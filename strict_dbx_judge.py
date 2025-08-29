#!/usr/bin/env python3
"""
Databricks-only strict LLM-as-a-Judge with:
- Original Databricks secrets auth for OPENAI_API_KEY (no simplification)
- Zillow Labs gateway as OPENAI_BASE_URL
- Databricks Workspace file export helper
- Deterministic metric scoring (scores are frozen and reproducible)
- Optional: narrative justifications (if model available), aligned to frozen scores
"""

import os, io, re, json, base64, requests
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from difflib import SequenceMatcher

# ========== Authentication & Gateway (retain EXACT original style) ==========
try:
    from pyspark.dbutils import DBUtils
    dbutils = DBUtils(spark)  # type: ignore  # noqa: F821
    os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY") or dbutils.secrets.get("zillow-openai","openai-api-key")
except Exception:
    assert os.getenv("OPENAI_API_KEY"), "Set OPENAI_API_KEY via env or Databricks secret scope."

# Default to the internal gateway unless overridden.
os.environ.setdefault("OPENAI_BASE_URL", "https://api.zillowlabs.com/openai/v1")

# Optional dependencies for file parsing and OpenAI client
try:
    from docx import Document  # type: ignore
except Exception:
    Document = None  # type: ignore

try:
    from striprtf.striprtf import rtf_to_text  # type: ignore
except Exception:
    def rtf_to_text(x: str) -> str:  # fallback no-op
        return x

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore


# ========== Databricks Workspace file export (retain original style) ==========
def export_workspace_file_to_bytes(ws_path: str) -> bytes:
    """Export a /Workspace/... item to bytes (no local filesystem writes)."""
    # Normalize '/Workspace/...' → '/Users/...'
    if ws_path.startswith('/Workspace/'):
        ws_path = ws_path[len('/Workspace'):]
    ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()  # type: ignore
    api_url = ctx.apiUrl().get()
    try:
        token = ctx.apiToken().get()
    except Exception:
        token = os.getenv('DATABRICKS_TOKEN') or os.getenv('DBRKS_TOKEN')
        if not token:
            raise RuntimeError('No API token available. Enable notebook-scoped tokens or set DATABRICKS_TOKEN.')

    url = api_url.rstrip('/') + '/api/2.0/workspace/export'
    params = {'path': ws_path, 'format': 'BINARY', 'direct_download': 'true'}
    headers = {'Authorization': f'Bearer {token}'}
    r = requests.get(url, params=params, headers=headers, timeout=120)
    r.raise_for_status()
    try:
        data = r.json()
        return base64.b64decode(data['content'])
    except ValueError:
        return r.content


# ========== Loaders: DOCX/RTF fair/golden/profile ==========
def load_golden_from_docx_bytes(b: bytes) -> Dict[str, Any]:
    if not Document:
        raise RuntimeError("python-docx is required to parse DOCX.")
    doc = Document(io.BytesIO(b))  # type: ignore
    questions, q_idx, current_q, buf = {}, 0, None, []
    for p in doc.paragraphs:
        t = (p.text or '').strip()
        if not t:
            continue
        if t.endswith('?') and len(t) < 200:
            if current_q and buf:
                q_idx += 1
                questions[str(q_idx)] = {'question': current_q, 'response_template': '\n'.join(buf), 'variables': []}
                buf = []
            current_q = t
        else:
            buf.append(t)
    if current_q and buf:
        q_idx += 1
        questions[str(q_idx)] = {'question': current_q, 'response_template': '\n'.join(buf), 'variables': []}
    return {'questions': questions}


def load_profiles_from_rtf_bytes(b: bytes) -> Dict[str, Any]:
    text = rtf_to_text(b.decode('utf-8', errors='ignore'))
    profiles: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or ':' not in line:
            continue
        name, rest = line.split(':', 1)
        row: Dict[str, Any] = {'profile_id': name.strip()}
        for part in rest.split(','):
            if '=' not in part:
                continue
            k, v = [s.strip() for s in part.split('=', 1)]
            k = k.lower()
            def to_num(x):
                sx = str(x).replace(',', '').replace('$', '')
                try:
                    return int(sx)
                except Exception:
                    try:
                        return float(sx)
                    except Exception:
                        return x
            if k in ('income','annual_income'): row['annual_income'] = to_num(v)
            elif k in ('debts','monthly_debts'): row['monthly_debts'] = to_num(v)
            elif k in ('down','down_payment'):   row['down_payment']  = to_num(v)
            elif k in ('credit','credit_score','credit_score_range'): row['credit_score'] = v
            elif k in ('target','preferred_monthly_payment'): row['preferred_monthly_payment'] = to_num(v)
            elif k in ('max','comfortable_max_monthly_payment'): row['comfortable_max_monthly_payment'] = to_num(v)
        profiles.append(row)
    return {'profiles': profiles}


def load_fair_from_docx_bytes(b: bytes) -> Dict[str, Any]:
    if not Document:
        raise RuntimeError("python-docx is required to parse DOCX.")
    doc = Document(io.BytesIO(b))  # type: ignore
    bullets = [(p.text or '').strip() for p in doc.paragraphs if (p.text or '').strip()]
    return {'doc': bullets}


# ========== GPT-5 Model Selection (kept consistent with gateway) ==========
def _choose_chat_capable_model() -> Optional[str]:
    base = os.getenv("OPENAI_BASE_URL")
    key  = os.getenv("OPENAI_API_KEY")
    if not (OpenAI and key):
        return None
    try:
        client = OpenAI(base_url=base, api_key=key)
    except Exception:
        client = None

    seen: List[str] = []
    if client:
        try:
            seen = [m.id for m in client.models.list().data]
        except Exception:
            seen = []

    gpt5_seen = [m for m in seen if m.lower().startswith("gpt-5")]
    seed_candidates = [
        "gpt-5-2025-08-07",
        "gpt-5-mini-2025-08-07",
        "gpt-5",
        "gpt-5-mini",
    ]
    pool = list({*(gpt5_seen or seed_candidates)})

    def _rank(mid: str) -> int:
        ml = mid.lower(); s = 0
        if "2025-08-07" in ml: s += 100
        if ml.endswith("-mini") or "-mini-" in ml: s -= 1
        return s

    candidates = sorted(pool, key=_rank, reverse=True)

    def _supports_chat(model_id: str) -> bool:
        try:
            r = requests.post(
                f"{base}/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": model_id, "messages":[{"role":"user","content":"ping"}], "max_completion_tokens": 1},
                timeout=10,
            )
            return r.status_code == 200
        except Exception:
            return False

    for mid in candidates:
        if _supports_chat(mid):
            os.environ["OPENAI_DEPLOYMENT"] = mid
            return mid

    alias_guess = os.getenv("OPENAI_DEPLOYMENT") or "openai-gpt5-endpoint"
    if alias_guess not in candidates and _supports_chat(alias_guess):
        os.environ["OPENAI_DEPLOYMENT"] = alias_guess
        return alias_guess
    return None


# ========== Deterministic Scoring & Utilities ==========
STRICT_EXPECTED_ORDER = [
    "Personalization Accuracy",
    "Context based Personalization",
    "Next Step Identification",
    "Assumption Listing",
    "Assumption Trust",
    "Calculation Accuracy",
    "Faithfulness to Ground Truth",
    "Overall Accuracy",
    "Structured Presentation",
    "Coherence",
    "Completeness",
    "Fair Housing Classifier",
]

def _alpha_from_table_strict(table_markdown: str) -> int:
    def _to_num(score: str):
        s = (str(score) or "").strip().lower()
        if s in {"accurate","true","present"}: return 1.0
        if s in {"inaccurate","false","not present","not-present"}: return 0.0
        try:
            v = float(s)
            if 1.0 <= v <= 5.0: return (v-1.0)/4.0
            if 0.0 <= v <= 1.0: return v
        except Exception:
            return None
        return None
    lines = [ln.strip() for ln in (table_markdown or "").strip().split("\n") if ln.strip()]
    if len(lines) < 3:
        return 0
    exclude = {"Structured Presentation","Completeness"}
    vals: List[float] = []
    for ln in lines[2:]:
        if "|" not in ln: continue
        parts = [p.strip() for p in ln.split("|")]
        if len(parts) < 4: continue
        metric, score = parts[1], parts[2]
        if metric in exclude: continue
        v = _to_num(score)
        if v is not None:
            vals.append(v)
    return int(round((sum(vals)/len(vals))*100)) if vals else 0


def extract_dollar_amount(text: str, keyword_regex: str) -> Optional[float]:
    patterns = [
        rf'\$\s*(\d{{1,3}}(?:,\d{{3}})*(?:\.\d{{2}})?)\s*(?:{keyword_regex})',
        rf'(\d{{1,3}}(?:,\d{{3}})*(?:\.\d{{2}})?)\s*(?:dollars?\s*)?(?:{keyword_regex})',
        rf'{keyword_regex}[^0-9]*?\$?\s*(\d{{1,3}}(?:,\d{{3}})*(?:\.\d{{2}})?)',
    ]
    t = text.lower()
    for pat in patterns:
        m = re.findall(pat, t, flags=re.I)
        if m:
            s = (m[0] if isinstance(m[0], str) else m[0][0]).replace(',', '').replace('$', '')
            try:
                return float(s)
            except Exception:
                continue
    return None


def verify_numerical_accuracy(candidate: str, user_profile: Dict[str, Any]) -> Dict[str, bool]:
    results = {
        'income_correct': False,
        'debt_correct': False,
        'down_payment_correct': False,
        'credit_score_correct': False,
    }
    if not user_profile:
        return results

    mentioned_income = extract_dollar_amount(candidate, r'(?:income|salary|earn)')
    mentioned_debt   = extract_dollar_amount(candidate, r'(?:monthly debts?|monthly obligations?)')
    mentioned_down   = extract_dollar_amount(candidate, r'(?:down payment|downpayment|down)')

    if mentioned_income is not None and 'annual_income' in user_profile:
        actual = float(user_profile['annual_income'])
        if actual > 0 and abs(mentioned_income - actual) / actual <= 0.02:
            results['income_correct'] = True

    if mentioned_debt is not None and 'monthly_debts' in user_profile:
        actual = float(user_profile['monthly_debts'])
        if abs(mentioned_debt - actual) <= 25:
            results['debt_correct'] = True

    if mentioned_down is not None and 'down_payment' in user_profile:
        actual = float(user_profile['down_payment'])
        if abs(mentioned_down - actual) <= 200 or abs(mentioned_down - actual) / max(actual, 1.0) <= 0.02:
            results['down_payment_correct'] = True

    m = re.search(r'(\d{3})\s*[-–—]\s*(\d{3})', candidate)
    if m and 'credit_score' in user_profile:
        rng = f"{m.group(1)}-{m.group(2)}"
        if rng == str(user_profile.get('credit_score')):
            results['credit_score_correct'] = True

    return results


def compute_fair_housing(candidate: str, fair_doc: Dict[str, Any]) -> str:
    c = candidate.lower()
    # Simple doc-based guardrails using common protected terms; extend if doc has more structure
    protected = [
        'race','color','religion','sex','gender','sexual orientation','gender identity','national origin',
        'familial status','disability','age','source of income','voucher','section 8'
    ]
    issues = 0
    # Disallow explicit exclusion statements against protected terms
    for term in protected:
        pat = rf"\b(?:no|avoid|prefer|refuse|deny).{{0,30}}\b{re.escape(term)}\b"
        if re.search(pat, c, flags=re.I):
            issues += 1
    # ESA/service animal nuance
    if re.search(r"\bno\s+(?:pets|animals).{0,20}(?:allowed|permitted)\b", c, flags=re.I):
        if not re.search(r"assistance\s+animal|service\s+animal|emotional\s+support", c, flags=re.I):
            issues += 1
    # Occupancy rigid rules
    if re.search(r"\b(?:adults\s+only|no\s+minors)\b", c, flags=re.I):
        issues += 1
    if re.search(r"\bmaximum.{0,20}(?:people|persons).{0,20}(?:per|/).{0,10}bedroom\b", c, flags=re.I):
        issues += 1
    return "True" if issues == 0 else "False"


def best_golden_entry(golden: Dict[str, Any], question: str) -> Optional[Dict[str, Any]]:
    if not golden or 'questions' not in golden or not question:
        return None
    qn = question.lower().strip()
    best = None; best_score = 0.0
    for _, row in golden['questions'].items():
        cand = (row.get('question') or '').lower()
        score = SequenceMatcher(None, qn, cand).ratio()
        if score > best_score:
            best_score = score; best = row
    return best if best_score >= 0.55 else None


def compute_faithfulness(candidate: str, question: str, golden: Dict[str, Any]) -> str:
    entry = best_golden_entry(golden, question)
    if not entry:
        return "False"
    template = (entry.get('response_template') or '').lower()
    claim_pats = {
        'dti': r'\b(?:dti|debt-?\s*to-?\s*income)\b',
        'piti': r'\bpiti\b|principal[^.\n]*interest[^.\n]*tax[^.\n]*insurance',
        'pmi': r'\b(?:pmi|mortgage insurance)\b',
        'rate': r'(?:\brate\b|\binterest\b|\d+(?:\.\d+)?\s*%)',
        'term': r'\b(?:15|20|25|30)\s*-?\s*year\b',
        'tax': r'\bproperty\s+tax(?:es)?\b',
        'ins': r'\bhome(?:owner|owners|owner\'s|owners\')?\s+insurance\b|\binsurance\b',
        'hoa': r'\bhoa\b|homeowner.?s association',
        'stress': r'stress[- ]?test|rate shock|\+\s*0\.?.*%\b',
    }
    required = [k for k, pat in claim_pats.items() if re.search(pat, template, flags=re.I)]
    c = candidate.lower()
    covered = sum(1 for k in required if re.search(claim_pats[k], c, flags=re.I))
    coverage = covered / max(1, len(required))
    contradicted = any(re.search(p, c, flags=re.I) for p in [r"\bno\s+pmi\b", r"\bno\s+hoa\b", r"ignore.*rate shock"])
    numeric_implied = any(k in ("dti","piti","rate","pmi") for k in required)
    num_ok = True
    if numeric_implied:
        va = verify_numerical_accuracy(candidate, {})
        num_ok = any(va.values())
    return "True" if (coverage >= 0.6 and not contradicted and num_ok) else "False"


def compute_scores(candidate: str,
                   user_profile: Dict[str, Any],
                   question: str,
                   golden: Dict[str, Any],
                   fair_doc: Dict[str, Any]) -> Dict[str, str]:
    c = candidate.lower().strip()
    num_acc = verify_numerical_accuracy(candidate, user_profile)

    personalization = "Accurate" if (
        (("income" in c or "salary" in c) and num_acc['income_correct']) or
        (("debt" in c or "obligation" in c) and num_acc['debt_correct']) or
        (("down payment" in c or "downpayment" in c) and num_acc['down_payment_correct']) or
        (("credit" in c) and num_acc['credit_score_correct'])
    ) else "Inaccurate"

    features = [
        ("income", num_acc['income_correct']),
        ("debts", num_acc['debt_correct']),
        ("down", num_acc['down_payment_correct']),
        ("credit", num_acc['credit_score_correct']),
        ("rate%", bool(re.search(r'\d+(?:\.\d+)?\s*%.*rate', c))),
        ("dti", "dti" in c or "debt-to-income" in c),
        ("piti", "piti" in c or all(w in c for w in ["principal","interest","tax","insurance"])),
        ("pmi", "pmi" in c or "mortgage insurance" in c),
        ("tax", "property tax" in c or ("tax" in c and "propert" in c)),
        ("ins", "insurance" in c),
        ("hoa", "hoa" in c),
        ("term", bool(re.search(r'(?:15|20|25|30)\s*-?\s*year', c))),
    ]
    present = sum(1 for _, ok in features if ok)
    total = len(features)
    pct = present / total
    context_score = "1" if pct < 0.30 else "2" if pct < 0.50 else "3" if pct < 0.70 else "4" if pct < 0.85 else "5"

    action_specific = any(re.search(p, c, re.I) for p in [
        r'get pre-?approved', r'apply for', r'schedule.*appointment', r'upload.*document', r'complete.*application', r'lock.*rate', r'compare.*lenders'
    ])
    generic = any(p in c for p in ["speak with","contact","consult","talk to"])
    next_step = "Present" if action_specific and not generic else "Not Present"

    assumption_flags = sum(1 for p in [
        r'\d+(?:\.\d+)?\s*%\s*(?:interest|rate)', r'property tax', r'insurance', r'hoa', r'(?:pmi|mortgage insurance)', r'(?:15|20|25|30)\s*-?\s*year'
    ] if re.search(p, c, re.I))
    has_lang = any(w in c for w in ["assum", "we used", "assuming"])
    assumption_listing = "True" if has_lang and assumption_flags >= 3 else "False"
    assumption_trust = str(min(5, 1 + int(has_lang) + (1 if assumption_flags >= 3 else 0) + (1 if assumption_flags >= 5 else 0)))

    has_calc = ("dti" in c or "piti" in c or re.search(r'\d+(?:\.\d+)?\s*%.*rate', c))
    calc_acc = "True" if (not has_calc or sum(num_acc.values()) >= 2) else "False"

    faithful = compute_faithfulness(candidate, question, golden)
    fair_ok = compute_fair_housing(candidate, fair_doc)

    overall = "True" if (
        personalization == "Accurate" and faithful == "True" and calc_acc == "True" and int(context_score) >= 3
    ) else "False"

    structured = "5" if (candidate.count("\n") > 5 and re.search(r'^\s*(?:[-*•]|\d+\.)\s', candidate, re.M)) else "3" if ("\n\n" in candidate) else "2"
    coherence = "True" if any(w in c for w in ["because","since","therefore","thus"]) else "False"
    completeness = context_score

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
        "Fair Housing Classifier": fair_ok,
    }


def build_table(frozen_scores: Dict[str, str], justifications: Optional[Dict[str, str]] = None) -> str:
    lines = ["| Metric | Score | Justification |", "|--------|-------|---------------|"]
    justifications = justifications or {}
    for m in STRICT_EXPECTED_ORDER:
        s = frozen_scores.get(m, "")
        j = (justifications.get(m, "") or "").replace("\n", " ").strip()
        lines.append(f"| {m} | {s} | {j} |")
    return "\n".join(lines)


def backfill_justifications(candidate: str,
                            question: str,
                            user_profile: Dict[str, Any],
                            frozen_scores: Dict[str, str],
                            *,
                            min_w: int = 50,
                            max_w: int = 90,
                            max_tokens: int = 700) -> Dict[str, str]:
    model = os.getenv("OPENAI_DEPLOYMENT") or _choose_chat_capable_model()
    key   = os.getenv("OPENAI_API_KEY")
    base  = os.getenv("OPENAI_BASE_URL")
    if not (OpenAI and model and key):
        return {}
    client = OpenAI(base_url=base, api_key=key)
    system = (
        "You are an evaluator. For each requested metric, write ONE paragraph "
        f"({min_w}–{max_w} words) that supports the TARGET score for that metric. "
        "Ground in the question, user profile, and candidate answer. "
        "Return STRICT JSON mapping metric→paragraph."
    )
    need = list(STRICT_EXPECTED_ORDER)
    targets = {m: frozen_scores.get(m) for m in need}
    user = (
        "CONTEXT:\n"
        f"- Question: {question}\n"
        f"- User profile: {json.dumps(user_profile or {})}\n"
        f"- Candidate answer:\n{candidate}\n\n"
        f"NEED JUSTIFICATIONS FOR METRICS: {need}\n"
        f"TARGET SCORES: {json.dumps(targets)}"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        max_completion_tokens=max_tokens,
        temperature=0,
        top_p=0.1,
        seed=int(os.getenv("EVALUATION_SEED","42")),
    )
    txt = (resp.choices[0].message.content or "").strip()
    try:
        data = json.loads(txt)
        return {k: (str(v) if not isinstance(v, str) else v).strip() for k, v in data.items()}
    except Exception:
        m = re.search(r"\{[\s\S]*\}", txt)
        return json.loads(m.group(0)) if m else {}


@dataclass
class EvaluationResult:
    evaluation_results_markdown: str
    alpha: int
    model: Optional[str]
    seed: int
    deterministic: bool


def evaluate_llm_with_files_STRICT(
    question: str,
    candidate_answer: str,
    user_profile: Dict[str, Any],
    *,
    golden_dict: Dict[str, Any],
    fair_doc: Dict[str, Any],
    model: Optional[str] = None,
    seed: int = 42,
    generate_justifications: bool = True,
) -> EvaluationResult:
    chosen_model = model or os.getenv("OPENAI_DEPLOYMENT") or _choose_chat_capable_model()

    # 1) Deterministic frozen scores
    frozen = compute_scores(candidate_answer, user_profile or {}, question or "", golden_dict or {}, fair_doc or {})

    # 2) Optional narratives aligned to frozen scores
    just_map: Dict[str, str] = {}
    if generate_justifications:
        just_map = backfill_justifications(candidate_answer, question, user_profile or {}, frozen)

    # 3) Table + alpha
    table = build_table(frozen, just_map)
    alpha = _alpha_from_table_strict(table)

    return EvaluationResult(
        evaluation_results_markdown=table,
        alpha=int(round(alpha)),
        model=chosen_model,
        seed=seed,
        deterministic=True,
    )


if __name__ == "__main__":
    # Example usage inside Databricks: provide docs via Workspace exports
    # golden_bytes = export_workspace_file_to_bytes('/Workspace/Users/you/golden.docx')
    # golden = load_golden_from_docx_bytes(golden_bytes)
    # fair_doc = json.loads(export_workspace_file_to_bytes('/Workspace/Users/you/fair.json').decode('utf-8'))
    # For demo, minimal placeholders:
    golden = {"questions": {"1": {"question":"What factors were considered to calculate my Buyability?",
                                    "response_template":"We considered DTI, PITI, PMI, term, rate, taxes, insurance, HOA, stress test."}}}
    fair_doc = {"doc": []}

    question = "What factors were considered to calculate my Buyability?"
    user_profile = {"annual_income": 90000, "monthly_debts": 200, "down_payment": 18000, "credit_score": "660-719"}
    candidate = (
        "We estimated your Buyability using income, monthly debts, down payment, and your 660–719 credit tier. "
        "Assumed a 30-year fixed at today's rate, included taxes, insurance, and PMI if <20% down. "
        "Checked DTI thresholds and sized PITI to fit your budget; also ran a +0.5% rate shock. "
        "Next step: get pre-approved online and upload your documents to lock a rate."
    )

    res = evaluate_llm_with_files_STRICT(
        question=question,
        candidate_answer=candidate,
        user_profile=user_profile,
        golden_dict=golden,
        fair_doc=fair_doc,
        generate_justifications=False,
    )
    print(res.evaluation_results_markdown)
    print("Alpha:", res.alpha, "| Model:", res.model, "| Deterministic Scores:", res.deterministic)

