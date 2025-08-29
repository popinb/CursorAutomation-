# EXACT REPLACEMENT GUIDE: Where to Make Changes

## 🎯 **Results Summary**
- **OLD SCORE: 85** (inflated)
- **NEW SCORE: 42** (realistic)
- **IMPROVEMENT: 43 points lower** ✅

## 📝 **Changes Required in Your Original Code**

### **STEP 1: Replace Helper Functions** 
**Location:** Around lines 220-280 in your original code

**FIND THIS CODE:**
```python
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()

def _has_kw(s: str, pat: str) -> bool:
    return bool(re.search(pat, s, flags=re.I))

def _features(candidate: str, user_profile: dict) -> Dict[str, bool]:
    # ... existing feature detection code
```

**REPLACE WITH:**
```python
# Copy the entire helper functions section from fixed_scoring_system.py
# Lines 1-120 (everything before "IMPROVED FEATURE DETECTION")
```

### **STEP 2: Replace Feature Detection**
**Location:** Around lines 280-320 in your original code

**FIND THIS CODE:**
```python
def _features(candidate: str, user_profile: dict) -> Dict[str, bool]:
    c = _norm(candidate)
    has_income  = ("income" in c)
    has_debts   = ("debt" in c or "debts" in c)
    # ... rest of old feature detection
```

**REPLACE WITH:**
```python
def _features_improved(candidate: str, user_profile: dict) -> Dict[str, bool]:
    # Copy from fixed_scoring_system.py lines 130-165
```

### **STEP 3: Replace Bucket Function**
**Location:** Around lines 320-330 in your original code

**FIND THIS CODE:**
```python
def _bucket_1to5(present:int, relevant:int) -> str:
    if relevant <= 0: return "1"
    pct = present/relevant
    if pct < 0.20: return "1"
    if pct < 0.40: return "2"
    if pct < 0.60: return "3"
    if pct < 0.80: return "4"
    return "5"
```

**REPLACE WITH:**
```python
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
```

### **STEP 4: Replace Main Scoring Function**
**Location:** Around lines 330-400 in your original code

**FIND THIS CODE:**
```python
def deterministic_scores(candidate: str, user_profile: dict) -> Dict[str,str]:
    f = _features(candidate, user_profile)
    relevant_list = [
        "has_income","has_debts","has_down","has_credit","has_rate","has_dti",
        "has_piti","has_pmi","has_taxes","has_ins","has_hoa","has_term"
    ]
    present = sum(1 for k in relevant_list if f[k])
    relevant = len(relevant_list)

    personalization = "Accurate" if ((f["has_income"] and f["has_debts"]) or (f["has_down"] and f["has_credit"])) else "Inaccurate"
    context_score   = _bucket_1to5(present, relevant)
    next_step       = "Present" if f["has_next"] else "Not Present"
    named = sum(int(f[k]) for k in ["has_rate","has_term","has_pmi","has_taxes","has_ins","has_hoa"])
    assumption_listing = "True" if (f["mentions_assume"] or named >= 3) else "False"
    trust_pts = 1 + (1 if f["mentions_assume"] else 0) + (1 if named >= 3 else 0)
    assumption_trust = str(max(1, min(5, trust_pts)))
    calc_acc = "True"  # ❌ THIS WAS THE PROBLEM!
    core_mech = sum(int(f[k]) for k in ["has_dti","has_piti","has_pmi","has_rate"])
    faithful = "True" if core_mech >= 1 else "False"
    overall  = "True" if (personalization=="Accurate" and faithful=="True" and int(context_score) >= 3) else "False"
    text_len = len(_norm(candidate).split())
    bullets  = ("\n-" in candidate) or ("\n*" in candidate) or ("1." in candidate and "\n" in candidate)
    separators = (":" in candidate) or ("—" in candidate) or ("–" in candidate)
    struc_pts = (1 if bullets else 0) + (1 if separators else 0) + (1 if text_len > 120 else 0)
    structured = str(min(5, max(1, 2 + struc_pts)))  # ❌ ARTIFICIAL MINIMUM!
    coherence = "True" if (("because" in _norm(candidate)) or ("so that" in _norm(candidate)) or core_mech >= 2) else "False"
    completeness = _bucket_1to5(present, relevant)
    fh = "False" if f["hits_prot"] else "True"

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
```

**REPLACE WITH:**
```python
def deterministic_scores_FIXED(candidate: str, user_profile: dict, question: str = "", golden_dict: dict = None) -> Dict[str, str]:
    """COMPLETELY FIXED deterministic scoring system"""
    
    # Calculate individual scores
    scores = {}
    scores["Personalization Accuracy"] = score_personalization_accuracy(candidate, user_profile)
    scores["Context based Personalization"] = score_context_personalization(candidate, user_profile)
    scores["Next Step Identification"] = "Present" if _features_improved(candidate, user_profile)["has_next"] else "Not Present"
    scores["Assumption Listing"] = score_assumption_listing(candidate, user_profile)
    scores["Assumption Trust"] = score_assumption_trust(candidate, user_profile)
    scores["Calculation Accuracy"] = score_calculation_accuracy(candidate, user_profile)  # ✅ FIXED!
    scores["Faithfulness to Ground Truth"] = score_faithfulness(candidate, golden_dict or {}, question)
    scores["Structured Presentation"] = score_structured_presentation(candidate, user_profile)  # ✅ FIXED!
    scores["Coherence"] = score_coherence(candidate, user_profile)
    scores["Completeness"] = score_completeness(candidate, user_profile, question)
    scores["Fair Housing Classifier"] = score_fair_housing(candidate, user_profile)
    
    # Overall accuracy depends on others (calculate last)
    scores["Overall Accuracy"] = score_overall_accuracy(candidate, user_profile, scores)
    
    return scores
```

### **STEP 5: Add All Supporting Functions**
**Location:** Before the main scoring function

**ADD THESE FUNCTIONS:**
Copy from `fixed_scoring_system.py`:
- `score_personalization_accuracy()`
- `score_context_personalization()`
- `score_calculation_accuracy()` 
- `score_assumption_listing()`
- `score_assumption_trust()`
- `score_faithfulness()`
- `score_overall_accuracy()`
- `score_structured_presentation()`
- `score_coherence()`
- `score_completeness()`
- `score_fair_housing()`

### **STEP 6: Update Function Call**
**Location:** Around lines 420-430 in your original code

**FIND THIS CODE:**
```python
def _apply_deterministic_scores_to_rows(rows: List[Dict[str,str]], cand: str, prof: dict) -> List[Dict[str,str]]:
    fixed = deterministic_scores(cand, prof)  # ❌ OLD FUNCTION
    out = []
    for r in rows:
        m = r.get("metric")
        if m in fixed:
            r["score"] = fixed[m]
        out.append(r)
    return out
```

**REPLACE WITH:**
```python
def _apply_deterministic_scores_to_rows(rows: List[Dict[str,str]], cand: str, prof: dict, question: str = "", golden: dict = None) -> List[Dict[str,str]]:
    fixed = deterministic_scores_FIXED(cand, prof, question, golden)  # ✅ NEW FUNCTION
    out = []
    for r in rows:
        m = r.get("metric")
        if m in fixed:
            r["score"] = fixed[m]
        out.append(r)
    return out
```

### **STEP 7: Update Main Evaluator Call**
**Location:** Around lines 500+ in your original code

**FIND THIS CODE:**
```python
# --- Deterministic scores override (stabilize Alpha) ---
rows = _apply_deterministic_scores_to_rows(rows, candidate_answer, user_profile)
```

**REPLACE WITH:**
```python
# --- Deterministic scores override (stabilize Alpha) ---
rows = _apply_deterministic_scores_to_rows(rows, candidate_answer, user_profile, question, golden_dict)
```

## 🔧 **Key Fixes Applied**

1. ✅ **Calculation Accuracy**: Now validates actual math instead of hardcoded `"True"`
2. ✅ **Structured Presentation**: Removed artificial minimum score of 2
3. ✅ **Stricter Thresholds**: 90%+ needed for perfect scores instead of 80%
4. ✅ **Better Feature Detection**: Requires actual numbers/values, not just keywords
5. ✅ **Number Validation**: Checks if profile numbers are used correctly
6. ✅ **Enhanced Fair Housing**: More comprehensive bias detection
7. ✅ **Improved Coherence**: Checks for contradictions and logical flow

## 📊 **Expected Results**

After applying these changes:
- **Scores will be 20-40 points lower** (more realistic)
- **Better discrimination** between good and poor responses
- **Actual validation** instead of keyword counting
- **Harder to game** the system with keyword stuffing

## ⚠️ **Important Notes**

1. **Test thoroughly** - The new system is much stricter
2. **Update golden answers** if needed to match stricter criteria
3. **Consider gradual rollout** - Scores will drop significantly
4. **Monitor business impact** - Users may see lower scores initially

## 🚀 **Quick Implementation**

1. **Copy `fixed_scoring_system.py`** to your project
2. **Replace the 7 sections** above in your original code
3. **Test with your examples**
4. **Adjust thresholds** if needed for your specific use case