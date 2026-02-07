# LLM Judge Determinism Analysis

## Executive Summary

**Answer: This code is PARTIALLY deterministic.**

The implementation uses a hybrid approach where some components are deterministic while others rely on LLM calls that introduce non-deterministic behavior.

## Deterministic Components

### 1. **Deterministic Scoring System** ✅ FULLY DETERMINISTIC
```python
def deterministic_scores(candidate: str, user_profile: dict) -> Dict[str,str]:
```

This function provides rule-based, deterministic scoring for all 12 metrics:
- Uses regex patterns and keyword matching
- Applies mathematical formulas (e.g., percentage thresholds for 1-5 scoring)
- No randomness involved - same input always produces same output

**Key deterministic rules:**
- Personalization Accuracy: Based on presence of income+debts OR down+credit
- Context-based scoring: Uses `_bucket_1to5()` with fixed percentage thresholds
- Fair Housing: Regex pattern matching for protected class mentions
- Structured Presentation: Count-based scoring (bullets, separators, text length)

### 2. **Alpha Calculation** ✅ FULLY DETERMINISTIC
```python
def _alpha_from_table_strict(table_markdown: str) -> int:
```
- Fixed mathematical formula
- Excludes "Structured Presentation" and "Completeness" from average
- Consistent score mapping: Accurate/True/Present→1, Inaccurate/False/Not Present→0
- Same table always produces same alpha score

### 3. **Feature Extraction** ✅ FULLY DETERMINISTIC
```python
def _features(candidate: str, user_profile: dict) -> Dict[str, bool]:
```
- Rule-based keyword detection
- Regex pattern matching
- Boolean logic operations

## Non-Deterministic Components

### 1. **LLM-Generated Justifications** ❌ NON-DETERMINISTIC
```python
def _build_prompt_full_gt_messages_STRICT()
def _backfill_chunk_JSON()
def _backfill_single_metric()
```

**Issues:**
- LLM calls for generating 50-90 word justifications
- Temperature/sampling parameters not explicitly set to 0
- Different justification text on each run, even with same scores
- Backfill system makes additional LLM calls when justifications are too short

### 2. **Score Repair System** ❌ NON-DETERMINISTIC
```python
def _repair_scores_with_model()
```
- Uses LLM to fix invalid scores
- Could potentially change scores non-deterministically
- However, this is mitigated by the deterministic override system

### 3. **Scratchpad Generation** ❌ NON-DETERMINISTIC
```python
def _backfill_scratchpad_if_empty()
```
- LLM-generated reasoning explanations
- Varies between runs

## Critical Design Pattern: **DETERMINISTIC OVERRIDE**

The code implements a crucial pattern that ensures score determinism:

```python
# Primary judge pass (full GT in messages) - NON-DETERMINISTIC
chat = client.chat.completions.create(...)

# --- Deterministic scores override (stabilize Alpha) ---
rows = _apply_deterministic_scores_to_rows(rows, candidate_answer, user_profile)
```

**This means:**
1. LLM generates initial scores (non-deterministic)
2. **Deterministic function completely overwrites all scores**
3. Alpha calculation uses only the deterministic scores
4. Final evaluation score is deterministic

## Configuration Impact on Determinism

### Environment Variables
```python
os.environ['BACKFILL_MAX_WORKERS'] = '1'  # single-thread for consistency
```
- Single-threading reduces some variability
- But doesn't eliminate LLM non-determinism

### Missing Deterministic Controls
The code does **NOT** set:
- `temperature=0` in LLM calls
- `seed` parameter for reproducibility
- `top_p=1` or other sampling controls

## Determinism Assessment by Component

| Component | Determinism Level | Impact on Final Score |
|-----------|------------------|----------------------|
| **Metric Scores** | ✅ FULLY DETERMINISTIC | ✅ **HIGH** - Determines alpha |
| **Alpha Calculation** | ✅ FULLY DETERMINISTIC | ✅ **HIGH** - Final evaluation score |
| **Justification Text** | ❌ NON-DETERMINISTIC | ⚠️ **LOW** - Cosmetic only |
| **Scratchpad** | ❌ NON-DETERMINISTIC | ⚠️ **LOW** - Explanatory only |
| **Score Repair** | ❌ NON-DETERMINISTIC | ⚠️ **VERY LOW** - Overridden by deterministic scores |

## Recommendations for Full Determinism

### 1. **Add LLM Deterministic Parameters**
```python
chat = client.chat.completions.create(
    model=model,
    messages=messages,
    max_completion_tokens=mct,
    temperature=0,        # ADD THIS
    seed=42,             # ADD THIS
    top_p=1              # ADD THIS
)
```

### 2. **Pre-generate Templates for Justifications**
Replace LLM-generated justifications with deterministic templates:
```python
def generate_deterministic_justification(metric: str, score: str, features: dict) -> str:
    templates = {
        "Personalization Accuracy": {
            "Accurate": "The response demonstrates accurate personalization by incorporating {income_debt_factors}...",
            "Inaccurate": "The response lacks personalization as it fails to reference {missing_factors}..."
        }
    }
    return templates[metric][score].format(**features)
```

### 3. **Remove Score Repair System**
Since deterministic scores are applied anyway, the repair system adds unnecessary non-determinism.

## Conclusion

**Current State:** The code achieves **deterministic evaluation scores** (the most important aspect) while having non-deterministic explanatory text.

**For Evaluation Purposes:** This is acceptable since the alpha score and metric scores are deterministic.

**For Full Reproducibility:** Additional changes needed to make justifications and explanations deterministic.

**Primary Achievement:** The hybrid design successfully provides stable, reproducible evaluation scores while maintaining human-readable explanations (even if those explanations vary between runs).