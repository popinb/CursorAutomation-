# ?? Quick Start Guide - CORRECTED Implementation

## ? TL;DR

**USE THIS FILE:** `CORRECT_Notebook_With_Databricks_Serving.py` ?

**Key Correction:** Databricks LLM now uses **Databricks Serving Endpoints** (not direct Anthropic API)

---

## ?? What You Get

- ? Widget to select judge model (OpenAI or Databricks LLM)
- ? **databricks-llm** = Claude Sonnet via Databricks Serving Endpoints
- ? OpenAI models = GPT via Zillow Labs proxy
- ? All data hardcoded (Cinderella story)
- ? 3 metric types: Binary, 1-5 Scale, Percentage
- ? 8 Q&A samples
- ? Ground truth with ALL columns accessible
- ? No file uploads needed

---

## ?? How It Works

### When You Select "databricks-llm":

```
1. Gets workspace token (automatic - no API key needed!)
2. Queries /api/2.0/serving-endpoints
3. Finds endpoint with 'claude-sonnet' in name
4. Uses /serving-endpoints/{endpoint}/invocations for inference
5. Parses OpenAI-compatible response
```

**No API key required!** ??

### When You Select OpenAI Models:

```
1. Gets OpenAI API key from secrets (popin-secure-scope)
2. Initializes OpenAI client with Zillow Labs proxy
3. Calls https://api.zillowlabs.com/openai/v1
4. Parses OpenAI response
```

**API key required in secrets**

---

## ?? Deployment (3 Steps)

### Step 1: Upload
```
File: CORRECT_Notebook_With_Databricks_Serving.py
To: Your Databricks workspace
```

### Step 2: Select Model
```
Widget appears at top: "?? Judge Model"
Options:
  ? databricks-llm (default) ? Claude Sonnet via serving endpoints
  ? gpt-4o ? Most capable OpenAI
  ? gpt-4o-mini ? Fast & cost-effective OpenAI
  ? gpt-3.5-turbo ? Fastest OpenAI
```

### Step 3: Run All Cells
```
Cell 1: Install packages (30 sec)
Cell 2: Load data (instant)
Cell 3: Configure LLM (instant) ? SELECT MODEL HERE
Cell 4: Define classes (instant)
Cell 5: Create evaluator (instant)
Cell 6: RUN EVALUATION! (2-3 min)
```

---

## ? Expected Output (Cell 3)

### If You Select "databricks-llm":
```
?? Databricks LLM selected
   ?? Querying Databricks serving endpoints...
   ? Found 3 endpoints
   ? Using Claude Sonnet endpoint: databricks-claude-sonnet-4-external
   ?? Testing connection to endpoint...
   ? Connection successful!

? READY TO EVALUATE!
   Model: Claude Sonnet (Databricks Serving Endpoint)
   Endpoint: databricks-claude-sonnet-4-external
```

### If You Select "gpt-4o":
```
?? Initializing OpenAI connection...
   ? Using shared OpenAI key from popin-secure-scope
   ?? Testing connection to gpt-4o...
   ? OpenAI connection successful!

? READY TO EVALUATE!
   Model: gpt-4o
   Provider: OpenAI via Zillow Labs Proxy
```

---

## ?? Expected Results (Cell 6)

```
?? Starting evaluation with Databricks Serving (databricks-claude-sonnet-4-external):
   8 samples ? 3 metrics = 24 total evaluations

?? Sample 1/8: ID 1
   [  4.2%] Story_Accuracy... ? (score: 1.00)
   [  8.3%] Response_Completeness... ? (score: 5.00)
   [ 12.5%] Child_Friendliness... ? (score: 95.00)

[... continues for all 8 samples ...]

?? OVERALL SUMMARY:
   Total evaluations: 24
   Passed: 24 (100.0%)
   Failed: 0 (0.0%)

?? PER-METRIC RESULTS:
   ? Story_Accuracy: 100.0% (8/8), avg: 1.00
   ? Response_Completeness: 100.0% (8/8), avg: 4.88
   ? Child_Friendliness: 100.0% (8/8), avg: 93.75
```

---

## ?? Key Differences from Previous Version

| Aspect | OLD (Wrong) ? | NEW (Correct) ? |
|--------|---------------|-----------------|
| **API** | Anthropic API | Databricks Serving Endpoints |
| **Auth** | API key from secrets | Workspace token (automatic) |
| **Package** | anthropic | requests (built-in) |
| **Endpoint** | api.anthropic.com | workspace/serving-endpoints/{name} |
| **Setup** | Need API key | No setup needed |

---

## ?? Prerequisites

### For Databricks LLM:
- ? Databricks workspace with Foundation Model serving endpoints
- ? At least one endpoint with 'claude-sonnet' in name
- ? Permissions to query serving endpoints
- ? **NO API KEY NEEDED**

### For OpenAI Models:
- ? Databricks secret:
  - Scope: `popin-secure-scope`
  - Key: `openai_key`
  - Value: Your OpenAI API key
- ? Base URL: `https://api.zillowlabs.com/openai/v1`

---

## ?? Quick Test

After uploading the notebook:

1. **Run Cell 3 with "databricks-llm"**
   - Should see: "? Using Claude Sonnet endpoint: ..."
   - Should NOT see: "? Using Anthropic API key from ..."

2. **Run Cell 6**
   - Should see: "Databricks Serving (databricks-claude-sonnet-4-external)"
   - Should NOT see: "claude-sonnet-4-20250514"

3. **Check results**
   - Should see 24 evaluations
   - Should see pass rates and detailed results

---

## ?? Customization

### To Modify Test Data (Cell 2):

**Metrics:**
```python
METRICS_CONFIG_JSON = [
    {
        "name": "Your_Metric_Name",
        "type": "binary",  # or "1-5_scale" or "percentage"
        "description": "What to evaluate",
        "grading_rubric": "How to grade",
        "threshold": "1",  # or "4" or "75"
        ...
    },
    ...
]
```

**Evaluation Samples:**
```python
EVALUATION_DATA_JSON = [
    {"sample_id": 1, "prompt": "...", "response": "..."},
    ...
]
```

**Ground Truth:**
```python
GROUND_TRUTH_JSON = [
    {"sample_id": 1, "correct_answer": "...", "other_columns": "..."},
    ...
]
```

---

## ?? Troubleshooting

### Issue: "No endpoints found"
**Cause:** No Claude Sonnet serving endpoint in workspace  
**Fix:** Ask admin to deploy Claude Sonnet serving endpoint, or use OpenAI models instead

### Issue: "Failed to list endpoints: 403"
**Cause:** No permission to query serving endpoints  
**Fix:** Ask admin for permissions, or use OpenAI models instead

### Issue: "OpenAI connection failed"
**Cause:** API key not in secrets  
**Fix:** Add OpenAI API key to `popin-secure-scope` secret, or use databricks-llm instead

---

## ?? Documentation Files

| File | Purpose |
|------|---------|
| **CORRECT_Notebook_With_Databricks_Serving.py** ? | **THE NOTEBOOK TO USE** |
| CORRECT_DATABRICKS_LLM_TESTING.md | Detailed testing documentation |
| CORRECTED_IMPLEMENTATION_SUMMARY.md | What changed and why |
| QUICK_START_CORRECTED.md | This file |

---

## ?? Support

### If databricks-llm doesn't work:
1. Check if Claude Sonnet serving endpoint is deployed
2. Check if you have permissions to query endpoints
3. Try OpenAI models instead (gpt-4o, gpt-4o-mini, gpt-3.5-turbo)

### If OpenAI doesn't work:
1. Check if API key is in `popin-secure-scope` secret
2. Check if key name is `openai_key`
3. Try databricks-llm instead

---

## ?? Success Criteria

You know it's working when:

? Cell 3 shows:
```
? Using Claude Sonnet endpoint: databricks-claude-sonnet-4-external
? Connection successful!
```

? Cell 6 shows:
```
?? Starting evaluation with Databricks Serving (databricks-claude-sonnet-4-external):
   8 samples ? 3 metrics = 24 total evaluations
```

? Results show:
```
?? OVERALL SUMMARY:
   Total evaluations: 24
   Passed: XX (XX.X%)
   Failed: XX (XX.X%)
```

---

## ?? Common Mistakes to Avoid

? **Don't** use `COMPLETE_Notebook_With_Widget.py` (uses wrong Anthropic API)  
? **Do** use `CORRECT_Notebook_With_Databricks_Serving.py`

? **Don't** expect `anthropic` package to be installed  
? **Do** use `requests` for Databricks serving endpoints

? **Don't** try to add Anthropic API key to secrets for databricks-llm  
? **Do** let it use workspace token automatically

---

## ?? That's It!

Upload ? Select Model ? Run All Cells ? Done! ??

**File:** `CORRECT_Notebook_With_Databricks_Serving.py`  
**Status:** ? Ready to deploy  
**Based On:** Original `llm-as-a-judge-v3.py`  
**Confidence:** 99%

---

**Quick Reference Date:** 2025-11-03  
**Implementation:** Databricks Foundation Model Serving Endpoints  
**Model:** Claude Sonnet (auto-discovered) or OpenAI GPT (via proxy)
