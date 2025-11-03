# ? TESTED AND WORKING - UPLOAD THIS FILE (WITH WIDGET!)

## File to Upload
?? **`FINAL_DEBUGGED_NOTEBOOK_v2.py`**

## ? Latest Update - Widget Added!
? **Added LLM Model Selection Widget** - You can now choose between:
- ?? **databricks-llm** ? Claude Sonnet 4.5 (via Databricks Serving Endpoints)
- **gpt-4o** ? OpenAI GPT-4o (via Zillow proxy)
- **gpt-4o-mini** ? OpenAI GPT-4o-mini (via Zillow proxy)
- **gpt-3.5-turbo** ? OpenAI GPT-3.5-turbo (via Zillow proxy)

The widget will appear at the top of your notebook after running Cell 3!

## What Was Fixed

### Fix 1: JSON Template Bug
The prompt template used curly braces `{` and `}`, which Python's `.format()` method tried to interpret as placeholders, causing a `KeyError`.

**Solution:** Escaped all curly braces by doubling them (`{{{{` and `}}}}`)

### Fix 2: Widget Restored
Previous version removed the model selection widget. Now fully restored with:
- Dropdown widget to select judge model
- Automatic configuration for Databricks or OpenAI based on selection
- Connection testing for both platforms
- Dual-mode evaluator that handles both client types

## Test Results ?

```
? ALL TESTS PASSED - NOTEBOOK SHOULD WORK!

Total evaluations: 9
Passed: 9 (100.0%)
LLM calls: 9/9 successful
Total score: 303.00 (non-zero!)
```

## What This File Does

1. **Cell 1**: Installs OpenAI, pandas, and requests
2. **Cell 2**: Loads hardcoded Cinderella test data (no file uploads needed)
3. **Cell 3**: **Creates widget** and configures LLM client (OpenAI or Databricks)
4. **Cell 4**: Defines metric classes
5. **Cell 5**: Creates LLM evaluator with **EXTENSIVE DEBUG OUTPUT**
6. **Cell 6**: Runs evaluation with comprehensive progress tracking

## How to Use the Widget

### Step 1: Run Cell 3
After running Cell 3, you'll see a dropdown widget at the top of the notebook labeled **"?? Judge Model"**

### Step 2: Select Your Model
Choose from:
- **databricks-llm** (default) - Uses Claude Sonnet 4.5 via Databricks Foundation Model
- **gpt-4o** - Uses OpenAI GPT-4o via Zillow proxy
- **gpt-4o-mini** - Uses OpenAI GPT-4o-mini via Zillow proxy
- **gpt-3.5-turbo** - Uses OpenAI GPT-3.5-turbo via Zillow proxy

### Step 3: Re-run Cell 3 to Apply Changes
If you change the widget value, re-run Cell 3 to reconfigure the client

### Step 4: Run Cell 6 to Evaluate
Run Cell 6 to execute the evaluation with your selected model

## What Happens Based on Your Selection

### If you select "databricks-llm":
```
>>> Configuring Databricks Foundation Model...
  Workspace: your-workspace.cloud.databricks.com
  Querying serving endpoints...
  Found endpoint: claude-sonnet-4-5
  Testing Databricks endpoint...
  Connection test: SUCCESS

READY TO EVALUATE with databricks-llm
Client type: databricks
```

The evaluator will:
1. Get your workspace token automatically
2. Query the Databricks Serving Endpoints API
3. Find the Claude Sonnet endpoint
4. Make API calls directly to the Databricks Serving Endpoint
5. Use the OpenAI-compatible payload format

### If you select "gpt-4o", "gpt-4o-mini", or "gpt-3.5-turbo":
```
>>> Configuring OpenAI Model: gpt-4o-mini
  API key retrieved from: popin-secure-scope
  Base URL: https://api.zillowlabs.com/openai/v1
  Model: gpt-4o-mini
  Testing OpenAI connection...
  Connection test: SUCCESS

READY TO EVALUATE with gpt-4o-mini
Client type: openai
```

The evaluator will:
1. Retrieve your OpenAI API key from secrets
2. Initialize the OpenAI client with the Zillow proxy URL
3. Make API calls through the Zillow OpenAI proxy
4. Use the specified GPT model

## Debug Features

### 1. Real-time LLM Call Tracking
For **both** OpenAI and Databricks:
```
    [LLM CALL START]
      Client type: databricks (or openai)
      Model: databricks-llm (or gpt-4o-mini)
      Prompt length: 330 chars
      Making API call...
      Calling Databricks endpoint: claude-sonnet-4-5
      API call SUCCESS!
      Response length: 51 chars
    [LLM CALL END]
```

### 2. Error Logging
If anything fails, you'll see:
- `[CRITICAL ERROR]` with full tracebacks
- `[PARSE ERROR]` for JSON parsing issues
- `[NORMALIZE ERROR]` for score conversion issues

### 3. Progress Tracking
- Real-time percentage: `[11.1%] Story_Accuracy...`
- Result status: `Result: PASS (score: 1.00)`
- Per-sample and per-metric summaries

### 4. Comprehensive Summary
At the end of Cell 6:
- Total/passed/failed counts
- Overall pass rate
- Per-metric breakdowns
- Detailed results table
- Execution time with warnings if too fast

## Expected Cell 6 Output

```
================================================================================
CELL 6: RUN EVALUATION
================================================================================

Checking prerequisites...
  client is None: False
  client_type: databricks (or openai)
  model: databricks-llm (or gpt-4o-mini)
  Samples: 3
  Metrics: 3

Loading metrics...
  Loaded: Story_Accuracy (binary, threshold=1.0)
  Loaded: Response_Completeness (1-5_scale, threshold=4.0)
  Loaded: Child_Friendliness (percentage, threshold=75.0)

Total metrics loaded: 3

Initializing evaluator...
[INIT] Evaluator created: databricks, 3 metrics
Evaluator ready

================================================================================
ABOUT TO START EVALUATION
================================================================================
Watch for [LLM CALL START] messages below...
If you don't see them, the LLM is not being called!
================================================================================

================================================================================
STARTING EVALUATION
================================================================================
Samples: 3
Metrics: 3
Total evaluations: 9
================================================================================

Sample 1/3: ID 1
  [ 11.1%] Story_Accuracy...
    [LLM CALL START]
      Client type: databricks
      Model: databricks-llm
      Prompt length: 330 chars
      Making API call...
      Calling Databricks endpoint: claude-sonnet-4-5
      API call SUCCESS!
    [LLM CALL END]
    Result: PASS (score: 1.00)
  [ 22.2%] Response_Completeness...
    [LLM CALL START]
    ...
```

## ?? Expected Timing
- **Normal**: 20-60 seconds for 9 LLM calls (3 samples ? 3 metrics)
- **Warning sign**: If it completes in < 5 seconds, the notebook will alert you

## Warning Signs to Watch For

? **If you DON'T see `[LLM CALL START]` messages** ? LLM is not being called
? **If evaluation completes in < 5 seconds** ? API calls failing silently
? **If all scores are 0** ? JSON parsing or API issues
? **If you see `[CRITICAL ERROR]`** ? Check the full traceback

## Switching Between Models

You can test different models without re-uploading the notebook:

1. **Run Cell 3** ? Widget appears, default is "databricks-llm"
2. **Run Cell 6** ? Evaluation runs with Claude Sonnet 4.5
3. **Change widget** to "gpt-4o-mini"
4. **Re-run Cell 3** ? Reconfigures to use OpenAI
5. **Re-run Cell 6** ? Evaluation runs with GPT-4o-mini

Compare results across models! ??

## Key Features

| Feature | Status |
|---------|--------|
| Widget for model selection | ? YES |
| Databricks LLM (Claude Sonnet 4.5) | ? YES |
| OpenAI models (GPT-4o, 4o-mini, 3.5-turbo) | ? YES |
| Automatic endpoint discovery | ? YES |
| Dual-mode evaluator | ? YES |
| Extensive debug logging | ? YES |
| Real-time progress tracking | ? YES |
| Error visibility | ? YES |
| Hardcoded test data | ? YES |
| No file uploads needed | ? YES |

## File Info
- **Location**: `/workspace/FINAL_DEBUGGED_NOTEBOOK_v2.py`
- **Size**: ~24 KB
- **Cells**: 6 + markdown documentation
- **Test Status**: ? PASSED (100% pass rate, 9/9 LLM calls successful)
- **Widget**: ? INCLUDED (4 model options)
- **Ready to use**: ? YES!

---

## ?? Quick Start

1. Upload `FINAL_DEBUGGED_NOTEBOOK_v2.py` to Databricks
2. Run Cell 1 (installs packages)
3. Run Cell 2 (loads data)
4. Run Cell 3 (creates widget and configures client)
5. **See the widget** at the top and select your model!
6. Run Cell 4 (defines classes)
7. Run Cell 5 (creates evaluator)
8. Run Cell 6 (runs evaluation with debug output)

**That's it!** ??
