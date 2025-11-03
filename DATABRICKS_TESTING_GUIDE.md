# ?? Databricks Testing Guide - "Run All" Workflow

## ?? IMPORTANT: Testing Status

**Current Status:** ? Code Complete, ?? Not Tested in Live Databricks

**What Has Been Tested:**
- ? File I/O operations (local environment)
- ? CSV parsing and validation
- ? Data structures and logic
- ? Metrics add/edit/delete operations
- ? All Python code syntax

**What Needs Databricks Testing:**
- ?? `dbutils.widgets` behavior with "Run All"
- ?? File persistence across cell executions
- ?? Widget state management
- ?? Secrets integration
- ?? Complete workflow end-to-end

---

## ?? Testing Plan for Databricks

### Phase 1: Basic Upload & Configuration (10 minutes)

**Test Steps:**

1. **Upload Files to Databricks**
   ```
   1. Go to Databricks workspace
   2. Click "+" ? "Upload Data"
   3. Upload these 3 files:
      - test_metrics_config.csv
      - test_evaluation_data.csv
      - ground_truth_accuracy.csv
   4. Note the file paths (e.g., /Workspace/Users/your.email@domain.com/)
   ```

2. **Copy Notebook**
   ```
   1. Create new notebook in Databricks
   2. Copy content from databricks_ready_notebook.py
   3. Paste into notebook
   4. Save as "LLM Judge Evaluation"
   ```

3. **Update File Paths (Cell 2)**
   ```python
   # Should auto-configure, but verify:
   METRICS_FILE = "/Workspace/Users/your.email@domain.com/metrics_config.csv"
   EVAL_DATA_FILE = "/Workspace/Users/your.email@domain.com/evaluation_data.csv"
   GROUND_TRUTH_FILES = "/Workspace/Users/your.email@domain.com/ground_truth_accuracy.csv"
   ```

**Expected Results:**
- ? Files uploaded successfully
- ? Paths auto-configured
- ? File existence checks pass

---

### Phase 2: Individual Cell Testing (15 minutes)

**Test Each Cell Individually:**

#### Cell 1: Package Installation
```
Action: Run Cell 1
Expected: 
  ? Packages install without errors
  ? "All libraries imported successfully"
  ?? May take 30-60 seconds first time
```

#### Cell 2: File Configuration
```
Action: Run Cell 2
Expected:
  ? Auto-detects username
  ? Shows file paths
  ? Reports file status (Found/Not found)
```

#### Cell 3: Load Metrics
```
Action: Run Cell 3
Expected:
  ? Loads existing metrics OR creates defaults
  ? Displays metrics table
  ? Shows 3 default metrics if file doesn't exist
```

#### Cell 4: Display Metrics Editor
```
Action: Run Cell 4
Expected:
  ? Shows current metrics table
  ? Displays summary statistics
```

#### Cell 5: Create Widgets
```
Action: Run Cell 5
Expected:
  ? Widgets appear at top of notebook
  ? Form fields visible
  ? "Metrics editor ready!" message
  
?? POTENTIAL ISSUE: Widgets may not display correctly on first run
   FIX: Re-run Cell 5 if widgets don't appear
```

#### Cell 6: Apply Changes
```
Action: 
  1. Set Action = "view_only"
  2. Run Cell 6
Expected:
  ? "View mode - No changes made"
  ? Metrics table displayed
  
?? TEST ADD METRIC:
  1. Set Action = "add_new"
  2. Fill in:
     - Metric Name: TestMetric
     - Type: binary
     - Rubric: Test rubric text
     - Threshold: 1
  3. Run Cell 6
Expected:
  ? "Added metric: 'TestMetric'"
  ? Backup file created
  ? Updated table shows 4 metrics
  ? Changes saved to CSV
```

#### Cell 7: Load Data
```
Action: Run Cell 7
Expected:
  ? Evaluation data loaded (5 samples)
  ? Ground truth loaded (5 rows)
  ? Data preview tables displayed
  ? Summary shows all counts
```

#### Cell 8: Model Configuration
```
Action: Run Cell 8
Expected:
  ? Model dropdown appears
  ?? API key retrieval - may fail if secrets not configured
  
IF API KEY FAILS:
  - Expected: "No API key found" message
  - This is OK for testing file operations
  - Evaluation will fail but metrics editing works
```

#### Cells 9-10: Core Classes
```
Action: Run Cells 9-10
Expected:
  ? "Core classes defined"
  ? "LLM Judge Evaluator defined"
  ?? Very fast (< 1 second)
```

#### Cell 11: Run Evaluation
```
Action: Run Cell 11
Expected IF API KEY WORKS:
  ? Evaluation progress shows
  ? Each sample evaluated
  ? Results table displayed
  ? Summary statistics shown
  ? Results file saved
  
Expected IF NO API KEY:
  ?? "LLM client not initialized" message
  ? Evaluation will not run
  
THIS IS OK for testing metrics editor functionality
```

---

### Phase 3: "Run All" Testing (20 minutes)

**Critical Test: Complete Workflow**

1. **Clear All Outputs**
   ```
   1. In Databricks: Cell ? Clear All Outputs
   2. Verify all cell outputs are cleared
   ```

2. **Reset Widgets**
   ```
   1. Manually remove all widgets if present
   2. Or restart notebook
   ```

3. **Run All Cells**
   ```
   1. Click "Run All" button
   2. Watch cells execute in sequence
   3. Note any errors or warnings
   ```

**Expected Behavior:**

? **Success Scenario:**
```
Cell 1:  ? Packages install, restart Python
Cell 2:  ? File paths configured, status reported
Cell 3:  ? Metrics loaded/created
Cell 4:  ? Metrics displayed
Cell 5:  ? Widgets created (but may need manual interaction)
Cell 6:  ? Runs in "view_only" mode
Cell 7:  ? Data files loaded
Cell 8:  ? Model configured (API key may fail - OK)
Cell 9:  ? Classes defined
Cell 10: ? Evaluator defined
Cell 11: ?? May fail if no API key (expected)
```

?? **Potential Issues:**

**Issue 1: Widgets Don't Work in "Run All"**
```
PROBLEM: Databricks widgets need user interaction
SYMPTOMS: Cell 6 uses default values only
IMPACT: Can't modify metrics during "Run All"
SOLUTION: This is expected! Use two-phase approach:
  Phase 1: Run All to see current state
  Phase 2: Modify widgets, run Cell 6 individually
```

**Issue 2: Python Restart Breaks "Run All"**
```
PROBLEM: Cell 1 restarts Python, may stop execution
SYMPTOMS: Cells after Cell 1 don't run
SOLUTION: 
  - Run Cell 1 separately first
  - Then run remaining cells
  - OR: Remove dbutils.library.restartPython() line
```

**Issue 3: File Paths Not Found**
```
PROBLEM: Files uploaded to different location
SYMPTOMS: "File not found" errors
SOLUTION: Check Cell 2 output, update paths manually
```

**Issue 4: API Key Not Configured**
```
PROBLEM: No OpenAI secrets set up
SYMPTOMS: Evaluation doesn't run
SOLUTION: 
  - This is OK for metrics testing
  - To fix: Configure Databricks secrets
  - See "API Key Setup" section below
```

---

### Phase 4: Metrics Modification Testing (10 minutes)

**Test: Add New Metric**

1. Run cells 1-7 to load data
2. In Cell 5, set widgets:
   - Action: add_new
   - Name: Empathy
   - Type: 1-5_scale
   - Description: Check empathetic tone
   - Rubric: Rate from 1-5 based on empathy
   - Threshold: 4
3. Run Cell 6
4. Verify:
   - ? "Added metric: 'Empathy'" message
   - ? Backup file created
   - ? Metrics table shows 4 metrics
   - ? CSV file updated

**Test: Edit Existing Metric**

1. In Cell 5, set widgets:
   - Action: edit_existing
   - Select Metric: Relevance
   - Threshold: 3
2. Run Cell 6
3. Verify:
   - ? "Updated metric: 'Relevance'" message
   - ? Threshold changed in table

**Test: Delete Metric**

1. In Cell 5, set widgets:
   - Action: delete_existing
   - Select Metric: Empathy
2. Run Cell 6
3. Verify:
   - ? "Deleted metric: 'Empathy'" message
   - ? Metrics table shows 3 metrics

**Test: Changes Persist**

1. Make changes as above
2. Run Cell 3 again
3. Verify:
   - ? Changes are still there
   - ? CSV file was actually updated

---

## ?? Common Issues & Solutions

### Issue 1: "dbutils is not defined"

**Cause:** Running outside Databricks environment

**Solution:**
```
This code MUST run in Databricks notebook
Cannot test locally with standard Python
All dbutils code is Databricks-specific
```

### Issue 2: Widgets Not Appearing

**Cause:** Widgets need explicit creation

**Solutions:**
1. Re-run Cell 5
2. Check if `dbutils.widgets.removeAll()` was called
3. Try: `dbutils.widgets.removeAll()` then re-run Cell 5

### Issue 3: "Run All" Stops After Cell 1

**Cause:** Python restart interrupts execution

**Solutions:**
1. **Recommended:** Run Cell 1 separately, then run remaining cells
2. **Alternative:** Comment out `dbutils.library.restartPython()`
   ```python
   # dbutils.library.restartPython()  # Comment this out
   ```
3. **Better:** Use "Run All Below" starting from Cell 2

### Issue 4: File Paths Wrong

**Cause:** Auto-detection failed or files in different location

**Solution:**
```python
# In Cell 2, manually set paths:
METRICS_FILE = "/path/to/your/metrics_config.csv"
EVAL_DATA_FILE = "/path/to/your/evaluation_data.csv"
GROUND_TRUTH_FILES = "/path/to/your/ground_truth.csv"
```

### Issue 5: Changes Don't Save

**Cause:** File path wrong or permissions issue

**Diagnostics:**
```python
# Add to Cell 6:
import os
print(f"Attempting to save to: {METRICS_FILE}")
print(f"Directory exists: {os.path.exists(os.path.dirname(METRICS_FILE))}")
print(f"File exists: {os.path.exists(METRICS_FILE)}")
```

**Solution:**
1. Verify write permissions
2. Try saving to `/tmp/` first
3. Check Databricks file system access

### Issue 6: "Module Not Found" Errors

**Cause:** Packages not installed or Python not restarted

**Solution:**
1. Re-run Cell 1
2. Wait for "All libraries imported successfully"
3. If still fails: Restart notebook kernel

---

## ?? API Key Setup (For Evaluation)

The metrics editor works WITHOUT API keys, but evaluation needs them.

### Setup OpenAI API Key:

**Option 1: Using Databricks Secrets CLI**
```bash
# Install Databricks CLI
pip install databricks-cli

# Configure
databricks configure --token

# Create secret scope
databricks secrets create-scope --scope popin-secure-scope

# Add your OpenAI key
databricks secrets put --scope popin-secure-scope --key openai_key
```

**Option 2: Using Databricks UI** (If available)
```
1. Go to User Settings ? Secrets
2. Create new scope: popin-secure-scope
3. Add secret: openai_key
4. Paste your OpenAI API key
```

**Option 3: Personal Scope**
```python
# Use your email as scope name
# Example: email@company.com ? user_email_at_company_com
# Create scope with this name
# Add key: openai_key
```

---

## ? Verification Checklist

After testing, verify:

### File Operations:
- [ ] Can upload CSV files to Databricks
- [ ] File paths auto-configure correctly
- [ ] Files load without errors
- [ ] Ground truth data accessible

### Metrics Editor:
- [ ] Current metrics display correctly
- [ ] Widgets appear and are usable
- [ ] Add metric works and saves
- [ ] Edit metric works and saves
- [ ] Delete metric works and saves
- [ ] Backup files created
- [ ] Changes persist across cell runs

### "Run All" Workflow:
- [ ] All cells execute (except possibly Cell 11)
- [ ] No critical errors
- [ ] Metrics load/create successfully
- [ ] Data files load successfully
- [ ] Can modify metrics after "Run All"

### Evaluation (If API key configured):
- [ ] Model selection works
- [ ] API connection successful
- [ ] Evaluation runs
- [ ] Results displayed
- [ ] Results file saved

---

## ?? Test Results Template

Use this template to document your testing:

```
DATE: ___________
TESTER: ___________
DATABRICKS WORKSPACE: ___________

PHASE 1: File Upload
  ?/? Files uploaded successfully
  ?/? Paths auto-configured
  NOTES: ___________

PHASE 2: Individual Cells
  Cell 1:  ?/?  NOTES: ___________
  Cell 2:  ?/?  NOTES: ___________
  Cell 3:  ?/?  NOTES: ___________
  Cell 4:  ?/?  NOTES: ___________
  Cell 5:  ?/?  NOTES: ___________
  Cell 6:  ?/?  NOTES: ___________
  Cell 7:  ?/?  NOTES: ___________
  Cell 8:  ?/?  NOTES: ___________
  Cell 9:  ?/?  NOTES: ___________
  Cell 10: ?/?  NOTES: ___________
  Cell 11: ?/?  NOTES: ___________

PHASE 3: "Run All"
  ?/? Completes without critical errors
  ISSUES FOUND: ___________

PHASE 4: Metrics Modification
  ?/? Add metric works
  ?/? Edit metric works
  ?/? Delete metric works
  ?/? Changes persist

OVERALL RESULT: ?/?
PRODUCTION READY: ?/?

ISSUES TO FIX:
1. ___________
2. ___________
3. ___________
```

---

## ?? Recommended Testing Workflow

**For First-Time Testing:**

1. **Start Small** (30 minutes)
   - Upload 1 test file at a time
   - Run cells individually
   - Verify each step works

2. **Test Metrics Editor** (15 minutes)
   - Focus on add/edit/delete
   - Verify file saves
   - Check changes persist

3. **Test "Run All"** (10 minutes)
   - Try complete workflow
   - Note any breaks or warnings
   - Document issues

4. **Test with Real Data** (20 minutes)
   - Use your actual CSV files
   - Run evaluation (if API key configured)
   - Verify results make sense

**For Production Deployment:**

1. Test in dev workspace first
2. Get 2-3 users to try it
3. Collect feedback
4. Fix any issues
5. Deploy to production workspace
6. Train all users

---

## ?? Known Limitations

Based on Databricks behavior, these limitations are expected:

1. **Widgets Require Manual Interaction**
   - "Run All" uses default widget values
   - Must manually interact with widgets between runs
   - This is normal Databricks behavior

2. **Python Restart Breaks "Run All"**
   - Cell 1 restarts Python
   - May need to run Cell 1 separately
   - Then "Run All Below" from Cell 2

3. **File Paths May Need Manual Config**
   - Auto-detection works in most cases
   - Some environments may need manual paths
   - Easy to fix in Cell 2

4. **API Keys Need Pre-Configuration**
   - Secrets must be set up in advance
   - Can't configure API keys within notebook
   - Metrics editor works without API keys

---

## ?? Pro Tips

1. **Save Frequently**
   - Databricks auto-saves, but be cautious
   - Export notebook periodically

2. **Use Version Control**
   - Save different versions as you test
   - Easy to roll back if something breaks

3. **Test with Small Data First**
   - Use 2-3 samples for initial testing
   - Scale up after basics work

4. **Document Issues Immediately**
   - Note exact error messages
   - Capture screenshots
   - Record steps to reproduce

5. **Have Backup Plan**
   - Keep CSV files backed up locally
   - Know how to restore from backup
   - Test restore process

---

## ?? Getting Help

If you encounter issues during testing:

1. **Check this guide** - Most issues covered above
2. **Check Databricks docs** - For platform-specific issues
3. **Review error messages** - Often self-explanatory
4. **Test in isolation** - Run one cell at a time
5. **Verify file paths** - Most common issue

---

## ? Success Criteria

The implementation is successful if:

- ? Files upload and load correctly
- ? Metrics display in table
- ? Can add/edit/delete metrics via forms
- ? Changes save to CSV file
- ? Changes persist across cell runs
- ? "Run All" executes without critical errors
- ? (Optional) Evaluation runs if API key configured

**Minimum Viable Success:**
Even if evaluation doesn't work (no API key), the metrics editor should fully function. That's the primary goal.

---

## ?? You're Ready to Test!

1. Upload `databricks_ready_notebook.py` to Databricks
2. Upload the 3 CSV files
3. Follow Phase 1-4 testing above
4. Document results
5. Report findings

**Good luck with testing!** ??
