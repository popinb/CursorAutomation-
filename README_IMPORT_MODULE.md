# How to Import and Use the Databricks Metrics Editor Module

## ?? Files You Need

1. **`databricks_metrics_editor.py`** - The main module (REQUIRED)
2. **`example_usage.py`** - Example notebook showing how to use it
3. **`test_metrics_editor.py`** - Test suite to verify it works
4. **`METRICS_EDITOR_README.md`** - Full documentation

## ?? Quick Setup (3 Steps)

### Step 1: Upload the module to Databricks

1. Go to your Databricks workspace
2. Navigate to `/Workspace/Users/your_email@company.com/`
3. Click "Upload" and upload `databricks_metrics_editor.py`

### Step 2: Import in your notebook

```python
# Databricks notebook source
from databricks_metrics_editor import MetricsEditor

# Initialize
editor = MetricsEditor()
```

### Step 3: Use it!

```python
# Load your data
editor.load_data(
    evaluation_data_path="evaluation_data.csv",
    metrics_config_path="metrics_config.csv",
    ground_truth_files="gt1.csv;gt2.csv"
)

# Show interactive editor
editor.show_editor()

# After editing in the UI above, apply changes
editor.apply_changes()

# Save to file
editor.save_metrics("metrics_edited.csv")
```

## ?? Simple Example

```python
# Databricks notebook source

# COMMAND ----------
# Import
from databricks_metrics_editor import MetricsEditor
editor = MetricsEditor()

# COMMAND ----------
# Load data (files should be in /Workspace/Users/your_email/)
editor.load_data(
    metrics_config_path="my_metrics.csv"
)
editor.display_summary()

# COMMAND ----------
# Show editor - non-technical users can edit here
editor.show_editor()

# COMMAND ----------
# Apply the changes
editor.apply_changes()
display(editor.metrics_config_data)

# COMMAND ----------
# Save edited file
editor.save_metrics("metrics_final.csv")
```

## ?? What Users See

When you call `editor.show_editor()`, users see:

```
???????????????????????????????????????????????
?  ?? Metrics Configuration Editor            ?
?                            ? Auto-Saved     ?
???????????????????????????????????????????????
?  ?? How to Use                              ?
?  1. Edit any cell - saves automatically     ?
?  2. Add or delete rows as needed            ?
?  3. Run editor.apply_changes() to load      ?
?  4. Run editor.save_metrics() to save       ?
???????????????????????????????????????????????
?  TOTAL METRICS: 5    COLUMNS: 4             ?
???????????????????????????????????????????????
?  Row ? metric_name ? type  ? weight ? [???] ?
?   0  ? [editable]  ? [...]  ? [...]  ?Delete?
?   1  ? [editable]  ? [...]  ? [...]  ?Delete?
???????????????????????????????????????????????
?  ? Add New Metric                          ?
?  [form fields...]                           ?
?  [? Add Metric button]                     ?
???????????????????????????????????????????????
```

## ?? API Quick Reference

| Method | Description | Example |
|--------|-------------|---------|
| `MetricsEditor()` | Initialize editor | `editor = MetricsEditor()` |
| `load_data()` | Load CSV files | `editor.load_data(metrics_config_path="file.csv")` |
| `show_editor()` | Display UI | `editor.show_editor()` |
| `apply_changes()` | Load edits | `editor.apply_changes()` |
| `save_metrics()` | Save to CSV | `editor.save_metrics("output.csv")` |
| `display_summary()` | Show summary | `editor.display_summary()` |

## ?? Troubleshooting

### "ModuleNotFoundError: No module named 'databricks_metrics_editor'"

**Solution:**
```python
# Make sure file is in your workspace, then:
import sys
sys.path.append('/Workspace/Users/your_email@company.com/')
from databricks_metrics_editor import MetricsEditor
```

### "File not found"

**Solution:** Upload your CSV files to `/Workspace/Users/your_email@company.com/`

Or use absolute paths:
```python
editor.load_data(
    metrics_config_path="/Workspace/Users/your_email@company.com/metrics.csv"
)
```

### "Cannot display HTML"

**Solution:** Make sure you're running in a Databricks notebook (not a Python script)

## ? Test It

Run the test file to verify everything works:

```python
# Upload test_metrics_editor.py and run it
# It will test all functionality automatically
```

## ?? That's It!

You now have a working interactive metrics editor that:
- ? Loads CSV files automatically
- ? Provides a beautiful UI for editing
- ? Auto-saves changes
- ? Works for technical and non-technical users
- ? Saves edited data back to CSV

**Happy editing!** ??
