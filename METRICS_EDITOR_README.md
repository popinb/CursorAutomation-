# Databricks Metrics Editor

A simple, interactive metrics configuration editor for Databricks notebooks.

## ?? Quick Start

### Step 1: Upload the module to Databricks

Upload `databricks_metrics_editor.py` to your Databricks workspace:
- Go to `/Workspace/Users/your_email@company.com/`
- Upload the Python file

### Step 2: Use in your notebook

```python
# Import the module
from databricks_metrics_editor import MetricsEditor

# Initialize
editor = MetricsEditor()

# Load data
editor.load_data(
    evaluation_data_path="evaluation_data.csv",
    metrics_config_path="metrics_config.csv",
    ground_truth_files="ground_truth_1.csv;ground_truth_2.csv"
)

# Show interactive editor
editor.show_editor()

# After editing, apply changes
editor.apply_changes()

# Save to file
editor.save_metrics("metrics_edited.csv")
```

## ?? Complete API Reference

### Initialize Editor

```python
editor = MetricsEditor()
```

### Load Data

```python
editor.load_data(
    evaluation_data_path="evaluation_data.csv",      # Optional
    metrics_config_path="metrics_config.csv",        # Optional
    ground_truth_files="file1.csv;file2.csv"         # Optional, semicolon separated
)
```

**File paths can be:**
- Just the filename: `"metrics.csv"` (searches common locations)
- Absolute path: `"/Workspace/Users/user@email.com/metrics.csv"`
- Semicolon or comma separated for ground truth files

### Show Interactive Editor

```python
editor.show_editor()
```

Opens a beautiful UI where non-technical users can:
- ?? Edit any cell directly
- ? Add new rows
- ??? Delete rows
- ?? Auto-saves changes

### Apply Changes

```python
updated_df = editor.apply_changes()
```

Loads the changes made in the UI into memory.

### Save to File

```python
# Save with default location
editor.save_metrics("output.csv")

# Save to custom location
editor.save_metrics("output.csv", location="/Workspace/Shared/")
```

### Access Data

```python
# Get evaluation data
eval_df = editor.evaluation_data

# Get metrics configuration
metrics_df = editor.metrics_config_data

# Get ground truth data (dict of DataFrames)
gt_dict = editor.ground_truth_data

# Example: Access specific ground truth file
accuracy_df = editor.ground_truth_data.get("ground_truth_accuracy.csv")
```

### Display Summary

```python
editor.display_summary()
```

Shows a formatted summary of all loaded data.

### Get Summary as Dict

```python
summary = editor.get_summary()
print(summary)
```

Returns:
```python
{
    'user': 'user@email.com',
    'evaluation_data': {'loaded': True, 'rows': 100, 'columns': [...]},
    'metrics_config': {'loaded': True, 'rows': 5, 'columns': [...]},
    'ground_truth': {'loaded': True, 'files': ['file1.csv', 'file2.csv']}
}
```

### Cleanup

```python
editor.cleanup()
```

Removes temporary files (optional, runs automatically).

## ?? Complete Example

```python
# Databricks notebook source

# COMMAND ----------
# Import and initialize
from databricks_metrics_editor import MetricsEditor
editor = MetricsEditor()

# COMMAND ----------
# Load all data
editor.load_data(
    evaluation_data_path="evaluation_data.csv",
    metrics_config_path="metrics_config.csv",
    ground_truth_files="gt_accuracy.csv;gt_safety.csv"
)
editor.display_summary()

# COMMAND ----------
# Show interactive editor (for non-technical users)
editor.show_editor()

# COMMAND ----------
# Apply changes after editing
editor.apply_changes()
display(editor.metrics_config_data)

# COMMAND ----------
# Save edited metrics
editor.save_metrics("metrics_final.csv")

# COMMAND ----------
# Use the data in your code
for idx, row in editor.metrics_config_data.iterrows():
    print(f"Metric: {row['metric_name']}, Weight: {row['weight']}")
```

## ?? Features

### For Non-Technical Users
- ? Beautiful, intuitive UI
- ? No code required to edit
- ? Add/edit/delete rows with clicks
- ? Auto-saves changes
- ? Real-time updates

### For Developers
- ? Simple API (4 main methods)
- ? Auto-detects files in workspace
- ? Works with any CSV structure
- ? Easy integration with existing code
- ? Pandas DataFrame output

## ?? File Location

The module searches for files in these locations (in order):
1. `/Workspace/Users/{current_user}/filename.csv`
2. Current directory
3. `/Workspace/Shared/filename.csv`
4. `/dbfs/FileStore/filename.csv`
5. `/tmp/filename.csv`

## ?? Installation

**Option 1: Upload directly to Databricks**
1. Upload `databricks_metrics_editor.py` to your user workspace
2. Import in notebooks: `from databricks_metrics_editor import MetricsEditor`

**Option 2: Install from workspace**
```python
# If in a specific folder
import sys
sys.path.append('/Workspace/Users/your_email@company.com/')
from databricks_metrics_editor import MetricsEditor
```

## ?? Troubleshooting

**Issue: "Cannot display HTML"**
- Solution: Make sure you're running in a Databricks notebook

**Issue: "File not found"**
- Solution: Use `editor._find_file_in_workspace("yourfile.csv")` to debug
- Upload file to `/Workspace/Users/your_email@company.com/`

**Issue: Changes not applied**
- Solution: Make sure to run `editor.apply_changes()` after editing

## ?? Notes

- Changes are auto-saved to a temp file as you edit
- The temp file is cleaned up automatically
- All DataFrames use pandas format
- Compatible with Databricks Runtime 10.4+

## ?? That's It!

Simple, clean, and easy to use for both technical and non-technical users!
