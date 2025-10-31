# ?? Databricks Metrics Configuration Editor

## Simple Editor for Non-Technical Users

### ? Features
- ?? Edit metrics configuration in a beautiful, intuitive table
- ? Add new rows easily
- ??? Delete rows with one click
- ?? Simple save process
- ?? Automatically finds and updates your config file

---

## ?? Quick Start

### Option 1: **SUPER SIMPLE** (Recommended)
Use `databricks_super_simple.py` - The easiest option!

**Steps:**
1. Import the notebook into Databricks
2. Click "Run All"
3. Edit the table that appears
4. Click "SAVE CHANGES" button
5. Run the cell below the table
6. Done! ?

---

## ?? Which File to Use?

| File | Best For | Complexity |
|------|----------|------------|
| `databricks_super_simple.py` | Everyone | ? Simple |
| `databricks_autosave_final.py` | More features | ?? Medium |
| `databricks_simple_editor.py` | Customization | ?? Medium |

---

## ?? How It Works

1. **Load**: Finds your CSV file automatically
2. **Edit**: Beautiful HTML table for easy editing
3. **Save**: One-click save to update the file
4. **Done**: File is updated on Databricks workspace

---

## ?? File Locations

The editor automatically searches these locations:
- `/Workspace/Users/<your_username>/sample_metrics_config_simplified.csv`
- `/dbfs/FileStore/sample_metrics_config_simplified.csv`
- `/Workspace/Shared/sample_metrics_config_simplified.csv`

If the file doesn't exist, it creates a sample one for you!

---

## ?? Tips for Non-Technical Users

### Editing Values
- **Text**: Just type normally
- **Numbers**: Type numbers (e.g., `1.5`, `42`)
- **True/False**: Type `true` or `false`
- The editor automatically converts types!

### Adding Rows
1. Scroll to "Add New Row" section
2. Fill in the fields
3. Click "Add Row"
4. Click "SAVE CHANGES" when done

### Deleting Rows
1. Click the ??? button next to the row
2. Confirm deletion
3. Click "SAVE CHANGES"

---

## ?? Troubleshooting

### "No changes detected"
- Make sure you clicked "SAVE CHANGES" button
- Run the cell below the table

### "File not found"
- The editor creates a sample file automatically
- Check the file path shown in the interface

### Save button not working
1. Open browser console (F12)
2. Copy the JSON shown
3. Paste in the widget that appears
4. Run the cell

---

## ?? Customization

Want to edit a different file? Change this line:
```python
CONFIG_FILE = "your_file_name.csv"
```

---

## ?? Important Notes

- Always click "SAVE CHANGES" after editing
- Changes are not saved until you run the save cell
- The file is updated on Databricks workspace, not locally

---

## ?? Support

If you need help:
1. Check the file path shown in the interface
2. Make sure you have write permissions
3. Try creating the file manually first
4. Check Databricks logs for errors

---

## ?? Example Configuration

```csv
metric_name,metric_type,weight,enabled
faithfulness,llm_judge,1.0,true
relevance,llm_judge,0.8,true
coherence,llm_judge,0.6,false
```

---

**Made with ?? for non-technical users**
