# ?? How to Run in Databricks

## Complete Step-by-Step Guide

---

## ?? Method 1: Import the Python File (Easiest)

### Step 1: Download the File
```
1. Locate: databricks_super_simple.py (in your workspace folder)
2. Download it to your computer
```

### Step 2: Open Databricks
```
1. Go to your Databricks workspace URL
   Example: https://your-company.cloud.databricks.com
2. Log in with your credentials
```

### Step 3: Import the Notebook
```
1. In Databricks, look at the left sidebar
2. Click "Workspace" 
3. Navigate to your desired folder (usually "Users/your-email")
4. Right-click in the folder area
5. Select "Import"
```

### Step 4: Upload the File
```
1. In the Import dialog:
   - Click "Browse" or drag-and-drop
   - Select "databricks_super_simple.py" from your computer
   - Click "Import"
```

### Step 5: Open the Notebook
```
1. The notebook should open automatically
2. If not, find it in your folder and click it
3. You'll see the notebook with 3-4 cells
```

### Step 6: Run the Notebook
```
Option A - Run All Cells (Recommended):
   1. Click "Run All" at the top of the notebook
   2. Wait 5-10 seconds for all cells to execute
   3. Beautiful table editor will appear!

Option B - Run One Cell at a Time:
   1. Click on Cell 1
   2. Press Shift + Enter (or click ?? Run)
   3. Wait for it to finish
   4. Repeat for Cell 2, Cell 3
```

### Step 7: Start Editing!
```
1. You'll see a beautiful table with your metrics
2. Click any cell to edit
3. Type your changes
4. Click "?? SAVE CHANGES" button
5. Run the cell below (Cell 3 or 4)
6. Done! ?
```

---

## ?? Method 2: Copy-Paste Code (Alternative)

### Step 1: Create New Notebook
```
1. In Databricks, click "Workspace" in sidebar
2. Navigate to your folder
3. Click "Create" ? "Notebook"
4. Name it: "Metrics Config Editor"
5. Language: Python
6. Click "Create"
```

### Step 2: Copy the Code
```
1. Open databricks_super_simple.py in a text editor
2. Select all (Ctrl/Cmd + A)
3. Copy (Ctrl/Cmd + C)
```

### Step 3: Paste into Databricks
```
1. In your new Databricks notebook
2. Delete the default cell if present
3. Click "+ Code" to add a cell
4. Paste the entire code (Ctrl/Cmd + V)
```

### Step 4: Split into Cells
```
The code has markers like:
# COMMAND ----------

These tell Databricks where to split cells.
Databricks will automatically create separate cells.
```

### Step 5: Run All Cells
```
1. Click "Run All" at the top
2. Wait for execution
3. Start editing!
```

---

## ?? Method 3: Direct Import via URL (If File is on GitHub/Cloud)

### If You Have the File on GitHub:
```
1. In Databricks, click "Workspace"
2. Click "Import"
3. Select "URL"
4. Paste the raw GitHub URL
5. Click "Import"
6. Run All!
```

---

## ?? Visual Guide

### What You'll See in Databricks:

```
???????????????????????????????????????????????????????????
? Databricks Workspace                                    ?
???????????????????????????????????????????????????????????
?                                                          ?
?  ?? Workspace                                            ?
?    ?? ?? Users                                           ?
?        ?? ?? your.email@company.com                      ?
?            ?? ?? Metrics Config Editor ? YOU ARE HERE    ?
?            ?? ?? Other notebooks...                      ?
?            ?? ...                                        ?
?                                                          ?
???????????????????????????????????????????????????????????
```

### Your Notebook Will Look Like:

```
???????????????????????????????????????????????????????????
? ?? Metrics Config Editor                    [Run All ??] ?
???????????????????????????????????????????????????????????
?                                                          ?
? [Cell 1] ??  Setup and Configuration                     ?
? ? Loaded: /Workspace/Users/you/config.csv               ?
? ?? 3 rows, 4 columns                                     ?
?                                                          ?
???????????????????????????????????????????????????????????
?                                                          ?
? [Cell 2] ??  Interactive Editor                          ?
?                                                          ?
?  ?????????????????????????????????????????????????      ?
?  ? ? Metrics Configuration Editor               ?      ?
?  ? [?? SAVE CHANGES]                              ?      ?
?  ?                                                ?      ?
?  ?  # ? metric_name ? type ? weight ? Delete     ?      ?
?  ?  1 ? faith...    ? llm  ?  1.0   ?  ???        ?      ?
?  ?  2 ? relev...    ? llm  ?  0.8   ?  ???        ?      ?
?  ?????????????????????????????????????????????????      ?
?                                                          ?
???????????????????????????????????????????????????????????
?                                                          ?
? [Cell 3] ??  Apply Changes                               ?
? ??  No changes yet. Edit table above and run again.     ?
?                                                          ?
???????????????????????????????????????????????????????????
```

---

## ?? Complete Workflow Example

### First Time Setup (One-time, ~3 minutes):

```
1. [You] Download databricks_super_simple.py
        ?
2. [You] Go to Databricks ? Workspace ? Your folder
        ?
3. [You] Right-click ? Import ? Select file
        ?
4. [You] Click "Import"
        ?
5. [Databricks] Notebook opens automatically
        ?
6. [You] Click "Run All" button at top
        ?
7. [Databricks] Executes cells (5-10 seconds)
        ?
8. [Databricks] Shows beautiful table editor
        ?
9. ? READY TO USE!
```

### Regular Usage (Every time, ~2 minutes):

```
1. [You] Open notebook in Databricks
        ?
2. [You] Click "Run All"
        ?
3. [You] Edit table (click cells, type changes)
        ?
4. [You] Click "?? SAVE CHANGES" button
        ?
5. [You] Run Cell 3 (below the table)
        ?
6. [Databricks] Saves to CSV file
        ?
7. [Databricks] Shows confirmation: "? SUCCESS!"
        ?
8. ? DONE! File updated.
```

---

## ?? Troubleshooting

### Problem: "Can't find Import option"
```
Solution:
1. Make sure you're in the "Workspace" section (left sidebar)
2. Navigate to a folder (not at root level)
3. Right-click in the empty space of the folder
4. You should see "Import" option
```

### Problem: "Import fails"
```
Solutions:
1. Make sure file is .py extension
2. Try Method 2 (copy-paste) instead
3. Check file isn't corrupted
4. Try downloading file again
```

### Problem: "Cells don't appear after import"
```
Solution:
1. Databricks should auto-split cells at "# COMMAND ----------"
2. If not, you can manually split:
   - Click where you want to split
   - Press Ctrl/Cmd + Shift + -
```

### Problem: "Table doesn't appear"
```
Solution:
1. Make sure you ran Cell 2 (the one with displayHTML)
2. Try clicking "Clear" ? "Clear State" at top
3. Run All again
4. Wait a few seconds for HTML to render
```

### Problem: "Save button doesn't work"
```
Solution:
1. Click the button
2. You should see confirmation message
3. Then run Cell 3 (the cell below)
4. If still not working, press F12 to open console
5. Copy the JSON from console
6. Paste it manually in the widget that appears
```

### Problem: "File not found"
```
Solution:
Don't worry! The notebook creates a sample file automatically.
1. Check the file path shown in the interface
2. The editor will create: /Workspace/Users/you/sample_metrics_config_simplified.csv
3. If you have your own file, make sure it's in the right location
```

### Problem: "Permission denied"
```
Solution:
1. Make sure you have write access to your workspace folder
2. Try using /dbfs/FileStore/ location instead
3. Contact your Databricks admin if needed
```

---

## ?? File Location Tips

### Where Your CSV File Will Be Saved:

**Default location:**
```
/Workspace/Users/your.email@company.com/sample_metrics_config_simplified.csv
```

**Alternative locations (if default fails):**
```
/dbfs/FileStore/sample_metrics_config_simplified.csv
/Workspace/Shared/sample_metrics_config_simplified.csv
```

### How to Find Your CSV File:

```
Method 1 - Via Workspace:
1. Click "Workspace" in sidebar
2. Navigate to: Users ? your.email@company.com
3. Look for .csv files

Method 2 - Via DBFS:
1. Click "Data" in sidebar
2. Click "DBFS"
3. Navigate to FileStore folder
4. Look for your file

Method 3 - In Notebook:
Run this code in a cell:
```python
import os
print("Files in your directory:")
for file in os.listdir(f"/Workspace/Users/{current_user}"):
    if file.endswith('.csv'):
        print(f"  - {file}")
```
```

---

## ?? Configuration Options

### Change the CSV Filename:

Edit this line in Cell 1:
```python
CONFIG_FILE = "your_custom_filename.csv"
```

### Change the File Location:

Edit this line in Cell 1:
```python
FILE_PATH = "/your/custom/path/filename.csv"
```

### Use a Different CSV File:

Just change the CONFIG_FILE variable to point to your existing file!

---

## ?? Quick Reference Commands

### Keyboard Shortcuts in Databricks:

| Action | Shortcut |
|--------|----------|
| Run current cell | `Shift + Enter` |
| Run all cells | `Ctrl/Cmd + Shift + A` |
| Add cell below | `B` |
| Delete cell | `D D` (press D twice) |
| Edit mode | `Enter` |
| Command mode | `Esc` |
| Save notebook | `Ctrl/Cmd + S` |

---

## ? Success Checklist

After importing and running, you should see:

- [ ] ? Cell 1 shows: "Loaded: [file path]"
- [ ] ? Cell 2 displays beautiful HTML table
- [ ] ? Table is interactive (can click cells)
- [ ] ? "SAVE CHANGES" button is visible
- [ ] ? Can add new rows
- [ ] ? Can delete rows
- [ ] ? Running Cell 3 shows success message

If you see all of these, **you're all set!** ??

---

## ?? Next Steps

Now that you know how to run it:

1. **Try editing** some values
2. **Add a new row** using the form at bottom
3. **Delete a row** using the ??? button
4. **Click "SAVE CHANGES"** to test the save functionality
5. **Run Cell 3** to apply your changes
6. **Check the success message** to confirm it worked

**You're ready!** Start editing your metrics configuration! ?

---

## ?? Need More Help?

- **First time?** ? Read `QUICK_START_GUIDE.md`
- **Want details?** ? Read `DATABRICKS_EDITOR_README.md`
- **Technical info?** ? Read `IMPLEMENTATION_SUMMARY.md`
- **See improvements?** ? Read `BEFORE_AFTER_COMPARISON.md`

---

**Happy Editing!** ??
