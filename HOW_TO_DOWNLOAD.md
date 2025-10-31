# ?? How to Download Your Auto-Saved Metrics File

## ?? Quick Answer

Your auto-saved file is at: `/Workspace/Users/[your-email]/sample_metrics_config_simplified.csv`

---

## ?? Method 1: Direct Download from Databricks UI (EASIEST)

### Step-by-Step:

1. **Open Databricks Workspace sidebar** (left side)

2. **Navigate to your file:**
   - Click "Workspace" 
   - Click "Users"
   - Click your email/username folder
   - Find `sample_metrics_config_simplified.csv`

3. **Download:**
   - **Right-click** the file ? "Download"
   - OR **Click** the file ? Click **3 dots (?)** ? "Download"

**Done!** File downloads to your computer.

---

## ?? Method 2: Using the Notebook (Add This Cell)

Add a new cell to your notebook with this code:

```python
# DOWNLOAD CELL - Add this as Cell 8

print("="*80)
print("?? DOWNLOAD YOUR METRICS FILE")
print("="*80)

# Get the file
import pandas as pd

# Load current metrics
current_df, current_path = find_and_load()

print(f"?? Current file: {current_path}")
print(f"?? Rows: {len(current_df)}")
print()

# Display download options
print("?? DOWNLOAD OPTIONS:")
print()
print("OPTION 1: Direct Download")
print("  1. Go to Workspace ? Users ? [your-name]")
print(f"  2. Find file: {os.path.basename(current_path)}")
print("  3. Right-click ? Download")
print()
print("OPTION 2: Download via File Browser")
print("  1. Click 'Data' in left sidebar")
print("  2. Navigate to: " + current_path)
print("  3. Click download icon")
print()
print("OPTION 3: Copy to DBFS FileStore (downloadable URL)")

# Button to copy to FileStore
dbutils.widgets.dropdown("download_action", "Select Action", 
                         ["Select Action", "Copy to FileStore", "View Current Data"])

action = dbutils.widgets.get("download_action")

if action == "Copy to FileStore":
    try:
        # Copy to FileStore (creates downloadable URL)
        filestore_path = f"/dbfs/FileStore/{os.path.basename(current_path)}"
        current_df.to_csv(filestore_path, index=False)
        
        # Get Databricks workspace URL
        workspace_url = spark.conf.get("spark.databricks.workspaceUrl")
        download_url = f"https://{workspace_url}/files/{os.path.basename(current_path)}"
        
        print("="*80)
        print("? SUCCESS! File copied to FileStore")
        print("="*80)
        print(f"?? Download URL:")
        print(f"   {download_url}")
        print()
        print("?? Click the URL above to download")
        print("="*80)
        
    except Exception as e:
        print(f"? Error: {e}")
        print()
        print("?? Try Option 1 or 2 instead")

elif action == "View Current Data":
    print("="*80)
    print("?? CURRENT DATA PREVIEW")
    print("="*80)
    display(current_df)

print()
print("="*80)
```

**Then:**
1. Select "Copy to FileStore" from dropdown
2. Run the cell
3. Click the URL that appears
4. File downloads!

---

## ?? Method 3: Command Line Download (For Advanced Users)

If you have Databricks CLI installed:

```bash
databricks workspace export /Workspace/Users/[your-email]/sample_metrics_config_simplified.csv ./metrics_config.csv
```

---

## ?? Method 4: Export from Notebook Display

**Add this simple cell:**

```python
# CELL: Quick Export

import pandas as pd
from IPython.display import FileLink

# Load current data
current_df, _ = find_and_load()

# Save to local notebook directory
export_path = "/tmp/metrics_export.csv"
current_df.to_csv(export_path, index=False)

print("? File ready for download!")
print()
print("Download link:")
display(FileLink(export_path))
```

**Then:** Click the link that appears to download!

---

## ?? Method 5: Download Button in UI (Best for Non-Tech Users)

I can add a download button directly in your editor UI! Want me to add this?

---

## ?? Troubleshooting

### "Can't find the file"
- File is saved in: `/Workspace/Users/[your-email]/`
- Run Cell 6 in the notebook to see exact location
- Check the widget at top of notebook for filename

### "No download option"
- You might not have permissions
- Ask your Databricks admin for workspace access
- Try Method 2 (FileStore) instead

### "Download URL doesn't work"
- Make sure you're logged into Databricks
- Try right-click ? "Open in new tab"
- Use Method 1 (Direct Download) instead

---

## ?? Recommended Method by User Type

| User Type | Best Method | Why |
|-----------|-------------|-----|
| Non-Technical | Method 1 (UI) | Simplest, no code |
| Occasional Users | Method 2 (Notebook) | One-click from notebook |
| Regular Users | Method 5 (UI Button) | Built into editor |
| Advanced Users | Method 3 (CLI) | Scriptable, automated |
| Quick Export | Method 4 (Display) | Fastest for one-time |

---

## ?? What You're Downloading

- **File type:** CSV (opens in Excel, Google Sheets, etc.)
- **Contents:** Your metrics configuration table
- **Columns:** metric_name, metric_type, weight, enabled (or your custom columns)
- **Format:** Standard CSV, comma-separated

---

## ?? Pro Tips

1. **Regular backups:** Download weekly to keep versions
2. **Name versions:** Save as `metrics_config_2025-10-31.csv`
3. **Share with team:** Upload to shared folder in Databricks
4. **Edit offline:** Download ? Edit in Excel ? Re-upload to Databricks

---

## ?? Want Me to Add a Download Button?

I can modify your notebook to include a "?? Download File" button in the UI!

Just say: "Add download button" and I'll update the code.

---

**Need help?** Let me know which method you want to use and I can walk you through it! ??
