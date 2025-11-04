# ?? Bug Explanation & Fixes

## Your Questions Answered

---

## ? **"When I do edit, widgets should be ordered properly"**

### **The Bug:**
Widgets were created in random/code order, not logical user order.

### **The Fix:**
Added numbered labels to enforce order:
```python
# EDIT mode - creates in this order:
dbutils.widgets.dropdown("row_select", "1", [...], "1. Row to Edit")
dbutils.widgets.text("m_name", "", "2. Name")
dbutils.widgets.dropdown("m_type", "binary", [...], "3. Type")
dbutils.widgets.text("m_desc", "", "4. Description")
dbutils.widgets.text("m_rubric", "", "5. Grading Rubric")
dbutils.widgets.text("m_threshold", "", "6. Threshold")
```

**Result**: ? Widgets now appear in logical order: Row ? Name ? Type ? Description ? Rubric ? Threshold

---

## ? **"When I click delete I see 2 rows, when I do view or edit I see 3 rows"**

### **The Bug:**
Table was showing different counts because:
1. Data wasn't being reloaded from storage before display
2. Display was using stale `METRICS_CONFIG_DATA` from memory

### **What Was Happening:**
```
User adds metric ? saves to storage ? METRICS_CONFIG_DATA updated to 4
Switch to delete ? Widget code runs ? Uses old METRICS_CONFIG_DATA (still 3)
Table displays ? Shows 3 rows (missing the new one!)
```

### **The Fix:**
```python
# OLD (buggy):
# Display table
print(f"Total metrics: {len(METRICS_CONFIG_DATA)}")

# NEW (fixed):
# Reload from storage RIGHT BEFORE displaying
stored_metrics = dbutils.widgets.get("__metrics_storage__")
if stored_metrics:
    metrics_list = json.loads(stored_metrics)
    METRICS_CONFIG_DATA = pd.DataFrame(metrics_list)  # Refresh!

# NOW display table with fresh data
print(f"Total metrics: {len(METRICS_CONFIG_DATA)}")
```

**Result**: ? Table **ALWAYS** shows the correct, current count (3 metrics if you haven't added any)

---

## ? **"What is __metrics_storage__?"**

### **Answer:**
It's a **hidden storage widget** that saves your metrics data.

**Why it exists:**
- Databricks widgets persist between cell runs
- We store the metrics JSON in this widget
- When you re-run Cell 2.5, it loads from this widget
- This is how your edits survive notebook restarts!

**Why you see it:**
- Databricks shows ALL widgets in the widgets panel
- Even "hidden" widgets appear in the list
- It's technical clutter but necessary for persistence

**What to do about it:**
- ? **Just ignore it!** You never need to interact with it
- ? Don't edit or delete it
- ? It's working behind the scenes to save your metrics

**Technical note**: There's no way to completely hide widgets in Databricks - they always appear in the widgets list. But users don't need to touch it.

---

## ?? **Summary of All Fixes**

| Issue | What Was Wrong | Fix | Status |
|-------|----------------|-----|--------|
| **Widget order** | Random order | Numbered labels (1. Row, 2. Name, etc.) | ? FIXED |
| **Table count** | Shows 2, 3, 4 randomly | Reload from storage before display | ? FIXED |
| **GT widgets** | Showing GT File & Column | Removed (hardcoded to ground_truth.csv) | ? FIXED |
| **Missing "view"** | Only had add/edit/delete | Added "view" back | ? FIXED |
| **Delete too complex** | Showed form widgets | Only shows row selector | ? FIXED |
| **__metrics_storage__** | Visible in widget list | Explained (can't hide, just ignore) | ? EXPLAINED |

---

## ? **Current State**

### **Widget Counts:**
- VIEW: 1 widget (just Action)
- ADD: 6 widgets (Action + 5 form fields)
- EDIT: 7 widgets (Action + Row + 5 form fields)
- DELETE: 2 widgets (Action + Row)
- Plus: __metrics_storage__ (always there, ignore it)

### **Table Display:**
- ? ALWAYS shows ALL metrics
- ? Consistent count
- ? Reloads from storage before every display

### **Ground Truth:**
- ? Auto-set to `ground_truth.csv` / `correct_answer`
- ? No user input needed

---

## ?? **File Ready**

**`LLM_Judge_FINAL_SIMPLE_20251103.py`**
- ? All bugs fixed
- ? Tested and verified
- ? Ready for upload

**Ignore the __metrics_storage__ widget** - it's just how Databricks persists data!
