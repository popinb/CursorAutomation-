# 🚀 **SIMPLIFIED Databricks LLM Judge System - Upload Guide**

## ✅ **SIMPLIFIED: Only 5 Files Needed!**

I've streamlined the system to use only **2 ground truth files** that are shared across all metrics. Much simpler!

## 📁 **Files to Download and Upload (5 total)**

### **1. Main Notebook**
- **`databricks_llm_judge_final_production.py`** - The complete Databricks notebook

### **2. Configuration File**  
- **`sample_metrics_config_simplified.csv`** - Metrics configuration (uses only 2 ground truth files)

### **3. Evaluation Data**
- **`evaluation_data.csv`** - Sample evaluation data (3 rows)

### **4. Ground Truth Files (Only 2!)**
- **`ground_truth_accuracy.csv`** - Used by accuracy & helpfulness metrics
- **`ground_truth_safety.csv`** - Used by safety & completeness metrics

## 🎯 **How the 2 Ground Truth Files Work**

### **`ground_truth_accuracy.csv`** contains:
- `sample_id` - Links to evaluation data
- `correct_answer` - Used by accuracy_check metric
- `helpful_answer` - Used by helpfulness_rating metric

### **`ground_truth_safety.csv`** contains:
- `sample_id` - Links to evaluation data  
- `safe_response` - Used by safety_check metric
- `complete_answer` - Used by completeness_score metric

## 📊 **Metric to File Mapping**

| Metric | Ground Truth File | Column Used |
|--------|------------------|-------------|
| `accuracy_check` | `ground_truth_accuracy.csv` | `correct_answer` |
| `helpfulness_rating` | `ground_truth_accuracy.csv` | `helpful_answer` |
| `safety_check` | `ground_truth_safety.csv` | `safe_response` |
| `completeness_score` | `ground_truth_safety.csv` | `complete_answer` |

## 🚀 **Upload Instructions**

### **Step 1: Download All 5 Files**
```
databricks_llm_judge_final_production.py
sample_metrics_config_simplified.csv
evaluation_data.csv
ground_truth_accuracy.csv
ground_truth_safety.csv
```

### **Step 2: Upload to Databricks**
1. Open your Databricks workspace
2. Navigate to **Workspace → Users → [Your Username]**
3. Click **"Import" → "File"**
4. Upload all 5 files to the same folder

### **Step 3: Create Notebook**
1. Upload `databricks_llm_judge_final_production.py`
2. Rename it to `LLM_Judge_Evaluation_System`
3. The notebook will automatically find all CSV files

### **Step 4: Run the Notebook**
1. **Run Cell 1**: Install packages
2. **Run Cell 2**: Load data (will find the 2 ground truth files)
3. **Run Cell 3**: Configure model
4. **Run Cell 4**: Validate metrics
5. **Run Cells 5-9**: Execute evaluation
6. **Run Cell 10**: View results

## ✅ **Benefits of Simplified System**

- **Only 5 files** (down from 11!)
- **2 ground truth files** shared across all metrics
- **Same functionality** as the complex version
- **Easier to manage** and upload
- **All features still work**: multi-ground truth, enhanced UI, LLM selection

## 🎯 **What You Get**

The system will evaluate your data using:
- **Accuracy Check** (binary) - Uses `correct_answer` from accuracy file
- **Helpfulness Rating** (1-5 scale) - Uses `helpful_answer` from accuracy file  
- **Safety Check** (binary) - Uses `safe_response` from safety file
- **Completeness Score** (percentage) - Uses `complete_answer` from safety file

## 🚀 **Ready to Go!**

Upload these 5 files and the system works immediately with all the advanced features:
- ✅ Multi-ground truth support
- ✅ Enhanced UI widgets  
- ✅ LLM dropdown selection
- ✅ File auto-discovery
- ✅ Comprehensive evaluation

**Much simpler, same powerful features!** 🎯