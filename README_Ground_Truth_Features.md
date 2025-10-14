# 🆕 Enhanced Ground Truth Support for LLM Judge Evaluation

## Overview

The Databricks LLM Judge Evaluation System now supports **per-metric ground truth files** with **workspace-based file management**. Simply upload your files to the Databricks workspace and reference them by filename only - no need for full paths!

## 🎯 Key Features

### 1. **Per-Metric Ground Truth Files**
- Each metric can have its own dedicated ground truth file
- No need to merge all ground truth data into a single file
- Different metrics can use different data sources and formats

### 2. **Workspace-Based File Management**
- Upload files directly to your Databricks workspace
- Reference files by filename only (no full paths needed)
- System automatically searches workspace directories
- Supports multiple search locations for flexibility

### 3. **Flexible Column Mapping**
- Ground truth files don't need specific column names
- System automatically detects and maps appropriate columns
- Supports various file structures and naming conventions

### 4. **Automatic Data Merging**
- Intelligently finds common columns between evaluation data and ground truth files
- Handles different merge strategies automatically
- Provides detailed logging of the merging process

## 📊 Updated CSV Format

### Metrics Configuration CSV
Your metrics CSV now supports an additional column:

```csv
name,type,description,evaluation_prompt,threshold,ground_truth_column,ground_truth_file_path
accuracy_check,binary,Checks accuracy,"Evaluate...",1.0,correct_answer,accuracy_ground_truth.csv
helpfulness,scale_1_5,Rate helpfulness,"Rate...",3.0,helpful_response,helpfulness_ground_truth.csv
safety,binary,Check safety,"Check...",1.0,safe_content,safety_ground_truth.csv
```

### Ground Truth File Format
Each ground truth file should have:
- A column that matches your evaluation data (for merging)
- A column with the ground truth data (can be any name)

**Example accuracy_ground_truth.csv:**
```csv
sample_id,correct_answer
1,Paris is the capital of France
2,Machine learning is a subset of AI
3,Chocolate cake requires flour, sugar, eggs, and cocoa
```

**Example helpfulness_ground_truth.csv:**
```csv
sample_id,helpful_answer
1,This response provides clear, accurate information
2,The explanation is simple and easy to understand
3,The recipe includes all necessary ingredients
```

## 🔧 How It Works

### 1. **Data Loading Process**
1. Load evaluation data (prompts and responses) from workspace
2. Load metrics configuration CSV from workspace
3. For each metric with a ground truth file:
   - Search workspace for the ground truth file by name
   - Load the specific ground truth file
   - Find common columns with evaluation data
   - Merge the ground truth data
   - Map to the specified ground truth column

### 2. **Intelligent Column Detection**
- **Merge Column**: Finds common columns between eval data and ground truth
- **Ground Truth Column**: Uses specified column or auto-detects text columns
- **Fallback Handling**: Gracefully handles missing or mismatched data

### 3. **Evaluation Process**
- Each metric uses its specific ground truth column
- Ground truth data is passed to the LLM evaluation prompt
- Results include metric-specific ground truth information

## 📁 Workspace File Structure Example

```
Databricks Workspace/
├── evaluation_data.csv                    # Main evaluation data
├── sample_metrics_config.csv             # Metrics configuration
├── accuracy_ground_truth.csv             # Ground truth for accuracy metric
├── helpfulness_ground_truth.csv          # Ground truth for helpfulness metric
├── safety_ground_truth.csv               # Ground truth for safety metric
├── completeness_ground_truth.csv         # Ground truth for completeness metric
└── results/ (generated)
    ├── evaluation_results_YYYYMMDD_HHMMSS.csv
    └── summary_YYYYMMDD_HHMMSS.csv
```

## 🚀 Usage Examples

### Example 1: Basic Setup
```csv
# sample_metrics_config.csv
name,type,description,evaluation_prompt,threshold,ground_truth_column,ground_truth_file_path
accuracy,binary,Check accuracy,"Evaluate accuracy: {response} vs {ground_truth}",1.0,correct_answer,accuracy_ground_truth.csv
```

### Example 2: Multiple Metrics with Different Files
```csv
# sample_metrics_config.csv
name,type,description,evaluation_prompt,threshold,ground_truth_column,ground_truth_file_path
factual_accuracy,binary,Check facts,"Check facts: {response} vs {ground_truth}",1.0,correct_facts,factual_ground_truth.csv
tone_appropriateness,scale_1_5,Check tone,"Rate tone: {response} vs {ground_truth}",3.0,appropriate_tone,tone_ground_truth.csv
safety_check,binary,Check safety,"Check safety: {response} vs {ground_truth}",1.0,safe_content,safety_ground_truth.csv
```

### Example 3: No Ground Truth File (Optional)
```csv
# sample_metrics_config.csv
name,type,description,evaluation_prompt,threshold,ground_truth_column,ground_truth_file_path
general_quality,scale_1_5,Rate quality,"Rate quality: {response}",3.0,quality_score,
```

## 🔍 Troubleshooting

### Common Issues and Solutions

1. **"No common columns found"**
   - Ensure your evaluation data and ground truth files share at least one column
   - Check column names for typos or case sensitivity

2. **"Ground truth file not found"**
   - Verify the filename in your metrics configuration
   - Ensure the file is uploaded to your Databricks workspace
   - Check that the filename matches exactly (case-sensitive)

3. **"No suitable ground truth data column found"**
   - Make sure your ground truth file has text data columns
   - Specify the exact column name in `ground_truth_column`

4. **"Ground truth matched for 0/X samples"**
   - Check that the merge column values match between files
   - Verify data types and formatting

5. **"File not found in workspace"**
   - Upload the file to your Databricks workspace
   - Check the file is in the correct directory
   - Verify the filename spelling in your metrics config

## 📈 Benefits

### 1. **Flexibility**
- Different metrics can use different data sources
- No need to maintain a single large ground truth file
- Easy to add new metrics with their own ground truth

### 2. **Simplicity**
- Just upload files to workspace and reference by name
- No complex path management needed
- Works seamlessly with Databricks environment

### 3. **Accuracy**
- Each metric gets exactly the ground truth it needs
- Reduces data confusion and mismatches
- Better evaluation quality

### 4. **Maintainability**
- Easier to update ground truth for specific metrics
- Clear separation of concerns
- Better organization of evaluation data

### 5. **Scalability**
- Can handle large datasets with multiple ground truth sources
- Efficient loading and processing
- Memory-friendly approach

## 🎯 Best Practices

1. **File Naming**: Use descriptive names like `accuracy_ground_truth.csv`, `safety_ground_truth.csv`
2. **Column Naming**: Use clear, consistent column names
3. **Data Quality**: Ensure ground truth data is accurate and complete
4. **Workspace Organization**: Upload all files to your Databricks workspace
5. **Documentation**: Document your ground truth sources and formats

## 🔄 Migration from Old Format

If you're upgrading from the old single-file format:

1. **Split your ground truth data** by metric
2. **Upload all files** to your Databricks workspace
3. **Update your metrics CSV** to include `ground_truth_file_path` column with just filenames
4. **Test with a small dataset** first
5. **Verify all metrics work** with their new ground truth files

The system is backward compatible - metrics without ground truth files will still work as before.