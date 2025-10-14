# 🆕 Enhanced Ground Truth Support for LLM Judge Evaluation

## Overview

The Databricks LLM Judge Evaluation System now supports **per-metric ground truth files**, allowing you to have distinct ground truth data for each evaluation metric. This provides much more flexibility and accuracy in your evaluations.

## 🎯 Key Features

### 1. **Per-Metric Ground Truth Files**
- Each metric can have its own dedicated ground truth file
- No need to merge all ground truth data into a single file
- Different metrics can use different data sources and formats

### 2. **Flexible Column Mapping**
- Ground truth files don't need specific column names
- System automatically detects and maps appropriate columns
- Supports various file structures and naming conventions

### 3. **Automatic Data Merging**
- Intelligently finds common columns between evaluation data and ground truth files
- Handles different merge strategies automatically
- Provides detailed logging of the merging process

## 📊 Updated CSV Format

### Metrics Configuration CSV
Your metrics CSV now supports an additional column:

```csv
name,type,description,evaluation_prompt,threshold,ground_truth_column,ground_truth_file_path
accuracy_check,binary,Checks accuracy,"Evaluate...",1.0,correct_answer,/path/to/accuracy_gt.csv
helpfulness,scale_1_5,Rate helpfulness,"Rate...",3.0,helpful_response,/path/to/helpfulness_gt.csv
safety,binary,Check safety,"Check...",1.0,safe_content,/path/to/safety_gt.csv
```

### Ground Truth File Format
Each ground truth file should have:
- A column that matches your evaluation data (for merging)
- A column with the ground truth data (can be any name)

**Example accuracy_gt.csv:**
```csv
sample_id,correct_answer
1,Paris is the capital of France
2,Machine learning is a subset of AI
3,Chocolate cake requires flour, sugar, eggs, and cocoa
```

**Example helpfulness_gt.csv:**
```csv
question_id,helpful_response
1,This response provides clear, accurate information
2,The explanation is simple and easy to understand
3,The recipe includes all necessary ingredients
```

## 🔧 How It Works

### 1. **Data Loading Process**
1. Load evaluation data (prompts and responses)
2. Load metrics configuration CSV
3. For each metric with a ground truth file:
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

## 📁 File Structure Example

```
your_project/
├── evaluation_data.csv          # Main evaluation data
├── metrics_config.csv           # Metrics configuration
├── ground_truth/
│   ├── accuracy_gt.csv         # Ground truth for accuracy metric
│   ├── helpfulness_gt.csv      # Ground truth for helpfulness metric
│   ├── safety_gt.csv           # Ground truth for safety metric
│   └── completeness_gt.csv     # Ground truth for completeness metric
└── results/
    ├── evaluation_results.csv   # Generated results
    └── summary.csv             # Generated summary
```

## 🚀 Usage Examples

### Example 1: Basic Setup
```csv
# metrics_config.csv
name,type,description,evaluation_prompt,threshold,ground_truth_column,ground_truth_file_path
accuracy,binary,Check accuracy,"Evaluate accuracy: {response} vs {ground_truth}",1.0,correct_answer,/data/accuracy_gt.csv
```

### Example 2: Multiple Metrics with Different Files
```csv
# metrics_config.csv
name,type,description,evaluation_prompt,threshold,ground_truth_column,ground_truth_file_path
factual_accuracy,binary,Check facts,"Check facts: {response} vs {ground_truth}",1.0,correct_facts,/data/facts_gt.csv
tone_appropriateness,scale_1_5,Check tone,"Rate tone: {response} vs {ground_truth}",3.0,appropriate_tone,/data/tone_gt.csv
safety_check,binary,Check safety,"Check safety: {response} vs {ground_truth}",1.0,safe_content,/data/safety_gt.csv
```

### Example 3: No Ground Truth File (Optional)
```csv
# metrics_config.csv
name,type,description,evaluation_prompt,threshold,ground_truth_column,ground_truth_file_path
general_quality,scale_1_5,Rate quality,"Rate quality: {response}",3.0,quality_score,
```

## 🔍 Troubleshooting

### Common Issues and Solutions

1. **"No common columns found"**
   - Ensure your evaluation data and ground truth files share at least one column
   - Check column names for typos or case sensitivity

2. **"Ground truth file not found"**
   - Verify the file path in your metrics configuration
   - Ensure the file exists and is accessible

3. **"No suitable ground truth data column found"**
   - Make sure your ground truth file has text data columns
   - Specify the exact column name in `ground_truth_column`

4. **"Ground truth matched for 0/X samples"**
   - Check that the merge column values match between files
   - Verify data types and formatting

## 📈 Benefits

### 1. **Flexibility**
- Different metrics can use different data sources
- No need to maintain a single large ground truth file
- Easy to add new metrics with their own ground truth

### 2. **Accuracy**
- Each metric gets exactly the ground truth it needs
- Reduces data confusion and mismatches
- Better evaluation quality

### 3. **Maintainability**
- Easier to update ground truth for specific metrics
- Clear separation of concerns
- Better organization of evaluation data

### 4. **Scalability**
- Can handle large datasets with multiple ground truth sources
- Efficient loading and processing
- Memory-friendly approach

## 🎯 Best Practices

1. **File Naming**: Use descriptive names like `accuracy_gt.csv`, `safety_gt.csv`
2. **Column Naming**: Use clear, consistent column names
3. **Data Quality**: Ensure ground truth data is accurate and complete
4. **File Organization**: Keep ground truth files in a dedicated folder
5. **Documentation**: Document your ground truth sources and formats

## 🔄 Migration from Old Format

If you're upgrading from the old single-file format:

1. **Split your ground truth data** by metric
2. **Update your metrics CSV** to include `ground_truth_file_path` column
3. **Test with a small dataset** first
4. **Verify all metrics work** with their new ground truth files

The system is backward compatible - metrics without ground truth files will still work as before.