# Enhanced LLM Judge Evaluation System

## 🚀 Overview

This enhanced LLM Judge evaluation system supports **per-metric ground truth files**, allowing each evaluation metric to use its own dedicated ground truth source. This provides maximum flexibility for complex evaluation scenarios where different metrics require different types of reference data.

## ✨ Key Features

- **🗂️ Per-metric ground truth files** - Each metric can specify its own ground truth file
- **📊 Flexible column mapping** - No fixed column names required
- **🔍 Auto-detection** - System automatically finds appropriate ground truth columns
- **📈 Enhanced reporting** - Ground truth coverage tracking and detailed metrics
- **💾 Complete export** - All data including ground truth values saved for audit

## 📋 Setup Instructions

### 1. Prepare Your Metrics Configuration CSV

Create a CSV file with the following columns:

| Column | Required | Description | Example |
|--------|----------|-------------|---------|
| `name` | ✅ | Metric identifier | `accuracy_check` |
| `type` | ✅ | Metric type | `binary`, `scale_1_5`, `percentage` |
| `description` | ✅ | Human readable description | `Checks factual accuracy` |
| `evaluation_prompt` | ✅ | LLM prompt template | `"Evaluate accuracy..."` |
| `threshold` | ✅ | Pass/fail threshold | `1.0`, `3.0`, `0.7` |
| `ground_truth_file_path` | ❌ | Path to ground truth file | `/path/to/gt.csv` |
| `ground_truth_column` | ❌ | Column name in GT file | `correct_answer` |

### 2. Prepare Ground Truth Files (Optional)

Each metric can have its own ground truth file:
- **Format**: CSV with any column structure
- **Flexibility**: System auto-detects ground truth columns
- **Fallback**: If no file specified, uses main evaluation data

### 3. Prepare Evaluation Data

CSV file with your prompts and responses:
- **Required columns**: System auto-detects prompt/response columns
- **Common names**: `prompt`, `question`, `query` / `response`, `answer`, `output`

## 🔧 Usage Example

### Sample Files Structure
```
/workspace/
├── sample_evaluation_data.csv          # Main evaluation data
├── sample_metrics_config.csv           # Metrics configuration
├── sample_accuracy_gt.csv              # Ground truth for accuracy metric
├── sample_helpfulness_gt.csv           # Ground truth for helpfulness metric
├── sample_relevance_gt.csv             # Ground truth for relevance metric
└── enhanced_llm_judge_notebook.py      # Enhanced notebook
```

### Sample Metrics Configuration
```csv
name,type,description,evaluation_prompt,threshold,ground_truth_file_path,ground_truth_column
accuracy_check,binary,Checks factual accuracy,"Evaluate accuracy...",1.0,/workspace/sample_accuracy_gt.csv,correct_answer
helpfulness_rating,scale_1_5,Rates helpfulness,"Rate helpfulness...",3.0,/workspace/sample_helpfulness_gt.csv,helpfulness_score
relevance_score,percentage,Measures relevance,"Score relevance...",70.0,/workspace/sample_relevance_gt.csv,relevance_percentage
completeness,binary,Checks completeness,"Check completeness...",1.0,,ground_truth
```

### Sample Ground Truth Files

**accuracy_gt.csv:**
```csv
question_id,correct_answer,source
1,Paris,Encyclopedia
2,Machine learning uses algorithms...,Textbook
3,Mix ingredients and bake at 350°F...,Recipe
```

**helpfulness_gt.csv:**
```csv
prompt_id,helpfulness_score,notes
1,4,Very helpful with detail
2,5,Excellent explanation
3,3,Adequate but basic
```

## 🎯 Metric Types

### Binary Metrics
- **Range**: 0 or 1
- **Threshold**: Usually 1.0 (must pass)
- **Use cases**: Accuracy checks, completeness validation

### Scale 1-5 Metrics  
- **Range**: 1 to 5
- **Threshold**: Usually 3.0 (above average)
- **Use cases**: Quality ratings, helpfulness scores

### Percentage Metrics
- **Range**: 0 to 100 (or 0.0 to 1.0)
- **Threshold**: Usually 0.7 or 70.0
- **Use cases**: Relevance scores, similarity measures

## 📊 Enhanced Outputs

The system provides comprehensive results including:

1. **Individual scores** per metric per sample
2. **Pass/fail status** based on thresholds
3. **Explanations** from the LLM judge
4. **Ground truth values used** for each evaluation
5. **Coverage statistics** showing GT availability
6. **Summary metrics** with pass rates and averages

## 🔄 Running the System

1. **Upload files** using the Databricks widgets
2. **Configure model** (Databricks LLM or OpenAI)
3. **Run evaluation** - system handles all ground truth loading
4. **View results** in enhanced display format
5. **Export data** with complete audit trail

## 🛠️ Advanced Features

### Auto-Detection
- **Column detection**: Finds prompt/response columns automatically
- **Ground truth detection**: Identifies GT columns by keywords
- **File validation**: Checks file existence and format

### Error Handling
- **Missing files**: Graceful fallback to main evaluation data
- **Column mismatches**: Auto-mapping and fallback strategies
- **Length mismatches**: Automatic padding/truncation

### Performance
- **Parallel processing**: Where possible
- **Rate limiting**: Built-in delays for Databricks endpoints
- **Memory efficient**: Streaming processing for large datasets

## 📈 Benefits Over Standard System

1. **Separation of concerns** - Each metric has its own GT source
2. **Flexibility** - No need to merge all GT into one file
3. **Scalability** - Easy to add new metrics with new GT sources
4. **Maintainability** - GT files can be updated independently
5. **Auditability** - Complete tracking of what GT was used

## 🚨 Best Practices

1. **File organization** - Keep GT files organized by metric type
2. **Column naming** - Use descriptive column names in GT files
3. **Data quality** - Ensure GT files have good coverage
4. **Version control** - Track changes to GT files
5. **Testing** - Validate GT files before running evaluations

## 🔍 Troubleshooting

### Common Issues

**Ground truth file not found:**
- Check file paths in metrics config
- Ensure files exist and are readable

**Column not found:**
- System will auto-detect, but specify column names for clarity
- Check column names in GT files

**Low coverage:**
- Check GT file length matches evaluation data
- Verify column names and data quality

**Performance issues:**
- Use smaller batches for large datasets
- Consider using Databricks LLM to avoid rate limits

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Verify file formats and paths
3. Run the test functionality in Cell 11
4. Check Databricks logs for detailed error messages

---

**Happy evaluating with enhanced ground truth support!** 🎯📊