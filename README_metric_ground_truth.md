# LLM Judge Evaluation System with Metric-Specific Ground Truth

This enhanced version of the LLM Judge Evaluation System allows each metric to have its own ground truth file, providing maximum flexibility for complex evaluation scenarios.

## 🎯 Key Enhancement

Each evaluation metric can now specify its own ground truth CSV file, allowing you to:
- Use different ground truth sources for different metrics
- Share ground truth files between metrics
- Have metrics that don't require ground truth at all
- Use different join keys for different ground truth files

## 📁 File Structure

```
/workspace/
├── llm_judge_evaluation_with_metric_ground_truth.py  # Main notebook
├── sample_evaluation_data.csv                        # Your data to evaluate
├── sample_metrics_config.csv                         # Metric definitions
└── ground_truth/                                     # Ground truth files
    ├── accuracy_gt.csv
    ├── completeness_gt.csv
    └── relevance_gt.csv
```

## 📊 Metrics Configuration CSV Format

The metrics configuration CSV must include these columns:

| Column | Description | Example |
|--------|-------------|---------|
| `name` | Metric identifier | `accuracy_check` |
| `type` | Metric type | `binary`, `scale_1_5`, `percentage` |
| `description` | Human-readable description | `Checks factual accuracy` |
| `evaluation_prompt` | LLM prompt with placeholders | `Evaluate accuracy...{prompt}...{response}...{ground_truth}` |
| `threshold` | Pass/fail threshold | `1.0`, `3.0`, `0.7` |
| `ground_truth_path` | Path to ground truth CSV | `/workspace/ground_truth/accuracy_gt.csv` |
| `join_key` | Column to join with eval data | `prompt_id`, `id`, `question_num` |

### Example metrics_config.csv:
```csv
name,type,description,evaluation_prompt,threshold,ground_truth_path,join_key
accuracy_check,binary,"Checks accuracy","Is accurate? {prompt} {response} {ground_truth}",1.0,/workspace/ground_truth/accuracy_gt.csv,prompt_id
style_check,percentage,"Checks style","Rate style {prompt} {response}",0.7,,
```

## 📄 Ground Truth File Format

Each ground truth CSV file should contain:
1. A key column that matches your evaluation data (e.g., `prompt_id`, `id`, `question_num`)
2. One or more columns with ground truth data (the first non-key column will be used)

### Example accuracy_gt.csv:
```csv
prompt_id,correct_answer
1,"Paris is the capital of France"
2,"Machine learning is a subset of AI"
```

## 🔧 How It Works

1. **Metric Definition**: Each metric in your metrics CSV specifies its own ground truth file path
2. **Ground Truth Loading**: The system loads each ground truth file when initializing metrics
3. **Flexible Joining**: Each metric can use a different join key to match with evaluation data
4. **Smart Defaults**: If no join key is specified, the system will auto-detect common keys or use row index
5. **Evaluation**: During evaluation, each metric retrieves its specific ground truth for each sample

## 🚀 Usage Example

1. **Prepare your evaluation data** (CSV with prompts and responses)
2. **Create ground truth files** for metrics that need them
3. **Define your metrics** in a CSV with ground truth file paths
4. **Run the notebook** - it will automatically load and use the correct ground truth for each metric

## 💡 Benefits

- **Flexibility**: Different metrics can use different ground truth sources
- **Reusability**: Multiple metrics can share the same ground truth file
- **Simplicity**: Metrics without ground truth can leave the path empty
- **Scalability**: Easy to add new metrics with their own ground truth

## 📝 Tips

- Use descriptive file names for ground truth files (e.g., `factual_accuracy_gt.csv`)
- Keep ground truth files in a dedicated directory for organization
- Use consistent join keys across related files
- Document which ground truth file each metric uses

## 🔍 Troubleshooting

If ground truth isn't loading:
1. Check file paths are absolute or relative to notebook location
2. Verify join key exists in both evaluation data and ground truth file
3. Ensure ground truth CSV has at least 2 columns (key + ground truth)
4. Check for typos in column names

## 📈 Example Output

The system will show:
- Which ground truth file is being used for each metric
- How many ground truth entries were loaded
- Warnings if ground truth can't be found for specific samples
- Final evaluation results with metric-specific ground truth considered