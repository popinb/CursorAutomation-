# PM Metrics Evaluation Framework

A flexible, PM-friendly framework for evaluating AI model responses using LLM-as-a-Judge methodology with MLflow integration.

## 🎯 Overview

This framework allows Product Managers at Zillow to:
- **Define custom metrics** for evaluating AI responses
- **Run deterministic evaluations** using LLM judges
- **Track experiments** with MLflow for easy comparison
- **Visualize results** with built-in charts and dashboards
- **Export results** for stakeholder reporting

## 🚀 Quick Start

### 1. Setup
```python
# Clone the notebook: PM_Metrics_Evaluation_Framework.py
# Update the Configuration section (Section 2) with your details
```

### 2. Configure Your Evaluation
```python
# In Section 2, update these key variables:
EXPERIMENT_NAME = "Your_Experiment_Name"
PM_NAME = "Your_Name" 
PROJECT_NAME = "Your_Project"
DATA_FILE_PATH = "/path/to/your/data.csv"
```

### 3. Choose Your Metrics
```python
# In Section 4, select which metrics to run:
ACTIVE_METRICS = [
    "Accuracy",        # Binary: Is the response factually correct?
    "Helpfulness",     # Scale 1-5: How helpful is the response?
    "Personalization", # Scale 1-5: How well personalized?
    # Add your custom metrics here
]
```

### 4. Run Evaluation
Execute all cells in order. The framework will:
- Load your data
- Run evaluations using LLM judges
- Log results to MLflow
- Generate visualizations
- Export results for analysis

## 📊 Example Metrics

### 1. Accuracy (Binary Metric)
Evaluates whether responses are factually correct and complete.
- **Score**: 0 (Incorrect) or 1 (Correct)
- **Threshold**: 1 (must be perfect to pass)
- **Use Case**: Fact-checking, compliance verification

### 2. Helpfulness (Scale Metric)
Measures how useful the response is for the user.
- **Score**: 1-5 scale
- **Threshold**: 3 (adequate or better)
- **Use Case**: User experience evaluation, content quality

### 3. Personalization (Scale Metric)
Assesses how well the response uses user context.
- **Score**: 1-5 scale  
- **Threshold**: 3 (adequate personalization)
- **Use Case**: Personalization effectiveness, user engagement

## 🔧 Adding Custom Metrics

### Step 1: Create a Prompt Template
```python
YOUR_METRIC_PROMPT = """
You are an expert evaluator assessing [WHAT YOU'RE MEASURING].

### Task
[CLEAR DESCRIPTION OF EVALUATION TASK]

### Materials
**User Question:**
```
{prompt}
```

**AI Response:**
```
{response}
```

**User Context:**
```
{user_context}
```

### Evaluation Criteria
1. [CRITERION 1]: [Description]
2. [CRITERION 2]: [Description]
3. [CRITERION 3]: [Description]

### Scoring
- **[SCORE] ([LABEL])**: [Description]
- **[SCORE] ([LABEL])**: [Description]

### Output Format
Return ONLY this JSON:
```json
{{
  "your_metric_score": <score>,
  "explanation": "<brief reasoning>"
}}
```
"""
```

### Step 2: Add to Metrics Registry
```python
METRICS_REGISTRY = {
    # ... existing metrics ...
    
    "YourMetric": {
        "prompt_template": YOUR_METRIC_PROMPT,
        "threshold": 3,  # Adjust based on your scale
        "description": "Brief description of what this measures",
        "score_type": "scale"  # "binary" (0/1) or "scale" (1-5)
    }
}
```

### Step 3: Activate the Metric
```python
ACTIVE_METRICS = [
    "Accuracy",
    "Helpfulness", 
    "Personalization",
    "YourMetric",  # Add your metric here
]
```

## 📈 Understanding Results

### MLflow Dashboard
- **Experiments**: Compare different runs and configurations
- **Metrics**: Track performance over time
- **Artifacts**: Access detailed logs and visualizations
- **Parameters**: See what settings were used for each run

### Exported Files
- **detailed_results_[RUN_NAME].csv**: Individual sample scores and details
- **summary_stats_[RUN_NAME].json**: Aggregate statistics and metadata
- **evaluation_results_[RUN_NAME].png**: Visualization charts

### Key Metrics to Monitor
- **Mean Score**: Average performance across all samples
- **Pass Rate**: Percentage of samples meeting the threshold
- **Standard Deviation**: Consistency of performance
- **Overall Score**: Minimum score across all metrics (most conservative)

## 🎨 Customization Options

### Data Sources
The framework supports:
- **CSV files**: Standard comma-separated format
- **JSON files**: Structured data format
- **Sample data**: Built-in demo data for testing

### Required Columns
Your data must include:
- **Prompt column**: User questions/queries
- **Response column**: AI model responses
- **User context column**: Personalization data (optional)

### Judge Models
Configure which LLM models to use as judges:
```python
JUDGE_MODELS = ["gpt-4o"]  # Single judge
JUDGE_MODELS = ["gpt-4o", "gpt-4o-mini"]  # Ensemble (majority vote)
```

### Visualization Options
- **Bar charts**: Metric scores and pass rates
- **Heatmaps**: Score distribution across samples
- **Pie charts**: Overall pass/fail distribution
- **Custom plots**: Add your own visualization code

## 🔍 Troubleshooting

### Common Issues

**"Missing required columns"**
- Ensure your data has the columns specified in `PROMPT_COLUMN`, `RESPONSE_COLUMN`
- Update column names in the configuration section

**"API key not found"**
- Update the secrets scope and key names in the configuration
- Ensure you have access to the required API keys

**"Low pass rates"**
- Review your metric thresholds - they might be too strict
- Check if your data quality matches the metric expectations
- Consider adjusting prompt templates for clarity

**"MLflow logging errors"**
- Verify MLflow is properly configured
- Check experiment naming conventions
- Ensure you have write permissions

### Best Practices

1. **Start Small**: Test with a small dataset (10-20 samples) first
2. **Validate Metrics**: Review a few manual examples to ensure metrics work as expected
3. **Set Realistic Thresholds**: Based on your use case and data quality
4. **Document Changes**: Keep track of metric definitions and threshold adjustments
5. **Regular Reviews**: Periodically review and update metrics based on business needs

## 📋 Data Format Requirements

### Input Data Structure
```csv
prompt,response,user_context
"What's the best neighborhood for families?","Based on your needs, I recommend...","Family with 2 kids, $800k budget"
"How much house can I afford?","With your income, you can afford...","$100k income, $20k down payment"
```

### Output Data Structure
The framework adds these columns:
- `[MetricName]`: Raw scores (0-1 for binary, 1-5 for scale)
- `[MetricName]_status`: Pass/Fail status with emojis
- `[MetricName]_details`: Detailed judge reasoning
- `overall_score`: Minimum score across all metrics
- `overall_status`: Overall pass/fail status

## 🤝 Support & Feedback

### Getting Help
1. **Check the troubleshooting section** above
2. **Review the example metrics** for template guidance
3. **Test with sample data** to isolate issues
4. **Contact the development team** for technical support

### Providing Feedback
- **Metric suggestions**: What new metrics would be valuable?
- **Usability improvements**: How can we make this more PM-friendly?
- **Feature requests**: What additional functionality do you need?
- **Bug reports**: Any issues or unexpected behavior?

## 🔄 Version History

### v1.0 (Current)
- Initial PM-friendly framework
- 3 example metrics (Accuracy, Helpfulness, Personalization)
- MLflow integration with visualizations
- Comprehensive documentation and examples
- Batch processing and export capabilities

### Planned Features
- **Web interface**: GUI for non-technical users
- **Automated scheduling**: Regular evaluation runs
- **Advanced analytics**: Trend analysis and insights
- **Integration APIs**: Connect with other Zillow tools
- **Template library**: Pre-built metrics for common use cases

---

## 📞 Contact

For questions, support, or feedback about this framework, please reach out to the AI/ML Platform team.

**Happy evaluating!** 🚀