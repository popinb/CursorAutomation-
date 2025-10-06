# Zillow LLM Judge - Generalized Metrics Evaluation System

A flexible framework for Product Managers to evaluate AI responses using custom LLM-based metrics with MLflow 3.0 visualization.

## 🚀 Quick Start

1. **Configure Metrics**: Add your custom metrics in section 4 of the notebook
2. **Set Data Source**: Update the data path in the configuration
3. **Run Evaluation**: Execute the evaluation pipeline
4. **View Results**: Check MLflow UI for detailed visualizations

## 📊 Features

- **Easy Configuration**: Add/remove metrics through simple configuration
- **Multiple Metric Types**: Binary, categorical, and continuous metrics
- **Ensemble Evaluation**: Use multiple LLM judges for reliable scoring
- **MLflow Integration**: Automatic logging and visualization
- **Deterministic Results**: Consistent evaluation across runs
- **Zillow-Specific**: Designed for real estate and financial contexts

## 🎯 Example Metrics Included

### 1. Response Quality (Binary)
Evaluates if the AI response is helpful and relevant to the user's question.

### 2. Personalization Accuracy (Binary)
Checks if the AI correctly uses user's personal information without distortion.

### 3. Helpfulness Scale (Categorical 1-5)
Rates how helpful the response is on a 1-5 scale.

## 🔧 Configuration

### Adding a New Metric

```python
NEW_METRIC = MetricConfig(
    name="your_metric_name",
    description="What this metric measures",
    metric_type=MetricType.BINARY,  # or CATEGORICAL, CONTINUOUS
    prompt_template="""
    Your evaluation prompt here...
    Use {prompt}, {response}, {user_profile}, {context} as needed
    """,
    threshold=1.0,  # Optional threshold for pass/fail
    required_variables=["prompt", "response"]
)
```

### Available Variables for Prompts
- `{prompt}`: User's question/query
- `{response}`: AI's response
- `{user_profile}`: User's personal information
- `{context}`: Additional context (if available)

### Metric Types
- **BINARY**: 0/1 or True/False responses
- **CATEGORICAL**: 1-5 scale or custom categories
- **CONTINUOUS**: Any numeric value

## 📈 MLflow Visualizations

The system automatically creates:
- Distribution plots for each metric
- Time series analysis
- Summary statistics tables
- Pass/fail rates
- Detailed scoring breakdowns

## 🛠️ Advanced Features

### Ensemble Evaluation
Use multiple LLM models for more reliable scoring:
```python
EVALUATION_CONFIG = EvaluationConfig(
    judge_models=["gpt-4o", "gpt-4o-mini"],  # Multiple models
    max_concurrency=2
)
```

### Batch Processing
For large datasets, process in batches:
```python
results = run_batch_evaluation("/path/to/large_dataset.csv", batch_size=50)
```

### Custom Model Configuration
Configure different models for different metrics or adjust concurrency limits.

## 📋 Data Format

Your input CSV should have these columns:
- `prompt`: User questions/queries
- `response`: AI responses to evaluate
- `user_profile`: User personal information (optional)
- `context`: Additional context (optional)

## 🔍 Troubleshooting

### Common Issues

1. **API Key Issues**
   - Ensure OpenAI API key is correctly set
   - Check that the key has sufficient credits

2. **Data Format Issues**
   - Ensure required columns exist
   - Check data types and formats

3. **Evaluation Errors**
   - Verify prompt templates use correct variable names
   - Ensure JSON output format is properly specified

4. **Performance Issues**
   - Reduce max_concurrency if hitting rate limits
   - Use fewer judge models for faster evaluation

### Best Practices

1. **Start Simple**: Begin with 1-2 metrics and expand gradually
2. **Test Prompts**: Validate templates with a few examples first
3. **Use Ensemble**: Multiple judge models provide more reliable results
4. **Set Thresholds**: Define clear pass/fail criteria
5. **Monitor Costs**: LLM evaluation can be expensive with large datasets

## 📊 Example Output

The system provides:
- Individual scores for each sample
- Mean, std, min, max statistics
- Pass/fail rates based on thresholds
- Detailed explanations for each evaluation
- Rich visualizations in MLflow UI

## 🚀 Getting Started

1. Clone this notebook
2. Update the configuration in section 5
3. Add your metrics in section 4
4. Run the evaluation pipeline
5. Check MLflow UI for results

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the example metrics for guidance
3. Contact the ML team for advanced support

## 🔄 Version History

- **v1.0**: Initial release with basic metrics
- **v1.1**: Added ensemble evaluation support
- **v1.2**: Enhanced MLflow visualizations
- **v1.3**: Added batch processing capabilities

---

**Note**: This system is designed specifically for Zillow's context and should be used with appropriate data privacy and security considerations.