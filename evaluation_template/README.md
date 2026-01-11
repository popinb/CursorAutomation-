# LLM Response Evaluation Template

A flexible, configurable template for evaluating LLM responses using customizable metrics and MLflow tracking. This template transforms the original Databricks notebook into a reusable framework that works with any dataset and evaluation criteria.

## 🚀 Quick Start

1. **Install Dependencies**
```bash
pip install mlflow>=3.0 langchain_openai pyyaml pandas httpx pydantic matplotlib
```

2. **Set Environment Variables**
```bash
export OPENAI_API_KEY="your-openai-api-key"
export OPENAI_API_BASE="https://api.openai.com/v1"  # Optional: custom endpoint
```

3. **Prepare Your Dataset**
Create a CSV file with your evaluation data. At minimum, you need:
- A column with prompts/questions
- A column with model responses to evaluate

4. **Configure Your Evaluation**
Copy and customize one of the example configurations or create your own:
```bash
cp examples/chatbot_evaluation.yaml config/my_evaluation.yaml
```

5. **Run the Evaluation**
```bash
python src/evaluation_template.py config/my_evaluation.yaml
```

## 📁 Project Structure

```
evaluation_template/
├── src/
│   └── evaluation_template.py     # Main evaluation pipeline
├── config/
│   ├── evaluation_config.yaml    # Main configuration template
│   └── metric_templates.yaml     # Library of reusable metrics
├── examples/
│   ├── chatbot_evaluation.yaml   # Chatbot quality evaluation
│   ├── qa_evaluation.yaml        # Q&A system evaluation
│   └── creative_writing_evaluation.yaml  # Creative writing evaluation
├── data/                          # Place your CSV datasets here
├── results/                       # Evaluation results are saved here
└── README.md                      # This file
```

## 🔧 Configuration Guide

### Dataset Configuration

```yaml
dataset:
  file_path: "./data/your_dataset.csv"
  columns:
    prompt: "question"           # Column containing input prompts
    response: "model_answer"     # Column containing responses to evaluate
    ground_truth: "correct_answer"  # Optional: reference answers
    metadata: "context"         # Optional: additional context
```

### Metrics Configuration

Define custom metrics with flexible prompt templates:

```yaml
metrics:
  your_metric_name:
    enabled: true
    description: "What this metric measures"
    prompt_template: |
      You are an impartial evaluator.
      Your task is to evaluate...
      
      ### Materials
      * **User Query:** {prompt}
      * **AI Response:** {response}
      * **Ground Truth:** {ground_truth}  # Optional
      
      ### Evaluation Guidelines
      Rate on a scale of 1-5...
      
      ### Output Format
      Return only this JSON:
      ```json
      {{
        "your_metric_score": <1-5 integer>,
        "explanation": "<explanation>"
      }}
      ```
    
    score_key: "your_metric_score"
    threshold: 3    # Minimum score for "passing"
    scale: 5        # Maximum possible score
```

### Judge Models Configuration

```yaml
judges:
  models:
    - "gpt-4o"              # Primary judge
    - "gpt-4o-mini"         # Additional judge for ensemble
  api:
    openai_api_key_env: "OPENAI_API_KEY"
    openai_base_url_env: "OPENAI_API_BASE"
    temperature: 0
    timeout: 60
```

### Composite Scoring

Combine multiple metrics into summary scores:

```yaml
composite_metrics:
  overall_quality:
    description: "Overall response quality"
    method: "weighted_average"  # Options: average, weighted_average, minimum
    metrics:
      - "accuracy"
      - "clarity"
      - "helpfulness"
    weights:  # Only for weighted_average
      accuracy: 0.4
      clarity: 0.3
      helpfulness: 0.3
```

## 📊 Built-in Metrics Library

The template includes ready-to-use metrics in `config/metric_templates.yaml`:

### Content Quality
- **Accuracy**: Factual correctness of responses
- **Completeness**: How thoroughly the response addresses the query
- **Relevance**: How relevant the response is to the input

### Style & Format
- **Clarity**: How clear and understandable the response is
- **Coherence**: Logical structure and consistency

### Specialized
- **Helpfulness**: How useful the response is to the user
- **Safety**: Whether the response is safe and appropriate
- **Creativity**: Originality and creative quality (for creative tasks)

### Templates
- **Binary Quality**: Simple pass/fail assessment
- **5-Point Scale**: Standard Likert scale evaluation
- **Ground Truth Comparison**: Compare against reference answers

## 🎯 Use Cases & Examples

### 1. Chatbot Quality Evaluation
Evaluate customer service chatbot responses for helpfulness, clarity, and safety.
```bash
python src/evaluation_template.py examples/chatbot_evaluation.yaml
```

### 2. Question Answering Systems
Compare Q&A system outputs against ground truth answers.
```bash
python src/evaluation_template.py examples/qa_evaluation.yaml
```

### 3. Creative Writing Assessment
Evaluate creative writing for style, creativity, and engagement.
```bash
python src/evaluation_template.py examples/creative_writing_evaluation.yaml
```

## 📈 MLflow Integration

The template automatically tracks experiments in MLflow:

- **Experiment Tracking**: All runs are organized by experiment name
- **Metric Logging**: Individual and composite scores are logged
- **Configuration Tracking**: Full configuration is saved with each run
- **Visualization**: Use MLflow UI to compare runs and visualize results

View your results:
```bash
mlflow ui
```

Then navigate to `http://localhost:5000` to explore your evaluation results.

## 🛠️ Customization Guide

### Creating Custom Metrics

1. **Define the Evaluation Criteria**: What specific aspect do you want to measure?

2. **Write the Prompt Template**: Create a clear prompt that instructs the LLM judge:
   ```yaml
   your_custom_metric:
     prompt_template: |
       You are evaluating {specific_aspect}.
       
       Materials: {prompt}, {response}
       
       Guidelines: Rate 1-5 where...
       
       Output: JSON with score and explanation
   ```

3. **Configure Scoring**: Set the score key, threshold, and scale.

4. **Test and Iterate**: Run on a small dataset first and refine.

### Adding New Data Sources

The template currently supports CSV files. To add other formats:

1. Extend the `DataLoader` class in `src/evaluation_template.py`
2. Add format-specific loading methods
3. Update the configuration schema

### Ensemble Evaluation

Use multiple judge models for more robust evaluation:
- **Discrete Scores**: Uses majority voting
- **Continuous Scores**: Uses averaging
- **Conflict Resolution**: Ties are broken by selecting the lower score (conservative approach)

## 🔍 Output Format

The template generates several output files:

1. **Detailed Results** (`evaluation_results_<timestamp>.csv`):
   - All individual scores and explanations
   - Composite scores
   - Pass/fail status for each metric

2. **Summary Statistics** (`evaluation_summary_<timestamp>.csv`):
   - Mean, std, min, max for each metric
   - Overall performance statistics

3. **MLflow Artifacts**:
   - Configuration files
   - Metric plots and visualizations

## 🚨 Best Practices

### Metric Design
- **Clear Criteria**: Make evaluation criteria as specific as possible
- **Balanced Scales**: Use consistent scaling across metrics
- **Examples**: Include examples in prompts when helpful
- **Validation**: Test metrics on known good/bad examples first

### Dataset Preparation
- **Quality Control**: Ensure your dataset is clean and representative
- **Size Considerations**: Start with smaller datasets for initial testing
- **Ground Truth**: Include reference answers when possible for accuracy metrics

### Judge Model Selection
- **Capability**: Choose models appropriate for your evaluation complexity
- **Consistency**: Use the same judge models across experiments for comparability
- **Cost**: Balance evaluation quality with API costs

### Threshold Setting
- **Domain-Specific**: Set thresholds based on your specific use case requirements
- **Validation**: Use held-out data to validate threshold settings
- **Business Impact**: Consider the real-world implications of false positives/negatives

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional ready-to-use metrics
- Support for other data formats
- Advanced visualization features
- Integration with other ML platforms

## 📝 License

This project is provided as-is for educational and research purposes. Please ensure compliance with your organization's policies and the terms of service of any APIs used.

## 🆘 Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure your OpenAI API key is correctly set
2. **Column Not Found**: Check that your dataset columns match the configuration
3. **JSON Parsing Errors**: Some judge models may return malformed JSON; consider adding retry logic
4. **Rate Limits**: Implement delays between API calls if hitting rate limits

### Getting Help

- Check the example configurations for working templates
- Review the metric templates library for standard patterns
- Ensure your prompt templates follow the expected format
- Test with a small dataset first to validate configuration