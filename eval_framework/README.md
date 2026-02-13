# Evaluation Framework

A flexible, template-based evaluation framework for assessing AI model responses using LLM judges and MLflow tracking.

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare your data:**
   - Create CSV files with your evaluation data
   - Required columns: prompt/question and response/answer
   - Optional: additional context columns

3. **Configure your evaluation:**
   - Copy `config_template.yaml` to create your configuration
   - Define your metrics using natural language prompts
   - Set up judge models and scoring thresholds

4. **Run evaluation:**
   ```python
   from evaluation_framework import run_evaluation
   results = run_evaluation("your_config.yaml")
   ```

## 📁 Project Structure

```
eval_framework/
├── evaluation_framework.py    # Main framework code
├── config_template.yaml       # Configuration template
├── evaluation_template.ipynb  # Jupyter notebook interface
├── notebook_converter.py      # Convert existing notebooks
├── examples/                  # Example configurations
│   ├── personalization_eval_config.yaml
│   ├── simple_qa_config.yaml
│   └── custom_scoring_config.yaml
└── README.md                  # This file
```

## 🔧 Configuration Guide

### Basic Structure

```yaml
experiment:
  name: "your_experiment_name"
  
data:
  ground_truth_dir: "./your_data/"
  columns:
    prompt: "question_column"
    response: "answer_column"
    
models:
  judge_models:
    - "gpt-4o"
    
metrics:
  - name: "your_metric"
    prompt_template: |
      Your evaluation prompt here...
    score_range: [1, 5]
    threshold: 3
```

### Defining Metrics

Metrics are defined using natural language prompts. The framework automatically:
- Formats prompts with your data
- Calls LLM judges to evaluate
- Parses scores from JSON responses
- Tracks results in MLflow

Example metric definition:
```yaml
- name: "accuracy"
  description: "Evaluates response accuracy"
  prompt_template: |
    Evaluate if this answer is accurate.
    
    Question: {prompt}
    Answer: {response}
    
    Score from 1-5 where 5 is most accurate.
    
    Return JSON: {"accuracy_score": <score>, "explanation": "<reason>"}
  
  score_range: [1, 5]
  threshold: 4
```

### Scoring Systems

The framework supports various scoring approaches:

- **Binary (0/1)**: Pass/fail evaluations
- **Scale (1-5, 1-10)**: Graduated scoring
- **Percentage (0-100)**: Percentage-based metrics
- **Custom ranges**: Any numeric range you define

### Data Format

Your CSV files should include:
- **Required**: Prompt/question column and response/answer column
- **Optional**: Additional context columns (user info, metadata, etc.)

Example CSV structure:
```csv
prompt,response,user_context
"What is 2+2?","The answer is 4","math_student"
"Explain photosynthesis","Plants convert light...","biology_teacher"
```

## 🎯 Features

### 1. Flexible Metric Definition
- Define metrics using natural language
- No coding required for new metrics
- Supports any scoring system

### 2. Ensemble Evaluation
- Use multiple judge models
- Automatic score aggregation
- Increased reliability

### 3. MLflow Integration
- Automatic experiment tracking
- Metric visualization
- Result comparison across runs

### 4. Batch Processing
- Process multiple CSV files
- Parallel evaluation support
- Efficient handling of large datasets

### 5. Detailed Analysis
- Score distributions
- Failed case analysis
- Exportable results

## 📊 Viewing Results

### In Notebook
The evaluation notebook provides:
- Summary statistics
- Score distributions
- Failed case analysis
- Sample results

### In MLflow
```bash
mlflow ui --backend-store-uri file://./mlruns
```

Then navigate to http://localhost:5000 to see:
- Experiment runs
- Metric comparisons
- Detailed logs
- Artifacts

## 🔄 Converting Existing Notebooks

If you have an existing evaluation notebook, use the converter:

```python
from notebook_converter import convert_notebook_to_config
convert_notebook_to_config("old_notebook.ipynb", "new_config.yaml")
```

Note: You'll need to manually extract prompt templates and variables.

## 📝 Example Use Cases

### 1. Q&A Evaluation
```yaml
metrics:
  - name: "correctness"
    prompt_template: |
      Is this answer correct?
      Question: {prompt}
      Answer: {response}
      Score 1-5...
```

### 2. Personalization Check
```yaml
metrics:
  - name: "personalization"
    prompt_template: |
      Does this response use the user's personal info?
      User Info: {user_context}
      Response: {response}
      Score 1-5...
```

### 3. Safety Evaluation
```yaml
metrics:
  - name: "safety"
    prompt_template: |
      Is this response safe and appropriate?
      Response: {response}
      Score 1 for safe, 0 for unsafe...
```

## 🤝 Contributing

To add new features:
1. Extend the `MetricDefinition` class for new metric types
2. Add new evaluator types by extending `BaseEvaluator`
3. Create new example configurations

## 📄 License

This framework is provided as-is for evaluation purposes.

## 🆘 Troubleshooting

### Common Issues

1. **Missing columns error**: Ensure your CSV has the columns specified in config
2. **API errors**: Check your OpenAI API key and rate limits
3. **Memory issues**: Process large datasets in batches

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🚧 Roadmap

- [ ] Support for more LLM providers
- [ ] Real-time evaluation dashboard
- [ ] Automatic metric suggestion
- [ ] Statistical significance testing
- [ ] Export to various formats