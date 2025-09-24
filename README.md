# Universal Evaluation Template System

A flexible, MLflow-integrated evaluation system that allows anyone to evaluate any dataset with custom metrics and scoring systems. This template system makes it easy to set up comprehensive evaluations without writing custom code.

## 🚀 Features

- **Universal Dataset Support**: Works with any CSV dataset containing prompts and responses
- **Custom Metrics**: Define your own evaluation metrics with natural language prompts
- **MLflow Integration**: Automatic logging and visualization of evaluation results
- **Ensemble Evaluation**: Use multiple judge models for robust evaluation
- **Domain-Specific Templates**: Pre-built metric templates for different domains (financial, medical, etc.)
- **Flexible Configuration**: YAML-based configuration system
- **Response Generation**: Support for various response generation methods

## 📁 Project Structure

```
template_evaluation_system/
├── template_evaluation_system.py    # Main evaluation system
├── metric_templates.py              # Pre-built metric templates
├── generate_config.py               # Configuration generator utility
├── example_usage.py                 # Usage examples
├── config_template.yaml             # Configuration template
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## 🛠️ Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up your OpenAI API key**:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   ```

## 📊 Quick Start

### 1. Prepare Your Dataset

Your dataset should be a CSV file with at least these columns:
- **prompt**: User questions or prompts
- **response**: Model responses to evaluate

Optional columns:
- **ground_truth**: Ground truth answers for accuracy evaluation
- **user_features**: User personalization features (JSON string)

Example dataset:
```csv
prompt,response,ground_truth,user_features
"What is the capital of France?","Paris is the capital of France.","Paris","{}"
"How do I bake a cake?","Mix ingredients and bake at 350°F for 30 minutes.","Mix and bake","{\"cooking_experience\": \"beginner\"}"
```

### 2. Generate Configuration

Use the configuration generator to create your evaluation setup:

```bash
python generate_config.py \
    --dataset data/your_dataset.csv \
    --prompt-col prompt \
    --response-col response \
    --experiment my_evaluation \
    --metrics accuracy relevance helpfulness \
    --output config.yaml
```

### 3. Run Evaluation

```bash
python template_evaluation_system.py --config config.yaml --output results/
```

## 🎯 Available Metrics

### General Metrics
- **accuracy**: Measures correctness against ground truth
- **relevance**: Evaluates how relevant the response is to the query
- **helpfulness**: Assesses how helpful the response is
- **safety**: Evaluates safety and appropriateness
- **factual_accuracy**: Checks for factual correctness
- **completeness**: Measures how complete the response is
- **clarity**: Evaluates clarity and understandability
- **personalization**: Assesses use of user personalization features

### Domain-Specific Metrics

#### Financial Advice
- **risk_assessment**: Evaluates risk communication
- **regulatory_compliance**: Checks regulatory compliance

#### Medical Advice
- **safety_first**: Prioritizes safety in medical advice

## 🔧 Configuration

### Basic Configuration

```yaml
dataset:
  path: "data/your_dataset.csv"
  prompt_column: "prompt"
  response_column: "response"
  ground_truth_column: "ground_truth"  # Optional
  user_features_column: "user_features"  # Optional

models:
  response_model: "GOLDEN_RESPONSE"  # or LLM_gpt-4o, LLM_gpt-4, FIRST_CALL
  judge_models: ["gpt-4o", "gpt-4"]

api_keys:
  openai_api_key: "your-openai-api-key"
  openai_base_url: "https://api.openai.com/v1"  # Optional

evaluation:
  experiment_name: "my_evaluation"
  run_name: "run_1"  # Optional
  metrics:
    accuracy:
      prompt_template: "Your evaluation prompt here..."
      threshold: 3.0
      output_schema:
        fields:
          accuracy_score: int
          explanation: str
```

### Custom Metrics

You can define custom metrics by adding them to your configuration:

```yaml
evaluation:
  metrics:
    my_custom_metric:
      prompt_template: |
        You are an impartial evaluator.
        Evaluate the response based on your custom criteria.
        
        User Query: {prompt}
        Model Response: {response}
        
        Rate from 1-5 where:
        1 = Poor
        5 = Excellent
        
        Return JSON: {{"my_custom_metric_score": <1-5>, "explanation": "<reasoning>"}}
      threshold: 3.0
      output_schema:
        fields:
          my_custom_metric_score: int
          explanation: str
```

## 📈 MLflow Integration

The system automatically logs evaluation results to MLflow:

1. **Experiment Tracking**: Each evaluation creates a new experiment
2. **Run Logging**: Individual runs are logged with metrics and parameters
3. **Artifact Storage**: Detailed evaluation results are stored as artifacts
4. **Dashboard**: View results in the MLflow UI

### Accessing Results

```python
import mlflow

# List experiments
experiments = mlflow.search_experiments()
print(experiments)

# Get specific run
run = mlflow.get_run("your_run_id")
print(run.data.metrics)
```

## 🎨 Examples

### Example 1: Basic Evaluation

```python
from template_evaluation_system import UniversalEvaluator, EvaluationConfig

# Load configuration
config = EvaluationConfig("config.yaml")

# Load dataset
import pandas as pd
df = pd.read_csv("data/your_dataset.csv")

# Run evaluation
evaluator = UniversalEvaluator(config)
results = evaluator.evaluate_dataset(df)

# View results
print(results[["prompt", "accuracy", "relevance"]].head())
```

### Example 2: Custom Metrics

```python
# Define custom metric
custom_metric = {
    "creativity": {
        "prompt_template": """
        Evaluate the creativity of the response.
        
        User Query: {prompt}
        Model Response: {response}
        
        Rate creativity from 1-5:
        1 = Not creative
        5 = Highly creative
        
        Return JSON: {{"creativity_score": <1-5>, "explanation": "<reasoning>"}}
        """,
        "threshold": 3.0,
        "output_schema": {
            "fields": {
                "creativity_score": "int",
                "explanation": "str"
            }
        }
    }
}

# Add to configuration
config.evaluation.metrics.update(custom_metric)
```

### Example 3: Domain-Specific Evaluation

```python
# Use financial domain metrics
config_dict = generate_config(
    dataset_path="financial_data.csv",
    prompt_column="prompt",
    response_column="response",
    experiment_name="financial_eval",
    metrics=["risk_assessment", "regulatory_compliance"],
    domain="financial_advice",
    user_features_column="user_profile"
)
```

## 🔍 Advanced Usage

### Response Generation

The system supports different response generation methods:

1. **GOLDEN_RESPONSE**: Use existing responses in your dataset
2. **LLM_gpt-4o**: Generate responses using GPT-4o
3. **LLM_gpt-4**: Generate responses using GPT-4
4. **FIRST_CALL**: Use external API for response generation

### Ensemble Evaluation

Use multiple judge models for more robust evaluation:

```yaml
models:
  judge_models: ["gpt-4o", "gpt-4", "claude-3-sonnet"]
```

The system will use majority voting for final scores.

### Custom API Integration

For external APIs, add configuration:

```yaml
api:
  base_url: "https://your-api-endpoint.com"
  headers:
    apikey: "your-api-key"
  timeout: 60.0
```

## 📊 Output Format

The system generates several output files:

1. **evaluation_results_TIMESTAMP.csv**: Detailed evaluation results
2. **evaluation_results_TIMESTAMP_summary.csv**: Summary statistics
3. **MLflow artifacts**: Detailed logs and metrics

### Result Columns

- **Original columns**: All original dataset columns
- **Metric scores**: Individual scores for each metric (e.g., `accuracy`, `relevance`)
- **Status indicators**: Pass/fail indicators (e.g., `accuracy_status`)
- **Details**: Detailed evaluation explanations (`{metric}_details`)

## 🛠️ Troubleshooting

### Common Issues

1. **Missing API Key**: Ensure `OPENAI_API_KEY` is set
2. **Column Not Found**: Check that your dataset has the required columns
3. **Model Errors**: Verify that your judge models are available
4. **Configuration Errors**: Validate your YAML configuration

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your metric templates to `metric_templates.py`
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For questions and support:
1. Check the examples in `example_usage.py`
2. Review the configuration template in `config_template.yaml`
3. Open an issue on GitHub

## 🔄 Migration from Databricks

If you're migrating from the original Databricks notebook:

1. Export your dataset as CSV
2. Convert your prompt templates to the new format
3. Update your configuration using the generator
4. Run the evaluation system

The new system provides the same functionality with better modularity and easier customization.