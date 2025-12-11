# Generalized LLM Evaluation System

A flexible, directory-based evaluation system for LLM responses that supports multiple metric types, auto-generated prompts, and various output formats.

## Features

- **Flexible Metrics Configuration**: Load metrics from CSV files with support for binary, scale (1-5), and percentage metrics
- **Auto-Generated Prompts**: Automatically generate evaluation prompts from grading rubrics
- **Predefined Scenarios**: Built-in evaluation scenarios for common use cases
- **Custom Scenarios**: Create your own evaluation scenarios with custom metrics
- **Multiple Output Formats**: Export results to CSV, JSON, and other formats
- **MLflow Integration**: Log evaluation results to MLflow for experiment tracking
- **CLI Interface**: Easy-to-use command-line interface for running evaluations

## Quick Start

### 1. Basic Usage with CSV Metrics

```python
from generalized_evaluator import GeneralizedEvaluator

# Create evaluator with CSV metrics configuration
evaluator = GeneralizedEvaluator(
    judge_model="gpt-4",
    metrics_config_path="my_metrics.csv"
)

# Define evaluation data
evaluation_data = [
    {
        'prompt': 'What is the capital of France?',
        'response': 'The capital of France is Paris.',
        'ground_truth': 'Paris'
    }
]

# Run evaluation
results_df = evaluator.evaluate_dataset(evaluation_data)

# View results
print(results_df)
```

### 2. Using Predefined Scenarios

```python
from evaluation_config_manager import EvaluationConfigManager

# Initialize configuration manager
config_manager = EvaluationConfigManager()

# Create predefined scenarios
config_manager.create_predefined_scenarios()

# List available scenarios
print(config_manager.list_scenarios())

# Create evaluator for a specific scenario
evaluator = config_manager.create_evaluator("financial_advisory")

# Run evaluation
results_df = evaluator.evaluate_dataset(evaluation_data)
```

### 3. Command Line Interface

```bash
# Create sample metrics CSV
python run_evaluation.py --create-sample-metrics

# Run evaluation with custom metrics
python run_evaluation.py --metrics-csv my_metrics.csv --data evaluation_data.json

# Run evaluation with predefined scenario
python run_evaluation.py --scenario financial_advisory --sample-data

# List available scenarios
python run_evaluation.py --list-scenarios
```

## Metrics Configuration

### CSV Format

Create a CSV file with the following columns:

| Column | Description | Required | Example |
|--------|-------------|----------|---------|
| `name` | Metric name | Yes | `accuracy` |
| `type` | Metric type | Yes | `binary`, `1-5_scale`, `percentage` |
| `description` | What the metric evaluates | Yes | `Evaluates if the response is factually accurate` |
| `grading_rubric` | Detailed grading criteria | No | `Check if the response contains correct information...` |
| `evaluation_prompt` | Custom evaluation prompt | No | `You are an expert evaluator...` |
| `threshold` | Pass/fail threshold | Yes | `0.5`, `3.0`, `70.0` |
| `ground_truth_column` | Column name for ground truth | Yes | `ground_truth` |
| `ground_truth_file_path` | Path to ground truth file | No | `ground_truth_data.csv` |

### Example CSV

```csv
name,type,description,grading_rubric,threshold,ground_truth_column,ground_truth_file_path
accuracy,binary,Evaluates if the response is factually accurate,"Check if the response contains correct information and facts. Look for any false statements or misleading information.",0.5,ground_truth,ground_truth_data.csv
completeness,1-5_scale,Evaluates how complete the response is,"Rate from 1-5: 1=very incomplete, missing key information; 2=mostly incomplete; 3=partially complete; 4=mostly complete; 5=completely addresses all aspects of the question.",3.0,ground_truth,ground_truth_data.csv
helpfulness,percentage,Evaluates how helpful the response is to the user,"Rate from 0-100%: Consider how actionable, clear, and useful the response is for the user's specific needs.",70.0,ground_truth,ground_truth_data.csv
```

## Predefined Scenarios

### 1. Basic Quality Assessment (`basic_quality`)

Evaluates general LLM response quality with metrics for:
- Accuracy (binary)
- Completeness (1-5 scale)
- Helpfulness (percentage)

### 2. Financial Advisory (`financial_advisory`)

Evaluates financial advisory responses with metrics for:
- Personalization accuracy (binary)
- Calculation accuracy (binary)
- Compliance (binary)
- Structured presentation (1-5 scale)

### 3. Customer Support (`customer_support`)

Evaluates customer support responses with metrics for:
- Empathy (1-5 scale)
- Problem solving (1-5 scale)
- Clarity (percentage)

## Creating Custom Scenarios

```python
from evaluation_config_manager import EvaluationConfigManager

# Initialize configuration manager
config_manager = EvaluationConfigManager()

# Define custom metrics
custom_metrics = [
    {
        'name': 'code_quality',
        'type': '1-5_scale',
        'description': 'Evaluates code quality and readability',
        'grading_rubric': 'Rate from 1-5: 1=very poor quality; 5=excellent quality',
        'threshold': 3.0,
        'ground_truth_column': 'quality_standards',
        'ground_truth_file_path': 'quality_standards.json'
    }
]

# Create custom scenario
config_manager.create_scenario(
    name="code_evaluation",
    description="Evaluation for code quality and correctness",
    metrics=custom_metrics
)
```

## Evaluation Data Format

Evaluation data should be a list of dictionaries with the following structure:

```python
evaluation_data = [
    {
        'prompt': 'The user question or prompt',
        'response': 'The LLM response to evaluate',
        'ground_truth': 'Expected answer or reference',
        # Additional fields can be included based on your metrics
        'user_profile': {...},  # For personalization metrics
        'context': {...},       # For context-aware metrics
    }
]
```

## Output Formats

The system exports results in multiple formats:

### 1. CSV Format
- Detailed results with all metric scores
- Easy to import into spreadsheets or analysis tools

### 2. JSON Format
- Structured data for programmatic access
- Includes metadata and configuration

### 3. Summary Statistics
- Overall pass rates and metrics
- Per-metric performance analysis

### 4. MLflow Logging
- Experiment tracking and comparison
- Parameter and metric logging
- Artifact storage

## Advanced Usage

### Custom Metric Types

You can extend the system to support additional metric types by:

1. Adding new values to the `MetricType` enum
2. Updating the `_simulate_llm_evaluation` method
3. Modifying the `auto_generate_evaluation_prompt` method

### Ground Truth Integration

The system supports multiple ground truth data sources:

- CSV files with column-based matching
- JSON files with structured data
- In-memory dictionaries
- Database connections (extensible)

### Batch Processing

Process multiple evaluation scenarios in parallel:

```python
scenarios = ["basic_quality", "financial_advisory", "customer_support"]
results = {}

for scenario in scenarios:
    evaluator = config_manager.create_evaluator(scenario)
    results[scenario] = evaluator.evaluate_dataset(evaluation_data)
```

## CLI Reference

### Basic Commands

```bash
# Create sample metrics CSV
python run_evaluation.py --create-sample-metrics

# List available scenarios
python run_evaluation.py --list-scenarios

# Run evaluation with CSV metrics
python run_evaluation.py --metrics-csv metrics.csv --data data.json

# Run evaluation with predefined scenario
python run_evaluation.py --scenario financial_advisory --sample-data

# Run with custom output directory
python run_evaluation.py --scenario basic_quality --data data.json --output-dir /path/to/results

# Run with verbose output
python run_evaluation.py --scenario basic_quality --data data.json --verbose
```

### Command Line Options

| Option | Description | Required |
|--------|-------------|----------|
| `--metrics-csv` | Path to CSV metrics configuration | No* |
| `--scenario` | Name of predefined scenario | No* |
| `--data` | Path to JSON evaluation data | No** |
| `--sample-data` | Use sample evaluation data | No** |
| `--judge-model` | LLM model for evaluation | No |
| `--output-dir` | Output directory for results | No |
| `--verbose` | Print detailed results | No |
| `--create-sample-metrics` | Create sample metrics CSV | No |
| `--list-scenarios` | List available scenarios | No |

*Either `--metrics-csv` or `--scenario` is required
**Either `--data` or `--sample-data` is required

## Examples

See `example_usage.py` for comprehensive examples of:
- Basic CSV-based evaluation
- Using predefined scenarios
- Creating custom scenarios
- Exporting results and MLflow logging
- Batch processing multiple scenarios

## Requirements

- Python 3.7+
- pandas
- mlflow
- pyyaml (optional, for YAML configuration support)

## Installation

```bash
pip install pandas mlflow pyyaml
```

## Contributing

To extend the system:

1. Add new metric types to `MetricType` enum
2. Implement evaluation logic in `GeneralizedEvaluator`
3. Create new predefined scenarios in `EvaluationConfigManager`
4. Add CLI options in `run_evaluation.py`

## License

This project is licensed under the MIT License - see the LICENSE file for details.