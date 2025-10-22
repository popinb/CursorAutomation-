# Directory Handling Guide for LLM Evaluation Framework

This guide explains how to use the generalized directory handling feature that replaces hardcoded paths with a flexible, configurable system.

## Overview

The directory handler provides:
- **Configurable paths** via JSON configuration files
- **Environment variable** support for overriding paths
- **Multi-environment support** (local, Databricks, cloud storage)
- **User-specific directories** for multi-user scenarios
- **Automatic directory creation** (optional)
- **Timestamped file naming** utilities
- **Path resolution** for relative and absolute paths

## Quick Start

### 1. Basic Usage

```python
from directory_handler import get_directory_handler

# Get default directory handler
dir_handler = get_directory_handler()

# Get directory paths
data_dir = dir_handler.get_dir('data')
results_dir = dir_handler.get_dir('results')

# Create a timestamped file
result_file = dir_handler.get_timestamped_file_path('results', 'evaluation', '.csv')
```

### 2. With Configuration File

Create a `config.json` file:

```json
{
  "directories": {
    "data": "path/to/data",
    "results": "path/to/results",
    "reports": "path/to/reports"
  },
  "auto_create_dirs": true
}
```

Use it in your code:

```python
dir_handler = get_directory_handler('config.json')
```

### 3. With Environment Variables

Set environment variables to override default paths:

```bash
export WORKSPACE_ROOT=/path/to/workspace
export LLM_EVAL_DATA_DIR=/path/to/data
export LLM_EVAL_RESULTS_DIR=/path/to/results
```

## Directory Types

The framework supports these directory types:

| Directory Type | Purpose | Default Location |
|---------------|---------|------------------|
| `data` | Input data files | `{workspace}/data` |
| `results` | Evaluation results | `{workspace}/results` |
| `logs` | Log files | `{workspace}/logs` |
| `configs` | Configuration files | `{workspace}/configs` |
| `metrics` | Metric definitions | `{workspace}/metrics` |
| `ground_truth` | Ground truth data | `{workspace}/ground_truth` |
| `reports` | HTML/PDF reports | `{workspace}/reports` |
| `artifacts` | MLflow artifacts | `{workspace}/artifacts` |
| `mlflow` | MLflow runs | `{workspace}/mlflow_runs` |
| `temp` | Temporary files | `{workspace}/temp` |

## Configuration Methods

### 1. JSON Configuration File

```json
{
  "directories": {
    "data": "/absolute/path/to/data",
    "results": "./relative/path/to/results",
    "reports": "~/user/home/reports"
  },
  "auto_create_dirs": true,
  "framework": {
    "judge_model": "gpt-4",
    "mlflow_enabled": true
  }
}
```

### 2. Environment Variables

All directory paths can be overridden using environment variables:

- `WORKSPACE_ROOT` - Base workspace directory
- `LLM_EVAL_DATA_DIR` - Data directory
- `LLM_EVAL_RESULTS_DIR` - Results directory
- `LLM_EVAL_LOGS_DIR` - Logs directory
- `LLM_EVAL_CONFIGS_DIR` - Configs directory
- `LLM_EVAL_METRICS_DIR` - Metrics directory
- `LLM_EVAL_GROUND_TRUTH_DIR` - Ground truth directory
- `LLM_EVAL_REPORTS_DIR` - Reports directory
- `LLM_EVAL_ARTIFACTS_DIR` - Artifacts directory
- `LLM_EVAL_MLFLOW_DIR` - MLflow directory
- `LLM_EVAL_TEMP_DIR` - Temp directory

### 3. Programmatic Configuration

```python
from directory_handler import DirectoryHandler

# Create custom directory handler
dir_handler = DirectoryHandler(auto_create_dirs=True)

# Override specific directories
dir_handler.dirs['data'] = '/custom/data/path'
dir_handler.dirs['results'] = '/custom/results/path'

# Resolve paths (converts relative to absolute)
dir_handler._resolve_paths()
```

## Environment-Specific Examples

### Local Development

```json
{
  "directories": {
    "data": "./data",
    "results": "./results",
    "reports": "./reports"
  }
}
```

### Databricks

```json
{
  "directories": {
    "data": "/dbfs/FileStore/llm_evaluation/data",
    "results": "/dbfs/FileStore/llm_evaluation/results",
    "logs": "/Workspace/Users/${USER}/logs",
    "reports": "/Workspace/Users/${USER}/reports",
    "artifacts": "/dbfs/FileStore/llm_evaluation/artifacts"
  },
  "auto_create_dirs": false
}
```

### AWS S3

```bash
export LLM_EVAL_DATA_DIR=s3://my-bucket/llm-evaluation/data
export LLM_EVAL_RESULTS_DIR=s3://my-bucket/llm-evaluation/results
export LLM_EVAL_ARTIFACTS_DIR=s3://my-bucket/llm-evaluation/artifacts
```

### Azure Blob Storage

```bash
export LLM_EVAL_DATA_DIR=https://myaccount.blob.core.windows.net/llm-eval/data
export LLM_EVAL_RESULTS_DIR=https://myaccount.blob.core.windows.net/llm-eval/results
```

## User-Specific Paths

Support multiple users with isolated directories:

```python
# Get user-specific directory
user_results = dir_handler.get_user_specific_path('results', username='alice')
# Creates: {results_dir}/alice/

# Get MLflow experiment path
mlflow_path = dir_handler.get_mlflow_experiment_path(
    experiment_name='my_experiment',
    username='alice@company.com'
)
# Returns: /Users/alice@company.com/my_experiment
```

## Utility Functions

### Timestamped Files

```python
# Create timestamped filename
file_path = dir_handler.get_timestamped_file_path(
    'results',
    'evaluation',
    '.csv'
)
# Returns: /path/to/results/evaluation_20231014_153022.csv
```

### List Files

```python
# List all CSV files in data directory
csv_files = dir_handler.list_files('data', '*.csv')

# List all files recursively
all_files = dir_handler.list_files('data', '*', recursive=True)
```

### Clean Temporary Files

```python
# Remove temp files older than 24 hours
removed_count = dir_handler.clean_temp_files(older_than_hours=24)
```

### Path Resolution

```python
# Resolve relative path
abs_path = dir_handler.resolve_path('subdir/file.txt', relative_to='data')

# Resolve absolute path (returns as-is)
abs_path = dir_handler.resolve_path('/absolute/path/file.txt')
```

## Integration with Evaluation Framework

### Using with LLMEvaluationFramework

```python
from llm_evaluation_framework import LLMEvaluationFramework

# Initialize with config
framework = LLMEvaluationFramework(config_path='config.json')

# Run evaluation - automatically uses configured directories
results = framework.run_evaluation(
    evaluation_data='test_data.csv',  # Looks in data directory
    metrics_config='metrics.csv'      # Looks in configs directory
)
# Results saved to configured results directory
```

### Using with ZillowEvalsWrapperV2

```python
from evals_wrapper_v2 import ZillowEvalsWrapperV2

# Initialize wrapper
wrapper = ZillowEvalsWrapperV2(config_path='config.json')

# Evaluate - results automatically saved to configured directory
result = wrapper.evaluate_response(
    candidate_answer="...",
    question="...",
    save_results=True  # Saves to results directory
)
```

## Migration from Hardcoded Paths

Replace hardcoded paths like:

```python
# Old way
results_path = "/Workspace/Users/user@company.com/results.csv"
report_path = "/Workspace/Users/user@company.com/llm_evaluation_report_20251014_190922.html"
```

With:

```python
# New way
dir_handler = get_directory_handler()
results_path = dir_handler.get_timestamped_file_path('results', 'evaluation', '.csv')
report_path = dir_handler.get_timestamped_file_path('reports', 'llm_evaluation_report', '.html')
```

## Best Practices

1. **Use configuration files** for production deployments
2. **Use environment variables** for sensitive paths or cloud deployments
3. **Enable auto_create_dirs** for local development, disable for production
4. **Use timestamped files** for results to avoid overwrites
5. **Use user-specific paths** in multi-user environments
6. **Clean temporary files** regularly to save disk space
7. **Use relative paths** in config files for portability

## Troubleshooting

### Directories not created

Check if `auto_create_dirs` is set to `true` in your configuration.

### Permission errors

Ensure the process has write permissions to the parent directories.

### Path not found

Use `dir_handler.get_config_summary()` to verify your configuration.

### Environment variables not working

Ensure variables are exported (not just set) and start with `LLM_EVAL_`.

## Example Run Script

```bash
#!/bin/bash

# Set environment for production
export WORKSPACE_ROOT=/data/llm_evaluation
export LLM_EVAL_DATA_DIR=/data/llm_evaluation/prod/data
export LLM_EVAL_RESULTS_DIR=/data/llm_evaluation/prod/results
export LLM_EVAL_MLFLOW_DIR=/data/mlflow

# Run evaluation
python run_evaluation.py --config production.json
```

This generalized directory system makes the evaluation framework portable across different environments while maintaining consistency and ease of use.