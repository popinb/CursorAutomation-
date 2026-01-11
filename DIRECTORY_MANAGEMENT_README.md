# Generalized Directory Management for LLM Evaluation

This system replaces hardcoded directory paths with a flexible, environment-aware directory management system that works seamlessly across local development, Databricks, cloud environments, and containers.

## 🎯 Problem Solved

**Before (Hardcoded Paths):**
```python
# ❌ This breaks in different environments
out_dir = "/tmp/llm_eval_artifacts"
csv_path = os.path.join(out_dir, f"results_{int(time.time())}.csv")
mlflow.set_experiment(f"/Users/{username}/llm_evaluation_experiment")
```

**After (Generalized Directory Management):**
```python
# ✅ This works everywhere
dm = create_directory_manager()  # Auto-detects environment
csv_path = dm.get_output_path("results.csv")
mlflow.set_experiment(dm.get_mlflow_experiment_path())
```

## 🚀 Quick Start

### Basic Usage

```python
from directory_manager import create_directory_manager

# Auto-detect environment and create appropriate directory structure
with create_directory_manager() as dm:
    # Get paths for different file types
    output_file = dm.get_output_path("results.csv")
    log_file = dm.get_log_path("evaluation.log")
    temp_file = dm.get_temp_path("processing", ".tmp")
    
    # Load assets (automatically finds correct path)
    golden_responses = dm.get_asset_path("golden_responses")
    
    # MLflow integration
    experiment_path = dm.get_mlflow_experiment_path()
```

### Environment-Specific Setup

```python
from directory_manager import get_local_manager, get_databricks_manager, get_cloud_manager

# Local development
local_dm = get_local_manager(workspace_root="./my_project")

# Databricks
databricks_dm = get_databricks_manager(username="user@company.com")

# Cloud/Container
cloud_dm = get_cloud_manager(workspace_root="/app")
```

## 🏗️ Architecture

### Core Components

1. **DirectoryManager**: Main class that handles all directory operations
2. **DirectoryConfig**: Configuration dataclass for customizing behavior
3. **Environment Detection**: Automatic detection of runtime environment
4. **Path Resolution**: Smart path resolution based on environment

### Supported Environments

| Environment | Detection Method | Default Paths |
|-------------|------------------|---------------|
| **Local** | Default fallback | `./workspace`, `./assets`, `./outputs` |
| **Databricks** | `/databricks` exists or `DATABRICKS_RUNTIME_VERSION` env var | `/Workspace/Users/{user}`, `/tmp/llm_eval_artifacts` |
| **Cloud** | AWS/Azure/GCP env vars | `/app`, `/tmp/outputs` |
| **Container** | `/.dockerenv` exists | `/workspace`, `/workspace/outputs` |

## 📁 Directory Structure

The system creates and manages these directories:

```
workspace_root/
├── assets/                 # Ground truth data, configs
│   ├── golden_responses.json
│   ├── buyability_profiles.json
│   └── fair_housing_guide.json
├── outputs/               # Evaluation results
│   ├── results_20241022_143022.csv
│   └── logs/             # Log files
└── temp/                 # Temporary processing files
```

## 🔧 Configuration Options

### Basic Configuration

```python
from directory_manager import DirectoryConfig, DirectoryManager

config = DirectoryConfig(
    workspace_root="/my/workspace",
    output_filename_pattern="results_{timestamp}.csv",
    environment="local",
    create_missing_dirs=True,
    cleanup_temp_on_exit=True
)

dm = DirectoryManager(config)
```

### Advanced Configuration

```python
config = DirectoryConfig(
    workspace_root="/custom/workspace",
    assets_dir="/shared/assets",
    output_dir="/fast/storage/outputs",
    temp_dir="/tmp/processing",
    
    # File naming patterns
    output_filename_pattern="eval_results_{timestamp}.csv",
    log_filename_pattern="eval_log_{timestamp}.log",
    
    # MLflow settings
    mlflow_experiment_path="/experiments/llm_evaluation",
    mlflow_artifact_path="results",
    
    # Asset file mappings
    asset_files={
        "golden_responses": "golden_responses.json",
        "buyability_profiles": "buyability_profiles.json",
        "metrics_config": "metrics_config.csv"
    }
)
```

## 🔄 Migration Guide

### Step 1: Identify Hardcoded Paths

Find hardcoded paths in your code:
```python
# ❌ Hardcoded paths to replace
out_dir = "/tmp/llm_eval_artifacts"
assets_path = "./assets/golden_responses.json"
mlflow_experiment = "/Users/user@company.com/llm_evaluation_experiment"
```

### Step 2: Replace with Directory Manager

```python
# ✅ Generalized approach
from directory_manager import create_directory_manager

dm = create_directory_manager()

# Replace hardcoded paths
out_dir = dm.output_dir
assets_path = dm.get_asset_path("golden_responses")
mlflow_experiment = dm.get_mlflow_experiment_path()
```

### Step 3: Update File Operations

```python
# ❌ Before
csv_path = os.path.join("/tmp/llm_eval_artifacts", f"results_{int(time.time())}.csv")
results_df.to_csv(csv_path, index=False)

# ✅ After  
csv_path = dm.get_output_path("results.csv")
results_df.to_csv(csv_path, index=False)
```

### Step 4: Use Migration Helper

```python
from config_examples import migrate_hardcoded_paths

# Define your old hardcoded paths
old_paths = {
    'output_dir': '/tmp/llm_eval_artifacts',
    'assets_dir': './assets',
    'mlflow_experiment': '/Users/user@company.com/llm_evaluation_experiment'
}

# Create compatible directory manager
dm = migrate_hardcoded_paths(old_paths)
```

## 🌍 Environment Examples

### Local Development

```python
from directory_manager import get_local_manager

# Setup for local development
dm = get_local_manager(workspace_root="./my_project")

# All paths will be relative to ./my_project
output_file = dm.get_output_path("results.csv")
# → ./my_project/outputs/results_20241022_143022.csv
```

### Databricks

```python
from directory_manager import get_databricks_manager

# Setup for Databricks
dm = get_databricks_manager(username="user@company.com")

# Paths optimized for Databricks
output_file = dm.get_output_path("results.csv")
# → /tmp/llm_eval_artifacts/results_20241022_143022.csv

experiment = dm.get_mlflow_experiment_path()
# → /Users/user@company.com/llm_evaluation_experiment
```

### Docker Container

```python
from directory_manager import get_cloud_manager

# Setup for containerized deployment
dm = get_cloud_manager(workspace_root="/app")

# Container-appropriate paths
output_file = dm.get_output_path("results.csv")
# → /app/outputs/results_20241022_143022.csv
```

## 🧪 Testing

### Test Configuration

```python
from config_examples import get_testing_config
from directory_manager import DirectoryManager

# Isolated test environment
config = get_testing_config("/tmp/test_eval")
dm = DirectoryManager(config)

# All operations isolated to /tmp/test_eval
```

### Validation

```python
# Validate environment setup
validation = dm.validate_environment()

print(f"Environment: {validation['environment']}")
print(f"Workspace exists: {validation['workspace_root']['exists']}")
print(f"Assets readable: {validation['assets_dir']['readable']}")

# Check specific assets
for asset_name, asset_info in validation['assets'].items():
    status = "✅" if asset_info['exists'] else "❌"
    print(f"{asset_name}: {status} {asset_info['path']}")
```

## 📊 Integration with Evaluation Pipeline

### Complete Example

```python
from generalized_evaluation import run_evaluation_pipeline
from directory_manager import create_directory_manager
import pandas as pd

# Create evaluation data
eval_data = pd.DataFrame({
    'prompt': ['What is 2+2?'],
    'response': ['2+2 equals 4'],
    'ground_truth': ['4']
})

metrics_config = pd.DataFrame({
    'name': ['Accuracy'],
    'type': ['binary'],
    'description': ['Is the answer correct?'],
    'grading_rubric': ['Score 1 if correct, 0 otherwise'],
    'threshold': [1.0],
    'ground_truth_column': ['ground_truth'],
    'ground_truth_file_path': ['']
})

# Run with auto-detected environment
results = run_evaluation_pipeline(
    evaluation_data=eval_data,
    metrics_config_data=metrics_config,
    judge_model="gpt-4"
)

# Results automatically saved to appropriate location
print(f"Results saved with {len(results)} evaluations")
```

## 🔍 Troubleshooting

### Common Issues

1. **FileNotFoundError for assets**
   ```python
   # Check if assets exist
   validation = dm.validate_environment()
   print(validation['assets'])
   ```

2. **Permission errors**
   ```python
   # Check directory permissions
   validation = dm.validate_environment()
   print(f"Output writable: {validation['output_dir']['writable']}")
   ```

3. **Environment detection issues**
   ```python
   # Force specific environment
   dm = create_directory_manager(environment="local")
   ```

### Debug Information

```python
# Get comprehensive environment info
validation = dm.validate_environment()
import json
print(json.dumps(validation, indent=2))
```

## 🎯 Benefits

| Benefit | Description |
|---------|-------------|
| **🌍 Cross-platform** | Works on Windows, macOS, Linux |
| **☁️ Multi-environment** | Local, Databricks, AWS, Azure, GCP |
| **🔧 Configurable** | Customize paths and behavior |
| **🧪 Testable** | Easy to test with isolated configs |
| **📁 Auto-setup** | Creates directories automatically |
| **🔒 Safe** | Handles missing files gracefully |
| **🚀 Performance** | Environment-specific optimizations |

## 📚 API Reference

### DirectoryManager Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `get_asset_path(name)` | Get path to asset file | `Path` |
| `get_output_path(filename)` | Get path for output file | `Path` |
| `get_temp_path(prefix, suffix)` | Get temporary file path | `Path` |
| `get_log_path(name)` | Get path for log file | `Path` |
| `get_mlflow_experiment_path()` | Get MLflow experiment path | `str` |
| `validate_environment()` | Validate setup | `Dict` |
| `cleanup_temp_files()` | Clean up temp files | `None` |

### Factory Functions

| Function | Description |
|----------|-------------|
| `create_directory_manager()` | Auto-detect environment |
| `get_local_manager()` | Local development setup |
| `get_databricks_manager()` | Databricks setup |
| `get_cloud_manager()` | Cloud/container setup |

---

## 🤝 Contributing

To extend the directory management system:

1. Add new environment detection in `_detect_environment()`
2. Create environment-specific defaults in `_apply_*_defaults()`
3. Add configuration templates in `config_examples.py`
4. Update documentation and tests

## 📄 License

This directory management system is part of the LLM evaluation framework and follows the same license terms.