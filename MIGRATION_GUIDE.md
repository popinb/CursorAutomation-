# Migration Guide: From Databricks Notebook to Universal Template

This guide helps you migrate from the original Databricks notebook to the new Universal Evaluation Template System.

## 🔄 What Changed

### Before (Databricks Notebook)
- Hardcoded for specific dataset structure
- Fixed evaluation metrics
- Databricks-specific dependencies
- Manual configuration in code

### After (Universal Template)
- Works with any CSV dataset
- Configurable metrics via YAML
- Platform-agnostic
- Easy configuration generation

## 📋 Migration Steps

### 1. Export Your Dataset

From your Databricks notebook, export your dataset as CSV:

```python
# In Databricks
df.write.mode("overwrite").option("header", "true").csv("dbfs:/FileStore/your_dataset.csv")

# Download to local machine
dbutils.fs.cp("dbfs:/FileStore/your_dataset.csv", "file:/tmp/your_dataset.csv")
```

### 2. Convert Your Prompt Templates

#### Original Databricks Format:
```python
PERSONALIZATION_ACCURACY_PROMPT = """You are an impartial evaluator.
Your task is to decide whether the AI assistant's answer **accurately incorporates the user's personalized information**...

### Materials
* **User Query:**
  ```
  {prompt}  
  ```

* **User Personalization Features:**
  ```
  {user_personalization_features}  
  ```

* **AI Response:**
  ```
  {response}  
  ```

### Evaluation Guidelines
...
"""
```

#### New Template Format:
```yaml
evaluation:
  metrics:
    personalization_accuracy:
      prompt_template: |
        You are an impartial evaluator.
        Your task is to decide whether the AI assistant's answer **accurately incorporates the user's personalized information**...
        
        User Query: {prompt}
        User Personalization Features: {user_personalization_features}
        AI Response: {response}
        
        Rate from 1-5 where:
        1 = Poor
        5 = Excellent
        
        Return JSON: {{"personalization_accuracy_score": <1-5>, "explanation": "<reasoning>"}}
      threshold: 3.0
      output_schema:
        fields:
          personalization_accuracy_score: int
          explanation: str
```

### 3. Generate Configuration

Use the configuration generator:

```bash
python generate_config.py \
    --dataset your_dataset.csv \
    --prompt-col prompt \
    --response-col response \
    --experiment your_experiment \
    --metrics personalization_accuracy context_personalization general_personalization \
    --user-features-col user_personalization_features \
    --output your_config.yaml
```

### 4. Update API Keys

Edit your configuration file to include your API keys:

```yaml
api_keys:
  openai_api_key: "your-actual-openai-key"
  openai_base_url: "https://api.openai.com/v1"  # or your custom endpoint
```

### 5. Run Evaluation

```bash
python template_evaluation_system.py --config your_config.yaml --output results/
```

## 🔧 Key Differences

### Dataset Structure
- **Before**: Fixed column names (`prompt`, `response`, `user_personalization_features`)
- **After**: Configurable column names via YAML

### Metric Definition
- **Before**: Python variables with hardcoded templates
- **After**: YAML configuration with structured schemas

### Model Configuration
- **Before**: Hardcoded model lists
- **After**: Configurable via YAML

### MLflow Integration
- **Before**: Manual MLflow calls
- **After**: Automatic integration with configurable experiments

## 📊 Feature Mapping

| Databricks Feature | Template Equivalent |
|-------------------|-------------------|
| `PERSONALIZATION_ACCURACY_PROMPT` | `personalization_accuracy` metric |
| `CONTEXT_PERSONALIZATION_PROMPT` | `context_personalization` metric |
| `GENERAL_PERSONALIZATION_PROMPT` | `general_personalization` metric |
| `JUDGE_MODELS` | `models.judge_models` in config |
| `EXPERIMENT_NAME` | `evaluation.experiment_name` in config |
| `RESPONSE_MODEL` | `models.response_model` in config |

## 🎯 Benefits of Migration

1. **Flexibility**: Works with any dataset structure
2. **Maintainability**: YAML configuration is easier to manage
3. **Reusability**: Templates can be shared across projects
4. **Extensibility**: Easy to add new metrics and domains
5. **Platform Independence**: Works anywhere Python runs

## 🚀 Quick Migration Example

### Step 1: Prepare Your Data
```python
# Export from Databricks
df = spark.sql("SELECT * FROM your_table").toPandas()
df.to_csv("migrated_dataset.csv", index=False)
```

### Step 2: Generate Config
```bash
python generate_config.py \
    --dataset migrated_dataset.csv \
    --prompt-col prompt \
    --response-col response \
    --experiment migrated_evaluation \
    --metrics accuracy relevance personalization \
    --user-features-col user_personalization_features \
    --output migrated_config.yaml
```

### Step 3: Run Evaluation
```bash
python template_evaluation_system.py --config migrated_config.yaml
```

### Step 4: View Results
```bash
mlflow ui
# Open http://localhost:5000 in your browser
```

## 🔍 Troubleshooting

### Common Issues

1. **Column Not Found**: Check your column names in the configuration
2. **API Key Missing**: Set `OPENAI_API_KEY` environment variable
3. **Model Not Available**: Check your `openai_base_url` configuration
4. **YAML Syntax Error**: Validate your configuration file

### Getting Help

1. Run the test suite: `python test_template.py`
2. Check examples: `python example_usage.py`
3. Use interactive setup: `python quick_start.py`
4. Review documentation: `README.md`

## 📈 Performance Comparison

| Aspect | Databricks | Template System |
|--------|------------|-----------------|
| Setup Time | 30+ minutes | 5 minutes |
| Configuration | Code changes | YAML file |
| Reusability | Low | High |
| Maintenance | High | Low |
| Platform Lock-in | High | None |

## 🎉 Next Steps

After migration:

1. **Customize Metrics**: Add domain-specific metrics
2. **Automate**: Set up CI/CD pipelines
3. **Scale**: Run evaluations on larger datasets
4. **Monitor**: Use MLflow for experiment tracking
5. **Share**: Distribute templates across teams

The new system provides all the functionality of the original Databricks notebook with much greater flexibility and ease of use!