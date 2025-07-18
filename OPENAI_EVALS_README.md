# OpenAI Evals Framework Implementation: Zillow Home Affordability Judge

This document explains the **completely separate** OpenAI Evals framework implementation for the Zillow Home Affordability Judge, which is distinct from the custom LLM-as-a-judge implementation.

## 📁 Files Overview

### Core Implementation Files
- **`openaieval_zillow_judge.py`** - Full OpenAI Evals framework integration (requires `evals` package)
- **`standalone_openai_evals.py`** - Standalone implementation following OpenAI Evals patterns
- **`run_openai_evals.py`** - CLI script with both programmatic and CLI execution modes

### Configuration Files  
- **`zillow_eval_registry.yaml`** - OpenAI Evals registry configuration
- **`zillow_eval_samples.jsonl`** - Evaluation samples in JSONL format

### Documentation
- **`CREATE_CUSTOM_GPT_INSTRUCTIONS.md`** - Instructions for creating ChatGPT Custom GPT
- **`OPENAI_EVALS_README.md`** - This comprehensive guide

## 🔄 Implementation Approaches

### 1. Full OpenAI Evals Framework (`openaieval_zillow_judge.py`)

**Requirements:**
```bash
pip install evals
```

**Features:**
- ✅ Complete OpenAI Evals framework integration
- ✅ Uses `Eval` base class and framework patterns
- ✅ Compatible with `oaieval` CLI command
- ✅ Registry-based configuration
- ✅ Built-in recording and logging
- ✅ Framework-level parallelization

**Usage:**
```bash
# Via CLI (requires evals installation)
oaieval gpt-4 zillow-affordability --registry_path .

# Via Python script
python run_openai_evals.py --mode cli --model gpt-4
```

### 2. Standalone OpenAI Evals-Compatible (`standalone_openai_evals.py`)

**Requirements:**
```bash
pip install openai  # Only standard OpenAI library
```

**Features:**
- ✅ OpenAI Evals framework patterns and structure
- ✅ No additional dependencies beyond OpenAI
- ✅ Compatible result formats and logging
- ✅ Deterministic evaluation (seed=42)
- ✅ Structured recording with timestamps
- ✅ Framework-style configuration

**Usage:**
```bash
# Direct execution
python standalone_openai_evals.py

# Via CLI script
python run_openai_evals.py --mode programmatic
```

## 🎯 Evaluation Framework

### 12-Metric Evaluation Criteria

Both implementations use the same comprehensive evaluation framework:

1. **Personalization Accuracy** (Boolean: Accurate/Inaccurate)
2. **Context-based Personalization** (Scale: 1-5)  
3. **Next Step Identification** (Boolean: Present/Not Present)
4. **Assumption Listing** (Boolean: True/False)
5. **Assumption Trust** (Scale: 1-5)
6. **Calculation Accuracy** (Boolean: True/False)
7. **Faithfulness to Ground Truth** (Boolean: True/False)
8. **Overall Accuracy** (Boolean: True/False)
9. **Structured Presentation** (Scale: 1-5)
10. **Coherence** (Boolean: True/False)
11. **Completeness** (Scale: 1-5)
12. **Fair Housing Classifier** (Boolean: True/False)

### Scoring System

- **Boolean metrics**: True/Accurate/Present = 10 points, False/Inaccurate/Not Present = 0 points
- **Scale metrics**: 1-5 scale converted to 2,4,6,8,10 points
- **Alpha scoring**: Excludes Completeness and Structured Presentation (10 metrics, 100 points max)
- **Full scoring**: Includes all 12 metrics (120 points max)

## 🚀 Quick Start

### Option 1: Standalone (Recommended for Testing)

```bash
# 1. Ensure API credentials are set
export OPENAI_API_KEY="popinb_zillowlabs__hs7x0vTjbLwjKhNStdgL1Dd"
export OPENAI_BASE_URL="https://api.zillowlabs.com/openai/v1"

# 2. Run standalone evaluation
python standalone_openai_evals.py
```

### Option 2: Full OpenAI Evals Framework

```bash
# 1. Install OpenAI Evals
pip install evals

# 2. Run with CLI integration
python run_openai_evals.py --mode cli --model gpt-4

# 3. Or run programmatically
python run_openai_evals.py --mode programmatic
```

## 📊 Sample Evaluation Results

The implementation was tested with 5 diverse samples:

```
📊 Average Alpha Score: 53.2%
📈 Average Full Score: 52.3%
🔢 Total Tokens: 18,841

Sample Results:
- Sample 1: Alpha 24/100 (24.0%) - Poor quality response
- Sample 2: Alpha 24/100 (24.0%) - Poor quality response  
- Sample 3: Alpha 96/100 (96.0%) - High quality response
- Sample 4: Alpha 24/100 (24.0%) - Poor quality response
- Sample 5: Alpha 98/100 (98.0%) - High quality response
```

## 🔧 Configuration Options

### Environment Variables
```bash
OPENAI_API_KEY="popinb_zillowlabs__hs7x0vTjbLwjKhNStdgL1Dd"
OPENAI_BASE_URL="https://api.zillowlabs.com/openai/v1"
```

### Evaluation Parameters
```python
# Deterministic evaluation settings
temperature=0          # Greedy decoding
seed=42               # Reproducible results  
max_tokens=4000       # Extended responses
model="gpt-4"         # Judge model
```

### Registry Configuration (`zillow_eval_registry.yaml`)
```yaml
zillow-affordability.dev.v0:
  class: openaieval_zillow_judge:ZillowAffordabilityEval
  args:
    samples_jsonl: zillow_eval_samples.jsonl
    seed: 42
    judge_model: gpt-4
    max_tokens: 4000
    temperature: 0
```

## 📁 Sample Data Format

Evaluation samples are stored in JSONL format (`zillow_eval_samples.jsonl`):

```json
{
  "question": "what is my buyability?",
  "candidate_answer": "Your buyability is $400,000.",
  "user_profile": {
    "annual_income": null,
    "monthly_debts": null,
    "down_payment": null,
    "credit_score": null
  }
}
```

## 📝 Output and Logging

### Structured Logging (OpenAI Evals Style)

Each evaluation produces structured logs in JSONL format:

```json
{
  "type": "sample_evaluation",
  "timestamp": 1752851312.123,
  "sample_id": 1,
  "input": {...},
  "output": "...",
  "scores": {
    "alpha_score": 24,
    "full_score": 28,
    "alpha_percentage": 24.0,
    "full_percentage": 23.3
  }
}
```

### Results Files

- **`evals_log_[timestamp].jsonl`** - Structured evaluation logs
- **`openai_evals_compatible_results.json`** - Complete results summary
- **`openai_evals_results.json`** - Programmatic execution results

## 🔍 Framework Comparison

| Feature | OpenAI Evals Framework | Custom LLM Judge | Standalone Evals |
|---------|----------------------|------------------|------------------|
| **Installation** | `pip install evals` | No dependencies | `pip install openai` |
| **CLI Integration** | ✅ `oaieval` command | ⚠️ Custom script | ✅ Compatible patterns |
| **Registry Config** | ✅ YAML-based | ❌ Code-based | ✅ Compatible format |
| **Structured Logging** | ✅ Built-in | ⚠️ Manual | ✅ Framework patterns |
| **Reproducibility** | ✅ Framework-level | ✅ Manual seeds | ✅ Deterministic |
| **Parallelization** | ✅ Built-in | ⚠️ Manual | ⚠️ Manual |
| **Custom Parameters** | ⚠️ Framework limits | ✅ Full control | ✅ Full control |
| **API Flexibility** | ⚠️ Framework patterns | ✅ Direct control | ✅ Direct control |

## 🎛️ Advanced Usage

### Custom Model Configuration

```python
# Use o1-preview for enhanced reasoning
evaluator = ZillowAffordabilityEval(
    model="o1-preview",
    max_tokens=10000,
    seed=42
)
```

### Batch Evaluation

```python
# Process multiple samples with custom recorder
recorder = EvalRecorder("batch_evaluation.jsonl")
results = evaluator.run_eval(recorder)
```

### CLI Options

```bash
# Run with specific configuration
python run_openai_evals.py \
    --mode both \
    --model o1-preview \
    --eval-name zillow-affordability-o1 \
    --max-samples 10
```

## 🔬 Technical Implementation Details

### OpenAI Evals Framework Integration

The implementation follows OpenAI Evals patterns:

1. **Eval Class**: Inherits from `evals.eval.Eval` base class
2. **Registry**: Uses YAML configuration for eval specifications
3. **Completion Functions**: Compatible with framework completion function interface
4. **Recording**: Uses `RecorderBase` for structured logging
5. **Metrics**: Framework-compatible metric definitions and aggregation

### Deterministic Evaluation

- **Temperature**: 0 (greedy decoding)
- **Seed**: 42 (reproducible results)
- **Parameters**: Fixed frequency/presence penalties
- **Model**: Consistent judge model across runs

### Error Handling

- Graceful degradation for missing knowledge base files
- Comprehensive error logging and reporting
- Fallback to default samples if JSONL file missing
- Token limit management and warnings

## 🏆 Benefits of OpenAI Evals Framework Approach

1. **Standardization**: Industry-standard evaluation patterns
2. **Reproducibility**: Framework-level deterministic execution  
3. **Scalability**: Built-in support for large-scale evaluations
4. **Integration**: Compatible with existing OpenAI Evals workflows
5. **Community**: Access to shared evaluation methodologies
6. **Documentation**: Well-documented framework patterns
7. **Maintenance**: Reduced custom code maintenance burden

## 🚀 Next Steps

1. **Install Full Framework**: `pip install evals` for complete integration
2. **Extend Samples**: Add more evaluation samples to `zillow_eval_samples.jsonl`
3. **Custom Metrics**: Define additional domain-specific metrics
4. **Model Comparison**: Evaluate different judge models (GPT-4, o1-preview)
5. **Production Integration**: Integrate with CI/CD pipelines
6. **Human Validation**: Collect human annotations for evaluation quality assessment

## 📞 Support and Usage

This OpenAI Evals implementation provides a **completely separate and distinct** approach from the custom LLM-as-a-judge implementation, offering:

- Industry-standard evaluation framework patterns
- Enhanced reproducibility and logging
- CLI integration capabilities  
- Structured configuration management
- Framework-compatible result formats

Choose the approach that best fits your evaluation workflow and infrastructure requirements!