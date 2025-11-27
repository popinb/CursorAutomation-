# Quick Start Guide - LLM Judge Application

Get your LLM-as-a-Judge system running in under 5 minutes!

## 🚀 Prerequisites

- Python 3.7+ (tested with Python 3.13)
- OpenAI API key

## ⚡ Quick Setup

### 1. Clone and Setup Environment

```bash
# Navigate to your project directory
cd /path/to/your/project

# Create virtual environment
python3 -m venv llm_judge_env
source llm_judge_env/bin/activate  # On Windows: llm_judge_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Set Your OpenAI API Key

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 3. Validate Installation

```bash
python3 test_setup.py
```

You should see: `🎉 All tests passed! Your setup is ready.`

## 🎯 Basic Usage Examples

### Simple Evaluation

```python
from llm_judge_app import LLMJudgeApp

# Use the example configuration
judge = LLMJudgeApp(config_path="config_example.yaml")

# Evaluate a response
result = judge.evaluate(
    candidate_answer="Python is a high-level programming language known for its readability and versatility.",
    question="What is Python programming language?",
    context={"domain": "programming"}
)

print(f"Overall Score: {result['overall_score']:.3f}")
```

### Run the Interactive Demo

```bash
python3 demo.py
```

This will run 5 comprehensive demos showing all features.

## 📝 Create Your First Custom Judge

### 1. Create Configuration File (`my_judge.yaml`)

```yaml
name: "My Custom Judge"
description: "Evaluates responses for my specific use case"

model: "gpt-4"
temperature: 0.2
max_tokens: 2000

knowledge_sources: []

metrics:
  - name: "Accuracy"
    type: "binary"
    description: "Is the information correct?"
    weight: 2.0
    scoring_criteria:
      true_indicators: ["correct", "accurate", "right"]
      false_indicators: ["wrong", "incorrect", "false"]

  - name: "Clarity"
    type: "scale"
    description: "How clear is the explanation (1-5)?"
    weight: 1.5
    scoring_criteria:
      min_scale: 1
      max_scale: 5

evaluation_prompt_template: |
  You are {judge_name}. {judge_description}
  
  Please evaluate this response: {candidate_answer}
  
  Question: {question}
  
  Rate on:
  - Accuracy: True/False
  - Clarity: 1-5 scale
  
  Provide your evaluation with clear justifications.
```

### 2. Use Your Custom Judge

```python
from llm_judge_app import LLMJudgeApp

judge = LLMJudgeApp(config_path="my_judge.yaml")
result = judge.evaluate(
    candidate_answer="Your text to evaluate...",
    question="What was the question?"
)
```

## 🔧 OpenAI Evals Integration

```python
from llm_judge_app import LLMJudgeOAIEval

# Initialize for OpenAI Evals framework
evals_judge = LLMJudgeOAIEval(config_path="my_judge.yaml")

# Format sample for OpenAI Evals
sample = {
    "input": {
        "candidate_answer": "Response to evaluate",
        "question": "Original question"
    }
}

result = evals_judge.eval_sample(sample)
```

## 📊 Batch Processing

```python
# Evaluate multiple responses at once
samples = [
    {
        "candidate_answer": "Response 1",
        "question": "Question 1",
        "context": {"id": 1}
    },
    {
        "candidate_answer": "Response 2",
        "question": "Question 2", 
        "context": {"id": 2}
    }
]

results = judge.batch_evaluate(samples)
print(f"Average score: {results['batch_statistics']['average_score']:.3f}")
```

## 🎛️ Metric Types Reference

| Type | Example | Scoring |
|------|---------|---------|
| `binary` | True/False, Yes/No | `{"true_indicators": ["yes"], "false_indicators": ["no"]}` |
| `scale` | 1-5 rating | `{"min_scale": 1, "max_scale": 5}` |
| `categorical` | Low/Medium/High | `{"categories": ["Low", "Medium", "High"]}` |
| `numeric` | 0-100 score | `{"min_value": 0, "max_value": 100}` |
| `text` | Open response | `{"text_evaluation": true}` |

## 🔥 Pro Tips

### 1. **Weight Your Metrics**
```yaml
metrics:
  - name: "Critical Feature"
    weight: 3.0  # 3x more important
  - name: "Nice to Have"
    weight: 1.0  # Standard importance
```

### 2. **Add Knowledge Sources**
```yaml
knowledge_sources:
  - "guidelines.json"      # Your evaluation criteria
  - "examples.txt"         # Good/bad examples
  - "domain_knowledge.yaml" # Specialized knowledge
```

### 3. **Export Results**
```python
# Save results for analysis
judge.export_results("results.json")
judge.export_results("results.yaml")

# Generate summary report
report = judge.generate_report()
with open("report.md", "w") as f:
    f.write(report)
```

### 4. **Error Handling**
```python
try:
    result = judge.evaluate(candidate_answer, question)
except Exception as e:
    print(f"Evaluation failed: {e}")
    # Handle error appropriately
```

## 🚨 Common Issues & Solutions

### API Key Not Set
```bash
export OPENAI_API_KEY="sk-your-key-here"
```

### Import Errors
```bash
# Activate virtual environment
source llm_judge_env/bin/activate
pip install -r requirements.txt
```

### Configuration Errors
- Check YAML syntax
- Ensure all required fields are present
- Validate file paths for knowledge sources

### Low Evaluation Scores
- Review your prompt template
- Check metric definitions
- Ensure knowledge sources are relevant

## 📚 Next Steps

1. **Customize metrics** for your domain
2. **Add knowledge sources** with domain expertise
3. **Integrate with your workflow** (CI/CD, API, batch processing)
4. **Scale evaluation** with batch processing
5. **Monitor performance** with detailed reports

## 🆘 Need Help?

1. Run `python3 test_setup.py` to validate installation
2. Check `demo.py` for working examples
3. Review `LLM_JUDGE_README.md` for detailed documentation
4. Examine existing configs like `config_example.yaml`

---

**Happy Evaluating! 🎉**

Convert any Custom GPT evaluation logic into a robust, automated LLM-as-a-Judge system.