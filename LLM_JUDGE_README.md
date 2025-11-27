# LLM-as-a-Judge Application

A flexible Python framework to convert Custom GPT evaluations into automated LLM-based judgments using the OpenAI Evals framework.

## 🎯 Overview

This application provides a comprehensive solution for automating evaluation processes by:

- **Converting Custom GPT logic** into structured, reproducible evaluations
- **Integrating with OpenAI Evals** framework for standardized evaluation workflows  
- **Supporting multiple metric types** (binary, scale, categorical, numeric, text)
- **Enabling batch processing** for large-scale evaluations
- **Providing flexible configuration** through YAML files or programmatic setup

## 🚀 Key Features

### ✨ Multi-Metric Evaluation
- **Binary metrics**: True/False assessments (e.g., accuracy, compliance)
- **Scale metrics**: 1-5 rating scales (e.g., quality, helpfulness)
- **Categorical metrics**: Predefined categories (e.g., tone, style)
- **Numeric metrics**: Numerical scores with custom ranges
- **Text metrics**: Open-ended text evaluations

### 🧠 Knowledge-Based Evaluation
- Load multiple knowledge sources (JSON, YAML, TXT files)
- Incorporate domain-specific guidelines and examples
- Reference ground truth data and rubrics
- Contextual evaluation based on user profiles

### 🔧 Flexible Configuration
- YAML-based configuration for easy customization
- Programmatic configuration for dynamic setups
- Weighted metrics for importance-based scoring
- Customizable prompt templates

### 📊 Advanced Analytics
- Batch evaluation with statistical summaries
- Detailed per-metric scoring and justifications
- Export results in JSON/YAML formats
- Automated report generation

### 🔌 Framework Integration
- Native OpenAI Evals compatibility
- Standalone operation capability
- Easy integration with existing workflows
- Extensible architecture for custom metrics

## 📦 Installation

1. **Set up Python environment:**
```bash
python3 -m venv llm_judge_env
source llm_judge_env/bin/activate  # On Windows: llm_judge_env\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set OpenAI API key:**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## 🏃‍♂️ Quick Start

### 1. Basic Usage

```python
from llm_judge_app import LLMJudgeApp

# Initialize with configuration file
judge = LLMJudgeApp(config_path="config_example.yaml")

# Evaluate a response
result = judge.evaluate(
    candidate_answer="Python is a versatile programming language...",
    question="What is Python?",
    context={"user_level": "beginner"}
)

print(f"Overall Score: {result['overall_score']:.3f}")
```

### 2. Run the Demo

```bash
python demo.py
```

This will demonstrate:
- Basic evaluation functionality
- Batch processing capabilities
- OpenAI Evals integration
- Custom configuration creation
- Result export and reporting

## 📝 Configuration

### YAML Configuration Example

```yaml
name: "Content Quality Judge"
description: "Evaluates content for accuracy, helpfulness, and clarity"

# Model settings
model: "gpt-4"
temperature: 0.2
max_tokens: 2000

# Knowledge sources
knowledge_sources:
  - "knowledge/guidelines.json"
  - "knowledge/examples.txt"
  - "knowledge/rubric.yaml"

# Evaluation metrics
metrics:
  - name: "Content Accuracy"
    type: "binary"
    description: "Whether content is factually accurate"
    weight: 2.0
    scoring_criteria:
      true_indicators: ["accurate", "correct", "factual"]
      false_indicators: ["inaccurate", "incorrect", "wrong"]

  - name: "Helpfulness"
    type: "scale"
    description: "How helpful the response is (1-5)"
    weight: 1.5
    scoring_criteria:
      min_scale: 1
      max_scale: 5

# Evaluation prompt template
evaluation_prompt_template: |
  You are {judge_name}. {judge_description}
  
  Evaluate this response based on the provided criteria...
```

### Programmatic Configuration

```python
from llm_judge_app import EvaluationMetric, MetricType, JudgeConfig, LLMJudgeApp

# Define metrics
metrics = [
    EvaluationMetric(
        name="Relevance",
        type=MetricType.SCALE,
        description="How relevant the answer is",
        scoring_criteria={"min_scale": 1, "max_scale": 5},
        weight=2.0
    )
]

# Create configuration
config = JudgeConfig(
    name="Custom Judge",
    description="Evaluates response quality",
    metrics=metrics,
    knowledge_sources=[],
    evaluation_prompt_template="Evaluate: {candidate_answer}"
)

# Initialize judge
judge = LLMJudgeApp(config=config)
```

## 🔄 Usage Patterns

### Single Evaluation

```python
result = judge.evaluate(
    candidate_answer="Your response here...",
    question="Original question",
    context={"domain": "technical", "level": "intermediate"},
    reference_answer="Ground truth answer (optional)"
)
```

### Batch Evaluation

```python
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

batch_results = judge.batch_evaluate(samples)
print(f"Average score: {batch_results['batch_statistics']['average_score']}")
```

### OpenAI Evals Integration

```python
from llm_judge_app import LLMJudgeOAIEval

# Initialize for OpenAI Evals
evals_judge = LLMJudgeOAIEval(config_path="config.yaml")

# Evaluate single sample
sample = {
    "input": {
        "candidate_answer": "Response to evaluate",
        "question": "Original question",
        "context": {"metadata": "value"}
    }
}

result = evals_judge.eval_sample(sample)
```

## 📊 Output Format

### Evaluation Result Structure

```json
{
  "overall_score": 0.85,
  "metric_scores": {
    "Content Accuracy": {
      "raw_score": true,
      "normalized_score": 1.0,
      "weight": 2.0,
      "type": "binary"
    },
    "Helpfulness": {
      "raw_score": 4,
      "normalized_score": 0.75,
      "weight": 1.5,
      "type": "scale"
    }
  },
  "justifications": {
    "Content Accuracy": "The response provides factually correct information...",
    "Helpfulness": "The answer addresses the question with practical examples..."
  },
  "detailed_evaluation": "Full LLM evaluation text...",
  "metadata": {
    "timestamp": "2024-01-15T10:30:00",
    "model": "gpt-4",
    "question": "What is Python?",
    "candidate_answer_length": 156
  }
}
```

## 🎛️ Metric Types

### Binary Metrics
- **Purpose**: True/False evaluations
- **Use cases**: Accuracy, compliance, presence of required elements
- **Configuration**: Define true/false indicators

### Scale Metrics  
- **Purpose**: Numerical ratings (e.g., 1-5, 1-10)
- **Use cases**: Quality, helpfulness, clarity
- **Configuration**: Set min/max scale values

### Categorical Metrics
- **Purpose**: Predefined category selection
- **Use cases**: Tone (formal/casual), style, classification
- **Configuration**: Define category list

### Numeric Metrics
- **Purpose**: Custom numerical scores
- **Use cases**: Percentage scores, custom calculations
- **Configuration**: Set min/max value ranges

## 📈 Analytics and Reporting

### Export Results

```python
# Export evaluation history
judge.export_results("results.json", format="json")
judge.export_results("results.yaml", format="yaml")
```

### Generate Reports

```python
# Generate summary report
report = judge.generate_report()
print(report)

# Save report to file
with open("evaluation_report.md", 'w') as f:
    f.write(report)
```

### Batch Statistics

```python
batch_results = judge.batch_evaluate(samples)
stats = batch_results['batch_statistics']

print(f"Total samples: {stats['total_samples']}")
print(f"Success rate: {stats['successful_evaluations']}/{stats['total_samples']}")
print(f"Average score: {stats['average_score']:.3f}")
print(f"Metric averages: {stats['metric_averages']}")
```

## 🛠️ Customization

### Adding Custom Metrics

1. **Define in YAML:**
```yaml
metrics:
  - name: "Custom Metric"
    type: "scale"  # or binary, categorical, numeric, text
    description: "Description of what this evaluates"
    weight: 1.0
    scoring_criteria:
      min_scale: 1
      max_scale: 5
```

2. **Define programmatically:**
```python
custom_metric = EvaluationMetric(
    name="Custom Metric",
    type=MetricType.SCALE,
    description="Custom evaluation criteria",
    scoring_criteria={"min_scale": 1, "max_scale": 5},
    weight=1.0
)
```

### Creating Knowledge Sources

1. **JSON format** (`knowledge/guidelines.json`):
```json
{
  "evaluation_guidelines": {
    "accuracy": "Verify all factual claims",
    "completeness": "Address all parts of the question"
  }
}
```

2. **Text format** (`knowledge/examples.txt`):
```text
Example high-quality response:
Q: How to improve code quality?
A: Use meaningful variable names, write unit tests, follow style guides...

Example low-quality response:
Q: How to improve code quality?
A: Make it better.
```

3. **YAML format** (`knowledge/rubric.yaml`):
```yaml
scoring_rubric:
  excellent: "Comprehensive, accurate, highly useful"
  good: "Accurate and useful"
  satisfactory: "Mostly accurate"
  needs_improvement: "Some inaccuracies"
  poor: "Major issues"
```

### Custom Prompt Templates

```yaml
evaluation_prompt_template: |
  You are {judge_name}, an expert evaluator.
  
  {judge_description}
  
  ## Knowledge Base
  {knowledge_context}
  
  ## Task
  Evaluate this response: {candidate_answer}
  
  Question: {question}
  Reference: {reference_answer}
  Context: {context}
  
  ## Metrics
  {metrics_description}
  
  Provide detailed scores and justifications.
```

## 🔧 Advanced Features

### Error Handling

The framework includes comprehensive error handling:
- Invalid configuration detection
- Missing knowledge source handling
- API failure recovery
- Malformed response parsing

### Logging

```python
import logging
logging.basicConfig(level=logging.INFO)

# Logs will show evaluation progress and any issues
```

### Performance Optimization

- Batch API calls for efficiency
- Caching of knowledge sources
- Configurable timeout settings
- Memory-efficient result storage

## 🧪 Testing and Validation

### Running Tests

```bash
# Run the demo to validate installation
python demo.py

# Test specific functionality
python -c "from llm_judge_app import LLMJudgeApp; print('✅ Import successful')"
```

### Validation Checklist

- [ ] OpenAI API key configured
- [ ] Dependencies installed
- [ ] Configuration file valid
- [ ] Knowledge sources accessible
- [ ] Sample evaluation runs successfully

## 📚 Examples and Use Cases

### 1. Code Review Evaluation
- Evaluate code review comments for helpfulness and accuracy
- Metrics: Relevance, Constructiveness, Technical Accuracy

### 2. Customer Support Response Assessment
- Judge customer service responses for quality and compliance
- Metrics: Helpfulness, Professionalism, Issue Resolution

### 3. Educational Content Evaluation
- Assess educational materials for accuracy and clarity
- Metrics: Accuracy, Completeness, Age Appropriateness

### 4. Content Moderation
- Evaluate content for policy compliance and quality
- Metrics: Policy Compliance, Quality, Appropriateness

### 5. Technical Documentation Review
- Judge technical documentation for completeness and clarity
- Metrics: Accuracy, Completeness, Clarity, Examples

## 🤝 Integration Examples

### CI/CD Pipeline Integration

```python
def evaluate_responses_pipeline():
    judge = LLMJudgeApp(config_path="ci_config.yaml")
    
    # Load test cases
    with open("test_cases.json") as f:
        test_cases = json.load(f)
    
    # Run evaluations
    results = judge.batch_evaluate(test_cases)
    
    # Check quality gate
    avg_score = results['batch_statistics']['average_score']
    if avg_score < 0.7:
        raise ValueError(f"Quality gate failed: {avg_score:.3f} < 0.7")
    
    return results
```

### API Service Wrapper

```python
from flask import Flask, request, jsonify

app = Flask(__name__)
judge = LLMJudgeApp(config_path="api_config.yaml")

@app.route('/evaluate', methods=['POST'])
def evaluate_endpoint():
    data = request.json
    result = judge.evaluate(
        candidate_answer=data['answer'],
        question=data.get('question', ''),
        context=data.get('context', {})
    )
    return jsonify(result)
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure virtual environment is activated
   - Install all requirements: `pip install -r requirements.txt`

2. **API Key Issues**
   - Set environment variable: `export OPENAI_API_KEY="key"`
   - Verify key has sufficient credits and permissions

3. **Configuration Errors**
   - Validate YAML syntax
   - Check file paths for knowledge sources
   - Ensure all required fields are present

4. **Knowledge Source Issues**
   - Verify file paths are correct
   - Check file format (JSON/YAML/TXT)
   - Ensure files are readable

5. **Evaluation Failures**
   - Check API rate limits
   - Verify prompt template formatting
   - Review error logs for specific issues

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# This will show detailed debug information
judge = LLMJudgeApp(config_path="config.yaml")
```

## 📄 License

This project is provided as-is for educational and evaluation purposes. Please ensure compliance with your organization's policies when using AI evaluation systems and processing sensitive data.

## 🤝 Contributing

To extend the framework:

1. **Add new metric types** by extending the `MetricType` enum and corresponding extraction methods
2. **Create custom knowledge source loaders** for different file formats
3. **Implement new scoring algorithms** for specialized evaluation needs
4. **Add integration adapters** for other evaluation frameworks

## 📞 Support

For questions and support:
- Review the demo script for usage examples
- Check the troubleshooting section for common issues  
- Examine the existing Zillow Judge implementation for reference patterns
- Review OpenAI Evals documentation for framework integration details

---

**Built with ❤️ for automated LLM evaluation workflows**