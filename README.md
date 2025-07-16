# Custom GPT LLM as a Judge - OpenAI Evals Framework

This project converts a Custom GPT evaluation setup into a structured OpenAI evals framework implementation. It simulates the exact instructions and knowledge sources that would be present in a Custom GPT, providing a systematic way to evaluate LLM responses using LLM-as-a-judge methodology.

## 🎯 Overview

The system replicates the functionality of a Custom GPT that acts as an AI evaluator, complete with:

- **Custom Instructions**: Detailed system prompts that define the evaluator's role and methodology
- **Knowledge Sources**: Simulated knowledge base containing evaluation criteria and guidelines
- **LLM Judge Logic**: Sophisticated evaluation algorithms that provide scores and reasoning
- **OpenAI Evals Integration**: Full compatibility with the OpenAI evals framework structure

## 🏗️ Architecture

```
Custom GPT Judge Evaluator
├── CustomGPTKnowledgeBase     # Simulates uploaded knowledge files
├── CustomGPTJudge             # Core evaluation logic
├── CustomGPTEvaluator         # OpenAI evals compatible interface
├── EvalDataLoader             # Data loading utilities
└── Configuration Files        # YAML configs and sample data
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Required dependencies (see requirements.txt)

### Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your OpenAI API key**:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```
   
   Or create a `.env` file:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

4. **Run the evaluation**:
   ```bash
   python custom_gpt_judge_eval.py
   ```

## 📊 Usage Examples

### Basic Evaluation

```python
import asyncio
from custom_gpt_judge_eval import CustomGPTEvaluator, EvalDataLoader

# Initialize evaluator
evaluator = CustomGPTEvaluator(
    api_key="your-openai-api-key",
    model="gpt-4",
    knowledge_sources=["evaluation_guidelines.pdf", "scoring_rubric.md"]
)

# Load sample data
samples = EvalDataLoader.create_sample_data()

# Run evaluation
results = await evaluator.evaluate_samples(samples)

# Generate report
report = evaluator.generate_report(results)
print(f"Pass Rate: {report['pass_rate']:.2%}")
print(f"Average Score: {report['average_score']:.2f}/5.0")
```

### Custom Evaluation Sample

```python
from custom_gpt_judge_eval import EvalSample

# Create a custom evaluation sample
sample = EvalSample(
    input=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing."}
    ],
    ideal="A clear explanation of quantum computing principles",
    metadata={
        "response_to_evaluate": "Quantum computing uses quantum bits that can exist in multiple states simultaneously...",
        "evaluation_criteria": "Check for technical accuracy and clarity.",
        "category": "technical"
    }
)

# Evaluate single sample
result = await evaluator.evaluate_single_sample(sample)
print(f"Score: {result.score}/5.0")
print(f"Reasoning: {result.reasoning}")
```

### Loading Data from JSONL

```python
# Load evaluation samples from JSONL file
samples = EvalDataLoader.load_from_jsonl("custom_gpt_samples.jsonl")

# Run evaluation on loaded samples
results = await evaluator.evaluate_samples(samples)
```

## 📁 File Structure

```
├── custom_gpt_judge_eval.py      # Main evaluator implementation
├── evals_config.yaml             # OpenAI evals configuration
├── custom_gpt_samples.jsonl      # Sample evaluation data
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── custom_gpt_evaluation_report.json  # Generated report (after running)
```

## 🔧 Configuration

### OpenAI Evals YAML Configuration

The `evals_config.yaml` file follows the OpenAI evals format:

```yaml
custom-gpt-judge:
  id: custom-gpt-judge.dev.v1
  description: Custom GPT LLM as a Judge evaluator
  metrics: [accuracy, average_score, pass_rate]

custom-gpt-judge.dev.v1:
  class: custom_gpt_judge_eval:CustomGPTEvaluator
  args:
    knowledge_sources:
      - evaluation_guidelines.pdf
      - quality_standards.txt
    model: gpt-4
    temperature: 0.1
```

### Sample Data Format

Evaluation samples follow the OpenAI evals JSONL format:

```json
{
  "input": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
  ],
  "ideal": "Paris",
  "metadata": {
    "response_to_evaluate": "The capital of France is Paris...",
    "evaluation_criteria": "Check for factual accuracy.",
    "category": "factual"
  }
}
```

## 🎯 Custom GPT Instructions

The evaluator uses detailed instructions that replicate a Custom GPT's behavior:

### Core Responsibilities
1. Evaluate responses for accuracy, relevance, and quality
2. Provide detailed reasoning for assessments
3. Score responses on a scale of 0-5
4. Identify strengths and areas for improvement

### Evaluation Process
1. Read the original question/prompt carefully
2. Analyze the response against evaluation criteria
3. Consider knowledge base guidelines
4. Provide step-by-step reasoning
5. Assign numerical score with justification

### Scoring Criteria
- **5**: Exceptional - Exceeds all expectations
- **4**: Good - Meets all expectations well
- **3**: Satisfactory - Meets basic expectations
- **2**: Below Average - Partially meets expectations
- **1**: Poor - Significant issues present
- **0**: Unacceptable - Fails to meet basic requirements

## 📚 Knowledge Sources

The system simulates three types of knowledge sources typically uploaded to a Custom GPT:

1. **evaluation_guidelines.pdf** - General evaluation methodology
2. **quality_standards.txt** - Quality criteria and standards
3. **scoring_rubric.md** - Detailed scoring guidelines

These are simulated as text content but can be replaced with actual file loading in production.

## 📈 Evaluation Report

The system generates comprehensive reports including:

- **Overall Metrics**: Pass rate, average score, total samples
- **Score Distribution**: Breakdown by score ranges
- **Detailed Results**: Individual sample scores and reasoning
- **Category Analysis**: Performance by response category

Example report output:
```
CUSTOM GPT JUDGE EVALUATION REPORT
==================================================
Eval Name: custom-gpt-judge
Total Samples: 10
Passed: 8
Failed: 2
Pass Rate: 80.00%
Average Score: 3.75/5.0

Score Distribution:
  4-5: 6 samples
  3-4: 2 samples
  2-3: 1 samples
  1-2: 1 samples
  0-1: 0 samples
```

## 🔄 Integration with OpenAI Evals

This implementation is designed to be compatible with the official OpenAI evals framework. You can:

1. **Use with oaieval CLI**: Register the evaluator and run with standard commands
2. **Extend existing templates**: Build upon model-graded evaluation patterns
3. **Custom evaluation logic**: Implement domain-specific evaluation criteria
4. **Batch processing**: Handle large datasets efficiently

### Running with oaieval (if integrated)

```bash
# After proper registration in evals framework
oaieval gpt-4 custom-gpt-judge --max_samples 100
```

## 🛠️ Customization

### Adding New Knowledge Sources

```python
class CustomGPTKnowledgeBase:
    def _load_knowledge_sources(self) -> str:
        # Load actual files instead of simulated content
        knowledge_content = ""
        for source in self.knowledge_sources:
            with open(source, 'r') as f:
                knowledge_content += f.read() + "\n\n"
        return knowledge_content
```

### Custom Evaluation Criteria

Modify the `custom_instructions` in `CustomGPTJudge` to add specific evaluation criteria for your use case:

```python
self.custom_instructions = """
Your custom evaluation instructions here...

DOMAIN-SPECIFIC CRITERIA:
- Technical accuracy for coding problems
- Safety considerations for medical advice
- Legal compliance for financial guidance
"""
```

### Scoring Customization

Adjust scoring scales and thresholds:

```python
# Custom scoring scale (e.g., 1-10 instead of 0-5)
def _parse_evaluation_response(self, response: str) -> Dict[str, Any]:
    # Custom parsing logic for different scoring scales
    pass
```

## 🧪 Testing

The system includes comprehensive sample data covering various categories:

- **Factual Questions**: Geography, history, science
- **Mathematical Problems**: Basic arithmetic, word problems
- **Creative Tasks**: Poetry, storytelling, creative writing
- **Instructional Content**: How-to guides, tutorials
- **Professional Advice**: Customer service, technical support

Run tests with:
```bash
python -m pytest tests/  # If test suite is added
```

## 📝 Best Practices

### 1. Evaluation Design
- **Clear criteria**: Define specific, measurable evaluation criteria
- **Balanced datasets**: Include diverse response types and quality levels
- **Consistent scoring**: Use detailed rubrics for reliable assessments

### 2. LLM Judge Configuration
- **Low temperature**: Use temperature=0.1 for consistent evaluations
- **Detailed prompts**: Provide comprehensive instructions and examples
- **Chain of thought**: Ask for step-by-step reasoning before scoring

### 3. Quality Assurance
- **Human validation**: Regularly check judge decisions against human evaluation
- **Inter-rater reliability**: Test consistency across multiple evaluation runs
- **Bias detection**: Monitor for systematic biases in scoring patterns

## 🔍 Troubleshooting

### Common Issues

1. **API Rate Limits**
   ```python
   # Add delays between requests
   await asyncio.sleep(0.1)
   ```

2. **Parsing Errors**
   ```python
   # Implement robust error handling
   try:
       score = float(score_match.group(1))
   except (AttributeError, ValueError):
       score = 0.0  # Default fallback
   ```

3. **Memory Issues with Large Datasets**
   ```python
   # Process in batches
   for batch in chunked(samples, batch_size=10):
       results.extend(await evaluator.evaluate_samples(batch))
   ```

## 🤝 Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is provided as-is for educational and research purposes. Please ensure compliance with OpenAI's usage policies when using their API.

## 🔗 Related Resources

- [OpenAI Evals Documentation](https://github.com/openai/evals)
- [LLM as a Judge Paper](https://arxiv.org/abs/2306.05685)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Custom GPT Guidelines](https://help.openai.com/en/articles/8554397-creating-a-gpt)

## 📞 Support

For questions or issues:

1. Check the troubleshooting section above
2. Review the OpenAI evals documentation
3. Open an issue with detailed reproduction steps

---

**Note**: This implementation provides a foundation for converting Custom GPT evaluators to the OpenAI evals framework. Customize it according to your specific evaluation needs and requirements.