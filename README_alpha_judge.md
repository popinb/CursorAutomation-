# Alpha Judge – Draft 2 (Replica, No Pairwise) - Local Version

This is a local version of the Databricks notebook that implements LLM-as-a-Judge evaluation using OpenAI's `o3` model.

## Features

- **Single-system judging** (no pairwise ranking)
- **Ground-truth–aware grading** with embedded fallback texts
- **Comprehensive metrics**: task adherence, completeness, factuality, safety/compliance, numeracy, ground-truth alignment
- **Programmatic checks**: numeric recall, protected-class coverage, occupancy rules coverage
- **Weighted final scoring** (0-100 scale)
- **Fallback scoring** when OpenAI is unavailable
- **Synthetic candidate generation** for testing

## Ground Truths Included

- **Fair Housing Guidelines**: Protected classes, occupancy rules (Keating/BOCA), prohibited acts
- **BuyAbility Golden Response**: Payment breakdown example with specific numbers
- **Buyability Profile**: Numeric scenario for Arkansas

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements_alpha_judge.txt
```

### 2. Configure OpenAI API Key

Option A: Environment variable
```bash
export OPENAI_API_KEY="your_api_key_here"
```

Option B: Create .env file
```bash
cp config_alpha_judge.env .env
# Edit .env and add your API key
```

Option C: Set in the script
```python
OPENAI_API_KEY = "your_api_key_here"
```

### 3. Optional: Add Custom Ground Truth Files

Place these files in the `ground_truths/` directory to override embedded text:
- `Zillow_Fair_Housing_Classifier.docx`
- `godenresponsealpha.docx` 
- `buyabilityprofile2.rtf`

## Usage

### Basic Run

```bash
python alpha_judge_local.py
```

### With Custom Configuration

```bash
export ALPHA_JUDGE_MODEL="o3"
export ALPHA_SAVE_DIR="./my_results"
python alpha_judge_local.py
```

### Custom Evaluation Dataset

Edit the `EVAL_SAMPLES` list in the script to add your own prompts and ground truth references.

### Use Real Candidate Generations

Set `SYNTHESIZE_CANDIDATES = False` and populate `CANDIDATE_OUTPUTS` with your model outputs.

## Output

The script generates:
- **Console output**: Real-time evaluation progress and results
- **CSV file**: Detailed results in spreadsheet format
- **JSONL file**: Results in JSON Lines format for programmatic processing
- **Summary statistics**: Overall performance metrics

## Evaluation Metrics

### LLM Judge Scores (0-5 scale)
- **Task Adherence**: Did the answer address the asked task?
- **Completeness**: Did it include all key parts?
- **Factuality**: Are statements correct relative to references?
- **Safety/Compliance**: Avoids unsafe/discriminatory guidance
- **Numeracy**: Numbers are correct and consistent
- **Ground Truth Alignment**: Content overlap with references

### Programmatic Metrics
- **Numeric Recall**: Percentage of numbers correctly recalled
- **Protected Class Coverage**: Fair housing compliance check
- **Occupancy Rule Hits**: Specific rule mention detection
- **Fuzzy Ground Truth Overlap**: Text similarity scoring

### Final Score (0-100)
Weighted combination of all judge scores, converted to percentage.

## Configuration Options

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `OPENAI_API_KEY` | Required | Your OpenAI API key |
| `ALPHA_JUDGE_MODEL` | `o3` | Judge model to use |
| `ALPHA_SAVE_DIR` | `./alpha_judge_runs` | Results output directory |
| `ALPHA_GT_DIR` | `./ground_truths` | Ground truth files directory |
| `ALPHA_RUN_NAME` | Auto-generated | Name for this evaluation run |

## Troubleshooting

### OpenAI API Issues
- Verify your API key is correct
- Check your OpenAI account has access to the `o3` model
- Ensure you have sufficient credits

### Missing Dependencies
- Install optional packages: `pip install python-docx striprtf`
- The script will fall back to embedded text if these aren't available

### Offline Mode
If OpenAI is unavailable, the script automatically switches to heuristic scoring based on:
- Text length and content analysis
- Fuzzy string matching
- Basic rule-based checks

## Example Output

```
================================================================================
ALPHA JUDGE - DRAFT 2 (REPLICA, NO PAIRWISE)
================================================================================

Starting evaluation at 2024-01-15 10:30:00
Run name: alpha-judge-d2-20240115-103000

Running evaluation...

[1/4] Evaluating: buyability_A
  Prompt: Given these user inputs: income $90,000; monthly debts $200; down payment $18,000; credit score 660–719; target monthly payment $2,500; location Georgia; and rate 6.88%. Explain their BuyAbility and provide a transparent monthly payment breakdown and next steps.
  Candidate: Your BuyAbility is about $320,000. Monthly payment could be around $2,500 comprised of roughly $2,000 principal & interest, $212 taxes, and $100 insurance. You should be fine in Georgia.
  Final Score: 78.5/100 (pass)

...

EVALUATION RESULTS
================================================================================
[Results table displayed here]

Results saved:
  CSV: ./alpha_judge_runs/alpha-judge-d2-20240115-103000-20240115-103045/results.csv
  JSONL: ./alpha_judge_runs/alpha-judge-d2-20240115-103000-20240115-103045/results.jsonl

Summary Statistics:
  Total samples: 4
  Average score: 76.2
  Pass rate: 75.0%
  Score range: 68.5 - 82.1

Evaluation completed at 2024-01-15 10:30:45
```

## License

This script is based on the original Databricks notebook and maintains the same evaluation methodology and ground truth content.