# Annotation Mode - Human Evaluation UI

A comprehensive web-based annotation interface for human evaluators to manually score LLM responses using the same 12 metrics as the automated Zillow Judge Evaluator.

## Overview

The Annotation Mode provides:

1. **Web-based Annotation UI** (`annotation_mode.py`) - A Flask application with a beautiful, modern interface
2. **Notebook Version** (`annotation_mode_notebook.py`) - For Databricks/Jupyter environments
3. **Comparison Tools** (`annotation_comparison.py`) - Compare human annotations with LLM judge scores

## Quick Start

### Option 1: Web Application (Recommended)

```bash
# Install dependencies
pip install flask pandas numpy

# Run the annotation server
python annotation_mode.py --port 5000

# Open in browser
# http://localhost:5000?annotator=your_name
```

### Option 2: Databricks/Jupyter Notebook

1. Upload `annotation_mode_notebook.py` to Databricks
2. Run all cells
3. Use the interactive HTML interface in the notebook output

## Features

### Beautiful Modern UI
- Clean, professional design
- Intuitive scoring interface
- Real-time progress tracking
- Mobile-responsive layout

### All 12 Evaluation Metrics
The annotation interface includes all metrics from the Zillow Judge Evaluator:

| Metric | Type | Description |
|--------|------|-------------|
| Personalization Accuracy | Binary | Matches user-specific figures |
| Context-Based Personalization | 1-5 Scale | Percentage of customizations included |
| Next-Step Identification | Binary | Actionable guidance provided |
| Assumption Listing | Binary | Explicit statement of assumptions |
| Assumption Trust | 1-5 Scale | Transparency about limitations |
| Calculation Accuracy | Binary | Mathematical correctness |
| Faithfulness to Ground Truth | Binary | Alignment with knowledge sources |
| Fair Housing Compliance | Binary | Adherence to fair housing laws |
| Overall Accuracy | Binary | Holistic correctness |
| Structured Presentation | 1-5 Scale | Quality of formatting |
| Coherence | Binary | Logical consistency |
| Completeness | 1-5 Scale | Coverage of question aspects |

### Annotation Persistence
- Automatic saving of annotations
- Resume from where you left off
- Export to CSV/JSON formats

### Multiple Annotators
- Support for multiple annotators
- Unique annotator IDs
- Independent progress tracking

## File Structure

```
annotation_mode.py           # Main Flask application
annotation_mode_notebook.py  # Databricks/Jupyter version
annotation_comparison.py     # Compare human vs LLM scores
annotation_data/             # Data directory (auto-created)
  samples.json              # Loaded samples
  annotations.json          # Saved annotations
```

## Usage Guide

### 1. Loading Samples

**Demo Samples**: The system includes 3 demo samples for testing. They are loaded automatically when you start the application.

**Custom Samples**: You can add samples via the API:

```python
import requests

sample = {
    "sample_id": "my_sample_001",
    "question": "What factors affect my home buying power?",
    "candidate_answer": "Your buying power is determined by...",
    "user_profile": {"annual_income": 90000, "monthly_debts": 200},
    "ground_truth": "Should mention income, debts, credit score..."
}

requests.post("http://localhost:5000/api/samples", json=sample)
```

**CSV Import**: Load samples from a CSV file:

```python
# Required columns: sample_id, question, candidate_answer
# Optional: user_profile (JSON string), ground_truth
```

### 2. Annotating Samples

1. **Navigate to a sample** using the Previous/Next buttons
2. **Score each metric** by clicking the appropriate option:
   - Binary metrics: Click "True/False" or "Accurate/Inaccurate"
   - Scale metrics: Use the slider (1-5)
3. **Add justifications** (optional) for each score
4. **Add notes** for any additional observations
5. **Submit** the annotation

### 3. Exporting Annotations

Click the "Export" tab to download your annotations:

- **CSV**: Spreadsheet format for analysis
- **JSON**: Full data with all metadata

Or use the API:

```bash
# Download CSV
curl http://localhost:5000/api/export?format=csv -o annotations.csv

# Download JSON
curl http://localhost:5000/api/export?format=json -o annotations.json
```

### 4. Comparing with LLM Scores

Use the comparison module to analyze agreement:

```python
from annotation_comparison import AnnotationComparator

comparator = AnnotationComparator()
comparator.load_human_annotations("human_annotations.csv")
comparator.load_llm_scores("llm_scores.csv")

# Get agreement summary
summary = comparator.get_agreement_summary()
print(f"Overall Agreement: {summary['overall_agreement_rate']:.1%}")

# Generate detailed report
print(comparator.generate_report())

# Export results
comparator.export_dataframe("comparison_results.csv")
```

Or from command line:

```bash
python annotation_comparison.py human_annotations.csv llm_scores.csv output_dir/
```

## API Reference

### Samples API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/samples` | GET | Get all samples |
| `/api/samples` | POST | Add a new sample |

### Annotations API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/annotations` | POST | Save an annotation |
| `/api/annotations/<sample_id>` | GET | Get annotations for a sample |
| `/api/progress` | GET | Get annotation progress |
| `/api/export` | GET | Export annotations (format=csv/json) |

## Comparison Metrics

The comparison module calculates:

- **Overall Agreement Rate**: Percentage of matching scores
- **Cohen's Kappa**: Inter-rater agreement for binary metrics
- **Pearson Correlation**: Correlation for scale metrics
- **Per-Metric Statistics**: Detailed breakdown by metric

### Interpreting Cohen's Kappa

| Kappa | Interpretation |
|-------|----------------|
| 0.81-1.00 | Almost perfect agreement |
| 0.61-0.80 | Substantial agreement |
| 0.41-0.60 | Moderate agreement |
| 0.21-0.40 | Fair agreement |
| 0.00-0.20 | Slight agreement |
| < 0.00 | Poor agreement |

## Configuration

### Environment Variables

```bash
# Flask server settings
export FLASK_PORT=5000
export FLASK_HOST=0.0.0.0
export FLASK_DEBUG=false

# Data directory
export ANNOTATION_DATA_DIR=./annotation_data
```

### Command Line Options

```bash
python annotation_mode.py --help

Options:
  --port PORT    Port to run server on (default: 5000)
  --host HOST    Host to bind to (default: 0.0.0.0)
  --debug        Enable debug mode
```

## Databricks Integration

For Databricks, use the notebook version which:

1. Works without Flask (pure HTML)
2. Uses Databricks widgets for configuration
3. Saves annotations to workspace files
4. Compatible with `displayHTML()`

### Databricks Widgets

- `annotator_id`: Your annotator identifier
- `samples_csv`: Path to samples CSV (optional)
- `annotations_csv`: Output file for annotations

## Best Practices

### For Annotators

1. **Read the full response** before scoring any metric
2. **Be consistent** in how you interpret metrics
3. **Use justifications** for borderline cases
4. **Take breaks** to avoid fatigue affecting quality

### For Project Leads

1. **Calibrate annotators** using test samples first
2. **Check inter-annotator agreement** regularly
3. **Review disagreements** to improve guidelines
4. **Document edge cases** as they arise

## Troubleshooting

### Web UI Issues

**Page doesn't load:**
- Check if Flask is running
- Verify port is not in use
- Check firewall settings

**Annotations not saving:**
- Check file permissions
- Verify `annotation_data/` directory exists
- Check browser console for errors

### Comparison Issues

**No comparisons generated:**
- Verify sample IDs match between files
- Check metric names are consistent
- Ensure files are properly formatted

## License

This module is part of the Zillow Judge Evaluator project.

---

For more information, see the main [README.md](README.md) or contact the project maintainers.
