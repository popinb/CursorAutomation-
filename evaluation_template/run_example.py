#!/usr/bin/env python3
"""
Example script demonstrating how to use the LLM Evaluation Template

This script shows how to:
1. Set up the evaluation environment
2. Run evaluations with different configurations
3. Access and analyze results
"""

import os
import sys
import argparse
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from evaluation_template import EvaluationPipeline


def run_sample_evaluation():
    """Run a sample evaluation using the provided sample dataset."""
    
    # Check if API key is set
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  Warning: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key: export OPENAI_API_KEY='your-key-here'")
        print("Proceeding with demo configuration...")
    
    # Create a simple evaluation configuration for the sample data
    sample_config = {
        'dataset': {
            'file_path': './data/sample_dataset.csv',
            'columns': {
                'prompt': 'prompt',
                'response': 'response',
                'ground_truth': 'ground_truth',
                'metadata': 'metadata'
            }
        },
        'experiment': {
            'name': 'sample_evaluation_demo',
            'run_name_prefix': 'demo_run',
            'output_directory': './results/demo/'
        },
        'judges': {
            'models': ['gpt-4o-mini'],  # Use cheaper model for demo
            'api': {
                'openai_api_key_env': 'OPENAI_API_KEY',
                'openai_base_url_env': 'OPENAI_API_BASE',
                'temperature': 0,
                'timeout': 60
            }
        },
        'metrics': {
            'accuracy': {
                'enabled': True,
                'description': 'Factual accuracy compared to ground truth',
                'prompt_template': '''You are an impartial evaluator.
Your task is to assess how factually accurate the response is compared to the ground truth.

### Materials

* **Question:**
```
{prompt}
```

* **Response:**
```
{response}
```

* **Ground Truth:**
```
{ground_truth}
```

### Evaluation Guidelines

Rate the factual accuracy on a scale of 1-5:
- 1: Completely incorrect
- 2: Mostly incorrect with some correct elements  
- 3: Partially correct but missing key information
- 4: Mostly correct with minor inaccuracies
- 5: Completely accurate

### Output Format

Return only this JSON (no extra text):
```json
{{
  "accuracy_score": <1-5 integer>,
  "explanation": "<brief explanation>"
}}
```''',
                'score_key': 'accuracy_score',
                'threshold': 3,
                'scale': 5
            },
            'clarity': {
                'enabled': True,
                'description': 'How clear and understandable the response is',
                'prompt_template': '''You are an impartial evaluator.
Your task is to assess the clarity of the response.

### Materials

* **Question:**
```
{prompt}
```

* **Response:**
```
{response}
```

### Evaluation Guidelines

Rate the clarity on a scale of 1-5:
- 1: Very unclear or confusing
- 2: Somewhat unclear
- 3: Generally clear
- 4: Clear and easy to understand
- 5: Exceptionally clear

### Output Format

Return only this JSON (no extra text):
```json
{{
  "clarity_score": <1-5 integer>,
  "explanation": "<brief explanation>"
}}
```''',
                'score_key': 'clarity_score',
                'threshold': 3,
                'scale': 5
            }
        },
        'composite_metrics': {
            'overall_quality': {
                'description': 'Overall response quality',
                'method': 'average',
                'metrics': ['accuracy', 'clarity']
            }
        },
        'mlflow': {
            'enabled': True,
            'tracking_uri': None,
            'experiment_tags': {
                'evaluation_type': 'demo',
                'version': '1.0'
            }
        }
    }
    
    # Save temporary config file
    import yaml
    os.makedirs('./config', exist_ok=True)
    config_path = './config/demo_config.yaml'
    
    with open(config_path, 'w') as f:
        yaml.dump(sample_config, f, default_flow_style=False)
    
    print("🚀 Running sample evaluation...")
    print(f"Configuration saved to: {config_path}")
    
    try:
        # Run the evaluation
        pipeline = EvaluationPipeline(config_path)
        results_df = pipeline.run_evaluation()
        
        print("\n" + "="*60)
        print("📊 SAMPLE EVALUATION RESULTS")
        print("="*60)
        
        # Display summary
        print(f"\nEvaluated {len(results_df)} samples")
        
        metrics = ['accuracy', 'clarity', 'overall_quality']
        for metric in metrics:
            if metric in results_df.columns:
                mean_score = results_df[metric].mean()
                print(f"{metric.replace('_', ' ').title()}: {mean_score:.3f}")
        
        print(f"\nDetailed results saved to: ./results/demo/")
        print(f"MLflow tracking URI: file://{os.path.abspath('./mlruns')}")
        print(f"View results: mlflow ui")
        
        return results_df
        
    except Exception as e:
        print(f"❌ Error running evaluation: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure OPENAI_API_KEY is set correctly")
        print("2. Check that the sample dataset exists at ./data/sample_dataset.csv") 
        print("3. Verify internet connection for API calls")
        raise


def main():
    """Main entry point for example script."""
    parser = argparse.ArgumentParser(description='Run LLM evaluation examples')
    parser.add_argument('--config', help='Path to evaluation config file')
    parser.add_argument('--demo', action='store_true', 
                       help='Run demo evaluation with sample data')
    
    args = parser.parse_args()
    
    if args.demo or not args.config:
        print("🎯 Running demo evaluation with sample data...")
        run_sample_evaluation()
    else:
        print(f"🎯 Running evaluation with config: {args.config}")
        pipeline = EvaluationPipeline(args.config)
        results_df = pipeline.run_evaluation()
        print("✅ Evaluation completed successfully!")


if __name__ == '__main__':
    main()