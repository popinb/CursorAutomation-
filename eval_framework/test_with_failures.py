#!/usr/bin/env python3
"""
Test the evaluation framework with some intentional failures
"""

import os
import json
from datetime import datetime
from pathlib import Path

# Create test data with some problematic responses
def create_test_data_with_issues():
    """Create test data that will trigger some failures"""
    
    test_dir = Path("test_data_failures")
    test_dir.mkdir(exist_ok=True)
    
    # Create CSV with some problematic responses
    csv_content = """prompt,response,user_personalization_features
"What's my mortgage rate?","Mortgage rates vary widely depending on many factors.","credit_score: 750, income: $95000"
"How much can I borrow?","Based on your income of $95,000 and credit score of 750, you could qualify for a loan up to $380,000.","credit_score: 750, income: $95000"
"Should I get a 15 or 30 year loan?","Both have pros and cons. 15-year loans have lower rates but higher payments.","loan_amount: $300000, income: $95000"
"What's the process to buy a home?","The home buying process involves getting pre-approved, finding a home, making an offer, getting inspections, and closing. Based on your $95,000 income, you're in a good position to start.","income: $95000"
"Can I afford a $400k house?","That might be stretching your budget. Generally, you should keep your housing costs under 28% of income.","income: $95000, current_savings: $80000"
"""
    
    csv_file = test_dir / "test_responses.csv"
    with open(csv_file, 'w') as f:
        f.write(csv_content)
    
    # Create config for this test
    config = {
        "experiment": {
            "name": "failure_test_evaluation",
            "run_name_prefix": "failure_test"
        },
        "data": {
            "ground_truth_dir": "./test_data_failures/",
            "columns": {
                "prompt": "prompt",
                "response": "response",
                "personalization": "user_personalization_features"
            },
            "files": ["test_responses.csv"]
        },
        "models": {
            "response_generation": {"type": "ground_truth"},
            "judge_models": ["gpt-4o"]
        },
        "metrics": [
            {
                "name": "personalization_accuracy",
                "description": "Checks if personal info is used accurately",
                "prompt_template": "Check if personal details are used...",
                "score_range": [0, 1],
                "threshold": 1,
                "output_format": "json"
            },
            {
                "name": "specificity",
                "description": "How specific and detailed the response is",
                "prompt_template": "Rate specificity 1-5...",
                "score_range": [1, 5],
                "threshold": 4,
                "output_format": "json"
            },
            {
                "name": "actionability", 
                "description": "How actionable the advice is",
                "prompt_template": "Rate actionability 1-5...",
                "score_range": [1, 5],
                "threshold": 4,
                "output_format": "json"
            },
            {
                "name": "completeness",
                "description": "How complete the answer is",
                "prompt_template": "Rate completeness 1-5...",
                "score_range": [1, 5],
                "threshold": 4,
                "output_format": "json"
            }
        ],
        "output": {
            "results_dir": "./failure_test_results/",
            "save_raw_responses": True,
            "save_detailed_scores": True,
            "generate_summary": True
        },
        "mlflow": {
            "enable_logging": False
        }
    }
    
    config_file = "failure_test_config.yaml"
    import yaml
    with open(config_file, 'w') as f:
        yaml.dump(config, f)
    
    return config_file


def evaluate_with_failures(sample, metric_name):
    """Simulate evaluations that will produce some failures"""
    
    response = sample['response']
    user_info = sample.get('user_personalization_features', '')
    
    if metric_name == 'personalization_accuracy':
        # Fail if response doesn't mention specific numbers from user info
        numbers = ['750', '95000', '95,000', '300000', '300,000', '400k', '80000']
        has_personalization = any(num in response for num in numbers)
        
        return {
            "personalization_accuracy_score": 1 if has_personalization else 0,
            "explanation": "Uses specific personal data" if has_personalization else "No personal data referenced"
        }
    
    elif metric_name == 'specificity':
        # Rate based on presence of specific numbers, percentages, or concrete advice
        specifics = ['$', '%', 'month', 'year', '000']
        specific_count = sum(1 for s in specifics if s in response)
        
        if specific_count >= 3:
            score = 5
        elif specific_count >= 2:
            score = 4
        elif specific_count >= 1:
            score = 3
        else:
            score = 2
            
        return {
            "specificity_score": score,
            "explanation": f"Contains {specific_count} specific elements"
        }
    
    elif metric_name == 'actionability':
        # Check for action words and next steps
        action_words = ['should', 'can', 'start', 'get', 'apply', 'contact', 'consider']
        has_actions = sum(1 for word in action_words if word.lower() in response.lower())
        
        if has_actions >= 3:
            score = 5
        elif has_actions >= 2:
            score = 4
        elif has_actions >= 1:
            score = 3
        else:
            score = 2
            
        return {
            "actionability_score": score,
            "explanation": f"Contains {has_actions} actionable elements"
        }
    
    elif metric_name == 'completeness':
        # Rate based on response length and coverage
        word_count = len(response.split())
        
        if word_count > 40:
            score = 5
        elif word_count > 25:
            score = 4
        elif word_count > 15:
            score = 3
        else:
            score = 2
            
        return {
            "completeness_score": score,
            "explanation": f"Response has {word_count} words"
        }


def run_failure_test():
    """Run test with failures"""
    
    print("\n🧪 TESTING FRAMEWORK WITH FAILURE CASES")
    print("="*70)
    
    # Create test data and config
    config_file = create_test_data_with_issues()
    print(f"✓ Created test data with problematic responses")
    print(f"✓ Configuration: {config_file}")
    
    # Load config
    import yaml
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load test data
    from pathlib import Path
    data_dir = Path(config['data']['ground_truth_dir'])
    csv_file = data_dir / config['data']['files'][0]
    
    # Parse CSV
    with open(csv_file, 'r') as f:
        lines = f.readlines()
    
    headers = lines[0].strip().split(',')
    samples = []
    
    for line in lines[1:]:
        values = []
        current = ""
        in_quotes = False
        
        for char in line:
            if char == '"':
                in_quotes = not in_quotes
            elif char == ',' and not in_quotes:
                values.append(current.strip().strip('"'))
                current = ""
            else:
                current += char
        
        values.append(current.strip().strip('"'))
        samples.append(dict(zip(headers, values)))
    
    print(f"\n✓ Loaded {len(samples)} test samples")
    
    # Run evaluations
    results = {
        'metrics': {},
        'summary': {}
    }
    
    print("\n📊 Running Evaluations...")
    print("-"*70)
    
    for metric in config['metrics']:
        metric_name = metric['name']
        print(f"\nEvaluating: {metric_name}")
        
        evaluations = []
        scores = []
        
        for i, sample in enumerate(samples):
            eval_result = evaluate_with_failures(sample, metric_name)
            evaluations.append(eval_result)
            score = eval_result[f"{metric_name}_score"]
            scores.append(score)
            
            # Show individual results
            status = "✅" if score >= metric['threshold'] else "❌"
            print(f"  Sample {i+1}: Score={score} {status} - {eval_result['explanation']}")
        
        results['metrics'][metric_name] = evaluations
        
        # Calculate summary
        results['summary'][metric_name] = {
            'scores': scores,
            'mean': sum(scores) / len(scores),
            'pass_rate': sum(1 for s in scores if s >= metric['threshold']) / len(scores),
            'threshold': metric['threshold'],
            'failures': sum(1 for s in scores if s < metric['threshold'])
        }
    
    # Display summary
    print("\n" + "="*70)
    print("📈 EVALUATION SUMMARY")
    print("="*70)
    
    print("\nMETRIC PERFORMANCE:")
    print("-"*50)
    print(f"{'Metric':<25} {'Mean':<8} {'Pass %':<10} {'Failed'}")
    print("-"*50)
    
    for metric_name, summary in results['summary'].items():
        print(f"{metric_name:<25} {summary['mean']:<8.2f} {summary['pass_rate']*100:<10.0f} {summary['failures']}")
    
    # Show failed cases
    print("\n\n❌ FAILED CASES BREAKDOWN:")
    print("-"*50)
    
    for metric_name, summary in results['summary'].items():
        if summary['failures'] > 0:
            print(f"\n{metric_name} (threshold: {summary['threshold']}):")
            
            for i, score in enumerate(summary['scores']):
                if score < summary['threshold']:
                    print(f"  Sample {i+1}: '{samples[i]['prompt'][:40]}...'")
                    print(f"    Score: {score}")
                    print(f"    Issue: {results['metrics'][metric_name][i]['explanation']}")
    
    # Save results
    output_dir = Path(config['output']['results_dir'])
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"failure_test_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            'samples': samples,
            'evaluations': results['metrics'],
            'summary': results['summary']
        }, f, indent=2)
    
    print(f"\n\n✅ Results saved to: {results_file}")
    
    print("\n" + "="*70)
    print("✅ FAILURE TEST COMPLETED!")
    print("="*70)


if __name__ == "__main__":
    run_failure_test()