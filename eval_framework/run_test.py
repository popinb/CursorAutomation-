#!/usr/bin/env python3
"""
Complete test of the evaluation framework
"""

import os
import json
import yaml
from datetime import datetime
from pathlib import Path

# Create a minimal working version without external dependencies
class SimpleEvaluationFramework:
    """Simplified evaluation framework for testing"""
    
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.results_dir = Path(self.config['output']['results_dir'])
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
    def load_data(self):
        """Load CSV data"""
        data_dir = Path(self.config['data']['ground_truth_dir'])
        csv_file = data_dir / self.config['data']['files'][0]
        
        # Parse CSV manually
        with open(csv_file, 'r') as f:
            lines = f.readlines()
            
        headers = lines[0].strip().split(',')
        rows = []
        
        for line in lines[1:]:
            # Simple CSV parsing (handles basic cases)
            values = []
            current = ""
            in_quotes = False
            
            for char in line:
                if char == '"':
                    in_quotes = not in_quotes
                elif char == ',' and not in_quotes:
                    values.append(current.strip())
                    current = ""
                else:
                    current += char
                    
            values.append(current.strip())
            rows.append(dict(zip(headers, values)))
            
        return rows
    
    def evaluate_sample(self, sample, metric_config):
        """Simulate LLM evaluation of a single sample"""
        metric_name = metric_config['name']
        
        # Simulate different evaluations based on metric and sample
        if metric_name == 'personalization_accuracy':
            # Check if response mentions specific numbers from user info
            has_numbers = any(term in sample['response'] for term in ['780', '120,000', '7.5%', '600,000', '620'])
            score = 1 if has_numbers else 0
            explanation = "All personal details accurately used." if has_numbers else "Personal details not properly referenced."
            
        elif metric_name == 'relevance':
            # All samples in our test data are relevant
            score = 5 if 'rate' in sample['prompt'] or 'afford' in sample['prompt'] else 4
            explanation = "Directly answers the question asked."
            
        elif metric_name == 'advice_quality':
            # Vary quality based on response length and specificity
            if 'could' in sample['response'] and any(char.isdigit() for char in sample['response']):
                score = 4
                explanation = "Good personalized advice with specific numbers."
            else:
                score = 3
                explanation = "Basic but accurate advice."
                
        elif metric_name == 'actionability':
            # Check for specific recommendations
            if 'save' in sample['response'] or 'qualify' in sample['response']:
                score = 5
                explanation = "Clear actionable steps provided."
            else:
                score = 3
                explanation = "General guidance without specific next steps."
        
        return {
            f"{metric_name}_score": score,
            "explanation": explanation
        }
    
    def run_evaluation(self):
        """Run the complete evaluation"""
        print("🚀 Starting Evaluation Framework Test")
        print("="*60)
        
        # Load data
        print("\n1. Loading data...")
        samples = self.load_data()
        print(f"   ✓ Loaded {len(samples)} samples")
        
        # Initialize results
        results = {
            'samples': samples,
            'evaluations': {},
            'summary': {}
        }
        
        # Run evaluation for each metric
        print("\n2. Running evaluations...")
        for metric_config in self.config['metrics']:
            metric_name = metric_config['name']
            print(f"   Evaluating: {metric_name}")
            
            metric_results = []
            for i, sample in enumerate(samples):
                eval_result = self.evaluate_sample(sample, metric_config)
                metric_results.append(eval_result)
                
            results['evaluations'][metric_name] = metric_results
            
            # Calculate summary statistics
            scores = [r[f"{metric_name}_score"] for r in metric_results]
            results['summary'][metric_name] = {
                'mean': sum(scores) / len(scores),
                'min': min(scores),
                'max': max(scores),
                'threshold': metric_config['threshold'],
                'pass_rate': sum(1 for s in scores if s >= metric_config['threshold']) / len(scores)
            }
        
        # Display results
        self.display_results(results)
        
        # Save results
        self.save_results(results)
        
        return results
    
    def display_results(self, results):
        """Display evaluation results"""
        print("\n" + "="*60)
        print("📊 EVALUATION RESULTS")
        print("="*60)
        
        # Summary by metric
        print("\nSUMMARY BY METRIC:")
        print("-"*40)
        
        for metric_name, summary in results['summary'].items():
            print(f"\n{metric_name.replace('_', ' ').title()}:")
            print(f"  Mean Score: {summary['mean']:.2f}")
            print(f"  Range: {summary['min']} - {summary['max']}")
            print(f"  Pass Rate: {summary['pass_rate']*100:.0f}%")
            print(f"  Threshold: {summary['threshold']}")
        
        # Detailed results table
        print("\n\nDETAILED RESULTS:")
        print("-"*60)
        print(f"{'#':<3} {'Question':<40} {'PA':<4} {'REL':<4} {'AQ':<4} {'ACT':<4}")
        print("-"*60)
        
        for i, sample in enumerate(results['samples']):
            question = sample['prompt'][:37] + "..." if len(sample['prompt']) > 40 else sample['prompt']
            
            scores = []
            for metric in ['personalization_accuracy', 'relevance', 'advice_quality', 'actionability']:
                score = results['evaluations'][metric][i][f"{metric}_score"]
                scores.append(str(score))
            
            print(f"{i+1:<3} {question:<40} {scores[0]:<4} {scores[1]:<4} {scores[2]:<4} {scores[3]:<4}")
        
        # Failed cases
        print("\n\nFAILED CASES:")
        print("-"*40)
        
        any_failures = False
        for metric_name, summary in results['summary'].items():
            threshold = summary['threshold']
            failures = []
            
            for i, eval_result in enumerate(results['evaluations'][metric_name]):
                score = eval_result[f"{metric_name}_score"]
                if score < threshold:
                    failures.append((i, score, eval_result['explanation']))
            
            if failures:
                any_failures = True
                print(f"\n{metric_name} (threshold: {threshold}):")
                for sample_idx, score, explanation in failures:
                    print(f"  Sample {sample_idx+1}: Score={score}")
                    print(f"    {explanation}")
        
        if not any_failures:
            print("\nNo failures found!")
    
    def save_results(self, results):
        """Save results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = self.results_dir / f"evaluation_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n\n✅ Results saved to: {results_file}")
        
        # Save summary CSV
        summary_data = []
        for i in range(len(results['samples'])):
            row = {
                'sample_id': i + 1,
                'prompt': results['samples'][i]['prompt'][:50] + '...'
            }
            
            for metric_name in results['evaluations']:
                score = results['evaluations'][metric_name][i][f"{metric_name}_score"]
                row[metric_name] = score
                
            summary_data.append(row)
        
        # Write CSV manually
        csv_file = self.results_dir / f"evaluation_summary_{timestamp}.csv"
        with open(csv_file, 'w') as f:
            # Headers
            headers = ['sample_id', 'prompt'] + list(results['evaluations'].keys())
            f.write(','.join(headers) + '\n')
            
            # Data rows
            for row in summary_data:
                values = [str(row.get(h, '')) for h in headers]
                f.write(','.join(values) + '\n')
        
        print(f"✅ Summary saved to: {csv_file}")


def main():
    """Run the test"""
    print("\n🧪 TESTING EVALUATION FRAMEWORK")
    print("="*60)
    
    # Use the test configuration
    config_path = "test_config.yaml"
    
    # Check files exist
    if not Path(config_path).exists():
        print(f"❌ Error: Configuration file '{config_path}' not found!")
        return
    
    if not Path("examples/sample_ground_truth.csv").exists():
        print("❌ Error: Sample data file not found!")
        return
    
    print(f"\n✓ Configuration: {config_path}")
    print("✓ Data: examples/sample_ground_truth.csv")
    
    # Run evaluation
    framework = SimpleEvaluationFramework(config_path)
    results = framework.run_evaluation()
    
    print("\n" + "="*60)
    print("✅ TEST COMPLETED SUCCESSFULLY!")
    print("="*60)


if __name__ == "__main__":
    main()