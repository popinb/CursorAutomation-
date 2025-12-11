#!/usr/bin/env python3
"""
Evaluation Runner CLI

A command-line tool for running LLM evaluations with different configurations.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

from generalized_evaluator import GeneralizedEvaluator, create_sample_metrics_csv
from evaluation_config_manager import EvaluationConfigManager, create_sample_evaluation_data


def run_evaluation_from_csv(
    metrics_csv_path: str,
    evaluation_data: List[Dict[str, Any]],
    judge_model: str = "gpt-4",
    output_dir: str = "/tmp/evaluation_results"
) -> Dict[str, Any]:
    """
    Run evaluation using a CSV metrics configuration.
    
    Args:
        metrics_csv_path: Path to CSV file containing metrics configuration
        evaluation_data: List of evaluation samples
        judge_model: LLM model to use for evaluation
        output_dir: Directory to save results
        
    Returns:
        Dictionary containing evaluation results and summary
    """
    print(f"🚀 Starting evaluation with metrics from: {metrics_csv_path}")
    print(f"📊 Evaluating {len(evaluation_data)} samples")
    print(f"🤖 Using judge model: {judge_model}")
    
    # Create evaluator
    evaluator = GeneralizedEvaluator(
        judge_model=judge_model,
        metrics_config_path=metrics_csv_path
    )
    
    # Run evaluation
    results_df = evaluator.evaluate_dataset(evaluation_data)
    
    # Generate summary
    summary = evaluator._generate_summary(results_df)
    
    # Export results
    exported_files = evaluator.export_results(results_df, output_dir)
    
    # Log to MLflow
    try:
        evaluator.log_to_mlflow(results_df)
    except Exception as e:
        print(f"⚠️ MLflow logging failed: {e}")
    
    return {
        'results_df': results_df,
        'summary': summary,
        'exported_files': exported_files
    }


def run_evaluation_from_scenario(
    scenario_name: str,
    evaluation_data: List[Dict[str, Any]],
    output_dir: str = "/tmp/evaluation_results"
) -> Dict[str, Any]:
    """
    Run evaluation using a predefined scenario.
    
    Args:
        scenario_name: Name of the predefined scenario
        evaluation_data: List of evaluation samples
        output_dir: Directory to save results
        
    Returns:
        Dictionary containing evaluation results and summary
    """
    print(f"🚀 Starting evaluation with scenario: {scenario_name}")
    print(f"📊 Evaluating {len(evaluation_data)} samples")
    
    # Initialize configuration manager
    config_manager = EvaluationConfigManager()
    
    # Create predefined scenarios if they don't exist
    if not config_manager.list_scenarios():
        config_manager.create_predefined_scenarios()
    
    # Get scenario
    scenario = config_manager.get_scenario(scenario_name)
    if not scenario:
        available_scenarios = config_manager.list_scenarios()
        raise ValueError(f"Scenario '{scenario_name}' not found. Available scenarios: {available_scenarios}")
    
    print(f"📋 Scenario: {scenario.description}")
    print(f"🤖 Judge model: {scenario.judge_model}")
    
    # Create evaluator
    evaluator = config_manager.create_evaluator(scenario_name)
    
    # Run evaluation
    results_df = evaluator.evaluate_dataset(evaluation_data)
    
    # Generate summary
    summary = evaluator._generate_summary(results_df)
    
    # Export results
    exported_files = evaluator.export_results(results_df, output_dir)
    
    # Log to MLflow
    try:
        evaluator.log_to_mlflow(results_df)
    except Exception as e:
        print(f"⚠️ MLflow logging failed: {e}")
    
    return {
        'results_df': results_df,
        'summary': summary,
        'exported_files': exported_files
    }


def load_evaluation_data(data_path: str) -> List[Dict[str, Any]]:
    """
    Load evaluation data from a JSON file.
    
    Args:
        data_path: Path to JSON file containing evaluation data
        
    Returns:
        List of evaluation samples
    """
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and 'samples' in data:
        return data['samples']
    else:
        raise ValueError("Invalid data format. Expected list of samples or dict with 'samples' key.")


def print_results_summary(results: Dict[str, Any]):
    """Print a formatted summary of evaluation results."""
    summary = results['summary']
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS SUMMARY")
    print("="*60)
    print(f"Total Evaluations: {summary['total_evaluations']}")
    print(f"Passed: {summary['passed_evaluations']}")
    print(f"Failed: {summary['failed_evaluations']}")
    print(f"Overall Pass Rate: {summary['overall_pass_rate']:.1f}%")
    
    print(f"\nPer-Metric Results:")
    print("-" * 40)
    for metric_name, metric_summary in summary['metrics_summary'].items():
        print(f"{metric_name}:")
        print(f"  Pass Rate: {metric_summary['pass_rate']:.1f}% ({metric_summary['passed']}/{metric_summary['total']})")
        if metric_summary['average_score'] is not None:
            print(f"  Average Score: {metric_summary['average_score']:.2f}")
        print(f"  Threshold: {metric_summary['threshold']}")
        print()
    
    print("Exported Files:")
    print("-" * 40)
    for format_name, file_path in results['exported_files'].items():
        print(f"  {format_name}: {file_path}")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Run LLM evaluations with flexible configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run evaluation with custom metrics CSV
  python run_evaluation.py --metrics-csv metrics.csv --data evaluation_data.json
  
  # Run evaluation with predefined scenario
  python run_evaluation.py --scenario financial_advisory --data evaluation_data.json
  
  # Create sample metrics CSV
  python run_evaluation.py --create-sample-metrics
  
  # List available scenarios
  python run_evaluation.py --list-scenarios
  
  # Run with sample data
  python run_evaluation.py --scenario basic_quality --sample-data
        """
    )
    
    # Configuration options
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument(
        '--metrics-csv',
        help='Path to CSV file containing metrics configuration'
    )
    config_group.add_argument(
        '--scenario',
        help='Name of predefined evaluation scenario'
    )
    config_group.add_argument(
        '--create-sample-metrics',
        action='store_true',
        help='Create a sample metrics CSV file'
    )
    config_group.add_argument(
        '--list-scenarios',
        action='store_true',
        help='List available predefined scenarios'
    )
    
    # Data options
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument(
        '--data',
        help='Path to JSON file containing evaluation data'
    )
    data_group.add_argument(
        '--sample-data',
        action='store_true',
        help='Use sample evaluation data'
    )
    
    # Other options
    parser.add_argument(
        '--judge-model',
        default='gpt-4',
        help='LLM model to use for evaluation (default: gpt-4)'
    )
    parser.add_argument(
        '--output-dir',
        default='/tmp/evaluation_results',
        help='Directory to save results (default: /tmp/evaluation_results)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed results'
    )
    
    args = parser.parse_args()
    
    try:
        # Handle special commands
        if args.create_sample_metrics:
            create_sample_metrics_csv("sample_metrics.csv")
            print("✅ Created sample_metrics.csv")
            return
        
        if args.list_scenarios:
            config_manager = EvaluationConfigManager()
            config_manager.create_predefined_scenarios()
            scenarios = config_manager.list_scenarios()
            print("Available scenarios:")
            for scenario_name in scenarios:
                scenario = config_manager.get_scenario(scenario_name)
                print(f"  - {scenario_name}: {scenario.description}")
            return
        
        # Load evaluation data
        if args.sample_data:
            if not args.scenario:
                print("❌ --sample-data requires --scenario to be specified")
                sys.exit(1)
            evaluation_data = create_sample_evaluation_data(args.scenario)
        elif args.data:
            evaluation_data = load_evaluation_data(args.data)
        else:
            print("❌ Must specify either --data or --sample-data")
            sys.exit(1)
        
        # Run evaluation
        if args.metrics_csv:
            results = run_evaluation_from_csv(
                metrics_csv_path=args.metrics_csv,
                evaluation_data=evaluation_data,
                judge_model=args.judge_model,
                output_dir=args.output_dir
            )
        elif args.scenario:
            results = run_evaluation_from_scenario(
                scenario_name=args.scenario,
                evaluation_data=evaluation_data,
                output_dir=args.output_dir
            )
        else:
            print("❌ Must specify either --metrics-csv or --scenario")
            sys.exit(1)
        
        # Print results
        print_results_summary(results)
        
        if args.verbose:
            print("\nDetailed Results:")
            print("-" * 60)
            print(results['results_df'].to_string(index=False))
        
        print("\n✅ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()