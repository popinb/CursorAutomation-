#!/usr/bin/env python3
"""
Example Usage of Generalized Directory Feature

This script demonstrates how to use the generalized evaluation system
with different configurations and scenarios.
"""

import json
import os
from pathlib import Path
from generalized_evaluator import GeneralizedEvaluator, create_sample_metrics_csv
from evaluation_config_manager import EvaluationConfigManager, create_sample_evaluation_data


def example_1_basic_csv_evaluation():
    """Example 1: Basic evaluation using CSV metrics configuration."""
    print("="*80)
    print("EXAMPLE 1: Basic CSV-based Evaluation")
    print("="*80)
    
    # Create sample metrics CSV
    create_sample_metrics_csv("example_metrics.csv")
    
    # Create sample evaluation data
    evaluation_data = [
        {
            'prompt': 'What is the capital of France?',
            'response': 'The capital of France is Paris.',
            'ground_truth': 'Paris'
        },
        {
            'prompt': 'How do I calculate mortgage payments?',
            'response': 'You can calculate mortgage payments using the formula: P = L[c(1 + c)^n]/[(1 + c)^n - 1] where P is payment, L is loan amount, c is monthly interest rate, and n is number of payments.',
            'ground_truth': 'Mortgage payment calculation formula'
        },
        {
            'prompt': 'What is machine learning?',
            'response': 'Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.',
            'ground_truth': 'Machine learning definition'
        }
    ]
    
    # Create evaluator
    evaluator = GeneralizedEvaluator(
        judge_model="gpt-4",
        metrics_config_path="example_metrics.csv"
    )
    
    # Run evaluation
    print("Running evaluation...")
    results_df = evaluator.evaluate_dataset(evaluation_data)
    
    # Display results
    print("\nResults:")
    print(results_df.to_string(index=False))
    
    # Generate summary
    summary = evaluator._generate_summary(results_df)
    print(f"\n📊 SUMMARY")
    print(f"Total Evaluations: {summary['total_evaluations']}")
    print(f"Passed: {summary['passed_evaluations']}")
    print(f"Pass Rate: {summary['overall_pass_rate']:.1f}%")
    
    return results_df, summary


def example_2_predefined_scenarios():
    """Example 2: Using predefined evaluation scenarios."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Predefined Scenarios")
    print("="*80)
    
    # Initialize configuration manager
    config_manager = EvaluationConfigManager()
    
    # Create predefined scenarios
    config_manager.create_predefined_scenarios()
    
    # List available scenarios
    print("Available scenarios:")
    for scenario_name in config_manager.list_scenarios():
        scenario = config_manager.get_scenario(scenario_name)
        print(f"  - {scenario_name}: {scenario.description}")
    
    # Run evaluation for financial advisory scenario
    print(f"\nRunning Financial Advisory Evaluation...")
    
    # Create sample data for financial advisory
    sample_data = create_sample_evaluation_data("financial_advisory")
    
    # Create evaluator
    evaluator = config_manager.create_evaluator("financial_advisory")
    
    # Run evaluation
    results_df = evaluator.evaluate_dataset(sample_data)
    
    # Display results
    print("\nResults:")
    print(results_df.to_string(index=False))
    
    # Generate summary
    summary = evaluator._generate_summary(results_df)
    print(f"\n📊 SUMMARY")
    print(f"Total Evaluations: {summary['total_evaluations']}")
    print(f"Passed: {summary['passed_evaluations']}")
    print(f"Pass Rate: {summary['overall_pass_rate']:.1f}%")
    
    return results_df, summary


def example_3_custom_scenario():
    """Example 3: Creating a custom evaluation scenario."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Custom Scenario Creation")
    print("="*80)
    
    # Initialize configuration manager
    config_manager = EvaluationConfigManager()
    
    # Define custom metrics for code quality evaluation
    code_quality_metrics = [
        {
            'name': 'syntax_correctness',
            'type': 'binary',
            'description': 'Evaluates if the code has correct syntax',
            'grading_rubric': 'Check if the code follows proper syntax rules for the programming language. Look for syntax errors, missing semicolons, incorrect brackets, etc.',
            'threshold': 0.5,
            'ground_truth_column': 'expected_syntax',
            'ground_truth_file_path': 'code_examples.json'
        },
        {
            'name': 'code_quality',
            'type': '1-5_scale',
            'description': 'Evaluates overall code quality',
            'grading_rubric': 'Rate from 1-5: 1=very poor quality, many issues; 2=poor quality, several issues; 3=average quality, some issues; 4=good quality, minor issues; 5=excellent quality, clean code.',
            'threshold': 3.0,
            'ground_truth_column': 'quality_standards',
            'ground_truth_file_path': 'quality_standards.json'
        },
        {
            'name': 'efficiency',
            'type': 'percentage',
            'description': 'Evaluates code efficiency',
            'grading_rubric': 'Rate from 0-100%: Consider time complexity, space complexity, and overall efficiency of the algorithm or approach used.',
            'threshold': 70.0,
            'ground_truth_column': 'efficiency_benchmarks',
            'ground_truth_file_path': 'efficiency_benchmarks.json'
        }
    ]
    
    # Create custom scenario
    scenario_path = config_manager.create_scenario(
        name="code_quality",
        description="Evaluation for code quality and correctness",
        metrics=code_quality_metrics
    )
    
    print(f"Created custom scenario: {scenario_path}")
    
    # Create sample data for code quality evaluation
    code_evaluation_data = [
        {
            'prompt': 'Write a function to calculate the factorial of a number',
            'response': 'def factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    return n * factorial(n - 1)',
            'ground_truth': 'Factorial function implementation',
            'expected_syntax': 'Python function syntax',
            'quality_standards': 'Clean, readable code',
            'efficiency_benchmarks': 'O(n) time complexity'
        },
        {
            'prompt': 'Write a function to find the maximum element in a list',
            'response': 'def find_max(lst):\n    max_val = lst[0]\n    for i in range(1, len(lst)):\n        if lst[i] > max_val:\n            max_val = lst[i]\n    return max_val',
            'ground_truth': 'Maximum element finder',
            'expected_syntax': 'Python function syntax',
            'quality_standards': 'Clean, readable code',
            'efficiency_benchmarks': 'O(n) time complexity'
        }
    ]
    
    # Create evaluator for custom scenario
    evaluator = config_manager.create_evaluator("code_quality")
    
    # Run evaluation
    print("Running code quality evaluation...")
    results_df = evaluator.evaluate_dataset(code_evaluation_data)
    
    # Display results
    print("\nResults:")
    print(results_df.to_string(index=False))
    
    # Generate summary
    summary = evaluator._generate_summary(results_df)
    print(f"\n📊 SUMMARY")
    print(f"Total Evaluations: {summary['total_evaluations']}")
    print(f"Passed: {summary['passed_evaluations']}")
    print(f"Pass Rate: {summary['overall_pass_rate']:.1f}%")
    
    return results_df, summary


def example_4_export_and_mlflow():
    """Example 4: Exporting results and MLflow logging."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Export and MLflow Logging")
    print("="*80)
    
    # Create evaluator
    evaluator = GeneralizedEvaluator(
        judge_model="gpt-4",
        metrics_config_path="example_metrics.csv"
    )
    
    # Create sample data
    evaluation_data = [
        {
            'prompt': 'What is the capital of France?',
            'response': 'The capital of France is Paris.',
            'ground_truth': 'Paris'
        },
        {
            'prompt': 'How do I bake a chocolate cake?',
            'response': 'To bake a chocolate cake, you need flour, sugar, cocoa powder, eggs, butter, and milk. Mix the dry ingredients, then add wet ingredients, and bake at 350°F for 30 minutes.',
            'ground_truth': 'Chocolate cake baking instructions'
        }
    ]
    
    # Run evaluation
    print("Running evaluation...")
    results_df = evaluator.evaluate_dataset(evaluation_data)
    
    # Export results to various formats
    print("\nExporting results...")
    exported_files = evaluator.export_results(results_df, "/tmp/example_evaluation_results")
    
    print("Exported files:")
    for format_name, file_path in exported_files.items():
        print(f"  {format_name}: {file_path}")
    
    # Log to MLflow
    print("\nLogging to MLflow...")
    try:
        evaluator.log_to_mlflow(results_df, "example_evaluation_run")
        print("✅ Successfully logged to MLflow")
    except Exception as e:
        print(f"⚠️ MLflow logging failed: {e}")
    
    return exported_files


def example_5_batch_processing():
    """Example 5: Batch processing with multiple scenarios."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Batch Processing")
    print("="*80)
    
    # Initialize configuration manager
    config_manager = EvaluationConfigManager()
    config_manager.create_predefined_scenarios()
    
    # Define scenarios to test
    scenarios_to_test = ["basic_quality", "customer_support"]
    
    # Create sample data for each scenario
    all_results = {}
    
    for scenario_name in scenarios_to_test:
        print(f"\nTesting scenario: {scenario_name}")
        
        # Create evaluator
        evaluator = config_manager.create_evaluator(scenario_name)
        
        # Create sample data
        sample_data = create_sample_evaluation_data(scenario_name)
        
        # Run evaluation
        results_df = evaluator.evaluate_dataset(sample_data)
        
        # Generate summary
        summary = evaluator._generate_summary(results_df)
        
        # Store results
        all_results[scenario_name] = {
            'results_df': results_df,
            'summary': summary
        }
        
        print(f"  Pass Rate: {summary['overall_pass_rate']:.1f}%")
    
    # Compare results across scenarios
    print(f"\n📊 COMPARISON ACROSS SCENARIOS")
    print("-" * 50)
    for scenario_name, results in all_results.items():
        summary = results['summary']
        print(f"{scenario_name}:")
        print(f"  Pass Rate: {summary['overall_pass_rate']:.1f}%")
        print(f"  Total Evaluations: {summary['total_evaluations']}")
        print()
    
    return all_results


def main():
    """Run all examples."""
    print("Generalized Directory Feature - Example Usage")
    print("=" * 80)
    
    try:
        # Example 1: Basic CSV evaluation
        example_1_basic_csv_evaluation()
        
        # Example 2: Predefined scenarios
        example_2_predefined_scenarios()
        
        # Example 3: Custom scenario creation
        example_3_custom_scenario()
        
        # Example 4: Export and MLflow
        example_4_export_and_mlflow()
        
        # Example 5: Batch processing
        example_5_batch_processing()
        
        print("\n" + "="*80)
        print("✅ All examples completed successfully!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()