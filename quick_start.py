#!/usr/bin/env python3
"""
Quick Start Script for Universal Evaluation Template System
==========================================================

This script provides an interactive way to set up and run evaluations.
"""

import os
import sys
import pandas as pd
from pathlib import Path
from generate_config import generate_config
from template_evaluation_system import UniversalEvaluator, EvaluationConfig
import yaml


def interactive_setup():
    """Interactive setup for evaluation configuration."""
    print("🚀 Universal Evaluation Template System - Quick Start")
    print("=" * 60)
    
    # Get dataset information
    print("\n📊 Dataset Configuration")
    print("-" * 30)
    
    dataset_path = input("Enter path to your dataset CSV file: ").strip()
    if not os.path.exists(dataset_path):
        print(f"❌ File not found: {dataset_path}")
        return None
    
    # Load dataset to inspect columns
    try:
        df = pd.read_csv(dataset_path)
        print(f"✅ Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
        print(f"Available columns: {', '.join(df.columns)}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None
    
    # Get column names
    prompt_col = input(f"Enter prompt column name (default: 'prompt'): ").strip() or "prompt"
    response_col = input(f"Enter response column name (default: 'response'): ").strip() or "response"
    
    # Check for optional columns
    ground_truth_col = None
    user_features_col = None
    
    if input("Do you have a ground truth column? (y/n): ").lower().startswith('y'):
        ground_truth_col = input("Enter ground truth column name: ").strip()
    
    if input("Do you have user features/personalization data? (y/n): ").lower().startswith('y'):
        user_features_col = input("Enter user features column name: ").strip()
    
    # Get experiment information
    print("\n🔬 Experiment Configuration")
    print("-" * 30)
    
    experiment_name = input("Enter experiment name: ").strip()
    if not experiment_name:
        experiment_name = "my_evaluation"
    
    # Get API key
    print("\n🔑 API Configuration")
    print("-" * 30)
    
    openai_key = input("Enter your OpenAI API key: ").strip()
    if not openai_key:
        print("❌ OpenAI API key is required")
        return None
    
    # Select metrics
    print("\n📏 Metric Selection")
    print("-" * 30)
    
    from metric_templates import list_available_metrics, list_available_domains
    
    print("Available general metrics:")
    general_metrics = list_available_metrics()
    for i, metric in enumerate(general_metrics, 1):
        print(f"  {i}. {metric}")
    
    print("\nAvailable domain-specific metrics:")
    domains = list_available_domains()
    for i, domain in enumerate(domains, 1):
        print(f"  {i}. {domain}")
    
    # Get metric selection
    selected_metrics = []
    print("\nSelect metrics (comma-separated numbers, or press Enter for default):")
    metric_input = input("Metrics: ").strip()
    
    if metric_input:
        try:
            indices = [int(x.strip()) - 1 for x in metric_input.split(',')]
            selected_metrics = [general_metrics[i] for i in indices if 0 <= i < len(general_metrics)]
        except (ValueError, IndexError):
            print("Invalid selection, using default metrics")
            selected_metrics = ["accuracy", "relevance", "helpfulness"]
    else:
        selected_metrics = ["accuracy", "relevance", "helpfulness"]
    
    # Check for domain-specific metrics
    domain = None
    if input("Do you want to add domain-specific metrics? (y/n): ").lower().startswith('y'):
        print("Select domain (comma-separated numbers):")
        domain_input = input("Domains: ").strip()
        if domain_input:
            try:
                indices = [int(x.strip()) - 1 for x in domain_input.split(',')]
                if indices and 0 <= indices[0] < len(domains):
                    domain = domains[indices[0]]
                    domain_metrics = list_available_metrics(domain)
                    print(f"Available metrics for {domain}: {', '.join(domain_metrics)}")
                    domain_metric_input = input("Select domain metrics (comma-separated): ").strip()
                    if domain_metric_input:
                        domain_indices = [int(x.strip()) - 1 for x in domain_metric_input.split(',')]
                        selected_metrics.extend([domain_metrics[i] for i in domain_indices if 0 <= i < len(domain_metrics)])
            except (ValueError, IndexError):
                print("Invalid domain selection")
    
    print(f"\nSelected metrics: {', '.join(selected_metrics)}")
    
    # Generate configuration
    print("\n⚙️  Generating Configuration")
    print("-" * 30)
    
    try:
        config_dict = generate_config(
            dataset_path=dataset_path,
            prompt_column=prompt_col,
            response_column=response_col,
            experiment_name=experiment_name,
            metrics=selected_metrics,
            domain=domain,
            ground_truth_column=ground_truth_col,
            user_features_column=user_features_col,
            openai_api_key=openai_key
        )
        
        config_path = f"{experiment_name}_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        print(f"✅ Configuration saved to: {config_path}")
        
        return config_path, dataset_path
        
    except Exception as e:
        print(f"❌ Error generating configuration: {e}")
        return None


def run_evaluation(config_path, dataset_path):
    """Run the evaluation with the generated configuration."""
    print("\n🔬 Running Evaluation")
    print("-" * 30)
    
    try:
        # Load configuration
        config = EvaluationConfig(config_path)
        
        # Load dataset
        df = pd.read_csv(dataset_path)
        print(f"✅ Dataset loaded: {len(df)} rows")
        
        # Create evaluator
        evaluator = UniversalEvaluator(config)
        print("✅ Evaluator created")
        
        # Run evaluation
        print("🚀 Starting evaluation...")
        results = evaluator.evaluate_dataset(df)
        
        # Save results
        output_path = f"{config.experiment_name}_results.csv"
        results.to_csv(output_path, index=False)
        print(f"✅ Results saved to: {output_path}")
        
        # Print summary
        print("\n📊 Evaluation Summary")
        print("-" * 30)
        metric_cols = [col for col in results.columns if col.endswith('_score') or col in config.metrics.keys()]
        if metric_cols:
            summary = results[metric_cols].mean()
            for metric, score in summary.items():
                print(f"{metric}: {score:.3f}")
        
        print(f"\n🎉 Evaluation complete! Check MLflow UI for detailed results.")
        print(f"Experiment: {config.experiment_name}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error running evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main interactive function."""
    try:
        # Interactive setup
        setup_result = interactive_setup()
        if not setup_result:
            print("❌ Setup failed. Exiting.")
            return
        
        config_path, dataset_path = setup_result
        
        # Ask if user wants to run evaluation
        if input("\nDo you want to run the evaluation now? (y/n): ").lower().startswith('y'):
            results = run_evaluation(config_path, dataset_path)
            if results is not None:
                print("\n✅ Quick start completed successfully!")
            else:
                print("\n❌ Evaluation failed. Check the error messages above.")
        else:
            print(f"\n✅ Configuration created: {config_path}")
            print("Run the evaluation later with:")
            print(f"python template_evaluation_system.py --config {config_path}")
    
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()