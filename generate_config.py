"""
Configuration Generator
======================

Utility to generate configuration files for the evaluation system.
This helps users quickly set up evaluation configurations without manually editing YAML.
"""

import argparse
import yaml
from pathlib import Path
from metric_templates import METRIC_TEMPLATES, DOMAIN_TEMPLATES, list_available_metrics, list_available_domains


def generate_config(
    dataset_path: str,
    prompt_column: str,
    response_column: str,
    experiment_name: str,
    metrics: list = None,
    domain: str = None,
    ground_truth_column: str = None,
    user_features_column: str = None,
    response_model: str = "GOLDEN_RESPONSE",
    judge_models: list = None,
    openai_api_key: str = "your-openai-api-key",
    openai_base_url: str = "https://api.openai.com/v1"
):
    """Generate a configuration file for the evaluation system."""
    
    if judge_models is None:
        judge_models = ["gpt-4o", "gpt-4"]
    
    if metrics is None:
        if domain:
            metrics = list_available_metrics(domain)
        else:
            metrics = ["accuracy", "relevance", "helpfulness"]
    
    # Get metric configurations
    metric_configs = {}
    for metric in metrics:
        try:
            metric_configs[metric] = get_metric_template(metric, domain)
        except ValueError as e:
            print(f"Warning: {e}")
            continue
    
    config = {
        "dataset": {
            "path": dataset_path,
            "prompt_column": prompt_column,
            "response_column": response_column
        },
        "models": {
            "response_model": response_model,
            "judge_models": judge_models
        },
        "api_keys": {
            "openai_api_key": openai_api_key,
            "openai_base_url": openai_base_url
        },
        "evaluation": {
            "experiment_name": experiment_name,
            "run_name": f"{experiment_name}_run_1",
            "metrics": metric_configs
        }
    }
    
    # Add optional columns
    if ground_truth_column:
        config["dataset"]["ground_truth_column"] = ground_truth_column
    
    if user_features_column:
        config["dataset"]["user_features_column"] = user_features_column
    
    return config


def get_metric_template(metric_name: str, domain: str = None):
    """Get metric template from the templates module."""
    if domain and domain in DOMAIN_TEMPLATES:
        if metric_name in DOMAIN_TEMPLATES[domain]:
            return DOMAIN_TEMPLATES[domain][metric_name]
    
    if metric_name in METRIC_TEMPLATES:
        return METRIC_TEMPLATES[metric_name]
    
    raise ValueError(f"Metric template '{metric_name}' not found")


def main():
    """Command-line interface for configuration generation."""
    parser = argparse.ArgumentParser(description="Generate evaluation configuration files")
    
    # Required arguments
    parser.add_argument("--dataset", required=True, help="Path to dataset CSV file")
    parser.add_argument("--prompt-col", required=True, help="Name of prompt column")
    parser.add_argument("--response-col", required=True, help="Name of response column")
    parser.add_argument("--experiment", required=True, help="MLflow experiment name")
    parser.add_argument("--output", required=True, help="Output configuration file path")
    
    # Optional arguments
    parser.add_argument("--metrics", nargs="+", help="List of metrics to evaluate")
    parser.add_argument("--domain", help="Domain for specialized metrics")
    parser.add_argument("--ground-truth-col", help="Name of ground truth column")
    parser.add_argument("--user-features-col", help="Name of user features column")
    parser.add_argument("--response-model", default="GOLDEN_RESPONSE", 
                       choices=["GOLDEN_RESPONSE", "LLM_gpt-4o", "LLM_gpt-4", "FIRST_CALL"],
                       help="Model for response generation")
    parser.add_argument("--judge-models", nargs="+", default=["gpt-4o", "gpt-4"],
                       help="Models to use for evaluation")
    parser.add_argument("--openai-key", default="your-openai-api-key",
                       help="OpenAI API key")
    parser.add_argument("--openai-url", default="https://api.openai.com/v1",
                       help="OpenAI API base URL")
    
    args = parser.parse_args()
    
    # List available options if requested
    if args.metrics is None:
        print("Available metrics:")
        if args.domain:
            print(f"Domain-specific metrics for '{args.domain}':")
            for metric in list_available_metrics(args.domain):
                print(f"  - {metric}")
        else:
            print("General metrics:")
            for metric in list_available_metrics():
                print(f"  - {metric}")
        
        print("\nAvailable domains:")
        for domain in list_available_domains():
            print(f"  - {domain}")
        
        print("\nUse --metrics to specify which metrics to include in the configuration.")
        return
    
    # Generate configuration
    config = generate_config(
        dataset_path=args.dataset,
        prompt_column=args.prompt_col,
        response_column=args.response_col,
        experiment_name=args.experiment,
        metrics=args.metrics,
        domain=args.domain,
        ground_truth_column=args.ground_truth_col,
        user_features_column=args.user_features_col,
        response_model=args.response_model,
        judge_models=args.judge_models,
        openai_api_key=args.openai_key,
        openai_base_url=args.openai_url
    )
    
    # Save configuration
    with open(args.output, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"✅ Configuration saved to {args.output}")
    print(f"📊 Metrics included: {', '.join(config['evaluation']['metrics'].keys())}")
    if args.domain:
        print(f"🏷️  Domain: {args.domain}")


if __name__ == "__main__":
    main()