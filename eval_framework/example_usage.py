#!/usr/bin/env python3
"""
Example usage of the Evaluation Framework
"""

import os
import asyncio
from pathlib import Path

# Set up environment
os.environ["OPENAI_API_KEY"] = "your-api-key-here"  # Replace with your key

# Import the framework
from evaluation_framework import run_evaluation


def main():
    """Run a simple evaluation example"""
    
    # Path to configuration file
    config_path = "examples/simple_qa_config.yaml"
    
    # Check if config exists
    if not Path(config_path).exists():
        print(f"Error: Configuration file '{config_path}' not found!")
        print("Please create a configuration file first.")
        return
    
    print("Starting evaluation...")
    print(f"Using configuration: {config_path}")
    print("-" * 50)
    
    try:
        # Run the evaluation
        results = run_evaluation(config_path)
        
        print("\n" + "="*50)
        print("EVALUATION COMPLETE!")
        print("="*50)
        
        # Display summary
        print(f"\nTotal samples evaluated: {len(results)}")
        
        # Get metric columns
        metric_cols = [col for col in results.columns 
                      if not col.endswith('_details') 
                      and not col.endswith('_status')
                      and col not in ['prompt', 'response', 'source_file']]
        
        print("\nResults Summary:")
        for metric in metric_cols:
            if metric in results.columns:
                mean_score = results[metric].mean()
                print(f"  {metric}: {mean_score:.3f}")
        
        print(f"\nDetailed results saved to: {Path(config_path).parent / 'results'}")
        
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()