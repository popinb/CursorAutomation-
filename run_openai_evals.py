#!/usr/bin/env python3
"""
Script to run Zillow Home Affordability evaluation using OpenAI Evals framework.

This script provides both programmatic and CLI-compatible execution of the evaluation.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, Optional

def setup_environment():
    """Set up the environment for OpenAI Evals."""
    # Set API credentials
    api_key = os.getenv("OPENAI_API_KEY", "popinb_zillowlabs__hs7x0vTjbLwjKhNStdgL1Dd")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.zillowlabs.com/openai/v1")
    
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_BASE_URL"] = base_url
    
    return api_key, base_url

def run_openai_evals_cli(model: str, eval_name: str, max_samples: Optional[int] = None):
    """
    Run evaluation using OpenAI Evals CLI command.
    
    Args:
        model: Model name to evaluate (e.g., 'gpt-4', 'o1-preview')
        eval_name: Name of the evaluation (e.g., 'zillow-affordability')
        max_samples: Maximum number of samples to evaluate
    """
    import subprocess
    
    print(f"🤖 Running OpenAI Evals CLI")
    print(f"📊 Model: {model}")
    print(f"🔍 Evaluation: {eval_name}")
    if max_samples:
        print(f"📝 Max Samples: {max_samples}")
    print("=" * 50)
    
    # Construct CLI command
    cmd = ["oaieval", model, eval_name]
    if max_samples:
        cmd.extend(["--max_samples", str(max_samples)])
    
    # Add registry path if needed
    cmd.extend(["--registry_path", "."])
    
    try:
        # Run the CLI command
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        print("✅ Evaluation completed successfully!")
        print("\n📋 STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("\n⚠️ STDERR:")
            print(result.stderr)
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Evaluation failed with exit code {e.returncode}")
        print(f"📋 STDOUT:\n{e.stdout}")
        print(f"⚠️ STDERR:\n{e.stderr}")
        
    except FileNotFoundError:
        print("❌ OpenAI Evals CLI not found. Install with: pip install evals")
        print("💡 Alternatively, run programmatic evaluation below.")

def run_programmatic_evaluation():
    """Run the evaluation programmatically without CLI."""
    try:
        from openaieval_zillow_judge import ZillowAffordabilityEval
        
        print("🤖 Running Programmatic OpenAI Evals Evaluation")
        print("=" * 50)
        
        # Setup API credentials
        api_key, base_url = setup_environment()
        
        # Create mock completion function
        class MockCompletionFn:
            def __init__(self, api_key: str, base_url: str = None):
                import openai
                self.client = openai.OpenAI(
                    api_key=api_key,
                    base_url=base_url
                )
            
            def __call__(self, prompt, **kwargs):
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=prompt,
                    **kwargs
                )
                return MockCompletion(response.choices[0].message.content)
        
        class MockCompletion:
            def __init__(self, content):
                self.content = content
            
            def get_completions(self):
                return [self.content]
        
        class MockRecorder:
            def __init__(self):
                self.events = []
            
            def record_event(self, event_type, **kwargs):
                self.events.append({"type": event_type, **kwargs})
        
        # Initialize evaluator
        completion_fn = MockCompletionFn(api_key, base_url)
        evaluator = ZillowAffordabilityEval(
            completion_fns=[completion_fn],
            samples_jsonl="zillow_eval_samples.jsonl",
            name="zillow_affordability_openai_evals"
        )
        
        # Run evaluation
        recorder = MockRecorder()
        results = evaluator.run(recorder)
        
        # Display results
        print(f"\n📊 EVALUATION RESULTS:")
        print("=" * 50)
        print(f"🎯 Average Alpha Score: {results['summary']['avg_alpha_percentage']:.1f}%")
        print(f"📈 Average Full Score: {results['summary']['avg_full_percentage']:.1f}%")
        print(f"✅ Successful Evaluations: {results['summary']['successful_evaluations']}")
        print(f"❌ Error Rate: {results['summary']['error_rate']:.1f}%")
        
        print(f"\n📋 DETAILED SAMPLE RESULTS:")
        print("=" * 50)
        for i, result in enumerate(results["results"], 1):
            if "error" not in result:
                print(f"Sample {i}:")
                print(f"  🎯 Alpha: {result['alpha_score']}/{result['alpha_max']} ({result['alpha_percentage']:.1f}%)")
                print(f"  📊 Full: {result['full_score']}/{result['full_max']} ({result['full_percentage']:.1f}%)")
            else:
                print(f"Sample {i}: ❌ {result['error']}")
        
        # Save results to file
        output_file = "openai_evals_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Results saved to: {output_file}")
        
        return results
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure openaieval_zillow_judge.py is in the current directory")
        return None
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return None

def compare_frameworks():
    """Compare the different evaluation frameworks."""
    print("\n🔍 FRAMEWORK COMPARISON:")
    print("=" * 70)
    
    print("📊 OpenAI Evals Framework:")
    print("  ✅ Standardized evaluation structure")
    print("  ✅ Built-in recording and logging")
    print("  ✅ Registry-based configuration") 
    print("  ✅ CLI integration (oaieval command)")
    print("  ✅ Reproducible evaluation runs")
    print("  ✅ Framework-level parallelization")
    print("  ✅ Built-in metrics aggregation")
    print("  ⚠️  Requires OpenAI Evals installation")
    
    print("\n📊 Custom LLM Judge (previous implementation):")
    print("  ✅ Direct API control")
    print("  ✅ Custom parameter optimization")
    print("  ✅ Flexible evaluation logic")
    print("  ✅ No additional dependencies")
    print("  ⚠️  Manual recording/logging required")
    print("  ⚠️  Custom CLI implementation needed")

def main():
    """Main function to run the evaluation."""
    parser = argparse.ArgumentParser(
        description="Run Zillow Home Affordability evaluation using OpenAI Evals framework"
    )
    parser.add_argument(
        "--mode", 
        choices=["cli", "programmatic", "both"],
        default="programmatic",
        help="Evaluation mode: CLI, programmatic, or both"
    )
    parser.add_argument(
        "--model",
        default="gpt-4", 
        help="Model to evaluate (default: gpt-4)"
    )
    parser.add_argument(
        "--eval-name",
        default="zillow-affordability",
        help="Evaluation name (default: zillow-affordability)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples to evaluate"
    )
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Only set up environment and show configuration"
    )
    
    args = parser.parse_args()
    
    print("🤖 OPENAI EVALS ZILLOW HOME AFFORDABILITY JUDGE")
    print("=" * 70)
    print("Framework-based evaluation with standardized logging and metrics")
    print("=" * 70)
    
    # Setup environment
    api_key, base_url = setup_environment()
    
    print(f"🔧 Environment Configuration:")
    print(f"  API Key: {api_key[:20]}...")
    print(f"  Base URL: {base_url}")
    
    if args.setup_only:
        print("✅ Environment setup complete. Use --mode to run evaluations.")
        return
    
    # Run evaluation based on mode
    if args.mode in ["cli", "both"]:
        print(f"\n🚀 Running CLI Evaluation...")
        run_openai_evals_cli(args.model, args.eval_name, args.max_samples)
    
    if args.mode in ["programmatic", "both"]:
        print(f"\n🚀 Running Programmatic Evaluation...")
        results = run_programmatic_evaluation()
        
        if results and args.mode == "both":
            print(f"\n📊 Evaluation completed in both modes!")
    
    # Show framework comparison
    compare_frameworks()
    
    print(f"\n✅ OpenAI Evals execution completed!")

if __name__ == "__main__":
    main()