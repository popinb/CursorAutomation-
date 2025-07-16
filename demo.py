#!/usr/bin/env python3
"""
Custom GPT Judge Evaluator - Demonstration Script
=================================================

This script demonstrates how to use the Custom GPT Judge Evaluator system
to convert a Custom GPT setup into the OpenAI evals framework.

Run this script to see the system in action with sample evaluation data.
"""

import os
import asyncio
import json
from pathlib import Path

# Import our custom evaluator
from custom_gpt_judge_eval import (
    CustomGPTEvaluator,
    EvalDataLoader,
    EvalSample
)


async def run_basic_demo():
    """Run a basic demonstration of the Custom GPT Judge Evaluator."""
    
    print("🤖 Custom GPT Judge Evaluator - Demo")
    print("=" * 50)
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: Please set OPENAI_API_KEY environment variable")
        print("   You can get an API key from: https://platform.openai.com/api-keys")
        return
    
    print("✅ OpenAI API key found")
    
    # Initialize the evaluator
    print("\n🔧 Initializing Custom GPT Judge Evaluator...")
    
    knowledge_sources = [
        "evaluation_guidelines.pdf",
        "quality_standards.txt", 
        "scoring_rubric.md"
    ]
    
    evaluator = CustomGPTEvaluator(
        api_key=api_key,
        model="gpt-4",  # Use GPT-4 for best evaluation quality
        knowledge_sources=knowledge_sources
    )
    
    print(f"   Model: {evaluator.judge.model}")
    print(f"   Knowledge Sources: {len(knowledge_sources)} files")
    print(f"   Eval Name: {evaluator.eval_name}")
    
    # Load sample data
    print("\n📊 Loading evaluation samples...")
    
    # First try to load from JSONL file, fall back to sample data
    jsonl_file = "custom_gpt_samples.jsonl"
    if Path(jsonl_file).exists():
        samples = EvalDataLoader.load_from_jsonl(jsonl_file)
        print(f"   Loaded {len(samples)} samples from {jsonl_file}")
    else:
        samples = EvalDataLoader.create_sample_data()
        print(f"   Using {len(samples)} built-in sample data points")
    
    # Show sample categories
    categories = set()
    for sample in samples:
        if sample.metadata and "category" in sample.metadata:
            categories.add(sample.metadata["category"])
    
    print(f"   Categories: {', '.join(sorted(categories))}")
    
    # Run evaluation
    print(f"\n🔍 Running evaluation on {len(samples)} samples...")
    print("   This may take a few minutes depending on API response times...")
    
    try:
        results = await evaluator.evaluate_samples(samples)
        print(f"   ✅ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"   ❌ Evaluation failed: {str(e)}")
        return
    
    # Generate and display report
    print("\n📈 Generating evaluation report...")
    report = evaluator.generate_report(results)
    
    # Display summary
    print("\n" + "=" * 50)
    print("📋 EVALUATION SUMMARY")
    print("=" * 50)
    print(f"📊 Total Samples: {report['total_samples']}")
    print(f"✅ Passed: {report['passed_samples']}")
    print(f"❌ Failed: {report['failed_samples']}")
    print(f"📈 Pass Rate: {report['pass_rate']:.1%}")
    print(f"⭐ Average Score: {report['average_score']:.2f}/5.0")
    
    # Score distribution
    print(f"\n📊 Score Distribution:")
    for score_range, count in report['score_distribution'].items():
        percentage = (count / report['total_samples']) * 100
        bar = "█" * int(percentage / 5)  # Simple bar chart
        print(f"   {score_range}: {count:2d} samples ({percentage:4.1f}%) {bar}")
    
    # Show detailed results for first few samples
    print(f"\n🔍 Sample Results (showing first 3):")
    print("-" * 50)
    
    for i, result in enumerate(report['detailed_results'][:3]):
        print(f"\n📝 Sample {i+1}:")
        print(f"   Score: {result['score']}/5.0")
        print(f"   Status: {'✅ PASSED' if result['passed'] else '❌ FAILED'}")
        
        # Truncate reasoning for display
        reasoning = result['reasoning'] or "No reasoning provided"
        if len(reasoning) > 150:
            reasoning = reasoning[:150] + "..."
        print(f"   Reasoning: {reasoning}")
    
    # Save detailed report
    output_file = "custom_gpt_evaluation_report.json"
    print(f"\n💾 Saving detailed report to: {output_file}")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"   ✅ Report saved successfully!")
    except Exception as e:
        print(f"   ❌ Failed to save report: {str(e)}")
    
    print(f"\n🎉 Demo completed! Check {output_file} for full results.")


async def run_custom_sample_demo():
    """Demonstrate evaluating a custom sample."""
    
    print("\n" + "=" * 50)
    print("🧪 CUSTOM SAMPLE EVALUATION DEMO")
    print("=" * 50)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Skipping custom sample demo - no API key")
        return
    
    # Initialize evaluator
    evaluator = CustomGPTEvaluator(api_key=api_key)
    
    # Create a custom evaluation sample
    custom_sample = EvalSample(
        input=[
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "Explain what a Python function is and give a simple example."}
        ],
        ideal="A function is a reusable block of code that performs a specific task.",
        metadata={
            "response_to_evaluate": """
A Python function is a reusable block of code that performs a specific task. 
Functions help organize code and avoid repetition. Here's a simple example:

def greet(name):
    return f"Hello, {name}!"

# Using the function
message = greet("Alice")
print(message)  # Output: Hello, Alice!

This function takes a name as input and returns a greeting message.
            """.strip(),
            "evaluation_criteria": "Check for technical accuracy, clarity, and inclusion of a practical example.",
            "category": "programming"
        }
    )
    
    print("📝 Evaluating custom sample:")
    print(f"   Question: {custom_sample.input[1]['content']}")
    print(f"   Response length: {len(custom_sample.metadata['response_to_evaluate'])} characters")
    
    # Evaluate the custom sample
    try:
        result = await evaluator.evaluate_single_sample(custom_sample)
        
        print(f"\n🎯 Evaluation Result:")
        print(f"   Score: {result.score}/5.0")
        print(f"   Status: {'✅ PASSED' if result.passed else '❌ FAILED'}")
        print(f"   Reasoning: {result.reasoning}")
        
    except Exception as e:
        print(f"   ❌ Evaluation failed: {str(e)}")


def display_system_info():
    """Display information about the Custom GPT Judge system."""
    
    print("\n" + "=" * 50)
    print("ℹ️  SYSTEM INFORMATION")
    print("=" * 50)
    
    print("🏗️  Architecture:")
    print("   ├── CustomGPTKnowledgeBase - Simulates knowledge files")
    print("   ├── CustomGPTJudge - Core evaluation logic")
    print("   ├── CustomGPTEvaluator - OpenAI evals interface")
    print("   └── EvalDataLoader - Data handling utilities")
    
    print("\n📚 Knowledge Sources (simulated):")
    print("   ├── evaluation_guidelines.pdf - General methodology")
    print("   ├── quality_standards.txt - Quality criteria")
    print("   └── scoring_rubric.md - Scoring guidelines")
    
    print("\n⚙️  Features:")
    print("   ✅ Custom GPT instruction simulation")
    print("   ✅ Knowledge base integration")
    print("   ✅ Chain-of-thought reasoning")
    print("   ✅ Structured scoring (0-5 scale)")
    print("   ✅ OpenAI evals compatibility")
    print("   ✅ Comprehensive reporting")
    print("   ✅ Async processing")
    print("   ✅ Error handling & resilience")


async def main():
    """Main demonstration function."""
    
    print("🎬 Custom GPT Judge Evaluator - Complete Demo")
    print("=" * 60)
    
    # Display system information
    display_system_info()
    
    # Run basic demo
    await run_basic_demo()
    
    # Run custom sample demo
    await run_custom_sample_demo()
    
    print("\n" + "=" * 60)
    print("✨ Demo completed! Key takeaways:")
    print("   • Custom GPT instructions converted to structured evaluator")
    print("   • Knowledge sources simulated and integrated")
    print("   • LLM-as-a-judge methodology implemented")
    print("   • Full OpenAI evals framework compatibility")
    print("   • Comprehensive evaluation reports generated")
    print("\n🚀 Ready to adapt this system for your specific use case!")


if __name__ == "__main__":
    # Check Python version
    import sys
    if sys.version_info < (3, 8):
        print("❌ This script requires Python 3.8 or higher")
        sys.exit(1)
    
    # Run the demonstration
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {str(e)}")
        print("💡 Make sure you have set OPENAI_API_KEY and installed requirements.txt")