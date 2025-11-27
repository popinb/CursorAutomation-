#!/usr/bin/env python3
"""
Demo script for LLM Judge Application

This script demonstrates how to use the LLM Judge framework to evaluate
responses using custom criteria and knowledge bases.
"""

import os
import json
from pathlib import Path
from llm_judge_app import LLMJudgeApp, LLMJudgeOAIEval, EvaluationMetric, MetricType, JudgeConfig


def create_sample_knowledge_base():
    """Create sample knowledge base files for demonstration."""
    
    # Create knowledge directory
    knowledge_dir = Path("knowledge")
    knowledge_dir.mkdir(exist_ok=True)
    
    # Sample guidelines
    guidelines = {
        "content_guidelines": {
            "accuracy": "All factual claims must be verifiable and current",
            "completeness": "Answers should address all parts of the question",
            "tone": "Professional but accessible tone is preferred",
            "technical_depth": "Adjust technical complexity to user's apparent expertise level"
        },
        "quality_indicators": {
            "good": ["specific examples", "clear explanations", "actionable advice"],
            "bad": ["vague statements", "unsupported claims", "irrelevant information"]
        }
    }
    
    with open(knowledge_dir / "guidelines.json", 'w') as f:
        json.dump(guidelines, f, indent=2)
    
    # Sample examples
    examples_text = """
Example High-Quality Response:
Q: How do I improve my Python code's performance?
A: Here are three specific strategies: 1) Use list comprehensions instead of loops where possible, 2) Profile your code with cProfile to identify bottlenecks, 3) Consider using numpy for numerical operations. For example, [list comprehension example].

Example Low-Quality Response:
Q: How do I improve my Python code's performance?
A: Make it faster by writing better code and using good practices.
    """
    
    with open(knowledge_dir / "examples.txt", 'w') as f:
        f.write(examples_text)
    
    # Sample rubric
    rubric = {
        "scoring_rubric": {
            "5_excellent": "Exceeds expectations with comprehensive, accurate, and highly useful content",
            "4_good": "Meets expectations with accurate and useful content",
            "3_satisfactory": "Partially meets expectations with mostly accurate content",
            "2_needs_improvement": "Below expectations with some inaccuracies or gaps",
            "1_poor": "Significantly below expectations with major issues"
        }
    }
    
    with open(knowledge_dir / "rubric.yaml", 'w') as f:
        import yaml
        yaml.dump(rubric, f, indent=2)
    
    print("✅ Sample knowledge base created in 'knowledge/' directory")


def demo_basic_evaluation():
    """Demonstrate basic evaluation functionality."""
    
    print("\n🔍 Demo 1: Basic Evaluation")
    print("=" * 50)
    
    # Create sample knowledge base if it doesn't exist
    if not Path("knowledge").exists():
        create_sample_knowledge_base()
    
    # Initialize the judge
    try:
        judge = LLMJudgeApp(config_path="config_example.yaml")
        print("✅ LLM Judge initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing judge: {e}")
        print("💡 Make sure you have set OPENAI_API_KEY environment variable")
        return
    
    # Sample evaluation
    question = "What are the best practices for writing clean Python code?"
    
    candidate_answer = """
    Clean Python code follows several key principles:
    
    1. Use descriptive variable names like 'user_count' instead of 'n'
    2. Follow PEP 8 style guidelines for formatting
    3. Write docstrings for functions and classes
    4. Keep functions small and focused on single tasks
    5. Use list comprehensions for simple iterations
    6. Handle exceptions appropriately with try-except blocks
    
    These practices improve code readability and maintainability.
    """
    
    context = {
        "user_expertise": "beginner",
        "code_type": "general_practices"
    }
    
    try:
        result = judge.evaluate(
            candidate_answer=candidate_answer,
            question=question,
            context=context
        )
        
        print(f"Overall Score: {result['overall_score']:.3f}")
        print("\nMetric Scores:")
        for metric_name, metric_data in result['metric_scores'].items():
            print(f"  {metric_name}: {metric_data['normalized_score']:.3f}")
        
        print(f"\nDetailed Evaluation:\n{result['detailed_evaluation'][:500]}...")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")


def demo_batch_evaluation():
    """Demonstrate batch evaluation functionality."""
    
    print("\n📊 Demo 2: Batch Evaluation")
    print("=" * 50)
    
    # Sample data for batch evaluation
    samples = [
        {
            "candidate_answer": "Python is good for data science because it has libraries.",
            "question": "Why is Python popular for data science?",
            "context": {"user_expertise": "beginner"}
        },
        {
            "candidate_answer": "Python excels in data science due to comprehensive libraries like pandas for data manipulation, NumPy for numerical computing, and scikit-learn for machine learning. Its readable syntax and strong community support make it accessible to researchers and analysts.",
            "question": "Why is Python popular for data science?",
            "context": {"user_expertise": "intermediate"}
        },
        {
            "candidate_answer": "Python's data science ecosystem includes pandas, NumPy, matplotlib, seaborn, scikit-learn, TensorFlow, and PyTorch. The language's interpreted nature facilitates rapid prototyping, while Jupyter notebooks provide interactive development environments. Strong integration with databases and APIs enables comprehensive data pipelines.",
            "question": "Why is Python popular for data science?",
            "context": {"user_expertise": "advanced"}
        }
    ]
    
    try:
        if not Path("knowledge").exists():
            create_sample_knowledge_base()
        
        judge = LLMJudgeApp(config_path="config_example.yaml")
        
        batch_results = judge.batch_evaluate(samples)
        
        print(f"Batch Results:")
        print(f"  Total samples: {batch_results['batch_statistics']['total_samples']}")
        print(f"  Successful evaluations: {batch_results['batch_statistics']['successful_evaluations']}")
        print(f"  Average score: {batch_results['batch_statistics']['average_score']:.3f}")
        
        print(f"\nMetric Averages:")
        for metric, avg_score in batch_results['batch_statistics']['metric_averages'].items():
            print(f"  {metric}: {avg_score:.3f}")
            
    except Exception as e:
        print(f"❌ Batch evaluation failed: {e}")


def demo_openai_evals_integration():
    """Demonstrate OpenAI Evals framework integration."""
    
    print("\n🔧 Demo 3: OpenAI Evals Integration")
    print("=" * 50)
    
    try:
        if not Path("knowledge").exists():
            create_sample_knowledge_base()
        
        # Initialize with OpenAI Evals interface
        evals_judge = LLMJudgeOAIEval(config_path="config_example.yaml")
        
        # Sample in OpenAI Evals format
        sample = {
            "input": {
                "candidate_answer": "Machine learning is a subset of AI that uses algorithms to learn patterns from data.",
                "question": "What is machine learning?",
                "context": {"domain": "AI/ML basics"},
                "reference_answer": "Machine learning is a method of data analysis that automates analytical model building."
            }
        }
        
        result = evals_judge.eval_sample(sample)
        
        print(f"OpenAI Evals Result:")
        print(f"  Score: {result['score']:.3f}")
        print(f"  Evaluation completed successfully ✅")
        
        # Batch evaluation with multiple samples
        samples = [sample] * 2  # Duplicate for demo
        batch_result = evals_judge.run_eval(samples)
        
        print(f"\nBatch evaluation:")
        print(f"  Processed {len(samples)} samples")
        print(f"  Average score: {batch_result['batch_statistics']['average_score']:.3f}")
        
    except Exception as e:
        print(f"❌ OpenAI Evals integration failed: {e}")


def demo_custom_configuration():
    """Demonstrate creating a custom judge configuration programmatically."""
    
    print("\n⚙️ Demo 4: Custom Configuration")
    print("=" * 50)
    
    # Create custom metrics
    metrics = [
        EvaluationMetric(
            name="Relevance",
            type=MetricType.SCALE,
            description="How relevant the answer is to the question",
            scoring_criteria={"min_scale": 1, "max_scale": 5},
            weight=2.0
        ),
        EvaluationMetric(
            name="Accuracy",
            type=MetricType.BINARY,
            description="Whether the information is factually correct",
            scoring_criteria={"verification_required": True},
            weight=3.0
        ),
        EvaluationMetric(
            name="Style",
            type=MetricType.CATEGORICAL,
            description="Writing style appropriateness",
            scoring_criteria={"categories": ["Formal", "Casual", "Technical"]},
            weight=1.0
        )
    ]
    
    # Create custom configuration
    config = JudgeConfig(
        name="Custom Code Review Judge",
        description="Evaluates code review comments for quality and helpfulness",
        metrics=metrics,
        knowledge_sources=[],  # No external knowledge sources for this demo
        evaluation_prompt_template="""
You are {judge_name}. {judge_description}

Evaluate this code review comment:

Question: {question}
Comment: {candidate_answer}

Rate on these metrics:
{metrics_description}

Provide scores and justifications for each metric.
        """,
        model="gpt-4",
        temperature=0.1
    )
    
    try:
        # Initialize with custom config
        custom_judge = LLMJudgeApp(config=config)
        
        result = custom_judge.evaluate(
            candidate_answer="This function could be improved by adding error handling and using more descriptive variable names. Consider refactoring the nested loops for better readability.",
            question="Please review this code for quality improvements",
        )
        
        print(f"Custom Judge Evaluation:")
        print(f"  Overall Score: {result['overall_score']:.3f}")
        print(f"  Metrics evaluated: {len(result['metric_scores'])}")
        
        # Generate report
        report = custom_judge.generate_report()
        print(f"\nGenerated Report Preview:")
        print(report[:300] + "...")
        
    except Exception as e:
        print(f"❌ Custom configuration demo failed: {e}")


def demo_export_and_reporting():
    """Demonstrate result export and reporting features."""
    
    print("\n📈 Demo 5: Export and Reporting")
    print("=" * 50)
    
    try:
        if not Path("knowledge").exists():
            create_sample_knowledge_base()
        
        judge = LLMJudgeApp(config_path="config_example.yaml")
        
        # Run a few evaluations
        samples = [
            "Python is a programming language.",
            "Python is a versatile, high-level programming language known for its readability and extensive library ecosystem.",
            "Python is an interpreted, object-oriented programming language with dynamic semantics and strong typing."
        ]
        
        for i, answer in enumerate(samples):
            judge.evaluate(
                candidate_answer=answer,
                question="What is Python?",
                context={"sample_id": i+1}
            )
        
        # Export results
        judge.export_results("evaluation_results.json", format="json")
        judge.export_results("evaluation_results.yaml", format="yaml")
        
        print("✅ Results exported to evaluation_results.json and evaluation_results.yaml")
        
        # Generate report
        report = judge.generate_report()
        
        with open("evaluation_report.md", 'w') as f:
            f.write(report)
        
        print("✅ Report generated and saved to evaluation_report.md")
        print(f"\nReport Preview:")
        print(report[:400] + "...")
        
    except Exception as e:
        print(f"❌ Export/reporting demo failed: {e}")


def main():
    """Run all demonstrations."""
    
    print("🚀 LLM Judge Application Demo")
    print("=" * 50)
    print("This demo shows how to convert Custom GPT evaluations into")
    print("automated LLM-as-a-Judge systems using OpenAI Evals framework.")
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  Warning: OPENAI_API_KEY environment variable not set.")
        print("   Some demos may fail without API access.")
        print("   Set your API key: export OPENAI_API_KEY='your-key-here'")
    
    try:
        demo_basic_evaluation()
        demo_batch_evaluation()
        demo_openai_evals_integration()
        demo_custom_configuration()
        demo_export_and_reporting()
        
        print("\n🎉 All demos completed successfully!")
        print("\nNext steps:")
        print("1. Customize config_example.yaml for your use case")
        print("2. Add your own knowledge sources")
        print("3. Define metrics specific to your evaluation needs")
        print("4. Integrate with your existing OpenAI Evals workflow")
        
    except KeyboardInterrupt:
        print("\n\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")


if __name__ == "__main__":
    main()