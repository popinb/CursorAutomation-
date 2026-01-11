#!/usr/bin/env python3
"""
Demo script showing the evaluation template in action

This script demonstrates the complete evaluation pipeline with mock LLM responses
to show how the template works with the Zillow Co-Pilot metrics.
"""

import sys
import os
import json
import pandas as pd
from unittest.mock import Mock, patch
import tempfile

# Add src to path
sys.path.insert(0, 'src')

from evaluation_template import EvaluationPipeline, EvaluationConfig, DataLoader, LLMJudge, CompositeScorer


class MockEvaluationPipeline(EvaluationPipeline):
    """Modified pipeline that uses mock LLM responses instead of real API calls."""
    
    def __init__(self, config_path: str):
        """Initialize with mock LLM judge."""
        self.config = EvaluationConfig(config_path)
        self.data_loader = DataLoader(self.config)
        self.llm_judge = self._create_mock_judge()
        self.composite_scorer = CompositeScorer(self.config)
    
    def _create_mock_judge(self):
        """Create a mock LLM judge with realistic responses."""
        judge = LLMJudge.__new__(LLMJudge)
        judge.config = self.config
        judge.judges_config = self.config.judges_config
        
        # Create mock models
        mock_model1 = Mock()
        mock_model2 = Mock()
        mock_model1.model_name = "mock-gpt-4o"
        mock_model2.model_name = "mock-gpt-4o-mini"
        
        # Define realistic mock responses for each sample
        self.mock_responses = self._generate_mock_responses()
        
        # Set up response behavior
        def mock_invoke(prompt):
            # Determine which metric and sample based on prompt content
            metric_name = self._identify_metric(prompt)
            sample_idx = self._identify_sample(prompt)
            
            response = Mock()
            response.content = json.dumps(self.mock_responses[sample_idx][metric_name])
            return response
        
        mock_model1.invoke = mock_invoke
        mock_model2.invoke = mock_invoke
        
        judge.models = [mock_model1, mock_model2]
        return judge
    
    def _identify_metric(self, prompt):
        """Identify which metric is being evaluated based on prompt content."""
        if "structured presentation" in prompt.lower():
            return "structured_presentation"
        elif "personalization accuracy" in prompt.lower():
            return "personalization_accuracy"
        elif "actionability" in prompt.lower():
            return "actionability_guidance"
        elif "coherence" in prompt.lower():
            return "coherence"
        else:
            return "structured_presentation"  # default
    
    def _identify_sample(self, prompt):
        """Identify which data sample is being evaluated based on prompt content."""
        # Simple heuristic based on content
        if "Seattle" in prompt:
            return 0
        elif "95,000" in prompt:
            return 1
        elif "Austin" in prompt:
            return 2
        elif "900k" in prompt:
            return 3
        elif "refinance" in prompt.lower():
            return 4
        elif "relocation" in prompt.lower():
            return 5
        elif "FHA" in prompt:
            return 6
        elif "Miami" in prompt:
            return 7
        else:
            return 0  # default
    
    def _generate_mock_responses(self):
        """Generate realistic mock responses for each sample."""
        return [
            # Sample 0: Seattle house search - Good response
            {
                "structured_presentation": {
                    "structured_presentation_score": 5,
                    "explanation": "Excellent structure with clear headings (## Immediate Actions, ## Seattle Market Insights), bullets, and organized sections"
                },
                "personalization_accuracy": {
                    "personalization_accuracy_score": 1,
                    "explanation": "Correctly uses $800k budget and Seattle location from user context"
                },
                "actionability_guidance": {
                    "actionability_guidance_score": 1,
                    "explanation": "Clear next steps: get pre-approved, connect with agent, start home search"
                },
                "coherence": {
                    "coherence_score": 1,
                    "explanation": "Logically consistent throughout with no contradictions"
                }
            },
            # Sample 1: Income affordability - Good personalization
            {
                "structured_presentation": {
                    "structured_presentation_score": 4,
                    "explanation": "Well structured with clear headings and bullets for analysis breakdown"
                },
                "personalization_accuracy": {
                    "personalization_accuracy_score": 1,
                    "explanation": "Accurately uses $95k income, 750 credit score, and calculated DTI ratio"
                },
                "actionability_guidance": {
                    "actionability_guidance_score": 1,
                    "explanation": "Specific next steps provided: get pre-approved, explore neighborhoods, review properties"
                },
                "coherence": {
                    "coherence_score": 1,
                    "explanation": "Consistent financial calculations and logical flow"
                }
            },
            # Sample 2: Austin homes - Poor structure and guidance
            {
                "structured_presentation": {
                    "structured_presentation_score": 2,
                    "explanation": "Poor structure - wall of text with no clear organization, headings, or bullets"
                },
                "personalization_accuracy": {
                    "personalization_accuracy_score": 1,
                    "explanation": "Mentions Austin correctly, no incorrect financial facts stated"
                },
                "actionability_guidance": {
                    "actionability_guidance_score": 0,
                    "explanation": "Vague guidance like 'consider getting pre-approved' without specific next steps"
                },
                "coherence": {
                    "coherence_score": 0,
                    "explanation": "Jumps between topics without clear connection, mentions unrelated info like traffic"
                }
            },
            # Sample 3: Unrealistic budget - Good honest assessment
            {
                "structured_presentation": {
                    "structured_presentation_score": 4,
                    "explanation": "Well organized with clear sections for reality check, risks, and alternatives"
                },
                "personalization_accuracy": {
                    "personalization_accuracy_score": 1,
                    "explanation": "Correctly uses $60k income and $900k target, calculations are accurate"
                },
                "actionability_guidance": {
                    "actionability_guidance_score": 1,
                    "explanation": "Provides realistic alternatives and specific next steps for the user's situation"
                },
                "coherence": {
                    "coherence_score": 1,
                    "explanation": "Logical progression from problem identification to solutions"
                }
            },
            # Sample 4: Refinancing - Poor personalization
            {
                "structured_presentation": {
                    "structured_presentation_score": 2,
                    "explanation": "Minimal structure, mostly generic bullet points without clear organization"
                },
                "personalization_accuracy": {
                    "personalization_accuracy_score": 0,
                    "explanation": "Fails to use specific user data like current 6.5% rate, loan balance, or credit score"
                },
                "actionability_guidance": {
                    "actionability_guidance_score": 0,
                    "explanation": "Generic advice without specific next steps tailored to user's situation"
                },
                "coherence": {
                    "coherence_score": 1,
                    "explanation": "Generally coherent but repetitive without contradictions"
                }
            },
            # Sample 5: Quick sale - Excellent response
            {
                "structured_presentation": {
                    "structured_presentation_score": 5,
                    "explanation": "Excellent structure with timeline, action items, and clear decision framework"
                },
                "personalization_accuracy": {
                    "personalization_accuracy_score": 1,
                    "explanation": "Uses correct home value ($420k) and location (Denver) from context"
                },
                "actionability_guidance": {
                    "actionability_guidance_score": 1,
                    "explanation": "Very specific timeline and actionable steps for quick sale strategy"
                },
                "coherence": {
                    "coherence_score": 1,
                    "explanation": "Logical flow from urgent need to strategic timeline to specific actions"
                }
            },
            # Sample 6: Loan comparison - Poor structure
            {
                "structured_presentation": {
                    "structured_presentation_score": 2,
                    "explanation": "Poor structure with repetitive sentences and no clear organization"
                },
                "personalization_accuracy": {
                    "personalization_accuracy_score": 1,
                    "explanation": "No specific financial facts used, but none incorrectly stated either"
                },
                "actionability_guidance": {
                    "actionability_guidance_score": 0,
                    "explanation": "Generic information without specific guidance for user's situation"
                },
                "coherence": {
                    "coherence_score": 0,
                    "explanation": "Repetitive content with overlapping information that reduces clarity"
                }
            },
            # Sample 7: Investment property - Good specialized response
            {
                "structured_presentation": {
                    "structured_presentation_score": 5,
                    "explanation": "Excellent structure with market overview, strategy, considerations, and next steps"
                },
                "personalization_accuracy": {
                    "personalization_accuracy_score": 1,
                    "explanation": "Correctly uses $1.2M budget and Miami location, appropriate for experienced investor"
                },
                "actionability_guidance": {
                    "actionability_guidance_score": 1,
                    "explanation": "Specific next steps: market analysis, local connections, financing review"
                },
                "coherence": {
                    "coherence_score": 1,
                    "explanation": "Logical flow from market overview to strategy to implementation"
                }
            }
        ]


def run_demo_evaluation():
    """Run a complete evaluation demo with mock responses."""
    
    print("🏠 ZILLOW CO-PILOT EVALUATION DEMO")
    print("=" * 60)
    print("This demo shows the evaluation template in action with realistic")
    print("mock responses to demonstrate the complete pipeline.\n")
    
    # Initialize mock pipeline
    print("🔧 Initializing evaluation pipeline...")
    pipeline = MockEvaluationPipeline('config/zillow_copilot_evaluation.yaml')
    
    # Load and display dataset info
    print("📊 Loading evaluation dataset...")
    df = pipeline.data_loader.load_dataset()
    print(f"   Loaded {len(df)} real estate assistant responses")
    print(f"   Columns: {list(df.columns)}")
    
    # Run evaluation
    print("\n🎯 Running evaluation with 4 Zillow metrics...")
    
    for metric_name, metric_config in pipeline.config.metrics_config.items():
        if not metric_config.get('enabled', True):
            continue
            
        print(f"\n   📏 Evaluating {metric_name}...")
        print(f"      Description: {metric_config['description']}")
        
        # Run evaluation for this metric
        eval_results = pipeline.llm_judge.evaluate_single_metric(
            metric_name, metric_config, df
        )
        
        # Add results to dataframe
        df[metric_name] = eval_results['scores']
        df[f"{metric_name}_details"] = eval_results['details']
        
        # Add threshold-based status
        threshold = metric_config.get('threshold', 0)
        df[f"{metric_name}_status"] = ["✅" if s >= threshold else "❌" for s in eval_results['scores']]
        
        print(f"      Mean score: {eval_results['mean_score']:.2f}")
        print(f"      Pass rate: {(df[f'{metric_name}_status'] == '✅').mean()*100:.1f}%")
    
    # Compute composite scores
    print(f"\n📈 Computing composite scores...")
    df = pipeline.composite_scorer.compute_composite_scores(df)
    
    # Display detailed results
    print(f"\n📋 DETAILED EVALUATION RESULTS")
    print("=" * 60)
    
    for idx, row in df.iterrows():
        print(f"\n🔍 Sample {idx + 1}: {row['user_query'][:80]}...")
        print(f"   Response quality: {row['copilot_response'][:100]}...")
        print()
        
        # Show metric scores
        metrics = ['structured_presentation', 'personalization_accuracy', 'actionability_guidance', 'coherence']
        for metric in metrics:
            score = row[metric]
            status = row[f"{metric}_status"]
            explanation = row[f"{metric}_details"]['final_score']
            threshold = pipeline.config.metrics_config[metric]['threshold']
            
            print(f"   {metric.replace('_', ' ').title()}: {score} {status} (threshold: {threshold})")
        
        # Show composite scores
        print(f"   Overall Quality: {row['overall_quality']:.2f}")
        print(f"   Co-Pilot Readiness: {row['copilot_readiness']:.0f}")
    
    # Summary statistics
    print(f"\n📊 SUMMARY STATISTICS")
    print("=" * 60)
    
    metrics = ['structured_presentation', 'personalization_accuracy', 'actionability_guidance', 'coherence']
    for metric in metrics:
        mean_score = df[metric].mean()
        threshold = pipeline.config.metrics_config[metric]['threshold']
        pass_rate = (df[metric] >= threshold).mean() * 100
        
        print(f"{metric.replace('_', ' ').title():<25} Mean: {mean_score:.2f}  Pass Rate: {pass_rate:.1f}%")
    
    print(f"\nComposite Scores:")
    print(f"Overall Quality (avg):     {df['overall_quality'].mean():.2f}")
    print(f"Co-Pilot Readiness (pass): {(df['copilot_readiness'] >= 1).sum()}/{len(df)} responses")
    
    # Save results
    output_path = "results/demo_evaluation_results.csv"
    os.makedirs("results", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n💾 Results saved to: {output_path}")
    
    return df


def analyze_results(df):
    """Analyze and interpret the evaluation results."""
    
    print(f"\n🔍 RESULTS ANALYSIS")
    print("=" * 60)
    
    # Find best and worst performing responses
    best_idx = df['overall_quality'].idxmax()
    worst_idx = df['overall_quality'].idxmin()
    
    print(f"\n🏆 BEST PERFORMING RESPONSE (Sample {best_idx + 1}):")
    print(f"   Query: {df.iloc[best_idx]['user_query'][:100]}...")
    print(f"   Overall Quality: {df.iloc[best_idx]['overall_quality']:.2f}")
    print(f"   Strengths: High structure and clear guidance")
    
    print(f"\n🚨 LOWEST PERFORMING RESPONSE (Sample {worst_idx + 1}):")
    print(f"   Query: {df.iloc[worst_idx]['user_query'][:100]}...")
    print(f"   Overall Quality: {df.iloc[worst_idx]['overall_quality']:.2f}")
    print(f"   Issues: Poor structure and vague guidance")
    
    # Metric-specific insights
    print(f"\n📊 METRIC INSIGHTS:")
    
    # Structure insights
    struct_scores = df['structured_presentation']
    print(f"   Structured Presentation: {struct_scores.mean():.1f}/5 avg")
    print(f"     {(struct_scores >= 4).sum()}/{len(df)} responses well-structured")
    
    # Personalization insights  
    pers_scores = df['personalization_accuracy']
    print(f"   Personalization Accuracy: {pers_scores.mean():.1f}/1 avg")
    print(f"     {pers_scores.sum()}/{len(df)} responses accurate")
    
    # Actionability insights
    action_scores = df['actionability_guidance']
    print(f"   Actionability & Guidance: {action_scores.mean():.1f}/1 avg")
    print(f"     {action_scores.sum()}/{len(df)} responses actionable")
    
    # Coherence insights
    coh_scores = df['coherence']
    print(f"   Coherence: {coh_scores.mean():.1f}/1 avg")
    print(f"     {coh_scores.sum()}/{len(df)} responses coherent")
    
    # Production readiness
    ready_count = (df['copilot_readiness'] >= 1).sum()
    print(f"\n🚀 PRODUCTION READINESS:")
    print(f"   {ready_count}/{len(df)} responses meet all minimum standards")
    print(f"   Success rate: {ready_count/len(df)*100:.1f}%")


def main():
    """Main demo execution."""
    try:
        # Run the evaluation demo
        results_df = run_demo_evaluation()
        
        # Analyze the results
        analyze_results(results_df)
        
        print(f"\n🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("This demo showed how the evaluation template:")
        print("✅ Loads and processes real estate assistant conversations")
        print("✅ Evaluates responses using 4 specialized Zillow metrics")
        print("✅ Computes composite scores for overall assessment")
        print("✅ Provides detailed insights and production readiness metrics")
        print("✅ Saves structured results for further analysis")
        print("\nThe template is ready for production use with actual LLM APIs!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()