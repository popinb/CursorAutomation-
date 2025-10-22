"""
Evaluation Configuration Manager

This module provides utilities for managing different evaluation configurations
and making it easy to set up evaluation scenarios for different use cases.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from generalized_evaluator import GeneralizedEvaluator, MetricConfig, MetricType


@dataclass
class EvaluationScenario:
    """Configuration for a specific evaluation scenario."""
    name: str
    description: str
    metrics_config_path: str
    ground_truth_data: Optional[Dict[str, Any]] = None
    assets_directory: Optional[str] = None
    judge_model: str = "gpt-4"
    output_directory: str = "/tmp/evaluation_results"


class EvaluationConfigManager:
    """
    Manager for different evaluation configurations and scenarios.
    """
    
    def __init__(self, config_directory: str = "evaluation_configs"):
        """
        Initialize the configuration manager.
        
        Args:
            config_directory: Directory containing evaluation configuration files
        """
        self.config_directory = Path(config_directory)
        self.config_directory.mkdir(exist_ok=True)
        self.scenarios = {}
        self._load_scenarios()
    
    def create_scenario(
        self,
        name: str,
        description: str,
        metrics: List[Dict[str, Any]],
        ground_truth_data: Optional[Dict[str, Any]] = None,
        assets_directory: Optional[str] = None,
        judge_model: str = "gpt-4"
    ) -> str:
        """
        Create a new evaluation scenario.
        
        Args:
            name: Name of the scenario
            description: Description of what this scenario evaluates
            metrics: List of metric configurations
            ground_truth_data: Optional ground truth data
            assets_directory: Directory containing asset files
            judge_model: LLM model to use for evaluation
            
        Returns:
            Path to the created configuration file
        """
        # Create metrics CSV
        metrics_path = self.config_directory / f"{name}_metrics.csv"
        self._create_metrics_csv(metrics, metrics_path)
        
        # Create scenario configuration
        scenario = EvaluationScenario(
            name=name,
            description=description,
            metrics_config_path=str(metrics_path),
            ground_truth_data=ground_truth_data,
            assets_directory=assets_directory,
            judge_model=judge_model,
            output_directory=f"/tmp/evaluation_results/{name}"
        )
        
        # Save scenario configuration
        scenario_path = self.config_directory / f"{name}_scenario.json"
        with open(scenario_path, 'w') as f:
            json.dump(asdict(scenario), f, indent=2)
        
        self.scenarios[name] = scenario
        print(f"✅ Created evaluation scenario: {name}")
        print(f"   Metrics: {metrics_path}")
        print(f"   Config: {scenario_path}")
        
        return str(scenario_path)
    
    def _create_metrics_csv(self, metrics: List[Dict[str, Any]], output_path: Path):
        """Create a metrics CSV file from metric configurations."""
        import pandas as pd
        
        # Ensure all required fields are present
        for metric in metrics:
            if 'threshold' not in metric:
                metric['threshold'] = 0.5
            if 'ground_truth_column' not in metric:
                metric['ground_truth_column'] = 'ground_truth'
            if 'ground_truth_file_path' not in metric:
                metric['ground_truth_file_path'] = ''
        
        df = pd.DataFrame(metrics)
        df.to_csv(output_path, index=False)
    
    def _load_scenarios(self):
        """Load existing scenario configurations."""
        for config_file in self.config_directory.glob("*_scenario.json"):
            try:
                with open(config_file, 'r') as f:
                    scenario_data = json.load(f)
                    scenario = EvaluationScenario(**scenario_data)
                    self.scenarios[scenario.name] = scenario
            except Exception as e:
                print(f"⚠️ Error loading scenario from {config_file}: {e}")
    
    def get_scenario(self, name: str) -> Optional[EvaluationScenario]:
        """Get a scenario by name."""
        return self.scenarios.get(name)
    
    def list_scenarios(self) -> List[str]:
        """List all available scenarios."""
        return list(self.scenarios.keys())
    
    def create_evaluator(self, scenario_name: str) -> GeneralizedEvaluator:
        """
        Create an evaluator for a specific scenario.
        
        Args:
            scenario_name: Name of the scenario to use
            
        Returns:
            Configured GeneralizedEvaluator instance
        """
        scenario = self.get_scenario(scenario_name)
        if not scenario:
            raise ValueError(f"Scenario '{scenario_name}' not found")
        
        return GeneralizedEvaluator(
            judge_model=scenario.judge_model,
            metrics_config_path=scenario.metrics_config_path,
            ground_truth_data=scenario.ground_truth_data,
            assets_directory=scenario.assets_directory
        )
    
    def create_predefined_scenarios(self):
        """Create some predefined evaluation scenarios."""
        
        # 1. Basic Quality Assessment
        basic_metrics = [
            {
                'name': 'accuracy',
                'type': 'binary',
                'description': 'Evaluates if the response is factually accurate',
                'grading_rubric': 'Check if the response contains correct information and facts. Look for any false statements or misleading information.',
                'threshold': 0.5,
                'ground_truth_column': 'ground_truth',
                'ground_truth_file_path': 'ground_truth_data.csv'
            },
            {
                'name': 'completeness',
                'type': '1-5_scale',
                'description': 'Evaluates how complete the response is',
                'grading_rubric': 'Rate from 1-5: 1=very incomplete, missing key information; 2=mostly incomplete; 3=partially complete; 4=mostly complete; 5=completely addresses all aspects of the question.',
                'threshold': 3.0,
                'ground_truth_column': 'ground_truth',
                'ground_truth_file_path': 'ground_truth_data.csv'
            },
            {
                'name': 'helpfulness',
                'type': 'percentage',
                'description': 'Evaluates how helpful the response is to the user',
                'grading_rubric': 'Rate from 0-100%: Consider how actionable, clear, and useful the response is for the user\'s specific needs.',
                'threshold': 70.0,
                'ground_truth_column': 'ground_truth',
                'ground_truth_file_path': 'ground_truth_data.csv'
            }
        ]
        
        self.create_scenario(
            name="basic_quality",
            description="Basic quality assessment for general LLM responses",
            metrics=basic_metrics
        )
        
        # 2. Financial Advisory Evaluation
        financial_metrics = [
            {
                'name': 'personalization_accuracy',
                'type': 'binary',
                'description': 'Evaluates if the response is personalized to the user\'s financial situation',
                'grading_rubric': 'Check if the response uses specific user financial data (income, debts, down payment, credit score) and provides personalized calculations rather than generic advice.',
                'threshold': 0.5,
                'ground_truth_column': 'user_profile',
                'ground_truth_file_path': 'user_profiles.csv'
            },
            {
                'name': 'calculation_accuracy',
                'type': 'binary',
                'description': 'Evaluates if financial calculations are mathematically correct',
                'grading_rubric': 'Verify that DTI calculations, mortgage payment estimates, and other financial computations are mathematically accurate based on the provided user data.',
                'threshold': 0.5,
                'ground_truth_column': 'user_profile',
                'ground_truth_file_path': 'user_profiles.csv'
            },
            {
                'name': 'compliance',
                'type': 'binary',
                'description': 'Evaluates fair housing and regulatory compliance',
                'grading_rubric': 'Check that the response complies with fair housing laws, avoids discriminatory language, and includes appropriate disclaimers about financial advice.',
                'threshold': 0.5,
                'ground_truth_column': 'compliance_guidelines',
                'ground_truth_file_path': 'compliance_guide.json'
            },
            {
                'name': 'structured_presentation',
                'type': '1-5_scale',
                'description': 'Evaluates the structure and clarity of the response',
                'grading_rubric': 'Rate from 1-5: 1=poor structure, hard to follow; 2=basic structure; 3=clear organization; 4=well-structured with headings/lists; 5=excellent structure with clear sections and formatting.',
                'threshold': 3.0,
                'ground_truth_column': 'ground_truth',
                'ground_truth_file_path': 'ground_truth_data.csv'
            }
        ]
        
        self.create_scenario(
            name="financial_advisory",
            description="Evaluation for financial advisory responses (like Zillow Buyability)",
            metrics=financial_metrics
        )
        
        # 3. Customer Support Evaluation
        support_metrics = [
            {
                'name': 'empathy',
                'type': '1-5_scale',
                'description': 'Evaluates the level of empathy and understanding shown',
                'grading_rubric': 'Rate from 1-5: 1=no empathy, dismissive; 2=minimal empathy; 3=some understanding; 4=good empathy and acknowledgment; 5=excellent empathy and emotional intelligence.',
                'threshold': 3.0,
                'ground_truth_column': 'ground_truth',
                'ground_truth_file_path': 'ground_truth_data.csv'
            },
            {
                'name': 'problem_solving',
                'type': '1-5_scale',
                'description': 'Evaluates the quality of problem-solving approach',
                'grading_rubric': 'Rate from 1-5: 1=no solution provided; 2=generic response; 3=basic solution; 4=good step-by-step approach; 5=comprehensive solution with multiple options.',
                'threshold': 3.0,
                'ground_truth_column': 'ground_truth',
                'ground_truth_file_path': 'ground_truth_data.csv'
            },
            {
                'name': 'clarity',
                'type': 'percentage',
                'description': 'Evaluates how clear and understandable the response is',
                'grading_rubric': 'Rate from 0-100%: Consider use of simple language, clear instructions, and avoidance of jargon.',
                'threshold': 80.0,
                'ground_truth_column': 'ground_truth',
                'ground_truth_file_path': 'ground_truth_data.csv'
            }
        ]
        
        self.create_scenario(
            name="customer_support",
            description="Evaluation for customer support responses",
            metrics=support_metrics
        )
        
        print("✅ Created predefined evaluation scenarios:")
        print("   - basic_quality: General LLM response quality")
        print("   - financial_advisory: Financial advisory responses")
        print("   - customer_support: Customer support responses")


def create_sample_evaluation_data(scenario_name: str, num_samples: int = 5) -> List[Dict[str, Any]]:
    """
    Create sample evaluation data for testing.
    
    Args:
        scenario_name: Name of the scenario to create data for
        num_samples: Number of sample data points to create
        
    Returns:
        List of sample evaluation data
    """
    if scenario_name == "basic_quality":
        return [
            {
                'prompt': 'What is the capital of France?',
                'response': 'The capital of France is Paris.',
                'ground_truth': 'Paris'
            },
            {
                'prompt': 'How do I bake a chocolate cake?',
                'response': 'To bake a chocolate cake, you need flour, sugar, cocoa powder, eggs, butter, and milk. Mix the dry ingredients, then add wet ingredients, and bake at 350°F for 30 minutes.',
                'ground_truth': 'Chocolate cake baking instructions'
            },
            {
                'prompt': 'What is machine learning?',
                'response': 'Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.',
                'ground_truth': 'Machine learning definition'
            }
        ]
    
    elif scenario_name == "financial_advisory":
        return [
            {
                'prompt': 'What factors were considered to calculate my Buyability?',
                'response': 'Your personalized BuyAbility estimate is $318,431, based on your specific financial profile. This calculation uses your $90,000 annual income, $200 monthly debts, $18,000 down payment, and credit score range of 660-719.',
                'ground_truth': 'Buyability calculation factors',
                'user_profile': {
                    'annual_income': 90000,
                    'monthly_debts': 200,
                    'down_payment': 18000,
                    'credit_score': '660-719'
                }
            },
            {
                'prompt': 'Is my Buyability personalized to me?',
                'response': 'Yes, your BuyAbility is personalized to you. We used your income and debts to calculate it.',
                'ground_truth': 'Personalization confirmation',
                'user_profile': {
                    'annual_income': 100000,
                    'monthly_debts': 200,
                    'down_payment': 30000,
                    'credit_score': 'good'
                }
            }
        ]
    
    elif scenario_name == "customer_support":
        return [
            {
                'prompt': 'I can\'t log into my account. What should I do?',
                'response': 'I understand how frustrating it can be when you can\'t access your account. Let me help you resolve this. First, try resetting your password using the "Forgot Password" link on the login page. If that doesn\'t work, please check that you\'re using the correct email address.',
                'ground_truth': 'Account login troubleshooting'
            },
            {
                'prompt': 'My order hasn\'t arrived yet. Can you help?',
                'response': 'I\'m sorry to hear your order is delayed. Let me check the status for you. Could you please provide your order number so I can look up the tracking information and see what\'s happening?',
                'ground_truth': 'Order tracking assistance'
            }
        ]
    
    else:
        # Generic sample data
        return [
            {
                'prompt': f'Sample question {i+1}',
                'response': f'Sample response {i+1}',
                'ground_truth': f'Sample ground truth {i+1}'
            }
            for i in range(num_samples)
        ]


def main():
    """Example usage of the evaluation configuration manager."""
    print("=== Evaluation Configuration Manager ===\n")
    
    # Initialize configuration manager
    config_manager = EvaluationConfigManager()
    
    # Create predefined scenarios
    config_manager.create_predefined_scenarios()
    
    # List available scenarios
    print("\nAvailable scenarios:")
    for scenario_name in config_manager.list_scenarios():
        scenario = config_manager.get_scenario(scenario_name)
        print(f"  - {scenario_name}: {scenario.description}")
    
    # Example: Run evaluation for financial advisory scenario
    print("\n" + "="*60)
    print("Running Financial Advisory Evaluation")
    print("="*60)
    
    # Create evaluator for financial advisory scenario
    evaluator = config_manager.create_evaluator("financial_advisory")
    
    # Create sample data
    sample_data = create_sample_evaluation_data("financial_advisory")
    
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
    
    # Export results
    exported_files = evaluator.export_results(results_df)
    print(f"\n📁 Exported files:")
    for format_name, file_path in exported_files.items():
        print(f"  {format_name}: {file_path}")


if __name__ == "__main__":
    main()