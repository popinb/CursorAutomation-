"""
Example: Generalized LLM Evaluation with Flexible Directory Management

This example demonstrates how to use the generalized evaluation system
that replaces hardcoded directory paths with a flexible directory management system.
"""

import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

# Import the generalized evaluation system
from generalized_evaluation import (
    run_evaluation_pipeline,
    auto_generate_evaluation_prompt,
    load_metrics_from_csv,
    MetricType,
    MetricConfig
)
from directory_manager import (
    DirectoryManager,
    DirectoryConfig,
    create_directory_manager,
    get_local_manager,
    get_databricks_manager,
    get_cloud_manager
)


def create_sample_evaluation_data() -> pd.DataFrame:
    """Create sample evaluation data for demonstration."""
    return pd.DataFrame({
        'prompt': [
            'What is my home buying budget?',
            'Can I afford a $400k house?',
            'What factors affect my mortgage rate?',
            'How much should I save for a down payment?',
            'What is PMI and when do I need it?'
        ],
        'response': [
            'Based on your $80k income and $300 monthly debts, your budget is approximately $285k.',
            'With your current financial profile, a $400k house would require $2,800/month payments.',
            'Your credit score, down payment, loan term, and market rates affect your mortgage rate.',
            'Aim for 3-20% down payment. For a $300k house, save $9k-$60k plus closing costs.',
            'PMI is required when down payment is less than 20% of home value.'
        ],
        'ground_truth': [
            'Budget calculation based on 36% DTI rule and user financial profile.',
            'Affordability analysis with payment breakdown and DTI consideration.',
            'Credit score, down payment, loan term, market conditions affect rates.',
            'Down payment recommendations with specific dollar amounts.',
            'PMI explanation with threshold and cost information.'
        ]
    })


def create_sample_metrics_config() -> pd.DataFrame:
    """Create sample metrics configuration for demonstration."""
    return pd.DataFrame({
        'name': [
            'Financial_Accuracy',
            'Personalization_Quality', 
            'Completeness',
            'Clarity',
            'Actionability'
        ],
        'type': [
            'binary',
            'scale_1_5',
            'scale_1_5', 
            'scale_1_5',
            'binary'
        ],
        'description': [
            'Are the financial calculations and recommendations accurate?',
            'How well personalized is the response to the user?',
            'How complete is the response in addressing the question?',
            'How clear and understandable is the response?',
            'Does the response provide actionable next steps?'
        ],
        'grading_rubric': [
            'Score 1 if financial calculations are mathematically correct and recommendations align with industry standards. Score 0 if calculations are wrong or recommendations are inappropriate.',
            'Score 1-5 based on personalization level: 1=generic response, 3=some personalization, 5=highly personalized with specific user data.',
            'Score 1-5 based on completeness: 1=minimal information, 3=adequate coverage, 5=comprehensive and thorough.',
            'Score 1-5 based on clarity: 1=confusing/unclear, 3=mostly clear, 5=crystal clear and well-structured.',
            'Score 1 if response includes specific next steps or actions. Score 0 if response is purely informational without guidance.'
        ],
        'threshold': [1.0, 3.0, 3.0, 3.0, 1.0],
        'ground_truth_column': ['ground_truth'] * 5,
        'ground_truth_file_path': [''] * 5
    })


def demo_local_environment():
    """Demonstrate evaluation in local environment."""
    print("="*60)
    print("🏠 LOCAL ENVIRONMENT DEMO")
    print("="*60)
    
    # Create local directory manager
    local_dm = get_local_manager(workspace_root="./demo_workspace")
    
    # Show environment details
    validation = local_dm.validate_environment()
    print(f"Environment: {validation['environment']}")
    print(f"Workspace: {validation['workspace_root']['path']}")
    print(f"Assets: {validation['assets_dir']['path']}")
    print(f"Output: {validation['output_dir']['path']}")
    
    # Create sample data
    eval_data = create_sample_evaluation_data()
    metrics_config = create_sample_metrics_config()
    
    # Run evaluation
    results = run_evaluation_pipeline(
        evaluation_data=eval_data,
        metrics_config_data=metrics_config,
        judge_model="gpt-4",
        directory_manager=local_dm,
        save_results=True,
        log_to_mlflow=False  # Skip MLflow for local demo
    )
    
    print(f"\n📊 Evaluation completed! Results shape: {results.shape}")
    return results


def demo_databricks_environment():
    """Demonstrate evaluation in Databricks environment."""
    print("="*60)
    print("🧱 DATABRICKS ENVIRONMENT DEMO")
    print("="*60)
    
    # Create Databricks directory manager
    databricks_dm = get_databricks_manager(username="demo_user")
    
    # Show environment details
    validation = databricks_dm.validate_environment()
    print(f"Environment: {validation['environment']}")
    print(f"Workspace: {validation['workspace_root']['path']}")
    print(f"MLflow Experiment: {databricks_dm.get_mlflow_experiment_path()}")
    
    # Create sample data
    eval_data = create_sample_evaluation_data()
    metrics_config = create_sample_metrics_config()
    
    # Run evaluation
    results = run_evaluation_pipeline(
        evaluation_data=eval_data,
        metrics_config_data=metrics_config,
        judge_model="gpt-4-turbo",
        directory_manager=databricks_dm,
        save_results=True,
        log_to_mlflow=False  # Set to True in real Databricks environment
    )
    
    print(f"\n📊 Evaluation completed! Results shape: {results.shape}")
    return results


def demo_cloud_environment():
    """Demonstrate evaluation in cloud environment."""
    print("="*60)
    print("☁️ CLOUD ENVIRONMENT DEMO")
    print("="*60)
    
    # Create cloud directory manager
    cloud_dm = get_cloud_manager(workspace_root="/app/workspace")
    
    # Show environment details
    validation = cloud_dm.validate_environment()
    print(f"Environment: {validation['environment']}")
    print(f"Workspace: {validation['workspace_root']['path']}")
    print(f"Output: {validation['output_dir']['path']}")
    
    # Create sample data
    eval_data = create_sample_evaluation_data()
    metrics_config = create_sample_metrics_config()
    
    # Run evaluation
    results = run_evaluation_pipeline(
        evaluation_data=eval_data,
        metrics_config_data=metrics_config,
        judge_model="claude-3-sonnet",
        directory_manager=cloud_dm,
        save_results=True,
        log_to_mlflow=False
    )
    
    print(f"\n📊 Evaluation completed! Results shape: {results.shape}")
    return results


def demo_custom_configuration():
    """Demonstrate custom directory configuration."""
    print("="*60)
    print("⚙️ CUSTOM CONFIGURATION DEMO")
    print("="*60)
    
    # Create custom configuration
    custom_config = DirectoryConfig(
        workspace_root="/custom/workspace",
        assets_dir="/custom/assets",
        output_dir="/custom/outputs",
        temp_dir="/tmp/custom_eval",
        output_filename_pattern="custom_results_{timestamp}.csv",
        mlflow_experiment_path="/custom/experiments/llm_eval",
        environment="local",
        create_missing_dirs=True
    )
    
    # Create directory manager with custom config
    custom_dm = DirectoryManager(custom_config)
    
    # Show configuration
    print(f"Custom workspace: {custom_dm.workspace_root}")
    print(f"Custom assets: {custom_dm.assets_dir}")
    print(f"Custom output: {custom_dm.output_dir}")
    print(f"Custom temp: {custom_dm.temp_dir}")
    print(f"MLflow experiment: {custom_dm.get_mlflow_experiment_path()}")
    
    # Generate some example paths
    print(f"\nExample paths:")
    print(f"Output file: {custom_dm.get_output_path()}")
    print(f"Log file: {custom_dm.get_log_path()}")
    print(f"Temp file: {custom_dm.get_temp_path('test', '.tmp')}")


def demo_migrating_hardcoded_paths():
    """Demonstrate how to migrate from hardcoded paths."""
    print("="*60)
    print("🔄 MIGRATION FROM HARDCODED PATHS DEMO")
    print("="*60)
    
    # OLD WAY (hardcoded paths - DON'T DO THIS)
    print("❌ Old way (hardcoded paths):")
    old_hardcoded_paths = {
        "output_dir": "/tmp/llm_eval_artifacts",
        "csv_path": "/tmp/llm_eval_artifacts/results_20241022_123456.csv",
        "mlflow_experiment": "/Users/user@company.com/llm_evaluation_experiment",
        "assets_dir": "./assets",
        "golden_responses": "./assets/golden_responses.json"
    }
    
    for key, path in old_hardcoded_paths.items():
        print(f"  {key}: {path}")
    
    print("\n✅ New way (generalized directory management):")
    
    # NEW WAY (generalized directory management)
    dm = create_directory_manager()
    
    new_flexible_paths = {
        "output_dir": str(dm.output_dir),
        "csv_path": str(dm.get_output_path("results.csv")),
        "mlflow_experiment": dm.get_mlflow_experiment_path(),
        "assets_dir": str(dm.assets_dir),
        "golden_responses": str(dm.get_asset_path("golden_responses"))
    }
    
    for key, path in new_flexible_paths.items():
        print(f"  {key}: {path}")
    
    print("\n🎯 Benefits of the new approach:")
    benefits = [
        "✅ Works across different environments (local, Databricks, cloud)",
        "✅ Automatic path resolution based on environment detection",
        "✅ Configurable directory structure",
        "✅ Automatic directory creation",
        "✅ Environment-specific optimizations",
        "✅ Easy to test and maintain",
        "✅ No more hardcoded paths breaking in different environments"
    ]
    
    for benefit in benefits:
        print(f"  {benefit}")


def demo_original_code_transformation():
    """Show how the original hardcoded code transforms to generalized version."""
    print("="*60)
    print("🔀 CODE TRANSFORMATION DEMO")
    print("="*60)
    
    print("Original hardcoded code:")
    print("-" * 40)
    original_code = '''
# OLD CODE (hardcoded paths)
out_dir = "/tmp/llm_eval_artifacts"
os.makedirs(out_dir, exist_ok=True)
csv_path = os.path.join(out_dir, f"results_{int(time.time())}.csv")
results_df.to_csv(csv_path, index=False)

mlflow.set_experiment(f"/Users/{username}/llm_evaluation_experiment")
mlflow.log_artifact(csv_path, artifact_path="tables")
'''
    print(original_code)
    
    print("Transformed generalized code:")
    print("-" * 40)
    transformed_code = '''
# NEW CODE (generalized directory management)
dm = create_directory_manager()  # Auto-detects environment
csv_path = dm.get_output_path("results.csv")  # Environment-appropriate path
results_df.to_csv(csv_path, index=False)

mlflow.set_experiment(dm.get_mlflow_experiment_path())  # Environment-specific
mlflow.log_artifact(str(csv_path), artifact_path=dm.config.mlflow_artifact_path)
'''
    print(transformed_code)
    
    print("Key improvements:")
    improvements = [
        "🎯 No hardcoded paths - works everywhere",
        "🔧 Environment auto-detection", 
        "📁 Automatic directory creation",
        "⚙️ Configurable settings",
        "🧪 Easy testing with different configs",
        "🔒 Better error handling and validation"
    ]
    
    for improvement in improvements:
        print(f"  {improvement}")


def main():
    """Run all demonstrations."""
    print("🚀 GENERALIZED DIRECTORY MANAGEMENT DEMO")
    print("="*60)
    
    try:
        # Demo different environments
        demo_local_environment()
        print("\n")
        
        # Demo custom configuration  
        demo_custom_configuration()
        print("\n")
        
        # Demo migration from hardcoded paths
        demo_migrating_hardcoded_paths()
        print("\n")
        
        # Demo code transformation
        demo_original_code_transformation()
        
        print("\n" + "="*60)
        print("✅ ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        print("\n🎯 Next Steps:")
        next_steps = [
            "1. Replace hardcoded paths in your evaluation code",
            "2. Use create_directory_manager() for auto-detection", 
            "3. Configure DirectoryConfig for custom setups",
            "4. Test in your target environments",
            "5. Enjoy flexible, maintainable code! 🎉"
        ]
        
        for step in next_steps:
            print(f"  {step}")
            
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("This is expected if assets directory doesn't exist.")
        print("The directory manager handles missing files gracefully in production.")


if __name__ == "__main__":
    main()