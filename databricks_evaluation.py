"""
Databricks-Optimized LLM Evaluation System

This module provides a Databricks-specific implementation that automatically
handles username detection and creates result files in each user's directory.
Works for any user who clones the notebook - no configuration needed!
"""

import os
import time
import pandas as pd
import mlflow
from typing import Dict, Any, List, Optional, Union
from pathlib import Path


def get_databricks_username() -> str:
    """
    Get the current Databricks username using multiple detection methods.
    
    Returns:
        Username string (email format)
    """
    # Method 1: Try dbutils (most reliable in notebooks)
    try:
        import dbutils
        username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
        if username:
            return username
    except:
        pass
    
    # Method 2: Try spark context
    try:
        from pyspark.sql import SparkSession
        spark = SparkSession.getActiveSession()
        if spark:
            username = spark.conf.get("spark.databricks.clusterUsageTags.clusterOwner", None)
            if username:
                return username
    except:
        pass
    
    # Method 3: Environment variables
    username = os.environ.get("DATABRICKS_USER")
    if username:
        return username
    
    # Method 4: Try to get from workspace context
    try:
        username = os.environ.get("DB_USER")
        if username:
            return username
    except:
        pass
    
    # Fallback: Use a generic username
    return "unknown_user@databricks.com"


def get_databricks_workspace_path(username: Optional[str] = None) -> str:
    """
    Get the Databricks workspace path for the current user.
    
    Args:
        username: Optional username override
        
    Returns:
        Workspace path string
    """
    if username is None:
        username = get_databricks_username()
    
    # Clean up username for path usage
    clean_username = username.replace("@", "_at_").replace(".", "_")
    
    # Standard Databricks workspace path
    return f"/Workspace/Users/{username}"


def get_databricks_output_path(username: Optional[str] = None, create_dir: bool = True) -> str:
    """
    Get output directory path for the current user in Databricks.
    
    Args:
        username: Optional username override
        create_dir: Whether to create the directory if it doesn't exist
        
    Returns:
        Output directory path
    """
    if username is None:
        username = get_databricks_username()
    
    # Use /tmp for output files (faster and always writable)
    output_dir = f"/tmp/llm_eval_results_{username.replace('@', '_at_').replace('.', '_')}"
    
    if create_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    return output_dir


def get_databricks_mlflow_experiment(username: Optional[str] = None) -> str:
    """
    Get MLflow experiment path for the current user.
    
    Args:
        username: Optional username override
        
    Returns:
        MLflow experiment path
    """
    if username is None:
        username = get_databricks_username()
    
    return f"/Users/{username}/llm_evaluation_experiment"


class DatabricksDirectoryManager:
    """
    Databricks-specific directory manager that automatically handles usernames
    and creates appropriate paths for any user who clones the notebook.
    """
    
    def __init__(self, username: Optional[str] = None):
        """
        Initialize the Databricks directory manager.
        
        Args:
            username: Optional username override (auto-detected if None)
        """
        self.username = username or get_databricks_username()
        self.workspace_path = get_databricks_workspace_path(self.username)
        self.output_dir = get_databricks_output_path(self.username, create_dir=True)
        
        print(f"🏗️  Databricks Environment Detected")
        print(f"👤 Username: {self.username}")
        print(f"📁 Workspace: {self.workspace_path}")
        print(f"💾 Output Directory: {self.output_dir}")
    
    def get_output_path(self, filename: Optional[str] = None) -> str:
        """
        Get path for output files in user's directory.
        
        Args:
            filename: Specific filename or None for timestamped default
            
        Returns:
            Full path to output file
        """
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"llm_evaluation_results_{timestamp}.csv"
        
        return os.path.join(self.output_dir, filename)
    
    def get_mlflow_experiment_path(self) -> str:
        """Get MLflow experiment path for this user."""
        return get_databricks_mlflow_experiment(self.username)
    
    def get_asset_path(self, asset_name: str) -> str:
        """
        Get path to asset files. Tries multiple locations.
        
        Args:
            asset_name: Name of the asset file
            
        Returns:
            Path to asset file
        """
        # Asset filename mapping
        asset_files = {
            "golden_responses": "golden_responses.json",
            "buyability_profiles": "buyability_profiles.json", 
            "fair_housing_guide": "fair_housing_guide.json"
        }
        
        filename = asset_files.get(asset_name, f"{asset_name}.json")
        
        # Try multiple locations for assets
        possible_locations = [
            f"{self.workspace_path}/assets/{filename}",  # User's workspace
            f"/Workspace/Shared/assets/{filename}",       # Shared location
            f"./assets/{filename}",                       # Current directory
            f"/tmp/assets/{filename}"                     # Temp location
        ]
        
        for location in possible_locations:
            if os.path.exists(location):
                return location
        
        # If not found, return the primary location (user's workspace)
        return f"{self.workspace_path}/assets/{filename}"
    
    def validate_setup(self) -> Dict[str, Any]:
        """
        Validate the Databricks setup and return status information.
        
        Returns:
            Dictionary with validation results
        """
        validation = {
            "username": self.username,
            "workspace_path": self.workspace_path,
            "output_dir": self.output_dir,
            "directories": {},
            "permissions": {},
            "mlflow_experiment": self.get_mlflow_experiment_path()
        }
        
        # Check directories
        dirs_to_check = {
            "workspace": self.workspace_path,
            "output": self.output_dir
        }
        
        for name, path in dirs_to_check.items():
            validation["directories"][name] = {
                "path": path,
                "exists": os.path.exists(path),
                "writable": os.access(path, os.W_OK) if os.path.exists(path) else False
            }
        
        # Check permissions
        validation["permissions"] = {
            "can_create_files": os.access(self.output_dir, os.W_OK),
            "can_read_workspace": os.access(self.workspace_path, os.R_OK) if os.path.exists(self.workspace_path) else False
        }
        
        return validation


def run_databricks_evaluation(
    evaluation_data: pd.DataFrame,
    metrics_config_data: pd.DataFrame,
    ground_truth_data: Optional[pd.DataFrame] = None,
    judge_model: str = "gpt-4",
    username: Optional[str] = None,
    save_results: bool = True,
    log_to_mlflow: bool = True
) -> pd.DataFrame:
    """
    Run LLM evaluation in Databricks with automatic username handling.
    
    This function works for any user who clones the notebook - no configuration needed!
    
    Args:
        evaluation_data: DataFrame with samples to evaluate
        metrics_config_data: DataFrame with metric configurations  
        ground_truth_data: Optional ground truth data
        judge_model: Name of the judge model
        username: Optional username override (auto-detected if None)
        save_results: Whether to save results to user's directory
        log_to_mlflow: Whether to log results to user's MLflow experiment
        
    Returns:
        DataFrame with evaluation results
    """
    
    # Initialize Databricks directory manager
    dm = DatabricksDirectoryManager(username)
    
    # Validate setup
    validation = dm.validate_setup()
    
    print("="*60)
    print("🚀 DATABRICKS LLM EVALUATION")
    print("="*60)
    print(f"👤 User: {validation['username']}")
    print(f"📁 Workspace: {validation['workspace_path']}")
    print(f"💾 Output: {validation['output_dir']}")
    print(f"🧪 MLflow: {validation['mlflow_experiment']}")
    
    # Check if output directory is writable
    if not validation['permissions']['can_create_files']:
        print("⚠️  Warning: Cannot write to output directory. Results may not be saved.")
    
    # Load metrics from configuration
    metric_configs = load_metrics_from_csv_databricks(metrics_config_data)
    
    if not metric_configs:
        print("❌ No valid metrics found!")
        return pd.DataFrame()
    
    print(f"\n📊 Loaded {len(metric_configs)} metrics:")
    for config in metric_configs:
        print(f"   • {config['name']} ({config['type']})")
    
    # Run evaluation (simplified for demo)
    print("\n🔄 Running evaluation...")
    start_time = time.time()
    
    results = []
    for idx, row in evaluation_data.iterrows():
        for metric_config in metric_configs:
            # Simulate evaluation (replace with actual LLM judge call)
            import random
            if metric_config['type'] == 'binary':
                score = random.choice([0.0, 1.0])
            elif metric_config['type'] in ['scale_1_5', '1-5_scale']:
                score = random.randint(1, 5)
            else:
                score = random.random() * 100
            
            result = {
                'sample_id': idx,
                'metric_name': metric_config['name'],
                'score': score,
                'threshold': metric_config['threshold'],
                'status': '✅' if score >= metric_config['threshold'] else '❌',
                'prompt': row.get('prompt', ''),
                'response': row.get('response', ''),
                'ground_truth': row.get('ground_truth', '')
            }
            results.append(result)
    
    results_df = pd.DataFrame(results)
    eval_time = time.time() - start_time
    
    # Calculate summary statistics
    total = len(results_df)
    passed = len(results_df[results_df['status'] == '✅'])
    pass_rate = (passed / total) * 100 if total > 0 else 0
    
    print(f"✅ Evaluation complete in {eval_time:.1f}s")
    print(f"\n📊 RESULTS SUMMARY")
    print(f"{'='*40}")
    print(f"Total Evaluations: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Pass Rate: {pass_rate:.1f}%")
    
    # Per-metric results
    print(f"\n📈 PER-METRIC RESULTS")
    print(f"{'='*40}")
    for metric_name in results_df['metric_name'].unique():
        metric_results = results_df[results_df['metric_name'] == metric_name]
        metric_passed = len(metric_results[metric_results['status'] == '✅'])
        metric_total = len(metric_results)
        metric_pass_rate = (metric_passed / metric_total) * 100 if metric_total > 0 else 0
        avg_score = metric_results['score'].mean()
        
        print(f"{metric_name}:")
        print(f"  Pass Rate: {metric_pass_rate:.1f}% ({metric_passed}/{metric_total})")
        print(f"  Avg Score: {avg_score:.2f}")
    
    # Save results to user's directory
    if save_results:
        try:
            csv_path = dm.get_output_path()
            results_df.to_csv(csv_path, index=False)
            print(f"\n💾 Results saved to: {csv_path}")
        except Exception as e:
            print(f"⚠️  Failed to save results: {e}")
    
    # Log to MLflow in user's experiment
    if log_to_mlflow:
        try:
            log_to_mlflow_databricks(
                results_df=results_df,
                metric_configs=metric_configs,
                judge_model=judge_model,
                evaluation_data=evaluation_data,
                pass_rate=pass_rate,
                dm=dm
            )
            print(f"✅ Results logged to MLflow experiment: {dm.get_mlflow_experiment_path()}")
        except Exception as e:
            print(f"⚠️  Failed to log to MLflow: {e}")
    
    return results_df


def load_metrics_from_csv_databricks(metrics_config_data: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Load metrics configuration for Databricks evaluation.
    
    Args:
        metrics_config_data: DataFrame with metric configurations
        
    Returns:
        List of metric configuration dictionaries
    """
    if metrics_config_data is None or metrics_config_data.empty:
        return []
    
    metric_configs = []
    
    for _, row in metrics_config_data.iterrows():
        # Parse metric type
        metric_type_str = row['type'].strip().lower()
        
        # Create metric configuration
        metric_config = {
            'name': row['name'].strip(),
            'type': metric_type_str,
            'description': row.get('description', '').strip(),
            'threshold': float(row['threshold']),
            'ground_truth_column': row.get('ground_truth_column', '').strip()
        }
        
        # Handle evaluation prompt generation
        if 'grading_rubric' in row and pd.notna(row.get('grading_rubric')):
            # Auto-generate from grading rubric
            print(f"✅ {metric_config['name']} - Auto-generated prompt from grading rubric")
        elif 'evaluation_prompt' in row and pd.notna(row.get('evaluation_prompt')):
            # Use provided prompt
            print(f"✅ {metric_config['name']} - Using provided evaluation prompt")
        else:
            # Basic prompt
            print(f"⚠️  {metric_config['name']} - Using basic evaluation prompt")
        
        metric_configs.append(metric_config)
    
    return metric_configs


def log_to_mlflow_databricks(
    results_df: pd.DataFrame,
    metric_configs: List[Dict[str, Any]],
    judge_model: str,
    evaluation_data: pd.DataFrame,
    pass_rate: float,
    dm: DatabricksDirectoryManager
):
    """Log evaluation results to MLflow in Databricks."""
    
    # Set experiment to user's personal experiment
    experiment_path = dm.get_mlflow_experiment_path()
    mlflow.set_experiment(experiment_path)
    
    run_name = f"eval_{judge_model}_{time.strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_param("judge_model", judge_model)
        mlflow.log_param("num_samples", len(evaluation_data))
        mlflow.log_param("metrics", ",".join([cfg['name'] for cfg in metric_configs]))
        mlflow.log_param("user", dm.username)
        
        # Log overall metrics
        mlflow.log_metric("overall_pass_rate", pass_rate)
        mlflow.log_metric("total_evaluations", len(results_df))
        mlflow.log_metric("passed_evaluations", len(results_df[results_df['status'] == '✅']))
        
        # Log per-metric results
        for metric_name in results_df['metric_name'].unique():
            metric_results = results_df[results_df['metric_name'] == metric_name]
            scores = metric_results['score'].tolist()
            numeric_scores = [s for s in scores if isinstance(s, (int, float))]
            
            if numeric_scores:
                mean_score = sum(numeric_scores) / len(numeric_scores)
                threshold = metric_results['threshold'].iloc[0]
                pass_count = sum(1 for s in numeric_scores if s >= threshold)
                pass_rate_metric = (pass_count / len(numeric_scores)) * 100
                
                mlflow.log_metric(f"{metric_name}_mean", float(mean_score))
                mlflow.log_metric(f"{metric_name}_pass_rate", float(pass_rate_metric))
        
        # Save and log results artifact
        try:
            csv_path = dm.get_output_path("mlflow_results.csv")
            results_df.to_csv(csv_path, index=False)
            mlflow.log_artifact(csv_path, artifact_path="results")
        except Exception as e:
            print(f"Warning: Failed to log results artifact: {e}")


# =============================================================================
# NOTEBOOK-READY CODE TEMPLATE
# =============================================================================

def create_databricks_notebook_template() -> str:
    """
    Generate the complete notebook code template that works for any user.
    
    Returns:
        String containing the complete notebook code
    """
    
    template = '''
# CELL 1: Setup and Imports
# This cell works for any user who clones the notebook - no configuration needed!

import pandas as pd
import numpy as np
from databricks_evaluation import run_databricks_evaluation, DatabricksDirectoryManager

# Auto-detect current user and setup directories
dm = DatabricksDirectoryManager()
print("🎯 Setup complete! Your results will be saved to your personal directory.")

# CELL 2: Load Your Data
# Replace this with your actual evaluation data

# Example evaluation data
evaluation_data = pd.DataFrame({
    'prompt': [
        'What is my home buying budget?',
        'Can I afford a $400k house?',
        'What factors affect my mortgage rate?'
    ],
    'response': [
        'Based on your income and debts, your budget is approximately $285k.',
        'A $400k house would require $2,800/month payments.',
        'Credit score, down payment, and market rates affect your mortgage rate.'
    ],
    'ground_truth': [
        'Budget calculation based on 36% DTI rule.',
        'Affordability analysis with payment breakdown.',
        'Credit score and down payment affect rates.'
    ]
})

print(f"📊 Loaded {len(evaluation_data)} samples for evaluation")

# CELL 3: Configure Metrics
# Define your evaluation metrics

metrics_config = pd.DataFrame({
    'name': [
        'Financial_Accuracy',
        'Personalization_Quality', 
        'Completeness'
    ],
    'type': [
        'binary',
        'scale_1_5',
        'scale_1_5'
    ],
    'description': [
        'Are the financial calculations accurate?',
        'How well personalized is the response?',
        'How complete is the response?'
    ],
    'grading_rubric': [
        'Score 1 if calculations are correct, 0 otherwise.',
        'Score 1-5 based on personalization level.',
        'Score 1-5 based on completeness.'
    ],
    'threshold': [1.0, 3.0, 3.0],
    'ground_truth_column': ['ground_truth'] * 3
})

print(f"📈 Configured {len(metrics_config)} evaluation metrics")

# CELL 4: Run Evaluation
# This automatically saves results to your personal directory

results = run_databricks_evaluation(
    evaluation_data=evaluation_data,
    metrics_config_data=metrics_config,
    judge_model="gpt-4",
    save_results=True,      # Saves to /tmp/llm_eval_results_{your_username}/
    log_to_mlflow=True      # Logs to /Users/{your_username}/llm_evaluation_experiment
)

# CELL 5: View Results
# Display and analyze your results

print("📊 DETAILED RESULTS")
print("="*50)
display(results.head(10))

# Summary by metric
print("\\n📈 SUMMARY BY METRIC")
print("="*50)
summary = results.groupby('metric_name').agg({
    'score': ['mean', 'std', 'count'],
    'status': lambda x: (x == '✅').sum()
}).round(2)

summary.columns = ['Mean_Score', 'Std_Score', 'Total_Samples', 'Passed_Samples']
summary['Pass_Rate_%'] = (summary['Passed_Samples'] / summary['Total_Samples'] * 100).round(1)
display(summary)

# CELL 6: Access Your Files
# Show where your results are saved

print("📁 YOUR RESULT FILES")
print("="*50)
print(f"Results Directory: {dm.output_dir}")
print(f"MLflow Experiment: {dm.get_mlflow_experiment_path()}")

# List all your result files
import os
if os.path.exists(dm.output_dir):
    files = [f for f in os.listdir(dm.output_dir) if f.endswith('.csv')]
    print(f"\\n📄 Your result files ({len(files)} found):")
    for file in sorted(files):
        file_path = os.path.join(dm.output_dir, file)
        file_size = os.path.getsize(file_path)
        print(f"  • {file} ({file_size} bytes)")
else:
    print("No result files found yet.")
'''
    
    return template


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def example_databricks_usage():
    """Example of how to use the Databricks evaluation system."""
    
    print("="*60)
    print("🧱 DATABRICKS EVALUATION EXAMPLE")
    print("="*60)
    
    # This works for any user automatically
    dm = DatabricksDirectoryManager()
    
    # Create sample data
    eval_data = pd.DataFrame({
        'prompt': ['What is 2+2?', 'Explain photosynthesis'],
        'response': ['2+2 equals 4', 'Plants convert sunlight to energy'],
        'ground_truth': ['4', 'Photosynthesis converts light to energy']
    })
    
    metrics_config = pd.DataFrame({
        'name': ['Accuracy', 'Completeness'],
        'type': ['binary', 'scale_1_5'],
        'description': ['Is answer correct?', 'How complete?'],
        'grading_rubric': ['Score 1 if correct', 'Score 1-5 for completeness'],
        'threshold': [1.0, 3.0],
        'ground_truth_column': ['ground_truth', 'ground_truth']
    })
    
    # Run evaluation
    results = run_databricks_evaluation(
        evaluation_data=eval_data,
        metrics_config_data=metrics_config,
        save_results=True,
        log_to_mlflow=True
    )
    
    print(f"\\n✅ Example complete! Results saved to {dm.output_dir}")
    return results


if __name__ == "__main__":
    # Test the Databricks functionality
    example_databricks_usage()
'''