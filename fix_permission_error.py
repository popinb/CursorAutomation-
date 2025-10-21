"""
Solution for fixing the PermissionError when saving LLM evaluation results.
This module provides alternative approaches to handle file saving with proper permissions.
"""

import os
import tempfile
import time
from pathlib import Path

def get_safe_output_directory(base_name="llm_eval_artifacts"):
    """
    Get a safe output directory for saving evaluation results.
    Tries multiple locations in order of preference.
    """
    # Option 1: Try user's home directory
    home_dir = Path.home()
    user_output_dir = home_dir / base_name
    try:
        user_output_dir.mkdir(exist_ok=True)
        # Test write permissions
        test_file = user_output_dir / ".test_write"
        test_file.touch()
        test_file.unlink()
        return str(user_output_dir)
    except (PermissionError, OSError):
        pass
    
    # Option 2: Try current working directory
    cwd_output_dir = Path.cwd() / base_name
    try:
        cwd_output_dir.mkdir(exist_ok=True)
        # Test write permissions
        test_file = cwd_output_dir / ".test_write"
        test_file.touch()
        test_file.unlink()
        return str(cwd_output_dir)
    except (PermissionError, OSError):
        pass
    
    # Option 3: Use system temp directory with proper subdirectory
    temp_base = tempfile.gettempdir()
    user_temp_dir = Path(temp_base) / f"user_{os.getuid()}" / base_name
    try:
        user_temp_dir.mkdir(parents=True, exist_ok=True)
        # Test write permissions
        test_file = user_temp_dir / ".test_write"
        test_file.touch()
        test_file.unlink()
        return str(user_temp_dir)
    except (PermissionError, OSError):
        pass
    
    # Option 4: Create a temporary directory (guaranteed to work)
    temp_dir = tempfile.mkdtemp(prefix=f"{base_name}_")
    return temp_dir


def save_results_safely(results_df, base_filename="results"):
    """
    Save results DataFrame safely, handling permission errors.
    Returns the path where the file was saved.
    """
    # Get a safe output directory
    out_dir = get_safe_output_directory()
    
    # Create filename with timestamp
    timestamp = int(time.time())
    csv_filename = f"{base_filename}_{timestamp}.csv"
    csv_path = os.path.join(out_dir, csv_filename)
    
    # Save the results
    try:
        results_df.to_csv(csv_path, index=False)
        print(f"✅ Results saved successfully to: {csv_path}")
        return csv_path
    except Exception as e:
        # Fallback: Use NamedTemporaryFile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            results_df.to_csv(tmp_file, index=False)
            print(f"✅ Results saved to temporary file: {tmp_file.name}")
            return tmp_file.name


# Modified version of your code section with the fix:
def save_evaluation_results(results_df, mlflow, dbutils, JUDGE_MODEL, EVALUATION_DATA, metric_configs):
    """
    Save evaluation results with proper error handling for permissions.
    """
    # Log to MLflow
    mlflow.set_experiment(f"/Users/{dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()}/llm_evaluation_experiment")
    
    with mlflow.start_run(run_name=f"eval_{JUDGE_MODEL}_{time.strftime('%Y%m%d_%H%M%S')}"):
        mlflow.log_param("judge_model", JUDGE_MODEL)
        mlflow.log_param("num_samples", len(EVALUATION_DATA))
        mlflow.log_param("metrics", ",".join([cfg.name for cfg in metric_configs]))
        
        # Calculate metrics
        total = len(results_df)
        passed = len(results_df[results_df['status'] == '✅'])
        pass_rate = (passed / total) * 100 if total > 0 else 0
        
        mlflow.log_metric("overall_pass_rate", pass_rate)
        mlflow.log_metric("total_evaluations", total)
        mlflow.log_metric("passed_evaluations", passed)
        
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
        
        # Save results using the safe method
        csv_path = save_results_safely(results_df)
        
        # Log artifact to MLflow
        try:
            mlflow.log_artifact(csv_path, artifact_path="tables")
            print(f"✅ Results logged to MLflow")
        except Exception as e:
            print(f"⚠️ Could not log to MLflow: {e}")
            print(f"   Results are still saved locally at: {csv_path}")


# Example of how to use in your original code:
"""
Replace this section in your code:

    # Save results
    out_dir = "/tmp/llm_eval_artifacts"
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"results_{int(time.time())}.csv")
    results_df.to_csv(csv_path, index=False)
    mlflow.log_artifact(csv_path, artifact_path="tables")

With:

    # Save results using the safe method
    csv_path = save_results_safely(results_df)
    mlflow.log_artifact(csv_path, artifact_path="tables")
"""