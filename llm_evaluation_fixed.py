#!/usr/bin/env python3
"""
Fixed LLM Evaluation Script with Proper File Handling

This script fixes the PermissionError issue when saving evaluation results
by implementing better directory and file handling practices.
"""

import os
import time
import tempfile
import pandas as pd
from pathlib import Path


def auto_generate_evaluation_prompt(metric_name: str, metric_type: str, description: str, grading_rubric: str) -> str:
    """
    Auto-generate evaluation prompt from grading rubric.
    PM just provides grading rubric, code handles the rest!
    """
    
    # Add grading rubric section
    rubric_section = f"\n**Grading Rubric:**\n{grading_rubric}\n" if grading_rubric else ""
    
    # Build complete evaluation prompt
    prompt = f"""You are an expert evaluator. Your task: {description}

{rubric_section}
**Evaluation Details:**
- User Query: {{prompt}}
- AI Response: {{response}}
- Ground Truth Reference: {{ground_truth}}

**Instructions:**
Carefully evaluate the AI response using the grading rubric above.

**Required Output Format:**
Return ONLY a valid JSON object with these two fields:
{{
  "score": <your_score>,
  "explanation": "Brief explanation of your score"
}}

Do not include any other text outside the JSON object."""
    
    return prompt


def create_safe_output_directory():
    """
    Create a safe output directory for saving results.
    
    Returns:
        str: Path to the created directory
    """
    # Try multiple directory options in order of preference
    possible_dirs = [
        # 1. Current workspace directory (most reliable)
        os.path.join(os.getcwd(), "llm_eval_results"),
        # 2. User's home directory
        os.path.expanduser("~/llm_eval_results"),
        # 3. System temp directory with user permissions
        os.path.join(tempfile.gettempdir(), f"llm_eval_results_{os.getuid() if hasattr(os, 'getuid') else 'user'}"),
        # 4. Fallback to Python's tempfile
        tempfile.mkdtemp(prefix="llm_eval_")
    ]
    
    for directory in possible_dirs:
        try:
            # Create directory with proper permissions
            Path(directory).mkdir(parents=True, exist_ok=True)
            
            # Test write permissions by creating a test file
            test_file = os.path.join(directory, "test_write.tmp")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            
            print(f"✅ Using output directory: {directory}")
            return directory
            
        except (PermissionError, OSError) as e:
            print(f"⚠️ Cannot use directory {directory}: {e}")
            continue
    
    # If all else fails, use current directory
    fallback_dir = os.getcwd()
    print(f"🔄 Falling back to current directory: {fallback_dir}")
    return fallback_dir


def safe_save_results(results_df, output_dir: str, filename_prefix: str = "results") -> str:
    """
    Safely save results DataFrame to CSV with error handling.
    
    Args:
        results_df: DataFrame containing results
        output_dir: Directory to save the file
        filename_prefix: Prefix for the filename
        
    Returns:
        str: Path to the saved file, or None if failed
    """
    timestamp = int(time.time())
    filename = f"{filename_prefix}_{timestamp}.csv"
    file_path = os.path.join(output_dir, filename)
    
    try:
        # Save with proper error handling
        results_df.to_csv(file_path, index=False)
        print(f"✅ Results saved to: {file_path}")
        return file_path
        
    except PermissionError as e:
        print(f"❌ Permission denied saving to {file_path}: {e}")
        
        # Try alternative filename with random suffix
        import random
        alt_filename = f"{filename_prefix}_{timestamp}_{random.randint(1000, 9999)}.csv"
        alt_path = os.path.join(output_dir, alt_filename)
        
        try:
            results_df.to_csv(alt_path, index=False)
            print(f"✅ Results saved to alternative path: {alt_path}")
            return alt_path
        except Exception as e2:
            print(f"❌ Failed to save to alternative path: {e2}")
            return None
            
    except Exception as e:
        print(f"❌ Unexpected error saving results: {e}")
        return None


def load_metrics_from_csv():
    """Load metrics from CSV and auto-generate evaluation prompts."""
    # This is a placeholder - you would need to implement the actual CSV loading
    # based on your METRICS_CONFIG_DATA
    print("📋 Loading metrics configuration...")
    
    # Example metric configs for demonstration
    metric_configs = []
    
    # You would replace this with your actual CSV loading logic
    # if METRICS_CONFIG_DATA is None:
    #     return []
    
    print("✅ Metrics loaded successfully")
    return metric_configs


def main_evaluation_with_fixed_saving():
    """
    Main evaluation function with fixed file saving.
    
    This replaces the problematic section in your original code.
    """
    print("🚀 STARTING LLM EVALUATION WITH FIXED FILE HANDLING")
    print("="*60)
    
    # Create safe output directory
    output_dir = create_safe_output_directory()
    
    # Load metrics (you would implement this based on your data)
    metric_configs = load_metrics_from_csv()
    
    if not metric_configs:
        print("❌ No valid metrics found!")
        return
    
    # Placeholder for your actual evaluation logic
    # You would replace this with your actual LLMJudgeEvaluator code
    print("📊 Running evaluations...")
    
    # Create sample results DataFrame for demonstration
    results_data = {
        'metric_name': ['accuracy', 'relevance', 'completeness'],
        'score': [0.85, 0.92, 0.78],
        'status': ['✅', '✅', '✅'],
        'threshold': [0.8, 0.8, 0.7]
    }
    results_df = pd.DataFrame(results_data)
    
    # Calculate summary statistics
    total = len(results_df)
    passed = len(results_df[results_df['status'] == '✅'])
    pass_rate = (passed / total) * 100 if total > 0 else 0
    
    print(f"\n📊 RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total Evaluations: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Pass Rate: {pass_rate:.1f}%")
    print(f"{'='*60}")
    
    # FIXED: Safe file saving with proper error handling
    saved_path = safe_save_results(results_df, output_dir, "evaluation_results")
    
    if saved_path:
        print(f"✅ Results successfully saved to: {saved_path}")
        
        # Optional: Try to log to MLflow with better error handling
        try:
            # Only import and use MLflow if available
            import mlflow
            
            # Set experiment with error handling
            try:
                # Use a simpler experiment name that doesn't require user context
                mlflow.set_experiment("llm_evaluation_experiment")
                
                with mlflow.start_run(run_name=f"eval_{time.strftime('%Y%m%d_%H%M%S')}"):
                    mlflow.log_param("num_samples", len(results_df))
                    mlflow.log_param("metrics", ",".join([cfg.get('name', 'unknown') for cfg in metric_configs]))
                    
                    mlflow.log_metric("overall_pass_rate", pass_rate)
                    mlflow.log_metric("total_evaluations", total)
                    mlflow.log_metric("passed_evaluations", passed)
                    
                    # Log the saved file as an artifact
                    mlflow.log_artifact(saved_path, artifact_path="tables")
                
                print(f"✅ Results logged to MLflow")
                
            except Exception as mlflow_error:
                print(f"⚠️ MLflow logging failed (non-critical): {mlflow_error}")
                print("📁 Results are still saved locally")
                
        except ImportError:
            print("⚠️ MLflow not available, skipping cloud logging")
            print("📁 Results are saved locally")
            
    else:
        print("❌ Failed to save results to any location")
        print("📋 Results are available in memory as 'results_df'")
    
    return results_df


if __name__ == "__main__":
    # Run the fixed evaluation
    results = main_evaluation_with_fixed_saving()
    print("\n✅ Evaluation completed!")