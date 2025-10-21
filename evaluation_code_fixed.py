#!/usr/bin/env python3
"""
Fixed LLM Evaluation Code - Addresses PermissionError Issue

This is the corrected version of your evaluation code that fixes the
PermissionError when saving results to /tmp/llm_eval_artifacts/
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
        os.path.join(os.getcwd(), "llm_eval_artifacts"),
        # 2. User's home directory
        os.path.expanduser("~/llm_eval_artifacts"),
        # 3. System temp directory with user permissions
        os.path.join(tempfile.gettempdir(), f"llm_eval_artifacts_{os.getuid() if hasattr(os, 'getuid') else 'user'}"),
        # 4. Fallback to Python's tempfile
        tempfile.mkdtemp(prefix="llm_eval_artifacts_")
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
    # NOTE: Replace this with your actual METRICS_CONFIG_DATA loading logic
    # if METRICS_CONFIG_DATA is None:
    #     return []
    
    metric_configs = []
    
    # Your existing CSV loading logic would go here
    # for _, row in METRICS_CONFIG_DATA.iterrows():
    #     ... (your existing code)
    
    return metric_configs


# Your existing evaluation code with FIXED file saving section
def run_evaluation():
    """Main evaluation function with fixed file handling."""
    
    # Load metrics
    metric_configs = load_metrics_from_csv()

    if not metric_configs:
        print("❌ No valid metrics!")
    else:
        # Create evaluator (replace with your actual evaluator)
        # evaluator = LLMJudgeEvaluator(
        #     judge_model=JUDGE_MODEL,
        #     metrics=metric_configs,
        #     ground_truth_data=ground_truth_data
        # )
        
        # Run evaluation (replace with your actual evaluation data)
        print("="*60)
        print("🚀 STARTING EVALUATION")
        print("="*60)
        
        start_time = time.time()
        # results_df = evaluator.evaluate_dataset(EVALUATION_DATA)
        
        # For demonstration, create sample results
        results_df = pd.DataFrame({
            'metric_name': ['accuracy', 'relevance', 'completeness'],
            'score': [0.85, 0.92, 0.78],
            'status': ['✅', '✅', '✅'],
            'threshold': [0.8, 0.8, 0.7]
        })
        
        eval_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"✅ COMPLETE in {eval_time:.1f}s")
        print(f"{'='*60}")
        
        # Display summary
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
        
        # Per-metric results
        for metric_name in results_df['metric_name'].unique():
            metric_results = results_df[results_df['metric_name'] == metric_name]
            metric_passed = len(metric_results[metric_results['status'] == '✅'])
            metric_total = len(metric_results)
            metric_pass_rate = (metric_passed / metric_total) * 100 if metric_total > 0 else 0
            avg_score = metric_results['score'].mean()
            
            print(f"\n🔹 {metric_name}:")
            print(f"   Pass Rate: {metric_pass_rate:.1f}% ({metric_passed}/{metric_total})")
            print(f"   Avg Score: {avg_score:.2f}")
            print(f"   Threshold: {metric_results['threshold'].iloc[0]}")
        
        # Store globally
        globals()['results_df'] = results_df
        
        # FIXED: Safe directory creation and file saving
        out_dir = create_safe_output_directory()
        saved_path = safe_save_results(results_df, out_dir, "results")
        
        # Log to MLflow with better error handling
        try:
            import mlflow
            
            # Use a simpler experiment name that works in all environments
            experiment_name = "llm_evaluation_experiment"
            
            # Try to get username safely
            try:
                import getpass
                username = getpass.getuser()
                experiment_name = f"/Users/{username}/llm_evaluation_experiment"
            except:
                # Fallback to simple name if user detection fails
                pass
            
            try:
                mlflow.set_experiment(experiment_name)
            except:
                # Fallback to default experiment
                mlflow.set_experiment("llm_evaluation_experiment")
            
            run_name = f"eval_{time.strftime('%Y%m%d_%H%M%S')}"
            # Replace JUDGE_MODEL with a default if not defined
            judge_model = globals().get('JUDGE_MODEL', 'default_model')
            
            with mlflow.start_run(run_name=run_name):
                mlflow.log_param("judge_model", judge_model)
                # Replace EVALUATION_DATA with actual data length
                num_samples = globals().get('EVALUATION_DATA', [])
                mlflow.log_param("num_samples", len(num_samples) if hasattr(num_samples, '__len__') else len(results_df))
                mlflow.log_param("metrics", ",".join([cfg.get('name', 'unknown') for cfg in metric_configs]))
                
                mlflow.log_metric("overall_pass_rate", pass_rate)
                mlflow.log_metric("total_evaluations", total)
                mlflow.log_metric("passed_evaluations", passed)
                
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
                
                # Log the saved file as an artifact if it was saved successfully
                if saved_path:
                    mlflow.log_artifact(saved_path, artifact_path="tables")
            
            print(f"\n✅ Results logged to MLflow")
            
        except ImportError:
            print(f"\n⚠️ MLflow not available, skipping cloud logging")
            print(f"📁 Results saved locally to: {saved_path if saved_path else 'memory only'}")
            
        except Exception as mlflow_error:
            print(f"\n⚠️ MLflow logging failed: {mlflow_error}")
            print(f"📁 Results saved locally to: {saved_path if saved_path else 'memory only'}")


if __name__ == "__main__":
    print("🚀 Running Fixed LLM Evaluation")
    run_evaluation()