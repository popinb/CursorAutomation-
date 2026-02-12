"""
Fixed version of the LLM evaluation code with proper permission handling.
"""

import os
import time
import tempfile
from pathlib import Path
import pandas as pd
import mlflow

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


def get_safe_output_directory(base_name="llm_eval_artifacts"):
    """
    Get a safe output directory for saving evaluation results.
    Tries multiple locations in order of preference.
    """
    # Option 1: Try current working directory
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
    
    # Option 2: Try user's home directory
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
    
    # Option 3: Use system temp directory with user-specific subdirectory
    try:
        temp_base = tempfile.gettempdir()
        # Create user-specific subdirectory to avoid conflicts
        user_id = os.environ.get('USER', 'default')
        user_temp_dir = Path(temp_base) / f"{user_id}_{base_name}"
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


def load_metrics_from_csv():
    """Load metrics from CSV and auto-generate evaluation prompts."""
    if METRICS_CONFIG_DATA is None:
        return []
    
    metric_configs = []
    for _, row in METRICS_CONFIG_DATA.iterrows():
        metric_type_str = row['type'].strip().lower()
        if metric_type_str == 'binary':
            metric_type = MetricType.BINARY
        elif metric_type_str in ['1-5_scale', 'scale_1_5']:
            metric_type = MetricType.SCALE_1_5
        elif metric_type_str == 'percentage':
            metric_type = MetricType.PERCENTAGE
        else:
            metric_type = MetricType.BINARY
        
        # Check if using new format (grading_rubric) or old format (evaluation_prompt)
        if 'grading_rubric' in row and pd.notna(row.get('grading_rubric')):
            # NEW FORMAT: Auto-generate evaluation prompt from grading rubric
            grading_rubric = str(row['grading_rubric']).strip()
            prompt_template = auto_generate_evaluation_prompt(
                metric_name=row['name'].strip(),
                metric_type=metric_type_str,
                description=row.get('description', '').strip(),
                grading_rubric=grading_rubric
            )
            print(f"✅ {row['name'].strip()} - Auto-generated prompt from grading rubric")
        elif 'evaluation_prompt' in row and pd.notna(row.get('evaluation_prompt')):
            # OLD FORMAT: Use evaluation_prompt directly (backward compatibility)
            prompt_template = row['evaluation_prompt'].strip()
            print(f"✅ {row['name'].strip()} - Using provided evaluation_prompt")
        else:
            # Fallback: Generate basic prompt from description
            prompt_template = auto_generate_evaluation_prompt(
                metric_name=row['name'].strip(),
                metric_type=metric_type_str,
                description=row.get('description', '').strip(),
                grading_rubric=""
            )
            print(f"⚠️ {row['name'].strip()} - No rubric or prompt provided, using basic prompt")
        
        # Ground truth file matching (PM provides filename, code matches to uploaded files)
        gt_file_value = row.get('ground_truth_file_path', '').strip()
        
        metric_config = MetricConfig(
            name=row['name'].strip(),
            metric_type=metric_type,
            description=row.get('description', '').strip(),
            prompt_template=prompt_template,
            threshold=float(row['threshold']),
            ground_truth_column=row['ground_truth_column'].strip(),
            ground_truth_file_path=gt_file_value
        )
        
        # Show which GT file this metric will use
        if gt_file_value:
            gt_filename = os.path.basename(gt_file_value)
            print(f"   📚 Ground truth: {gt_filename} → column '{row['ground_truth_column'].strip()}'")
        
        metric_configs.append(metric_config)
    
    return metric_configs


# Load metrics
metric_configs = load_metrics_from_csv()

if not metric_configs:
    print("❌ No valid metrics!")
else:
    # Create evaluator
    evaluator = LLMJudgeEvaluator(
        judge_model=JUDGE_MODEL,
        metrics=metric_configs,
        ground_truth_data=ground_truth_data
    )
    
    # Run evaluation
    print("="*60)
    print("🚀 STARTING EVALUATION")
    print("="*60)
    
    start_time = time.time()
    results_df = evaluator.evaluate_dataset(EVALUATION_DATA)
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
    
    # Log to MLflow
    mlflow.set_experiment(f"/Users/{dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()}/llm_evaluation_experiment")
    
    with mlflow.start_run(run_name=f"eval_{JUDGE_MODEL}_{time.strftime('%Y%m%d_%H%M%S')}"):
        mlflow.log_param("judge_model", JUDGE_MODEL)
        mlflow.log_param("num_samples", len(EVALUATION_DATA))
        mlflow.log_param("metrics", ",".join([cfg.name for cfg in metric_configs]))
        
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
        
        # FIXED SECTION: Save results with proper permission handling
        try:
            # Get a safe output directory
            out_dir = get_safe_output_directory()
            csv_path = os.path.join(out_dir, f"results_{int(time.time())}.csv")
            
            # Save the results
            results_df.to_csv(csv_path, index=False)
            print(f"\n✅ Results saved successfully to: {csv_path}")
            
            # Log artifact to MLflow
            mlflow.log_artifact(csv_path, artifact_path="tables")
            print(f"✅ Results logged to MLflow")
            
        except Exception as e:
            # Fallback: Use NamedTemporaryFile if all else fails
            print(f"⚠️ Could not save to preferred location: {e}")
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, 
                                           prefix='llm_eval_results_') as tmp_file:
                results_df.to_csv(tmp_file, index=False)
                csv_path = tmp_file.name
                print(f"✅ Results saved to temporary file: {csv_path}")
                
                try:
                    mlflow.log_artifact(csv_path, artifact_path="tables")
                    print(f"✅ Results logged to MLflow")
                except Exception as mlflow_error:
                    print(f"⚠️ Could not log to MLflow: {mlflow_error}")
                    print(f"   Results are still saved locally at: {csv_path}")