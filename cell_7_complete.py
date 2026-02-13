# MAGIC %md
# MAGIC ## Cell 7: Run Evaluation
# MAGIC
# MAGIC **Purpose**: This cell executes the actual evaluation process, running all configured metrics against all evaluation samples.
# MAGIC
# MAGIC **What it does**:
# MAGIC - Loads metrics configuration from the CSV file and auto-generates evaluation prompts from grading rubrics
# MAGIC - **ENHANCED**: Always parses ALL columns from ground truth files regardless of CSV specification
# MAGIC - Creates the LLM Judge Evaluator instance with the selected model and metrics
# MAGIC - Runs the evaluation process for all samples and all metrics with rich ground truth context
# MAGIC - Displays real-time progress as each sample and metric is evaluated
# MAGIC - Generates comprehensive results including scores, explanations, and pass/fail status
# MAGIC - Logs all results to MLflow for experiment tracking and comparison
# MAGIC - Provides detailed summary statistics including overall pass rates and per-metric performance
# MAGIC
# MAGIC **When to run**: Run this cell after Cells 1-6 to execute the evaluation
# MAGIC
# MAGIC **Expected output**: Real-time evaluation progress, comprehensive results summary, and MLflow logging confirmation

# COMMAND ----------

def auto_generate_evaluation_prompt(metric_name: str, metric_type: str, description: str, grading_rubric: str) -> str:
    """
    Auto-generate evaluation prompt from grading rubric.
    PM just provides grading rubric, code handles the rest!
    
    ENHANCED: Now optimized for rich ground truth context with all columns.
    """
    
    # Add grading rubric section
    rubric_section = f"\n**Grading Rubric:**\n{grading_rubric}\n" if grading_rubric else ""
    
    # Build complete evaluation prompt with enhanced ground truth instructions
    prompt = f"""You are an expert evaluator. Your task: {description}

{rubric_section}
**Evaluation Details:**
- User Query: {{prompt}}
- AI Response: {{response}}
- Ground Truth Reference: {{ground_truth}}

**Ground Truth Context:**
The ground truth reference above contains ALL available reference data for this evaluation, including multiple data points that may be relevant to your assessment. Use this comprehensive reference information to make a thorough evaluation.

**Instructions:**
Carefully evaluate the AI response using the grading rubric above and the comprehensive ground truth reference data.

**Required Output Format:**
Return ONLY a valid JSON object with these two fields:
{{
  "score": <your_score>,
  "explanation": "Brief explanation of your score based on the rubric and ground truth reference"
}}

Do not include any other text outside the JSON object."""
    
    return prompt

def load_metrics_from_csv():
    """
    Load metrics from CSV and auto-generate evaluation prompts.
    
    ENHANCED: Always parses ALL columns from ground truth files for maximum context,
    regardless of what's specified in the CSV file.
    """
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
            print(f"✅ {row['name'].strip()} - Auto-generated prompt from grading rubric (ENHANCED)")
        elif 'evaluation_prompt' in row and pd.notna(row.get('evaluation_prompt')):
            # OLD FORMAT: Use evaluation_prompt directly (backward compatibility)
            prompt_template = row['evaluation_prompt'].strip()
            print(f"✅ {row['name'].strip()} - Using provided evaluation_prompt (ENHANCED)")
        else:
            # Fallback: Generate basic prompt from description
            prompt_template = auto_generate_evaluation_prompt(
                metric_name=row['name'].strip(),
                metric_type=metric_type_str,
                description=row.get('description', '').strip(),
                grading_rubric=""
            )
            print(f"⚠️ {row['name'].strip()} - No rubric or prompt provided, using basic prompt (ENHANCED)")
        
        # Ground truth file matching (PM provides filename, code matches to uploaded files)
        gt_file_value = row.get('ground_truth_file_path', '').strip()
        
        # 🎯 ENHANCED: Always use ALL columns regardless of CSV specification
        use_all_columns = True  # Force to True - always parse all columns
        
        # Keep ground_truth_column for backward compatibility, but we'll ignore it and use all columns
        gt_column = row.get('ground_truth_column', '').strip()
        
        metric_config = MetricConfig(
            name=row['name'].strip(),
            metric_type=metric_type,
            description=row.get('description', '').strip(),
            prompt_template=prompt_template,
            threshold=float(row['threshold']),
            ground_truth_column=gt_column,  # Keep for compatibility but won't be used
            ground_truth_file_path=gt_file_value,
            use_all_columns=use_all_columns  # Always True
        )
        
        # Show which GT file this metric will use with enhanced messaging
        if gt_file_value:
            gt_filename = os.path.basename(gt_file_value)
            print(f"   📚 Ground truth: {gt_filename} → 🎯 ALL COLUMNS (Enhanced Context)")
            if gt_column:
                print(f"      ℹ️  Note: CSV specified column '{gt_column}' but using ALL columns for richer context")
        
        metric_configs.append(metric_config)
    
    return metric_configs

# Load metrics with enhanced processing
print("🚀 LOADING METRICS WITH ENHANCED GROUND TRUTH ACCESS")
print("="*70)
print("🎯 Key Enhancement: ALL columns from ground truth files will be parsed")
print("   regardless of what's specified in your CSV configuration!")
print("="*70)

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
    print("="*70)
    print("🚀 STARTING ENHANCED EVALUATION")
    print("="*70)
    print("🎯 Each metric now has access to ALL ground truth columns for richer context!")
    print("📊 This provides much more comprehensive reference data for evaluation")
    print("="*70)
    
    start_time = time.time()
    results_df = evaluator.evaluate_dataset(EVALUATION_DATA)
    eval_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"✅ ENHANCED EVALUATION COMPLETE in {eval_time:.1f}s")
    print(f"{'='*70}")
    
    # Display summary
    total = len(results_df)
    passed = len(results_df[results_df['status'] == '✅'])
    pass_rate = (passed / total) * 100 if total > 0 else 0
    
    print(f"\n📊 ENHANCED RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Total Evaluations: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Pass Rate: {pass_rate:.1f}%")
    print(f"🎯 Enhancement: All metrics used comprehensive ground truth context")
    print(f"{'='*70}")
    
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
        print(f"   🎯 Used: Enhanced ground truth with all columns")
    
    # Store globally
    globals()['results_df'] = results_df
    
    # Log to MLflow with enhanced metadata
    mlflow.set_experiment(f"/Users/{dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()}/llm_evaluation_experiment")
    
    with mlflow.start_run(run_name=f"enhanced_eval_{JUDGE_MODEL}_{time.strftime('%Y%m%d_%H%M%S')}"):
        mlflow.log_param("judge_model", JUDGE_MODEL)
        mlflow.log_param("num_samples", len(EVALUATION_DATA))
        mlflow.log_param("metrics", ",".join([cfg.name for cfg in metric_configs]))
        mlflow.log_param("enhancement", "all_columns_ground_truth")
        mlflow.log_param("ground_truth_mode", "comprehensive_all_columns")
        
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
        
        # Log enhancement details
        mlflow.log_param("ground_truth_columns_used", "all_available")
        mlflow.log_param("context_richness", "maximum")
    
    print(f"\n✅ Enhanced results logged to MLflow")
    print(f"🎯 Experiment tagged with 'all_columns_ground_truth' enhancement")
    print(f"📊 All metrics benefited from comprehensive ground truth context")

# COMMAND ----------