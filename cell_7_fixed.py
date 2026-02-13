# MAGIC %md
# MAGIC ## Cell 7: Run Evaluation (FIXED)
# MAGIC
# MAGIC **Purpose**: This cell executes the actual evaluation process with robust error handling for CSV data issues.
# MAGIC
# MAGIC **FIXED**: Added comprehensive data validation and error handling for threshold values and other CSV parsing issues.

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

def safe_float_conversion(value, default=0.0, field_name="value"):
    """
    Safely convert a value to float with comprehensive error handling.
    
    Args:
        value: The value to convert
        default: Default value if conversion fails
        field_name: Name of the field for error reporting
    
    Returns:
        float: Converted value or default
    """
    if pd.isna(value) or value is None:
        print(f"⚠️  Warning: {field_name} is null/NaN, using default {default}")
        return default
    
    # Convert to string and clean up
    str_value = str(value).strip().lower()
    
    # Handle empty strings
    if not str_value:
        print(f"⚠️  Warning: {field_name} is empty, using default {default}")
        return default
    
    # Handle boolean-like strings
    if str_value in ['true', '==true', 'yes', '1']:
        print(f"⚠️  Warning: {field_name} contains '{value}', interpreting as 1.0")
        return 1.0
    elif str_value in ['false', '==false', 'no', '0']:
        print(f"⚠️  Warning: {field_name} contains '{value}', interpreting as 0.0")
        return 0.0
    
    # Handle percentage strings
    if str_value.endswith('%'):
        try:
            return float(str_value[:-1])
        except ValueError:
            print(f"⚠️  Warning: Could not parse percentage '{value}' for {field_name}, using default {default}")
            return default
    
    # Handle decimal/fraction strings
    if '/' in str_value:
        try:
            parts = str_value.split('/')
            if len(parts) == 2:
                return float(parts[0]) / float(parts[1])
        except ValueError:
            pass
    
    # Try direct float conversion
    try:
        return float(str_value)
    except ValueError:
        print(f"⚠️  Warning: Could not convert '{value}' to float for {field_name}, using default {default}")
        return default

def safe_string_conversion(value, default="", field_name="value"):
    """
    Safely convert a value to string with error handling.
    
    Args:
        value: The value to convert
        default: Default value if conversion fails
        field_name: Name of the field for error reporting
    
    Returns:
        str: Converted value or default
    """
    if pd.isna(value) or value is None:
        return default
    
    try:
        return str(value).strip()
    except Exception as e:
        print(f"⚠️  Warning: Could not convert '{value}' to string for {field_name}: {e}, using default '{default}'")
        return default

def validate_metric_type(metric_type_str):
    """
    Validate and normalize metric type string.
    
    Args:
        metric_type_str: The metric type string from CSV
        
    Returns:
        MetricType: Validated metric type enum
    """
    if pd.isna(metric_type_str):
        print("⚠️  Warning: metric type is null, defaulting to 'binary'")
        return MetricType.BINARY
    
    metric_type_clean = str(metric_type_str).strip().lower()
    
    # Handle various formats
    if metric_type_clean in ['binary', 'bool', 'boolean', 'true/false', 'pass/fail']:
        return MetricType.BINARY
    elif metric_type_clean in ['1-5_scale', 'scale_1_5', '1-5', 'scale', 'rating', '1to5']:
        return MetricType.SCALE_1_5
    elif metric_type_clean in ['percentage', 'percent', '%', '0-100']:
        return MetricType.PERCENTAGE
    else:
        print(f"⚠️  Warning: Unknown metric type '{metric_type_str}', defaulting to 'binary'")
        return MetricType.BINARY

def load_metrics_from_csv():
    """
    Load metrics from CSV with comprehensive error handling and data validation.
    
    ENHANCED: Always parses ALL columns from ground truth files for maximum context,
    regardless of what's specified in the CSV file.
    FIXED: Robust error handling for malformed CSV data.
    """
    if METRICS_CONFIG_DATA is None:
        print("❌ No metrics configuration data available")
        return []
    
    print(f"📊 Processing {len(METRICS_CONFIG_DATA)} rows from metrics CSV...")
    
    metric_configs = []
    for idx, row in METRICS_CONFIG_DATA.iterrows():
        try:
            print(f"\n🔄 Processing row {idx + 1}: ", end="")
            
            # Safely extract and validate name
            name = safe_string_conversion(row.get('name', ''), f"unnamed_metric_{idx}", "name")
            if not name or name.startswith("unnamed_metric_"):
                print(f"⚠️  Row {idx + 1}: Missing or invalid name, skipping")
                continue
            
            print(f"'{name}'")
            
            # Safely validate metric type
            metric_type = validate_metric_type(row.get('type', 'binary'))
            
            # Safely extract description
            description = safe_string_conversion(row.get('description', ''), f"Evaluation metric: {name}", "description")
            
            # Safely convert threshold with appropriate defaults based on metric type
            if metric_type == MetricType.BINARY:
                default_threshold = 0.5
            elif metric_type == MetricType.SCALE_1_5:
                default_threshold = 3.0
            else:  # PERCENTAGE
                default_threshold = 70.0
            
            threshold = safe_float_conversion(row.get('threshold', default_threshold), default_threshold, f"threshold for {name}")
            
            # Safely extract ground truth info
            gt_file_value = safe_string_conversion(row.get('ground_truth_file_path', ''), "", "ground_truth_file_path")
            gt_column = safe_string_conversion(row.get('ground_truth_column', ''), "", "ground_truth_column")
            
            # Check if using new format (grading_rubric) or old format (evaluation_prompt)
            grading_rubric = safe_string_conversion(row.get('grading_rubric', ''), "", "grading_rubric")
            evaluation_prompt = safe_string_conversion(row.get('evaluation_prompt', ''), "", "evaluation_prompt")
            
            if grading_rubric:
                # NEW FORMAT: Auto-generate evaluation prompt from grading rubric
                prompt_template = auto_generate_evaluation_prompt(
                    metric_name=name,
                    metric_type=metric_type.value,
                    description=description,
                    grading_rubric=grading_rubric
                )
                print(f"   ✅ Auto-generated prompt from grading rubric (ENHANCED)")
            elif evaluation_prompt:
                # OLD FORMAT: Use evaluation_prompt directly (backward compatibility)
                prompt_template = evaluation_prompt
                print(f"   ✅ Using provided evaluation_prompt (ENHANCED)")
            else:
                # Fallback: Generate basic prompt from description
                prompt_template = auto_generate_evaluation_prompt(
                    metric_name=name,
                    metric_type=metric_type.value,
                    description=description,
                    grading_rubric=""
                )
                print(f"   ⚠️  No rubric or prompt provided, using basic prompt (ENHANCED)")
            
            # 🎯 ENHANCED: Always use ALL columns regardless of CSV specification
            use_all_columns = True  # Force to True - always parse all columns
            
            metric_config = MetricConfig(
                name=name,
                metric_type=metric_type,
                description=description,
                prompt_template=prompt_template,
                threshold=threshold,
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
            else:
                print(f"   📚 No ground truth file specified")
            
            print(f"   🎯 Threshold: {threshold} ({metric_type.value})")
            
            metric_configs.append(metric_config)
            
        except Exception as e:
            print(f"❌ Error processing row {idx + 1}: {e}")
            print(f"   Row data: {dict(row)}")
            continue
    
    print(f"\n✅ Successfully loaded {len(metric_configs)} metrics")
    return metric_configs

# Load metrics with enhanced processing and error handling
print("🚀 LOADING METRICS WITH ENHANCED GROUND TRUTH ACCESS & ERROR HANDLING")
print("="*80)
print("🎯 Key Enhancement: ALL columns from ground truth files will be parsed")
print("   regardless of what's specified in your CSV configuration!")
print("🛡️  Added: Comprehensive error handling for malformed CSV data")
print("="*80)

# Debug: Show the raw CSV data first
if METRICS_CONFIG_DATA is not None:
    print(f"\n🔍 DEBUG: Raw CSV data preview:")
    print(f"   Columns: {list(METRICS_CONFIG_DATA.columns)}")
    print(f"   Shape: {METRICS_CONFIG_DATA.shape}")
    
    # Show problematic threshold values
    if 'threshold' in METRICS_CONFIG_DATA.columns:
        unique_thresholds = METRICS_CONFIG_DATA['threshold'].unique()
        print(f"   Unique threshold values: {unique_thresholds}")
        
        # Identify problematic values
        problematic = []
        for val in unique_thresholds:
            if pd.notna(val):
                try:
                    float(val)
                except ValueError:
                    problematic.append(val)
        
        if problematic:
            print(f"   ⚠️  Problematic threshold values found: {problematic}")
            print(f"   🛠️  These will be handled automatically with safe conversion")
    
    print(f"\n📋 First few rows:")
    display(METRICS_CONFIG_DATA.head())

metric_configs = load_metrics_from_csv()

if not metric_configs:
    print("❌ No valid metrics loaded!")
    print("💡 Please check your metrics CSV file for:")
    print("   • Valid 'name' column with non-empty values")
    print("   • Valid 'type' column (binary, 1-5_scale, percentage)")
    print("   • Valid 'threshold' column with numeric values")
    print("   • Either 'grading_rubric' or 'evaluation_prompt' column")
else:
    # Create evaluator
    evaluator = LLMJudgeEvaluator(
        judge_model=JUDGE_MODEL,
        metrics=metric_configs,
        ground_truth_data=ground_truth_data
    )
    
    # Run evaluation
    print("="*80)
    print("🚀 STARTING ENHANCED EVALUATION WITH ERROR HANDLING")
    print("="*80)
    print("🎯 Each metric now has access to ALL ground truth columns for richer context!")
    print("📊 This provides much more comprehensive reference data for evaluation")
    print("🛡️  Enhanced with robust error handling for data issues")
    print("="*80)
    
    start_time = time.time()
    results_df = evaluator.evaluate_dataset(EVALUATION_DATA)
    eval_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"✅ ENHANCED EVALUATION COMPLETE in {eval_time:.1f}s")
    print(f"{'='*80}")
    
    # Display summary
    total = len(results_df)
    passed = len(results_df[results_df['status'] == '✅'])
    pass_rate = (passed / total) * 100 if total > 0 else 0
    
    print(f"\n📊 ENHANCED RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"Total Evaluations: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Pass Rate: {pass_rate:.1f}%")
    print(f"🎯 Enhancement: All metrics used comprehensive ground truth context")
    print(f"🛡️  Error Handling: Robust data validation applied")
    print(f"{'='*80}")
    
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
        mlflow.log_param("error_handling", "robust_csv_validation")
        
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
        mlflow.log_param("data_validation", "comprehensive")
    
    print(f"\n✅ Enhanced results logged to MLflow")
    print(f"🎯 Experiment tagged with 'all_columns_ground_truth' enhancement")
    print(f"📊 All metrics benefited from comprehensive ground truth context")
    print(f"🛡️  Data validation ensured robust processing of CSV issues")

# COMMAND ----------