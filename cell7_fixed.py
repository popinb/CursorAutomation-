# COMMAND ----------

# Check if we should use custom metrics from Cell 5 or load from config
if METRICS_CONFIG_DATA is not None:
    # If CSV file was uploaded, use metrics from CSV
    CUSTOM_METRICS = load_metrics_from_config()
elif not CUSTOM_METRICS:
    # If no custom metrics defined in Cell 5 and no CSV, use defaults
    CUSTOM_METRICS = load_metrics_from_config()
# else: Use the CUSTOM_METRICS already defined in Cell 5

# Process metrics and create evaluator
metric_configs = process_custom_metrics(CUSTOM_METRICS)

if not metric_configs:
    print("❌ No valid metrics to evaluate!")
    print("Please define metrics in Cell 5 or upload a metrics configuration CSV file")
else:
    # Create evaluator
    evaluator = LLMJudgeEvaluator(
        judge_model=JUDGE_MODEL,
        metrics=metric_configs
    )
    
    # Run evaluation
    print("="*60)
    print("🚀 STARTING EVALUATION")
    print("="*60)
    
    start_time = time.time()
    results_df = evaluator.evaluate_dataset(EVALUATION_DATA)
    eval_time = time.time() - start_time
    
    print("\n" + "="*60)
    print(f"✅ EVALUATION COMPLETE in {eval_time:.1f} seconds")
    print("="*60)
    
    # Display sample results
    print("\n📊 Sample Results:")
    display_cols = ['prompt', 'response'] + [f"{m.name}_score" for m in metric_configs] + [f"{m.name}_status" for m in metric_configs]
    display_cols = [col for col in display_cols if col in results_df.columns]
    display(results_df[display_cols].head())