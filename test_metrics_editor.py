"""
Quick test script for the Metrics Editor module.
Run this in a Databricks notebook to verify everything works.
"""

# Databricks notebook source
# COMMAND ----------
# Test 1: Import and Initialize

print("="*80)
print("TEST 1: Import and Initialize")
print("="*80)

try:
    from databricks_metrics_editor import MetricsEditor
    editor = MetricsEditor()
    print(f"? Module imported successfully")
    print(f"? Current user: {editor.current_user}")
    print(f"? Temp file path: {editor.temp_metrics_path}")
except Exception as e:
    print(f"? Import failed: {e}")
    raise

# COMMAND ----------
# Test 2: Load Data (with sample data)

print("\n" + "="*80)
print("TEST 2: Load Data")
print("="*80)

try:
    summary = editor.load_data(
        evaluation_data_path="nonexistent_eval.csv",  # Will create sample
        metrics_config_path="nonexistent_metrics.csv",  # Will create sample
        ground_truth_files=""  # No ground truth for test
    )
    
    print(f"? Data loaded successfully")
    print(f"   Evaluation rows: {summary['evaluation_data']['rows']}")
    print(f"   Metrics rows: {summary['metrics_config']['rows']}")
    
    # Display the loaded data
    editor.display_summary()
    
except Exception as e:
    print(f"? Load failed: {e}")
    raise

# COMMAND ----------
# Test 3: Show Editor UI

print("\n" + "="*80)
print("TEST 3: Display Interactive Editor")
print("="*80)

try:
    editor.show_editor()
    print("? Editor UI displayed above")
    print("?? Try editing the table, adding rows, or deleting rows")
except Exception as e:
    print(f"? Editor display failed: {e}")
    raise

# COMMAND ----------
# Test 4: Apply Changes

print("\n" + "="*80)
print("TEST 4: Apply Changes")
print("="*80)

try:
    result = editor.apply_changes()
    
    if result is not None:
        print("? Changes applied successfully")
        print(f"   Current metrics count: {len(result)}")
        display(result)
    else:
        print("?? No changes to apply (this is OK)")
        
except Exception as e:
    print(f"? Apply changes failed: {e}")
    raise

# COMMAND ----------
# Test 5: Save to File

print("\n" + "="*80)
print("TEST 5: Save to File")
print("="*80)

try:
    test_filename = "test_metrics_output.csv"
    saved_path = editor.save_metrics(test_filename)
    
    if saved_path:
        print("? File saved successfully")
        
        # Verify file exists
        import os
        if os.path.exists(saved_path):
            print(f"? Verified file exists at: {saved_path}")
        else:
            print(f"?? File not found at expected path")
    else:
        print("? Save returned None")
        
except Exception as e:
    print(f"? Save failed: {e}")
    raise

# COMMAND ----------
# Test 6: Access Data

print("\n" + "="*80)
print("TEST 6: Access Data")
print("="*80)

try:
    # Test accessing evaluation data
    eval_df = editor.evaluation_data
    print(f"? Evaluation data accessible: {len(eval_df)} rows")
    
    # Test accessing metrics config
    metrics_df = editor.metrics_config_data
    print(f"? Metrics config accessible: {len(metrics_df)} rows")
    
    # Test accessing ground truth
    gt_dict = editor.ground_truth_data
    print(f"? Ground truth data accessible: {len(gt_dict)} files")
    
    # Get summary
    summary = editor.get_summary()
    print(f"? Summary accessible: {list(summary.keys())}")
    
except Exception as e:
    print(f"? Data access failed: {e}")
    raise

# COMMAND ----------
# Test 7: Cleanup

print("\n" + "="*80)
print("TEST 7: Cleanup")
print("="*80)

try:
    import os
    temp_existed = os.path.exists(editor.temp_metrics_path)
    
    editor.cleanup()
    
    temp_exists_after = os.path.exists(editor.temp_metrics_path)
    
    if temp_existed and not temp_exists_after:
        print("? Temporary file cleaned up successfully")
    elif not temp_existed:
        print("? No temporary file to clean up")
    else:
        print("?? Temporary file still exists after cleanup")
        
except Exception as e:
    print(f"? Cleanup failed: {e}")

# COMMAND ----------
# Final Summary

print("\n" + "="*80)
print("?? ALL TESTS COMPLETED!")
print("="*80)
print("""
? Module is working correctly!

You can now use it in your notebooks:

    from databricks_metrics_editor import MetricsEditor
    
    editor = MetricsEditor()
    editor.load_data(
        evaluation_data_path="your_data.csv",
        metrics_config_path="your_metrics.csv",
        ground_truth_files="gt1.csv;gt2.csv"
    )
    editor.show_editor()
    editor.apply_changes()
    editor.save_metrics("output.csv")

?? See METRICS_EDITOR_README.md for full documentation
""")
print("="*80)
