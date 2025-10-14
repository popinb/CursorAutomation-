#!/usr/bin/env python3
"""
Manual Validation Script for Databricks LLM Judge Evaluation Notebook
Validates all components without Databricks dependencies
"""

import os
import sys
import pandas as pd
import tempfile
import json

def validate_file_structure():
    """Validate that all required files exist and have correct structure"""
    print("📁 Validating file structure...")
    
    required_files = [
        'databricks_llm_judge_notebook.py',
        'sample_metrics_config.csv',
        'evaluation_data.csv',
        'correct_answer.csv',
        'helpful_answer.csv',
        'safe_response.csv',
        'complete_answer.csv'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ All required files present")
    return True

def validate_csv_files():
    """Validate CSV file formats and content"""
    print("📊 Validating CSV files...")
    
    # Test evaluation data
    try:
        eval_df = pd.read_csv('evaluation_data.csv')
        required_cols = ['sample_id', 'prompt', 'response']
        for col in required_cols:
            if col not in eval_df.columns:
                print(f"❌ Missing column in evaluation_data.csv: {col}")
                return False
        print("✅ evaluation_data.csv format correct")
    except Exception as e:
        print(f"❌ Error reading evaluation_data.csv: {e}")
        return False
    
    # Test metrics config
    try:
        metrics_df = pd.read_csv('sample_metrics_config.csv')
        required_cols = ['name', 'type', 'description', 'evaluation_prompt', 'threshold', 'ground_truth_column', 'ground_truth_file_path']
        for col in required_cols:
            if col not in metrics_df.columns:
                print(f"❌ Missing column in sample_metrics_config.csv: {col}")
                return False
        
        # Check metric types
        valid_types = ['binary', 'scale_1_5', 'percentage']
        for metric_type in metrics_df['type']:
            if metric_type not in valid_types:
                print(f"❌ Invalid metric type: {metric_type}")
                return False
        
        print("✅ sample_metrics_config.csv format correct")
    except Exception as e:
        print(f"❌ Error reading sample_metrics_config.csv: {e}")
        return False
    
    # Test ground truth files
    ground_truth_files = ['correct_answer.csv', 'helpful_answer.csv', 'safe_response.csv', 'complete_answer.csv']
    for file in ground_truth_files:
        try:
            gt_df = pd.read_csv(file)
            if 'sample_id' not in gt_df.columns:
                print(f"❌ Missing sample_id column in {file}")
                return False
            if len(gt_df) == 0:
                print(f"❌ Empty ground truth file: {file}")
                return False
            print(f"✅ {file} format correct")
        except Exception as e:
            print(f"❌ Error reading {file}: {e}")
            return False
    
    return True

def validate_notebook_syntax():
    """Validate notebook Python syntax"""
    print("🐍 Validating notebook syntax...")
    
    try:
        with open('databricks_llm_judge_notebook.py', 'r') as f:
            content = f.read()
        
        # Check for common syntax issues
        if 'import pandas as pd' not in content:
            print("❌ Missing pandas import")
            return False
        
        if 'class LLMJudgeEvaluator' not in content:
            print("❌ Missing LLMJudgeEvaluator class")
            return False
        
        if 'def load_any_csv' not in content:
            print("❌ Missing load_any_csv function")
            return False
        
        if 'def load_ground_truth_for_metric' not in content:
            print("❌ Missing load_ground_truth_for_metric function")
            return False
        
        # Try to compile the code
        compile(content, 'databricks_llm_judge_notebook.py', 'exec')
        print("✅ Notebook syntax is valid")
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax error in notebook: {e}")
        return False
    except Exception as e:
        print(f"❌ Error validating notebook: {e}")
        return False

def validate_ground_truth_mapping():
    """Validate ground truth file mapping"""
    print("🔗 Validating ground truth mapping...")
    
    try:
        # Load metrics config
        metrics_df = pd.read_csv('sample_metrics_config.csv')
        
        # Check that ground_truth_column values match expected filenames
        expected_mappings = {
            'correct_answer': 'correct_answer.csv',
            'helpful_answer': 'helpful_answer.csv',
            'safe_response': 'safe_response.csv',
            'complete_answer': 'complete_answer.csv'
        }
        
        for _, row in metrics_df.iterrows():
            gt_column = row['ground_truth_column']
            if gt_column in expected_mappings:
                expected_file = expected_mappings[gt_column]
                if not os.path.exists(expected_file):
                    print(f"❌ Ground truth file not found: {expected_file}")
                    return False
                print(f"✅ {gt_column} -> {expected_file} mapping correct")
        
        print("✅ Ground truth mapping validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Error validating ground truth mapping: {e}")
        return False

def validate_data_consistency():
    """Validate data consistency across files"""
    print("🔄 Validating data consistency...")
    
    try:
        # Load evaluation data
        eval_df = pd.read_csv('evaluation_data.csv')
        eval_sample_ids = set(eval_df['sample_id'])
        
        # Check ground truth files have matching sample IDs
        ground_truth_files = ['correct_answer.csv', 'helpful_answer.csv', 'safe_response.csv', 'complete_answer.csv']
        
        for file in ground_truth_files:
            gt_df = pd.read_csv(file)
            gt_sample_ids = set(gt_df['sample_id'])
            
            if not eval_sample_ids.issubset(gt_sample_ids):
                missing_ids = eval_sample_ids - gt_sample_ids
                print(f"❌ Missing sample IDs in {file}: {missing_ids}")
                return False
            
            print(f"✅ {file} has all required sample IDs")
        
        print("✅ Data consistency validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Error validating data consistency: {e}")
        return False

def validate_json_prompts():
    """Validate that evaluation prompts contain proper JSON format"""
    print("📝 Validating JSON prompts...")
    
    try:
        metrics_df = pd.read_csv('sample_metrics_config.csv')
        
        for _, row in metrics_df.iterrows():
            prompt = row['evaluation_prompt']
            
            # Check for required placeholders
            if '{prompt}' not in prompt:
                print(f"❌ Missing {{prompt}} placeholder in metric: {row['name']}")
                return False
            
            if '{response}' not in prompt:
                print(f"❌ Missing {{response}} placeholder in metric: {row['name']}")
                return False
            
            if '{ground_truth}' not in prompt:
                print(f"❌ Missing {{ground_truth}} placeholder in metric: {row['name']}")
                return False
            
            # Check for JSON format in prompts
            if 'Return JSON:' in prompt or 'JSON:' in prompt:
                print(f"✅ {row['name']} has JSON format instructions")
            else:
                print(f"⚠️ {row['name']} may be missing JSON format instructions")
        
        print("✅ JSON prompts validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Error validating JSON prompts: {e}")
        return False

def main():
    """Run all validation checks"""
    print("🚀 Manual Validation for Databricks LLM Judge Evaluation Notebook")
    print("=" * 70)
    
    validations = [
        ("File Structure", validate_file_structure),
        ("CSV Files", validate_csv_files),
        ("Notebook Syntax", validate_notebook_syntax),
        ("Ground Truth Mapping", validate_ground_truth_mapping),
        ("Data Consistency", validate_data_consistency),
        ("JSON Prompts", validate_json_prompts)
    ]
    
    passed = 0
    total = len(validations)
    
    for validation_name, validation_func in validations:
        print(f"\n🔍 Running: {validation_name}")
        if validation_func():
            passed += 1
            print(f"✅ {validation_name} PASSED")
        else:
            print(f"❌ {validation_name} FAILED")
    
    print("\n" + "=" * 70)
    print(f"🏁 Validation Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 ALL VALIDATIONS PASSED!")
        print("\n📋 The notebook is ready for Databricks upload!")
        print("\n🚀 Upload Instructions:")
        print("1. Upload databricks_llm_judge_notebook.py to Databricks as a notebook")
        print("2. Upload all CSV files to your Databricks workspace")
        print("3. Update the widget values in the notebook:")
        print("   - evaluation_data_path: 'evaluation_data.csv'")
        print("   - metrics_config_path: 'sample_metrics_config.csv'")
        print("4. Run the notebook cells in order")
        print("\n✨ The system will automatically find and load your ground truth files!")
        return True
    else:
        print("❌ Some validations failed. Please fix the issues before uploading.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)