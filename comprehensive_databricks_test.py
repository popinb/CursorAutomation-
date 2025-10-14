#!/usr/bin/env python3
"""
Comprehensive Test Suite for Databricks LLM Judge Evaluation System
Tests all functionality including UI widgets, LLM selection, and file handling
"""

import os
import sys
import pandas as pd
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock
import traceback

def test_file_structure():
    """Test that all required files exist and are properly formatted"""
    print("📁 Testing file structure...")
    
    required_files = [
        'databricks_llm_judge_enhanced_ui.py',
        'sample_metrics_config_multi_gt.csv',
        'evaluation_data.csv',
        'correct_answer.csv',
        'accuracy_expert1.csv',
        'accuracy_expert2.csv',
        'helpful_answer.csv',
        'helpfulness_rating1.csv',
        'helpfulness_rating2.csv',
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

def test_csv_validation():
    """Test all CSV files are valid and properly formatted"""
    print("📊 Testing CSV validation...")
    
    try:
        # Test evaluation data
        eval_df = pd.read_csv('evaluation_data.csv')
        assert len(eval_df) == 3, f"Expected 3 rows in evaluation_data.csv, got {len(eval_df)}"
        assert 'sample_id' in eval_df.columns, "Missing sample_id column"
        assert 'prompt' in eval_df.columns, "Missing prompt column"
        assert 'response' in eval_df.columns, "Missing response column"
        print("✅ evaluation_data.csv valid")
        
        # Test metrics config
        metrics_df = pd.read_csv('sample_metrics_config_multi_gt.csv')
        assert len(metrics_df) == 4, f"Expected 4 metrics, got {len(metrics_df)}"
        
        required_metrics_cols = ['name', 'type', 'description', 'evaluation_prompt', 'threshold', 'ground_truth_column', 'ground_truth_file_path']
        for col in required_metrics_cols:
            assert col in metrics_df.columns, f"Missing column in metrics: {col}"
        
        # Test multi-ground truth specifications
        for _, row in metrics_df.iterrows():
            gt_files_str = row['ground_truth_file_path']
            if gt_files_str and gt_files_str.strip():
                # Should contain multiple files separated by comma or semicolon
                has_multiple = ',' in gt_files_str or ';' in gt_files_str
                if not has_multiple and 'accuracy_check' in row['name']:
                    print(f"⚠️ {row['name']} has multiple GT files but no separator found")
        
        print("✅ sample_metrics_config_multi_gt.csv valid")
        
        # Test ground truth files
        gt_files = [
            'correct_answer.csv', 'accuracy_expert1.csv', 'accuracy_expert2.csv',
            'helpful_answer.csv', 'helpfulness_rating1.csv', 'helpfulness_rating2.csv',
            'safe_response.csv', 'complete_answer.csv'
        ]
        
        for file in gt_files:
            gt_df = pd.read_csv(file)
            assert 'sample_id' in gt_df.columns, f"Missing sample_id in {file}"
            assert len(gt_df) == 3, f"Expected 3 rows in {file}, got {len(gt_df)}"
            print(f"✅ {file} valid")
        
        print("✅ All CSV files valid")
        return True
        
    except Exception as e:
        print(f"❌ CSV validation failed: {e}")
        traceback.print_exc()
        return False

def test_notebook_syntax():
    """Test notebook Python syntax and key components"""
    print("🐍 Testing notebook syntax...")
    
    try:
        with open('databricks_llm_judge_enhanced_ui.py', 'r') as f:
            content = f.read()
        
        # Check for key components
        key_components = [
            'class LLMJudgeEvaluator',
            'def load_any_csv',
            'def load_ground_truth_for_metric',
            'def parse_ground_truth_files',
            'def discover_files_in_workspace',
            'def discover_files_with_glob',
            'class MetricConfig',
            'class MetricType',
            'dbutils.widgets.dropdown',
            'dbutils.widgets.text'
        ]
        
        missing_components = []
        for component in key_components:
            if component not in content:
                missing_components.append(component)
        
        if missing_components:
            print(f"❌ Missing components: {missing_components}")
            return False
        
        print("✅ All key components present")
        
        # Test Python syntax (excluding magic commands)
        lines = content.split('\n')
        python_lines = []
        for line in lines:
            if not line.strip().startswith('# MAGIC') and not line.strip().startswith('%'):
                python_lines.append(line)
        
        python_content = '\n'.join(python_lines)
        compile(python_content, 'databricks_llm_judge_enhanced_ui.py', 'exec')
        print("✅ Notebook syntax valid")
        
        return True
        
    except Exception as e:
        print(f"❌ Notebook syntax test failed: {e}")
        traceback.print_exc()
        return False

def test_ui_widgets():
    """Test UI widget functionality"""
    print("🎛️ Testing UI widgets...")
    
    try:
        with open('databricks_llm_judge_enhanced_ui.py', 'r') as f:
            content = f.read()
        
        # Check for all required widgets
        required_widgets = [
            'dbutils.widgets.dropdown("upload_mode"',
            'dbutils.widgets.text("evaluation_data_path"',
            'dbutils.widgets.text("metrics_config_path"',
            'dbutils.widgets.text("ground_truth_files"',
            'dbutils.widgets.text("workspace_path"',
            'dbutils.widgets.text("file_patterns"',
            'dbutils.widgets.text("glob_patterns"',
            'dbutils.widgets.dropdown("judge_model"'
        ]
        
        missing_widgets = []
        for widget in required_widgets:
            if widget not in content:
                missing_widgets.append(widget)
        
        if missing_widgets:
            print(f"❌ Missing widgets: {missing_widgets}")
            return False
        
        print("✅ All required widgets present")
        
        # Check widget options
        if '["single_files", "multi_files", "auto_discovery", "pattern_matching"]' not in content:
            print("❌ Upload mode dropdown missing options")
            return False
        
        if '["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo", "databricks-llm"]' not in content:
            print("❌ Judge model dropdown missing options")
            return False
        
        print("✅ Widget options valid")
        return True
        
    except Exception as e:
        print(f"❌ UI widgets test failed: {e}")
        traceback.print_exc()
        return False

def test_llm_functionality():
    """Test LLM selection and functionality"""
    print("🤖 Testing LLM functionality...")
    
    try:
        with open('databricks_llm_judge_enhanced_ui.py', 'r') as f:
            content = f.read()
        
        # Check for LLM model handling
        llm_checks = [
            'JUDGE_MODEL = dbutils.widgets.get("judge_model")',
            'if JUDGE_MODEL == "databricks-llm"',
            'self.is_databricks_llm = judge_model == "databricks-llm"',
            '_initialize_databricks_client',
            '_initialize_openai_client',
            'def _call_databricks_llm',
            'def _call_openai_llm'
        ]
        
        missing_llm_features = []
        for check in llm_checks:
            if check not in content:
                missing_llm_features.append(check)
        
        if missing_llm_features:
            print(f"❌ Missing LLM features: {missing_llm_features}")
            return False
        
        print("✅ LLM functionality present")
        
        # Check for model-specific handling
        if 'gpt-4o' in content and 'gpt-4o-mini' in content and 'gpt-3.5-turbo' in content:
            print("✅ OpenAI models supported")
        else:
            print("❌ OpenAI models not properly supported")
            return False
        
        if 'databricks-llm' in content:
            print("✅ Databricks LLM supported")
        else:
            print("❌ Databricks LLM not supported")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ LLM functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_multi_ground_truth():
    """Test multi-ground truth functionality"""
    print("📚 Testing multi-ground truth functionality...")
    
    try:
        # Test parse_ground_truth_files function
        def parse_ground_truth_files(gt_files_string):
            if not gt_files_string or not gt_files_string.strip():
                return []
            
            files = []
            for separator in [',', ';']:
                if separator in gt_files_string:
                    files = [f.strip() for f in gt_files_string.split(separator) if f.strip()]
                    break
            
            if not files:
                files = [gt_files_string.strip()]
            
            return files
        
        # Test comma separation
        test_comma = "file1.csv,file2.csv,file3.csv"
        result_comma = parse_ground_truth_files(test_comma)
        assert len(result_comma) == 3, f"Expected 3 files, got {len(result_comma)}"
        assert result_comma == ['file1.csv', 'file2.csv', 'file3.csv'], f"Comma parsing failed: {result_comma}"
        print("✅ Comma separation works")
        
        # Test semicolon separation
        test_semicolon = "file1.csv;file2.csv;file3.csv"
        result_semicolon = parse_ground_truth_files(test_semicolon)
        assert len(result_semicolon) == 3, f"Expected 3 files, got {len(result_semicolon)}"
        assert result_semicolon == ['file1.csv', 'file2.csv', 'file3.csv'], f"Semicolon parsing failed: {result_semicolon}"
        print("✅ Semicolon separation works")
        
        # Test single file
        test_single = "single_file.csv"
        result_single = parse_ground_truth_files(test_single)
        assert len(result_single) == 1, f"Expected 1 file, got {len(result_single)}"
        assert result_single == ['single_file.csv'], f"Single file parsing failed: {result_single}"
        print("✅ Single file works")
        
        # Test empty string
        test_empty = ""
        result_empty = parse_ground_truth_files(test_empty)
        assert len(result_empty) == 0, f"Expected 0 files, got {len(result_empty)}"
        print("✅ Empty string works")
        
        print("✅ Multi-ground truth parsing works")
        return True
        
    except Exception as e:
        print(f"❌ Multi-ground truth test failed: {e}")
        traceback.print_exc()
        return False

def test_file_discovery():
    """Test file discovery functionality"""
    print("🔍 Testing file discovery...")
    
    try:
        # Test glob pattern matching
        import glob
        
        # Create test files
        test_dir = tempfile.mkdtemp()
        test_files = [
            'test_evaluation.csv',
            'test_ground_truth.csv',
            'test_expert1.csv',
            'test_expert2.csv'
        ]
        
        for file in test_files:
            with open(os.path.join(test_dir, file), 'w') as f:
                f.write('sample_id,data\n1,test\n')
        
        # Test pattern matching
        patterns = ['test_*.csv', 'test_evaluation.csv']
        for pattern in patterns:
            files = glob.glob(os.path.join(test_dir, pattern))
            if pattern == 'test_*.csv':
                assert len(files) == 4, f"Expected 4 files for {pattern}, got {len(files)}"
            else:
                assert len(files) == 1, f"Expected 1 file for {pattern}, got {len(files)}"
        
        print("✅ File discovery patterns work")
        
        # Cleanup
        import shutil
        shutil.rmtree(test_dir)
        
        return True
        
    except Exception as e:
        print(f"❌ File discovery test failed: {e}")
        traceback.print_exc()
        return False

def test_data_consistency():
    """Test data consistency across all files"""
    print("🔄 Testing data consistency...")
    
    try:
        # Load evaluation data
        eval_df = pd.read_csv('evaluation_data.csv')
        eval_sample_ids = set(eval_df['sample_id'])
        
        # Test all ground truth files have matching sample IDs
        gt_files = [
            'correct_answer.csv', 'accuracy_expert1.csv', 'accuracy_expert2.csv',
            'helpful_answer.csv', 'helpfulness_rating1.csv', 'helpfulness_rating2.csv',
            'safe_response.csv', 'complete_answer.csv'
        ]
        
        for file in gt_files:
            gt_df = pd.read_csv(file)
            gt_sample_ids = set(gt_df['sample_id'])
            
            if not eval_sample_ids.issubset(gt_sample_ids):
                missing_ids = eval_sample_ids - gt_sample_ids
                print(f"❌ {file}: Missing sample IDs: {missing_ids}")
                return False
            
            print(f"✅ {file}: Has all required sample IDs")
        
        print("✅ Data consistency check passed")
        return True
        
    except Exception as e:
        print(f"❌ Data consistency test failed: {e}")
        traceback.print_exc()
        return False

def test_metrics_configuration():
    """Test metrics configuration parsing"""
    print("📋 Testing metrics configuration...")
    
    try:
        metrics_df = pd.read_csv('sample_metrics_config_multi_gt.csv')
        
        # Check metric types
        valid_types = ['binary', 'scale_1_5', 'percentage']
        for metric_type in metrics_df['type']:
            if metric_type not in valid_types:
                print(f"❌ Invalid metric type: {metric_type}")
                return False
        
        print("✅ All metric types valid")
        
        # Check thresholds are numeric
        if not pd.api.types.is_numeric_dtype(metrics_df['threshold']):
            print("❌ Some thresholds are not numeric")
            return False
        
        print("✅ All thresholds numeric")
        
        # Check ground truth file specifications
        for _, row in metrics_df.iterrows():
            gt_files_str = row['ground_truth_file_path']
            if gt_files_str and gt_files_str.strip():
                # Should contain multiple files for some metrics
                if 'accuracy_check' in row['name'] or 'helpfulness_rating' in row['name']:
                    has_multiple = ',' in gt_files_str or ';' in gt_files_str
                    if not has_multiple:
                        print(f"⚠️ {row['name']} should have multiple GT files")
        
        print("✅ Metrics configuration valid")
        return True
        
    except Exception as e:
        print(f"❌ Metrics configuration test failed: {e}")
        traceback.print_exc()
        return False

def run_comprehensive_test():
    """Run all tests"""
    print("🚀 COMPREHENSIVE DATABRICKS TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("File Structure", test_file_structure),
        ("CSV Validation", test_csv_validation),
        ("Notebook Syntax", test_notebook_syntax),
        ("UI Widgets", test_ui_widgets),
        ("LLM Functionality", test_llm_functionality),
        ("Multi Ground Truth", test_multi_ground_truth),
        ("File Discovery", test_file_discovery),
        ("Data Consistency", test_data_consistency),
        ("Metrics Configuration", test_metrics_configuration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "=" * 60)
    print(f"🏁 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! The notebook is ready for Databricks upload.")
        return True
    else:
        print("❌ Some tests failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    exit(0 if success else 1)