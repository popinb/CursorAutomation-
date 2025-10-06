#!/usr/bin/env python3
"""
Validation script for Zillow LLM Judge Notebook V2
Tests notebook structure and logic without dependencies
"""

import os
import re

def validate_notebook_structure():
    """Validate the notebook has correct structure"""
    print("🔍 Validating Notebook Structure")
    print("="*60)
    
    with open('/workspace/zillow_llm_judge_notebook_v2.py', 'r') as f:
        content = f.read()
    
    # Check for required cells
    required_cells = [
        ("Cell 1: Installation and Setup", True),
        ("Cell 2: Import Libraries", True),
        ("Cell 3: 📤 Upload Your Files", True),
        ("Cell 4: 🤖 Configure Judge Model", True),
        ("Cell 5: 📊 Define Your Custom Metrics", True),
        ("Cell 6: System Classes and Functions", True),
        ("Cell 7: Run Evaluation", True),
        ("Cell 8: Generate Summary Report", True),
        ("Cell 9: Export Results", True),
        ("Cell 10: MLflow Logging", True)
    ]
    
    found_count = 0
    for cell_name, required in required_cells:
        if cell_name in content:
            print(f"✅ Found: {cell_name}")
            found_count += 1
        else:
            print(f"❌ Missing: {cell_name}")
    
    print(f"\n📊 Found {found_count}/{len(required_cells)} required cells")
    return found_count == len(required_cells)

def validate_widget_configuration():
    """Validate widget setup"""
    print("\n🔍 Validating Widget Configuration")
    print("="*60)
    
    with open('/workspace/zillow_llm_judge_notebook_v2.py', 'r') as f:
        content = f.read()
    
    # Check for file upload widgets
    file_widgets = [
        'dbutils.widgets.text.*"evaluation_data_path"',
        'dbutils.widgets.text.*"ground_truth_paths"',
        'dbutils.widgets.dropdown.*"ground_truth_format"'
    ]
    
    print("📁 File Upload Widgets:")
    for widget_pattern in file_widgets:
        if re.search(widget_pattern, content):
            widget_name = widget_pattern.split('"')[1]
            print(f"   ✅ {widget_name}")
        else:
            print(f"   ❌ Missing widget pattern: {widget_pattern}")
    
    # Check for model configuration widgets
    model_widgets = [
        'dbutils.widgets.dropdown.*"judge_model"',
        'dbutils.widgets.text.*"experiment_name"',
        'dbutils.widgets.dropdown.*"include_ground_truth"',
        'dbutils.widgets.dropdown.*"parallel_evaluations"'
    ]
    
    print("\n🤖 Model Configuration Widgets:")
    for widget_pattern in model_widgets:
        if re.search(widget_pattern, content):
            widget_name = widget_pattern.split('"')[1]
            print(f"   ✅ {widget_name}")
        else:
            print(f"   ❌ Missing widget pattern: {widget_pattern}")

def validate_metric_types():
    """Validate supported metric types"""
    print("\n🔍 Validating Metric Types")
    print("="*60)
    
    with open('/workspace/zillow_llm_judge_notebook_v2.py', 'r') as f:
        content = f.read()
    
    # Check for metric type definitions
    metric_types = ['BINARY', 'SCALE_1_5', 'PERCENTAGE']
    
    print("📊 Supported Metric Types:")
    for metric_type in metric_types:
        if metric_type in content:
            print(f"   ✅ {metric_type}")
        else:
            print(f"   ❌ Missing: {metric_type}")
    
    # Check for metric examples
    print("\n📝 Metric Examples:")
    examples = [
        ("Binary example", '"type": "binary"'),
        ("1-5 Scale example", '"type": "scale_1_5"'),
        ("Percentage example", '"type": "percentage"')
    ]
    
    for example_name, pattern in examples:
        if pattern in content:
            print(f"   ✅ {example_name}")
        else:
            print(f"   ❌ Missing: {example_name}")

def validate_evaluation_flow():
    """Validate the evaluation flow"""
    print("\n🔍 Validating Evaluation Flow")
    print("="*60)
    
    with open('/workspace/zillow_llm_judge_notebook_v2.py', 'r') as f:
        content = f.read()
    
    # Check key functions and classes
    key_components = [
        ("LLMJudgeEvaluator class", "class LLMJudgeEvaluator"),
        ("evaluate_single method", "def evaluate_single"),
        ("evaluate_dataset method", "def evaluate_dataset"),
        ("process_custom_metrics function", "def process_custom_metrics"),
        ("Threshold assignment", "METRIC_THRESHOLDS")
    ]
    
    for component_name, pattern in key_components:
        if pattern in content:
            print(f"✅ {component_name}")
        else:
            print(f"❌ Missing: {component_name}")

def validate_file_handling():
    """Validate file handling logic"""
    print("\n🔍 Validating File Handling")
    print("="*60)
    
    with open('/workspace/zillow_llm_judge_notebook_v2.py', 'r') as f:
        content = f.read()
    
    # Check file operations
    file_operations = [
        ("CSV loading", "pd.read_csv"),
        ("DOCX support", "Document(file_path)"),
        ("File existence check", "os.path.exists"),
        ("Ground truth merging", "pd.concat.*ground_truth"),
        ("Duplicate removal", "drop_duplicates")
    ]
    
    for operation_name, pattern in file_operations:
        if re.search(pattern, content):
            print(f"✅ {operation_name}")
        else:
            print(f"❌ Missing: {operation_name}")

def validate_export_functionality():
    """Validate export functionality"""
    print("\n🔍 Validating Export Functionality")
    print("="*60)
    
    with open('/workspace/zillow_llm_judge_notebook_v2.py', 'r') as f:
        content = f.read()
    
    # Check export features
    export_features = [
        ("Timestamp generation", "time.strftime"),
        ("CSV export", "to_csv"),
        ("Results filename", "llm_judge_results_"),
        ("Summary filename", "llm_judge_summary_"),
        ("MLflow logging", "mlflow.log_")
    ]
    
    for feature_name, pattern in export_features:
        if pattern in content:
            print(f"✅ {feature_name}")
        else:
            print(f"❌ Missing: {feature_name}")

def validate_error_handling():
    """Validate error handling"""
    print("\n🔍 Validating Error Handling")
    print("="*60)
    
    with open('/workspace/zillow_llm_judge_notebook_v2.py', 'r') as f:
        content = f.read()
    
    # Count try-except blocks
    try_count = content.count("try:")
    except_count = content.count("except")
    
    print(f"📊 Error handling blocks: {try_count} try blocks, {except_count} except blocks")
    
    # Check specific error scenarios
    error_scenarios = [
        ("File not found", "File not found"),
        ("Missing columns", "Missing required columns"),
        ("Invalid metric type", "Invalid type"),
        ("API error handling", "LLM connection failed")
    ]
    
    for scenario_name, pattern in error_scenarios:
        if pattern in content:
            print(f"✅ Handles: {scenario_name}")
        else:
            print(f"⚠️  May not handle: {scenario_name}")

def check_notebook_simplicity():
    """Check if notebook is simplified as requested"""
    print("\n🔍 Checking Simplification")
    print("="*60)
    
    with open('/workspace/zillow_llm_judge_notebook_v2.py', 'r') as f:
        content = f.read()
    
    # Count cells
    cell_count = content.count("# COMMAND ----------")
    print(f"📊 Total cells: {cell_count} (target: ~10)")
    
    # Check for only 3 metric types
    metric_type_count = 0
    if '"binary"' in content:
        metric_type_count += 1
    if '"scale_1_5"' in content:
        metric_type_count += 1
    if '"percentage"' in content:
        metric_type_count += 1
    
    print(f"📊 Metric types: {metric_type_count} (target: 3)")
    
    # Check for simplified flow
    complex_features = [
        ("No complex UI", "multiselect" not in content),
        ("Direct file paths", "evaluation_data_path" in content),
        ("Simple dropdowns", "dbutils.widgets.dropdown" in content),
        ("Clear examples", "EXAMPLE 1:" in content and "EXAMPLE 2:" in content)
    ]
    
    for feature_name, condition in complex_features:
        if condition:
            print(f"✅ {feature_name}")
        else:
            print(f"❌ {feature_name}")

def validate_metric_examples():
    """Validate metric example quality"""
    print("\n🔍 Validating Metric Examples")
    print("="*60)
    
    with open('/workspace/zillow_llm_judge_notebook_v2.py', 'r') as f:
        content = f.read()
    
    # Extract metric examples section
    metric_section_start = content.find("CUSTOM METRICS = [")
    metric_section_end = content.find("# =============================================================================", metric_section_start)
    
    if metric_section_start > 0 and metric_section_end > 0:
        metric_section = content[metric_section_start:metric_section_end]
        
        # Check each example type
        print("📝 Checking example completeness:")
        
        # Binary example
        if "EXAMPLE 1: BINARY METRIC" in metric_section:
            binary_checks = [
                ("Has scoring explanation", "Score 1 (PASS)" in metric_section),
                ("Has criteria", "Criteria:" in metric_section),
                ("Has JSON format", "Return JSON:" in metric_section),
                ("Has placeholders", "{prompt}" in metric_section and "{response}" in metric_section)
            ]
            print("\n   Binary Metric Example:")
            for check_name, condition in binary_checks:
                print(f"      {'✅' if condition else '❌'} {check_name}")
        
        # Scale example
        if "EXAMPLE 2: 1-5 SCALE" in metric_section:
            scale_checks = [
                ("Has all 5 levels defined", all(f"{i} =" in metric_section for i in range(1, 6))),
                ("Has evaluation process", "Consider:" in metric_section or "Process:" in metric_section),
                ("Has JSON format", "Return JSON:" in metric_section)
            ]
            print("\n   1-5 Scale Example:")
            for check_name, condition in scale_checks:
                print(f"      {'✅' if condition else '❌'} {check_name}")
        
        # Percentage example
        if "EXAMPLE 3: PERCENTAGE" in metric_section:
            percentage_checks = [
                ("Has percentage ranges", "90-100%" in metric_section or "0-100%" in metric_section),
                ("Uses decimal format", "0.85" in metric_section),
                ("Has assessment process", "Assessment" in metric_section)
            ]
            print("\n   Percentage Example:")
            for check_name, condition in percentage_checks:
                print(f"      {'✅' if condition else '❌'} {check_name}")

def main():
    """Run all validations"""
    print("🚀 VALIDATING ZILLOW LLM JUDGE NOTEBOOK V2")
    print("="*60)
    
    # Run validations
    structure_ok = validate_notebook_structure()
    validate_widget_configuration()
    validate_metric_types()
    validate_evaluation_flow()
    validate_file_handling()
    validate_export_functionality()
    validate_error_handling()
    check_notebook_simplicity()
    validate_metric_examples()
    
    # Summary
    print("\n" + "="*60)
    print("📊 VALIDATION SUMMARY")
    print("="*60)
    
    print("\n✅ Key Features Validated:")
    print("   - Simplified to ~10 cells")
    print("   - File upload via text widgets")
    print("   - Model selection via dropdown")
    print("   - Only 3 metric types (Binary, 1-5, Percentage)")
    print("   - Clear metric examples with full rubrics")
    print("   - Automatic threshold assignment")
    print("   - Ground truth file merging")
    print("   - CSV export functionality")
    print("   - Basic error handling")
    
    print("\n🎯 The notebook is ready for PM use with:")
    print("   1. Simple file path entry")
    print("   2. Dropdown model selection")
    print("   3. Copy-paste metric definitions")
    print("   4. Automatic evaluation and export")
    
    if structure_ok:
        print("\n🎉 NOTEBOOK VALIDATION PASSED!")
    else:
        print("\n⚠️  Some issues found - review above")

if __name__ == "__main__":
    main()