# 🚀 DATABRICKS LLM EVALUATION - SINGLE CELL VERSION
# Copy this entire cell into your Databricks notebook
# Works for ANY user automatically - no configuration needed!

import os
import time
import pandas as pd
import mlflow
from typing import Dict, Any, List, Optional

# =============================================================================
# AUTO-DETECT CURRENT USER AND SETUP DIRECTORIES
# =============================================================================

def get_current_username():
    """Get current Databricks username automatically."""
    try:
        # Method 1: Try dbutils (most reliable)
        username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
        if username:
            return username
    except:
        pass
    
    try:
        # Method 2: Try spark context
        username = spark.conf.get("spark.databricks.clusterUsageTags.clusterOwner", None)
        if username:
            return username
    except:
        pass
    
    # Method 3: Environment variable fallback
    return os.environ.get("DATABRICKS_USER", "unknown_user@databricks.com")

def setup_user_directories(username=None):
    """Setup directories for current user automatically."""
    if username is None:
        username = get_current_username()
    
    # Create user-specific output directory
    clean_username = username.replace("@", "_at_").replace(".", "_")
    output_dir = f"/tmp/llm_eval_results_{clean_username}"
    os.makedirs(output_dir, exist_ok=True)
    
    return {
        'username': username,
        'output_dir': output_dir,
        'mlflow_experiment': f"/Users/{username}/llm_evaluation_experiment",
        'workspace_path': f"/Workspace/Users/{username}"
    }

# Auto-setup for current user
USER_CONFIG = setup_user_directories()

print("🎯 AUTOMATIC SETUP COMPLETE!")
print("="*50)
print(f"👤 Username: {USER_CONFIG['username']}")
print(f"📁 Your Results Directory: {USER_CONFIG['output_dir']}")
print(f"🧪 Your MLflow Experiment: {USER_CONFIG['mlflow_experiment']}")

# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def auto_generate_evaluation_prompt(metric_name: str, metric_type: str, description: str, grading_rubric: str) -> str:
    """Auto-generate evaluation prompt from grading rubric."""
    rubric_section = f"\n**Grading Rubric:**\n{grading_rubric}\n" if grading_rubric else ""
    
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

def load_metrics_from_csv():
    """Load metrics from CSV and auto-generate evaluation prompts."""
    if METRICS_CONFIG_DATA is None:
        return []
    
    metric_configs = []
    for _, row in METRICS_CONFIG_DATA.iterrows():
        metric_type_str = row['type'].strip().lower()
        if metric_type_str == 'binary':
            metric_type = 'binary'
        elif metric_type_str in ['1-5_scale', 'scale_1_5']:
            metric_type = 'scale_1_5'
        elif metric_type_str == 'percentage':
            metric_type = 'percentage'
        else:
            metric_type = 'binary'
        
        # Check format and generate prompt
        if 'grading_rubric' in row and pd.notna(row.get('grading_rubric')):
            grading_rubric = str(row['grading_rubric']).strip()
            prompt_template = auto_generate_evaluation_prompt(
                metric_name=row['name'].strip(),
                metric_type=metric_type_str,
                description=row.get('description', '').strip(),
                grading_rubric=grading_rubric
            )
            print(f"✅ {row['name'].strip()} - Auto-generated prompt from grading rubric")
        elif 'evaluation_prompt' in row and pd.notna(row.get('evaluation_prompt')):
            prompt_template = row['evaluation_prompt'].strip()
            print(f"✅ {row['name'].strip()} - Using provided evaluation_prompt")
        else:
            prompt_template = auto_generate_evaluation_prompt(
                metric_name=row['name'].strip(),
                metric_type=metric_type_str,
                description=row.get('description', '').strip(),
                grading_rubric=""
            )
            print(f"⚠️ {row['name'].strip()} - No rubric or prompt provided, using basic prompt")
        
        metric_config = {
            'name': row['name'].strip(),
            'metric_type': metric_type,
            'description': row.get('description', '').strip(),
            'prompt_template': prompt_template,
            'threshold': float(row['threshold']),
            'ground_truth_column': row['ground_truth_column'].strip(),
            'ground_truth_file_path': row.get('ground_truth_file_path', '').strip()
        }
        
        metric_configs.append(metric_config)
    
    return metric_configs

# =============================================================================
# YOUR DATA - REPLACE WITH YOUR ACTUAL DATA
# =============================================================================

# STEP 1: Replace with your evaluation data
EVALUATION_DATA = pd.DataFrame({
    'prompt': [
        'What is my home buying budget?',
        'Can I afford a $400k house?', 
        'What factors affect my mortgage rate?',
        'How much should I save for a down payment?',
        'What is PMI and when do I need it?'
    ],
    'response': [
        'Based on your $80k income and $300 monthly debts, your budget is approximately $285k.',
        'With your current financial profile, a $400k house would require $2,800/month payments.',
        'Your credit score, down payment, loan term, and market rates affect your mortgage rate.',
        'Aim for 3-20% down payment. For a $300k house, save $9k-$60k plus closing costs.',
        'PMI is required when down payment is less than 20% of home value.'
    ],
    'ground_truth': [
        'Budget calculation based on 36% DTI rule and user financial profile.',
        'Affordability analysis with payment breakdown and DTI consideration.',
        'Credit score, down payment, loan term, market conditions affect rates.',
        'Down payment recommendations with specific dollar amounts.',
        'PMI explanation with threshold and cost information.'
    ]
})

# STEP 2: Replace with your metrics configuration
METRICS_CONFIG_DATA = pd.DataFrame({
    'name': [
        'Financial_Accuracy',
        'Personalization_Quality', 
        'Completeness',
        'Clarity',
        'Actionability'
    ],
    'type': [
        'binary',
        'scale_1_5',
        'scale_1_5', 
        'scale_1_5',
        'binary'
    ],
    'description': [
        'Are the financial calculations and recommendations accurate?',
        'How well personalized is the response to the user?',
        'How complete is the response in addressing the question?',
        'How clear and understandable is the response?',
        'Does the response provide actionable next steps?'
    ],
    'grading_rubric': [
        'Score 1 if financial calculations are mathematically correct and recommendations align with industry standards. Score 0 if calculations are wrong or recommendations are inappropriate.',
        'Score 1-5 based on personalization level: 1=generic response, 3=some personalization, 5=highly personalized with specific user data.',
        'Score 1-5 based on completeness: 1=minimal information, 3=adequate coverage, 5=comprehensive and thorough.',
        'Score 1-5 based on clarity: 1=confusing/unclear, 3=mostly clear, 5=crystal clear and well-structured.',
        'Score 1 if response includes specific next steps or actions. Score 0 if response is purely informational without guidance.'
    ],
    'threshold': [1.0, 3.0, 3.0, 3.0, 1.0],
    'ground_truth_column': ['ground_truth'] * 5,
    'ground_truth_file_path': [''] * 5
})

print(f"\n📊 Loaded {len(EVALUATION_DATA)} samples for evaluation")
print(f"📈 Configured {len(METRICS_CONFIG_DATA)} metrics")

# =============================================================================
# RUN EVALUATION - SAVES TO YOUR DIRECTORY AUTOMATICALLY
# =============================================================================

def run_evaluation():
    """Run evaluation and save results to current user's directory."""
    
    print("\n" + "="*60)
    print("🚀 STARTING EVALUATION")
    print("="*60)
    print(f"👤 User: {USER_CONFIG['username']}")
    print(f"💾 Results will be saved to: {USER_CONFIG['output_dir']}")
    
    # Load metrics
    metric_configs = load_metrics_from_csv()
    
    if not metric_configs:
        print("❌ No valid metrics!")
        return pd.DataFrame()
    
    # Run evaluation (replace with actual LLM judge)
    print("\n🔄 Running evaluation...")
    start_time = time.time()
    
    results = []
    for idx, row in EVALUATION_DATA.iterrows():
        for metric_config in metric_configs:
            # Simulate scoring (replace with actual evaluation)
            import random
            if metric_config['metric_type'] == 'binary':
                score = random.choice([0.0, 1.0])
            elif metric_config['metric_type'] == 'scale_1_5':
                score = random.randint(1, 5)
            else:
                score = random.random() * 100
            
            result = {
                'sample_id': idx,
                'metric_name': metric_config['name'],
                'score': score,
                'threshold': metric_config['threshold'],
                'status': '✅' if score >= metric_config['threshold'] else '❌',
                'prompt': row.get('prompt', ''),
                'response': row.get('response', ''),
                'ground_truth': row.get('ground_truth', '')
            }
            results.append(result)
    
    results_df = pd.DataFrame(results)
    eval_time = time.time() - start_time
    
    # Calculate summary
    total = len(results_df)
    passed = len(results_df[results_df['status'] == '✅'])
    pass_rate = (passed / total) * 100 if total > 0 else 0
    
    print(f"✅ COMPLETE in {eval_time:.1f}s")
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
    
    # Save results to user's directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = f"{USER_CONFIG['output_dir']}/results_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n💾 Results saved to: {csv_path}")
    
    # Log to MLflow in user's experiment
    try:
        mlflow.set_experiment(USER_CONFIG['mlflow_experiment'])
        
        with mlflow.start_run(run_name=f"eval_gpt4_{timestamp}"):
            mlflow.log_param("judge_model", "gpt-4")
            mlflow.log_param("num_samples", len(EVALUATION_DATA))
            mlflow.log_param("user", USER_CONFIG['username'])
            
            mlflow.log_metric("overall_pass_rate", pass_rate)
            mlflow.log_metric("total_evaluations", total)
            mlflow.log_metric("passed_evaluations", passed)
            
            # Log per-metric results
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
            
            # Log results file
            mlflow.log_artifact(csv_path, artifact_path="tables")
        
        print(f"✅ Results logged to MLflow: {USER_CONFIG['mlflow_experiment']}")
        
    except Exception as e:
        print(f"⚠️ MLflow logging failed: {e}")
    
    # Store globally for analysis
    globals()['results_df'] = results_df
    
    return results_df

# RUN THE EVALUATION
results_df = run_evaluation()

# =============================================================================
# VIEW RESULTS
# =============================================================================

print("\n" + "="*60)
print("📊 DETAILED RESULTS")
print("="*60)

# Show first 10 results
display(results_df.head(10))

# Summary statistics by metric
print("\n📈 SUMMARY BY METRIC")
print("="*50)

summary_stats = results_df.groupby('metric_name').agg({
    'score': ['mean', 'std', 'count'],
    'status': lambda x: (x == '✅').sum()
}).round(2)

summary_stats.columns = ['Mean_Score', 'Std_Score', 'Total_Samples', 'Passed_Samples']
summary_stats['Pass_Rate_%'] = (summary_stats['Passed_Samples'] / summary_stats['Total_Samples'] * 100).round(1)

display(summary_stats)

# Show your file locations
print("\n📁 YOUR RESULT FILES")
print("="*50)
print(f"Results Directory: {USER_CONFIG['output_dir']}")
print(f"MLflow Experiment: {USER_CONFIG['mlflow_experiment']}")

# List your result files
if os.path.exists(USER_CONFIG['output_dir']):
    files = [f for f in os.listdir(USER_CONFIG['output_dir']) if f.endswith('.csv')]
    print(f"\n📄 Your result files ({len(files)} found):")
    for file in sorted(files):
        file_path = os.path.join(USER_CONFIG['output_dir'], file)
        file_size = os.path.getsize(file_path)
        print(f"  • {file} ({file_size:,} bytes)")
else:
    print("No result files found.")

print(f"\n🎉 EVALUATION COMPLETE!")
print(f"📊 Results: {len(results_df)} evaluations")
print(f"💾 Saved to: {USER_CONFIG['output_dir']}")
print(f"🧪 MLflow: {USER_CONFIG['mlflow_experiment']}")