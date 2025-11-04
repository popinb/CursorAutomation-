# Databricks notebook source
# MAGIC %md
# MAGIC # LLM-as-a-Judge: v4 FIXED - Full Features Working
# MAGIC 
# MAGIC **Complete with Add/Edit/Delete/View - No widget storage, no sync issues**

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 1: Install Packages

# COMMAND ----------

%pip install openai pandas requests --quiet
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2: Load Data

# COMMAND ----------

import pandas as pd
import json

print("="*80)
print("CELL 2: LOADING DATA")
print("="*80)

# Hardcoded defaults
METRICS_CONFIG_JSON = [
    {
        "name": "Story_Accuracy",
        "type": "binary",
        "description": "Evaluate if the response is factually accurate about the Cinderella story",
        "grading_rubric": "Score 1 if all story facts are correct and align with the classic Cinderella tale. Score 0 if any facts are incorrect, made up, or contradict the original story.",
        "threshold": "1",
        "ground_truth_file_path": "ground_truth.csv",
        "ground_truth_column": "correct_answer"
    },
    {
        "name": "Response_Completeness",
        "type": "1-5_scale",
        "description": "Evaluate how complete and thorough the response is",
        "grading_rubric": "5=Fully complete, addresses all aspects comprehensively; 4=Mostly complete with minor gaps; 3=Partially complete, missing some details; 2=Barely complete, many gaps; 1=Incomplete or inadequate",
        "threshold": "4",
        "ground_truth_file_path": "ground_truth.csv",
        "ground_truth_column": "correct_answer"
    },
    {
        "name": "Child_Friendliness",
        "type": "percentage",
        "description": "Evaluate what percentage of the response is appropriate and understandable for children",
        "grading_rubric": "100%=Perfectly child-friendly language and content; 75%=Mostly appropriate with minor complex words; 50%=Somewhat child-friendly; 25%=Barely appropriate for children; 0%=Not suitable for children",
        "threshold": "75",
        "ground_truth_file_path": "ground_truth.csv",
        "ground_truth_column": "correct_answer"
    }
]

EVALUATION_DATA_JSON = [
    {"sample_id": 1, "prompt": "Who is Cinderella?", "response": "Cinderella is a young girl who lives with her mean stepmother and two stepsisters. With help from her Fairy Godmother, she goes to the prince's ball and loses her glass slipper at midnight."},
    {"sample_id": 2, "prompt": "What did the Fairy Godmother turn into a carriage?", "response": "The Fairy Godmother used her magic wand to turn a big orange pumpkin into a beautiful golden carriage."},
    {"sample_id": 3, "prompt": "What happened at midnight?", "response": "When the clock struck midnight, Cinderella had to run away from the ball. In her hurry, she lost one of her glass slippers on the palace steps."}
]

GROUND_TRUTH_JSON = [
    {"sample_id": 1, "correct_answer": "Cinderella is a kind young girl mistreated by her stepmother and stepsisters. With help from her Fairy Godmother, she attends a royal ball, loses her glass slipper at midnight, and is found by the prince.", "story_element": "Main plot"},
    {"sample_id": 2, "correct_answer": "A pumpkin", "story_element": "Magic transformation"},
    {"sample_id": 3, "correct_answer": "Cinderella had to leave the ball because the magic spell would break at midnight. She ran away and lost her glass slipper.", "story_element": "Midnight deadline"}
]

# Convert to DataFrames
METRICS_CONFIG_DATA = pd.DataFrame(METRICS_CONFIG_JSON)
EVALUATION_DATA = pd.DataFrame(EVALUATION_DATA_JSON)
GROUND_TRUTH_DATA_DF = pd.DataFrame(GROUND_TRUTH_JSON)
GROUND_TRUTH_DATA = {'ground_truth.csv': GROUND_TRUTH_DATA_DF}

print(f"\n? Data loaded:")
print(f"  Metrics: {len(METRICS_CONFIG_DATA)}")
print(f"  Samples: {len(EVALUATION_DATA)}")
print(f"  Ground truth: {len(GROUND_TRUTH_DATA_DF)}")
print("="*80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2.5: Full Metrics Editor (Add/Edit/Delete/View)

# COMMAND ----------

import json

print("="*80)
print("CELL 2.5: METRICS EDITOR - ALL FEATURES")
print("="*80)

# Create action dropdown
try:
    dbutils.widgets.dropdown("action", "view", ["view", "add", "edit", "delete"], "Action")
except:
    pass

action = dbutils.widgets.get("action")

# CREATE/REMOVE WIDGETS BASED ON ACTION
if action == "view":
    print("\n?? VIEW MODE")
    # Remove all form widgets
    for w in ["row_select", "m_name", "m_type", "m_desc", "m_rubric", "m_threshold", "save_btn"]:
        try:
            dbutils.widgets.remove(w)
        except:
            pass

elif action == "add":
    print("\n? ADD MODE")
    try:
        dbutils.widgets.text("m_name", "", "1. Name")
        dbutils.widgets.dropdown("m_type", "binary", ["binary", "1-5_scale", "percentage"], "2. Type")
        dbutils.widgets.text("m_desc", "", "3. Description")
        dbutils.widgets.text("m_rubric", "", "4. Grading Rubric")
        dbutils.widgets.text("m_threshold", "", "5. Threshold (1, 4, or 75)")
        dbutils.widgets.dropdown("save_btn", "no", ["no", "yes"], "6. ? Save?")
    except:
        pass
    # Remove row selector
    try:
        dbutils.widgets.remove("row_select")
    except:
        pass

elif action == "edit":
    print("\n?? EDIT MODE")
    try:
        dbutils.widgets.dropdown("row_select", "1", [str(i+1) for i in range(max(1, len(METRICS_CONFIG_DATA)))], "1. Row to Edit")
        dbutils.widgets.text("m_name", "", "2. Name")
        dbutils.widgets.dropdown("m_type", "binary", ["binary", "1-5_scale", "percentage"], "3. Type")
        dbutils.widgets.text("m_desc", "", "4. Description")
        dbutils.widgets.text("m_rubric", "", "5. Grading Rubric")
        dbutils.widgets.text("m_threshold", "", "6. Threshold")
        dbutils.widgets.dropdown("save_btn", "no", ["no", "yes"], "7. ? Save?")
    except:
        pass

elif action == "delete":
    print("\n??? DELETE MODE")
    try:
        dbutils.widgets.dropdown("row_select", "1", [str(i+1) for i in range(max(1, len(METRICS_CONFIG_DATA)))], "1. Row to Delete")
        dbutils.widgets.dropdown("save_btn", "no", ["no", "yes"], "2. ?? Confirm?")
    except:
        pass
    # Remove form widgets
    for w in ["m_name", "m_type", "m_desc", "m_rubric", "m_threshold"]:
        try:
            dbutils.widgets.remove(w)
        except:
            pass

# PROCESS ACTIONS
if action == "add":
    save_btn = dbutils.widgets.get("save_btn")
    m_name = dbutils.widgets.get("m_name").strip()
    
    if save_btn == "yes" and m_name:
        # Check for duplicates
        current_metrics = METRICS_CONFIG_DATA.to_dict('records')
        existing_names = [m['name'] for m in current_metrics]
        
        if m_name in existing_names:
            print(f"\n? ERROR: Metric '{m_name}' already exists!")
        else:
            # Add new metric
            new_metric = {
                "name": m_name,
                "type": dbutils.widgets.get("m_type"),
                "description": dbutils.widgets.get("m_desc"),
                "grading_rubric": dbutils.widgets.get("m_rubric"),
                "threshold": dbutils.widgets.get("m_threshold"),
                "ground_truth_file_path": "ground_truth.csv",
                "ground_truth_column": "correct_answer"
            }
            current_metrics.append(new_metric)
            METRICS_CONFIG_DATA = pd.DataFrame(current_metrics)
            
            print(f"\n? ADDED: {m_name}")
            
            # Auto-reset
            dbutils.widgets.remove("save_btn")
            dbutils.widgets.dropdown("save_btn", "no", ["no", "yes"], "6. ? Save?")
            dbutils.widgets.remove("m_name")
            dbutils.widgets.text("m_name", "", "1. Name")
    elif save_btn == "yes" and not m_name:
        print("\n??  Please enter a metric name")

elif action == "edit":
    save_btn = dbutils.widgets.get("save_btn")
    m_name = dbutils.widgets.get("m_name").strip()
    
    if save_btn == "yes" and m_name:
        row_idx = int(dbutils.widgets.get("row_select")) - 1
        current_metrics = METRICS_CONFIG_DATA.to_dict('records')
        
        if 0 <= row_idx < len(current_metrics):
            old_name = current_metrics[row_idx]['name']
            
            # Check if new name conflicts with other metrics
            existing_names = [m['name'] for i, m in enumerate(current_metrics) if i != row_idx]
            if m_name in existing_names:
                print(f"\n? ERROR: Metric name '{m_name}' already exists!")
            else:
                # Update metric
                current_metrics[row_idx] = {
                    "name": m_name,
                    "type": dbutils.widgets.get("m_type"),
                    "description": dbutils.widgets.get("m_desc"),
                    "grading_rubric": dbutils.widgets.get("m_rubric"),
                    "threshold": dbutils.widgets.get("m_threshold"),
                    "ground_truth_file_path": "ground_truth.csv",
                    "ground_truth_column": "correct_answer"
                }
                METRICS_CONFIG_DATA = pd.DataFrame(current_metrics)
                
                print(f"\n? EDITED: {old_name} ? {m_name}")
                
                # Auto-reset
                dbutils.widgets.remove("save_btn")
                dbutils.widgets.dropdown("save_btn", "no", ["no", "yes"], "7. ? Save?")
    elif save_btn == "yes" and not m_name:
        print("\n??  Please enter a metric name")

elif action == "delete":
    save_btn = dbutils.widgets.get("save_btn")
    
    if save_btn == "yes":
        row_idx = int(dbutils.widgets.get("row_select")) - 1
        current_metrics = METRICS_CONFIG_DATA.to_dict('records')
        
        if 0 <= row_idx < len(current_metrics):
            deleted_name = current_metrics[row_idx]['name']
            current_metrics.pop(row_idx)
            
            if current_metrics:
                METRICS_CONFIG_DATA = pd.DataFrame(current_metrics)
            else:
                METRICS_CONFIG_DATA = pd.DataFrame(columns=["name", "type", "description", "grading_rubric", "threshold", "ground_truth_file_path", "ground_truth_column"])
            
            print(f"\n??? DELETED: {deleted_name}")
            
            # Update row selector if metrics remain
            if len(current_metrics) > 0:
                dbutils.widgets.remove("row_select")
                dbutils.widgets.dropdown("row_select", "1", [str(i+1) for i in range(len(current_metrics))], "1. Row to Delete")
            
            # Auto-reset
            dbutils.widgets.remove("save_btn")
            dbutils.widgets.dropdown("save_btn", "no", ["no", "yes"], "2. ?? Confirm?")

# DISPLAY TABLE
print(f"\n?? CURRENT METRICS ({len(METRICS_CONFIG_DATA)} total)")
print("="*80)
for idx, row in METRICS_CONFIG_DATA.iterrows():
    gt = f"{row.get('ground_truth_file_path', '')}/{row.get('ground_truth_column', '')}"
    print(f"{idx+1}. {row['name']} ({row['type']}) - Threshold: {row['threshold']} - GT: {gt}")
print("="*80)

# Instructions
if action == "view":
    print("\n?? Select Action (add/edit/delete) to modify metrics")
elif action == "add":
    print("\n?? Fill form, set Save='yes', then re-run cell")
elif action == "edit":
    print("\n?? Select row, fill form, set Save='yes', then re-run cell")
elif action == "delete":
    print("\n?? Select row, set Confirm='yes', then re-run cell")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 3: Configure LLM (continues from here with all features from original v4...)

# COMMAND ----------

print("="*80)
print("CELL 3: CONFIGURE LLM")
print("="*80)

print(f"\n? METRICS_CONFIG_DATA has {len(METRICS_CONFIG_DATA)} metrics:")
for idx, row in METRICS_CONFIG_DATA.iterrows():
    print(f"  {idx+1}. {row['name']}")

print("\n?? These metrics will be used for evaluation in Cell 6")
print("="*80)

        .mtable tbody tr:last-child td {border-bottom: 2px solid #667eea;}
        .mrow {font-weight: 700; color: #667eea; font-size: 17px; text-align: center; width: 50px;}
        .mname {font-weight: 600; color: #1e40af; font-size: 14px;}
        .mtype {display: inline-block; padding: 4px 12px; border-radius: 12px; font-size: 10px; font-weight: 700; text-transform: uppercase;}
        .type-bin {background: #dbeafe; color: #1e40af;}
        .type-scale {background: #fef3c7; color: #92400e;}
        .type-pct {background: #d1fae5; color: #065f46;}
        .mthreshold {text-align: center; font-weight: 700; font-size: 15px; color: #dc2626;}
        .mgt {font-size: 13px;}
        .has-gt {color: #10b981; font-weight: 600;}
        .no-gt {color: #9ca3af;}
    </style>
    <table class="mtable">
        <thead>
            <tr>
                <th>#</th>
                <th>Name</th>
                <th>Type</th>
                <th style="width: 35%;">Description</th>
                <th>Threshold</th>
                <th>Ground Truth</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for idx, row in df.iterrows():
        type_class = "type-bin" if row['type'] == 'binary' else ("type-scale" if '1-5' in row['type'] else "type-pct")
        has_gt = row.get('ground_truth_file_path', '').strip()
        gt_col = row.get('ground_truth_column', '').strip()
        gt_display = f"<span class='has-gt'>? {gt_col if gt_col else 'all'}</span>" if has_gt else "<span class='no-gt'>? None</span>"
        
        desc_short = str(row.get('description', ''))[:80]
        if len(str(row.get('description', ''))) > 80:
            desc_short += "..."
        
        html += f"""
            <tr>
                <td class="mrow">{idx+1}</td>
                <td class="mname">{row['name']}</td>
                <td><span class="mtype {type_class}">{row['type']}</span></td>
                <td class="mgt">{desc_short}</td>
                <td class="mthreshold">{row['threshold']}</td>
                <td class="mgt">{gt_display}</td>
            </tr>
        """
    
    html += "</tbody></table>"
    return html

displayHTML(generate_metrics_table(METRICS_CONFIG_DATA))

# Show detailed table
if len(METRICS_CONFIG_DATA) > 0:
    print("\n?? Detailed Metrics:")
    display(METRICS_CONFIG_DATA)
else:
    print("\n?? No metrics configured!")

print("\n" + "="*80)
print("?? QUICK GUIDE:")
print("="*80)
print("?? TO ADD A NEW METRIC:")
print("   1. Set Action = 'add'")
print("   2. Fill in: Name, Type, Description, Grading Rubric, Threshold")
print("   3. (Optional) Ground Truth File and Column")
print("   4. Re-run this cell")
print()
print("?? TO EDIT AN EXISTING METRIC:")
print("   1. Set Action = 'edit'")
print("   2. Select the Row number")
print("   3. Update the form widgets with new values")
print("   4. Re-run this cell")
print()
print("?? TO DELETE A METRIC:")
print("   1. Set Action = 'delete'")
print("   2. Select the Row number to delete")
print("   3. Re-run this cell")
print()
print("?? TO JUST VIEW METRICS:")
print("   (Metrics are always displayed below)")

print("="*80)

print(f"\n? Metrics ready! Variable: METRICS_CONFIG_DATA ({len(METRICS_CONFIG_DATA)} metrics)")
print("?? Continue to Cell 3 to configure LLM and run evaluation")
print("="*80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 3: Configure LLM Judge Model

# COMMAND ----------

from openai import OpenAI
import requests
import os

print("="*80)
print("CELL 3: CONFIGURE LLM JUDGE MODEL")
print("="*80)

# Create widget for model selection
dbutils.widgets.dropdown(
    "judge_model",
    "databricks-llm",
    ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo", "databricks-llm"],
    "?? Judge Model"
)

JUDGE_MODEL = dbutils.widgets.get("judge_model")
print(f"\nSelected model: {JUDGE_MODEL}")

client = None
client_type = None

# Configure based on selection
if JUDGE_MODEL == "databricks-llm":
    print("\n>>> Configuring Databricks Foundation Model...")
    
    try:
        # Get workspace context (using correct Databricks API)
        dbutils_context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
        databricks_token = dbutils_context.apiToken().get()
        workspace_url = dbutils_context.browserHostName().get()
        
        print(f"  Workspace: {workspace_url}")
        
        # Find Claude Sonnet endpoint
        endpoints_url = f"https://{workspace_url}/api/2.0/serving-endpoints"
        headers = {"Authorization": f"Bearer {databricks_token}"}
        
        print(f"  Querying serving endpoints...")
        response = requests.get(endpoints_url, headers=headers)
        response.raise_for_status()
        
        endpoints = response.json().get("endpoints", [])
        claude_endpoint = None
        
        for endpoint in endpoints:
            endpoint_name = endpoint.get("name", "").lower()
            if "claude" in endpoint_name and "sonnet" in endpoint_name:
                claude_endpoint = endpoint.get("name")
                break
        
        if not claude_endpoint:
            raise ValueError("No Claude Sonnet endpoint found. Available endpoints: " + 
                           ", ".join([e.get("name", "") for e in endpoints]))
        
        print(f"  Found endpoint: {claude_endpoint}")
        
        # Store client config
        client = {
            'type': 'databricks',
            'workspace_url': workspace_url,
            'token': databricks_token,
            'headers': headers,
            'endpoint': claude_endpoint
        }
        client_type = "databricks"
        
        # Test connection
        print(f"\n  Testing Databricks endpoint...")
        test_url = f"https://{workspace_url}/serving-endpoints/{claude_endpoint}/invocations"
        test_payload = {
            "messages": [{"role": "user", "content": "Say OK"}],
            "max_tokens": 5
        }
        test_response = requests.post(test_url, json=test_payload, headers=headers)
        test_response.raise_for_status()
        print(f"  Connection test: SUCCESS")
        
    except Exception as e:
        print(f"\n  ERROR configuring Databricks LLM: {e}")
        raise

else:
    print(f"\n>>> Configuring OpenAI Model: {JUDGE_MODEL}")
    
    try:
        # Try multiple secret scopes for API key
        OPENAI_KEY = None
        scopes = ["popin-secure-scope", "user", "common"]
        
        for scope in scopes:
            try:
                OPENAI_KEY = dbutils.secrets.get(scope, "openai_key")
                print(f"  API key retrieved from: {scope}")
                break
            except:
                continue
        
        if not OPENAI_KEY:
            raise ValueError("Could not retrieve OpenAI API key from any scope")
        
        # Initialize OpenAI client with Zillow proxy
        client = OpenAI(
            base_url="https://api.zillowlabs.com/openai/v1",
            api_key=OPENAI_KEY
        )
        client_type = "openai"
        
        print(f"  Base URL: https://api.zillowlabs.com/openai/v1")
        print(f"  Model: {JUDGE_MODEL}")
        
        # Test connection
        print(f"\n  Testing OpenAI connection...")
        test_response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=5
        )
        print(f"  Connection test: SUCCESS")
        
    except Exception as e:
        print(f"\n  ERROR configuring OpenAI: {e}")
        raise

print(f"\n{'='*80}")
print(f"READY TO EVALUATE with {JUDGE_MODEL}")
print(f"Client type: {client_type}")
print(f"{'='*80}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 4: Define Classes

# COMMAND ----------

from enum import Enum
from dataclasses import dataclass

class MetricType(Enum):
    BINARY = "binary"
    SCALE_1_5 = "1-5_scale"
    PERCENTAGE = "percentage"

@dataclass
class MetricConfig:
    name: str
    description: str
    metric_type: MetricType
    prompt_template: str
    threshold: float
    ground_truth_column: str
    ground_truth_file_path: str = ""

print("Classes defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 5: Create Evaluator (WITH EXTENSIVE DEBUG)

# COMMAND ----------

import json
import re
import os
import pandas as pd
import requests

class LLMJudgeEvaluator:
    """LLM Judge Evaluator with extensive debugging."""
    
    def __init__(self, client, model, metrics, ground_truth_data, client_type):
        self.client = client
        self.model = model
        self.metrics = metrics
        self.ground_truth_data = ground_truth_data
        self.client_type = client_type
        print(f"[INIT] Evaluator created: {client_type}, {len(metrics)} metrics")
    
    def _escape_prompt_template(self, template):
        """Escape prompt template to handle JSON examples while preserving placeholders."""
        # Save actual placeholders
        placeholders = {
            '{prompt}': '<<<PROMPT_PLACEHOLDER>>>',
            '{response}': '<<<RESPONSE_PLACEHOLDER>>>',
            '{ground_truth}': '<<<GROUND_TRUTH_PLACEHOLDER>>>'
        }
        
        escaped = template
        for placeholder, marker in placeholders.items():
            escaped = escaped.replace(placeholder, marker)
        
        # Escape all remaining braces
        escaped = escaped.replace('{', '{{').replace('}', '}}')
        
        # Restore placeholders
        for placeholder, marker in placeholders.items():
            escaped = escaped.replace(marker, placeholder)
        
        return escaped
    
    def _get_ground_truth(self, metric, sample_idx):
        """Get ground truth for a sample."""
        try:
            filename = os.path.basename(metric.ground_truth_file_path) if metric.ground_truth_file_path else None
            if not filename or filename not in self.ground_truth_data:
                return "Not provided"
            
            df = self.ground_truth_data[filename]
            if sample_idx >= len(df):
                return f"Index {sample_idx} out of range"
            
            row = df.iloc[sample_idx]
            all_data = []
            for col, value in row.items():
                if pd.notna(value) and str(value).strip():
                    all_data.append(f"- {col}: {value}")
            
            return "Ground Truth:\n" + "\n".join(all_data) if all_data else "No data"
        except Exception as e:
            print(f"[ERROR] Ground truth error: {e}")
            return f"Error: {e}"
    
    def _call_llm(self, prompt):
        """Call LLM with extensive debugging."""
        print(f"\n    [LLM CALL START]", flush=True)
        print(f"      Client type: {self.client_type}", flush=True)
        print(f"      Model: {self.model}", flush=True)
        print(f"      Prompt length: {len(prompt)} chars", flush=True)
        
        try:
            print(f"      Making API call...", flush=True)
            
            if self.client_type == "databricks":
                # Databricks Serving Endpoint
                import requests
                
                workspace_url = self.client['workspace_url']
                endpoint = self.client['endpoint']
                headers = self.client['headers']
                
                url = f"https://{workspace_url}/serving-endpoints/{endpoint}/invocations"
                payload = {
                    "messages": [
                        {"role": "system", "content": "You are an expert evaluator. Respond with ONLY valid JSON: {\"score\": <number>, \"explanation\": \"<text>\"}"},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 500,
                    "temperature": 0.1
                }
                
                print(f"      Calling Databricks endpoint: {endpoint}", flush=True)
                response = requests.post(url, json=payload, headers=headers)
                response.raise_for_status()
                
                response_json = response.json()
                content = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                if not content:
                    raise ValueError(f"Empty response from Databricks: {response_json}")
                
            else:
                # OpenAI client
                print(f"      Calling OpenAI API (model: {self.model})", flush=True)
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert evaluator. Respond with ONLY valid JSON: {\"score\": <number>, \"explanation\": \"<text>\"}"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=500
                )
                
                content = response.choices[0].message.content
            
            print(f"      API call SUCCESS!", flush=True)
            print(f"      Response length: {len(content)} chars", flush=True)
            print(f"      Response preview: {content[:100]}...", flush=True)
            print(f"    [LLM CALL END]", flush=True)
            
            return content
            
        except Exception as e:
            print(f"\n    [LLM CALL FAILED]", flush=True)
            print(f"      Error: {str(e)}", flush=True)
            import traceback
            print(f"      Traceback:", flush=True)
            print(traceback.format_exc(), flush=True)
            return f'{{"score": 0, "explanation": "API call failed: {str(e)}"}}'
    
    def _parse_response(self, response, metric):
        """Parse LLM response."""
        content = response.strip()
        
        # Remove markdown if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        # Parse JSON
        try:
            data = json.loads(content)
            score = data.get('score', data.get('Score', 0))
            explanation = data.get('explanation', data.get('Explanation', 'No explanation'))
        except Exception as e:
            print(f"\n      [PARSE ERROR]", flush=True)
            print(f"        Error: {str(e)}", flush=True)
            print(f"        Content: {content[:200]}", flush=True)
            
            # Try regex fallback
            try:
                score_match = re.search(r'"score"\s*:\s*(\d+\.?\d*)', content, re.IGNORECASE)
                score = float(score_match.group(1)) if score_match else 0
                explanation = f"Parse error (using regex): {str(e)[:100]}"
                print(f"        Regex fallback: score={score}", flush=True)
            except Exception as e2:
                score = 0
                explanation = f"Complete parse failure: {str(e)[:100]}"
                print(f"        Regex also failed: {e2}", flush=True)
        
        # Normalize score
        try:
            score = float(score)
            if metric.metric_type == MetricType.BINARY:
                score = 1.0 if score > 0.5 else 0.0
            elif metric.metric_type == MetricType.SCALE_1_5:
                score = max(1.0, min(5.0, score))
            elif metric.metric_type == MetricType.PERCENTAGE:
                if score <= 1.0:
                    score = score * 100
                score = max(0.0, min(100.0, score))
        except Exception as e:
            print(f"      [NORMALIZE ERROR] {e}", flush=True)
            score = 0.0
        
        return score, str(explanation)[:500]
    
    def evaluate_single(self, prompt, response, metric, sample_idx):
        """Evaluate single sample."""
        try:
            # Get ground truth
            ground_truth = self._get_ground_truth(metric, sample_idx)
            
            # Escape template and format prompt
            safe_template = self._escape_prompt_template(metric.prompt_template)
            eval_prompt = safe_template.format(
                prompt=prompt,
                response=response,
                ground_truth=ground_truth
            )
            
            # Call LLM
            llm_response = self._call_llm(eval_prompt)
            
            # Parse response
            score, explanation = self._parse_response(llm_response, metric)
            
            # Determine status
            status = "PASS" if score >= metric.threshold else "FAIL"
            
            return {
                "score": score,
                "explanation": explanation,
                "status": status,
                "ground_truth_used": ground_truth != "Not provided"
            }
        except Exception as e:
            print(f"\n    [CRITICAL ERROR in evaluate_single]", flush=True)
            print(f"      Metric: {metric.name}", flush=True)
            print(f"      Error: {str(e)}", flush=True)
            import traceback
            print(traceback.format_exc(), flush=True)
            
            return {
                "score": 0,
                "explanation": f"Error: {str(e)}",
                "status": "FAIL",
                "ground_truth_used": False
            }
    
    def evaluate_dataset(self, eval_data):
        """Evaluate entire dataset."""
        results = []
        total = len(eval_data) * len(self.metrics)
        current = 0
        
        print(f"\n{'='*80}")
        print(f"STARTING EVALUATION")
        print(f"{'='*80}")
        print(f"Samples: {len(eval_data)}")
        print(f"Metrics: {len(self.metrics)}")
        print(f"Total evaluations: {total}")
        print(f"{'='*80}\n")
        
        for idx, row in eval_data.iterrows():
            sample_id = row.get('sample_id', idx)
            prompt = row.get('prompt', '')
            response = row.get('response', '')
            
            print(f"Sample {idx+1}/{len(eval_data)}: ID {sample_id}")
            
            for metric in self.metrics:
                current += 1
                progress = (current / total) * 100
                print(f"  [{progress:5.1f}%] {metric.name}...", flush=True)
                
                result = self.evaluate_single(prompt, response, metric, idx)
                
                print(f"    Result: {result['status']} (score: {result['score']:.2f})", flush=True)
                
                results.append({
                    'sample_id': sample_id,
                    'metric_name': metric.name,
                    'metric_type': metric.metric_type.value,
                    'score': result['score'],
                    'threshold': metric.threshold,
                    'status': result['status'],
                    'explanation': result['explanation'],
                    'ground_truth_used': result['ground_truth_used']
                })
        
        print(f"\n{'='*80}")
        print("EVALUATION DATASET COMPLETE")
        print(f"{'='*80}")
        return pd.DataFrame(results)

print("Evaluator class defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 6: RUN EVALUATION

# COMMAND ----------

import time

print("="*80)
print("CELL 6: RUN EVALUATION")
print("="*80)

# Check prerequisites
print("\nChecking prerequisites...")
print(f"  client is None: {client is None}")
if client is not None:
    print(f"  client_type: {client_type}")
    print(f"  model: {JUDGE_MODEL}")
print(f"  Samples: {len(EVALUATION_DATA)}")
print(f"  Metrics: {len(METRICS_CONFIG_DATA)}")

# Helper function
def safe_float(value, default=0.0):
    """Safely convert to float."""
    if pd.isna(value) or value is None:
        return default
    try:
        val_str = str(value).strip().lower()
        if val_str in ['true', '==true', 'yes', '1']:
            return 1.0
        if val_str in ['false', '==false', 'no', '0']:
            return 0.0
        if val_str.endswith('%'):
            return float(val_str[:-1])
        return float(val_str)
    except:
        return default

def generate_prompt(name, description, rubric):
    """Generate evaluation prompt."""
    return f"""You are an expert evaluator. Task: {description}

Grading Rubric:
{rubric}

Evaluation Details:
- User Query: {{prompt}}
- AI Response: {{response}}
- Ground Truth: {{ground_truth}}

IMPORTANT: Respond with ONLY valid JSON (no other text):
{{
  "score": <your_numeric_score>,
  "explanation": "Brief explanation"
}}"""

# Load metrics
print("\nLoading metrics...")
metric_configs = []

for idx, row in METRICS_CONFIG_DATA.iterrows():
    name = str(row['name']).strip()
    type_str = str(row['type']).strip().lower()
    
    if type_str in ['binary', 'bool']:
        mtype = MetricType.BINARY
    elif type_str in ['1-5_scale', 'scale']:
        mtype = MetricType.SCALE_1_5
    else:
        mtype = MetricType.PERCENTAGE
    
    description = str(row.get('description', '')).strip()
    rubric = str(row.get('grading_rubric', '')).strip()
    threshold = safe_float(row.get('threshold', 0.5))
    
    prompt_template = generate_prompt(name, description, rubric)
    
    metric_config = MetricConfig(
        name=name,
        metric_type=mtype,
        description=description,
        prompt_template=prompt_template,
        threshold=threshold,
        ground_truth_column=str(row.get('ground_truth_column', '')).strip(),
        ground_truth_file_path=str(row.get('ground_truth_file_path', '')).strip()
    )
    
    metric_configs.append(metric_config)
    print(f"  Loaded: {name} ({mtype.value}, threshold={threshold})")

print(f"\nTotal metrics loaded: {len(metric_configs)}")

# Initialize evaluator
if client is None:
    print("\nERROR: Client not initialized!")
else:
    print("\nInitializing evaluator...")
    evaluator = LLMJudgeEvaluator(
        client=client,
        model=JUDGE_MODEL,
        metrics=metric_configs,
        ground_truth_data=GROUND_TRUTH_DATA,
        client_type=client_type
    )
    
    print(f"Evaluator ready")
    print(f"\n{'='*80}")
    print("ABOUT TO START EVALUATION")
    print(f"{'='*80}")
    print("Watch for [LLM CALL START] messages below...")
    print("If you don't see them, the LLM is not being called!")
    print(f"{'='*80}\n")
    
    # Force flush
    import sys
    sys.stdout.flush()
    
    # Run evaluation
    start_time = time.time()
    results_df = evaluator.evaluate_dataset(EVALUATION_DATA)
    eval_time = time.time() - start_time
    
    # Results
    print(f"\n{'='*80}")
    print(f"EVALUATION COMPLETE in {eval_time:.1f}s")
    print(f"{'='*80}")
    
    if eval_time < 5.0:
        print(f"\nWARNING: Evaluation was very fast ({eval_time:.1f}s)")
        print(f"Expected time for {len(results_df)} LLM calls: 20-60 seconds")
        print(f"This suggests LLM calls are failing silently!")
    
    # Analyze results
    total = len(results_df)
    passed = len(results_df[results_df['status'] == 'PASS'])
    failed = len(results_df[results_df['status'] == 'FAIL'])
    pass_rate = (passed / total * 100) if total > 0 else 0
    
    print(f"\nRESULTS SUMMARY:")
    print(f"  Total: {total}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Pass rate: {pass_rate:.1f}%")
    print(f"  Time: {eval_time:.1f}s")
    
    # Check for all zeros
    if results_df['score'].sum() == 0:
        print(f"\nWARNING: All scores are 0!")
        print(f"This means:")
        print(f"  1. LLM calls are failing, OR")
        print(f"  2. LLM responses cannot be parsed, OR")
        print(f"  3. There's an error in evaluate_single")
        print(f"\nCheck the debug output above for [LLM CALL START] messages")
    
    # Show first few results
    print(f"\nFirst 3 results:")
    for i in range(min(3, len(results_df))):
        row = results_df.iloc[i]
        print(f"\n  {i+1}. Sample {row['sample_id']} - {row['metric_name']}:")
        print(f"     Score: {row['score']:.2f}")
        print(f"     Status: {row['status']}")
        print(f"     Explanation: {row['explanation'][:150]}")
    
    # Per-metric
    print(f"\nPer-metric results:")
    for metric_name in results_df['metric_name'].unique():
        metric_results = results_df[results_df['metric_name'] == metric_name]
        m_passed = len(metric_results[metric_results['status'] == 'PASS'])
        m_total = len(metric_results)
        m_rate = (m_passed / m_total * 100) if m_total > 0 else 0
        m_avg = metric_results['score'].mean()
        
        print(f"  {metric_name}: {m_rate:.1f}% pass ({m_passed}/{m_total}), avg score: {m_avg:.2f}")
    
    # Display full results
    print(f"\nDetailed results table:")
    display(results_df[['sample_id', 'metric_name', 'score', 'threshold', 'status', 'explanation']])
    
    print(f"\n{'='*80}")
    print("END OF CELL 6")
    print(f"{'='*80}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Debug Guide
# MAGIC 
# MAGIC **What to look for in Cell 6 output:**
# MAGIC 
# MAGIC 1. **[LLM CALL START]** - Should appear for each evaluation (9 times for 3 samples x 3 metrics)
# MAGIC    - If missing: LLM is NOT being called
# MAGIC 
# MAGIC 2. **API call SUCCESS!** - Should appear after each LLM call
# MAGIC    - If missing: API calls are failing
# MAGIC 
# MAGIC 3. **Evaluation time** - Should be 20-60 seconds for 9 LLM calls
# MAGIC    - If < 5 seconds: LLM calls are failing silently
# MAGIC 
# MAGIC 4. **All scores are 0** warning - Indicates parsing or API issues
# MAGIC 
# MAGIC 5. **ERROR messages** - Will show full traceback of any failures
