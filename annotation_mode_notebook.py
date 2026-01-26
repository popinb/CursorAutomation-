# Databricks notebook source
# MAGIC %md
# MAGIC # Annotation Mode - Human Evaluation UI
# MAGIC 
# MAGIC This notebook provides an interactive annotation interface for human evaluators
# MAGIC to manually score LLM responses using the same 12 metrics as the automated evaluator.
# MAGIC 
# MAGIC ## Features
# MAGIC - Beautiful, modern UI for annotation
# MAGIC - All 12 evaluation metrics with proper scoring types
# MAGIC - Sample navigation and progress tracking
# MAGIC - Annotation persistence (saves to CSV)
# MAGIC - Export functionality

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup & Configuration

# COMMAND ----------

import pandas as pd
import json
import os
from datetime import datetime
from html import escape
from typing import Dict, Any, List, Optional

# ============================================================
# Configuration - Adjust these paths as needed
# ============================================================

# For Databricks
try:
    # Create widgets for configuration
    dbutils.widgets.text("annotator_id", "annotator_1", "Annotator ID")
    dbutils.widgets.text("samples_csv", "", "Samples CSV Path (optional)")
    dbutils.widgets.text("annotations_csv", "annotation_results.csv", "Output Annotations CSV")
    
    ANNOTATOR_ID = dbutils.widgets.get("annotator_id")
    SAMPLES_CSV = dbutils.widgets.get("samples_csv")
    ANNOTATIONS_CSV = dbutils.widgets.get("annotations_csv")
    IN_DATABRICKS = True
except:
    # For Jupyter/local
    ANNOTATOR_ID = "annotator_1"
    SAMPLES_CSV = ""
    ANNOTATIONS_CSV = "annotation_results.csv"
    IN_DATABRICKS = False

print(f"Annotator ID: {ANNOTATOR_ID}")
print(f"Annotations will be saved to: {ANNOTATIONS_CSV}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Metrics Configuration

# COMMAND ----------

# Define the 12 evaluation metrics
METRICS = [
    {
        "name": "personalization_accuracy",
        "display_name": "Personalization Accuracy",
        "metric_type": "binary",
        "options": ["Accurate", "Inaccurate"],
        "description": "Does the response accurately match user-specific figures from the buyability profile?"
    },
    {
        "name": "context_based_personalization",
        "display_name": "Context-Based Personalization",
        "metric_type": "scale_1_5",
        "description": "What percentage of relevant customizations are included? (1=0-20%, 5=80-100%)"
    },
    {
        "name": "next_step_identification",
        "display_name": "Next-Step Identification",
        "metric_type": "binary",
        "options": ["Present", "Not Present"],
        "description": "Does the response provide actionable next steps or guidance?"
    },
    {
        "name": "assumption_listing",
        "display_name": "Assumption Listing",
        "metric_type": "binary",
        "options": ["True", "False"],
        "description": "Does the response explicitly state assumptions and limitations?"
    },
    {
        "name": "assumption_trust",
        "display_name": "Assumption Trust",
        "metric_type": "scale_1_5",
        "description": "How transparently does the response flag ambiguities and gaps? (1=silent, 5=comprehensive)"
    },
    {
        "name": "calculation_accuracy",
        "display_name": "Calculation Accuracy",
        "metric_type": "binary",
        "options": ["True", "False"],
        "description": "Are all mathematical calculations correct?"
    },
    {
        "name": "faithfulness_to_ground_truth",
        "display_name": "Faithfulness to Ground Truth",
        "metric_type": "binary",
        "options": ["True", "False"],
        "description": "Does the response align with established ground truth information?"
    },
    {
        "name": "fair_housing_compliance",
        "display_name": "Fair Housing Compliance",
        "metric_type": "binary",
        "options": ["True", "False"],
        "description": "Does the response comply with fair housing laws and avoid discriminatory language?"
    },
    {
        "name": "overall_accuracy",
        "display_name": "Overall Accuracy",
        "metric_type": "binary",
        "options": ["True", "False"],
        "description": "Does the response correctly and completely answer the question?"
    },
    {
        "name": "structured_presentation",
        "display_name": "Structured Presentation",
        "metric_type": "scale_1_5",
        "description": "How well is the content organized with headings, lists, and formatting? (1=wall of text, 5=excellent)"
    },
    {
        "name": "coherence",
        "display_name": "Coherence",
        "metric_type": "binary",
        "options": ["True", "False"],
        "description": "Is the response logically consistent without contradictions or excessive repetition?"
    },
    {
        "name": "completeness",
        "display_name": "Completeness",
        "metric_type": "scale_1_5",
        "description": "How completely does the response address all aspects of the question? (1=minimal, 5=comprehensive)"
    },
]

print(f"Loaded {len(METRICS)} metrics for evaluation")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sample Data

# COMMAND ----------

# Demo samples - Replace with your actual data
DEMO_SAMPLES = [
    {
        "sample_id": "sample_001",
        "question": "What factors were considered to calculate my Buyability?",
        "candidate_answer": """Your personalized BuyAbility estimate is $318,431, based on your specific financial profile.
        
This calculation uses your $90,000 annual income, $200 monthly debts, $18,000 down payment,
and credit score range of 660-719. With your monthly income of $7,500, lenders recommend
up to 36% DTI, giving you about $2,500 available for mortgage payments.

Your monthly payment breaks down as:
• Principal & Interest: $1,975
• Property Taxes: $212
• Homeowners Insurance: $106
• PMI: $207

You should consider getting pre-approved to start your home search in Georgia.""",
        "user_profile": {
            "annual_income": 90000,
            "monthly_debts": 200,
            "down_payment": 18000,
            "credit_score": "660-719"
        },
        "ground_truth": "Response should explain income, debts, down payment, credit score, and location factors."
    },
    {
        "sample_id": "sample_002",
        "question": "Can I afford to buy a home right now?",
        "candidate_answer": """Yes, based on your current financial situation, you appear to be in a good position to buy a home.

Your BuyAbility is approximately $320,000, which means you could comfortably afford homes in that price range.

Key factors in your favor:
1. Stable income of $90,000/year
2. Low debt-to-income ratio under 36%
3. Adequate down payment of $18,000

I recommend speaking with a mortgage lender to get pre-approved and start your home search.""",
        "user_profile": {
            "annual_income": 90000,
            "monthly_debts": 200,
            "down_payment": 18000,
            "credit_score": "660-719"
        },
        "ground_truth": "Should confirm affordability with specific figures and provide next steps."
    },
    {
        "sample_id": "sample_003",
        "question": "What's included in the monthly payment?",
        "candidate_answer": """Your monthly mortgage payment includes several components:

Principal: The amount that goes toward paying down your loan balance
Interest: The cost of borrowing money
Taxes: Property taxes for your home
Insurance: Homeowners insurance to protect your property

If your down payment is less than 20%, you'll also pay PMI (Private Mortgage Insurance).

These are often referred to as PITI - Principal, Interest, Taxes, and Insurance.""",
        "user_profile": {},
        "ground_truth": "Should detail P&I, taxes, insurance, and PMI with specific amounts if profile available."
    },
]

# Load samples from CSV if provided, otherwise use demo samples
if SAMPLES_CSV and os.path.exists(SAMPLES_CSV):
    samples_df = pd.read_csv(SAMPLES_CSV)
    SAMPLES = samples_df.to_dict('records')
    print(f"Loaded {len(SAMPLES)} samples from {SAMPLES_CSV}")
else:
    SAMPLES = DEMO_SAMPLES
    print(f"Using {len(SAMPLES)} demo samples")

# Display samples summary
for i, sample in enumerate(SAMPLES):
    print(f"  {i+1}. {sample['sample_id']}: {sample['question'][:50]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Annotation Interface

# COMMAND ----------

def generate_annotation_ui(samples: List[Dict], metrics: List[Dict], annotator_id: str) -> str:
    """Generate the annotation UI HTML."""
    
    samples_json = json.dumps(samples)
    metrics_json = json.dumps(metrics)
    
    html = f"""
    <style>
        .annotation-app {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f8fafc;
        }}
        
        .header {{
            background: linear-gradient(135deg, #2563eb, #7c3aed);
            color: white;
            padding: 24px 30px;
            border-radius: 16px;
            margin-bottom: 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3);
        }}
        
        .header h1 {{
            font-size: 28px;
            font-weight: 700;
            margin: 0;
        }}
        
        .header-subtitle {{
            opacity: 0.9;
            font-size: 14px;
            margin-top: 4px;
        }}
        
        .header-info {{
            display: flex;
            gap: 16px;
        }}
        
        .badge {{
            background: rgba(255,255,255,0.2);
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 14px;
            backdrop-filter: blur(10px);
        }}
        
        .main-grid {{
            display: grid;
            grid-template-columns: 1fr 380px;
            gap: 24px;
        }}
        
        .card {{
            background: white;
            border-radius: 16px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            padding: 24px;
            margin-bottom: 20px;
        }}
        
        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 16px;
            border-bottom: 2px solid #f1f5f9;
        }}
        
        .card-title {{
            font-size: 20px;
            font-weight: 700;
            color: #1e293b;
        }}
        
        .sample-nav {{
            display: flex;
            gap: 8px;
        }}
        
        .nav-btn {{
            padding: 8px 16px;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            background: white;
            cursor: pointer;
            font-weight: 600;
            font-size: 14px;
            transition: all 0.2s;
        }}
        
        .nav-btn:hover {{
            border-color: #2563eb;
            color: #2563eb;
        }}
        
        .nav-btn:disabled {{
            opacity: 0.5;
            cursor: not-allowed;
        }}
        
        .sample-box {{
            background: linear-gradient(135deg, #f8fafc, #f1f5f9);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 16px;
        }}
        
        .sample-label {{
            font-size: 11px;
            font-weight: 700;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 10px;
        }}
        
        .question-text {{
            font-size: 18px;
            font-weight: 600;
            color: #2563eb;
            line-height: 1.5;
        }}
        
        .response-text {{
            white-space: pre-wrap;
            font-size: 14px;
            line-height: 1.8;
            color: #334155;
        }}
        
        .profile-box {{
            background: linear-gradient(135deg, #dbeafe, #e0f2fe);
            border-radius: 12px;
            padding: 16px 20px;
            margin-top: 16px;
        }}
        
        .profile-grid {{
            display: flex;
            flex-wrap: wrap;
            gap: 16px;
        }}
        
        .profile-item {{
            font-size: 13px;
        }}
        
        .profile-key {{
            color: #64748b;
        }}
        
        .profile-value {{
            font-weight: 700;
            color: #1e293b;
        }}
        
        .ground-truth-box {{
            background: linear-gradient(135deg, #fef3c7, #fde68a);
            border-radius: 12px;
            padding: 16px 20px;
            margin-top: 16px;
            border-left: 4px solid #f59e0b;
        }}
        
        .metrics-panel {{
            position: sticky;
            top: 20px;
        }}
        
        .metric-card {{
            border: 2px solid #e2e8f0;
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 12px;
            transition: all 0.3s;
        }}
        
        .metric-card:hover {{
            border-color: #2563eb;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.1);
        }}
        
        .metric-card.completed {{
            border-color: #10b981;
            background: linear-gradient(135deg, #f0fdf4, #ecfdf5);
        }}
        
        .metric-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }}
        
        .metric-name {{
            font-weight: 700;
            font-size: 14px;
            color: #1e293b;
        }}
        
        .metric-type {{
            font-size: 10px;
            padding: 3px 8px;
            border-radius: 12px;
            background: #f1f5f9;
            color: #64748b;
            font-weight: 600;
        }}
        
        .metric-desc {{
            font-size: 12px;
            color: #64748b;
            margin-bottom: 12px;
            line-height: 1.5;
        }}
        
        .score-buttons {{
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }}
        
        .score-btn {{
            padding: 8px 16px;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            background: white;
            cursor: pointer;
            font-size: 13px;
            font-weight: 600;
            transition: all 0.2s;
        }}
        
        .score-btn:hover {{
            border-color: #2563eb;
            background: #f8fafc;
        }}
        
        .score-btn.selected {{
            border-color: #2563eb;
            background: #2563eb;
            color: white;
        }}
        
        .score-btn.positive.selected {{
            border-color: #10b981;
            background: #10b981;
        }}
        
        .score-btn.negative.selected {{
            border-color: #ef4444;
            background: #ef4444;
        }}
        
        .scale-container {{
            width: 100%;
        }}
        
        .scale-display {{
            font-size: 32px;
            font-weight: 800;
            color: #2563eb;
            text-align: center;
            margin-bottom: 8px;
        }}
        
        .scale-slider {{
            width: 100%;
            height: 8px;
            border-radius: 4px;
            background: #e2e8f0;
            outline: none;
            -webkit-appearance: none;
        }}
        
        .scale-slider::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 24px;
            height: 24px;
            border-radius: 50%;
            background: #2563eb;
            cursor: pointer;
            box-shadow: 0 2px 8px rgba(37, 99, 235, 0.4);
        }}
        
        .scale-labels {{
            display: flex;
            justify-content: space-between;
            font-size: 11px;
            color: #94a3b8;
            margin-top: 6px;
        }}
        
        .notes-area {{
            margin-top: 20px;
        }}
        
        .notes-input {{
            width: 100%;
            padding: 12px 16px;
            border: 2px solid #e2e8f0;
            border-radius: 12px;
            font-size: 14px;
            resize: vertical;
            min-height: 80px;
            font-family: inherit;
        }}
        
        .notes-input:focus {{
            outline: none;
            border-color: #2563eb;
        }}
        
        .submit-section {{
            display: flex;
            justify-content: flex-end;
            margin-top: 24px;
            padding-top: 20px;
            border-top: 2px solid #f1f5f9;
        }}
        
        .submit-btn {{
            padding: 14px 32px;
            background: linear-gradient(135deg, #10b981, #059669);
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 16px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s;
            box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
        }}
        
        .submit-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4);
        }}
        
        .progress-bar {{
            width: 100%;
            height: 6px;
            background: #e2e8f0;
            border-radius: 3px;
            margin-top: 12px;
            overflow: hidden;
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #10b981, #059669);
            border-radius: 3px;
            transition: width 0.5s ease;
        }}
        
        .toast {{
            position: fixed;
            bottom: 24px;
            right: 24px;
            padding: 16px 24px;
            border-radius: 12px;
            color: white;
            font-weight: 600;
            z-index: 1000;
            animation: slideUp 0.4s ease;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        }}
        
        .toast.success {{
            background: linear-gradient(135deg, #10b981, #059669);
        }}
        
        .toast.error {{
            background: linear-gradient(135deg, #ef4444, #dc2626);
        }}
        
        @keyframes slideUp {{
            from {{
                transform: translateY(100px);
                opacity: 0;
            }}
            to {{
                transform: translateY(0);
                opacity: 1;
            }}
        }}
        
        .results-panel {{
            margin-top: 24px;
            background: white;
            border-radius: 16px;
            padding: 24px;
            display: none;
        }}
        
        .results-table {{
            width: 100%;
            border-collapse: collapse;
        }}
        
        .results-table th, .results-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }}
        
        .results-table th {{
            background: #f8fafc;
            font-weight: 700;
            color: #64748b;
            font-size: 12px;
            text-transform: uppercase;
        }}
        
        @media (max-width: 1200px) {{
            .main-grid {{
                grid-template-columns: 1fr;
            }}
            .metrics-panel {{
                position: static;
            }}
        }}
    </style>
    
    <div class="annotation-app">
        <div class="header">
            <div>
                <h1>Annotation Mode</h1>
                <div class="header-subtitle">Human Evaluation Interface</div>
            </div>
            <div class="header-info">
                <div class="badge">Annotator: <strong>{annotator_id}</strong></div>
                <div class="badge">Progress: <strong id="progressText">0/{len(samples)}</strong></div>
            </div>
        </div>
        
        <div class="main-grid">
            <div class="content-section">
                <div class="card">
                    <div class="card-header">
                        <div class="card-title">Sample <span id="sampleId">-</span></div>
                        <div class="sample-nav">
                            <button class="nav-btn" onclick="navigate(-1)" id="prevBtn">← Previous</button>
                            <button class="nav-btn" onclick="navigate(1)" id="nextBtn">Next →</button>
                        </div>
                    </div>
                    
                    <div class="sample-box">
                        <div class="sample-label">Question</div>
                        <div class="question-text" id="questionText">-</div>
                    </div>
                    
                    <div class="sample-box">
                        <div class="sample-label">Candidate Response</div>
                        <div class="response-text" id="responseText">-</div>
                    </div>
                    
                    <div class="profile-box" id="profileBox" style="display: none;">
                        <div class="sample-label">User Profile</div>
                        <div class="profile-grid" id="profileContent"></div>
                    </div>
                    
                    <div class="ground-truth-box" id="gtBox" style="display: none;">
                        <div class="sample-label">Ground Truth Reference</div>
                        <div id="gtText" style="font-size: 14px;"></div>
                    </div>
                    
                    <div class="notes-area">
                        <div class="sample-label">Annotator Notes (optional)</div>
                        <textarea class="notes-input" id="notesInput" placeholder="Add any notes about this annotation..."></textarea>
                    </div>
                    
                    <div class="submit-section">
                        <button class="submit-btn" onclick="submitAnnotation()">Submit Annotation ✓</button>
                    </div>
                </div>
                
                <div class="results-panel" id="resultsPanel">
                    <div class="card-title" style="margin-bottom: 16px;">Saved Annotations</div>
                    <table class="results-table">
                        <thead>
                            <tr>
                                <th>Sample ID</th>
                                <th>Timestamp</th>
                                <th>Metrics Scored</th>
                            </tr>
                        </thead>
                        <tbody id="resultsBody"></tbody>
                    </table>
                </div>
            </div>
            
            <div class="metrics-panel">
                <div class="card">
                    <div class="card-header">
                        <div class="card-title">Metrics</div>
                        <span id="metricsProgress" style="font-size: 14px; color: #64748b;">0/12</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" id="progressFill" style="width: 0%;"></div>
                    </div>
                    <div id="metricsContainer" style="margin-top: 16px;"></div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const samples = {samples_json};
        const metrics = {metrics_json};
        const annotatorId = '{annotator_id}';
        
        let currentIndex = 0;
        let scores = {{}};
        let allAnnotations = [];
        
        // Initialize
        document.addEventListener('DOMContentLoaded', init);
        init();
        
        function init() {{
            renderMetrics();
            if (samples.length > 0) {{
                displaySample(0);
            }}
        }}
        
        function renderMetrics() {{
            const container = document.getElementById('metricsContainer');
            container.innerHTML = metrics.map((m, i) => {{
                let scoreHtml = '';
                
                if (m.metric_type === 'binary') {{
                    const opts = m.options || ['True', 'False'];
                    scoreHtml = `
                        <div class="score-buttons">
                            ${{opts.map(opt => `
                                <button class="score-btn ${{opt.includes('True') || opt === 'Accurate' || opt === 'Present' ? 'positive' : 'negative'}}" 
                                        data-metric="${{m.name}}" data-value="${{opt}}" 
                                        onclick="setScore('${{m.name}}', '${{opt}}', this)">
                                    ${{opt}}
                                </button>
                            `).join('')}}
                        </div>
                    `;
                }} else if (m.metric_type === 'scale_1_5') {{
                    scoreHtml = `
                        <div class="scale-container">
                            <div class="scale-display" id="display-${{m.name}}">-</div>
                            <input type="range" class="scale-slider" min="1" max="5" value="3" 
                                   id="slider-${{m.name}}" 
                                   oninput="setSlider('${{m.name}}', this.value)">
                            <div class="scale-labels">
                                <span>1 (Poor)</span>
                                <span>3 (Average)</span>
                                <span>5 (Excellent)</span>
                            </div>
                        </div>
                    `;
                }}
                
                return `
                    <div class="metric-card" id="metric-${{m.name}}">
                        <div class="metric-header">
                            <span class="metric-name">${{m.display_name}}</span>
                            <span class="metric-type">${{m.metric_type === 'scale_1_5' ? '1-5 Scale' : 'Binary'}}</span>
                        </div>
                        <div class="metric-desc">${{m.description}}</div>
                        ${{scoreHtml}}
                    </div>
                `;
            }}).join('');
        }}
        
        function displaySample(index) {{
            if (index < 0 || index >= samples.length) return;
            
            currentIndex = index;
            scores = {{}};
            const sample = samples[index];
            
            document.getElementById('sampleId').textContent = sample.sample_id;
            document.getElementById('questionText').textContent = sample.question;
            document.getElementById('responseText').textContent = sample.candidate_answer;
            document.getElementById('notesInput').value = '';
            
            // Profile
            const profileBox = document.getElementById('profileBox');
            const profileContent = document.getElementById('profileContent');
            if (sample.user_profile && Object.keys(sample.user_profile).length > 0) {{
                profileBox.style.display = 'block';
                profileContent.innerHTML = Object.entries(sample.user_profile)
                    .map(([k, v]) => `<div class="profile-item"><span class="profile-key">${{k}}:</span> <span class="profile-value">${{v}}</span></div>`)
                    .join('');
            }} else {{
                profileBox.style.display = 'none';
            }}
            
            // Ground truth
            const gtBox = document.getElementById('gtBox');
            if (sample.ground_truth) {{
                gtBox.style.display = 'block';
                document.getElementById('gtText').textContent = sample.ground_truth;
            }} else {{
                gtBox.style.display = 'none';
            }}
            
            // Reset UI
            document.querySelectorAll('.metric-card').forEach(c => c.classList.remove('completed'));
            document.querySelectorAll('.score-btn').forEach(b => b.classList.remove('selected'));
            document.querySelectorAll('.scale-display').forEach(d => d.textContent = '-');
            
            updateProgress();
            updateNavButtons();
        }}
        
        function setScore(metricName, value, btn) {{
            scores[metricName] = value;
            
            // Update UI
            document.querySelectorAll(`[data-metric="${{metricName}}"]`).forEach(b => b.classList.remove('selected'));
            btn.classList.add('selected');
            document.getElementById(`metric-${{metricName}}`).classList.add('completed');
            
            updateMetricsProgress();
        }}
        
        function setSlider(metricName, value) {{
            scores[metricName] = parseInt(value);
            document.getElementById(`display-${{metricName}}`).textContent = value;
            document.getElementById(`metric-${{metricName}}`).classList.add('completed');
            updateMetricsProgress();
        }}
        
        function updateMetricsProgress() {{
            const completed = Object.keys(scores).length;
            document.getElementById('metricsProgress').textContent = `${{completed}}/${{metrics.length}}`;
            document.getElementById('progressFill').style.width = `${{(completed / metrics.length) * 100}}%`;
        }}
        
        function updateProgress() {{
            const annotated = allAnnotations.filter(a => samples.some(s => s.sample_id === a.sample_id)).length;
            document.getElementById('progressText').textContent = `${{annotated}}/${{samples.length}}`;
        }}
        
        function updateNavButtons() {{
            document.getElementById('prevBtn').disabled = currentIndex === 0;
            document.getElementById('nextBtn').disabled = currentIndex === samples.length - 1;
        }}
        
        function navigate(dir) {{
            displaySample(currentIndex + dir);
        }}
        
        function submitAnnotation() {{
            if (Object.keys(scores).length < metrics.length) {{
                if (!confirm('Not all metrics scored. Submit anyway?')) return;
            }}
            
            const sample = samples[currentIndex];
            const annotation = {{
                sample_id: sample.sample_id,
                annotator_id: annotatorId,
                timestamp: new Date().toISOString(),
                notes: document.getElementById('notesInput').value,
                ...scores
            }};
            
            allAnnotations.push(annotation);
            showToast('Annotation saved!', 'success');
            updateResults();
            updateProgress();
            
            // Move to next
            if (currentIndex < samples.length - 1) {{
                setTimeout(() => navigate(1), 500);
            }}
        }}
        
        function updateResults() {{
            const panel = document.getElementById('resultsPanel');
            const tbody = document.getElementById('resultsBody');
            
            panel.style.display = 'block';
            tbody.innerHTML = allAnnotations.map(a => `
                <tr>
                    <td>${{a.sample_id}}</td>
                    <td>${{new Date(a.timestamp).toLocaleString()}}</td>
                    <td>${{Object.keys(a).filter(k => !['sample_id', 'annotator_id', 'timestamp', 'notes'].includes(k)).length}}</td>
                </tr>
            `).join('');
        }}
        
        function showToast(msg, type) {{
            const toast = document.createElement('div');
            toast.className = `toast ${{type}}`;
            toast.textContent = msg;
            document.body.appendChild(toast);
            setTimeout(() => toast.remove(), 3000);
        }}
        
        // Export function (can be called from Python)
        function getAnnotationsJSON() {{
            return JSON.stringify(allAnnotations, null, 2);
        }}
    </script>
    """
    
    return html

# Generate and display the UI
annotation_html = generate_annotation_ui(SAMPLES, METRICS, ANNOTATOR_ID)

# Display in Databricks
if IN_DATABRICKS:
    displayHTML(annotation_html)
else:
    # For Jupyter
    from IPython.display import HTML, display
    display(HTML(annotation_html))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Export Annotations
# MAGIC 
# MAGIC After completing annotations in the UI above, run this cell to export results.
# MAGIC You can also manually enter annotations below.

# COMMAND ----------

# ============================================================
# Manual Annotation Export Helper
# ============================================================

# Paste your annotations JSON here (copy from browser console: getAnnotationsJSON())
ANNOTATIONS_JSON = """
[]
"""

def save_annotations(json_str: str, output_path: str) -> None:
    """Save annotations from JSON string to CSV."""
    try:
        annotations = json.loads(json_str)
        if annotations:
            df = pd.DataFrame(annotations)
            df.to_csv(output_path, index=False)
            print(f"Saved {len(annotations)} annotations to {output_path}")
            display(df)
        else:
            print("No annotations to save.")
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
    except Exception as e:
        print(f"Error saving: {e}")

# Uncomment to save:
# save_annotations(ANNOTATIONS_JSON, ANNOTATIONS_CSV)

print("""
============================================================
EXPORT INSTRUCTIONS
============================================================
1. Complete your annotations in the UI above
2. Open browser developer console (F12)
3. Type: getAnnotationsJSON()
4. Copy the JSON output
5. Paste it into ANNOTATIONS_JSON variable above
6. Uncomment and run save_annotations()
============================================================
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Alternative: Widget-Based Annotation
# MAGIC 
# MAGIC If the HTML interface doesn't work well, use this simpler widget-based approach.

# COMMAND ----------

def create_widget_annotation_ui(samples: List[Dict], metrics: List[Dict]):
    """Create a simpler widget-based annotation interface."""
    
    if not IN_DATABRICKS:
        print("Widget-based UI only available in Databricks. Use the HTML interface above.")
        return
    
    # Sample selector
    sample_options = [f"{s['sample_id']}: {s['question'][:40]}..." for s in samples]
    dbutils.widgets.dropdown("current_sample", sample_options[0], sample_options, "Select Sample")
    
    # Create widgets for each metric
    for m in metrics:
        if m['metric_type'] == 'binary':
            opts = m.get('options', ['True', 'False'])
            dbutils.widgets.dropdown(f"score_{m['name']}", opts[0], opts, m['display_name'])
        elif m['metric_type'] == 'scale_1_5':
            dbutils.widgets.dropdown(f"score_{m['name']}", "3", ["1", "2", "3", "4", "5"], m['display_name'])
    
    dbutils.widgets.text("annotation_notes", "", "Notes")
    
    print("Widget-based annotation UI created!")
    print("1. Select a sample from the dropdown")
    print("2. Score each metric using the dropdowns")
    print("3. Run the next cell to save your annotation")

# Uncomment to create widgets:
# create_widget_annotation_ui(SAMPLES, METRICS)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC 
# MAGIC This notebook provides a complete human annotation interface for evaluating LLM responses.
# MAGIC 
# MAGIC **Features:**
# MAGIC - Interactive HTML-based annotation UI
# MAGIC - All 12 evaluation metrics (binary and scale)
# MAGIC - Sample navigation and progress tracking
# MAGIC - Export to CSV/JSON
# MAGIC 
# MAGIC **Usage:**
# MAGIC 1. Configure samples (use demo or load your own CSV)
# MAGIC 2. Run the annotation UI cell
# MAGIC 3. Score each sample on all metrics
# MAGIC 4. Export your annotations
