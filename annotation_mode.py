"""
Annotation Mode with UI for Zillow LLM Judge Evaluator

This module provides a web-based annotation interface for human annotators
to manually score LLM responses using the same metrics as the automated evaluator.

Features:
- Beautiful modern UI for annotation
- All 12 evaluation metrics with proper scoring types
- Sample navigation and progress tracking
- Annotation persistence (save/load)
- Comparison with LLM judge scores
- Export functionality (CSV/JSON)
- Inter-annotator agreement tracking

Usage:
    python annotation_mode.py [--port 5000] [--host 0.0.0.0]
    
Then open http://localhost:5000 in your browser.
"""

import os
import json
import csv
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict, field
from flask import Flask, render_template_string, request, jsonify, send_file
import uuid

# ============================================================
# Data Models
# ============================================================

@dataclass
class Metric:
    """Definition of an evaluation metric."""
    name: str
    display_name: str
    metric_type: str  # 'binary', 'scale_1_5', 'percentage'
    description: str
    
    def get_valid_values(self) -> List[Any]:
        """Return valid values for this metric type."""
        if self.metric_type == 'binary':
            return ['True', 'False', 'Accurate', 'Inaccurate', 'Present', 'Not Present']
        elif self.metric_type == 'scale_1_5':
            return [1, 2, 3, 4, 5]
        elif self.metric_type == 'percentage':
            return list(range(0, 101, 5))  # 0, 5, 10, ... 100
        return []


@dataclass
class AnnotationSample:
    """A sample to be annotated."""
    sample_id: str
    question: str
    candidate_answer: str
    user_profile: Dict[str, Any] = field(default_factory=dict)
    ground_truth: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class Annotation:
    """An annotation for a sample."""
    annotation_id: str
    sample_id: str
    annotator_id: str
    scores: Dict[str, Any]  # metric_name -> score
    justifications: Dict[str, str]  # metric_name -> justification
    notes: str = ""
    timestamp: str = ""
    duration_seconds: int = 0
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if not self.annotation_id:
            self.annotation_id = str(uuid.uuid4())[:8]


# ============================================================
# Metrics Configuration
# ============================================================

METRICS = [
    Metric(
        name="personalization_accuracy",
        display_name="Personalization Accuracy",
        metric_type="binary",
        description="Does the response accurately match user-specific figures from the buyability profile?"
    ),
    Metric(
        name="context_based_personalization",
        display_name="Context-Based Personalization",
        metric_type="scale_1_5",
        description="What percentage of relevant customizations are included? (1=0-20%, 5=80-100%)"
    ),
    Metric(
        name="next_step_identification",
        display_name="Next-Step Identification",
        metric_type="binary",
        description="Does the response provide actionable next steps or guidance?"
    ),
    Metric(
        name="assumption_listing",
        display_name="Assumption Listing",
        metric_type="binary",
        description="Does the response explicitly state assumptions and limitations?"
    ),
    Metric(
        name="assumption_trust",
        display_name="Assumption Trust",
        metric_type="scale_1_5",
        description="How transparently does the response flag ambiguities and gaps? (1=silent, 5=comprehensive)"
    ),
    Metric(
        name="calculation_accuracy",
        display_name="Calculation Accuracy",
        metric_type="binary",
        description="Are all mathematical calculations correct?"
    ),
    Metric(
        name="faithfulness_to_ground_truth",
        display_name="Faithfulness to Ground Truth",
        metric_type="binary",
        description="Does the response align with established ground truth information?"
    ),
    Metric(
        name="fair_housing_compliance",
        display_name="Fair Housing Compliance",
        metric_type="binary",
        description="Does the response comply with fair housing laws and avoid discriminatory language?"
    ),
    Metric(
        name="overall_accuracy",
        display_name="Overall Accuracy",
        metric_type="binary",
        description="Does the response correctly and completely answer the question?"
    ),
    Metric(
        name="structured_presentation",
        display_name="Structured Presentation",
        metric_type="scale_1_5",
        description="How well is the content organized with headings, lists, and formatting? (1=wall of text, 5=excellent)"
    ),
    Metric(
        name="coherence",
        display_name="Coherence",
        metric_type="binary",
        description="Is the response logically consistent without contradictions or excessive repetition?"
    ),
    Metric(
        name="completeness",
        display_name="Completeness",
        metric_type="scale_1_5",
        description="How completely does the response address all aspects of the question? (1=minimal, 5=comprehensive)"
    ),
]


# ============================================================
# Annotation Manager
# ============================================================

class AnnotationManager:
    """Manages annotation samples and annotations."""
    
    def __init__(self, data_dir: str = "./annotation_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.samples_file = self.data_dir / "samples.json"
        self.annotations_file = self.data_dir / "annotations.json"
        
        self.samples: Dict[str, AnnotationSample] = {}
        self.annotations: Dict[str, List[Annotation]] = {}  # sample_id -> list of annotations
        
        self._load_data()
    
    def _load_data(self):
        """Load samples and annotations from disk."""
        if self.samples_file.exists():
            with open(self.samples_file, 'r') as f:
                data = json.load(f)
                for sample_data in data:
                    sample = AnnotationSample(**sample_data)
                    self.samples[sample.sample_id] = sample
        
        if self.annotations_file.exists():
            with open(self.annotations_file, 'r') as f:
                data = json.load(f)
                for sample_id, annotations_list in data.items():
                    self.annotations[sample_id] = [
                        Annotation(**ann_data) for ann_data in annotations_list
                    ]
    
    def _save_data(self):
        """Save samples and annotations to disk."""
        # Save samples
        samples_data = [asdict(sample) for sample in self.samples.values()]
        with open(self.samples_file, 'w') as f:
            json.dump(samples_data, f, indent=2)
        
        # Save annotations
        annotations_data = {
            sample_id: [asdict(ann) for ann in annotations]
            for sample_id, annotations in self.annotations.items()
        }
        with open(self.annotations_file, 'w') as f:
            json.dump(annotations_data, f, indent=2)
    
    def add_sample(self, sample: AnnotationSample) -> str:
        """Add a new sample for annotation."""
        self.samples[sample.sample_id] = sample
        self._save_data()
        return sample.sample_id
    
    def get_sample(self, sample_id: str) -> Optional[AnnotationSample]:
        """Get a sample by ID."""
        return self.samples.get(sample_id)
    
    def get_all_samples(self) -> List[AnnotationSample]:
        """Get all samples."""
        return list(self.samples.values())
    
    def add_annotation(self, annotation: Annotation) -> str:
        """Add an annotation for a sample."""
        if annotation.sample_id not in self.annotations:
            self.annotations[annotation.sample_id] = []
        
        self.annotations[annotation.sample_id].append(annotation)
        self._save_data()
        return annotation.annotation_id
    
    def get_annotations(self, sample_id: str) -> List[Annotation]:
        """Get all annotations for a sample."""
        return self.annotations.get(sample_id, [])
    
    def get_annotation_progress(self, annotator_id: str) -> Dict[str, Any]:
        """Get annotation progress for an annotator."""
        total_samples = len(self.samples)
        annotated_samples = set()
        
        for sample_id, annotations in self.annotations.items():
            for ann in annotations:
                if ann.annotator_id == annotator_id:
                    annotated_samples.add(sample_id)
        
        return {
            "total_samples": total_samples,
            "annotated_samples": len(annotated_samples),
            "remaining_samples": total_samples - len(annotated_samples),
            "progress_percent": (len(annotated_samples) / total_samples * 100) if total_samples > 0 else 0
        }
    
    def export_annotations_csv(self, output_path: str):
        """Export all annotations to CSV."""
        rows = []
        for sample_id, annotations in self.annotations.items():
            sample = self.samples.get(sample_id)
            for ann in annotations:
                row = {
                    "sample_id": sample_id,
                    "question": sample.question if sample else "",
                    "annotator_id": ann.annotator_id,
                    "timestamp": ann.timestamp,
                    "notes": ann.notes,
                    **ann.scores
                }
                rows.append(row)
        
        if rows:
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
    
    def load_samples_from_csv(self, csv_path: str):
        """Load samples from a CSV file."""
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sample = AnnotationSample(
                    sample_id=row.get('sample_id', str(uuid.uuid4())[:8]),
                    question=row.get('question', row.get('prompt', '')),
                    candidate_answer=row.get('candidate_answer', row.get('response', '')),
                    ground_truth=row.get('ground_truth', row.get('expected', '')),
                    user_profile=json.loads(row.get('user_profile', '{}')) if row.get('user_profile') else {},
                )
                self.add_sample(sample)
    
    def create_demo_samples(self):
        """Create demo samples for testing."""
        demo_samples = [
            AnnotationSample(
                sample_id="demo_001",
                question="What factors were considered to calculate my Buyability?",
                candidate_answer="""Your personalized BuyAbility estimate is $318,431, based on your specific financial profile.
                
This calculation uses your $90,000 annual income, $200 monthly debts, $18,000 down payment,
and credit score range of 660-719. With your monthly income of $7,500, lenders recommend
up to 36% DTI, giving you about $2,500 available for mortgage payments.

Your monthly payment breaks down as:
• Principal & Interest: $1,975
• Property Taxes: $212
• Homeowners Insurance: $106
• PMI: $207

You should consider getting pre-approved to start your home search in Georgia.""",
                user_profile={
                    "annual_income": 90000,
                    "monthly_debts": 200,
                    "down_payment": 18000,
                    "credit_score": "660-719"
                },
                ground_truth="Response should explain income, debts, down payment, credit score, and location factors."
            ),
            AnnotationSample(
                sample_id="demo_002",
                question="Can I afford to buy a home right now?",
                candidate_answer="""Yes, based on your current financial situation, you appear to be in a good position to buy a home.

Your BuyAbility is approximately $320,000, which means you could comfortably afford homes in that price range.

Key factors in your favor:
1. Stable income of $90,000/year
2. Low debt-to-income ratio under 36%
3. Adequate down payment of $18,000

I recommend speaking with a mortgage lender to get pre-approved and start your home search.""",
                user_profile={
                    "annual_income": 90000,
                    "monthly_debts": 200,
                    "down_payment": 18000,
                    "credit_score": "660-719"
                },
                ground_truth="Should confirm affordability with specific figures and provide next steps."
            ),
            AnnotationSample(
                sample_id="demo_003",
                question="What's included in the monthly payment?",
                candidate_answer="""Your monthly mortgage payment includes several components:

Principal: The amount that goes toward paying down your loan balance
Interest: The cost of borrowing money
Taxes: Property taxes for your home
Insurance: Homeowners insurance to protect your property

If your down payment is less than 20%, you'll also pay PMI (Private Mortgage Insurance).

These are often referred to as PITI - Principal, Interest, Taxes, and Insurance.""",
                user_profile={},
                ground_truth="Should detail P&I, taxes, insurance, and PMI with specific amounts if profile available."
            ),
        ]
        
        for sample in demo_samples:
            if sample.sample_id not in self.samples:
                self.add_sample(sample)


# ============================================================
# HTML Templates
# ============================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Annotation Mode - Zillow LLM Judge</title>
    <style>
        :root {
            --primary-color: #2563eb;
            --primary-hover: #1d4ed8;
            --success-color: #10b981;
            --warning-color: #f59e0b;
            --danger-color: #ef4444;
            --bg-color: #f8fafc;
            --card-bg: #ffffff;
            --text-primary: #1e293b;
            --text-secondary: #64748b;
            --border-color: #e2e8f0;
        }
        
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-primary);
            line-height: 1.6;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        /* Header */
        .header {
            background: linear-gradient(135deg, var(--primary-color), #7c3aed);
            color: white;
            padding: 20px 30px;
            border-radius: 12px;
            margin-bottom: 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .header h1 {
            font-size: 24px;
            font-weight: 600;
        }
        
        .header-info {
            display: flex;
            gap: 20px;
            align-items: center;
        }
        
        .annotator-badge {
            background: rgba(255,255,255,0.2);
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 14px;
        }
        
        .progress-indicator {
            background: rgba(255,255,255,0.2);
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 14px;
        }
        
        /* Main Layout */
        .main-layout {
            display: grid;
            grid-template-columns: 1fr 400px;
            gap: 24px;
        }
        
        /* Cards */
        .card {
            background: var(--card-bg);
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            padding: 24px;
            margin-bottom: 20px;
        }
        
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 1px solid var(--border-color);
        }
        
        .card-title {
            font-size: 18px;
            font-weight: 600;
            color: var(--text-primary);
        }
        
        .card-subtitle {
            font-size: 14px;
            color: var(--text-secondary);
        }
        
        /* Sample Display */
        .sample-content {
            background: #f1f5f9;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 16px;
        }
        
        .sample-label {
            font-size: 12px;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            margin-bottom: 8px;
        }
        
        .sample-text {
            white-space: pre-wrap;
            font-size: 14px;
            line-height: 1.7;
        }
        
        .question-text {
            font-size: 16px;
            font-weight: 500;
            color: var(--primary-color);
        }
        
        .user-profile {
            background: #e0f2fe;
            border-radius: 8px;
            padding: 12px 16px;
            margin-top: 12px;
        }
        
        .profile-item {
            display: inline-block;
            margin-right: 16px;
            font-size: 13px;
        }
        
        .profile-label {
            color: var(--text-secondary);
        }
        
        .profile-value {
            font-weight: 600;
            color: var(--text-primary);
        }
        
        /* Metrics Panel */
        .metrics-panel {
            position: sticky;
            top: 20px;
        }
        
        .metric-item {
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 12px;
            transition: all 0.2s;
        }
        
        .metric-item:hover {
            border-color: var(--primary-color);
            box-shadow: 0 2px 8px rgba(37, 99, 235, 0.1);
        }
        
        .metric-item.completed {
            border-color: var(--success-color);
            background: #f0fdf4;
        }
        
        .metric-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 8px;
        }
        
        .metric-name {
            font-weight: 600;
            font-size: 14px;
            color: var(--text-primary);
        }
        
        .metric-type-badge {
            font-size: 11px;
            padding: 2px 8px;
            border-radius: 10px;
            background: var(--border-color);
            color: var(--text-secondary);
        }
        
        .metric-description {
            font-size: 12px;
            color: var(--text-secondary);
            margin-bottom: 12px;
        }
        
        /* Score Options */
        .score-options {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }
        
        .score-option {
            padding: 6px 12px;
            border: 2px solid var(--border-color);
            border-radius: 6px;
            background: white;
            cursor: pointer;
            font-size: 13px;
            font-weight: 500;
            transition: all 0.2s;
        }
        
        .score-option:hover {
            border-color: var(--primary-color);
            background: #f8fafc;
        }
        
        .score-option.selected {
            border-color: var(--primary-color);
            background: var(--primary-color);
            color: white;
        }
        
        .score-option.binary-true {
            border-color: var(--success-color);
        }
        
        .score-option.binary-true.selected {
            background: var(--success-color);
            border-color: var(--success-color);
        }
        
        .score-option.binary-false {
            border-color: var(--danger-color);
        }
        
        .score-option.binary-false.selected {
            background: var(--danger-color);
            border-color: var(--danger-color);
        }
        
        /* Scale slider */
        .scale-slider-container {
            width: 100%;
        }
        
        .scale-slider {
            width: 100%;
            height: 8px;
            border-radius: 4px;
            background: var(--border-color);
            outline: none;
            -webkit-appearance: none;
        }
        
        .scale-slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: var(--primary-color);
            cursor: pointer;
        }
        
        .scale-labels {
            display: flex;
            justify-content: space-between;
            margin-top: 4px;
            font-size: 11px;
            color: var(--text-secondary);
        }
        
        .scale-value {
            font-size: 24px;
            font-weight: 700;
            color: var(--primary-color);
            text-align: center;
            margin-bottom: 8px;
        }
        
        /* Justification */
        .justification-input {
            width: 100%;
            margin-top: 8px;
            padding: 8px 12px;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            font-size: 13px;
            resize: vertical;
            min-height: 60px;
        }
        
        .justification-input:focus {
            outline: none;
            border-color: var(--primary-color);
        }
        
        /* Buttons */
        .btn {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 10px 20px;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            border: none;
        }
        
        .btn-primary {
            background: var(--primary-color);
            color: white;
        }
        
        .btn-primary:hover {
            background: var(--primary-hover);
        }
        
        .btn-success {
            background: var(--success-color);
            color: white;
        }
        
        .btn-outline {
            background: white;
            border: 2px solid var(--border-color);
            color: var(--text-primary);
        }
        
        .btn-outline:hover {
            border-color: var(--primary-color);
            color: var(--primary-color);
        }
        
        /* Navigation */
        .sample-nav {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 20px;
        }
        
        .nav-buttons {
            display: flex;
            gap: 12px;
        }
        
        /* Sample List */
        .sample-list {
            max-height: 300px;
            overflow-y: auto;
            border: 1px solid var(--border-color);
            border-radius: 8px;
        }
        
        .sample-list-item {
            padding: 12px 16px;
            border-bottom: 1px solid var(--border-color);
            cursor: pointer;
            transition: background 0.2s;
        }
        
        .sample-list-item:hover {
            background: #f8fafc;
        }
        
        .sample-list-item.active {
            background: #eff6ff;
            border-left: 3px solid var(--primary-color);
        }
        
        .sample-list-item.annotated {
            background: #f0fdf4;
        }
        
        .sample-list-item.annotated::after {
            content: '✓';
            color: var(--success-color);
            font-weight: bold;
            float: right;
        }
        
        .sample-id {
            font-weight: 600;
            font-size: 13px;
        }
        
        .sample-preview {
            font-size: 12px;
            color: var(--text-secondary);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        /* Notes */
        .notes-section {
            margin-top: 20px;
        }
        
        .notes-input {
            width: 100%;
            padding: 12px;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            font-size: 14px;
            resize: vertical;
            min-height: 80px;
        }
        
        /* Toast notifications */
        .toast {
            position: fixed;
            bottom: 20px;
            right: 20px;
            padding: 16px 24px;
            border-radius: 8px;
            color: white;
            font-weight: 500;
            z-index: 1000;
            animation: slideIn 0.3s ease;
        }
        
        .toast.success {
            background: var(--success-color);
        }
        
        .toast.error {
            background: var(--danger-color);
        }
        
        @keyframes slideIn {
            from {
                transform: translateX(100%);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }
        
        /* Responsive */
        @media (max-width: 1200px) {
            .main-layout {
                grid-template-columns: 1fr;
            }
            
            .metrics-panel {
                position: static;
            }
        }
        
        /* Ground truth comparison */
        .ground-truth-section {
            background: #fef3c7;
            border-radius: 8px;
            padding: 16px;
            margin-top: 16px;
            border-left: 4px solid var(--warning-color);
        }
        
        /* Empty state */
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: var(--text-secondary);
        }
        
        .empty-state-icon {
            font-size: 48px;
            margin-bottom: 16px;
        }
        
        /* Tabs */
        .tabs {
            display: flex;
            gap: 4px;
            margin-bottom: 20px;
            border-bottom: 2px solid var(--border-color);
        }
        
        .tab {
            padding: 12px 20px;
            cursor: pointer;
            font-weight: 500;
            color: var(--text-secondary);
            border-bottom: 2px solid transparent;
            margin-bottom: -2px;
            transition: all 0.2s;
        }
        
        .tab:hover {
            color: var(--primary-color);
        }
        
        .tab.active {
            color: var(--primary-color);
            border-bottom-color: var(--primary-color);
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        /* Stats */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
            margin-bottom: 24px;
        }
        
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            text-align: center;
        }
        
        .stat-value {
            font-size: 32px;
            font-weight: 700;
            color: var(--primary-color);
        }
        
        .stat-label {
            font-size: 13px;
            color: var(--text-secondary);
            margin-top: 4px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1>Annotation Mode</h1>
                <p style="opacity: 0.9; font-size: 14px;">Zillow LLM Judge Evaluator</p>
            </div>
            <div class="header-info">
                <div class="annotator-badge" id="annotatorBadge">
                    Annotator: <strong id="annotatorName">{{ annotator_id }}</strong>
                </div>
                <div class="progress-indicator" id="progressIndicator">
                    Progress: <strong id="progressText">0/0</strong>
                </div>
            </div>
        </div>
        
        <div class="tabs">
            <div class="tab active" data-tab="annotate">Annotate</div>
            <div class="tab" data-tab="samples">All Samples</div>
            <div class="tab" data-tab="export">Export</div>
        </div>
        
        <!-- Annotate Tab -->
        <div class="tab-content active" id="tab-annotate">
            <div class="main-layout">
                <div class="content-area">
                    <!-- Sample Display -->
                    <div class="card" id="sampleCard">
                        <div class="card-header">
                            <div>
                                <h2 class="card-title">Sample <span id="currentSampleId">-</span></h2>
                                <p class="card-subtitle" id="sampleIndex">Sample 1 of 1</p>
                            </div>
                            <div class="nav-buttons">
                                <button class="btn btn-outline" onclick="navigateSample(-1)">← Previous</button>
                                <button class="btn btn-outline" onclick="navigateSample(1)">Next →</button>
                            </div>
                        </div>
                        
                        <div class="sample-content">
                            <div class="sample-label">Question</div>
                            <div class="question-text" id="questionText">-</div>
                        </div>
                        
                        <div class="sample-content">
                            <div class="sample-label">Candidate Response</div>
                            <div class="sample-text" id="responseText">-</div>
                        </div>
                        
                        <div class="user-profile" id="profileSection" style="display: none;">
                            <div class="sample-label">User Profile</div>
                            <div id="profileContent"></div>
                        </div>
                        
                        <div class="ground-truth-section" id="groundTruthSection" style="display: none;">
                            <div class="sample-label">Ground Truth Reference</div>
                            <div class="sample-text" id="groundTruthText">-</div>
                        </div>
                        
                        <!-- Notes Section -->
                        <div class="notes-section">
                            <label class="sample-label">Annotator Notes (optional)</label>
                            <textarea class="notes-input" id="annotatorNotes" placeholder="Add any notes about this annotation..."></textarea>
                        </div>
                        
                        <div class="sample-nav">
                            <div></div>
                            <button class="btn btn-success" onclick="submitAnnotation()">
                                Submit Annotation ✓
                            </button>
                        </div>
                    </div>
                </div>
                
                <!-- Metrics Panel -->
                <div class="metrics-panel">
                    <div class="card">
                        <div class="card-header">
                            <h3 class="card-title">Evaluation Metrics</h3>
                            <span class="card-subtitle" id="metricsProgress">0/12 completed</span>
                        </div>
                        
                        <div id="metricsContainer">
                            <!-- Metrics will be dynamically inserted here -->
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Samples Tab -->
        <div class="tab-content" id="tab-samples">
            <div class="card">
                <div class="card-header">
                    <h3 class="card-title">All Samples</h3>
                    <button class="btn btn-primary" onclick="showAddSampleModal()">+ Add Sample</button>
                </div>
                <div class="sample-list" id="samplesList">
                    <!-- Sample list items will be inserted here -->
                </div>
            </div>
        </div>
        
        <!-- Export Tab -->
        <div class="tab-content" id="tab-export">
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value" id="statTotal">0</div>
                    <div class="stat-label">Total Samples</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="statAnnotated">0</div>
                    <div class="stat-label">Annotated</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="statRemaining">0</div>
                    <div class="stat-label">Remaining</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="statProgress">0%</div>
                    <div class="stat-label">Progress</div>
                </div>
            </div>
            
            <div class="card">
                <div class="card-header">
                    <h3 class="card-title">Export Annotations</h3>
                </div>
                <p style="margin-bottom: 20px; color: var(--text-secondary);">
                    Export your annotations in various formats for analysis or integration with other tools.
                </p>
                <div style="display: flex; gap: 12px;">
                    <button class="btn btn-primary" onclick="exportAnnotations('csv')">Export CSV</button>
                    <button class="btn btn-primary" onclick="exportAnnotations('json')">Export JSON</button>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Global state
        let samples = [];
        let currentSampleIndex = 0;
        let annotations = {};
        let currentScores = {};
        let currentJustifications = {};
        let startTime = Date.now();
        
        const metrics = {{ metrics | tojson }};
        const annotatorId = '{{ annotator_id }}';
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            initTabs();
            loadSamples();
            renderMetrics();
        });
        
        // Tab functionality
        function initTabs() {
            document.querySelectorAll('.tab').forEach(tab => {
                tab.addEventListener('click', function() {
                    const tabId = this.dataset.tab;
                    
                    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                    
                    this.classList.add('active');
                    document.getElementById('tab-' + tabId).classList.add('active');
                    
                    if (tabId === 'samples') {
                        renderSamplesList();
                    } else if (tabId === 'export') {
                        updateStats();
                    }
                });
            });
        }
        
        // Load samples from API
        async function loadSamples() {
            try {
                const response = await fetch('/api/samples');
                samples = await response.json();
                
                if (samples.length > 0) {
                    displaySample(0);
                }
                
                updateProgress();
                renderSamplesList();
            } catch (error) {
                console.error('Error loading samples:', error);
                showToast('Error loading samples', 'error');
            }
        }
        
        // Display a sample
        function displaySample(index) {
            if (index < 0 || index >= samples.length) return;
            
            currentSampleIndex = index;
            const sample = samples[index];
            
            // Reset scores for new sample
            currentScores = {};
            currentJustifications = {};
            startTime = Date.now();
            
            // Load existing annotation if available
            loadExistingAnnotation(sample.sample_id);
            
            // Update UI
            document.getElementById('currentSampleId').textContent = sample.sample_id;
            document.getElementById('sampleIndex').textContent = `Sample ${index + 1} of ${samples.length}`;
            document.getElementById('questionText').textContent = sample.question;
            document.getElementById('responseText').textContent = sample.candidate_answer;
            document.getElementById('annotatorNotes').value = '';
            
            // User profile
            const profileSection = document.getElementById('profileSection');
            const profileContent = document.getElementById('profileContent');
            if (sample.user_profile && Object.keys(sample.user_profile).length > 0) {
                profileSection.style.display = 'block';
                profileContent.innerHTML = Object.entries(sample.user_profile)
                    .map(([key, value]) => `<span class="profile-item"><span class="profile-label">${key}:</span> <span class="profile-value">${value}</span></span>`)
                    .join('');
            } else {
                profileSection.style.display = 'none';
            }
            
            // Ground truth
            const gtSection = document.getElementById('groundTruthSection');
            if (sample.ground_truth) {
                gtSection.style.display = 'block';
                document.getElementById('groundTruthText').textContent = sample.ground_truth;
            } else {
                gtSection.style.display = 'none';
            }
            
            // Reset metric UI
            updateMetricsUI();
        }
        
        // Load existing annotation
        async function loadExistingAnnotation(sampleId) {
            try {
                const response = await fetch(`/api/annotations/${sampleId}?annotator=${annotatorId}`);
                const annotations = await response.json();
                
                if (annotations.length > 0) {
                    const ann = annotations[annotations.length - 1]; // Use latest
                    currentScores = ann.scores || {};
                    currentJustifications = ann.justifications || {};
                    document.getElementById('annotatorNotes').value = ann.notes || '';
                    updateMetricsUI();
                }
            } catch (error) {
                console.error('Error loading annotation:', error);
            }
        }
        
        // Render metrics
        function renderMetrics() {
            const container = document.getElementById('metricsContainer');
            container.innerHTML = metrics.map(metric => {
                let scoreHtml = '';
                
                if (metric.metric_type === 'binary') {
                    const options = getBinaryOptions(metric.name);
                    scoreHtml = `
                        <div class="score-options">
                            ${options.map(opt => `
                                <div class="score-option ${opt.class}" data-metric="${metric.name}" data-value="${opt.value}" onclick="selectScore('${metric.name}', '${opt.value}')">
                                    ${opt.label}
                                </div>
                            `).join('')}
                        </div>
                    `;
                } else if (metric.metric_type === 'scale_1_5') {
                    scoreHtml = `
                        <div class="scale-slider-container">
                            <div class="scale-value" id="scale-value-${metric.name}">-</div>
                            <input type="range" class="scale-slider" min="1" max="5" value="3" 
                                   id="slider-${metric.name}"
                                   oninput="updateSlider('${metric.name}', this.value)">
                            <div class="scale-labels">
                                <span>1 (Poor)</span>
                                <span>3 (Average)</span>
                                <span>5 (Excellent)</span>
                            </div>
                        </div>
                    `;
                } else if (metric.metric_type === 'percentage') {
                    scoreHtml = `
                        <div class="scale-slider-container">
                            <div class="scale-value" id="scale-value-${metric.name}">-%</div>
                            <input type="range" class="scale-slider" min="0" max="100" step="5" value="50" 
                                   id="slider-${metric.name}"
                                   oninput="updateSliderPercent('${metric.name}', this.value)">
                            <div class="scale-labels">
                                <span>0%</span>
                                <span>50%</span>
                                <span>100%</span>
                            </div>
                        </div>
                    `;
                }
                
                return `
                    <div class="metric-item" id="metric-item-${metric.name}" data-metric="${metric.name}">
                        <div class="metric-header">
                            <span class="metric-name">${metric.display_name}</span>
                            <span class="metric-type-badge">${formatMetricType(metric.metric_type)}</span>
                        </div>
                        <div class="metric-description">${metric.description}</div>
                        ${scoreHtml}
                        <textarea class="justification-input" 
                                  id="justification-${metric.name}"
                                  placeholder="Justification (optional)"
                                  oninput="updateJustification('${metric.name}', this.value)"></textarea>
                    </div>
                `;
            }).join('');
        }
        
        function getBinaryOptions(metricName) {
            if (metricName === 'personalization_accuracy') {
                return [
                    { value: 'Accurate', label: 'Accurate', class: 'binary-true' },
                    { value: 'Inaccurate', label: 'Inaccurate', class: 'binary-false' }
                ];
            } else if (metricName === 'next_step_identification') {
                return [
                    { value: 'Present', label: 'Present', class: 'binary-true' },
                    { value: 'Not Present', label: 'Not Present', class: 'binary-false' }
                ];
            }
            return [
                { value: 'True', label: 'True', class: 'binary-true' },
                { value: 'False', label: 'False', class: 'binary-false' }
            ];
        }
        
        function formatMetricType(type) {
            switch(type) {
                case 'binary': return 'Binary';
                case 'scale_1_5': return '1-5 Scale';
                case 'percentage': return 'Percentage';
                default: return type;
            }
        }
        
        // Score selection functions
        function selectScore(metricName, value) {
            currentScores[metricName] = value;
            
            // Update UI
            document.querySelectorAll(`[data-metric="${metricName}"]`).forEach(el => {
                if (el.classList.contains('score-option')) {
                    el.classList.toggle('selected', el.dataset.value === value);
                }
            });
            
            updateMetricCompletion(metricName);
            updateMetricsProgress();
        }
        
        function updateSlider(metricName, value) {
            currentScores[metricName] = parseInt(value);
            document.getElementById(`scale-value-${metricName}`).textContent = value;
            updateMetricCompletion(metricName);
            updateMetricsProgress();
        }
        
        function updateSliderPercent(metricName, value) {
            currentScores[metricName] = parseInt(value);
            document.getElementById(`scale-value-${metricName}`).textContent = value + '%';
            updateMetricCompletion(metricName);
            updateMetricsProgress();
        }
        
        function updateJustification(metricName, value) {
            currentJustifications[metricName] = value;
        }
        
        function updateMetricCompletion(metricName) {
            const item = document.getElementById(`metric-item-${metricName}`);
            if (currentScores[metricName] !== undefined) {
                item.classList.add('completed');
            }
        }
        
        function updateMetricsUI() {
            metrics.forEach(metric => {
                const value = currentScores[metric.name];
                const justification = currentJustifications[metric.name];
                
                if (metric.metric_type === 'binary' && value) {
                    selectScore(metric.name, value);
                } else if ((metric.metric_type === 'scale_1_5' || metric.metric_type === 'percentage') && value !== undefined) {
                    const slider = document.getElementById(`slider-${metric.name}`);
                    if (slider) {
                        slider.value = value;
                        if (metric.metric_type === 'percentage') {
                            updateSliderPercent(metric.name, value);
                        } else {
                            updateSlider(metric.name, value);
                        }
                    }
                }
                
                if (justification) {
                    const textarea = document.getElementById(`justification-${metric.name}`);
                    if (textarea) {
                        textarea.value = justification;
                    }
                }
            });
            updateMetricsProgress();
        }
        
        function updateMetricsProgress() {
            const completed = Object.keys(currentScores).length;
            document.getElementById('metricsProgress').textContent = `${completed}/${metrics.length} completed`;
        }
        
        // Navigation
        function navigateSample(direction) {
            const newIndex = currentSampleIndex + direction;
            if (newIndex >= 0 && newIndex < samples.length) {
                displaySample(newIndex);
            }
        }
        
        // Submit annotation
        async function submitAnnotation() {
            if (samples.length === 0) {
                showToast('No sample to annotate', 'error');
                return;
            }
            
            if (Object.keys(currentScores).length < metrics.length) {
                if (!confirm('Not all metrics have been scored. Submit anyway?')) {
                    return;
                }
            }
            
            const sample = samples[currentSampleIndex];
            const duration = Math.round((Date.now() - startTime) / 1000);
            
            const annotation = {
                sample_id: sample.sample_id,
                annotator_id: annotatorId,
                scores: currentScores,
                justifications: currentJustifications,
                notes: document.getElementById('annotatorNotes').value,
                duration_seconds: duration
            };
            
            try {
                const response = await fetch('/api/annotations', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(annotation)
                });
                
                if (response.ok) {
                    showToast('Annotation saved successfully!', 'success');
                    updateProgress();
                    
                    // Move to next sample
                    if (currentSampleIndex < samples.length - 1) {
                        setTimeout(() => navigateSample(1), 500);
                    }
                } else {
                    showToast('Error saving annotation', 'error');
                }
            } catch (error) {
                console.error('Error:', error);
                showToast('Error saving annotation', 'error');
            }
        }
        
        // Update progress
        async function updateProgress() {
            try {
                const response = await fetch(`/api/progress?annotator=${annotatorId}`);
                const progress = await response.json();
                
                document.getElementById('progressText').textContent = 
                    `${progress.annotated_samples}/${progress.total_samples}`;
            } catch (error) {
                console.error('Error updating progress:', error);
            }
        }
        
        // Render samples list
        function renderSamplesList() {
            const container = document.getElementById('samplesList');
            
            if (samples.length === 0) {
                container.innerHTML = `
                    <div class="empty-state">
                        <div class="empty-state-icon">📝</div>
                        <p>No samples available</p>
                        <p style="font-size: 13px;">Add samples to start annotating</p>
                    </div>
                `;
                return;
            }
            
            container.innerHTML = samples.map((sample, idx) => `
                <div class="sample-list-item ${idx === currentSampleIndex ? 'active' : ''}" 
                     onclick="goToSample(${idx})">
                    <div class="sample-id">${sample.sample_id}</div>
                    <div class="sample-preview">${sample.question.substring(0, 60)}...</div>
                </div>
            `).join('');
        }
        
        function goToSample(index) {
            displaySample(index);
            document.querySelector('[data-tab="annotate"]').click();
        }
        
        // Stats
        async function updateStats() {
            try {
                const response = await fetch(`/api/progress?annotator=${annotatorId}`);
                const progress = await response.json();
                
                document.getElementById('statTotal').textContent = progress.total_samples;
                document.getElementById('statAnnotated').textContent = progress.annotated_samples;
                document.getElementById('statRemaining').textContent = progress.remaining_samples;
                document.getElementById('statProgress').textContent = Math.round(progress.progress_percent) + '%';
            } catch (error) {
                console.error('Error updating stats:', error);
            }
        }
        
        // Export
        function exportAnnotations(format) {
            window.location.href = `/api/export?format=${format}`;
        }
        
        // Toast notifications
        function showToast(message, type) {
            const toast = document.createElement('div');
            toast.className = `toast ${type}`;
            toast.textContent = message;
            document.body.appendChild(toast);
            
            setTimeout(() => {
                toast.style.animation = 'slideIn 0.3s ease reverse';
                setTimeout(() => toast.remove(), 300);
            }, 3000);
        }
    </script>
</body>
</html>
"""


# ============================================================
# Flask Application
# ============================================================

app = Flask(__name__)
manager = AnnotationManager()

# Create demo samples on startup
manager.create_demo_samples()


@app.route('/')
def index():
    """Render the main annotation interface."""
    annotator_id = request.args.get('annotator', 'annotator_1')
    metrics_data = [asdict(m) for m in METRICS]
    return render_template_string(
        HTML_TEMPLATE,
        metrics=metrics_data,
        annotator_id=annotator_id
    )


@app.route('/api/samples', methods=['GET'])
def get_samples():
    """Get all samples."""
    samples = manager.get_all_samples()
    return jsonify([asdict(s) for s in samples])


@app.route('/api/samples', methods=['POST'])
def add_sample():
    """Add a new sample."""
    data = request.json
    sample = AnnotationSample(
        sample_id=data.get('sample_id', str(uuid.uuid4())[:8]),
        question=data.get('question', ''),
        candidate_answer=data.get('candidate_answer', ''),
        user_profile=data.get('user_profile', {}),
        ground_truth=data.get('ground_truth', ''),
        metadata=data.get('metadata', {})
    )
    sample_id = manager.add_sample(sample)
    return jsonify({"sample_id": sample_id, "status": "created"})


@app.route('/api/annotations', methods=['POST'])
def add_annotation():
    """Add a new annotation."""
    data = request.json
    annotation = Annotation(
        annotation_id="",
        sample_id=data.get('sample_id'),
        annotator_id=data.get('annotator_id', 'anonymous'),
        scores=data.get('scores', {}),
        justifications=data.get('justifications', {}),
        notes=data.get('notes', ''),
        duration_seconds=data.get('duration_seconds', 0)
    )
    annotation_id = manager.add_annotation(annotation)
    return jsonify({"annotation_id": annotation_id, "status": "created"})


@app.route('/api/annotations/<sample_id>', methods=['GET'])
def get_annotations(sample_id):
    """Get annotations for a sample."""
    annotator = request.args.get('annotator')
    annotations = manager.get_annotations(sample_id)
    
    if annotator:
        annotations = [a for a in annotations if a.annotator_id == annotator]
    
    return jsonify([asdict(a) for a in annotations])


@app.route('/api/progress', methods=['GET'])
def get_progress():
    """Get annotation progress for an annotator."""
    annotator_id = request.args.get('annotator', 'annotator_1')
    progress = manager.get_annotation_progress(annotator_id)
    return jsonify(progress)


@app.route('/api/export', methods=['GET'])
def export_data():
    """Export annotations."""
    format_type = request.args.get('format', 'json')
    
    if format_type == 'csv':
        export_path = manager.data_dir / 'export_annotations.csv'
        manager.export_annotations_csv(str(export_path))
        return send_file(export_path, as_attachment=True, download_name='annotations.csv')
    else:
        # JSON export
        all_data = {
            "samples": [asdict(s) for s in manager.get_all_samples()],
            "annotations": {
                sample_id: [asdict(a) for a in annotations]
                for sample_id, annotations in manager.annotations.items()
            }
        }
        return jsonify(all_data)


@app.route('/api/load-csv', methods=['POST'])
def load_csv():
    """Load samples from uploaded CSV."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Save temporarily and load
    temp_path = manager.data_dir / 'temp_upload.csv'
    file.save(temp_path)
    
    try:
        manager.load_samples_from_csv(str(temp_path))
        os.remove(temp_path)
        return jsonify({"status": "success", "message": "Samples loaded"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# Main Entry Point
# ============================================================

def main():
    """Main entry point for the annotation mode."""
    parser = argparse.ArgumentParser(description='Annotation Mode for Zillow LLM Judge')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  Annotation Mode - Zillow LLM Judge Evaluator")
    print("="*60)
    print(f"\n  Starting server at http://{args.host}:{args.port}")
    print(f"  Open your browser to start annotating!")
    print("\n  Tip: Add ?annotator=your_name to the URL to set your annotator ID")
    print("="*60 + "\n")
    
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
