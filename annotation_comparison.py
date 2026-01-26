"""
Annotation Comparison Module

This module provides tools to compare human annotations with LLM judge scores,
calculate inter-annotator agreement, and analyze annotation quality.

Features:
- Compare human vs LLM judge scores
- Calculate agreement metrics (Cohen's Kappa, Pearson correlation)
- Identify disagreements for review
- Generate comparison reports
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict

# Try to import scipy for statistical calculations
try:
    from scipy import stats
    from scipy.stats import pearsonr, spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Note: scipy not available. Some statistical calculations will be simplified.")


@dataclass
class ComparisonResult:
    """Result of comparing annotations."""
    metric_name: str
    sample_id: str
    human_score: Any
    llm_score: Any
    agreement: bool
    difference: Optional[float] = None
    notes: str = ""


class AnnotationComparator:
    """Compare human annotations with LLM judge scores."""
    
    def __init__(self):
        self.human_annotations: Dict[str, Dict[str, Any]] = {}  # sample_id -> {metric -> score}
        self.llm_scores: Dict[str, Dict[str, Any]] = {}  # sample_id -> {metric -> score}
        self.comparisons: List[ComparisonResult] = []
        
        # Define metric types for proper comparison
        self.metric_types = {
            "personalization_accuracy": "binary",
            "context_based_personalization": "scale_1_5",
            "next_step_identification": "binary",
            "assumption_listing": "binary",
            "assumption_trust": "scale_1_5",
            "calculation_accuracy": "binary",
            "faithfulness_to_ground_truth": "binary",
            "fair_housing_compliance": "binary",
            "overall_accuracy": "binary",
            "structured_presentation": "scale_1_5",
            "coherence": "binary",
            "completeness": "scale_1_5",
        }
        
        # Binary value mappings
        self.binary_positive = {"True", "Accurate", "Present", "1", 1, True}
        self.binary_negative = {"False", "Inaccurate", "Not Present", "0", 0, False}
    
    def load_human_annotations(self, filepath: str) -> int:
        """Load human annotations from CSV or JSON file."""
        path = Path(filepath)
        
        if path.suffix == '.csv':
            df = pd.read_csv(filepath)
            for _, row in df.iterrows():
                sample_id = row.get('sample_id', row.get('id', ''))
                if sample_id:
                    self.human_annotations[sample_id] = {}
                    for metric in self.metric_types:
                        if metric in row:
                            self.human_annotations[sample_id][metric] = row[metric]
        elif path.suffix == '.json':
            with open(filepath, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        sample_id = item.get('sample_id', '')
                        if sample_id:
                            self.human_annotations[sample_id] = {}
                            for metric in self.metric_types:
                                if metric in item or 'scores' in item:
                                    scores = item.get('scores', item)
                                    if metric in scores:
                                        self.human_annotations[sample_id][metric] = scores[metric]
                else:
                    self.human_annotations = data
        
        return len(self.human_annotations)
    
    def load_llm_scores(self, filepath: str) -> int:
        """Load LLM judge scores from CSV or JSON file."""
        path = Path(filepath)
        
        if path.suffix == '.csv':
            df = pd.read_csv(filepath)
            for _, row in df.iterrows():
                sample_id = row.get('sample_id', row.get('id', ''))
                if sample_id:
                    self.llm_scores[sample_id] = {}
                    for metric in self.metric_types:
                        if metric in row:
                            self.llm_scores[sample_id][metric] = row[metric]
        elif path.suffix == '.json':
            with open(filepath, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        sample_id = item.get('sample_id', '')
                        if sample_id:
                            self.llm_scores[sample_id] = {}
                            scores = item.get('detailed_scores', item.get('scores', item))
                            for metric in self.metric_types:
                                if metric in scores:
                                    score_data = scores[metric]
                                    if isinstance(score_data, dict):
                                        self.llm_scores[sample_id][metric] = score_data.get('score', score_data.get('numeric_score'))
                                    else:
                                        self.llm_scores[sample_id][metric] = score_data
                else:
                    self.llm_scores = data
        
        return len(self.llm_scores)
    
    def add_human_annotation(self, sample_id: str, scores: Dict[str, Any]):
        """Add a single human annotation."""
        self.human_annotations[sample_id] = scores
    
    def add_llm_score(self, sample_id: str, scores: Dict[str, Any]):
        """Add a single LLM score."""
        self.llm_scores[sample_id] = scores
    
    def _normalize_binary_score(self, score: Any) -> Optional[bool]:
        """Normalize binary scores to True/False."""
        if score is None:
            return None
        
        score_str = str(score).strip()
        
        if score in self.binary_positive or score_str.lower() in {'true', 'accurate', 'present', '1'}:
            return True
        elif score in self.binary_negative or score_str.lower() in {'false', 'inaccurate', 'not present', '0'}:
            return False
        
        return None
    
    def _normalize_scale_score(self, score: Any) -> Optional[float]:
        """Normalize scale scores to float."""
        if score is None:
            return None
        
        try:
            return float(score)
        except (ValueError, TypeError):
            return None
    
    def compare_all(self) -> List[ComparisonResult]:
        """Compare all annotations with LLM scores."""
        self.comparisons = []
        
        # Find common samples
        common_samples = set(self.human_annotations.keys()) & set(self.llm_scores.keys())
        
        for sample_id in common_samples:
            human_scores = self.human_annotations[sample_id]
            llm_scores = self.llm_scores[sample_id]
            
            for metric, metric_type in self.metric_types.items():
                human_score = human_scores.get(metric)
                llm_score = llm_scores.get(metric)
                
                if human_score is None or llm_score is None:
                    continue
                
                if metric_type == 'binary':
                    h_norm = self._normalize_binary_score(human_score)
                    l_norm = self._normalize_binary_score(llm_score)
                    
                    if h_norm is not None and l_norm is not None:
                        agreement = h_norm == l_norm
                        result = ComparisonResult(
                            metric_name=metric,
                            sample_id=sample_id,
                            human_score=human_score,
                            llm_score=llm_score,
                            agreement=agreement,
                            difference=0 if agreement else 1
                        )
                        self.comparisons.append(result)
                
                elif metric_type in ('scale_1_5', 'percentage'):
                    h_norm = self._normalize_scale_score(human_score)
                    l_norm = self._normalize_scale_score(llm_score)
                    
                    if h_norm is not None and l_norm is not None:
                        diff = abs(h_norm - l_norm)
                        # Agreement if within 1 point for scales
                        agreement = diff <= 1
                        result = ComparisonResult(
                            metric_name=metric,
                            sample_id=sample_id,
                            human_score=human_score,
                            llm_score=llm_score,
                            agreement=agreement,
                            difference=diff
                        )
                        self.comparisons.append(result)
        
        return self.comparisons
    
    def get_agreement_summary(self) -> Dict[str, Any]:
        """Get summary of agreement statistics."""
        if not self.comparisons:
            self.compare_all()
        
        total = len(self.comparisons)
        agreements = sum(1 for c in self.comparisons if c.agreement)
        
        # Per-metric statistics
        metric_stats = defaultdict(lambda: {'total': 0, 'agreed': 0, 'differences': []})
        
        for comp in self.comparisons:
            stat = metric_stats[comp.metric_name]
            stat['total'] += 1
            if comp.agreement:
                stat['agreed'] += 1
            if comp.difference is not None:
                stat['differences'].append(comp.difference)
        
        # Calculate per-metric agreement rates
        per_metric = {}
        for metric, stats in metric_stats.items():
            per_metric[metric] = {
                'agreement_rate': stats['agreed'] / stats['total'] if stats['total'] > 0 else 0,
                'total_comparisons': stats['total'],
                'mean_difference': np.mean(stats['differences']) if stats['differences'] else 0,
                'std_difference': np.std(stats['differences']) if len(stats['differences']) > 1 else 0,
            }
        
        return {
            'overall_agreement_rate': agreements / total if total > 0 else 0,
            'total_comparisons': total,
            'total_agreements': agreements,
            'total_disagreements': total - agreements,
            'per_metric': per_metric,
            'samples_compared': len(set(c.sample_id for c in self.comparisons)),
        }
    
    def get_disagreements(self, min_difference: float = 0) -> List[ComparisonResult]:
        """Get all disagreements, optionally filtered by minimum difference."""
        if not self.comparisons:
            self.compare_all()
        
        return [
            c for c in self.comparisons 
            if not c.agreement and (c.difference is None or c.difference > min_difference)
        ]
    
    def calculate_cohens_kappa(self, metric_name: str) -> Optional[float]:
        """Calculate Cohen's Kappa for a binary metric."""
        if self.metric_types.get(metric_name) != 'binary':
            return None
        
        if not self.comparisons:
            self.compare_all()
        
        # Get binary scores for this metric
        human_binary = []
        llm_binary = []
        
        for comp in self.comparisons:
            if comp.metric_name == metric_name:
                h = self._normalize_binary_score(comp.human_score)
                l = self._normalize_binary_score(comp.llm_score)
                if h is not None and l is not None:
                    human_binary.append(1 if h else 0)
                    llm_binary.append(1 if l else 0)
        
        if len(human_binary) < 2:
            return None
        
        # Calculate Cohen's Kappa
        n = len(human_binary)
        observed_agreement = sum(1 for h, l in zip(human_binary, llm_binary) if h == l) / n
        
        # Expected agreement
        p_h1 = sum(human_binary) / n
        p_l1 = sum(llm_binary) / n
        p_h0 = 1 - p_h1
        p_l0 = 1 - p_l1
        
        expected_agreement = (p_h1 * p_l1) + (p_h0 * p_l0)
        
        if expected_agreement == 1:
            return 1.0
        
        kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
        return kappa
    
    def calculate_correlation(self, metric_name: str) -> Optional[Tuple[float, float]]:
        """Calculate Pearson correlation for a scale metric. Returns (correlation, p-value)."""
        if self.metric_types.get(metric_name) not in ('scale_1_5', 'percentage'):
            return None
        
        if not HAS_SCIPY:
            return None
        
        if not self.comparisons:
            self.compare_all()
        
        human_scores = []
        llm_scores = []
        
        for comp in self.comparisons:
            if comp.metric_name == metric_name:
                h = self._normalize_scale_score(comp.human_score)
                l = self._normalize_scale_score(comp.llm_score)
                if h is not None and l is not None:
                    human_scores.append(h)
                    llm_scores.append(l)
        
        if len(human_scores) < 3:
            return None
        
        correlation, p_value = pearsonr(human_scores, llm_scores)
        return (correlation, p_value)
    
    def generate_report(self) -> str:
        """Generate a detailed comparison report."""
        summary = self.get_agreement_summary()
        disagreements = self.get_disagreements()
        
        report = []
        report.append("=" * 70)
        report.append("ANNOTATION COMPARISON REPORT")
        report.append("=" * 70)
        report.append("")
        
        report.append("OVERALL STATISTICS")
        report.append("-" * 40)
        report.append(f"Total Comparisons: {summary['total_comparisons']}")
        report.append(f"Samples Compared: {summary['samples_compared']}")
        report.append(f"Overall Agreement Rate: {summary['overall_agreement_rate']:.1%}")
        report.append(f"Total Agreements: {summary['total_agreements']}")
        report.append(f"Total Disagreements: {summary['total_disagreements']}")
        report.append("")
        
        report.append("PER-METRIC STATISTICS")
        report.append("-" * 40)
        
        for metric, stats in summary['per_metric'].items():
            metric_type = self.metric_types.get(metric, 'unknown')
            report.append(f"\n{metric} ({metric_type}):")
            report.append(f"  Agreement Rate: {stats['agreement_rate']:.1%}")
            report.append(f"  Comparisons: {stats['total_comparisons']}")
            
            if metric_type == 'binary':
                kappa = self.calculate_cohens_kappa(metric)
                if kappa is not None:
                    report.append(f"  Cohen's Kappa: {kappa:.3f}")
                    if kappa >= 0.8:
                        report.append("  Interpretation: Almost perfect agreement")
                    elif kappa >= 0.6:
                        report.append("  Interpretation: Substantial agreement")
                    elif kappa >= 0.4:
                        report.append("  Interpretation: Moderate agreement")
                    elif kappa >= 0.2:
                        report.append("  Interpretation: Fair agreement")
                    else:
                        report.append("  Interpretation: Slight/Poor agreement")
            else:
                report.append(f"  Mean Difference: {stats['mean_difference']:.2f}")
                report.append(f"  Std Difference: {stats['std_difference']:.2f}")
                
                corr = self.calculate_correlation(metric)
                if corr:
                    report.append(f"  Pearson Correlation: {corr[0]:.3f} (p={corr[1]:.4f})")
        
        if disagreements:
            report.append("")
            report.append("TOP DISAGREEMENTS (for review)")
            report.append("-" * 40)
            
            # Sort by difference (descending)
            sorted_disagreements = sorted(
                disagreements, 
                key=lambda x: x.difference if x.difference else 0,
                reverse=True
            )[:10]
            
            for i, d in enumerate(sorted_disagreements, 1):
                report.append(f"\n{i}. Sample: {d.sample_id}")
                report.append(f"   Metric: {d.metric_name}")
                report.append(f"   Human: {d.human_score} | LLM: {d.llm_score}")
                if d.difference:
                    report.append(f"   Difference: {d.difference}")
        
        report.append("")
        report.append("=" * 70)
        report.append("END OF REPORT")
        report.append("=" * 70)
        
        return "\n".join(report)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert comparisons to a pandas DataFrame."""
        if not self.comparisons:
            self.compare_all()
        
        data = []
        for comp in self.comparisons:
            data.append({
                'sample_id': comp.sample_id,
                'metric_name': comp.metric_name,
                'metric_type': self.metric_types.get(comp.metric_name, 'unknown'),
                'human_score': comp.human_score,
                'llm_score': comp.llm_score,
                'agreement': comp.agreement,
                'difference': comp.difference,
            })
        
        return pd.DataFrame(data)
    
    def export_report(self, filepath: str):
        """Export comparison report to file."""
        report = self.generate_report()
        with open(filepath, 'w') as f:
            f.write(report)
        print(f"Report exported to {filepath}")
    
    def export_dataframe(self, filepath: str):
        """Export comparison data to CSV."""
        df = self.to_dataframe()
        df.to_csv(filepath, index=False)
        print(f"Data exported to {filepath}")


def compare_annotations_cli(human_file: str, llm_file: str, output_dir: str = "."):
    """CLI function to compare annotations and generate reports."""
    comparator = AnnotationComparator()
    
    print(f"Loading human annotations from: {human_file}")
    n_human = comparator.load_human_annotations(human_file)
    print(f"  Loaded {n_human} samples")
    
    print(f"Loading LLM scores from: {llm_file}")
    n_llm = comparator.load_llm_scores(llm_file)
    print(f"  Loaded {n_llm} samples")
    
    print("\nComparing annotations...")
    comparator.compare_all()
    
    # Generate and print report
    report = comparator.generate_report()
    print("\n" + report)
    
    # Export files
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    comparator.export_report(str(output_path / "comparison_report.txt"))
    comparator.export_dataframe(str(output_path / "comparison_data.csv"))
    
    return comparator


# Example usage and demonstration
def demo():
    """Demonstrate the comparison functionality."""
    
    # Create sample data
    comparator = AnnotationComparator()
    
    # Add sample human annotations
    comparator.add_human_annotation("sample_001", {
        "personalization_accuracy": "Accurate",
        "context_based_personalization": 4,
        "next_step_identification": "Present",
        "assumption_listing": "True",
        "assumption_trust": 4,
        "calculation_accuracy": "True",
        "faithfulness_to_ground_truth": "True",
        "fair_housing_compliance": "True",
        "overall_accuracy": "True",
        "structured_presentation": 4,
        "coherence": "True",
        "completeness": 5,
    })
    
    comparator.add_human_annotation("sample_002", {
        "personalization_accuracy": "Accurate",
        "context_based_personalization": 3,
        "next_step_identification": "Present",
        "assumption_listing": "False",
        "assumption_trust": 3,
        "calculation_accuracy": "True",
        "faithfulness_to_ground_truth": "True",
        "fair_housing_compliance": "True",
        "overall_accuracy": "True",
        "structured_presentation": 4,
        "coherence": "True",
        "completeness": 4,
    })
    
    # Add sample LLM scores (simulating some disagreements)
    comparator.add_llm_score("sample_001", {
        "personalization_accuracy": "Accurate",
        "context_based_personalization": 5,  # Slight disagreement
        "next_step_identification": "Present",
        "assumption_listing": "True",
        "assumption_trust": 4,
        "calculation_accuracy": "True",
        "faithfulness_to_ground_truth": "True",
        "fair_housing_compliance": "True",
        "overall_accuracy": "True",
        "structured_presentation": 5,  # Slight disagreement
        "coherence": "True",
        "completeness": 5,
    })
    
    comparator.add_llm_score("sample_002", {
        "personalization_accuracy": "Inaccurate",  # Major disagreement
        "context_based_personalization": 4,
        "next_step_identification": "Present",
        "assumption_listing": "True",  # Disagreement
        "assumption_trust": 2,  # Major disagreement
        "calculation_accuracy": "True",
        "faithfulness_to_ground_truth": "True",
        "fair_housing_compliance": "True",
        "overall_accuracy": "False",  # Major disagreement
        "structured_presentation": 4,
        "coherence": "True",
        "completeness": 3,
    })
    
    # Compare and generate report
    comparator.compare_all()
    print(comparator.generate_report())
    
    return comparator


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) >= 3:
        human_file = sys.argv[1]
        llm_file = sys.argv[2]
        output_dir = sys.argv[3] if len(sys.argv) > 3 else "."
        compare_annotations_cli(human_file, llm_file, output_dir)
    else:
        print("Running demo mode...")
        print("Usage: python annotation_comparison.py <human_annotations.csv> <llm_scores.csv> [output_dir]")
        print("")
        demo()
