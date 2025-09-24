"""
Converter to transform existing Databricks notebooks to the new evaluation framework
"""

import re
import yaml
from typing import Dict, List, Any
import json


class NotebookToConfigConverter:
    """Convert notebook-style evaluation code to configuration format"""
    
    def __init__(self):
        self.config = {
            'experiment': {},
            'data': {
                'columns': {},
                'files': []
            },
            'models': {
                'response_generation': {},
                'judge_models': []
            },
            'metrics': [],
            'api': {
                'headers': {}
            },
            'output': {},
            'mlflow': {}
        }
        
    def extract_metric_from_prompt(self, name: str, prompt_text: str, 
                                   threshold: float = None) -> Dict[str, Any]:
        """Extract metric configuration from a prompt template"""
        
        # Try to extract score range from prompt
        score_range = [0, 1]  # default
        
        # Look for patterns like "1-5", "0-1", etc.
        range_patterns = [
            r'(?:score|rate).*?(\d+)[-–](\d+)',
            r'from\s+(\d+)\s+to\s+(\d+)',
            r'(\d+)\s+for.*?(\d+)\s+for',
        ]
        
        for pattern in range_patterns:
            match = re.search(pattern, prompt_text, re.IGNORECASE)
            if match:
                score_range = [int(match.group(1)), int(match.group(2))]
                break
                
        # Try to extract the score key name
        score_key_match = re.search(r'"(\w+_score)"', prompt_text)
        score_key = score_key_match.group(1) if score_key_match else f"{name.lower()}_score"
        
        # Clean up the prompt template
        clean_prompt = prompt_text.strip()
        
        # Determine threshold if not provided
        if threshold is None:
            if score_range == [0, 1]:
                threshold = 0.5
            else:
                # Use middle value for range
                threshold = (score_range[0] + score_range[1]) / 2
                
        metric_config = {
            'name': name.lower(),
            'description': f"Evaluates {name.replace('_', ' ')}",
            'prompt_template': clean_prompt,
            'score_range': score_range,
            'threshold': threshold,
            'output_format': 'json'
        }
        
        return metric_config
        
    def convert_from_notebook_variables(self, notebook_vars: Dict[str, Any]) -> Dict[str, Any]:
        """Convert notebook variables to configuration format"""
        
        # Extract experiment configuration
        if 'EXPERIMENT_NAME' in notebook_vars:
            self.config['experiment']['name'] = notebook_vars['EXPERIMENT_NAME']
        if 'RUN_NAME' in notebook_vars:
            self.config['experiment']['run_name_prefix'] = notebook_vars['RUN_NAME'].split('_')[0]
            
        # Extract data configuration
        if 'IN_PATH' in notebook_vars:
            import os
            self.config['data']['ground_truth_dir'] = os.path.dirname(notebook_vars['IN_PATH'])
            
        if 'PROMPT_COL_NAME' in notebook_vars:
            self.config['data']['columns']['prompt'] = notebook_vars['PROMPT_COL_NAME']
        if 'RESPONSE_COL_NAME' in notebook_vars:
            self.config['data']['columns']['response'] = notebook_vars['RESPONSE_COL_NAME']
        if 'PERSONALIZED_FEATURE_COL_NAME' in notebook_vars:
            self.config['data']['columns']['personalization'] = notebook_vars['PERSONALIZED_FEATURE_COL_NAME']
            
        # Extract model configuration
        if 'RESPONSE_MODEL' in notebook_vars:
            response_model = notebook_vars['RESPONSE_MODEL']
            if response_model == "GOLDEN_RESPONSE":
                self.config['models']['response_generation']['type'] = 'ground_truth'
            elif response_model == "FIRST_CALL":
                self.config['models']['response_generation']['type'] = 'api_endpoint'
            else:
                self.config['models']['response_generation']['type'] = 'llm'
                self.config['models']['response_generation']['model_name'] = response_model
                
        if 'JUDGE_MODELS' in notebook_vars:
            self.config['models']['judge_models'] = notebook_vars['JUDGE_MODELS']
            
        # Extract API configuration
        if 'BASE_URL' in notebook_vars:
            self.config['api']['base_url'] = notebook_vars['BASE_URL']
        if 'API_URL' in notebook_vars:
            self.config['api']['endpoints'] = {'response_generation': notebook_vars['API_URL']}
            
        # Extract output configuration
        if 'RESULTS_OUT_PATH' in notebook_vars:
            import os
            self.config['output']['results_dir'] = os.path.dirname(notebook_vars['RESULTS_OUT_PATH'])
            
        # MLflow configuration
        self.config['mlflow']['enable_logging'] = True
        
        return self.config
        
    def extract_metrics_from_prompts(self, prompt_templates: Dict[str, str], 
                                    thresholds: Dict[str, float] = None) -> List[Dict[str, Any]]:
        """Extract metric configurations from prompt template dictionary"""
        
        metrics = []
        thresholds = thresholds or {}
        
        for name, prompt in prompt_templates.items():
            threshold = thresholds.get(name)
            metric_config = self.extract_metric_from_prompt(name, prompt, threshold)
            metrics.append(metric_config)
            
        return metrics
        
    def save_config(self, output_path: str):
        """Save configuration to YAML file"""
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
            
            
def convert_notebook_to_config(notebook_path: str, output_path: str):
    """
    Convert a Databricks notebook to evaluation framework configuration
    
    This is a simplified converter - you'll need to manually extract
    the relevant variables from your notebook
    """
    
    converter = NotebookToConfigConverter()
    
    # Example: Extract these from your notebook
    notebook_vars = {
        'EXPERIMENT_NAME': '/golden_prompt_mlflow3_exp',
        'RUN_NAME': 'gpt-4o_GOLDEN_RESPONSE_20240315',
        'IN_PATH': './imputed_datasets/imputed_responses.csv',
        'PROMPT_COL_NAME': 'prompt',
        'RESPONSE_COL_NAME': 'response',
        'PERSONALIZED_FEATURE_COL_NAME': 'user_personalization_features',
        'RESPONSE_MODEL': 'GOLDEN_RESPONSE',
        'JUDGE_MODELS': ['gpt-4o'],
        'BASE_URL': 'https://example.com',
        'API_URL': 'https://example.com/api',
        'RESULTS_OUT_PATH': './results/evaluation_results.csv'
    }
    
    # Convert basic configuration
    converter.convert_from_notebook_variables(notebook_vars)
    
    # Example prompt templates from notebook
    prompt_templates = {
        'personalization_accuracy': '''You are an impartial evaluator...''',
        'context_personalization': '''You are an impartial evaluator...''',
        'general_personalization': '''You are an impartial evaluator...'''
    }
    
    thresholds = {
        'personalization_accuracy': 1,
        'context_personalization': 3,
        'general_personalization': 3
    }
    
    # Convert metrics
    converter.config['metrics'] = converter.extract_metrics_from_prompts(
        prompt_templates, thresholds
    )
    
    # Save configuration
    converter.save_config(output_path)
    print(f"Configuration saved to {output_path}")
    
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert notebook to config")
    parser.add_argument("notebook", help="Path to notebook file")
    parser.add_argument("output", help="Output configuration file path")
    args = parser.parse_args()
    
    convert_notebook_to_config(args.notebook, args.output)