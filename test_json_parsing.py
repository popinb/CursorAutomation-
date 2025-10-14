#!/usr/bin/env python3
"""Test JSON parsing functionality"""

import json
import re

print('🧪 TESTING JSON PARSING AND RESPONSE HANDLING')
print('='*50)

# Test different JSON response formats that LLMs might return
test_responses = [
    # Perfect JSON
    '{"factual_accuracy_score": 1, "explanation": "Response is accurate"}',
    
    # JSON with markdown
    '```json\n{"helpfulness_scale_score": 4, "explanation": "Very helpful"}\n```',
    
    # JSON with extra quotes (the KeyError issue we fixed)
    '{"\"completeness_check_score\"": 0.85, "explanation": "85% complete"}',
    
    # Non-JSON text response
    'This response gets a score of 1 because it is accurate and helpful.',
    
    # Percentage in text
    'The response covers about 85% of the question and deserves a score of 0.85',
    
    # Scale response in text
    'I would rate this response a 4 out of 5 for helpfulness'
]

def test_json_parsing(response_content, metric_name, metric_type):
    """Test JSON parsing with fallbacks"""
    # Clean response
    content = response_content.strip()
    
    # Remove markdown
    if content.startswith('```json'):
        content = content.replace('```json', '').replace('```', '').strip()
    elif content.startswith('```'):
        content = content.replace('```', '').strip()
    
    # Try JSON parsing
    try:
        result_json = json.loads(content)
        
        # Clean keys (remove extra quotes)
        cleaned_json = {}
        for key, value in result_json.items():
            clean_key = key.strip('"').strip("'").strip()
            cleaned_json[clean_key] = value
        
        # Find score
        score_key = f'{metric_name}_score'
        if score_key in cleaned_json:
            return cleaned_json[score_key], 'JSON exact match'
        
        # Flexible matching
        for key, value in cleaned_json.items():
            if metric_name.lower() in key.lower() and 'score' in key.lower():
                if isinstance(value, (int, float)):
                    return value, f'JSON flexible match: {key}'
        
        return 0, 'JSON parsed but no score found'
        
    except json.JSONDecodeError:
        # Fallback text parsing
        if metric_type == 'binary':
            if any(word in content.lower() for word in ['pass', 'correct', 'accurate', 'yes', 'true', '1']):
                return 1, 'Text parsing: binary pass'
            elif any(word in content.lower() for word in ['fail', 'incorrect', 'no', 'false', '0']):
                return 0, 'Text parsing: binary fail'
        
        elif metric_type == 'scale_1_5':
            numbers = re.findall(r'\b[1-5]\b', content)
            if numbers:
                return int(numbers[0]), f'Text parsing: scale {numbers[0]}'
        
        elif metric_type == 'percentage':
            percentages = re.findall(r'(\d+(?:\.\d+)?)', content)
            if percentages:
                score = float(percentages[0])
                if score > 1:
                    score = score / 100
                return score, f'Text parsing: percentage {score}'
        
        return 0, 'Text parsing failed'

# Test each response type
print('Testing different LLM response formats:')
for i, response in enumerate(test_responses, 1):
    print(f'\n--- Test Response {i} ---')
    print(f'Input: {response[:60]}...')
    
    # Test with different metric types
    test_cases = [
        ('factual_accuracy', 'binary'),
        ('helpfulness_scale', 'scale_1_5'),
        ('completeness_check', 'percentage')
    ]
    
    for metric_name, metric_type in test_cases:
        score, method = test_json_parsing(response, metric_name, metric_type)
        print(f'  {metric_name} ({metric_type}): score={score}, method={method}')

print('\n✅ JSON PARSING TEST COMPLETE')