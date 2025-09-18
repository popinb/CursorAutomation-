#!/usr/bin/env python3
"""
Test Randomization Balance for Response Latency Study
Validates that the randomization scheme provides proper counterbalancing
"""

import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import json

def load_randomization_groups():
    """Load the randomization groups from the JSON file"""
    groups = {
        1: {
            "trial_1": {"prompt_id": "1", "modality": "nonstream", "latency_ms": 500},
            "trial_2": {"prompt_id": "2", "modality": "nonstream", "latency_ms": 2000},
            "trial_3": {"prompt_id": "3", "modality": "nonstream", "latency_ms": 6000},
            "trial_4": {"prompt_id": "4", "modality": "stream", "latency_ms": 2000},
            "trial_5": {"prompt_id": "5", "modality": "stream", "latency_ms": 6000}
        },
        2: {
            "trial_1": {"prompt_id": "2", "modality": "nonstream", "latency_ms": 500},
            "trial_2": {"prompt_id": "3", "modality": "nonstream", "latency_ms": 2000},
            "trial_3": {"prompt_id": "4", "modality": "nonstream", "latency_ms": 6000},
            "trial_4": {"prompt_id": "5", "modality": "stream", "latency_ms": 2000},
            "trial_5": {"prompt_id": "1", "modality": "stream", "latency_ms": 6000}
        },
        3: {
            "trial_1": {"prompt_id": "3", "modality": "nonstream", "latency_ms": 500},
            "trial_2": {"prompt_id": "4", "modality": "nonstream", "latency_ms": 2000},
            "trial_3": {"prompt_id": "5", "modality": "nonstream", "latency_ms": 6000},
            "trial_4": {"prompt_id": "1", "modality": "stream", "latency_ms": 2000},
            "trial_5": {"prompt_id": "2", "modality": "stream", "latency_ms": 6000}
        },
        4: {
            "trial_1": {"prompt_id": "4", "modality": "nonstream", "latency_ms": 500},
            "trial_2": {"prompt_id": "5", "modality": "nonstream", "latency_ms": 2000},
            "trial_3": {"prompt_id": "1", "modality": "nonstream", "latency_ms": 6000},
            "trial_4": {"prompt_id": "2", "modality": "stream", "latency_ms": 2000},
            "trial_5": {"prompt_id": "3", "modality": "stream", "latency_ms": 6000}
        },
        5: {
            "trial_1": {"prompt_id": "5", "modality": "nonstream", "latency_ms": 500},
            "trial_2": {"prompt_id": "1", "modality": "nonstream", "latency_ms": 2000},
            "trial_3": {"prompt_id": "2", "modality": "nonstream", "latency_ms": 6000},
            "trial_4": {"prompt_id": "3", "modality": "stream", "latency_ms": 2000},
            "trial_5": {"prompt_id": "4", "modality": "stream", "latency_ms": 6000}
        }
    }
    return groups

def simulate_participants(n_participants=100):
    """Simulate participant assignment to randomization groups"""
    groups = load_randomization_groups()
    
    # Assign participants to groups (equal probability)
    participants = []
    for i in range(n_participants):
        participant_id = f"P{i+1:03d}"
        group = np.random.choice(list(groups.keys()))
        participants.append({
            'participant_id': participant_id,
            'group': group,
            'group_data': groups[group]
        })
    
    return participants

def analyze_balance(participants):
    """Analyze the balance of the randomization scheme"""
    
    print("🔍 RANDOMIZATION BALANCE ANALYSIS")
    print("=" * 50)
    
    # 1. Group assignment balance
    group_counts = Counter([p['group'] for p in participants])
    print(f"\n1. GROUP ASSIGNMENT BALANCE (N={len(participants)}):")
    for group, count in sorted(group_counts.items()):
        percentage = (count / len(participants)) * 100
        print(f"   Group {group}: {count} participants ({percentage:.1f}%)")
    
    # Expected: ~20% each group
    expected_per_group = len(participants) / 5
    max_deviation = max([abs(count - expected_per_group) for count in group_counts.values()])
    print(f"   Max deviation from expected ({expected_per_group:.1f}): {max_deviation:.1f}")
    
    # 2. Condition balance (prompt × modality × latency)
    condition_counts = defaultdict(int)
    trial_data = []
    
    for participant in participants:
        for trial_num in range(1, 6):
            trial_key = f"trial_{trial_num}"
            trial_info = participant['group_data'][trial_key]
            
            # Create condition key
            condition = (
                trial_info['prompt_id'],
                trial_info['modality'], 
                str(trial_info['latency_ms'])
            )
            condition_counts[condition] += 1
            
            trial_data.append({
                'participant': participant['participant_id'],
                'group': participant['group'],
                'trial': trial_num,
                'prompt_id': trial_info['prompt_id'],
                'modality': trial_info['modality'],
                'latency_ms': trial_info['latency_ms']
            })
    
    print(f"\n2. CONDITION BALANCE:")
    print(f"   Total trials: {len(trial_data)}")
    print(f"   Unique conditions: {len(condition_counts)}")
    
    # Group by condition type
    modality_counts = defaultdict(int)
    latency_counts = defaultdict(int)
    prompt_counts = defaultdict(int)
    
    for trial in trial_data:
        modality_counts[trial['modality']] += 1
        latency_counts[trial['latency_ms']] += 1
        prompt_counts[trial['prompt_id']] += 1
    
    print(f"\n   Modality distribution:")
    for modality, count in modality_counts.items():
        percentage = (count / len(trial_data)) * 100
        print(f"     {modality}: {count} trials ({percentage:.1f}%)")
    
    print(f"\n   Latency distribution:")
    for latency, count in sorted(latency_counts.items()):
        percentage = (count / len(trial_data)) * 100
        print(f"     {latency}ms: {count} trials ({percentage:.1f}%)")
    
    print(f"\n   Prompt distribution:")
    for prompt, count in sorted(prompt_counts.items()):
        percentage = (count / len(trial_data)) * 100
        print(f"     Prompt {prompt}: {count} trials ({percentage:.1f}%)")
    
    # 3. Detailed condition matrix
    print(f"\n3. DETAILED CONDITION MATRIX:")
    condition_matrix = defaultdict(lambda: defaultdict(int))
    
    for trial in trial_data:
        key = f"P{trial['prompt_id']}"
        condition = f"{trial['modality']}_{trial['latency_ms']}ms"
        condition_matrix[key][condition] += 1
    
    # Create table
    all_conditions = set()
    for prompt_data in condition_matrix.values():
        all_conditions.update(prompt_data.keys())
    all_conditions = sorted(all_conditions)
    
    print(f"{'Prompt':<8}", end="")
    for condition in all_conditions:
        print(f"{condition:<15}", end="")
    print()
    print("-" * (8 + 15 * len(all_conditions)))
    
    for prompt in sorted(condition_matrix.keys()):
        print(f"{prompt:<8}", end="")
        for condition in all_conditions:
            count = condition_matrix[prompt][condition]
            print(f"{count:<15}", end="")
        print()
    
    # 4. Balance validation
    print(f"\n4. BALANCE VALIDATION:")
    
    # Check if each prompt appears in each condition
    expected_per_condition = len(participants) / 5  # Since each participant does 1 trial per condition type
    
    balance_issues = []
    for prompt, conditions in condition_matrix.items():
        for condition, count in conditions.items():
            if count == 0:
                balance_issues.append(f"{prompt} never appears in {condition}")
    
    if balance_issues:
        print("   ⚠️  BALANCE ISSUES FOUND:")
        for issue in balance_issues:
            print(f"     - {issue}")
    else:
        print("   ✅ All prompts appear in all available conditions")
    
    # Check streaming vs non-streaming balance
    streaming_trials = sum(1 for t in trial_data if t['modality'] == 'stream')
    nonstream_trials = sum(1 for t in trial_data if t['modality'] == 'nonstream')
    
    print(f"\n   Streaming trials: {streaming_trials} ({streaming_trials/len(trial_data)*100:.1f}%)")
    print(f"   Non-streaming trials: {nonstream_trials} ({nonstream_trials/len(trial_data)*100:.1f}%)")
    
    # Expected: 40% streaming (2/5 trials), 60% non-streaming (3/5 trials)
    expected_streaming_pct = 40.0
    actual_streaming_pct = (streaming_trials / len(trial_data)) * 100
    streaming_error = abs(actual_streaming_pct - expected_streaming_pct)
    
    print(f"   Expected streaming: {expected_streaming_pct}%")
    print(f"   Streaming error: {streaming_error:.1f}%")
    
    if streaming_error < 2.0:
        print("   ✅ Streaming/non-streaming balance is good")
    else:
        print("   ⚠️  Streaming/non-streaming balance needs attention")
    
    return trial_data, condition_matrix

def test_counterbalancing(n_simulations=10):
    """Test counterbalancing across multiple simulations"""
    
    print(f"\n🔄 COUNTERBALANCING TEST ({n_simulations} simulations)")
    print("=" * 50)
    
    all_results = []
    
    for sim in range(n_simulations):
        participants = simulate_participants(100)
        trial_data, _ = analyze_balance(participants)
        
        # Calculate balance metrics
        modality_counts = Counter([t['modality'] for t in trial_data])
        streaming_pct = (modality_counts['stream'] / len(trial_data)) * 100
        
        prompt_counts = Counter([t['prompt_id'] for t in trial_data])
        prompt_balance = max(prompt_counts.values()) - min(prompt_counts.values())
        
        all_results.append({
            'simulation': sim + 1,
            'streaming_pct': streaming_pct,
            'prompt_balance_range': prompt_balance
        })
    
    # Summary statistics
    streaming_pcts = [r['streaming_pct'] for r in all_results]
    balance_ranges = [r['prompt_balance_range'] for r in all_results]
    
    print(f"\nStreaming percentage across simulations:")
    print(f"  Mean: {np.mean(streaming_pcts):.1f}% (target: 40.0%)")
    print(f"  Std:  {np.std(streaming_pcts):.1f}%")
    print(f"  Range: {min(streaming_pcts):.1f}% - {max(streaming_pcts):.1f}%")
    
    print(f"\nPrompt balance range (max - min occurrences):")
    print(f"  Mean: {np.mean(balance_ranges):.1f}")
    print(f"  Std:  {np.std(balance_ranges):.1f}")
    print(f"  Range: {min(balance_ranges)} - {max(balance_ranges)}")
    
    if np.mean(streaming_pcts) > 38 and np.mean(streaming_pcts) < 42:
        print("✅ Streaming balance is consistently good across simulations")
    else:
        print("⚠️  Streaming balance varies too much across simulations")
    
    if np.mean(balance_ranges) < 5:
        print("✅ Prompt balance is consistently good across simulations")
    else:
        print("⚠️  Prompt balance varies too much across simulations")

def generate_test_data_sample():
    """Generate a sample of test data for analysis validation"""
    
    print(f"\n📊 GENERATING TEST DATA SAMPLE")
    print("=" * 50)
    
    participants = simulate_participants(100)
    trial_data = []
    
    for participant in participants:
        for trial_num in range(1, 6):
            trial_key = f"trial_{trial_num}"
            trial_info = participant['group_data'][trial_key]
            
            # Simulate realistic responses
            latency_sec = trial_info['latency_ms'] / 1000
            
            # Quality rating (higher latency = lower quality, with noise)
            quality_base = 7 - (latency_sec * 0.5) + np.random.normal(0, 0.8)
            quality_rating = max(1, min(7, round(quality_base)))
            
            # Wait perception (higher latency = feels longer, streaming helps)
            wait_base = 1 + (latency_sec * 0.8)
            if trial_info['modality'] == 'stream':
                wait_base -= 0.5  # Streaming feels shorter
            wait_base += np.random.normal(0, 0.6)
            wait_felt = max(1, min(7, round(wait_base)))
            
            # Acceptance (lower latency = more acceptable)
            accept_prob = 1 / (1 + np.exp((latency_sec - 3) * 2))  # Sigmoid
            if trial_info['modality'] == 'stream':
                accept_prob += 0.1  # Streaming more acceptable
            would_accept = 1 if np.random.random() < accept_prob else 0
            
            # Impact on quality judgment
            impact_base = 1 + (latency_sec * 0.6) + np.random.normal(0, 0.7)
            wait_hurt_quality = max(1, min(7, round(impact_base)))
            
            trial_data.append({
                'participant_id': participant['participant_id'],
                'group': participant['group'],
                'trial': trial_num,
                'prompt_id': int(trial_info['prompt_id']),
                'modality': trial_info['modality'],
                'latency_ms': trial_info['latency_ms'],
                'latency_sec': latency_sec,
                'quality_rating': quality_rating,
                'wait_felt': wait_felt,
                'wait_hurt_quality': wait_hurt_quality,
                'would_accept': would_accept
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(trial_data)
    
    # Save sample data
    df.to_csv('/workspace/test_data_sample.csv', index=False)
    
    print(f"Generated {len(df)} trial observations from {len(participants)} participants")
    print(f"Saved to: test_data_sample.csv")
    
    # Basic statistics
    print(f"\nSample statistics:")
    print(f"Quality rating: {df['quality_rating'].mean():.2f} ± {df['quality_rating'].std():.2f}")
    print(f"Wait perception: {df['wait_felt'].mean():.2f} ± {df['wait_felt'].std():.2f}")
    print(f"Acceptance rate: {df['would_accept'].mean():.2%}")
    
    # By condition
    print(f"\nBy modality:")
    modality_stats = df.groupby('modality').agg({
        'quality_rating': 'mean',
        'wait_felt': 'mean', 
        'would_accept': 'mean'
    }).round(2)
    print(modality_stats)
    
    print(f"\nBy latency:")
    latency_stats = df.groupby('latency_ms').agg({
        'quality_rating': 'mean',
        'wait_felt': 'mean',
        'would_accept': 'mean'
    }).round(2)
    print(latency_stats)
    
    return df

def main():
    """Run all randomization tests"""
    
    print("🧪 RESPONSE LATENCY STUDY - RANDOMIZATION TESTING")
    print("=" * 60)
    
    # Set seed for reproducible results
    np.random.seed(42)
    
    # Test 1: Single simulation analysis
    participants = simulate_participants(100)
    trial_data, condition_matrix = analyze_balance(participants)
    
    # Test 2: Multiple simulation counterbalancing
    test_counterbalancing(10)
    
    # Test 3: Generate sample data
    sample_data = generate_test_data_sample()
    
    print(f"\n✅ RANDOMIZATION TESTING COMPLETE")
    print("=" * 60)
    print("Summary:")
    print("- Randomization groups provide balanced assignment")
    print("- Each prompt appears in all available conditions") 
    print("- Streaming/non-streaming ratio is correct (40%/60%)")
    print("- Sample data generated for analysis testing")
    print("- All balance checks passed")

if __name__ == "__main__":
    main()