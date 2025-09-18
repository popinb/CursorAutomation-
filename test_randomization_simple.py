#!/usr/bin/env python3
"""
Simple Randomization Balance Test (no external dependencies)
Validates the counterbalancing scheme for the Response Latency Study
"""

import random
from collections import defaultdict, Counter

def load_randomization_groups():
    """Load the randomization groups"""
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
    
    participants = []
    for i in range(n_participants):
        participant_id = f"P{i+1:03d}"
        group = random.choice(list(groups.keys()))
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
    for group in sorted(group_counts.keys()):
        count = group_counts[group]
        percentage = (count / len(participants)) * 100
        print(f"   Group {group}: {count} participants ({percentage:.1f}%)")
    
    expected_per_group = len(participants) / 5
    max_deviation = max([abs(count - expected_per_group) for count in group_counts.values()])
    print(f"   Max deviation from expected ({expected_per_group:.1f}): {max_deviation:.1f}")
    
    # 2. Create trial data
    trial_data = []
    for participant in participants:
        for trial_num in range(1, 6):
            trial_key = f"trial_{trial_num}"
            trial_info = participant['group_data'][trial_key]
            
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
    
    # Count by modality
    modality_counts = Counter([trial['modality'] for trial in trial_data])
    print(f"\n   Modality distribution:")
    for modality in sorted(modality_counts.keys()):
        count = modality_counts[modality]
        percentage = (count / len(trial_data)) * 100
        print(f"     {modality}: {count} trials ({percentage:.1f}%)")
    
    # Count by latency
    latency_counts = Counter([trial['latency_ms'] for trial in trial_data])
    print(f"\n   Latency distribution:")
    for latency in sorted(latency_counts.keys()):
        count = latency_counts[latency]
        percentage = (count / len(trial_data)) * 100
        print(f"     {latency}ms: {count} trials ({percentage:.1f}%)")
    
    # Count by prompt
    prompt_counts = Counter([trial['prompt_id'] for trial in trial_data])
    print(f"\n   Prompt distribution:")
    for prompt in sorted(prompt_counts.keys()):
        count = prompt_counts[prompt]
        percentage = (count / len(trial_data)) * 100
        print(f"     Prompt {prompt}: {count} trials ({percentage:.1f}%)")
    
    # 3. Detailed condition matrix
    print(f"\n3. DETAILED CONDITION MATRIX:")
    condition_matrix = defaultdict(lambda: defaultdict(int))
    
    for trial in trial_data:
        prompt_key = f"P{trial['prompt_id']}"
        condition = f"{trial['modality']}_{trial['latency_ms']}ms"
        condition_matrix[prompt_key][condition] += 1
    
    # Get all conditions
    all_conditions = set()
    for prompt_data in condition_matrix.values():
        all_conditions.update(prompt_data.keys())
    all_conditions = sorted(all_conditions)
    
    # Print table header
    print(f"{'Prompt':<8}", end="")
    for condition in all_conditions:
        print(f"{condition:<18}", end="")
    print()
    print("-" * (8 + 18 * len(all_conditions)))
    
    # Print table rows
    for prompt in sorted(condition_matrix.keys()):
        print(f"{prompt:<8}", end="")
        for condition in all_conditions:
            count = condition_matrix[prompt][condition]
            print(f"{count:<18}", end="")
        print()
    
    # 4. Balance validation
    print(f"\n4. BALANCE VALIDATION:")
    
    # Check streaming vs non-streaming balance
    streaming_trials = sum(1 for t in trial_data if t['modality'] == 'stream')
    nonstream_trials = sum(1 for t in trial_data if t['modality'] == 'nonstream')
    
    print(f"   Streaming trials: {streaming_trials} ({streaming_trials/len(trial_data)*100:.1f}%)")
    print(f"   Non-streaming trials: {nonstream_trials} ({nonstream_trials/len(trial_data)*100:.1f}%)")
    
    # Expected: 40% streaming (2/5 trials), 60% non-streaming (3/5 trials)
    expected_streaming_pct = 40.0
    actual_streaming_pct = (streaming_trials / len(trial_data)) * 100
    streaming_error = abs(actual_streaming_pct - expected_streaming_pct)
    
    print(f"   Expected streaming: {expected_streaming_pct}%")
    print(f"   Actual streaming: {actual_streaming_pct:.1f}%")
    print(f"   Streaming error: {streaming_error:.1f}%")
    
    if streaming_error < 2.0:
        print("   ✅ Streaming/non-streaming balance is excellent")
    elif streaming_error < 5.0:
        print("   ✅ Streaming/non-streaming balance is good")
    else:
        print("   ⚠️  Streaming/non-streaming balance needs attention")
    
    # Check prompt balance
    prompt_counts_list = list(prompt_counts.values())
    prompt_range = max(prompt_counts_list) - min(prompt_counts_list)
    print(f"   Prompt count range: {prompt_range} (max: {max(prompt_counts_list)}, min: {min(prompt_counts_list)})")
    
    if prompt_range <= 2:
        print("   ✅ Prompt balance is excellent")
    elif prompt_range <= 5:
        print("   ✅ Prompt balance is good")
    else:
        print("   ⚠️  Prompt balance needs attention")
    
    return trial_data

def test_theoretical_balance():
    """Test the theoretical perfect balance of the design"""
    print(f"\n🎯 THEORETICAL BALANCE ANALYSIS")
    print("=" * 50)
    
    groups = load_randomization_groups()
    
    # If we had exactly 100 participants (20 per group)
    theoretical_trials = []
    
    for group_id, group_data in groups.items():
        for _ in range(20):  # 20 participants per group
            for trial_num in range(1, 6):
                trial_key = f"trial_{trial_num}"
                trial_info = group_data[trial_key]
                
                theoretical_trials.append({
                    'group': group_id,
                    'prompt_id': trial_info['prompt_id'],
                    'modality': trial_info['modality'],
                    'latency_ms': trial_info['latency_ms']
                })
    
    print(f"With perfect 20 participants per group (N=100):")
    print(f"Total trials: {len(theoretical_trials)}")
    
    # Count conditions
    modality_counts = Counter([t['modality'] for t in theoretical_trials])
    prompt_counts = Counter([t['prompt_id'] for t in theoretical_trials])
    latency_counts = Counter([t['latency_ms'] for t in theoretical_trials])
    
    print(f"\nModality balance:")
    for modality, count in sorted(modality_counts.items()):
        pct = (count / len(theoretical_trials)) * 100
        print(f"  {modality}: {count} trials ({pct:.1f}%)")
    
    print(f"\nPrompt balance:")
    for prompt, count in sorted(prompt_counts.items()):
        pct = (count / len(theoretical_trials)) * 100
        print(f"  Prompt {prompt}: {count} trials ({pct:.1f}%)")
    
    print(f"\nLatency balance:")
    for latency, count in sorted(latency_counts.items()):
        pct = (count / len(theoretical_trials)) * 100
        print(f"  {latency}ms: {count} trials ({pct:.1f}%)")
    
    # Check if all prompts appear in all conditions
    condition_matrix = defaultdict(set)
    for trial in theoretical_trials:
        prompt = trial['prompt_id']
        condition = f"{trial['modality']}_{trial['latency_ms']}"
        condition_matrix[prompt].add(condition)
    
    print(f"\nCondition coverage per prompt:")
    all_possible_conditions = {
        'nonstream_500', 'nonstream_2000', 'nonstream_6000',
        'stream_2000', 'stream_6000'
    }
    
    for prompt in sorted(condition_matrix.keys()):
        conditions = condition_matrix[prompt]
        missing = all_possible_conditions - conditions
        print(f"  Prompt {prompt}: {len(conditions)}/5 conditions", end="")
        if missing:
            print(f" (missing: {', '.join(sorted(missing))})")
        else:
            print(" ✅")

def main():
    """Run all randomization tests"""
    
    print("🧪 RESPONSE LATENCY STUDY - RANDOMIZATION TESTING")
    print("=" * 60)
    
    # Set seed for reproducible results
    random.seed(42)
    
    # Test 1: Theoretical perfect balance
    test_theoretical_balance()
    
    # Test 2: Simulated realistic assignment
    print(f"\n" + "=" * 60)
    participants = simulate_participants(100)
    trial_data = analyze_balance(participants)
    
    # Test 3: Multiple simulations
    print(f"\n🔄 MULTIPLE SIMULATION TEST")
    print("=" * 50)
    
    streaming_percentages = []
    prompt_ranges = []
    
    for sim in range(10):
        sim_participants = simulate_participants(100)
        sim_trials = []
        
        for participant in sim_participants:
            for trial_num in range(1, 6):
                trial_key = f"trial_{trial_num}"
                trial_info = participant['group_data'][trial_key]
                sim_trials.append(trial_info)
        
        streaming_count = sum(1 for t in sim_trials if t['modality'] == 'stream')
        streaming_pct = (streaming_count / len(sim_trials)) * 100
        streaming_percentages.append(streaming_pct)
        
        prompt_counts = Counter([t['prompt_id'] for t in sim_trials])
        prompt_range = max(prompt_counts.values()) - min(prompt_counts.values())
        prompt_ranges.append(prompt_range)
    
    avg_streaming = sum(streaming_percentages) / len(streaming_percentages)
    avg_prompt_range = sum(prompt_ranges) / len(prompt_ranges)
    
    print(f"Across 10 simulations (N=100 each):")
    print(f"Average streaming percentage: {avg_streaming:.1f}% (target: 40.0%)")
    print(f"Streaming percentage range: {min(streaming_percentages):.1f}% - {max(streaming_percentages):.1f}%")
    print(f"Average prompt imbalance: {avg_prompt_range:.1f} trials")
    print(f"Prompt imbalance range: {min(prompt_ranges)} - {max(prompt_ranges)} trials")
    
    if abs(avg_streaming - 40.0) < 1.0:
        print("✅ Streaming balance is consistently excellent")
    elif abs(avg_streaming - 40.0) < 2.0:
        print("✅ Streaming balance is consistently good")
    else:
        print("⚠️  Streaming balance varies too much")
    
    if avg_prompt_range < 3.0:
        print("✅ Prompt balance is consistently excellent")
    elif avg_prompt_range < 5.0:
        print("✅ Prompt balance is consistently good")
    else:
        print("⚠️  Prompt balance varies too much")
    
    print(f"\n✅ RANDOMIZATION TESTING COMPLETE")
    print("=" * 60)
    print("SUMMARY:")
    print("- ✅ Theoretical design provides perfect counterbalancing")
    print("- ✅ Random assignment maintains good balance in practice")
    print("- ✅ Each prompt appears in all available conditions")
    print("- ✅ Streaming/non-streaming ratio is correct (40%/60%)")
    print("- ✅ Balance is consistent across multiple simulations")
    print("- ✅ Design is ready for implementation in Qualtrics")

if __name__ == "__main__":
    main()