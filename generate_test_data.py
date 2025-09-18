#!/usr/bin/env python3
"""
Generate realistic test data for the Response Latency Study
Creates CSV data that matches the expected Qualtrics output format
"""

import csv
import random
import math
from datetime import datetime, timedelta

def generate_realistic_responses(latency_sec, modality, prompt_id):
    """Generate realistic responses based on condition parameters"""
    
    # Base quality rating (higher latency = lower quality)
    quality_base = 6.5 - (latency_sec * 0.4)  # Decreases with latency
    if modality == 'stream':
        quality_base += 0.2  # Slight boost for streaming
    quality_base += random.normalvariate(0, 0.8)  # Add noise
    quality_rating = max(1, min(7, round(quality_base)))
    
    # Wait perception (higher latency = feels longer, streaming helps)
    wait_base = 1 + (latency_sec * 0.9)  # Strong relationship with actual latency
    if modality == 'stream':
        wait_base -= 0.6  # Streaming feels notably shorter
    wait_base += random.normalvariate(0, 0.7)  # Add noise
    wait_felt = max(1, min(7, round(wait_base)))
    
    # Impact on quality judgment
    impact_base = 1 + (latency_sec * 0.5)  # Moderate relationship
    if modality == 'stream':
        impact_base -= 0.3  # Streaming reduces negative impact
    impact_base += random.normalvariate(0, 0.8)
    wait_hurt_quality = max(1, min(7, round(impact_base)))
    
    # Acceptance (sigmoid function based on latency)
    accept_logit = 2 - (latency_sec * 1.2)  # Decreases with latency
    if modality == 'stream':
        accept_logit += 0.5  # Streaming increases acceptance
    accept_prob = 1 / (1 + math.exp(-accept_logit))
    would_accept = 7 if random.random() < accept_prob else random.randint(1, 4)
    
    # Perceived wait time in seconds (for slider question)
    perceived_base = latency_sec * random.uniform(0.7, 1.3)  # Some error around actual
    if modality == 'stream':
        perceived_base *= 0.8  # Streaming feels shorter
    perceived_wait = max(0, min(10, round(perceived_base * 2) / 2))  # Round to 0.5s
    
    return {
        'quality_rating': quality_rating,
        'wait_felt': wait_felt,
        'wait_hurt_quality': wait_hurt_quality,
        'would_accept': would_accept,
        'perceived_wait': perceived_wait
    }

def generate_participant_data(participant_id, group_assignment):
    """Generate data for one participant"""
    
    # Group assignments (same as randomization test)
    groups = {
        1: [
            {"prompt_id": "1", "modality": "nonstream", "latency_ms": 500},
            {"prompt_id": "2", "modality": "nonstream", "latency_ms": 2000},
            {"prompt_id": "3", "modality": "nonstream", "latency_ms": 6000},
            {"prompt_id": "4", "modality": "stream", "latency_ms": 2000},
            {"prompt_id": "5", "modality": "stream", "latency_ms": 6000}
        ],
        2: [
            {"prompt_id": "2", "modality": "nonstream", "latency_ms": 500},
            {"prompt_id": "3", "modality": "nonstream", "latency_ms": 2000},
            {"prompt_id": "4", "modality": "nonstream", "latency_ms": 6000},
            {"prompt_id": "5", "modality": "stream", "latency_ms": 2000},
            {"prompt_id": "1", "modality": "stream", "latency_ms": 6000}
        ],
        3: [
            {"prompt_id": "3", "modality": "nonstream", "latency_ms": 500},
            {"prompt_id": "4", "modality": "nonstream", "latency_ms": 2000},
            {"prompt_id": "5", "modality": "nonstream", "latency_ms": 6000},
            {"prompt_id": "1", "modality": "stream", "latency_ms": 2000},
            {"prompt_id": "2", "modality": "stream", "latency_ms": 6000}
        ],
        4: [
            {"prompt_id": "4", "modality": "nonstream", "latency_ms": 500},
            {"prompt_id": "5", "modality": "nonstream", "latency_ms": 2000},
            {"prompt_id": "1", "modality": "nonstream", "latency_ms": 6000},
            {"prompt_id": "2", "modality": "stream", "latency_ms": 2000},
            {"prompt_id": "3", "modality": "stream", "latency_ms": 6000}
        ],
        5: [
            {"prompt_id": "5", "modality": "nonstream", "latency_ms": 500},
            {"prompt_id": "1", "modality": "nonstream", "latency_ms": 2000},
            {"prompt_id": "2", "modality": "nonstream", "latency_ms": 6000},
            {"prompt_id": "3", "modality": "stream", "latency_ms": 2000},
            {"prompt_id": "4", "modality": "stream", "latency_ms": 6000}
        ]
    }
    
    trials = groups[group_assignment]
    
    # Generate responses for each trial
    trial_responses = {}
    perceived_waits = {}
    
    for i, trial in enumerate(trials, 1):
        latency_sec = trial['latency_ms'] / 1000
        responses = generate_realistic_responses(
            latency_sec, trial['modality'], trial['prompt_id']
        )
        
        # Store in format matching Qualtrics column names
        trial_responses[f'quality_rating_{i}'] = responses['quality_rating']
        trial_responses[f'wait_felt_{i}'] = responses['wait_felt']
        trial_responses[f'wait_hurt_quality_{i}'] = responses['wait_hurt_quality']
        trial_responses[f'would_accept_{i}'] = responses['would_accept']
        
        # Store trial metadata
        trial_responses[f'trial_{i}_prompt_id'] = trial['prompt_id']
        trial_responses[f'trial_{i}_modality'] = trial['modality']
        trial_responses[f'trial_{i}_latency_ms'] = trial['latency_ms']
        trial_responses[f'trial_{i}_tft_ms'] = 300
        trial_responses[f'trial_{i}_actual_latency_ms'] = trial['latency_ms'] + random.randint(-50, 50)
        
        # Perceived wait times (for slider question)
        perceived_waits[f'perceived_wait_{chr(65+i-1)}'] = responses['perceived_wait']
    
    # Generate individual difference measures
    # Max acceptable wait (influenced by their actual tolerance)
    avg_acceptance = sum([trial_responses[f'would_accept_{i}'] for i in range(1, 6)]) / 5
    max_acceptable_base = 2 + (avg_acceptance - 4) * 0.5  # Higher acceptance = higher tolerance
    max_acceptable_wait = max(0, min(10, round(max_acceptable_base * 2) / 2))
    
    # Streaming preference (influenced by their streaming vs non-streaming responses)
    stream_quality = sum([trial_responses[f'quality_rating_{i}'] for i in [4, 5]]) / 2
    nonstream_quality = sum([trial_responses[f'quality_rating_{i}'] for i in [1, 2, 3]]) / 3
    stream_wait = sum([trial_responses[f'wait_felt_{i}'] for i in [4, 5]]) / 2
    nonstream_wait = sum([trial_responses[f'wait_felt_{i}'] for i in [1, 2, 3]]) / 3
    
    # Preference logic
    if stream_quality > nonstream_quality and stream_wait < nonstream_wait:
        streaming_preference = 1  # Prefer streaming
    elif stream_quality < nonstream_quality and stream_wait > nonstream_wait:
        streaming_preference = 2  # Prefer non-streaming
    elif random.random() < 0.3:
        streaming_preference = 4  # It depends
    else:
        streaming_preference = 3  # No preference
    
    # Combine all data
    participant_data = {
        'ResponseId': participant_id,
        'group_assignment': group_assignment,
        'max_acceptable_wait': max_acceptable_wait,
        'streaming_preference': streaming_preference,
        **trial_responses,
        **perceived_waits
    }
    
    return participant_data

def generate_full_dataset(n_participants=100):
    """Generate complete dataset for analysis testing"""
    
    print(f"🔄 Generating test dataset with {n_participants} participants...")
    
    participants = []
    
    # Assign participants to groups (roughly equal)
    group_assignments = []
    for group in range(1, 6):
        group_assignments.extend([group] * (n_participants // 5))
    
    # Handle remainder
    remaining = n_participants - len(group_assignments)
    group_assignments.extend(random.choices(range(1, 6), k=remaining))
    
    random.shuffle(group_assignments)
    
    # Generate participant data
    for i in range(n_participants):
        participant_id = f"R_{i+1:03d}_{random.randint(1000000, 9999999)}"
        group = group_assignments[i]
        
        participant_data = generate_participant_data(participant_id, group)
        participants.append(participant_data)
        
        if (i + 1) % 20 == 0:
            print(f"  Generated {i + 1}/{n_participants} participants...")
    
    return participants

def save_dataset(participants, filename):
    """Save dataset to CSV file"""
    
    if not participants:
        print("❌ No participant data to save")
        return
    
    # Get all column names
    all_columns = set()
    for p in participants:
        all_columns.update(p.keys())
    
    # Sort columns for consistent output
    columns = sorted(all_columns)
    
    # Write CSV
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        
        # Add Qualtrics-style headers (2 extra rows)
        writer.writeheader()
        
        # Fake second header row (Qualtrics question text)
        fake_header = {col: f"Question text for {col}" for col in columns}
        fake_header['ResponseId'] = "Response ID"
        writer.writerow(fake_header)
        
        # Write participant data
        for participant in participants:
            writer.writerow(participant)
    
    print(f"✅ Dataset saved to: {filename}")
    print(f"   Participants: {len(participants)}")
    print(f"   Columns: {len(columns)}")

def analyze_generated_data(participants):
    """Quick analysis of generated data to verify realism"""
    
    print(f"\n📊 GENERATED DATA ANALYSIS")
    print("=" * 50)
    
    # Group distribution
    group_counts = {}
    for p in participants:
        group = p['group_assignment']
        group_counts[group] = group_counts.get(group, 0) + 1
    
    print("Group distribution:")
    for group in sorted(group_counts.keys()):
        count = group_counts[group]
        pct = (count / len(participants)) * 100
        print(f"  Group {group}: {count} participants ({pct:.1f}%)")
    
    # Quality ratings by condition
    print(f"\nQuality ratings by condition:")
    
    # Collect all trial data
    all_trials = []
    for p in participants:
        for trial in range(1, 6):
            trial_data = {
                'participant': p['ResponseId'],
                'group': p['group_assignment'],
                'trial': trial,
                'prompt_id': p[f'trial_{trial}_prompt_id'],
                'modality': p[f'trial_{trial}_modality'],
                'latency_ms': p[f'trial_{trial}_latency_ms'],
                'quality_rating': p[f'quality_rating_{trial}'],
                'wait_felt': p[f'wait_felt_{trial}'],
                'would_accept': p[f'would_accept_{trial}']
            }
            all_trials.append(trial_data)
    
    # Group by condition
    condition_stats = {}
    for trial in all_trials:
        key = f"{trial['modality']}_{trial['latency_ms']}ms"
        if key not in condition_stats:
            condition_stats[key] = {'quality': [], 'wait': [], 'accept': []}
        
        condition_stats[key]['quality'].append(trial['quality_rating'])
        condition_stats[key]['wait'].append(trial['wait_felt'])
        condition_stats[key]['accept'].append(1 if trial['would_accept'] >= 5 else 0)
    
    for condition in sorted(condition_stats.keys()):
        stats = condition_stats[condition]
        n = len(stats['quality'])
        quality_mean = sum(stats['quality']) / n
        wait_mean = sum(stats['wait']) / n
        accept_rate = sum(stats['accept']) / n
        
        print(f"  {condition:<20} (n={n:3d}): Quality={quality_mean:.2f}, Wait={wait_mean:.2f}, Accept={accept_rate:.2%}")
    
    # Individual differences
    max_waits = [p['max_acceptable_wait'] for p in participants]
    preferences = [p['streaming_preference'] for p in participants]
    
    print(f"\nIndividual differences:")
    print(f"  Max acceptable wait: {sum(max_waits)/len(max_waits):.2f}s (range: {min(max_waits):.1f}-{max(max_waits):.1f}s)")
    
    pref_labels = {1: "Streaming", 2: "Non-streaming", 3: "No preference", 4: "It depends"}
    pref_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for pref in preferences:
        pref_counts[pref] += 1
    
    print(f"  Streaming preferences:")
    for pref_id, label in pref_labels.items():
        count = pref_counts[pref_id]
        pct = (count / len(participants)) * 100
        print(f"    {label}: {count} ({pct:.1f}%)")

def main():
    """Generate and save test data"""
    
    print("🧪 RESPONSE LATENCY STUDY - TEST DATA GENERATION")
    print("=" * 60)
    
    # Set seed for reproducible data
    random.seed(42)
    
    # Generate dataset
    participants = generate_full_dataset(100)
    
    # Save to file
    filename = '/workspace/test_data_qualtrics_format.csv'
    save_dataset(participants, filename)
    
    # Analyze generated data
    analyze_generated_data(participants)
    
    print(f"\n✅ TEST DATA GENERATION COMPLETE")
    print("=" * 60)
    print("Files created:")
    print(f"- {filename}")
    print("\nThis data can be used to test the R analysis script.")
    print("The data includes realistic response patterns based on:")
    print("- Latency effects on quality and acceptance")
    print("- Streaming benefits for perceived wait time")
    print("- Individual differences in tolerance and preferences")
    print("- Proper counterbalancing across all conditions")

if __name__ == "__main__":
    main()