#!/usr/bin/env python3
"""
MASTER TEST RUNNER
Executes all test phases and generates comprehensive report
"""

import subprocess
import sys
import time
from datetime import datetime

print("=" * 100)
print("?? MASTER TEST SUITE - COMPREHENSIVE END-TO-END TESTING")
print("=" * 100)
print(f"Test Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 100)

# Track overall results
overall_start = time.time()
phase_results = {}

# ============================================================
# PHASE 0: Setup Test Data
# ============================================================

print("\n" + "=" * 100)
print("PHASE 0: SETUP TEST DATA")
print("=" * 100)

try:
    result = subprocess.run(
        ['python3', 'test_data_hardcoded.py'],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    if result.returncode == 0:
        print("? Test data created successfully")
        phase_results['Phase 0: Setup'] = 'PASS'
    else:
        print(f"? Test data creation failed")
        print(result.stderr)
        phase_results['Phase 0: Setup'] = 'FAIL'
        sys.exit(1)
except Exception as e:
    print(f"? Error in Phase 0: {e}")
    phase_results['Phase 0: Setup'] = 'ERROR'
    sys.exit(1)

# ============================================================
# PHASE 1: Core Functions
# ============================================================

print("\n" + "=" * 100)
print("PHASE 1: CORE FUNCTIONS TESTING")
print("=" * 100)

phase1_start = time.time()

try:
    result = subprocess.run(
        ['python3', 'test_phase1_core_functions.py'],
        capture_output=True,
        text=True,
        timeout=60
    )
    
    phase1_time = time.time() - phase1_start
    
    if result.returncode == 0:
        # Extract test results
        output = result.stdout
        if "PHASE 1 COMPLETE" in output and "16/16" in output:
            print(f"? PHASE 1 PASSED (16/16 tests in {phase1_time:.2f}s)")
            phase_results['Phase 1: Core Functions'] = 'PASS (16/16)'
            
            # Show summary
            lines = output.split('\n')
            for line in lines:
                if 'Test Results:' in line:
                    idx = lines.index(line)
                    for i in range(idx, min(idx+20, len(lines))):
                        if lines[i].strip().startswith('?') or lines[i].strip().startswith('?'):
                            print(f"   {lines[i].strip()}")
        else:
            print(f"??  PHASE 1 PARTIAL: Check output")
            phase_results['Phase 1: Core Functions'] = 'PARTIAL'
    else:
        print(f"? PHASE 1 FAILED")
        print(result.stderr)
        phase_results['Phase 1: Core Functions'] = 'FAIL'
except Exception as e:
    print(f"? Error in Phase 1: {e}")
    phase_results['Phase 1: Core Functions'] = 'ERROR'

# ============================================================
# PHASE 2: Widgets & UI
# ============================================================

print("\n" + "=" * 100)
print("PHASE 2: WIDGETS & UI WORKFLOW TESTING")
print("=" * 100)

phase2_start = time.time()

try:
    result = subprocess.run(
        ['python3', 'test_phase2_widgets_ui.py'],
        capture_output=True,
        text=True,
        timeout=60
    )
    
    phase2_time = time.time() - phase2_start
    
    if result.returncode == 0:
        output = result.stdout
        if "PHASE 2 COMPLETE" in output and "14/14" in output:
            print(f"? PHASE 2 PASSED (14/14 tests in {phase2_time:.2f}s)")
            phase_results['Phase 2: Widgets & UI'] = 'PASS (14/14)'
            
            # Show summary
            lines = output.split('\n')
            for line in lines:
                if 'Test Results:' in line:
                    idx = lines.index(line)
                    for i in range(idx, min(idx+16, len(lines))):
                        if lines[i].strip().startswith('?') or lines[i].strip().startswith('?'):
                            print(f"   {lines[i].strip()}")
        else:
            print(f"??  PHASE 2 PARTIAL: Check output")
            phase_results['Phase 2: Widgets & UI'] = 'PARTIAL'
    else:
        print(f"? PHASE 2 FAILED")
        print(result.stderr)
        phase_results['Phase 2: Widgets & UI'] = 'FAIL'
except Exception as e:
    print(f"? Error in Phase 2: {e}")
    phase_results['Phase 2: Widgets & UI'] = 'ERROR'

# ============================================================
# PHASE 3: Evaluation Workflow
# ============================================================

print("\n" + "=" * 100)
print("PHASE 3: COMPLETE EVALUATION WORKFLOW TESTING")
print("=" * 100)

phase3_start = time.time()

try:
    result = subprocess.run(
        ['python3', 'test_phase3_evaluation_workflow.py'],
        capture_output=True,
        text=True,
        timeout=120
    )
    
    phase3_time = time.time() - phase3_start
    
    if result.returncode == 0:
        output = result.stdout
        if "PHASE 3 COMPLETE" in output and "15/15" in output:
            print(f"? PHASE 3 PASSED (15/15 tests in {phase3_time:.2f}s)")
            phase_results['Phase 3: Evaluation Workflow'] = 'PASS (15/15)'
            
            # Show evaluation summary
            lines = output.split('\n')
            for line in lines:
                if 'EVALUATION SUMMARY:' in line:
                    idx = lines.index(line)
                    for i in range(idx, min(idx+6, len(lines))):
                        print(f"   {lines[i].strip()}")
                    break
        else:
            print(f"??  PHASE 3 PARTIAL: Check output")
            phase_results['Phase 3: Evaluation Workflow'] = 'PARTIAL'
    else:
        print(f"? PHASE 3 FAILED")
        print(result.stderr)
        phase_results['Phase 3: Evaluation Workflow'] = 'FAIL'
except Exception as e:
    print(f"? Error in Phase 3: {e}")
    phase_results['Phase 3: Evaluation Workflow'] = 'ERROR'

# ============================================================
# FINAL SUMMARY
# ============================================================

overall_time = time.time() - overall_start

print("\n" + "=" * 100)
print("?? FINAL TEST SUMMARY")
print("=" * 100)

print(f"\n??  Total Execution Time: {overall_time:.2f} seconds")
print(f"?? Test Completion: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

print("\n?? Phase Results:")
for phase_name, status in phase_results.items():
    if 'PASS' in status:
        icon = "?"
    elif 'PARTIAL' in status:
        icon = "??"
    elif 'FAIL' in status or 'ERROR' in status:
        icon = "?"
    else:
        icon = "??"
    
    print(f"   {icon} {phase_name}: {status}")

# Calculate overall status
all_passed = all('PASS' in status for status in phase_results.values())
any_failed = any('FAIL' in status or 'ERROR' in status for status in phase_results.values())

print("\n" + "=" * 100)

if all_passed:
    print("?? ALL PHASES PASSED - SYSTEM IS PRODUCTION READY!")
    print("=" * 100)
    print("\n? Summary:")
    print("   ? Phase 0: Test data setup - PASS")
    print("   ? Phase 1: Core functions (16 tests) - PASS")
    print("   ? Phase 2: Widgets & UI (14 tests) - PASS")
    print("   ? Phase 3: Evaluation workflow (15 tests) - PASS")
    print(f"   ? Total tests: 45")
    print(f"   ? All passed: ?")
    print(f"   ? Total time: {overall_time:.2f}s")
    print("\n?? RECOMMENDATION: Deploy to Databricks immediately!")
    exit_code = 0
elif any_failed:
    print("? SOME PHASES FAILED - REVIEW REQUIRED")
    print("=" * 100)
    print("\n??  Please review failed phases above")
    exit_code = 1
else:
    print("?? PARTIAL SUCCESS - REVIEW RECOMMENDED")
    print("=" * 100)
    exit_code = 2

print("\n" + "=" * 100)
print("?? MASTER TEST SUITE COMPLETE")
print("=" * 100)

sys.exit(exit_code)
