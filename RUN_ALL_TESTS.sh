#!/bin/bash

# Complete Test Suite for Corrected Databricks LLM Implementation
# Run this script to execute all test phases

echo "================================================================================"
echo "?? CORRECTED IMPLEMENTATION TEST SUITE"
echo "================================================================================"
echo ""
echo "Testing: Databricks Foundation Model Serving Endpoints Integration"
echo "Date: $(date)"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track results
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Function to run a test
run_test() {
    local test_file=$1
    local test_name=$2
    
    echo "================================================================================"
    echo "Running: $test_name"
    echo "File: $test_file"
    echo "================================================================================"
    
    if python3 "$test_file"; then
        echo -e "${GREEN}? PASSED${NC}: $test_name"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo -e "${RED}? FAILED${NC}: $test_name"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo ""
}

# Run all test phases
echo "Starting test execution..."
echo ""

run_test "test_corrected_phase1_data.py" "Phase 1: Core Data Loading (Hardcoded JSON)"
run_test "test_corrected_phase2_serving.py" "Phase 2: Databricks Serving Endpoints"
run_test "test_corrected_phase3_evaluator.py" "Phase 3: Dual-Mode LLM Evaluator"
run_test "test_corrected_phase4_e2e.py" "Phase 4: End-to-End Workflow"

# Summary
echo "================================================================================"
echo "?? TEST SUITE SUMMARY"
echo "================================================================================"
echo ""
echo "Total Tests: $TOTAL_TESTS"
echo -e "${GREEN}Passed: $PASSED_TESTS${NC}"
echo -e "${RED}Failed: $FAILED_TESTS${NC}"
echo ""

if [ $FAILED_TESTS -eq 0 ]; then
    PASS_RATE=100
    echo -e "${GREEN}Pass Rate: ${PASS_RATE}%${NC}"
    echo ""
    echo "================================================================================"
    echo -e "${GREEN}?? ALL TESTS PASSED - PRODUCTION READY!${NC}"
    echo "================================================================================"
    echo ""
    echo "? The corrected implementation is ready for deployment"
    echo "? Uses Databricks Foundation Model Serving Endpoints"
    echo "? Workspace token authentication (no API key needed)"
    echo "? Auto-discovers Claude Sonnet endpoint"
    echo "? OpenAI-compatible response format"
    echo ""
    echo "?? Main File: CORRECT_Notebook_With_Databricks_Serving.py"
    echo "?? Documentation: See QUICK_START_CORRECTED.md"
    echo ""
    exit 0
else
    PASS_RATE=$(( (PASSED_TESTS * 100) / TOTAL_TESTS ))
    echo -e "${YELLOW}Pass Rate: ${PASS_RATE}%${NC}"
    echo ""
    echo "================================================================================"
    echo -e "${RED}? SOME TESTS FAILED${NC}"
    echo "================================================================================"
    echo ""
    echo "Please review the test output above for details."
    echo ""
    exit 1
fi
