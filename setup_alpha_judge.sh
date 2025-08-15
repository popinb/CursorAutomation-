#!/bin/bash

echo "Setting up Alpha Judge - Draft 2 (Replica, No Pairwise)"
echo "========================================================"

# Check if Python 3 is available
if command -v python3 &> /dev/null; then
    echo "✓ Python 3 found: $(python3 --version)"
else
    echo "✗ Python 3 not found. Please install Python 3.8+ first."
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv alpha_judge_env

# Activate virtual environment
echo "Activating virtual environment..."
source alpha_judge_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements_alpha_judge.txt

echo ""
echo "Setup complete! To use Alpha Judge:"
echo ""
echo "1. Activate the virtual environment:"
echo "   source alpha_judge_env/bin/activate"
echo ""
echo "2. Set your OpenAI API key:"
echo "   export OPENAI_API_KEY='your_api_key_here'"
echo ""
echo "3. Run the full version (with OpenAI o3):"
echo "   python alpha_judge_local.py"
echo ""
echo "4. Or run the simple version (heuristic scoring):"
echo "   python alpha_judge_simple.py"
echo ""
echo "Note: The simple version works without OpenAI but uses heuristic scoring."
echo "      For full LLM-as-a-Judge evaluation, use the full version with OpenAI o3."