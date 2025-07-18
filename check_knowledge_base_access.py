#!/usr/bin/env python3
"""
Check OpenAI Evals Knowledge Base Access

This script verifies that the OpenAI Evals implementation has access to
and is properly loading the three knowledge base files.
"""

import os
import json
from pathlib import Path
from standalone_openai_evals import ZillowAffordabilityEval

def main():
    """Check knowledge base access in OpenAI Evals implementation."""
    
    print("🔍 CHECKING OPENAI EVALS KNOWLEDGE BASE ACCESS")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = ZillowAffordabilityEval(
        model="gpt-4",
        api_key=os.getenv("OPENAI_API_KEY", "test-key"),
        base_url=os.getenv("OPENAI_BASE_URL", "test-url"),
        seed=42,
        temperature=0,
        max_tokens=4000
    )
    
    print("📁 Knowledge Base Files Status:")
    print("=" * 40)
    
    # Check each knowledge base file
    knowledge_files = [
        ("Golden Responses", "golden_responses.json", evaluator.golden_responses),
        ("Buyability Profiles", "buyability_profiles.json", evaluator.buyability_profiles),
        ("Fair Housing Guide", "fair_housing_guide.json", evaluator.fair_housing_guide)
    ]
    
    all_loaded = True
    
    for name, filename, data in knowledge_files:
        filepath = Path("assets") / filename
        
        print(f"\n📋 {name}:")
        print(f"  File: {filepath}")
        print(f"  Exists: {'✅' if filepath.exists() else '❌'}")
        print(f"  Loaded: {'✅' if data else '❌'}")
        
        if filepath.exists():
            file_size = filepath.stat().st_size
            print(f"  Size: {file_size} bytes")
            
            if data:
                if isinstance(data, dict):
                    print(f"  Keys: {len(data)} items")
                    if data:
                        print(f"  Sample keys: {list(data.keys())[:3]}...")
                elif isinstance(data, list):
                    print(f"  Items: {len(data)} entries")
                print(f"  Status: ✅ Successfully loaded")
            else:
                print(f"  Status: ❌ Failed to load")
                all_loaded = False
        else:
            print(f"  Status: ❌ File not found")
            all_loaded = False
    
    print(f"\n🔧 Knowledge Base Integration Test:")
    print("=" * 40)
    
    # Test knowledge base formatting
    formatted_kb = evaluator._format_knowledge_base()
    
    print(f"Knowledge Base Length: {len(formatted_kb)} characters")
    print(f"Contains Golden Responses: {'golden_responses' in formatted_kb.lower()}")
    print(f"Contains Buyability Profiles: {'buyability' in formatted_kb.lower()}")
    print(f"Contains Fair Housing: {'fair housing' in formatted_kb.lower()}")
    
    if len(formatted_kb) > 100:
        print(f"\n📝 Knowledge Base Preview (first 200 chars):")
        print("-" * 40)
        print(formatted_kb[:200] + "...")
    
    # Test if knowledge base is included in evaluation prompt
    print(f"\n🔍 Prompt Integration Test:")
    print("=" * 40)
    
    test_prompt = evaluator._create_evaluation_prompt(
        question="test question",
        candidate_answer="test answer", 
        user_profile={"test": "profile"}
    )
    
    kb_in_prompt = len([line for line in test_prompt.split('\n') 
                       if any(kb_term in line.lower() 
                             for kb_term in ['golden', 'buyability', 'fair housing'])])
    
    print(f"Knowledge base sections in prompt: {kb_in_prompt}")
    print(f"Prompt length: {len(test_prompt)} characters")
    print(f"Knowledge base included: {'✅' if 'KNOWLEDGE BASE:' in test_prompt else '❌'}")
    
    # Overall status
    print(f"\n📊 OVERALL STATUS:")
    print("=" * 40)
    
    if all_loaded and len(formatted_kb) > 100:
        print("✅ All knowledge base files loaded successfully")
        print("✅ Knowledge base integrated into evaluation prompts")
        print("✅ OpenAI Evals implementation has full access")
    else:
        print("❌ Knowledge base access issues detected")
        if not all_loaded:
            print("   - Some files failed to load")
        if len(formatted_kb) < 100:
            print("   - Knowledge base formatting issues")
    
    # Detailed file contents check
    print(f"\n📋 DETAILED FILE CONTENTS:")
    print("=" * 40)
    
    for name, filename, data in knowledge_files:
        if data:
            print(f"\n{name}:")
            if isinstance(data, dict):
                for key, value in list(data.items())[:2]:  # Show first 2 items
                    if isinstance(value, str) and len(value) > 100:
                        preview = value[:100] + "..."
                    else:
                        preview = str(value)
                    print(f"  {key}: {preview}")
            elif isinstance(data, list):
                for i, item in enumerate(data[:2]):  # Show first 2 items
                    print(f"  Item {i+1}: {str(item)[:100]}...")


if __name__ == "__main__":
    main()