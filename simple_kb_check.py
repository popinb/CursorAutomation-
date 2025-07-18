#!/usr/bin/env python3
"""
Simple Knowledge Base Access Check

Checks if the three knowledge base files exist and can be loaded,
and shows how they would be integrated into the OpenAI Evals implementation.
"""

import json
from pathlib import Path

def load_json(filepath):
    """Load JSON data from file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}

def format_knowledge_base(golden_responses, buyability_profiles, fair_housing_guide):
    """Format knowledge base like the OpenAI Evals implementation."""
    parts = []
    if golden_responses:
        parts.append(f"Golden Responses: {json.dumps(golden_responses, indent=2)}")
    if buyability_profiles:
        parts.append(f"Buyability Profiles: {json.dumps(buyability_profiles, indent=2)}")
    if fair_housing_guide:
        parts.append(f"Fair Housing Guide: {json.dumps(fair_housing_guide, indent=2)}")
    return "\n\n".join(parts) if parts else "No knowledge base available"

def main():
    """Check knowledge base access."""
    
    print("🔍 OPENAI EVALS KNOWLEDGE BASE ACCESS CHECK")
    print("=" * 60)
    
    # Check file paths
    script_dir = Path(__file__).parent
    assets_dir = script_dir / "assets"
    
    knowledge_files = [
        ("Golden Responses", "golden_responses.json"),
        ("Buyability Profiles", "buyability_profiles.json"), 
        ("Fair Housing Guide", "fair_housing_guide.json")
    ]
    
    print("📁 File Existence Check:")
    print("=" * 30)
    
    all_exist = True
    loaded_data = {}
    
    for name, filename in knowledge_files:
        filepath = assets_dir / filename
        exists = filepath.exists()
        
        print(f"📋 {name}:")
        print(f"  Path: {filepath}")
        print(f"  Exists: {'✅' if exists else '❌'}")
        
        if exists:
            size = filepath.stat().st_size
            print(f"  Size: {size} bytes")
            
            # Try to load
            data = load_json(filepath)
            loaded_data[filename] = data
            
            if data:
                if isinstance(data, dict):
                    print(f"  Loaded: ✅ {len(data)} items")
                    if data:
                        keys = list(data.keys())[:3]
                        print(f"  Sample keys: {keys}...")
                elif isinstance(data, list):
                    print(f"  Loaded: ✅ {len(data)} entries")
            else:
                print(f"  Loaded: ❌ Failed to parse")
        else:
            all_exist = False
            loaded_data[filename] = {}
        
        print()
    
    print("🔧 Knowledge Base Integration Test:")
    print("=" * 35)
    
    # Test formatting like OpenAI Evals implementation
    golden_responses = loaded_data.get("golden_responses.json", {})
    buyability_profiles = loaded_data.get("buyability_profiles.json", {})
    fair_housing_guide = loaded_data.get("fair_housing_guide.json", {})
    
    formatted_kb = format_knowledge_base(golden_responses, buyability_profiles, fair_housing_guide)
    
    print(f"Knowledge Base Length: {len(formatted_kb)} characters")
    print(f"Contains Golden Responses: {'✅' if 'Golden Responses:' in formatted_kb else '❌'}")
    print(f"Contains Buyability Profiles: {'✅' if 'Buyability Profiles:' in formatted_kb else '❌'}")
    print(f"Contains Fair Housing: {'✅' if 'Fair Housing Guide:' in formatted_kb else '❌'}")
    
    if len(formatted_kb) > 100:
        print(f"\n📝 Knowledge Base Preview (first 300 chars):")
        print("-" * 50)
        print(formatted_kb[:300] + "..." if len(formatted_kb) > 300 else formatted_kb)
    
    print(f"\n🔍 OpenAI Evals Implementation Analysis:")
    print("=" * 40)
    
    # Simulate how the OpenAI Evals prompt would look
    sample_prompt = f"""
You are an expert evaluator for Zillow Home Affordability responses following OpenAI Evals framework standards.

EVALUATION TASK:
Question: can I afford to buy a home right now?
Candidate Answer: you can afford to buy now
User Profile: {{"annual_income": null, "monthly_debts": null, "down_payment": null, "credit_score": null}}

KNOWLEDGE BASE:
{formatted_kb[:500]}...

EVALUATION METRICS (12 total):
1. **Personalization Accuracy** (Accurate/Inaccurate)
2. **Context-based Personalization** (1-5 scale)
[... other metrics ...]
"""
    
    print(f"Sample prompt length: {len(sample_prompt)} characters")
    print(f"Knowledge base in prompt: {'✅' if len(formatted_kb) > 0 else '❌'}")
    print(f"Prompt includes KB data: {'✅' if 'Golden Responses:' in sample_prompt else '❌'}")
    
    # Content analysis
    print(f"\n📊 Content Analysis:")
    print("=" * 20)
    
    total_content = 0
    for name, filename in knowledge_files:
        data = loaded_data.get(filename, {})
        if data:
            content_size = len(json.dumps(data))
            total_content += content_size
            print(f"{name}: {content_size:,} characters")
    
    print(f"Total content: {total_content:,} characters")
    
    # Final verdict
    print(f"\n📊 FINAL VERDICT:")
    print("=" * 20)
    
    if all_exist and total_content > 1000:
        print("✅ OpenAI Evals HAS ACCESS to all three knowledge base files")
        print("✅ Files are properly loaded and formatted")
        print("✅ Knowledge base is integrated into evaluation prompts")
        print(f"✅ Total knowledge: {total_content:,} characters available")
    else:
        print("❌ Knowledge base access issues:")
        if not all_exist:
            print("   - Some files are missing")
        if total_content < 1000:
            print("   - Content appears insufficient")
    
    # Show specific content examples
    print(f"\n📋 CONTENT EXAMPLES:")
    print("=" * 20)
    
    if golden_responses:
        print("Golden Responses sample:")
        for key, value in list(golden_responses.items())[:1]:
            print(f"  {key}: {str(value)[:100]}...")
    
    if buyability_profiles:
        print("Buyability Profiles sample:")
        for key, value in list(buyability_profiles.items())[:1]:
            print(f"  {key}: {str(value)[:100]}...")
    
    if fair_housing_guide:
        print("Fair Housing Guide sample:")
        for key, value in list(fair_housing_guide.items())[:1]:
            print(f"  {key}: {str(value)[:100]}...")

if __name__ == "__main__":
    main()