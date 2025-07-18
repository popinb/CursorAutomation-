#!/usr/bin/env python3
"""
Comprehensive File Check for OpenAI Evals

Checks for all potential knowledge base and evaluation files that the 
OpenAI Evals implementation might need access to.
"""

import json
from pathlib import Path
import re

def find_evaluation_files():
    """Find all potential evaluation and knowledge base files."""
    
    print("🔍 COMPREHENSIVE FILE SEARCH FOR OPENAI EVALS")
    print("=" * 60)
    
    # Current working directory
    root = Path(".")
    
    # File patterns to look for
    patterns = [
        "*golden*",
        "*alpha*", 
        "*buyability*",
        "*fair*housing*",
        "*evaluation*",
        "*knowledge*",
        "*reference*",
        "*guide*"
    ]
    
    # Extensions to check
    extensions = [".json", ".yaml", ".yml", ".txt", ".md", ".docx"]
    
    found_files = {}
    
    for pattern in patterns:
        for ext in extensions:
            search_pattern = f"{pattern}{ext}"
            matches = list(root.glob(f"**/{search_pattern}"))
            if matches:
                found_files[pattern] = matches
    
    print("📁 POTENTIAL KNOWLEDGE BASE FILES:")
    print("=" * 40)
    
    for pattern, files in found_files.items():
        print(f"\n🔍 Pattern: {pattern}")
        for file in files:
            size = file.stat().st_size if file.exists() else 0
            print(f"  📄 {file} ({size} bytes)")
    
    return found_files

def check_current_implementation():
    """Check what the current OpenAI Evals implementation loads."""
    
    print("\n🔧 CURRENT OPENAI EVALS IMPLEMENTATION:")
    print("=" * 45)
    
    # Files currently loaded
    current_files = [
        "assets/golden_responses.json",
        "assets/buyability_profiles.json", 
        "assets/fair_housing_guide.json"
    ]
    
    print("Currently loaded files:")
    for file in current_files:
        filepath = Path(file)
        exists = filepath.exists()
        size = filepath.stat().st_size if exists else 0
        print(f"  {'✅' if exists else '❌'} {file} ({size} bytes)")
    
    return current_files

def check_references_in_code():
    """Check for references to additional files in the codebase."""
    
    print("\n🔍 CODE REFERENCES TO ADDITIONAL FILES:")
    print("=" * 40)
    
    # Files to search through
    search_files = [
        "llm_based_zillow_judge.py",
        "custom_gpt_llm_judge.py", 
        "zillow_judge_evaluator.py",
        "standalone_openai_evals.py"
    ]
    
    # Patterns to look for
    file_patterns = [
        r"goldenresponse[a-zA-Z]*\.(?:json|docx|txt)",
        r"alpha[a-zA-Z]*\.(?:json|docx|txt)",
        r"[a-zA-Z]*alpha[a-zA-Z]*\.(?:json|docx|txt)",
        r"evaluation[a-zA-Z]*\.(?:json|docx|txt)",
        r"reference[a-zA-Z]*\.(?:json|docx|txt)"
    ]
    
    references_found = []
    
    for file in search_files:
        filepath = Path(file)
        if filepath.exists():
            try:
                content = filepath.read_text()
                for pattern in file_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    for match in matches:
                        references_found.append((file, match))
            except Exception as e:
                print(f"  ⚠️ Error reading {file}: {e}")
    
    if references_found:
        print("References to additional files found:")
        for source_file, referenced_file in references_found:
            print(f"  📄 {source_file} → {referenced_file}")
    else:
        print("  ✅ No references to additional files found")
    
    return references_found

def check_missing_files():
    """Check for files that might be missing."""
    
    print("\n❓ POTENTIALLY MISSING FILES:")
    print("=" * 30)
    
    # Files that might be expected based on naming patterns
    potential_files = [
        "assets/goldenresponseforalpha.json",
        "assets/goldenresponsealpha.json", 
        "assets/alpha_responses.json",
        "assets/evaluation_criteria.json",
        "assets/scoring_rubric.json",
        "assets/reference_responses.json"
    ]
    
    missing_files = []
    
    for file in potential_files:
        filepath = Path(file)
        if not filepath.exists():
            missing_files.append(file)
        else:
            size = filepath.stat().st_size
            print(f"  ✅ Found: {file} ({size} bytes)")
    
    if missing_files:
        print("  Missing files (may or may not be needed):")
        for file in missing_files:
            print(f"    ❌ {file}")
    
    return missing_files

def recommend_openai_evals_updates():
    """Recommend updates to OpenAI Evals implementation."""
    
    print("\n💡 RECOMMENDATIONS FOR OPENAI EVALS:")
    print("=" * 40)
    
    # Check knowledge directory
    knowledge_dir = Path("knowledge")
    if knowledge_dir.exists():
        knowledge_files = list(knowledge_dir.glob("*.json"))
        if knowledge_files:
            print("📁 Additional files in knowledge/ directory:")
            for file in knowledge_files:
                size = file.stat().st_size
                print(f"  📄 {file} ({size} bytes)")
            print("  💡 Consider loading these in OpenAI Evals implementation")
    
    # Check for evaluation-specific files
    eval_files = []
    for pattern in ["*evaluation*", "*scoring*", "*rubric*", "*criteria*"]:
        matches = list(Path(".").glob(f"**/{pattern}.json"))
        eval_files.extend(matches)
    
    if eval_files:
        print("\n📊 Evaluation-specific files found:")
        for file in eval_files:
            size = file.stat().st_size
            print(f"  📄 {file} ({size} bytes)")
    
    return knowledge_files if knowledge_dir.exists() else [], eval_files

def main():
    """Main function to perform comprehensive file check."""
    
    # Run all checks
    found_files = find_evaluation_files()
    current_files = check_current_implementation()
    references = check_references_in_code()
    missing_files = check_missing_files()
    knowledge_files, eval_files = recommend_openai_evals_updates()
    
    # Summary
    print("\n📊 SUMMARY:")
    print("=" * 20)
    
    print(f"✅ Current files loaded: {len(current_files)}")
    print(f"🔍 File patterns found: {len(found_files)}")
    print(f"📄 Code references: {len(references)}")
    print(f"❓ Potentially missing: {len(missing_files)}")
    print(f"💡 Additional knowledge files: {len(knowledge_files)}")
    print(f"📊 Evaluation files: {len(eval_files)}")
    
    # Specific answer about goldenresponseforalpha
    print(f"\n🎯 SPECIFIC ANSWER ABOUT GOLDENRESPONSEFORALPHA:")
    print("=" * 50)
    
    goldenresponse_patterns = [
        "goldenresponseforalpha",
        "goldenresponsealpha", 
        "golden_response_alpha"
    ]
    
    found_goldenresponse = False
    for pattern in goldenresponse_patterns:
        files = list(Path(".").glob(f"**/*{pattern}*"))
        if files:
            found_goldenresponse = True
            print(f"✅ Found files matching '{pattern}':")
            for file in files:
                size = file.stat().st_size
                print(f"  📄 {file} ({size} bytes)")
    
    if not found_goldenresponse:
        print("❌ No files found matching goldenresponseforalpha patterns")
        print("💡 OpenAI Evals currently loads these golden responses:")
        print("   📄 assets/golden_responses.json (7,547 bytes)")
        print("   📄 This appears to be the main golden response file")
    
    # Final recommendation
    print(f"\n💡 FINAL RECOMMENDATION:")
    print("=" * 25)
    
    if len(knowledge_files) > 0 or len(eval_files) > 0:
        print("🔧 Consider updating OpenAI Evals to load additional files:")
        all_additional = knowledge_files + eval_files
        for file in all_additional[:5]:  # Show first 5
            print(f"   📄 {file}")
        if len(all_additional) > 5:
            print(f"   ... and {len(all_additional) - 5} more files")
    else:
        print("✅ OpenAI Evals implementation appears complete")
        print("✅ All necessary knowledge base files are loaded")

if __name__ == "__main__":
    main()