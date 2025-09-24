#!/usr/bin/env python3
"""
Final Test Report - Universal Evaluation Template System
======================================================

This script provides a comprehensive test report and identifies all failure points
in the Universal Evaluation Template System.
"""

import os
import sys
import json
import tempfile
import traceback
from pathlib import Path


class FinalTestReport:
    """Final comprehensive test report generator."""
    
    def __init__(self):
        self.results = []
        self.failure_points = []
        self.recommendations = []
    
    def test_core_system(self):
        """Test core system functionality."""
        print("🔬 Testing Core System Functionality")
        print("=" * 50)
        
        # Test 1: File structure
        self.test_file_structure()
        
        # Test 2: Core modules
        self.test_core_modules()
        
        # Test 3: Configuration system
        self.test_configuration_system()
        
        # Test 4: Metric templates
        self.test_metric_templates()
        
        # Test 5: Data handling
        self.test_data_handling()
        
        # Test 6: Error handling
        self.test_error_handling()
    
    def test_file_structure(self):
        """Test file structure and organization."""
        print("\n📁 Testing File Structure")
        print("-" * 30)
        
        required_files = [
            "template_evaluation_system.py",
            "metric_templates.py",
            "generate_config.py",
            "quick_start.py",
            "config_template.yaml",
            "requirements.txt",
            "README.md"
        ]
        
        required_dirs = [
            "data",
            "configs", 
            "results",
            "examples"
        ]
        
        missing_files = []
        missing_dirs = []
        
        for file in required_files:
            if not os.path.exists(file):
                missing_files.append(file)
            else:
                print(f"✅ {file}")
        
        for directory in required_dirs:
            if not os.path.exists(directory):
                missing_dirs.append(directory)
            else:
                print(f"✅ {directory}/")
        
        if missing_files:
            self.failure_points.append({
                "category": "File Structure",
                "issue": "Missing required files",
                "details": missing_files,
                "severity": "High"
            })
            print(f"❌ Missing files: {missing_files}")
        
        if missing_dirs:
            self.failure_points.append({
                "category": "File Structure", 
                "issue": "Missing required directories",
                "details": missing_dirs,
                "severity": "Medium"
            })
            print(f"❌ Missing directories: {missing_dirs}")
    
    def test_core_modules(self):
        """Test core module imports and functionality."""
        print("\n🐍 Testing Core Modules")
        print("-" * 30)
        
        # Test Python standard library imports
        try:
            import json
            import os
            import sys
            import tempfile
            from pathlib import Path
            print("✅ Standard library imports")
        except Exception as e:
            self.failure_points.append({
                "category": "Core Modules",
                "issue": "Standard library import failure",
                "details": str(e),
                "severity": "Critical"
            })
            print(f"❌ Standard library imports: {e}")
        
        # Test YAML support
        try:
            import yaml
            print("✅ YAML support available")
        except ImportError:
            self.failure_points.append({
                "category": "Core Modules",
                "issue": "YAML support missing",
                "details": "yaml module not available",
                "severity": "High"
            })
            print("❌ YAML support missing")
        
        # Test our custom modules
        try:
            sys.path.insert(0, os.getcwd())
            from metric_templates import METRIC_TEMPLATES, DOMAIN_TEMPLATES
            print("✅ Metric templates module")
        except Exception as e:
            self.failure_points.append({
                "category": "Core Modules",
                "issue": "Metric templates module failure",
                "details": str(e),
                "severity": "High"
            })
            print(f"❌ Metric templates module: {e}")
        
        # Test external dependencies
        external_deps = [
            ("pandas", "Data manipulation"),
            ("httpx", "HTTP client"),
            ("pydantic", "Data validation"),
            ("mlflow", "Experiment tracking")
        ]
        
        available_deps = 0
        for dep, description in external_deps:
            try:
                __import__(dep)
                print(f"✅ {dep} ({description})")
                available_deps += 1
            except ImportError:
                self.failure_points.append({
                    "category": "Dependencies",
                    "issue": f"Missing dependency: {dep}",
                    "details": f"{description} not available",
                    "severity": "Medium"
                })
                print(f"❌ {dep} ({description}) - Not available")
        
        print(f"\n📊 Dependencies: {available_deps}/{len(external_deps)} available")
    
    def test_configuration_system(self):
        """Test configuration system."""
        print("\n⚙️ Testing Configuration System")
        print("-" * 30)
        
        # Test configuration template
        if os.path.exists("config_template.yaml"):
            try:
                with open("config_template.yaml", 'r') as f:
                    import yaml
                    config = yaml.safe_load(f)
                
                required_sections = ["dataset", "models", "api_keys", "evaluation"]
                missing_sections = []
                
                for section in required_sections:
                    if section in config:
                        print(f"✅ {section} section")
                    else:
                        missing_sections.append(section)
                        print(f"❌ {section} section missing")
                
                if missing_sections:
                    self.failure_points.append({
                        "category": "Configuration",
                        "issue": "Missing configuration sections",
                        "details": missing_sections,
                        "severity": "High"
                    })
            except Exception as e:
                self.failure_points.append({
                    "category": "Configuration",
                    "issue": "Configuration template parsing error",
                    "details": str(e),
                    "severity": "High"
                })
                print(f"❌ Configuration template error: {e}")
        else:
            self.failure_points.append({
                "category": "Configuration",
                "issue": "Configuration template missing",
                "details": "config_template.yaml not found",
                "severity": "High"
            })
            print("❌ Configuration template missing")
    
    def test_metric_templates(self):
        """Test metric templates system."""
        print("\n📏 Testing Metric Templates")
        print("-" * 30)
        
        try:
            from metric_templates import METRIC_TEMPLATES, DOMAIN_TEMPLATES
            
            # Test general metrics
            expected_metrics = ["accuracy", "relevance", "helpfulness", "safety"]
            available_metrics = list(METRIC_TEMPLATES.keys())
            
            missing_metrics = [m for m in expected_metrics if m not in available_metrics]
            if missing_metrics:
                self.failure_points.append({
                    "category": "Metric Templates",
                    "issue": "Missing expected metrics",
                    "details": missing_metrics,
                    "severity": "Medium"
                })
                print(f"❌ Missing metrics: {missing_metrics}")
            else:
                print("✅ All expected metrics available")
            
            # Test domain templates
            expected_domains = ["financial_advice", "medical_advice"]
            available_domains = list(DOMAIN_TEMPLATES.keys())
            
            missing_domains = [d for d in expected_domains if d not in available_domains]
            if missing_domains:
                self.failure_points.append({
                    "category": "Metric Templates",
                    "issue": "Missing expected domains",
                    "details": missing_domains,
                    "severity": "Low"
                })
                print(f"❌ Missing domains: {missing_domains}")
            else:
                print("✅ All expected domains available")
            
            print(f"📊 Available metrics: {len(available_metrics)}")
            print(f"📊 Available domains: {len(available_domains)}")
            
        except Exception as e:
            self.failure_points.append({
                "category": "Metric Templates",
                "issue": "Metric templates system failure",
                "details": str(e),
                "severity": "High"
            })
            print(f"❌ Metric templates system error: {e}")
    
    def test_data_handling(self):
        """Test data handling capabilities."""
        print("\n📊 Testing Data Handling")
        print("-" * 30)
        
        # Test CSV handling
        try:
            # Create test CSV data
            test_csv = "prompt,response,ground_truth\nWhat is 2+2?,2+2 equals 4,4\nHow to cook?,Boil water,Boil"
            csv_path = "test_data.csv"
            
            with open(csv_path, 'w') as f:
                f.write(test_csv)
            
            # Test reading
            with open(csv_path, 'r') as f:
                lines = f.readlines()
            
            assert len(lines) == 3  # Header + 2 data rows
            assert "prompt,response,ground_truth" in lines[0]
            
            print("✅ CSV handling")
            
            # Clean up
            os.remove(csv_path)
            
        except Exception as e:
            self.failure_points.append({
                "category": "Data Handling",
                "issue": "CSV handling failure",
                "details": str(e),
                "severity": "Medium"
            })
            print(f"❌ CSV handling error: {e}")
        
        # Test JSON handling
        try:
            test_json = {"test": "data", "numbers": [1, 2, 3]}
            json_path = "test_data.json"
            
            with open(json_path, 'w') as f:
                json.dump(test_json, f)
            
            with open(json_path, 'r') as f:
                loaded = json.load(f)
            
            assert loaded["test"] == "data"
            assert loaded["numbers"] == [1, 2, 3]
            
            print("✅ JSON handling")
            
            # Clean up
            os.remove(json_path)
            
        except Exception as e:
            self.failure_points.append({
                "category": "Data Handling",
                "issue": "JSON handling failure",
                "details": str(e),
                "severity": "Medium"
            })
            print(f"❌ JSON handling error: {e}")
    
    def test_error_handling(self):
        """Test error handling capabilities."""
        print("\n🛡️ Testing Error Handling")
        print("-" * 30)
        
        # Test file not found handling
        try:
            with open("nonexistent_file.txt", 'r') as f:
                content = f.read()
            print("❌ File not found not handled properly")
        except FileNotFoundError:
            print("✅ File not found handled properly")
        except Exception as e:
            self.failure_points.append({
                "category": "Error Handling",
                "issue": "Unexpected file not found handling",
                "details": str(e),
                "severity": "Low"
            })
            print(f"❌ Unexpected file not found handling: {e}")
        
        # Test JSON parsing error handling
        try:
            json.loads("invalid json {")
            print("❌ JSON parsing error not handled properly")
        except json.JSONDecodeError:
            print("✅ JSON parsing error handled properly")
        except Exception as e:
            self.failure_points.append({
                "category": "Error Handling",
                "issue": "Unexpected JSON parsing error handling",
                "details": str(e),
                "severity": "Low"
            })
            print(f"❌ Unexpected JSON parsing error handling: {e}")
    
    def generate_recommendations(self):
        """Generate recommendations based on failure points."""
        print("\n💡 Generating Recommendations")
        print("=" * 50)
        
        # Categorize failure points
        categories = {}
        for failure in self.failure_points:
            category = failure["category"]
            if category not in categories:
                categories[category] = []
            categories[category].append(failure)
        
        # Generate recommendations for each category
        for category, failures in categories.items():
            print(f"\n{category} Issues:")
            print("-" * 20)
            
            for failure in failures:
                print(f"• {failure['issue']} (Severity: {failure['severity']})")
                
                if category == "Dependencies":
                    self.recommendations.append({
                        "priority": "High" if failure['severity'] == "High" else "Medium",
                        "action": f"Install {failure['details'].split(':')[0]}",
                        "command": f"pip install {failure['details'].split(':')[0]}",
                        "description": failure['details']
                    })
                elif category == "File Structure":
                    self.recommendations.append({
                        "priority": "High",
                        "action": "Create missing files/directories",
                        "command": "mkdir -p data configs results examples",
                        "description": f"Missing: {', '.join(failure['details'])}"
                    })
                elif category == "Configuration":
                    self.recommendations.append({
                        "priority": "High",
                        "action": "Fix configuration system",
                        "command": "Check config_template.yaml syntax",
                        "description": failure['issue']
                    })
                elif category == "Core Modules":
                    self.recommendations.append({
                        "priority": "Critical",
                        "action": "Fix core module issues",
                        "command": "Check Python installation and imports",
                        "description": failure['issue']
                    })
    
    def generate_final_report(self):
        """Generate the final comprehensive report."""
        print("\n" + "=" * 70)
        print("📋 FINAL TEST REPORT - Universal Evaluation Template System")
        print("=" * 70)
        
        # Summary statistics
        total_failures = len(self.failure_points)
        critical_failures = len([f for f in self.failure_points if f['severity'] == 'Critical'])
        high_failures = len([f for f in self.failure_points if f['severity'] == 'High'])
        medium_failures = len([f for f in self.failure_points if f['severity'] == 'Medium'])
        low_failures = len([f for f in self.failure_points if f['severity'] == 'Low'])
        
        print(f"\n📊 FAILURE SUMMARY:")
        print(f"Total Issues: {total_failures}")
        print(f"Critical: {critical_failures} 🔴")
        print(f"High: {high_failures} 🟠")
        print(f"Medium: {medium_failures} 🟡")
        print(f"Low: {low_failures} 🟢")
        
        # System status
        if critical_failures > 0:
            status = "🔴 CRITICAL ISSUES - System not functional"
        elif high_failures > 0:
            status = "🟠 HIGH ISSUES - System partially functional"
        elif medium_failures > 0:
            status = "🟡 MEDIUM ISSUES - System functional with limitations"
        else:
            status = "🟢 SYSTEM READY - All tests passed"
        
        print(f"\n🎯 SYSTEM STATUS: {status}")
        
        # Detailed failure report
        if self.failure_points:
            print(f"\n❌ DETAILED FAILURE REPORT:")
            print("-" * 40)
            
            for i, failure in enumerate(self.failure_points, 1):
                print(f"\n{i}. {failure['issue']}")
                print(f"   Category: {failure['category']}")
                print(f"   Severity: {failure['severity']}")
                print(f"   Details: {failure['details']}")
        
        # Recommendations
        if self.recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            print("-" * 20)
            
            # Sort by priority
            priority_order = {"Critical": 1, "High": 2, "Medium": 3, "Low": 4}
            sorted_recs = sorted(self.recommendations, key=lambda x: priority_order.get(x['priority'], 5))
            
            for i, rec in enumerate(sorted_recs, 1):
                print(f"\n{i}. {rec['action']} ({rec['priority']} Priority)")
                print(f"   Command: {rec['command']}")
                print(f"   Description: {rec['description']}")
        
        # Next steps
        print(f"\n🚀 NEXT STEPS:")
        print("-" * 15)
        
        if critical_failures > 0:
            print("1. Fix critical issues first")
            print("2. Ensure Python environment is properly set up")
            print("3. Install required dependencies")
        elif high_failures > 0:
            print("1. Address high-priority issues")
            print("2. Install missing dependencies")
            print("3. Test with example data")
        else:
            print("1. Set up your OpenAI API key")
            print("2. Run: python3 quick_start.py")
            print("3. Test with your own data")
        
        print("\n4. View MLflow dashboard for results")
        print("5. Customize metrics for your use case")
        
        # File structure
        print(f"\n📁 CURRENT FILE STRUCTURE:")
        print("-" * 30)
        for root, dirs, files in os.walk("."):
            level = root.replace(".", "").count(os.sep)
            indent = " " * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = " " * 2 * (level + 1)
            for file in files[:5]:  # Show first 5 files
                print(f"{subindent}{file}")
            if len(files) > 5:
                print(f"{subindent}... and {len(files) - 5} more files")
    
    def run_complete_test(self):
        """Run the complete test suite."""
        print("🧪 Universal Evaluation Template System - Final Test Report")
        print("=" * 70)
        
        # Run all tests
        self.test_core_system()
        
        # Generate recommendations
        self.generate_recommendations()
        
        # Generate final report
        self.generate_final_report()
        
        print("\n🎉 Test report completed!")
        print("Review the recommendations above to address any issues.")


def main():
    """Main function."""
    test_report = FinalTestReport()
    test_report.run_complete_test()


if __name__ == "__main__":
    main()