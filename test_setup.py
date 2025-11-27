#!/usr/bin/env python3
"""
Test script to validate LLM Judge Application setup

This script tests the basic functionality without requiring API calls.
"""

import os
import sys
import json
from pathlib import Path


def test_imports():
    """Test that all required modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        from llm_judge_app import (
            LLMJudgeApp, 
            LLMJudgeOAIEval, 
            EvaluationMetric, 
            MetricType, 
            JudgeConfig
        )
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_configuration_parsing():
    """Test configuration file parsing."""
    print("\n🔧 Testing configuration parsing...")
    
    try:
        # Test programmatic configuration
        from llm_judge_app import EvaluationMetric, MetricType, JudgeConfig
        
        metrics = [
            EvaluationMetric(
                name="Test Metric",
                type=MetricType.BINARY,
                description="Test metric for validation",
                scoring_criteria={"test": True},
                weight=1.0
            )
        ]
        
        config = JudgeConfig(
            name="Test Judge",
            description="Test configuration",
            metrics=metrics,
            knowledge_sources=[],
            evaluation_prompt_template="Test: {candidate_answer}"
        )
        
        print("✅ Programmatic configuration creation successful")
        
        # Test YAML configuration if file exists
        if Path("config_example.yaml").exists():
            try:
                from llm_judge_app import LLMJudgeApp
                # This will fail without API key, but should parse config successfully
                try:
                    judge = LLMJudgeApp(config_path="config_example.yaml")
                    print("✅ YAML configuration parsing successful")
                except Exception as e:
                    if "OpenAI" in str(e) or "API" in str(e):
                        print("✅ YAML configuration parsing successful (API key missing as expected)")
                    else:
                        print(f"❌ YAML configuration error: {e}")
                        return False
            except Exception as e:
                print(f"❌ YAML configuration error: {e}")
                return False
        else:
            print("⚠️  config_example.yaml not found, skipping YAML test")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_metric_types():
    """Test different metric types."""
    print("\n🎯 Testing metric types...")
    
    try:
        from llm_judge_app import MetricType, EvaluationMetric
        
        # Test all metric types
        metric_types = [
            (MetricType.BINARY, {"true_indicators": ["yes"], "false_indicators": ["no"]}),
            (MetricType.SCALE, {"min_scale": 1, "max_scale": 5}),
            (MetricType.CATEGORICAL, {"categories": ["A", "B", "C"]}),
            (MetricType.NUMERIC, {"min_value": 0, "max_value": 100}),
            (MetricType.TEXT, {"text_evaluation": True})
        ]
        
        for metric_type, criteria in metric_types:
            metric = EvaluationMetric(
                name=f"Test {metric_type.value}",
                type=metric_type,
                description=f"Test {metric_type.value} metric",
                scoring_criteria=criteria
            )
            print(f"  ✅ {metric_type.value} metric creation successful")
        
        print("✅ All metric types working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Metric type test failed: {e}")
        return False


def test_knowledge_base_creation():
    """Test knowledge base file creation and loading."""
    print("\n📚 Testing knowledge base functionality...")
    
    try:
        # Create test knowledge directory
        test_dir = Path("test_knowledge")
        test_dir.mkdir(exist_ok=True)
        
        # Create test files
        test_json = {"test": "data", "evaluation": {"criteria": "test"}}
        with open(test_dir / "test.json", 'w') as f:
            json.dump(test_json, f)
        
        test_text = "Test knowledge content\nExample: This is a test"
        with open(test_dir / "test.txt", 'w') as f:
            f.write(test_text)
        
        try:
            import yaml
            test_yaml = {"rubric": {"good": "test good", "bad": "test bad"}}
            with open(test_dir / "test.yaml", 'w') as f:
                yaml.dump(test_yaml, f)
            yaml_created = True
        except ImportError:
            print("  ⚠️  PyYAML not available, skipping YAML test")
            yaml_created = False
        
        print("✅ Knowledge base files created successfully")
        
        # Test configuration with knowledge sources (without API initialization)
        try:
            from llm_judge_app import JudgeConfig, EvaluationMetric, MetricType
            
            knowledge_sources = [
                str(test_dir / "test.json"),
                str(test_dir / "test.txt")
            ]
            if yaml_created:
                knowledge_sources.append(str(test_dir / "test.yaml"))
            
            config = JudgeConfig(
                name="Test Judge with Knowledge",
                description="Test with knowledge sources",
                metrics=[
                    EvaluationMetric(
                        name="Test",
                        type=MetricType.BINARY,
                        description="Test",
                        scoring_criteria={"test": True}
                    )
                ],
                knowledge_sources=knowledge_sources,
                evaluation_prompt_template="Test: {candidate_answer}"
            )
            
            print("✅ Knowledge source configuration successful")
        except Exception as e:
            print(f"❌ Knowledge source configuration failed: {e}")
            return False
        
        # Cleanup
        import shutil
        shutil.rmtree(test_dir)
        print("✅ Test cleanup successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Knowledge base test failed: {e}")
        return False


def test_dependencies():
    """Test required dependencies."""
    print("\n📦 Testing dependencies...")
    
    required_packages = [
        ('openai', 'OpenAI API client'),
        ('yaml', 'YAML parsing'),
        ('json', 'JSON parsing'),
        ('pathlib', 'Path handling'),
        ('dataclasses', 'Data classes'),
        ('enum', 'Enumerations'),
        ('typing', 'Type hints'),
        ('datetime', 'Date/time handling'),
        ('logging', 'Logging'),
        ('re', 'Regular expressions')
    ]
    
    missing_packages = []
    for package, description in required_packages:
        try:
            if package == 'yaml':
                import yaml
            else:
                __import__(package)
            print(f"  ✅ {package} ({description})")
        except ImportError:
            print(f"  ❌ {package} ({description}) - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    else:
        print("✅ All dependencies available")
        return True


def test_environment():
    """Test environment setup."""
    print("\n🌍 Testing environment...")
    
    # Check Python version
    if sys.version_info < (3, 7):
        print(f"❌ Python {sys.version_info.major}.{sys.version_info.minor} is too old. Need Python 3.7+")
        return False
    else:
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} is supported")
    
    # Check for OpenAI API key (optional)
    if os.getenv("OPENAI_API_KEY"):
        print("✅ OPENAI_API_KEY environment variable is set")
    else:
        print("⚠️  OPENAI_API_KEY not set (will need for actual evaluations)")
    
    # Check write permissions
    try:
        test_file = Path("test_write_permission.tmp")
        test_file.write_text("test")
        test_file.unlink()
        print("✅ Write permissions available")
    except Exception as e:
        print(f"❌ Write permission test failed: {e}")
        return False
    
    return True


def run_all_tests():
    """Run all validation tests."""
    print("🚀 LLM Judge Application Setup Validation")
    print("=" * 50)
    
    tests = [
        test_environment,
        test_dependencies,
        test_imports,
        test_configuration_parsing,
        test_metric_types,
        test_knowledge_base_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"💥 {test.__name__} failed")
        except Exception as e:
            print(f"💥 {test.__name__} crashed: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your setup is ready.")
        print("\nNext steps:")
        print("1. Set OPENAI_API_KEY environment variable")
        print("2. Run: python demo.py")
        print("3. Customize config_example.yaml for your use case")
        return True
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)