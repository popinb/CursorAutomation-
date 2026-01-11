"""
Usage Examples for the Generalized Directory Feature

This script demonstrates various ways to use the directory handler
in different environments (local, Databricks, cloud, etc.)
"""

import os
from directory_handler import DirectoryHandler, get_directory_handler
from llm_evaluation_framework import LLMEvaluationFramework
from evals_wrapper_v2 import ZillowEvalsWrapperV2


def example_1_basic_usage():
    """Example 1: Basic usage with default configuration."""
    print("\n=== Example 1: Basic Usage ===")
    
    # Get default directory handler
    dir_handler = get_directory_handler()
    
    # Print current configuration
    config = dir_handler.get_config_summary()
    print(f"Workspace root: {config['workspace_root']}")
    print("\nDirectories:")
    for dir_type, path in config['directories'].items():
        print(f"  {dir_type}: {path}")
    
    # Get specific directory paths
    data_dir = dir_handler.get_dir('data')
    results_dir = dir_handler.get_dir('results')
    print(f"\nData directory: {data_dir}")
    print(f"Results directory: {results_dir}")
    
    # Get a timestamped file path
    result_file = dir_handler.get_timestamped_file_path('results', 'evaluation', '.csv')
    print(f"Timestamped result file: {result_file}")


def example_2_custom_config():
    """Example 2: Using a custom configuration file."""
    print("\n=== Example 2: Custom Configuration ===")
    
    # Create a custom config
    custom_config = {
        "directories": {
            "data": "/custom/path/to/data",
            "results": "/custom/path/to/results",
            "logs": "./logs",  # Relative path
            "reports": "~/evaluation_reports"  # Home directory path
        },
        "auto_create_dirs": True
    }
    
    # Save custom config
    import json
    with open('custom_config.json', 'w') as f:
        json.dump(custom_config, f, indent=2)
    
    # Use custom config
    dir_handler = DirectoryHandler('custom_config.json')
    print("Custom directories created:")
    for dir_type in ['data', 'results', 'logs', 'reports']:
        print(f"  {dir_type}: {dir_handler.get_dir(dir_type)}")
    
    # Clean up
    os.remove('custom_config.json')


def example_3_environment_variables():
    """Example 3: Using environment variables to override paths."""
    print("\n=== Example 3: Environment Variables ===")
    
    # Set environment variables
    os.environ['WORKSPACE_ROOT'] = '/tmp/llm_eval_workspace'
    os.environ['LLM_EVAL_DATA_DIR'] = '/tmp/llm_eval_data'
    os.environ['LLM_EVAL_RESULTS_DIR'] = '/tmp/llm_eval_results'
    
    # Create directory handler (env vars will override defaults)
    dir_handler = DirectoryHandler()
    
    print("Environment variable overrides:")
    print(f"  WORKSPACE_ROOT: {os.environ['WORKSPACE_ROOT']}")
    print(f"  LLM_EVAL_DATA_DIR: {os.environ['LLM_EVAL_DATA_DIR']}")
    print(f"  LLM_EVAL_RESULTS_DIR: {os.environ['LLM_EVAL_RESULTS_DIR']}")
    
    print("\nResulting directories:")
    print(f"  data: {dir_handler.get_dir('data')}")
    print(f"  results: {dir_handler.get_dir('results')}")
    
    # Clean up env vars
    del os.environ['WORKSPACE_ROOT']
    del os.environ['LLM_EVAL_DATA_DIR']
    del os.environ['LLM_EVAL_RESULTS_DIR']


def example_4_databricks_setup():
    """Example 4: Databricks environment setup."""
    print("\n=== Example 4: Databricks Setup ===")
    
    # Simulate Databricks environment
    databricks_config = {
        "directories": {
            "data": "/dbfs/FileStore/llm_evaluation/data",
            "results": "/dbfs/FileStore/llm_evaluation/results",
            "logs": "/Workspace/Users/user@company.com/logs",
            "reports": "/Workspace/Users/user@company.com/reports",
            "artifacts": "/dbfs/FileStore/llm_evaluation/artifacts",
            "mlflow": "/databricks/mlflow"
        },
        "auto_create_dirs": False  # DBFS directories need special handling
    }
    
    # Save Databricks config
    import json
    with open('databricks_config.json', 'w') as f:
        json.dump(databricks_config, f, indent=2)
    
    # Use Databricks config
    dir_handler = DirectoryHandler('databricks_config.json', auto_create_dirs=False)
    
    print("Databricks directory structure:")
    for dir_type, path in dir_handler.dirs.items():
        if dir_type != 'workspace_root':
            print(f"  {dir_type}: {path}")
    
    # Get MLflow experiment path
    mlflow_path = dir_handler.get_mlflow_experiment_path(
        experiment_name="llm_evaluation_prod",
        username="user@company.com"
    )
    print(f"\nMLflow experiment path: {mlflow_path}")
    
    # Clean up
    os.remove('databricks_config.json')


def example_5_user_specific_paths():
    """Example 5: User-specific paths and multi-user support."""
    print("\n=== Example 5: User-Specific Paths ===")
    
    dir_handler = get_directory_handler()
    
    # Get user-specific directories for different users
    users = ['alice', 'bob', 'charlie']
    
    for user in users:
        user_results = dir_handler.get_user_specific_path('results', username=user)
        user_reports = dir_handler.get_user_specific_path('reports', username=user)
        
        print(f"\nUser: {user}")
        print(f"  Results: {user_results}")
        print(f"  Reports: {user_reports}")
        
        # Get MLflow experiment path for user
        mlflow_path = dir_handler.get_mlflow_experiment_path(
            experiment_name="personal_experiments",
            username=user
        )
        print(f"  MLflow: {mlflow_path}")


def example_6_file_management():
    """Example 6: File management utilities."""
    print("\n=== Example 6: File Management ===")
    
    dir_handler = get_directory_handler()
    
    # Create some test files
    test_files = []
    for i in range(3):
        filepath = dir_handler.get_timestamped_file_path('temp', f'test_{i}', '.txt')
        with open(filepath, 'w') as f:
            f.write(f"Test file {i}")
        test_files.append(filepath)
        print(f"Created: {os.path.basename(filepath)}")
    
    # List files in temp directory
    print("\nFiles in temp directory:")
    temp_files = dir_handler.list_files('temp', '*.txt')
    for f in temp_files:
        print(f"  {os.path.basename(f)}")
    
    # Clean old temp files (in this case, immediately)
    print("\nCleaning temp files...")
    removed = dir_handler.clean_temp_files(older_than_hours=0)
    print(f"Removed {removed} files")


def example_7_evaluation_framework_integration():
    """Example 7: Integration with the evaluation framework."""
    print("\n=== Example 7: Evaluation Framework Integration ===")
    
    # Create evaluation framework with custom paths
    framework = LLMEvaluationFramework()
    
    # Show framework configuration
    framework.print_config_summary()
    
    # Create wrapper with same directory handler
    wrapper = ZillowEvalsWrapperV2(directory_handler=framework.dir_handler)
    
    print("\nWrapper and framework share the same directory configuration")
    print(f"Results will be saved to: {framework.dir_handler.get_dir('results')}")
    print(f"Reports will be saved to: {framework.dir_handler.get_dir('reports')}")


def example_8_cloud_storage_simulation():
    """Example 8: Simulating cloud storage paths."""
    print("\n=== Example 8: Cloud Storage Simulation ===")
    
    # Simulate different cloud environments
    cloud_configs = {
        "AWS S3": {
            "data": "s3://my-bucket/llm-evaluation/data",
            "results": "s3://my-bucket/llm-evaluation/results",
            "artifacts": "s3://my-bucket/llm-evaluation/artifacts"
        },
        "Azure Blob": {
            "data": "https://myaccount.blob.core.windows.net/llm-eval/data",
            "results": "https://myaccount.blob.core.windows.net/llm-eval/results",
            "artifacts": "https://myaccount.blob.core.windows.net/llm-eval/artifacts"
        },
        "Google Cloud Storage": {
            "data": "gs://my-bucket/llm-evaluation/data",
            "results": "gs://my-bucket/llm-evaluation/results",
            "artifacts": "gs://my-bucket/llm-evaluation/artifacts"
        }
    }
    
    for cloud_type, paths in cloud_configs.items():
        print(f"\n{cloud_type} configuration:")
        for dir_type, path in paths.items():
            print(f"  {dir_type}: {path}")


def main():
    """Run all examples."""
    print("=" * 60)
    print("LLM Evaluation Framework - Directory Handler Examples")
    print("=" * 60)
    
    # Run examples
    example_1_basic_usage()
    example_2_custom_config()
    example_3_environment_variables()
    example_4_databricks_setup()
    example_5_user_specific_paths()
    example_6_file_management()
    example_7_evaluation_framework_integration()
    example_8_cloud_storage_simulation()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()