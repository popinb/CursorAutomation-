"""
Configuration Examples for Generalized Directory Management

This module provides pre-configured setups for common deployment scenarios,
making it easy to adapt the evaluation system to different environments.
"""

from directory_manager import DirectoryConfig, DirectoryManager, create_directory_manager
from typing import Dict, Any, Optional
import os


# =============================================================================
# ENVIRONMENT-SPECIFIC CONFIGURATIONS
# =============================================================================

def get_local_development_config(workspace_root: Optional[str] = None) -> DirectoryConfig:
    """Configuration for local development environment."""
    return DirectoryConfig(
        workspace_root=workspace_root or os.getcwd(),
        assets_dir=None,  # Will default to workspace_root/assets
        output_dir=None,  # Will default to workspace_root/outputs
        temp_dir=None,    # Will use system temp
        logs_dir=None,    # Will default to outputs/logs
        
        output_filename_pattern="results_{timestamp}.csv",
        log_filename_pattern="eval_log_{timestamp}.log",
        
        environment="local",
        create_missing_dirs=True,
        cleanup_temp_on_exit=True,
        
        mlflow_experiment_path="llm_evaluation_experiment",
        mlflow_artifact_path="results",
        
        asset_files={
            "golden_responses": "golden_responses.json",
            "buyability_profiles": "buyability_profiles.json", 
            "fair_housing_guide": "fair_housing_guide.json",
            "metrics_config": "metrics_config.csv"
        }
    )


def get_databricks_config(username: Optional[str] = None) -> DirectoryConfig:
    """Configuration for Databricks environment."""
    if username is None:
        username = os.environ.get("DATABRICKS_USER", "unknown_user")
    
    return DirectoryConfig(
        workspace_root=f"/Workspace/Users/{username}",
        assets_dir=f"/Workspace/Users/{username}/assets",
        output_dir="/tmp/llm_eval_artifacts", 
        temp_dir="/tmp/llm_eval_temp",
        logs_dir="/tmp/llm_eval_logs",
        
        output_filename_pattern="results_{timestamp}.csv",
        log_filename_pattern="eval_log_{timestamp}.log",
        
        environment="databricks",
        create_missing_dirs=True,
        cleanup_temp_on_exit=True,
        
        mlflow_experiment_path=f"/Users/{username}/llm_evaluation_experiment",
        mlflow_artifact_path="tables",
        
        asset_files={
            "golden_responses": "golden_responses.json",
            "buyability_profiles": "buyability_profiles.json", 
            "fair_housing_guide": "fair_housing_guide.json",
            "metrics_config": "metrics_config.csv"
        }
    )


def get_aws_lambda_config() -> DirectoryConfig:
    """Configuration for AWS Lambda environment."""
    return DirectoryConfig(
        workspace_root="/var/task",
        assets_dir="/var/task/assets",
        output_dir="/tmp/outputs",
        temp_dir="/tmp",
        logs_dir="/tmp/logs",
        
        output_filename_pattern="lambda_results_{timestamp}.csv",
        log_filename_pattern="lambda_log_{timestamp}.log",
        
        environment="cloud",
        create_missing_dirs=True,
        cleanup_temp_on_exit=True,
        
        mlflow_experiment_path="lambda_llm_evaluation",
        mlflow_artifact_path="results",
        
        asset_files={
            "golden_responses": "golden_responses.json",
            "buyability_profiles": "buyability_profiles.json", 
            "fair_housing_guide": "fair_housing_guide.json"
        }
    )


def get_docker_config() -> DirectoryConfig:
    """Configuration for Docker container environment."""
    return DirectoryConfig(
        workspace_root="/app",
        assets_dir="/app/assets",
        output_dir="/app/outputs",
        temp_dir="/tmp",
        logs_dir="/app/logs",
        
        output_filename_pattern="docker_results_{timestamp}.csv",
        log_filename_pattern="docker_log_{timestamp}.log",
        
        environment="container",
        create_missing_dirs=True,
        cleanup_temp_on_exit=True,
        
        mlflow_experiment_path="docker_llm_evaluation",
        mlflow_artifact_path="results",
        
        asset_files={
            "golden_responses": "golden_responses.json",
            "buyability_profiles": "buyability_profiles.json", 
            "fair_housing_guide": "fair_housing_guide.json"
        }
    )


def get_azure_functions_config() -> DirectoryConfig:
    """Configuration for Azure Functions environment."""
    return DirectoryConfig(
        workspace_root=os.environ.get("AzureWebJobsScriptRoot", "/home/site/wwwroot"),
        assets_dir=None,  # Will be derived from workspace_root
        output_dir="/tmp/outputs",
        temp_dir="/tmp",
        logs_dir="/tmp/logs",
        
        output_filename_pattern="azure_results_{timestamp}.csv",
        log_filename_pattern="azure_log_{timestamp}.log",
        
        environment="cloud",
        create_missing_dirs=True,
        cleanup_temp_on_exit=True,
        
        mlflow_experiment_path="azure_llm_evaluation",
        mlflow_artifact_path="results"
    )


def get_google_cloud_config() -> DirectoryConfig:
    """Configuration for Google Cloud Functions environment."""
    return DirectoryConfig(
        workspace_root="/workspace",
        assets_dir="/workspace/assets",
        output_dir="/tmp/outputs",
        temp_dir="/tmp",
        logs_dir="/tmp/logs",
        
        output_filename_pattern="gcp_results_{timestamp}.csv",
        log_filename_pattern="gcp_log_{timestamp}.log",
        
        environment="cloud",
        create_missing_dirs=True,
        cleanup_temp_on_exit=True,
        
        mlflow_experiment_path="gcp_llm_evaluation",
        mlflow_artifact_path="results"
    )


# =============================================================================
# USE CASE SPECIFIC CONFIGURATIONS
# =============================================================================

def get_testing_config(test_dir: str = "/tmp/test_eval") -> DirectoryConfig:
    """Configuration for testing environment with isolated directories."""
    return DirectoryConfig(
        workspace_root=test_dir,
        assets_dir=f"{test_dir}/test_assets",
        output_dir=f"{test_dir}/test_outputs",
        temp_dir=f"{test_dir}/test_temp",
        logs_dir=f"{test_dir}/test_logs",
        
        output_filename_pattern="test_results_{timestamp}.csv",
        log_filename_pattern="test_log_{timestamp}.log",
        
        environment="local",
        create_missing_dirs=True,
        cleanup_temp_on_exit=True,
        
        mlflow_experiment_path="test_llm_evaluation",
        mlflow_artifact_path="test_results"
    )


def get_production_config(base_dir: str = "/opt/llm_eval") -> DirectoryConfig:
    """Configuration for production environment with structured directories."""
    return DirectoryConfig(
        workspace_root=base_dir,
        assets_dir=f"{base_dir}/assets",
        output_dir=f"{base_dir}/outputs",
        temp_dir="/tmp/llm_eval",
        logs_dir=f"{base_dir}/logs",
        
        output_filename_pattern="prod_results_{timestamp}.csv",
        log_filename_pattern="prod_log_{timestamp}.log",
        
        environment="local",
        create_missing_dirs=True,
        cleanup_temp_on_exit=False,  # Keep temp files in production for debugging
        
        mlflow_experiment_path="production_llm_evaluation",
        mlflow_artifact_path="production_results"
    )


def get_ci_cd_config() -> DirectoryConfig:
    """Configuration for CI/CD pipeline environment."""
    return DirectoryConfig(
        workspace_root=os.environ.get("GITHUB_WORKSPACE", "/github/workspace"),
        assets_dir=None,
        output_dir="/tmp/ci_outputs",
        temp_dir="/tmp/ci_temp", 
        logs_dir="/tmp/ci_logs",
        
        output_filename_pattern="ci_results_{timestamp}.csv",
        log_filename_pattern="ci_log_{timestamp}.log",
        
        environment="container",
        create_missing_dirs=True,
        cleanup_temp_on_exit=True,
        
        mlflow_experiment_path="ci_llm_evaluation",
        mlflow_artifact_path="ci_results"
    )


# =============================================================================
# CONFIGURATION FACTORY
# =============================================================================

def create_config_for_environment(env_name: str, **kwargs) -> DirectoryConfig:
    """
    Factory function to create configuration for specific environments.
    
    Args:
        env_name: Name of the environment ('local', 'databricks', 'aws', etc.)
        **kwargs: Additional configuration parameters
        
    Returns:
        DirectoryConfig for the specified environment
    """
    config_map = {
        'local': get_local_development_config,
        'databricks': get_databricks_config,
        'aws': get_aws_lambda_config,
        'aws_lambda': get_aws_lambda_config,
        'docker': get_docker_config,
        'container': get_docker_config,
        'azure': get_azure_functions_config,
        'gcp': get_google_cloud_config,
        'google_cloud': get_google_cloud_config,
        'testing': get_testing_config,
        'test': get_testing_config,
        'production': get_production_config,
        'prod': get_production_config,
        'ci': get_ci_cd_config,
        'ci_cd': get_ci_cd_config
    }
    
    if env_name.lower() not in config_map:
        raise ValueError(f"Unknown environment: {env_name}. Available: {list(config_map.keys())}")
    
    config_func = config_map[env_name.lower()]
    
    # Get function signature to pass appropriate kwargs
    import inspect
    sig = inspect.signature(config_func)
    valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    
    return config_func(**valid_kwargs)


# =============================================================================
# MIGRATION HELPERS
# =============================================================================

def migrate_hardcoded_paths(old_paths: Dict[str, str]) -> DirectoryManager:
    """
    Helper function to migrate from hardcoded paths to directory manager.
    
    Args:
        old_paths: Dictionary of old hardcoded paths
        
    Returns:
        DirectoryManager configured to match old paths where possible
    """
    config = DirectoryConfig()
    
    # Map old paths to new configuration
    if 'workspace_root' in old_paths:
        config.workspace_root = old_paths['workspace_root']
    
    if 'assets_dir' in old_paths:
        config.assets_dir = old_paths['assets_dir']
    
    if 'output_dir' in old_paths:
        config.output_dir = old_paths['output_dir']
    
    if 'temp_dir' in old_paths:
        config.temp_dir = old_paths['temp_dir']
    
    if 'mlflow_experiment' in old_paths:
        config.mlflow_experiment_path = old_paths['mlflow_experiment']
    
    return DirectoryManager(config)


def validate_migration(old_paths: Dict[str, str], directory_manager: DirectoryManager) -> Dict[str, Any]:
    """
    Validate that migration from hardcoded paths is successful.
    
    Args:
        old_paths: Original hardcoded paths
        directory_manager: New DirectoryManager instance
        
    Returns:
        Validation results
    """
    validation = {
        'migration_successful': True,
        'path_mappings': {},
        'issues': []
    }
    
    # Check path mappings
    path_mappings = {
        'workspace_root': str(directory_manager.workspace_root),
        'assets_dir': str(directory_manager.assets_dir),
        'output_dir': str(directory_manager.output_dir),
        'temp_dir': str(directory_manager.temp_dir),
        'mlflow_experiment': directory_manager.get_mlflow_experiment_path()
    }
    
    validation['path_mappings'] = path_mappings
    
    # Check for issues
    for old_key, old_path in old_paths.items():
        if old_key in path_mappings:
            new_path = path_mappings[old_key]
            if old_path != new_path:
                validation['issues'].append(f"Path changed: {old_key} {old_path} -> {new_path}")
        else:
            validation['issues'].append(f"No mapping found for: {old_key}")
    
    if validation['issues']:
        validation['migration_successful'] = False
    
    return validation


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def example_usage():
    """Demonstrate configuration usage."""
    print("=== Configuration Examples ===\n")
    
    # Example 1: Auto-detection
    print("1. Auto-detection:")
    dm_auto = create_directory_manager()
    print(f"   Environment: {dm_auto.environment}")
    print(f"   Workspace: {dm_auto.workspace_root}")
    
    # Example 2: Specific environment
    print("\n2. Databricks environment:")
    config_db = get_databricks_config("user@company.com")
    dm_db = DirectoryManager(config_db)
    print(f"   Workspace: {dm_db.workspace_root}")
    print(f"   MLflow: {dm_db.get_mlflow_experiment_path()}")
    
    # Example 3: Testing environment
    print("\n3. Testing environment:")
    config_test = get_testing_config("/tmp/my_test")
    dm_test = DirectoryManager(config_test)
    print(f"   Test workspace: {dm_test.workspace_root}")
    print(f"   Test output: {dm_test.output_dir}")
    
    # Example 4: Migration from hardcoded paths
    print("\n4. Migration from hardcoded paths:")
    old_paths = {
        'output_dir': '/tmp/llm_eval_artifacts',
        'mlflow_experiment': '/Users/user@company.com/llm_evaluation_experiment'
    }
    dm_migrated = migrate_hardcoded_paths(old_paths)
    validation = validate_migration(old_paths, dm_migrated)
    print(f"   Migration successful: {validation['migration_successful']}")
    if validation['issues']:
        for issue in validation['issues']:
            print(f"   Issue: {issue}")


if __name__ == "__main__":
    example_usage()