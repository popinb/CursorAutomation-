"""
Directory Manager for LLM Evaluation System

This module provides a flexible directory management system that can handle
various deployment environments (local, cloud, container) and different
directory structures for assets, outputs, logs, and temporary files.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
from datetime import datetime
import tempfile
import shutil


@dataclass
class DirectoryConfig:
    """Configuration for directory paths and settings."""
    
    # Base directories
    workspace_root: Optional[str] = None
    assets_dir: Optional[str] = None
    output_dir: Optional[str] = None
    temp_dir: Optional[str] = None
    logs_dir: Optional[str] = None
    
    # File patterns and naming
    output_filename_pattern: str = "results_{timestamp}.csv"
    log_filename_pattern: str = "eval_log_{timestamp}.log"
    
    # Environment-specific settings
    environment: str = "auto"  # auto, local, databricks, cloud, container
    create_missing_dirs: bool = True
    cleanup_temp_on_exit: bool = True
    
    # MLflow settings
    mlflow_experiment_path: Optional[str] = None
    mlflow_artifact_path: str = "tables"
    
    # Asset file mappings
    asset_files: Dict[str, str] = field(default_factory=lambda: {
        "golden_responses": "golden_responses.json",
        "buyability_profiles": "buyability_profiles.json", 
        "fair_housing_guide": "fair_housing_guide.json"
    })


class DirectoryManager:
    """
    Manages directory paths and file operations across different environments.
    
    Features:
    - Auto-detects environment (local, Databricks, cloud, container)
    - Flexible directory configuration
    - Automatic directory creation
    - Path resolution and validation
    - Environment-specific optimizations
    - Cleanup management
    """
    
    def __init__(self, config: Optional[DirectoryConfig] = None):
        """
        Initialize the directory manager.
        
        Args:
            config: Directory configuration. If None, uses auto-detection.
        """
        self.config = config or DirectoryConfig()
        self.environment = self._detect_environment()
        self._temp_dirs_created: List[Path] = []
        
        # Apply environment-specific defaults
        self._apply_environment_defaults()
        
        # Resolve and validate paths
        self._resolve_paths()
        
        # Create directories if needed
        if self.config.create_missing_dirs:
            self._create_directories()
    
    def _detect_environment(self) -> str:
        """Auto-detect the current environment."""
        if self.config.environment != "auto":
            return self.config.environment
        
        # Check for Databricks
        if os.path.exists("/databricks") or "DATABRICKS_RUNTIME_VERSION" in os.environ:
            return "databricks"
        
        # Check for common cloud environments
        if any(key in os.environ for key in ["AWS_LAMBDA_FUNCTION_NAME", "GOOGLE_CLOUD_PROJECT", "AZURE_FUNCTIONS_ENVIRONMENT"]):
            return "cloud"
        
        # Check for container environments
        if os.path.exists("/.dockerenv") or os.environ.get("CONTAINER") == "true":
            return "container"
        
        # Default to local
        return "local"
    
    def _apply_environment_defaults(self):
        """Apply environment-specific default configurations."""
        if self.environment == "databricks":
            self._apply_databricks_defaults()
        elif self.environment == "cloud":
            self._apply_cloud_defaults()
        elif self.environment == "container":
            self._apply_container_defaults()
        else:
            self._apply_local_defaults()
    
    def _apply_databricks_defaults(self):
        """Apply Databricks-specific defaults."""
        if not self.config.workspace_root:
            # Try to get user workspace path
            username = os.environ.get("DATABRICKS_USER", "unknown_user")
            self.config.workspace_root = f"/Workspace/Users/{username}"
        
        if not self.config.output_dir:
            self.config.output_dir = "/tmp/llm_eval_artifacts"
        
        if not self.config.temp_dir:
            self.config.temp_dir = "/tmp/llm_eval_temp"
        
        if not self.config.mlflow_experiment_path:
            username = os.environ.get("DATABRICKS_USER", "unknown_user")
            self.config.mlflow_experiment_path = f"/Users/{username}/llm_evaluation_experiment"
    
    def _apply_cloud_defaults(self):
        """Apply cloud environment defaults."""
        if not self.config.workspace_root:
            self.config.workspace_root = os.environ.get("WORKSPACE_ROOT", "/app")
        
        if not self.config.output_dir:
            self.config.output_dir = os.environ.get("OUTPUT_DIR", "/tmp/outputs")
        
        if not self.config.temp_dir:
            self.config.temp_dir = tempfile.gettempdir()
    
    def _apply_container_defaults(self):
        """Apply container environment defaults."""
        if not self.config.workspace_root:
            self.config.workspace_root = "/workspace"
        
        if not self.config.output_dir:
            self.config.output_dir = "/workspace/outputs"
        
        if not self.config.temp_dir:
            self.config.temp_dir = "/tmp"
    
    def _apply_local_defaults(self):
        """Apply local development defaults."""
        if not self.config.workspace_root:
            # Use current working directory or script directory
            self.config.workspace_root = os.getcwd()
        
        if not self.config.output_dir:
            self.config.output_dir = os.path.join(self.config.workspace_root, "outputs")
        
        if not self.config.temp_dir:
            self.config.temp_dir = tempfile.gettempdir()
    
    def _resolve_paths(self):
        """Resolve all configured paths to absolute paths."""
        # Resolve workspace root
        if self.config.workspace_root:
            self.workspace_root = Path(self.config.workspace_root).resolve()
        else:
            self.workspace_root = Path.cwd()
        
        # Resolve assets directory
        if self.config.assets_dir:
            self.assets_dir = Path(self.config.assets_dir).resolve()
        else:
            self.assets_dir = self.workspace_root / "assets"
        
        # Resolve output directory
        if self.config.output_dir:
            self.output_dir = Path(self.config.output_dir).resolve()
        else:
            self.output_dir = self.workspace_root / "outputs"
        
        # Resolve temp directory
        if self.config.temp_dir:
            self.temp_dir = Path(self.config.temp_dir).resolve()
        else:
            self.temp_dir = Path(tempfile.gettempdir())
        
        # Resolve logs directory
        if self.config.logs_dir:
            self.logs_dir = Path(self.config.logs_dir).resolve()
        else:
            self.logs_dir = self.output_dir / "logs"
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.assets_dir,
            self.output_dir,
            self.temp_dir,
            self.logs_dir
        ]
        
        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                print(f"Warning: Cannot create directory {directory} due to permissions")
            except Exception as e:
                print(f"Warning: Failed to create directory {directory}: {e}")
    
    def get_asset_path(self, asset_name: str) -> Path:
        """
        Get the full path to an asset file.
        
        Args:
            asset_name: Name of the asset (e.g., 'golden_responses')
            
        Returns:
            Path to the asset file
            
        Raises:
            FileNotFoundError: If asset file doesn't exist
        """
        if asset_name in self.config.asset_files:
            filename = self.config.asset_files[asset_name]
        else:
            filename = f"{asset_name}.json"
        
        asset_path = self.assets_dir / filename
        
        if not asset_path.exists():
            raise FileNotFoundError(f"Asset file not found: {asset_path}")
        
        return asset_path
    
    def get_output_path(self, filename: Optional[str] = None, timestamp: bool = True) -> Path:
        """
        Get a path for output files.
        
        Args:
            filename: Specific filename, or None to use pattern
            timestamp: Whether to include timestamp in filename
            
        Returns:
            Path for output file
        """
        if filename is None:
            if timestamp:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = self.config.output_filename_pattern.format(timestamp=ts)
            else:
                filename = self.config.output_filename_pattern.format(timestamp="")
        
        return self.output_dir / filename
    
    def get_temp_path(self, prefix: str = "llm_eval", suffix: str = "", create_dir: bool = False) -> Path:
        """
        Get a temporary file or directory path.
        
        Args:
            prefix: Prefix for temp file/dir name
            suffix: Suffix for temp file/dir name
            create_dir: If True, create as directory instead of file
            
        Returns:
            Path to temporary file or directory
        """
        if create_dir:
            temp_path = self.temp_dir / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{suffix}"
            temp_path.mkdir(parents=True, exist_ok=True)
            self._temp_dirs_created.append(temp_path)
        else:
            # Create a unique temp file path
            import uuid
            temp_path = self.temp_dir / f"{prefix}_{uuid.uuid4().hex[:8]}{suffix}"
        
        return temp_path
    
    def get_log_path(self, log_name: Optional[str] = None, timestamp: bool = True) -> Path:
        """
        Get a path for log files.
        
        Args:
            log_name: Specific log name, or None to use pattern
            timestamp: Whether to include timestamp
            
        Returns:
            Path for log file
        """
        if log_name is None:
            if timestamp:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_name = self.config.log_filename_pattern.format(timestamp=ts)
            else:
                log_name = self.config.log_filename_pattern.format(timestamp="")
        
        return self.logs_dir / log_name
    
    def get_mlflow_experiment_path(self) -> str:
        """Get the MLflow experiment path for the current environment."""
        if self.config.mlflow_experiment_path:
            return self.config.mlflow_experiment_path
        
        # Generate default based on environment
        if self.environment == "databricks":
            username = os.environ.get("DATABRICKS_USER", "unknown_user")
            return f"/Users/{username}/llm_evaluation_experiment"
        else:
            return "llm_evaluation_experiment"
    
    def cleanup_temp_files(self):
        """Clean up temporary directories created during this session."""
        if not self.config.cleanup_temp_on_exit:
            return
        
        for temp_dir in self._temp_dirs_created:
            try:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Warning: Failed to cleanup temp directory {temp_dir}: {e}")
        
        self._temp_dirs_created.clear()
    
    def validate_environment(self) -> Dict[str, Any]:
        """
        Validate the current environment and return status information.
        
        Returns:
            Dictionary with validation results
        """
        validation = {
            "environment": self.environment,
            "workspace_root": {
                "path": str(self.workspace_root),
                "exists": self.workspace_root.exists(),
                "writable": os.access(self.workspace_root, os.W_OK) if self.workspace_root.exists() else False
            },
            "assets_dir": {
                "path": str(self.assets_dir),
                "exists": self.assets_dir.exists(),
                "readable": os.access(self.assets_dir, os.R_OK) if self.assets_dir.exists() else False
            },
            "output_dir": {
                "path": str(self.output_dir),
                "exists": self.output_dir.exists(),
                "writable": os.access(self.output_dir, os.W_OK) if self.output_dir.exists() else False
            },
            "temp_dir": {
                "path": str(self.temp_dir),
                "exists": self.temp_dir.exists(),
                "writable": os.access(self.temp_dir, os.W_OK) if self.temp_dir.exists() else False
            },
            "assets": {}
        }
        
        # Check asset files
        for asset_name, filename in self.config.asset_files.items():
            asset_path = self.assets_dir / filename
            validation["assets"][asset_name] = {
                "path": str(asset_path),
                "exists": asset_path.exists(),
                "readable": os.access(asset_path, os.R_OK) if asset_path.exists() else False
            }
        
        return validation
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup_temp_files()


def create_directory_manager(
    workspace_root: Optional[str] = None,
    environment: Optional[str] = None,
    **kwargs
) -> DirectoryManager:
    """
    Factory function to create a DirectoryManager with common configurations.
    
    Args:
        workspace_root: Override workspace root directory
        environment: Override environment detection
        **kwargs: Additional configuration options
        
    Returns:
        Configured DirectoryManager instance
    """
    config = DirectoryConfig()
    
    if workspace_root:
        config.workspace_root = workspace_root
    
    if environment:
        config.environment = environment
    
    # Apply any additional configuration
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return DirectoryManager(config)


# Convenience functions for common use cases
def get_local_manager(workspace_root: Optional[str] = None) -> DirectoryManager:
    """Get a DirectoryManager configured for local development."""
    return create_directory_manager(
        workspace_root=workspace_root,
        environment="local"
    )


def get_databricks_manager(username: Optional[str] = None) -> DirectoryManager:
    """Get a DirectoryManager configured for Databricks."""
    workspace_root = None
    if username:
        workspace_root = f"/Workspace/Users/{username}"
    
    return create_directory_manager(
        workspace_root=workspace_root,
        environment="databricks"
    )


def get_cloud_manager(workspace_root: Optional[str] = None) -> DirectoryManager:
    """Get a DirectoryManager configured for cloud environments."""
    return create_directory_manager(
        workspace_root=workspace_root,
        environment="cloud"
    )


if __name__ == "__main__":
    # Example usage and testing
    print("=== Directory Manager Test ===\n")
    
    # Test auto-detection
    with create_directory_manager() as dm:
        print(f"Detected environment: {dm.environment}")
        print(f"Workspace root: {dm.workspace_root}")
        print(f"Assets directory: {dm.assets_dir}")
        print(f"Output directory: {dm.output_dir}")
        print(f"Temp directory: {dm.temp_dir}")
        
        # Validate environment
        validation = dm.validate_environment()
        print(f"\nEnvironment validation:")
        for key, value in validation.items():
            if isinstance(value, dict) and "path" in value:
                status = "✅" if value.get("exists") else "❌"
                print(f"  {key}: {status} {value['path']}")
            elif key == "assets":
                print(f"  {key}:")
                for asset_name, asset_info in value.items():
                    status = "✅" if asset_info.get("exists") else "❌"
                    print(f"    {asset_name}: {status} {asset_info['path']}")
            else:
                print(f"  {key}: {value}")
        
        # Test path generation
        print(f"\nExample paths:")
        print(f"  Output file: {dm.get_output_path()}")
        print(f"  Log file: {dm.get_log_path()}")
        print(f"  Temp file: {dm.get_temp_path('test', '.tmp')}")
        print(f"  MLflow experiment: {dm.get_mlflow_experiment_path()}")