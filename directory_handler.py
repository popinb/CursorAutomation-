"""
Directory Handler for LLM Evaluation Framework

This module provides a generalized directory management system for handling
paths, configurations, and file locations in a flexible and portable way.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime


class DirectoryHandler:
    """
    Manages directory paths and file locations for the LLM evaluation framework.
    Supports environment variables, user-specific paths, and configurable directories.
    """
    
    def __init__(self, config_path: Optional[str] = None, auto_create_dirs: bool = True):
        """
        Initialize the directory handler.
        
        Args:
            config_path: Path to configuration file (JSON) with directory settings
            auto_create_dirs: Whether to automatically create directories if they don't exist
        """
        self.auto_create_dirs = auto_create_dirs
        
        # Default directory structure
        self.dirs = {
            'workspace_root': self._get_workspace_root(),
            'data': 'data',
            'results': 'results',
            'logs': 'logs',
            'configs': 'configs',
            'metrics': 'metrics',
            'ground_truth': 'ground_truth',
            'reports': 'reports',
            'artifacts': 'artifacts',
            'mlflow': 'mlflow_runs',
            'temp': 'temp'
        }
        
        # Load user config if provided
        if config_path:
            self._load_config(config_path)
        
        # Override with environment variables if set
        self._load_env_overrides()
        
        # Create absolute paths
        self._resolve_paths()
        
        # Create directories if needed
        if self.auto_create_dirs:
            self._create_directories()
    
    def _get_workspace_root(self) -> str:
        """Get the workspace root directory."""
        # Priority order:
        # 1. WORKSPACE_ROOT environment variable
        # 2. Current working directory
        return os.environ.get('WORKSPACE_ROOT', os.getcwd())
    
    def _load_config(self, config_path: str) -> None:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # Update directories from config
            if 'directories' in config:
                self.dirs.update(config['directories'])
                
            # Update settings
            if 'auto_create_dirs' in config:
                self.auto_create_dirs = config['auto_create_dirs']
                
        except FileNotFoundError:
            print(f"Warning: Config file not found at {config_path}, using defaults")
        except json.JSONDecodeError as e:
            print(f"Warning: Invalid JSON in config file: {e}")
    
    def _load_env_overrides(self) -> None:
        """Override directory paths with environment variables if set."""
        env_mappings = {
            'LLM_EVAL_DATA_DIR': 'data',
            'LLM_EVAL_RESULTS_DIR': 'results',
            'LLM_EVAL_LOGS_DIR': 'logs',
            'LLM_EVAL_CONFIGS_DIR': 'configs',
            'LLM_EVAL_METRICS_DIR': 'metrics',
            'LLM_EVAL_GROUND_TRUTH_DIR': 'ground_truth',
            'LLM_EVAL_REPORTS_DIR': 'reports',
            'LLM_EVAL_ARTIFACTS_DIR': 'artifacts',
            'LLM_EVAL_MLFLOW_DIR': 'mlflow',
            'LLM_EVAL_TEMP_DIR': 'temp'
        }
        
        for env_var, dir_key in env_mappings.items():
            if env_var in os.environ:
                self.dirs[dir_key] = os.environ[env_var]
    
    def _resolve_paths(self) -> None:
        """Convert relative paths to absolute paths."""
        workspace_root = Path(self.dirs['workspace_root'])
        
        for key, path in self.dirs.items():
            if key == 'workspace_root':
                continue
                
            path_obj = Path(path)
            
            # If path is not absolute, make it relative to workspace root
            if not path_obj.is_absolute():
                self.dirs[key] = str(workspace_root / path)
            else:
                self.dirs[key] = str(path_obj)
    
    def _create_directories(self) -> None:
        """Create directories if they don't exist."""
        for key, path in self.dirs.items():
            if key == 'workspace_root':
                continue
                
            path_obj = Path(path)
            if not path_obj.exists():
                try:
                    path_obj.mkdir(parents=True, exist_ok=True)
                    print(f"Created directory: {path}")
                except Exception as e:
                    print(f"Warning: Could not create directory {path}: {e}")
    
    def get_dir(self, dir_type: str) -> str:
        """
        Get the absolute path for a directory type.
        
        Args:
            dir_type: Type of directory (e.g., 'data', 'results', 'logs')
            
        Returns:
            Absolute path to the directory
        """
        if dir_type not in self.dirs:
            raise ValueError(f"Unknown directory type: {dir_type}")
        return self.dirs[dir_type]
    
    def get_file_path(self, dir_type: str, filename: str, create_dir: bool = True) -> str:
        """
        Get the full path for a file in a specific directory.
        
        Args:
            dir_type: Type of directory (e.g., 'data', 'results')
            filename: Name of the file
            create_dir: Whether to create the directory if it doesn't exist
            
        Returns:
            Full absolute path to the file
        """
        dir_path = self.get_dir(dir_type)
        
        if create_dir and self.auto_create_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            
        return str(Path(dir_path) / filename)
    
    def get_timestamped_file_path(self, dir_type: str, base_name: str, extension: str = '') -> str:
        """
        Get a timestamped file path for creating unique filenames.
        
        Args:
            dir_type: Type of directory
            base_name: Base name for the file
            extension: File extension (with or without dot)
            
        Returns:
            Full path with timestamp
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Ensure extension has a dot
        if extension and not extension.startswith('.'):
            extension = '.' + extension
            
        filename = f"{base_name}_{timestamp}{extension}"
        return self.get_file_path(dir_type, filename)
    
    def get_user_specific_path(self, dir_type: str, username: Optional[str] = None) -> str:
        """
        Get a user-specific subdirectory path.
        
        Args:
            dir_type: Type of directory
            username: Username (if None, tries to get from environment)
            
        Returns:
            User-specific directory path
        """
        if username is None:
            username = os.environ.get('USER', os.environ.get('USERNAME', 'default_user'))
            
        base_dir = self.get_dir(dir_type)
        user_dir = Path(base_dir) / username
        
        if self.auto_create_dirs:
            user_dir.mkdir(parents=True, exist_ok=True)
            
        return str(user_dir)
    
    def get_mlflow_experiment_path(self, experiment_name: Optional[str] = None, 
                                   username: Optional[str] = None) -> str:
        """
        Get MLflow experiment path following Databricks/MLflow conventions.
        
        Args:
            experiment_name: Name of the experiment
            username: Username (if None, tries to get from environment)
            
        Returns:
            MLflow experiment path
        """
        if username is None:
            username = os.environ.get('USER', os.environ.get('USERNAME', 'default_user'))
            
        if experiment_name is None:
            experiment_name = 'llm_evaluation_experiment'
            
        # Format: /Users/{username}/{experiment_name}
        return f"/Users/{username}/{experiment_name}"
    
    def resolve_path(self, path: Union[str, Path], relative_to: Optional[str] = None) -> str:
        """
        Resolve a path, making it absolute if needed.
        
        Args:
            path: Path to resolve
            relative_to: Directory type to use as base if path is relative
            
        Returns:
            Absolute path
        """
        path_obj = Path(path)
        
        if path_obj.is_absolute():
            return str(path_obj)
            
        if relative_to:
            base_dir = self.get_dir(relative_to)
            return str(Path(base_dir) / path)
        else:
            return str(Path(self.dirs['workspace_root']) / path)
    
    def list_files(self, dir_type: str, pattern: str = '*', recursive: bool = False) -> list:
        """
        List files in a directory matching a pattern.
        
        Args:
            dir_type: Type of directory
            pattern: Glob pattern for matching files
            recursive: Whether to search recursively
            
        Returns:
            List of file paths
        """
        dir_path = Path(self.get_dir(dir_type))
        
        if not dir_path.exists():
            return []
            
        if recursive:
            return [str(p) for p in dir_path.rglob(pattern) if p.is_file()]
        else:
            return [str(p) for p in dir_path.glob(pattern) if p.is_file()]
    
    def clean_temp_files(self, older_than_hours: int = 24) -> int:
        """
        Clean old temporary files.
        
        Args:
            older_than_hours: Remove files older than this many hours
            
        Returns:
            Number of files removed
        """
        import time
        
        temp_dir = Path(self.get_dir('temp'))
        if not temp_dir.exists():
            return 0
            
        current_time = time.time()
        cutoff_time = current_time - (older_than_hours * 3600)
        
        removed_count = 0
        for file_path in temp_dir.iterdir():
            if file_path.is_file():
                if file_path.stat().st_mtime < cutoff_time:
                    try:
                        file_path.unlink()
                        removed_count += 1
                    except Exception as e:
                        print(f"Warning: Could not remove {file_path}: {e}")
                        
        return removed_count
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of the current configuration."""
        return {
            'workspace_root': self.dirs['workspace_root'],
            'directories': {k: v for k, v in self.dirs.items() if k != 'workspace_root'},
            'auto_create_dirs': self.auto_create_dirs,
            'environment_overrides': {
                k: v for k, v in os.environ.items() 
                if k.startswith('LLM_EVAL_') or k == 'WORKSPACE_ROOT'
            }
        }
    
    def save_config(self, config_path: str) -> None:
        """
        Save the current configuration to a JSON file.
        
        Args:
            config_path: Path to save the configuration
        """
        config = {
            'directories': {k: v for k, v in self.dirs.items() if k != 'workspace_root'},
            'auto_create_dirs': self.auto_create_dirs
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Configuration saved to: {config_path}")


# Singleton instance for easy importing
_default_handler = None


def get_directory_handler(config_path: Optional[str] = None) -> DirectoryHandler:
    """
    Get the default directory handler instance.
    
    Args:
        config_path: Optional config path (only used on first call)
        
    Returns:
        DirectoryHandler instance
    """
    global _default_handler
    
    if _default_handler is None:
        _default_handler = DirectoryHandler(config_path)
        
    return _default_handler


def reset_handler() -> None:
    """Reset the default handler instance."""
    global _default_handler
    _default_handler = None