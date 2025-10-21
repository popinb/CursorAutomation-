"""
QUICK FIX SNIPPET - Replace the problematic section in your code

Replace this section in your original code:

# OLD CODE (PROBLEMATIC):
out_dir = "/tmp/llm_eval_artifacts"
os.makedirs(out_dir, exist_ok=True)
csv_path = os.path.join(out_dir, f"results_{int(time.time())}.csv")
results_df.to_csv(csv_path, index=False)
mlflow.log_artifact(csv_path, artifact_path="tables")

# NEW CODE (FIXED):
"""

import os
import time
import tempfile
from pathlib import Path


def create_safe_output_directory():
    """Create a safe output directory that works in any environment."""
    possible_dirs = [
        os.path.join(os.getcwd(), "llm_eval_artifacts"),  # Current directory
        os.path.expanduser("~/llm_eval_artifacts"),       # Home directory
        tempfile.mkdtemp(prefix="llm_eval_artifacts_")    # Temp directory
    ]
    
    for directory in possible_dirs:
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            # Test write permissions
            test_file = os.path.join(directory, "test.tmp")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            return directory
        except (PermissionError, OSError):
            continue
    
    return os.getcwd()  # Fallback to current directory


def safe_save_csv(results_df, output_dir):
    """Safely save CSV with error handling."""
    timestamp = int(time.time())
    csv_path = os.path.join(output_dir, f"results_{timestamp}.csv")
    
    try:
        results_df.to_csv(csv_path, index=False)
        return csv_path
    except PermissionError:
        # Try alternative filename
        import random
        alt_path = os.path.join(output_dir, f"results_{timestamp}_{random.randint(1000,9999)}.csv")
        results_df.to_csv(alt_path, index=False)
        return alt_path


# REPLACE YOUR PROBLEMATIC SECTION WITH THIS:
try:
    # Create safe output directory
    out_dir = create_safe_output_directory()
    print(f"✅ Using output directory: {out_dir}")
    
    # Save CSV safely
    csv_path = safe_save_csv(results_df, out_dir)
    print(f"✅ Results saved to: {csv_path}")
    
    # Log to MLflow with error handling
    try:
        mlflow.log_artifact(csv_path, artifact_path="tables")
        print(f"✅ Results logged to MLflow")
    except Exception as mlflow_error:
        print(f"⚠️ MLflow logging failed: {mlflow_error}")
        print(f"📁 Results still saved locally at: {csv_path}")
        
except Exception as e:
    print(f"❌ Error saving results: {e}")
    print(f"📋 Results available in memory as 'results_df'")


# That's it! This replaces the problematic section and handles all edge cases.