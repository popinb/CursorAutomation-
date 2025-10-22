import os
import getpass
from pathlib import Path
from typing import Optional, Tuple


def is_databricks() -> bool:
    """Best-effort detection of Databricks runtime."""
    if os.environ.get("DATABRICKS_RUNTIME_VERSION"):
        return True
    try:
        import dbruntime  # type: ignore
        return True
    except Exception:
        return False


def get_current_user() -> str:
    """Resolve current user with sensible fallbacks.

    Priority:
    1) MLFLOW_USER env var
    2) USER/USERNAME env var
    3) Databricks user via dbutils (if available)
    4) OS login user
    """
    user = os.environ.get("MLFLOW_USER") or os.environ.get("USER") or os.environ.get("USERNAME")
    if user:
        return user

    if is_databricks():
        try:
            # Lazy import to avoid requiring pyspark locally
            from pyspark.dbutils import DBUtils  # type: ignore
            from pyspark.sql import SparkSession  # type: ignore

            spark = SparkSession.builder.getOrCreate()
            dbutils = DBUtils(spark)
            return (
                dbutils.notebook.entry_point
                .getDbutils().notebook().getContext().userName().get()
            )
        except Exception:
            pass

    return getpass.getuser()


def resolve_mlflow_experiment(default_name: str = "llm_evaluation_experiment") -> str:
    """Return an experiment path/name portable across environments.

    - If MLFLOW_EXPERIMENT_PATH is set, use it literally
    - On Databricks, return /Users/<user>/<default_name>
    - Else return a plain name for local MLflow
    """
    exp_override = os.environ.get("MLFLOW_EXPERIMENT_PATH")
    if exp_override:
        return exp_override

    if is_databricks():
        return f"/Users/{get_current_user()}/{default_name}"

    return default_name


def get_artifacts_dir(subdir: str = "llm_eval_artifacts") -> Path:
    """Directory for writing local artifacts before MLflow logging.

    - If LLM_EVAL_ARTIFACTS_DIR is set, use it
    - On Databricks, use /dbfs/tmp/<subdir>
    - Locally, use ./artifacts/<subdir>
    """
    override = os.environ.get("LLM_EVAL_ARTIFACTS_DIR")
    if override:
        path_override = Path(override).expanduser().resolve()
        path_override.mkdir(parents=True, exist_ok=True)
        return path_override

    if is_databricks():
        base = Path("/dbfs/tmp")  # corresponds to dbfs:/tmp
    else:
        base = Path.cwd() / "artifacts"

    out_dir = base / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def resolve_ground_truth_path(path_str: str, default_dir_env: str = "GROUND_TRUTH_DIR") -> Optional[Path]:
    """Resolve a provided ground-truth file path.

    Tries in order:
    1) The path as given (absolute or relative)
    2) $GROUND_TRUTH_DIR/<path_str> if env var is set
    3) ./assets/<path_str>
    """
    if not path_str:
        return None

    candidate = Path(path_str)
    if candidate.is_file():
        return candidate.resolve()

    base_dir = os.environ.get(default_dir_env)
    if base_dir:
        candidate = Path(base_dir) / path_str
        if candidate.is_file():
            return candidate.resolve()

    candidate = Path.cwd() / "assets" / path_str
    if candidate.is_file():
        return candidate.resolve()

    # Not found, return original path for caller's error handling
    return Path(path_str)


def log_local_path_for_artifact_uri(artifact_uri: str) -> str:
    """Convert an MLflow artifact URI into a local/driver-visible path for printing."""
    if artifact_uri.startswith("dbfs:/"):
        # dbfs:/foo -> /dbfs/foo (driver-visible)
        return "/dbfs/" + artifact_uri[len("dbfs:/"):]
    if artifact_uri.startswith("file:"):
        # file:///abs/path -> /abs/path
        return artifact_uri.split("file://", 1)[-1]
    return artifact_uri


def save_and_log_artifact(path: Path, artifact_subdir: str = "") -> Tuple[str, str]:
    """Log an artifact with MLflow and return (printable_local_path, artifact_uri)."""
    import mlflow  # Local import to avoid hard dependency when unused

    absolute_path = Path(path).resolve()
    if artifact_subdir:
        mlflow.log_artifact(str(absolute_path), artifact_path=artifact_subdir)
        artifact_uri = mlflow.get_artifact_uri(f"{artifact_subdir}/{absolute_path.name}")
    else:
        mlflow.log_artifact(str(absolute_path))
        artifact_uri = mlflow.get_artifact_uri(absolute_path.name)

    return log_local_path_for_artifact_uri(artifact_uri), artifact_uri
