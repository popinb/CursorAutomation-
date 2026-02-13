# MAGIC %md
# MAGIC ## Cell 5: Core Classes
# MAGIC
# MAGIC **Purpose**: This cell defines the fundamental data structures and classes used throughout the evaluation system.
# MAGIC
# MAGIC **What it does**:
# MAGIC - Defines the MetricType enumeration for different types of evaluation metrics (binary, scale, percentage)
# MAGIC - Creates the MetricConfig dataclass to store configuration for each evaluation metric
# MAGIC - **NEW**: Enhanced to always use ALL columns from ground truth files for richer context
# MAGIC - Sets up the foundation for the evaluation logic in subsequent cells
# MAGIC
# MAGIC **When to run**: Run this cell after Cell 4 to initialize the core data structures
# MAGIC
# MAGIC **Expected output**: Confirmation message that core classes have been defined with all-columns support

# COMMAND ----------

class MetricType(Enum):
    BINARY = "binary"
    SCALE_1_5 = "1-5_scale"
    PERCENTAGE = "percentage"

@dataclass
class MetricConfig:
    name: str
    description: str
    metric_type: MetricType
    prompt_template: str
    threshold: float
    ground_truth_column: str  # Keep for backward compatibility, but will use all columns
    ground_truth_file_path: str = ""
    use_all_columns: bool = True  # Always True - parse all columns for richer context

print("✅ Core classes defined with enhanced all-columns ground truth support")
print("🎯 All metrics will now access ALL columns from ground truth files for richer evaluation context")