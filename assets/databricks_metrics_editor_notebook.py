# Databricks notebook source
# COMMAND ----------
# CELL 1: Import libraries and get current user

import pandas as pd
import os
import json
from datetime import datetime

try:
    current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
except Exception:
    current_user = "unknown_user"

# COMMAND ----------
# CELL 2: Create widgets for file paths

dbutils.widgets.text(
    "evaluation_data_path",
    "evaluation_data.csv",
    "?? Evaluation Data (CSV)"
)

dbutils.widgets.text(
    "metrics_config_path",
    "sample_metrics_config_simplified.csv",
    "?? Metrics Configuration (CSV)"
)

dbutils.widgets.text(
    "ground_truth_files",
    "ground_truth_accuracy.csv;ground_truth_safety.csv",
    "?? Ground Truth Files (semicolon separated)"
)

EVAL_DATA_PATH = dbutils.widgets.get("evaluation_data_path")
METRICS_CONFIG_PATH = dbutils.widgets.get("metrics_config_path")
GROUND_TRUTH_FILES_STRING = dbutils.widgets.get("ground_truth_files")

# COMMAND ----------
# CELL 3: Define helper functions


def parse_ground_truth_files(gt_files_string):
    """Parse semicolon or comma separated ground truth file paths."""
    if not gt_files_string or not str(gt_files_string).strip():
        return []

    gt_files_string = str(gt_files_string)

    files = []
    for separator in [";", ","]:
        if separator in gt_files_string:
            files = [f.strip() for f in gt_files_string.split(separator) if f.strip()]
            break

    if not files:
        files = [gt_files_string.strip()]

    return files


def find_file_in_workspace(filename):
    """Auto-detect file in common Databricks locations."""
    if os.path.isabs(filename) and os.path.exists(filename):
        return filename

    base_filename = os.path.basename(filename)

    try:
        user_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
    except Exception:
        user_name = None

    search_locations = []

    if user_name:
        search_locations.extend([
            f"/Workspace/Users/{user_name}/{base_filename}",
            f"/Workspace/Users/{user_name}/{filename}",
        ])

    search_locations.extend([
        filename,
        base_filename,
        f"./{base_filename}",
        f"/Workspace/Shared/{base_filename}",
        f"/dbfs/FileStore/{base_filename}",
        f"/tmp/{base_filename}",
        f"/Workspace/{base_filename}",
    ])

    for location in search_locations:
        if os.path.exists(location):
            return location

    return None


def load_csv_file(filename, file_type="data"):
    """Load CSV file with auto-detection. Returns DataFrame and resolved path."""
    file_path = None
    try:
        file_path = find_file_in_workspace(filename)

        if not file_path:
            return None, None

        df = pd.read_csv(file_path)
        return df, file_path

    except Exception:
        return None, file_path


def resolve_dbfs_path(path):
    """Convert dbfs:/ URIs to /dbfs/ filesystem paths for pandas."""
    if not path:
        return None
    if path.startswith("dbfs:/"):
        return path.replace("dbfs:/", "/dbfs/")
    return path


def to_dbfs_uri(path):
    """Convert /dbfs/ filesystem path back to dbfs:/ URI for display."""
    if not path:
        return None
    if path.startswith("/dbfs/"):
        return "dbfs:" + path[len("/dbfs") :]
    return path


def ensure_directory(path):
    """Create directory for the provided path if it does not exist."""
    if not path:
        return

    directory = os.path.dirname(path)
    if not directory:
        return

    if path.startswith("/dbfs/"):
        if directory.startswith("/dbfs/"):
            dbfs_dir = "dbfs:/" + directory[len("/dbfs/") :]
        else:
            dbfs_dir = "dbfs:/"
        try:
            dbutils.fs.mkdirs(dbfs_dir)
        except Exception:
            pass
    else:
        try:
            os.makedirs(directory, exist_ok=True)
        except Exception:
            pass


def autosave_dataframe(df, target_path):
    """Persist DataFrame to the requested path and return display-friendly path."""
    resolved_path = resolve_dbfs_path(target_path)
    if not resolved_path:
        return None

    ensure_directory(resolved_path)

    try:
        df.to_csv(resolved_path, index=False)
        return to_dbfs_uri(resolved_path)
    except Exception:
        return None

# COMMAND ----------
# CELL 4: Load evaluation data

evaluation_data_df, evaluation_data_path_resolved = load_csv_file(EVAL_DATA_PATH, "evaluation data")

if evaluation_data_df is None:
    evaluation_data_df = pd.DataFrame({
        "sample_id": [1, 2, 3],
        "prompt": [
            "What is the capital of France?",
            "Explain machine learning in simple terms",
            "How do I bake a chocolate cake?"
        ],
        "response": [
            "The capital of France is Paris, a beautiful city known for its culture and history.",
            "Machine learning is a type of AI where computers learn patterns from data to make predictions.",
            "To bake a chocolate cake, mix flour, cocoa, eggs, and sugar, then bake at 350?F for 30 minutes."
        ]
    })

EVALUATION_DATA = evaluation_data_df
EVALUATION_DATA_PATH_RESOLVED = evaluation_data_path_resolved

# COMMAND ----------
# CELL 5: Load metrics configuration

metrics_config_df, metrics_config_path_resolved = load_csv_file(METRICS_CONFIG_PATH, "metrics config")

if metrics_config_df is None:
    metrics_config_df = pd.DataFrame({
        "metric_name": ["faithfulness", "relevance", "coherence"],
        "metric_type": ["llm_judge", "llm_judge", "llm_judge"],
        "weight": [1.0, 0.8, 0.6],
        "enabled": [True, True, False]
    })

METRICS_CONFIG_DATA = metrics_config_df.copy()
METRICS_CONFIG_PATH_RESOLVED = metrics_config_path_resolved

# COMMAND ----------
# CELL 6: Load ground truth files

GROUND_TRUTH_FILES_LIST = parse_ground_truth_files(GROUND_TRUTH_FILES_STRING)
ground_truth_data = {}

if GROUND_TRUTH_FILES_LIST:
    for file_path in GROUND_TRUTH_FILES_LIST:
        if not file_path:
            continue

        filename = os.path.basename(file_path)
        found_path = find_file_in_workspace(file_path)

        if found_path:
            try:
                df = pd.read_csv(found_path)
                ground_truth_data[filename] = df
            except Exception:
                pass

GROUND_TRUTH_DATA = ground_truth_data

# COMMAND ----------
# CELL 7: Configure auto-save location

default_metrics_filename = os.path.basename(METRICS_CONFIG_PATH) if METRICS_CONFIG_PATH else "metrics_config.csv"
fallback_autosave_uri = f"dbfs:/FileStore/{default_metrics_filename}"

dbutils.widgets.text(
    "metrics_auto_save_path",
    fallback_autosave_uri,
    "?? Auto Save Location"
)

AUTO_SAVE_TARGET = dbutils.widgets.get("metrics_auto_save_path") or fallback_autosave_uri
AUTO_SAVE_TARGET = AUTO_SAVE_TARGET.strip()

# COMMAND ----------
# CELL 8: Interactive metrics editor with instant auto-save

import numpy as np
from ipywidgets import (
    VBox,
    HBox,
    Dropdown,
    Text,
    FloatText,
    Button,
    HTML,
    Output,
    Layout,
    ToggleButtons
)
from IPython.display import display, clear_output


def cast_value_to_dtype(value, dtype):
    """Convert widget input into the target dtype."""
    if value is None:
        return None

    if isinstance(value, float) and np.isnan(value):
        return None

    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return None
        value = stripped

    try:
        if pd.api.types.is_bool_dtype(dtype):
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower() in ["true", "1", "yes", "y", "enabled"]
            return bool(value)

        if pd.api.types.is_integer_dtype(dtype):
            return int(float(value))

        if pd.api.types.is_float_dtype(dtype):
            return float(value)

        return str(value)
    except Exception:
        return value


def format_widget_value(value, dtype):
    """Prepare a value for display inside a widget."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        if pd.api.types.is_bool_dtype(dtype):
            return False
        if pd.api.types.is_numeric_dtype(dtype):
            return None
        return ""

    if pd.api.types.is_bool_dtype(dtype):
        return bool(value)

    if pd.api.types.is_numeric_dtype(dtype):
        try:
            return float(value)
        except Exception:
            return None

    return str(value)


def build_metrics_editor():
    """Render an interactive form for editing metrics with auto-save."""
    global METRICS_CONFIG_DATA

    if METRICS_CONFIG_DATA is None or METRICS_CONFIG_DATA.empty:
        METRICS_CONFIG_DATA = pd.DataFrame(columns=["metric_name", "metric_type", "weight", "enabled"])

    columns = list(METRICS_CONFIG_DATA.columns)

    if not columns:
        display(HTML("<p><strong>No columns detected in the metrics configuration.</strong></p>"))
        return

    status_html = HTML()
    stats_html = HTML(
        """
        <div style=\"background:#e8f0fe;border-radius:8px;padding:12px 16px;margin-bottom:12px;\">
            <strong>Metrics Editor</strong><br>
            <span>Any change you make is saved immediately to the selected file.</span>
        </div>
        """
    )

    table_output = Output()

    def refresh_table_display():
        with table_output:
            clear_output(wait=True)
            display(METRICS_CONFIG_DATA)

    refresh_table_display()

    updating_form = {"active": False}
    updating_selector = {"active": False}

    row_selector = Dropdown(description="Metric", layout=Layout(width="400px"))

    def describe_row(idx):
        label_col = "metric_name" if "metric_name" in METRICS_CONFIG_DATA.columns else None
        if label_col and pd.notna(METRICS_CONFIG_DATA.loc[idx, label_col]):
            label = str(METRICS_CONFIG_DATA.loc[idx, label_col])
        else:
            label = f"Row {idx + 1}"
        return f"{idx + 1}. {label}"

    def refresh_selector(selected_index=None):
        updating_selector["active"] = True
        if not METRICS_CONFIG_DATA.empty:
            options = [(describe_row(idx), idx) for idx in range(len(METRICS_CONFIG_DATA))]
            row_selector.options = options
            if selected_index is not None and selected_index < len(METRICS_CONFIG_DATA):
                row_selector.value = selected_index
            else:
                row_selector.value = options[0][1]
        else:
            row_selector.options = [("No metrics yet", None)]
            row_selector.value = None
        updating_selector["active"] = False

    field_widgets = {}

    for column in columns:
        dtype = METRICS_CONFIG_DATA[column].dtype

        if pd.api.types.is_bool_dtype(dtype):
            widget = ToggleButtons(
                options=[("Enabled", True), ("Disabled", False)],
                description=column,
                button_style="info",
                layout=Layout(width="auto")
            )
        elif pd.api.types.is_numeric_dtype(dtype):
            widget = FloatText(description=column, layout=Layout(width="auto"))
        else:
            widget = Text(description=column, layout=Layout(width="auto"))

        field_widgets[column] = widget

    def load_row_into_form(row_index):
        updating_form["active"] = True
        if row_index is None or METRICS_CONFIG_DATA.empty:
            for column, widget in field_widgets.items():
                dtype = METRICS_CONFIG_DATA[column].dtype if column in METRICS_CONFIG_DATA else object
                widget.value = format_widget_value(None, dtype)
        else:
            for column, widget in field_widgets.items():
                dtype = METRICS_CONFIG_DATA[column].dtype
                value = METRICS_CONFIG_DATA.iloc[row_index][column] if column in METRICS_CONFIG_DATA.columns else None
                widget.value = format_widget_value(value, dtype)
        updating_form["active"] = False

    def update_stats_message(saved_path=None):
        total_metrics = len(METRICS_CONFIG_DATA)
        last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        path_text = saved_path or to_dbfs_uri(resolve_dbfs_path(AUTO_SAVE_TARGET)) or "not saved"
        status_html.value = (
            f"<div style='margin-top:8px;padding:10px;border-radius:6px;background:#f8f9fa;'>"
            f"<strong>Total metrics:</strong> {total_metrics} | "
            f"<strong>Columns:</strong> {', '.join(columns)}<br>"
            f"<strong>Last saved:</strong> {last_updated}<br>"
            f"<strong>Location:</strong> {path_text}"
            "</div>"
        )

    def persist_changes():
        saved_path = autosave_dataframe(METRICS_CONFIG_DATA, AUTO_SAVE_TARGET)
        refresh_table_display()
        update_stats_message(saved_path)

    def make_change_handler(column):
        dtype = METRICS_CONFIG_DATA[column].dtype

        def handle(change):
            if updating_form["active"] or updating_selector["active"]:
                return

            row_index = row_selector.value
            if row_index is None:
                return

            new_value = cast_value_to_dtype(change["new"], dtype)
            METRICS_CONFIG_DATA.at[row_index, column] = new_value
            persist_changes()
            if column == "metric_name":
                refresh_selector(row_index)

        return handle

    for column, widget in field_widgets.items():
        widget.observe(make_change_handler(column), names="value")

    def on_row_change(change):
        if updating_selector["active"]:
            return
        load_row_into_form(change["new"])

    row_selector.observe(on_row_change, names="value")

    def on_add_click(_):
        global METRICS_CONFIG_DATA
        new_row = {column: None for column in columns}
        METRICS_CONFIG_DATA = pd.concat([METRICS_CONFIG_DATA, pd.DataFrame([new_row])], ignore_index=True)
        refresh_selector(len(METRICS_CONFIG_DATA) - 1)
        load_row_into_form(len(METRICS_CONFIG_DATA) - 1)
        persist_changes()

    def on_duplicate_click(_):
        global METRICS_CONFIG_DATA
        idx = row_selector.value
        if idx is None:
            return
        duplicated_row = METRICS_CONFIG_DATA.iloc[idx].to_dict()
        METRICS_CONFIG_DATA = pd.concat([METRICS_CONFIG_DATA, pd.DataFrame([duplicated_row])], ignore_index=True)
        refresh_selector(len(METRICS_CONFIG_DATA) - 1)
        load_row_into_form(len(METRICS_CONFIG_DATA) - 1)
        persist_changes()

    def on_delete_click(_):
        global METRICS_CONFIG_DATA
        idx = row_selector.value
        if idx is None:
            return
        METRICS_CONFIG_DATA = METRICS_CONFIG_DATA.drop(index=idx).reset_index(drop=True)
        if METRICS_CONFIG_DATA.empty:
            refresh_selector(None)
            load_row_into_form(None)
        else:
            next_index = min(idx, len(METRICS_CONFIG_DATA) - 1)
            refresh_selector(next_index)
            load_row_into_form(row_selector.value)
        persist_changes()

    buttons = HBox([
        Button(description="Add metric", icon="plus", button_style="success", layout=Layout(width="150px"), tooltip="Add a new blank metric"),
        Button(description="Duplicate", icon="copy", layout=Layout(width="150px"), tooltip="Copy the selected metric"),
        Button(description="Delete", icon="trash", button_style="danger", layout=Layout(width="150px"), tooltip="Delete the selected metric")
    ])

    add_button, duplicate_button, delete_button = buttons.children
    add_button.on_click(on_add_click)
    duplicate_button.on_click(on_duplicate_click)
    delete_button.on_click(on_delete_click)

    form_layout = VBox(list(field_widgets.values()), layout=Layout(width="100%"))

    refresh_selector(0)
    load_row_into_form(row_selector.value)
    update_stats_message(to_dbfs_uri(resolve_dbfs_path(AUTO_SAVE_TARGET)))

    editor_ui = VBox(
        [
            stats_html,
            row_selector,
            form_layout,
            buttons,
            status_html,
            HTML("<hr>"),
            HTML("<strong>Preview of the saved file</strong>"),
            table_output,
        ],
        layout=Layout(width="100%", max_width="960px"),
    )

    display(editor_ui)


build_metrics_editor()

# COMMAND ----------
# CELL 9: ?? View final metrics snapshot

print("=" * 80)
print("?? EDITED METRICS CONFIGURATION")
print("=" * 80)
print(f"Total Metrics: {len(METRICS_CONFIG_DATA)}")
print(f"Columns: {', '.join(METRICS_CONFIG_DATA.columns.tolist())}")
print("=" * 80)

display(METRICS_CONFIG_DATA)
