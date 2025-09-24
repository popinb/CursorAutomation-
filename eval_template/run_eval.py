"""
Generic, config-driven LLM evaluation runner

Features
- Loads a dataset (CSV/JSON) with prompts/responses and optional user features
- Loads ground-truth/context files from a directory and injects them into prompts
- Uses user-defined prompt templates per metric
- Supports multiple judge models with majority/average aggregation
- Logs metrics, artifacts, and charts to MLflow

Usage
  python /workspace/eval_template/run_eval.py --config /workspace/eval_template/configs/example.yaml
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import pandas as pd
import yaml
import matplotlib.pyplot as plt

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


# -----------------------------
# Data structures
# -----------------------------


@dataclass
class MetricConfig:
    name: str
    template_path: str
    score_key: str
    threshold: float | int = 1
    aggregation: str = "majority"  # majority | average | min | max
    explanation_key: str = "explanation"


@dataclass
class RunConfig:
    experiment_name: str
    run_name: str
    dataset_path: str
    dataset_format: str = "csv"  # csv | json
    fields: Dict[str, str] = None  # mapping of logical -> column name
    judges: List[str] = None  # e.g., ["gpt-4o"]
    metrics: List[MetricConfig] = None
    ground_truth_dir: Optional[str] = None
    ground_truth_globs: Optional[List[str]] = None
    response_model: Optional[str] = None  # If you want to generate responses


# -----------------------------
# Helpers
# -----------------------------


def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        # Fallback raw bytes decode
        try:
            return path.read_bytes().decode("utf-8", errors="ignore")
        except Exception:
            return ""


def load_ground_truth(ground_truth_dir: Optional[str], patterns: Optional[List[str]]) -> str:
    if not ground_truth_dir:
        return ""

    base = Path(ground_truth_dir)
    if not base.exists():
        return ""

    collected: List[Tuple[str, str]] = []

    if not patterns:
        # default: all readable text-like files
        patterns = ["**/*.txt", "**/*.md", "**/*.json"]

    for pat in patterns:
        for file_path in base.glob(pat):
            if file_path.is_file():
                content = read_text_file(file_path)
                if content.strip():
                    collected.append((str(file_path), content))

    # Concatenate with headers so judges can attribute provenance
    parts: List[str] = []
    for fname, content in collected:
        parts.append(f"===== {fname} =====\n{content}\n")
    return "\n".join(parts)


def load_config(path: str) -> RunConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    metrics: List[MetricConfig] = []
    for m in data.get("metrics", []):
        metrics.append(
            MetricConfig(
                name=m["name"],
                template_path=m["template_path"],
                score_key=m.get("score_key", f"{m['name'].lower()}_score"),
                threshold=m.get("threshold", 1),
                aggregation=m.get("aggregation", "majority"),
                explanation_key=m.get("explanation_key", "explanation"),
            )
        )

    cfg = RunConfig(
        experiment_name=data["experiment_name"],
        run_name=data["run_name"],
        dataset_path=data["dataset"]["path"],
        dataset_format=data["dataset"].get("format", "csv"),
        fields=data.get("fields", {}),
        judges=data.get("judges", ["gpt-4o"]),
        metrics=metrics,
        ground_truth_dir=data.get("ground_truth", {}).get("dir"),
        ground_truth_globs=data.get("ground_truth", {}).get("globs"),
        response_model=data.get("response_generation", {}).get("model"),
    )
    return cfg


def load_dataset(cfg: RunConfig) -> pd.DataFrame:
    if cfg.dataset_format == "csv":
        df = pd.read_csv(cfg.dataset_path)
    elif cfg.dataset_format == "json":
        with open(cfg.dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported dataset format: {cfg.dataset_format}")

    return df


def render_template(template_path: str, variables: Dict[str, Any]) -> str:
    template_text = Path(template_path).read_text(encoding="utf-8")
    try:
        return template_text.format(**variables)
    except KeyError as exc:
        missing = str(exc)
        # Fill missing with blank and try again for robustness
        safe_vars = {k: ("" if v is None else v) for k, v in variables.items()}
        return template_text.format(**safe_vars)


def get_openai_client() -> OpenAI:
    if OpenAI is None:
        raise RuntimeError("openai package is not installed. Please add it to requirements and install.")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment")
    return OpenAI()


def call_judge(model: str, prompt: str, *, temperature: float = 0.0, max_retries: int = 2) -> Dict[str, Any]:
    client = get_openai_client()
    last_exc: Optional[Exception] = None
    for _ in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content or "{}"
            return json.loads(content)
        except Exception as exc:  # pragma: no cover
            last_exc = exc
    # On failure, return error envelope
    return {"error": str(last_exc) if last_exc else "unknown_error"}


def aggregate_scores(values: List[float | int], mode: str) -> float:
    if not values:
        return 0.0
    if mode == "average":
        return float(sum(values) / len(values))
    if mode == "min":
        return float(min(values))
    if mode == "max":
        return float(max(values))
    # majority (default)
    # Round values to nearest int for voting
    rounded = [int(round(float(v))) for v in values]
    counter = collections.Counter(rounded)
    # Pick most common; on tie pick lowest
    return float(max(counter.items(), key=lambda x: (x[1], -x[0]))[0])


def log_metric_histogram(metric_name: str, scores: List[float]) -> None:
    if not scores:
        return
    plt.figure(figsize=(5, 3))
    plt.hist(scores, bins=min(10, max(3, len(set(scores)))))
    plt.title(metric_name)
    plt.xlabel("score")
    plt.ylabel("count")
    mlflow.log_figure(plt.gcf(), f"figures/{metric_name}_hist.png")
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Config-driven LLM evaluation runner")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = load_dataset(cfg)

    # Column mappings with fallbacks
    prompt_col = cfg.fields.get("prompt", "prompt")
    response_col = cfg.fields.get("response", "response")
    upf_col = cfg.fields.get("user_personalization_features", "user_personalization_features")
    ucpf_col = cfg.fields.get("user_context_personalization_features", upf_col)

    ground_truth_text = load_ground_truth(cfg.ground_truth_dir, cfg.ground_truth_globs)

    # Prepare results containers
    per_metric_scores: Dict[str, List[float]] = {m.name: [] for m in cfg.metrics}
    per_row_details: List[Dict[str, Any]] = []

    mlflow.set_experiment(cfg.experiment_name)
    with mlflow.start_run(run_name=cfg.run_name):
        mlflow.log_params(
            {
                "dataset_path": cfg.dataset_path,
                "dataset_format": cfg.dataset_format,
                "judges": ",".join(cfg.judges or []),
                "num_rows": len(df),
            }
        )

        for row_idx, row in df.iterrows():
            variables = {
                "prompt": row.get(prompt_col, ""),
                "response": row.get(response_col, ""),
                "user_personalization_features": row.get(upf_col, ""),
                "user_context_personalization_features": row.get(ucpf_col, ""),
                "ground_truth": ground_truth_text,
            }

            row_detail: Dict[str, Any] = {"index": int(row_idx), "metrics": {}}

            for metric in cfg.metrics:
                eval_prompt = render_template(metric.template_path, variables)

                model_outputs: List[Dict[str, Any]] = []
                scores: List[float] = []

                for model in (cfg.judges or ["gpt-4o"]):
                    result_json = call_judge(model, eval_prompt)
                    model_outputs.append({"model": model, "result": result_json})

                    if isinstance(result_json, dict) and metric.score_key in result_json:
                        try:
                            scores.append(float(result_json[metric.score_key]))
                        except Exception:
                            pass

                final_score = aggregate_scores(scores, metric.aggregation)
                status = float(final_score) >= float(metric.threshold)

                per_metric_scores[metric.name].append(final_score)
                row_detail["metrics"][metric.name] = {
                    "final_score": final_score,
                    "threshold": metric.threshold,
                    "status": status,
                    "votes": scores,
                    "per_model": model_outputs,
                }

            per_row_details.append(row_detail)

        # Attach scores into DataFrame and log
        for metric in cfg.metrics:
            df[metric.name] = per_metric_scores[metric.name]
            mean_score = sum(per_metric_scores[metric.name]) / max(1, len(per_metric_scores[metric.name]))
            mlflow.log_metric(f"{metric.name}/mean", float(mean_score))
            log_metric_histogram(metric.name, [float(s) for s in per_metric_scores[metric.name]])

        # Save artifacts
        out_dir = Path("./eval_results")
        out_dir.mkdir(parents=True, exist_ok=True)
        results_csv = out_dir / f"results_{cfg.run_name}.csv"
        details_json = out_dir / f"details_{cfg.run_name}.json"
        df.to_csv(results_csv, index=False)
        details_json.write_text(json.dumps(per_row_details, indent=2), encoding="utf-8")

        mlflow.log_artifact(str(results_csv))
        mlflow.log_artifact(str(details_json))

    print(f"Saved results to {results_csv}")
    print(f"Saved details to {details_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

