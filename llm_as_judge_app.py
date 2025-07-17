"""LLM-as-a-Judge evaluation app
--------------------------------
This lightweight app turns any dataset of reference Q&A pairs into an
automatic evaluation harness powered by the official OpenAI "LLM-as-a-Judge"
pattern (see https://github.com/openai/evals).

The script expects a JSON-Lines file in which each line has **at minimum**
the following keys:

    {
      "id": "optional-case-id",
      "input": "<user question / prompt>",
      "ideal": "<ground-truth or expert answer>",
      "output": "<model under test answer>"  # may be filled in later
    }

If the *output* field is missing, the script will query the **Custom GPT**
(or whichever model name you pass via `--candidate-model`) to obtain an
answer first – this allows you to both *generate* and *evaluate* answers in
one pass.  Otherwise, it will skip generation and immediately grade the
existing answer.

For every row we call `pydantic_evals.evaluators.LLMJudge` which in turn
creates a short structured rubric prompt for GPT-4o (or another model you
specify with `--judge-model`).  The judge returns structured JSON which
contains a boolean pass/fail, a floating point score 0-1, and a free-text
reason.

A summary CSV/JSON with all scores is written to the path provided via
`--out` (defaults to `judge_results.json`).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from pydantic_evals.evaluators import EvaluatorContext, LLMJudge

# ---------------------------------------------------------------------------
# A tiny helper to call the candidate model (your Custom GPT)
# ---------------------------------------------------------------------------

async def _run_candidate_model(client: AsyncOpenAI, model_name: str, user_prompt: str) -> str:
    """Gets the model's chat completion with *no* system prompt supervision."""
    response = await client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Core evaluation wrapper
# ---------------------------------------------------------------------------

class LLMJudgeApp:
    """Thin wrapper around the pydantic-evals `LLMJudge` evaluator."""

    def __init__(
        self,
        judge_model: str = "openai:gpt-4o",
        candidate_model: str = "gpt-3.5-turbo",
        rubric: Optional[str] = None,
    ) -> None:
        self.judge_model = judge_model
        self.candidate_model = candidate_model

        # A short default rubric; feel free to edit for your domain.
        self.rubric = (
            rubric
            or """
You are grading a model answer.
Task: Compare the candidate answer with the ideal reference answer.
Rules:
1. If the candidate contradicts, hallucinates, or omits key information it fails.
2. If it is factually consistent and at least as complete and correct as the ideal answer, it passes.
3. Provide a score between 0 (completely wrong) and 1 (perfect) that reflects the answer quality.
Return as JSON: {"pass": <true|false>, "score": <float 0-1>, "reason": "<short explanation>"}.
"""
        )

        # Construct the LLMJudge evaluator object (async-capable).
        self.judge_evaluator = LLMJudge(
            rubric=self.rubric,
            model=self.judge_model,
            include_input=True,
            include_expected_output=True,
            # Ask for both pass/fail + score + reason
            score={"include_reason": False},
            assertion={"include_reason": True},
        )

        # Async OpenAI client reused across requests
        self._client = AsyncOpenAI()

    async def _grade_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Grades a single case and returns a dict with results."""
        input_prompt = case["input"]

        # Generate candidate answer if missing.
        if not case.get("output"):
            case["output"] = await _run_candidate_model(
                self._client, self.candidate_model, input_prompt
            )

        # Build evaluation context for LLMJudge
        ctx = EvaluatorContext(
            name=str(case.get("id")),
            inputs=input_prompt,
            output=case["output"],
            expected_output=case["ideal"],
            metadata=None,
            duration=0.0,
            # SpanTree not used by this evaluator; set to None safely.
            _span_tree=None,  # type: ignore[arg-type]
            attributes={},
            metrics={},
        )

        # Run judge (async)
        result_map = await self.judge_evaluator.evaluate_async(ctx)

        # `result_map` contains keys like "llmjudge_score" and/or "llmjudge_pass"
        return {
            "id": case.get("id"),
            "input": input_prompt,
            "ideal": case["ideal"],
            "output": case["output"],
            **result_map,
        }

    async def evaluate_dataset(self, path: Path) -> List[Dict[str, Any]]:
        """Load the JSONL dataset and evaluate every entry."""
        cases: List[Dict[str, Any]] = [json.loads(l) for l in path.read_text().splitlines() if l]
        graded: List[Dict[str, Any]] = []
        for case in cases:
            graded.append(await self._grade_case(case))
        return graded


# ---------------------------------------------------------------------------
# Command-line entry-point
# ---------------------------------------------------------------------------

async def _main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM-as-a-Judge evaluations on a JSONL file.")
    parser.add_argument("dataset", type=Path, help="Path to JSONL dataset file")
    parser.add_argument("--out", type=Path, default=Path("judge_results.json"), help="Output path")
    parser.add_argument("--judge-model", default="openai:gpt-4o", help="Model used as the judge")
    parser.add_argument("--candidate-model", default="gpt-3.5-turbo", help="Model under test")
    args = parser.parse_args()

    app = LLMJudgeApp(judge_model=args.judge_model, candidate_model=args.candidate_model)
    results = await app.evaluate_dataset(args.dataset)

    # Write as JSON
    args.out.write_text(json.dumps(results, indent=2))
    passed = sum(1 for r in results if r.get("llmjudge_pass"))
    print(f"Finished. {passed}/{len(results)} answers passed. Detailed results saved to {args.out}.")


if __name__ == "__main__":
    # Ensure the user has an OPENAI_API_KEY set
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Please set the OPENAI_API_KEY environment variable!")

    asyncio.run(_main())