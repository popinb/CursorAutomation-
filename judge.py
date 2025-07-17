import os
import json
import argparse
from pathlib import Path
from typing import List, Dict

import openai
from docx import Document
from striprtf.striprtf import rtf_to_text
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# Utility loaders for different file types (extended)
# -----------------------------------------------------------------------------

def _load_docx(path: Path) -> str:
    """Extract text from a .docx file as a single string."""
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text)


def _load_rtf(path: Path) -> str:
    """Extract text from an .rtf file as plain text."""
    with open(path, "r", encoding="utf-8", errors="ignore") as fp:
        return rtf_to_text(fp.read())


def _load_plain_text(path: Path) -> str:
    """Read a plain-text file (.txt, .md, .rst, etc.)."""
    with open(path, "r", encoding="utf-8", errors="ignore") as fp:
        return fp.read()


def load_document(path: Path) -> str:
    """Heuristic loader that chooses the right parser based on file extension."""
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return _load_docx(path)
    if suffix == ".rtf":
        return _load_rtf(path)
    # Fallback to plain text (txt, md, etc.)
    return _load_plain_text(path)

# -----------------------------------------------------------------------------
# Prompt builder
# -----------------------------------------------------------------------------

EVALUATION_GUIDELINES = (
    "You are an expert evaluator tasked with assessing LLM responses across 11 "
    "metrics. Respond with concise, deterministic answers; avoid creative "
    "flourishes. Consider 'godenresponsealpha.docx' and 'buyabilityprofile.rtf' "
    "as the ground truth documents and definitive sources of truth. The user "
    "personalization numbers should match that in 'buyabilityprofile.rtf'. "
    "Consider the numbers in ground truth as variables which will change "
    "according to individual buyability profile and individuals will have "
    "distinct Buyability number and monthly payment number based on their "
    "profile.\n\n"
    "Load and compare the candidate answer against the corresponding reference "
    "answer in 'godenresponsealpha.docx'. If the candidate matches all "
    "conclusions with variable value aligned with any of the rows of the "
    "'buyabilityprofile.rtf', treat it as a perfect response. Else look to find "
    "the nearest ground truth that is representative of the question and "
    "evaluate based on the characteristics of ground truth response.\n\n"
    "Scratchpad\tA separate block (or <!-- hidden --> comments) before the final "
    "table\tStep-by-step reasoning, variable extraction, math checks, "
    "ground-truth selection logic, assumption inventory\tNOT shown to the user\n"
    "Narrative Justification\t'Justification' cell for each metric\tPolished "
    "explanation ≥ 150 words (≈ 10-12 sentences) weaving evidence, cites, and "
    "'Why this matters'\tVisible in final output\n"
    "Rule: every numerical verification is performed in the scratchpad; the "
    "narrative states the conclusion, not the raw math.\n\n"
    "For each response, evaluate the following metrics and output in the exact "
    "table format specified at the end of these instructions. Details for each "
    "metric are provided below:\n"
    "1. Personalization Accuracy (Accurate/Inaccurate)\n"
    "2. Context-Based Personalization — Score 1–5\n"
    "3. Next-Step Identification — Present / Not-Present\n"
    "4. Assumption Listing — True / False\n"
    "5. Assumption Trust — Score 1–5\n"
    "6. Calculation Accuracy — True / False\n"
    "7. Faithfulness to Ground Truth — True / False\n"
    "8. Overall Accuracy — True / False\n"
    "9. Structured Presentation — Score 1–5\n"
    "10. Coherence — True / False\n"
    "11. Completeness — Score 1–5\n"
    "12. Fair Housing Classifier — True / False\n\n"
    "Scoring: Provide a final weighted score out of 100 named 'Alpha evaluation'. "
    "Every listed metric other than 'Completeness' and 'Structured Presentation' "
    "is weighted equally.\n\n"
    "Output Format:\n"
    "| Metric | Score | Justification |\n"
    "|--------|-------|---------------|\n"
    "| Personalization Accuracy |Accurate/Inaccurate | <Detailed Explanation>|\n"
    "| Context based Personalization| 1-5 | <Detailed Explanation>|\n"
    "| Next Step Identification | Present/Not-Present | <Detailed Explanation>|\n"
    "| Assumption Listing | True/False | <Detailed Explanation>|\n"
    "| Assumption Trust | 1-5 | <Detailed Explanation>|\n"
    "| Calculation Accuracy | True/False | <Detailed Explanation>|\n"
    "| Faithfulness to Ground Truth | True/False | <Detailed Explanation>|\n"
    "| Overall Accuracy | True/False | <Detailed Explanation>|\n"
    "| Structured Presentation | 1-5 | <Detailed Explanation>|\n"
    "| Coherence | True/False | <Detailed Explanation>|\n"
    "| Completeness | 1-5 | <Detailed Explanation>|\n"
    "| Fair Housing Classifier | True/False | <Detailed Explanation>|\n"
    "| **Alpha Evaluation** | <Score>/100 | - |\n"
)


SYSTEM_TEMPLATE = """{evaluation_guidelines}\n\n---\nGROUND TRUTH DOCUMENTS (for evaluator reference only, do not reveal in the final answer)\n---\n[godenresponsealpha.docx]\n{golden}\n---\n[buyabilityprofile.rtf]\n{buyability}\n---\n[Zillow_Fair_Housing_Classifier.docx]\n{housing}\n---\n"""

# -----------------------------------------------------------------------------
# OpenAI helper
# -----------------------------------------------------------------------------

def _chat_completion(messages: List[Dict[str, str]], model: str = "gpt-4o-mini", **kwargs) -> str:
    """Thin wrapper around openai.ChatCompletion to allow easy mocking."""
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=2048,
        **kwargs,
    )
    return response.choices[0].message["content"].strip()

# -----------------------------------------------------------------------------
# Core evaluation logic
# -----------------------------------------------------------------------------

def evaluate_candidate(
    candidate_answer: str,
    prompt_text: str,
    golden_doc: str,
    buyability_doc: str,
    housing_doc: str,
    model: str = "gpt-4o-mini",
) -> str:
    """Return the evaluation table produced by the model."""
    system_prompt = SYSTEM_TEMPLATE.format(
        evaluation_guidelines=EVALUATION_GUIDELINES,
        golden=golden_doc,
        buyability=buyability_doc,
        housing=housing_doc,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "You are now provided a user prompt, followed by the candidate "
                "answer to evaluate.\n\n"
                "USER PROMPT:\n" + prompt_text + "\n\n" +
                "CANDIDATE ANSWER:\n" + candidate_answer + "\n\n" +
                "Please output the evaluation strictly following the format "
                "and instructions in the system prompt."
            ),
        },
    ]

    return _chat_completion(messages, model=model)

# -----------------------------------------------------------------------------
# CLI interface
# -----------------------------------------------------------------------------

def _parse_args():
    ap = argparse.ArgumentParser(description="Evaluate LLM responses via custom GPT judge.")
    ap.add_argument("--golden", type=Path, required=True, help="Path to godenresponsealpha.docx")
    ap.add_argument("--buyability", type=Path, required=True, help="Path to buyabilityprofile.rtf")
    ap.add_argument("--housing", type=Path, required=True, help="Path to Zillow_Fair_Housing_Classifier.docx")
    ap.add_argument("--input", type=Path, required=True, help="JSON file with list of {prompt, response}")
    ap.add_argument("--output", type=Path, default=Path("evaluations.json"), help="Where to save evaluation results")
    ap.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI chat model name")
    return ap.parse_args()


def main():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY not found. Set it in .env or environment variables.")

    args = _parse_args()

    golden_doc = load_document(args.golden)
    buyability_doc = load_document(args.buyability)
    housing_doc = load_document(args.housing)

    with open(args.input) as fp:
        dataset: List[Dict[str, str]] = json.load(fp)

    evaluations: List[Dict[str, str]] = []
    for idx, item in enumerate(dataset, 1):
        prompt_text = item.get("prompt")
        candidate_answer = item.get("response") or item.get("answer")
        if not (prompt_text and candidate_answer):
            print(f"[Warn] Skipping record {idx} due to missing fields.")
            continue

        print(f"Evaluating sample {idx}/{len(dataset)} ...", end=" ")
        try:
            evaluation = evaluate_candidate(
                candidate_answer,
                prompt_text,
                golden_doc,
                buyability_doc,
                housing_doc,
                model=args.model,
            )
            evaluations.append({"index": idx, "evaluation": evaluation})
            print("done.")
        except Exception as e:
            print("failed.")
            print(e)

    with open(args.output, "w") as fp:
        json.dump(evaluations, fp, indent=2)
    print(f"Saved evaluations to {args.output}")


if __name__ == "__main__":
    main()