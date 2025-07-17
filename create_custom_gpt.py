import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from openai import OpenAI

# Instantiate a single OpenAI client
client = OpenAI()

# Paths to reference docs (adjust if different)
DOCS = [
    Path("docs/godenresponsealpha.txt"),
    Path("docs/buyabilityprofile.rtf"),
    Path("docs/Zillow_Fair_Housing_Classifier.txt"),
]

ASSISTANT_NAME = "BuyAbility Judge GPT"

EVALUATOR_INSTRUCTIONS = (
    "You are an expert evaluator tasked with assessing LLM responses across 11 metrics. "
    "Respond with concise, deterministic answers; avoid creative flourishes. "
    "Consider 'godenresponsealpha.docx' and 'buyabilityprofile.rtf' as the ground truth documents "
    "and definitive sources of truth. The user personalization numbers should match that in "
    "'buyabilityprofile.rtf'. The numbers in ground truth are variables which change per profile.\n\n"
    "Load and compare the candidate answer against the corresponding reference answer in "
    "'godenresponsealpha.docx'. If the candidate matches all conclusions with variable value aligned "
    "with any rows of 'buyabilityprofile.rtf', treat it as a perfect response. Else evaluate using the "
    "nearest ground truth. Follow the full rubric provided in the original project README.\n\n"
    "When responding, strictly output the evaluation table and final Alpha evaluation score as defined."
)

MODEL_NAME = "gpt-4o"


def upload_files(paths: List[Path]) -> List[str]:
    """Upload files to OpenAI and return the list of file IDs."""
    file_ids = []
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(p)
        print(f"Uploading {p} ...", end=" ")
        with open(p, "rb") as fp:
            resp = client.files.create(file=fp, purpose="assistants")
            file_ids.append(resp["id"])
        print("done.")
    return file_ids


def create_assistant(file_ids: List[str]):
    """Create the assistant and print its ID and dashboard link."""
    assistant = client.beta.assistants.create(
        name=ASSISTANT_NAME,
        instructions=EVALUATOR_INSTRUCTIONS,
        tools=[{"type": "retrieval"}],
        model=MODEL_NAME,
        file_ids=file_ids,
    )
    print("\nAssistant created!")
    print("ID:", assistant["id"])
    print("Manage it in the UI at: https://platform.openai.com/assistants")


def main():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("Set OPENAI_API_KEY in your env or .env file")

    file_ids = upload_files(DOCS)
    create_assistant(file_ids)


if __name__ == "__main__":
    main()