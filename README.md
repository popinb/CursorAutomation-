# Custom GPT Judge

This project provides a *deterministic* evaluator—powered by OpenAI GPT models—for scoring LLM answers with your bespoke rubric.

---

## 🗂️  Project Layout

- `judge.py` – CLI script that loads your ground-truth docs and evaluates candidate answers.
- `requirements.txt` – Python dependencies.
- `docs/` – *Put your reference docs here*  
  - `godenresponsealpha.docx`  
  - `buyabilityprofile.rtf`  
  - `Zillow_Fair_Housing_Classifier.docx`
- `.env` – Store your `OPENAI_API_KEY` here (see **Setup** below).
- Your dataset – JSON file containing the prompts and candidate answers to score.

## 📦  Setup

```bash
# 1. Clone or download this repo
cd /path/to/workspace

# 2. Create & activate a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate

# 3. Install requirements
pip install -r requirements.txt

# 4. Add your OpenAI key (create a copy of the template)
cp .env.example .env
# Edit .env and add: OPENAI_API_KEY=sk-...

# 5. Drop your ground-truth docs into a `docs/` folder (or elsewhere)
```

> **Note**: `judge.py` extracts text from `.docx` (via `python-docx`) and `.rtf` (via `striprtf`).

## 📑  Preparing the evaluation dataset

Create a JSON array with objects shaped like:

```json
[
  {
    "prompt": "<user question>",
    "response": "<candidate answer>"
  },
  ... (one object per sample) ...
]
```

Save it somewhere, e.g. `data/test_set.json`.

## 🚀  Running the judge

```bash
python judge.py \
  --golden docs/godenresponsealpha.docx \
  --buyability docs/buyabilityprofile.rtf \
  --housing docs/Zillow_Fair_Housing_Classifier.docx \
  --input data/test_set.json \
  --output results/evaluations.json \
  --model gpt-4o-mini
```

- The script iterates through every sample and calls the chat model *once* per sample with **temperature = 0** for determinism.
- Evaluations are saved as an array of objects (index + raw markdown table) in `evaluations.json`.

## ⚖️  How it works

1. **System prompt** – Encodes your detailed rubric (exact wording from your message) *plus* the full text of the three reference docs (kept hidden from the final answer).
2. **User turn** – Supplies the *specific user prompt* and the *candidate answer* to be judged.
3. **LLM output** – Should respect the rubric, generate a scratch-pad (hidden) and output the mandated table, including the *Alpha evaluation* out of 100.

Because the chain uses a single call with the ground-truth docs in-context, no vector DB is required—the simplest "tool-free" approach.

If doc sizes exceed the model’s context limit, consider:

- **Truncating** repetitive sections
- **Splitting** large tables
- Upgrading to a *128k* context model (e.g. `gpt-4o-128k`)

## 🛠️  Extending / Integrating

- Wrap `evaluate_candidate()` in your own code or serve it via FastAPI.  
- Swap the call with [OpenAI function-calling](https://platform.openai.com/docs/guides/function-calling) for structured JSON output.
- Replace in-context docs with embeddings + retrieval to save tokens.

---

Made with ❤️  to simplify rigorous LLM evaluation.