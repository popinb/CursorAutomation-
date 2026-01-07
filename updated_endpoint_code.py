# Databricks notebook source
# MAGIC %md
# MAGIC ## Updated Code for New Endpoint
# MAGIC 
# MAGIC New endpoint: `https://zgs-beth-copilot-stage.int.zgcp-itrc-nonprod-k8s.zg-int.net/voyager-eval`

# COMMAND ----------

# ============================================================
# STEP 1: Update the API_URL constant
# Replace your existing API_URL with this:
# ============================================================

API_URL = "https://zgs-beth-copilot-stage.int.zgcp-itrc-nonprod-k8s.zg-int.net/voyager-eval"

# ============================================================
# STEP 2: Updated helper functions for new endpoint
# Replace your existing _call_endpoint and call_endpoint
# ============================================================

async def _call_endpoint_new(
    client: httpx.AsyncClient,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Send a single POST request to the NEW endpoint."""
    try:
        resp = await client.post(
            API_URL,
            json=payload,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        return {"error": str(exc), "error_type": type(exc).__name__, "payload": payload}


async def call_endpoint_new(
    queries: list[str],
    *,
    zuid: str,
    conversation_id: str | None = None,
) -> list[dict[str, Any]]:
    """Call the NEW Beth-Copilot endpoint concurrently for all queries."""
    import uuid
    
    # Each query gets its own conversation_id if not provided
    payloads = []
    for query in queries:
        payload = {
            "prompt": query,  # Changed from "query" to "prompt"
            "zuid": zuid,     # At root level now
            "conversation_id": conversation_id or str(uuid.uuid4()),
        }
        payloads.append(payload)

    async with httpx.AsyncClient() as client:
        tasks = [_call_endpoint_new(client, p) for p in payloads]
        return await asyncio.gather(*tasks)


# ============================================================
# STEP 3: Test the new endpoint
# ============================================================

print("🔍 Testing new endpoint...")
print(f"URL: {API_URL}")

# Simple test
import uuid
test_conversation_id = str(uuid.uuid4())

test_payload = {
    "prompt": "What is BuyAbility?",
    "zuid": USER_ID,
    "conversation_id": test_conversation_id,
}

print(f"\nPayload: {test_payload}")

try:
    import requests
    resp = requests.post(
        API_URL,
        json=test_payload,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        timeout=60,
    )
    print(f"\nStatus: {resp.status_code}")
    print(f"Response: {resp.text[:500]}...")
except Exception as e:
    print(f"Error: {e}")

# COMMAND ----------

# ============================================================
# STEP 4: Updated Multi-turn function for new endpoint
# ============================================================

import uuid
import asyncio

async def run_multiturn_new_endpoint(
    turns: list[str],
    *,
    zuid: str,
    conversation_id: str | None = None,
    delay_seconds: float = 1.0,
) -> pd.DataFrame:
    """Multi-turn conversation using the NEW endpoint format."""
    
    # Same conversation_id for all turns (multi-turn requirement)
    conversation_id = conversation_id or str(uuid.uuid4())
    rows = []

    async with httpx.AsyncClient() as client:
        for turn_idx, user_text in enumerate(turns, start=1):
            print(f"  🔄 Turn {turn_idx}: {user_text[:50]}...")
            
            # NEW payload format
            payload = {
                "prompt": user_text,
                "zuid": zuid,
                "conversation_id": conversation_id,
            }
            
            raw = await _call_endpoint_new(client, payload)
            
            # Extract response - adjust based on actual response structure
            if "error" in raw:
                print(f"     ❌ Error: {raw['error']}")
                assistant_msg = ""
                error = raw["error"]
            else:
                # The response structure might be different - check and adjust
                # Try common patterns:
                assistant_msg = (
                    raw.get("response", "") or 
                    raw.get("message", "") or 
                    raw.get("responses", [{}])[0].get("message", "") or
                    raw.get("summary", "") or
                    str(raw)[:500]  # Fallback: show raw response
                )
                error = ""
                print(f"     ✅ Response received ({len(str(assistant_msg))} chars)")

            rows.append({
                "conversation_id": conversation_id,
                "turn": turn_idx,
                "user": user_text,
                "assistant": assistant_msg,
                "error": error,
                "raw_response": raw,  # Keep raw for debugging
            })

            # Delay between turns
            if turn_idx < len(turns):
                await asyncio.sleep(delay_seconds)

    return pd.DataFrame(rows)


# Test multi-turn
print("\n🔄 Testing multi-turn with new endpoint...")
turns = [
    "What factors were considered to calculate BuyAbility?",
    "What is the most important factor for me?",
]

df_multiturn = await run_multiturn_new_endpoint(
    turns,
    zuid=USER_ID,
)
display(df_multiturn)

# COMMAND ----------

# ============================================================
# STEP 5: Updated run_evaluations for new endpoint
# ============================================================

# If you need to fetch new responses for evaluation, update the fetch code:

async def fetch_responses_new_endpoint(
    df: pd.DataFrame,
    prompt_col: str,
    response_col: str,
    zuid: str,
) -> pd.DataFrame:
    """Fetch responses using the new endpoint."""
    import uuid
    
    prompts = df[prompt_col].tolist()
    print(f"Fetching {len(prompts)} responses from new endpoint...")
    
    responses = []
    async with httpx.AsyncClient() as client:
        for i, prompt in enumerate(prompts):
            print(f"  {i+1}/{len(prompts)}: {prompt[:50]}...")
            
            payload = {
                "prompt": prompt,
                "zuid": zuid,
                "conversation_id": str(uuid.uuid4()),
            }
            
            raw = await _call_endpoint_new(client, payload)
            
            # Extract the response text - adjust based on actual structure
            if "error" in raw:
                response_text = f"ERROR: {raw['error']}"
            else:
                response_text = (
                    raw.get("response", "") or 
                    raw.get("message", "") or 
                    raw.get("responses", [{}])[0].get("message", "") or
                    ""
                )
            
            responses.append(response_text)
            
            # Small delay to avoid overwhelming the API
            await asyncio.sleep(0.5)
    
    df[response_col] = responses
    return df


# Usage:
# df = await fetch_responses_new_endpoint(df, PROMPT_COL_NAME, RESPONSE_COL_NAME, USER_ID)
