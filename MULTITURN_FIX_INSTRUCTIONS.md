# Multi-Turn Conversation Fix for Databricks Notebook

## Problem Analysis

The **405 Method Not Allowed** error in your multi-turn example is likely caused by one of these issues:

1. **Async context issues in Databricks** - The `httpx.AsyncClient` created inside `run_multiturn_conversation` may have different behavior than expected
2. **Rate limiting** - The endpoint might be rejecting rapid sequential requests
3. **The original example in the Helpers cell runs automatically** when `RUN_MULTI_TURN_EXAMPLE = True`

## Solution

### Option 1: Use Synchronous Requests (Recommended)

Replace the multi-turn code with a synchronous version using `requests` library instead of `httpx`:

```python
# Add this AFTER your Helpers cell (replace the existing multi-turn code)

# ---------------------------------------------------------------------
# Fixed Multi-turn testing helper (Synchronous version)
# ---------------------------------------------------------------------
import uuid
import requests

def run_multiturn_sync(
    turns: list[str],
    *,
    zuid: str,
    login_memento: str,
    conversation_id: str | None = None,
    delay_between_turns: float = 2.0,  # Increase delay to avoid 405
    verbose: bool = True,
) -> tuple[pd.DataFrame, str]:
    """Run turns sequentially (synchronous version).
    
    More reliable in Databricks than async httpx.
    """
    conversation_id = conversation_id or f"mt_{uuid.uuid4().hex}"
    session_id = f"sid_{uuid.uuid4().hex}"
    rows = []

    for interaction_id, user_text in enumerate(turns, start=1):
        if verbose:
            print(f"  Turn {interaction_id}/{len(turns)}: Sending...")
        
        payload = {
            "query": user_text,
            "interaction_id": str(interaction_id),
            "conversation_id": conversation_id,
            "user_state": {
                "zuid": zuid,
                "session_id": session_id,
                "conversation_id": conversation_id,
            },
        }
        
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "x-z-login-memento": f"x-z-login-memento={login_memento}",
            "apikey": ZGS_BETH_COPILOT_KONG_KEY,
        }

        try:
            resp = requests.post(API_URL, json=payload, headers=headers, timeout=60)
            resp.raise_for_status()
            raw = resp.json()
            assistant_text = raw.get("responses", [{}])[0].get("message", "")
            error = ""
            if verbose:
                print(f"    ✅ Received ({len(assistant_text)} chars)")
                
        except requests.exceptions.HTTPError as e:
            error = f"HTTP {e.response.status_code}: {e.response.text[:100]}"
            assistant_text = ""
            if verbose:
                print(f"    ❌ {error}")
                
        except Exception as e:
            error = str(e)
            assistant_text = ""
            if verbose:
                print(f"    ❌ {error}")

        rows.append({
            "conversation_id": conversation_id,
            "turn": interaction_id,
            "user": user_text,
            "assistant": assistant_text,
            "error": error,
        })

        # IMPORTANT: Add delay between turns
        if interaction_id < len(turns):
            time.sleep(delay_between_turns)

    return pd.DataFrame(rows), conversation_id
```

### Option 2: Disable the Failing Example

In the **Helpers cell**, change:

```python
RUN_MULTI_TURN_EXAMPLE = True
```

to:

```python
RUN_MULTI_TURN_EXAMPLE = False  # Skip the example that causes 405
```

This will prevent the example with "I want to buy a home" from running.

### Option 3: Fix the Async Version

If you want to keep using async, add delays and better error handling:

```python
async def run_multiturn_sample(
    turns: list[str],
    *,
    zuid: str,
    login_memento: str,
) -> pd.DataFrame:
    conversation_id = f"mt_{RUN_NAME}_{uuid.uuid4().hex}"
    session_id = f"sid_{uuid.uuid4().hex}"

    rows = []
    async with httpx.AsyncClient() as client:
        for turn_idx, user_text in enumerate(turns, start=1):
            payload = {
                "query": user_text,
                "interaction_id": str(turn_idx),
                "conversation_id": conversation_id,
                "user_state": {
                    "zuid": zuid,
                    "session_id": session_id,
                    "conversation_id": conversation_id,
                },
            }

            # Use the same _call_endpoint function for consistency
            raw = await _call_endpoint(client, payload, login_memento)
            assistant_msg = raw.get("responses", [{}])[0].get("message", "")

            rows.append({
                "conversation_id": conversation_id,
                "turn": turn_idx,
                "user": user_text,
                "assistant": assistant_msg,
                "error": raw.get("error", ""),
            })
            
            # ADD THIS: Delay between turns to avoid rate limiting
            if turn_idx < len(turns):
                await asyncio.sleep(2.0)  # 2 second delay

    return pd.DataFrame(rows)
```

## Complete Fixed Cell for Your Notebook

Replace your multi-turn smoke test cell with this:

```python
# COMMAND ----------

# ------------------------------------------------------------
# Sample multi-turn smoke test (2 turns, same conversation_id)
# Uses SYNCHRONOUS requests for better reliability
# ------------------------------------------------------------
import uuid
import requests

def run_multiturn_sample(
    turns: list[str],
    *,
    zuid: str,
    login_memento: str,
    delay_seconds: float = 2.0,
) -> pd.DataFrame:
    """Synchronous multi-turn conversation helper."""
    conversation_id = f"mt_{RUN_NAME}_{uuid.uuid4().hex}"
    session_id = f"sid_{uuid.uuid4().hex}"
    rows = []

    for turn_idx, user_text in enumerate(turns, start=1):
        print(f"  🔄 Turn {turn_idx}: {user_text[:50]}...")
        
        payload = {
            "query": user_text,
            "interaction_id": str(turn_idx),
            "conversation_id": conversation_id,
            "user_state": {
                "zuid": zuid,
                "session_id": session_id,
                "conversation_id": conversation_id,
            },
        }
        
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "x-z-login-memento": f"x-z-login-memento={login_memento}",
            "apikey": ZGS_BETH_COPILOT_KONG_KEY,
        }

        try:
            resp = requests.post(API_URL, json=payload, headers=headers, timeout=60)
            resp.raise_for_status()
            raw = resp.json()
            assistant_msg = raw.get("responses", [{}])[0].get("message", "")
            error = ""
            print(f"     ✅ Response received")
        except requests.exceptions.HTTPError as e:
            error = f"HTTP {e.response.status_code}"
            assistant_msg = ""
            print(f"     ❌ Error: {error}")
        except Exception as e:
            error = str(e)
            assistant_msg = ""
            print(f"     ❌ Error: {error}")

        rows.append({
            "conversation_id": conversation_id,
            "turn": turn_idx,
            "user": user_text,
            "assistant": assistant_msg,
            "error": error,
        })

        # Delay between turns
        if turn_idx < len(turns):
            time.sleep(delay_seconds)

    return pd.DataFrame(rows)


# Your requested sample multi-turn run
turns = [
    "What factors were considered to calculate BuyAbility?",
    "What is the most important factor for me?",
]

print("🔄 Starting multi-turn conversation...")
df_multiturn = run_multiturn_sample(turns, zuid=USER_ID, login_memento=LOGIN_MEMENTO)
display(df_multiturn)

# Optional: persist the transcript
mt_out = f"{current_path}/results/multiturn_smoke_test_{RUN_NAME}.csv"
df_multiturn.to_csv(mt_out, index=False)
print(f"✅ Multi-turn transcript saved to {mt_out}")
```

## Key Changes Made

1. **Switched from async `httpx` to sync `requests`** - More reliable in Databricks
2. **Added 2-second delays between turns** - Prevents rate limiting (405 errors)
3. **Consistent session_id** - Using the same session_id across all turns
4. **Better error handling** - Catches and logs specific HTTP errors

## Also Fix the Original Helper Cell

In your **Helpers** cell, set:

```python
RUN_MULTI_TURN_EXAMPLE = False
```

This disables the original multi-turn example that was causing the error.
