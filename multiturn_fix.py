# Databricks notebook source
# MAGIC %md
# MAGIC ## Fixed Multi-Turn Conversation Support
# MAGIC 
# MAGIC This cell contains the corrected multi-turn conversation functions that properly
# MAGIC handle the Beth-Copilot API.

# COMMAND ----------

# ---------------------------------------------------------------------
# FIXED: Multi-turn conversation helper
# ---------------------------------------------------------------------
# Key fixes:
# 1. Reuse the same _call_endpoint function for consistency
# 2. Add proper delays between turns to avoid rate limiting
# 3. Better error handling and logging
# 4. Option to disable the example run

import uuid
import asyncio

async def run_multiturn_conversation_fixed(
    turns: list[str],
    *,
    zuid: str,
    login_memento: str,
    conversation_id: str | None = None,
    delay_between_turns: float = 1.0,  # Add delay to avoid rate limiting
    verbose: bool = True,
) -> tuple[pd.DataFrame, str]:
    """Run turns sequentially against the copilot endpoint.

    Multi-turn requirements:
    - conversation_id stays constant across turns
    - interaction_id increments each turn (1, 2, 3, ...)
    
    Fixes applied:
    - Added delay between turns to avoid rate limiting (405 errors)
    - Better error logging
    - Consistent session_id handling
    """
    conversation_id = conversation_id or f"mt_{uuid.uuid4().hex}"
    session_id = f"sid_{uuid.uuid4().hex}"  # Keep session_id constant across turns
    rows: list[dict[str, Any]] = []

    async with httpx.AsyncClient() as client:
        for interaction_id, user_text in enumerate(turns, start=1):
            if verbose:
                print(f"  Turn {interaction_id}/{len(turns)}: Sending request...")
            
            payload = {
                "query": user_text,
                "interaction_id": str(interaction_id),
                "conversation_id": conversation_id,
                "user_state": {
                    "zuid": zuid,
                    "session_id": session_id,  # Keep constant
                    "conversation_id": conversation_id,
                },
            }

            try:
                resp = await client.post(
                    API_URL,
                    json=payload,
                    headers={
                        "Accept": "application/json",
                        "Content-Type": "application/json",
                        "x-z-login-memento": f"x-z-login-memento={login_memento}",
                        "apikey": ZGS_BETH_COPILOT_KONG_KEY,
                    },
                    timeout=60.0,
                )
                resp.raise_for_status()
                raw = resp.json()
                assistant_text = raw.get("responses", [{}])[0].get("message", "")
                error = None
                
            except httpx.HTTPStatusError as e:
                # HTTP errors like 405, 500, etc.
                error = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
                assistant_text = ""
                if verbose:
                    print(f"    ❌ HTTP Error: {e.response.status_code}")
                    
            except httpx.RequestError as e:
                # Network errors
                error = f"Request error: {str(e)}"
                assistant_text = ""
                if verbose:
                    print(f"    ❌ Request Error: {str(e)}")
                    
            except Exception as e:
                error = f"{type(e).__name__}: {str(e)}"
                assistant_text = ""
                if verbose:
                    print(f"    ❌ Error: {str(e)}")

            rows.append({
                "conversation_id": conversation_id,
                "turn": interaction_id,
                "user": user_text,
                "assistant": assistant_text,
                "error": error or "",
            })
            
            if verbose and not error:
                print(f"    ✅ Response received ({len(assistant_text)} chars)")

            # Add delay between turns to avoid rate limiting
            if interaction_id < len(turns):
                await asyncio.sleep(delay_between_turns)

    return pd.DataFrame(rows), conversation_id


# ---------------------------------------------------------------------
# Alternative: Use requests library (sync) instead of httpx (async)
# This is more reliable in Databricks and avoids async issues
# ---------------------------------------------------------------------
import requests

def run_multiturn_sync(
    turns: list[str],
    *,
    zuid: str,
    login_memento: str,
    conversation_id: str | None = None,
    delay_between_turns: float = 1.5,
    verbose: bool = True,
) -> tuple[pd.DataFrame, str]:
    """Synchronous version of multi-turn conversation.
    
    This version uses the requests library instead of httpx,
    which is often more reliable in Databricks notebooks.
    """
    conversation_id = conversation_id or f"mt_{uuid.uuid4().hex}"
    session_id = f"sid_{uuid.uuid4().hex}"
    rows: list[dict[str, Any]] = []

    for interaction_id, user_text in enumerate(turns, start=1):
        if verbose:
            print(f"  Turn {interaction_id}/{len(turns)}: '{user_text[:50]}...'")
        
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
            resp = requests.post(
                API_URL,
                json=payload,
                headers=headers,
                timeout=60,
            )
            resp.raise_for_status()
            raw = resp.json()
            assistant_text = raw.get("responses", [{}])[0].get("message", "")
            error = None
            
            if verbose:
                print(f"    ✅ Response: {assistant_text[:100]}..." if len(assistant_text) > 100 else f"    ✅ Response: {assistant_text}")
                
        except requests.exceptions.HTTPError as e:
            error = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
            assistant_text = ""
            if verbose:
                print(f"    ❌ HTTP Error: {e.response.status_code} - {e.response.text[:100]}")
                
        except Exception as e:
            error = f"{type(e).__name__}: {str(e)}"
            assistant_text = ""
            if verbose:
                print(f"    ❌ Error: {str(e)}")

        rows.append({
            "conversation_id": conversation_id,
            "turn": interaction_id,
            "user": user_text,
            "assistant": assistant_text,
            "error": error or "",
        })

        # Delay between turns
        if interaction_id < len(turns):
            time.sleep(delay_between_turns)

    return pd.DataFrame(rows), conversation_id


# ---------------------------------------------------------------------
# Usage example - SET TO False TO SKIP
# ---------------------------------------------------------------------
RUN_MULTI_TURN_EXAMPLE = False  # <-- Set to True to run the example

if RUN_MULTI_TURN_EXAMPLE:
    print("\n🔄 Running multi-turn example (sync version)...")
    
    example_turns = [
        "What factors were considered to calculate BuyAbility?",
        "What is the most important factor for me?",
    ]
    
    try:
        mt_df, mt_conversation_id = run_multiturn_sync(
            example_turns,
            zuid=USER_ID,
            login_memento=LOGIN_MEMENTO,
            delay_between_turns=2.0,  # Longer delay to be safe
        )
        
        print(f"\n✅ Conversation ID: {mt_conversation_id}")
        display(mt_df)
        
        # Save results
        mt_out = f"{current_path}/results/multiturn_smoke_test_{RUN_NAME}.csv"
        mt_df.to_csv(mt_out, index=False)
        print(f"✅ Multi-turn transcript saved to {mt_out}")
        
    except Exception as e:
        print(f"❌ Multi-turn example failed: {e}")
        print("   This may be due to API configuration. Single-turn evaluation still works.")
else:
    print("ℹ️  Multi-turn example skipped (RUN_MULTI_TURN_EXAMPLE = False)")
