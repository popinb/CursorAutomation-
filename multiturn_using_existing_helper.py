# Databricks notebook source
# COMMAND ----------

# ------------------------------------------------------------
# Multi-turn using the EXISTING _call_endpoint function
# This MUST work since single-turn uses the same function!
# ------------------------------------------------------------
import uuid
import asyncio

async def run_multiturn_working(
    turns: list[str],
    *,
    zuid: str,
    login_memento: str,
    delay_seconds: float = 1.0,
) -> pd.DataFrame:
    """Multi-turn using the SAME _call_endpoint that works for single-turn."""
    
    conversation_id = f"mt_{uuid.uuid4().hex[:8]}"
    rows = []

    async with httpx.AsyncClient() as client:
        for turn_idx, user_text in enumerate(turns, start=1):
            print(f"  🔄 Turn {turn_idx}: {user_text[:50]}...")
            
            # Use EXACT same payload format as working single-turn
            payload = {
                "query": user_text,
                "interaction_id": str(turn_idx),
                "conversation_id": conversation_id,
                "user_state": {
                    "zuid": zuid,
                    "session_id": "sid",  # Same as single-turn
                    "conversation_id": "cid",  # Same as single-turn  
                },
            }
            
            # Use the EXISTING _call_endpoint function
            raw = await _call_endpoint(client, payload, login_memento)
            
            if "error" in raw:
                print(f"     ❌ Error: {raw['error']}")
                assistant_msg = ""
                error = raw["error"]
            else:
                assistant_msg = raw.get("responses", [{}])[0].get("message", "")
                error = ""
                print(f"     ✅ Response received ({len(assistant_msg)} chars)")

            rows.append({
                "conversation_id": conversation_id,
                "turn": turn_idx,
                "user": user_text,
                "assistant": assistant_msg,
                "error": error,
            })

            # Delay between turns
            if turn_idx < len(turns):
                await asyncio.sleep(delay_seconds)

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# Run the test
# ------------------------------------------------------------
turns = [
    "What factors were considered to calculate BuyAbility?",
    "What is the most important factor for me?",
]

print("🔄 Starting multi-turn conversation (using existing _call_endpoint)...")
df_multiturn = await run_multiturn_working(
    turns, 
    zuid=USER_ID, 
    login_memento=LOGIN_MEMENTO
)
display(df_multiturn)

# Save results
mt_out = f"{current_path}/results/multiturn_smoke_test_{RUN_NAME}.csv"
df_multiturn.to_csv(mt_out, index=False)
print(f"✅ Saved to {mt_out}")
