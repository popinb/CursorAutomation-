# Databricks notebook source
# MAGIC %md
# MAGIC # Voyager-Eval Endpoint Testing Notebook
# MAGIC 
# MAGIC This notebook tests the new voyager-eval streaming endpoint.
# MAGIC 
# MAGIC **Endpoint:** `https://zgs-beth-copilot-stage.int.zgcp-itrc-nonprod-k8s.zg-int.net/voyager-eval`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 1: Install Dependencies

# COMMAND ----------

%pip install httpx requests pandas --quiet

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2: Imports and Configuration

# COMMAND ----------

import requests
import httpx
import json
import uuid
import time
import asyncio
import pandas as pd
from typing import Any, Dict, List, Optional

# Configuration
API_URL = "https://zgs-beth-copilot-stage.int.zgcp-itrc-nonprod-k8s.zg-int.net/voyager-eval"

# Get user ID from secrets (or use test value)
try:
    USER_ID = dbutils.secrets.get(scope="hungc-secure-scope", key="stage_zuid")
except:
    USER_ID = "111"  # Default test value

print("✅ Configuration loaded")
print(f"   API_URL: {API_URL}")
print(f"   USER_ID: {USER_ID}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 3: Test Basic Connectivity

# COMMAND ----------

# Quick connectivity test
print("🔍 Testing endpoint connectivity...\n")

test_payload = {
    "prompt": "hi",
    "zuid": USER_ID,
    "conversation_id": str(uuid.uuid4()),
}

print(f"Payload: {json.dumps(test_payload, indent=2)}\n")

try:
    start = time.time()
    
    with requests.post(
        API_URL,
        json=test_payload,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        stream=True,
        timeout=120,
    ) as resp:
        print(f"✅ Status: {resp.status_code}")
        print(f"   Content-Type: {resp.headers.get('content-type', 'N/A')}")
        
        # Collect response
        chunks = []
        for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                chunks.append(chunk)
        
        full_response = "".join(chunks)
        elapsed = time.time() - start
        
        print(f"   Response time: {elapsed:.1f}s")
        print(f"   Response size: {len(full_response)} chars")
        print(f"\n📥 Response preview:\n{full_response[:1000]}...")
        
        # Try to parse
        try:
            response_data = json.loads(full_response)
            print(f"\n📦 Response keys: {list(response_data.keys())}")
        except:
            print("\n⚠️ Response is not JSON or is streaming format")
            
except requests.exceptions.Timeout:
    print(f"❌ Timeout after {time.time() - start:.1f}s")
except Exception as e:
    print(f"❌ Error: {type(e).__name__}: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 4: Streaming Response Helper Functions

# COMMAND ----------

def call_voyager_eval(
    prompt: str,
    *,
    zuid: str,
    conversation_id: Optional[str] = None,
    timeout: int = 120,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Call the voyager-eval endpoint with streaming support.
    
    Args:
        prompt: The user's query
        zuid: User ID
        conversation_id: Optional conversation ID (generated if not provided)
        timeout: Request timeout in seconds
        verbose: Print streaming output
    
    Returns:
        Parsed response dict or error dict
    """
    conversation_id = conversation_id or str(uuid.uuid4())
    
    payload = {
        "prompt": prompt,
        "zuid": zuid,
        "conversation_id": conversation_id,
    }
    
    try:
        with requests.post(
            API_URL,
            json=payload,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            stream=True,
            timeout=timeout,
        ) as resp:
            resp.raise_for_status()
            
            # Collect streaming response
            chunks = []
            for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
                if chunk:
                    chunks.append(chunk)
                    if verbose:
                        print(chunk, end="", flush=True)
            
            full_response = "".join(chunks)
            
            if verbose:
                print()  # Newline after streaming
            
            # Parse response
            try:
                return json.loads(full_response)
            except json.JSONDecodeError:
                # Return raw if not JSON
                return {"raw_response": full_response, "conversation_id": conversation_id}
                
    except requests.exceptions.Timeout:
        return {"error": f"Timeout after {timeout}s", "conversation_id": conversation_id}
    except requests.exceptions.HTTPError as e:
        return {"error": f"HTTP {e.response.status_code}: {e.response.text[:200]}", "conversation_id": conversation_id}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)}", "conversation_id": conversation_id}


async def call_voyager_eval_async(
    prompt: str,
    *,
    zuid: str,
    conversation_id: Optional[str] = None,
    timeout: int = 120,
) -> Dict[str, Any]:
    """Async version of call_voyager_eval."""
    conversation_id = conversation_id or str(uuid.uuid4())
    
    payload = {
        "prompt": prompt,
        "zuid": zuid,
        "conversation_id": conversation_id,
    }
    
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                API_URL,
                json=payload,
                headers={"Accept": "application/json", "Content-Type": "application/json"},
            ) as resp:
                resp.raise_for_status()
                
                chunks = []
                async for chunk in resp.aiter_text():
                    chunks.append(chunk)
                
                full_response = "".join(chunks)
                
                try:
                    return json.loads(full_response)
                except:
                    return {"raw_response": full_response, "conversation_id": conversation_id}
                    
    except Exception as e:
        return {"error": str(e), "conversation_id": conversation_id}


print("✅ Helper functions defined:")
print("   - call_voyager_eval(prompt, zuid, ...) - Synchronous")
print("   - call_voyager_eval_async(prompt, zuid, ...) - Async")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 5: Test Single Request

# COMMAND ----------

# Test single request
print("🔍 Testing single request...\n")

result = call_voyager_eval(
    "What is BuyAbility?",
    zuid=USER_ID,
    verbose=True,  # Show streaming output
)

print("\n" + "=" * 60)
print(f"📦 Result type: {type(result)}")
if isinstance(result, dict):
    print(f"📦 Keys: {list(result.keys())}")
    if "error" in result:
        print(f"❌ Error: {result['error']}")
    else:
        # Show some of the response
        for key, value in result.items():
            value_str = str(value)[:200]
            print(f"   {key}: {value_str}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 6: Multi-Turn Conversation

# COMMAND ----------

def run_multiturn_conversation(
    turns: List[str],
    *,
    zuid: str,
    conversation_id: Optional[str] = None,
    delay_between_turns: float = 1.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run a multi-turn conversation.
    
    All turns share the same conversation_id for context continuity.
    """
    conversation_id = conversation_id or str(uuid.uuid4())
    
    if verbose:
        print(f"🔄 Starting multi-turn conversation")
        print(f"   Conversation ID: {conversation_id}")
        print(f"   Turns: {len(turns)}")
        print("=" * 60)
    
    rows = []
    
    for turn_idx, user_text in enumerate(turns, start=1):
        if verbose:
            print(f"\n📤 Turn {turn_idx}: {user_text[:60]}...")
        
        start = time.time()
        result = call_voyager_eval(
            user_text,
            zuid=zuid,
            conversation_id=conversation_id,  # Same ID for all turns
            verbose=False,
        )
        elapsed = time.time() - start
        
        # Extract response (adjust based on actual response structure)
        if "error" in result:
            response_text = ""
            error = result["error"]
            if verbose:
                print(f"   ❌ Error: {error}")
        else:
            # Try to extract the response - adjust these based on actual structure
            response_text = (
                result.get("final_response") or
                result.get("response") or
                result.get("message") or
                result.get("raw_response", "")[:500] or
                ""
            )
            error = ""
            if verbose:
                print(f"   ✅ Response ({len(str(response_text))} chars, {elapsed:.1f}s)")
                if response_text:
                    print(f"   📥 {str(response_text)[:150]}...")
        
        rows.append({
            "conversation_id": conversation_id,
            "turn": turn_idx,
            "user": user_text,
            "assistant": response_text,
            "error": error,
            "response_time_s": elapsed,
            "raw_response": result,
        })
        
        # Delay between turns
        if turn_idx < len(turns):
            time.sleep(delay_between_turns)
    
    if verbose:
        print("\n" + "=" * 60)
        print(f"✅ Conversation complete!")
    
    return pd.DataFrame(rows)


print("✅ run_multiturn_conversation() defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 7: Test Multi-Turn Conversation

# COMMAND ----------

# Test multi-turn
turns = [
    "What factors were considered to calculate BuyAbility?",
    "What is the most important factor for me?",
]

print("🔄 Running multi-turn test...\n")

df_multiturn = run_multiturn_conversation(
    turns,
    zuid=USER_ID,
    delay_between_turns=2.0,  # 2 second delay
)

# Display results
display(df_multiturn[["turn", "user", "assistant", "error", "response_time_s"]])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 8: Batch Testing Multiple Prompts

# COMMAND ----------

async def batch_evaluate(
    prompts: List[str],
    *,
    zuid: str,
    max_concurrent: int = 3,
) -> pd.DataFrame:
    """
    Evaluate multiple prompts (each as separate conversation).
    """
    print(f"🔄 Evaluating {len(prompts)} prompts...")
    
    rows = []
    
    for i, prompt in enumerate(prompts):
        print(f"  {i+1}/{len(prompts)}: {prompt[:50]}...")
        
        result = await call_voyager_eval_async(
            prompt,
            zuid=zuid,
        )
        
        response_text = (
            result.get("final_response") or
            result.get("response") or
            result.get("raw_response", "")[:500] or
            ""
        )
        
        rows.append({
            "prompt": prompt,
            "response": response_text,
            "error": result.get("error", ""),
            "conversation_id": result.get("conversation_id", ""),
        })
        
        # Small delay to avoid overwhelming the API
        await asyncio.sleep(1)
    
    print(f"✅ Done!")
    return pd.DataFrame(rows)


# Test batch evaluation
test_prompts = [
    "What is BuyAbility?",
    "How do I calculate my down payment?",
    "What credit score do I need to buy a home?",
]

df_batch = await batch_evaluate(test_prompts, zuid=USER_ID)
display(df_batch)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 9: Inspect Response Structure

# COMMAND ----------

# Get a response and inspect its full structure
print("🔍 Inspecting full response structure...\n")

result = call_voyager_eval(
    "What is BuyAbility?",
    zuid=USER_ID,
)

def print_structure(obj, indent=0):
    """Pretty print nested dict/list structure."""
    prefix = "  " * indent
    if isinstance(obj, dict):
        print(f"{prefix}{{")
        for key, value in obj.items():
            if isinstance(value, (dict, list)):
                print(f"{prefix}  '{key}':")
                print_structure(value, indent + 2)
            else:
                value_preview = str(value)[:80]
                print(f"{prefix}  '{key}': {value_preview}...")
        print(f"{prefix}}}")
    elif isinstance(obj, list):
        print(f"{prefix}[")
        for i, item in enumerate(obj[:3]):  # Show first 3 items
            print_structure(item, indent + 1)
        if len(obj) > 3:
            print(f"{prefix}  ... ({len(obj)} items total)")
        print(f"{prefix}]")
    else:
        print(f"{prefix}{str(obj)[:100]}")

print_structure(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 10: Save Results

# COMMAND ----------

# Save multi-turn results
output_path = "/tmp/multiturn_test_results.csv"
df_multiturn[["conversation_id", "turn", "user", "assistant", "error"]].to_csv(output_path, index=False)
print(f"✅ Saved to {output_path}")

# Display final summary
print("\n📊 Summary:")
print(f"   Total turns: {len(df_multiturn)}")
print(f"   Errors: {(df_multiturn['error'] != '').sum()}")
print(f"   Avg response time: {df_multiturn['response_time_s'].mean():.1f}s")
