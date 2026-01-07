# Databricks notebook source
# COMMAND ----------

# ------------------------------------------------------------
# FIXED Multi-turn smoke test v2
# Uses EXACT same payload format as working single-turn calls
# ------------------------------------------------------------
import uuid
import requests

def run_multiturn_sample_v2(
    turns: list[str],
    *,
    zuid: str,
    login_memento: str,
    delay_seconds: float = 2.0,
) -> pd.DataFrame:
    """Multi-turn using EXACT same format as working single-turn."""
    
    # Use simple IDs like the working single-turn code
    conversation_id = str(uuid.uuid4().hex[:8])  # Shorter ID
    
    rows = []

    for turn_idx, user_text in enumerate(turns, start=1):
        print(f"  🔄 Turn {turn_idx}: {user_text[:50]}...")
        
        # Match EXACTLY the working payload format from call_endpoint
        payload = {
            "query": user_text,
            "interaction_id": str(turn_idx),
            "conversation_id": conversation_id,
            "user_state": {
                "zuid": zuid,
                "session_id": "sid",  # Hardcoded like working version
                "conversation_id": "cid",  # Hardcoded like working version
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
            print(f"     Status code: {resp.status_code}")
            resp.raise_for_status()
            raw = resp.json()
            assistant_msg = raw.get("responses", [{}])[0].get("message", "")
            error = ""
            print(f"     ✅ Response received")
        except requests.exceptions.HTTPError as e:
            error = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
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

        if turn_idx < len(turns):
            time.sleep(delay_seconds)

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# DEBUG: Test if the API_URL and credentials are correct
# ------------------------------------------------------------
def debug_single_request():
    """Test a single request to verify API is accessible."""
    print("🔍 DEBUG: Testing single request...")
    print(f"   API_URL: {API_URL}")
    print(f"   Has KONG_KEY: {bool(ZGS_BETH_COPILOT_KONG_KEY)}")
    print(f"   Has LOGIN_MEMENTO: {bool(LOGIN_MEMENTO)}")
    
    payload = {
        "query": "Hello, what is BuyAbility?",
        "interaction_id": "1",
        "conversation_id": "1",
        "user_state": {
            "zuid": USER_ID,
            "session_id": "sid",
            "conversation_id": "cid",
        },
    }
    
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "x-z-login-memento": f"x-z-login-memento={LOGIN_MEMENTO}",
        "apikey": ZGS_BETH_COPILOT_KONG_KEY,
    }
    
    print(f"   Headers: {list(headers.keys())}")
    
    try:
        resp = requests.post(API_URL, json=payload, headers=headers, timeout=60)
        print(f"   Status: {resp.status_code}")
        print(f"   Response headers: {dict(resp.headers)}")
        if resp.status_code == 405:
            print(f"   ❌ 405 Error - Method Not Allowed")
            print(f"   Response body: {resp.text[:500]}")
            # Try GET instead
            print("\n   🔍 Trying GET request instead...")
            resp_get = requests.get(API_URL, headers=headers, timeout=60)
            print(f"   GET Status: {resp_get.status_code}")
        else:
            print(f"   ✅ Request succeeded")
            return resp.json()
    except Exception as e:
        print(f"   ❌ Exception: {e}")
    
    return None

# Run debug first
debug_single_request()

# COMMAND ----------

# Then try multi-turn with fixed format
turns = [
    "What factors were considered to calculate BuyAbility?",
    "What is the most important factor for me?",
]

print("\n🔄 Starting multi-turn conversation (v2 - matching single-turn format)...")
df_multiturn = run_multiturn_sample_v2(turns, zuid=USER_ID, login_memento=LOGIN_MEMENTO)
display(df_multiturn)
