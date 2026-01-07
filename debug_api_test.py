# Databricks notebook source
# COMMAND ----------

# ------------------------------------------------------------
# DEBUG: Verify API access works with existing infrastructure
# Run this AFTER the Helpers cell to ensure all variables are set
# ------------------------------------------------------------

print("=" * 60)
print("🔍 DEBUG: Checking API access")
print("=" * 60)

# 1. Verify variables are set
print("\n1️⃣ Checking variables...")
try:
    print(f"   API_URL: {API_URL}")
    print(f"   USER_ID: {USER_ID[:10]}..." if USER_ID else "   USER_ID: NOT SET")
    print(f"   LOGIN_MEMENTO: {'SET' if LOGIN_MEMENTO else 'NOT SET'}")
    print(f"   ZGS_BETH_COPILOT_KONG_KEY: {'SET' if ZGS_BETH_COPILOT_KONG_KEY else 'NOT SET'}")
except NameError as e:
    print(f"   ❌ Variable not defined: {e}")
    print("   Make sure you run this AFTER the Constants cell!")

# 2. Test using the EXISTING call_endpoint function (which works for single-turn)
print("\n2️⃣ Testing with existing call_endpoint function...")
try:
    test_queries = ["What is BuyAbility?"]
    test_results = await call_endpoint(test_queries, zuid=USER_ID, login_memento=LOGIN_MEMENTO)
    
    if "error" in test_results[0]:
        print(f"   ❌ Error: {test_results[0]['error']}")
    else:
        response_text = test_results[0].get("responses", [{}])[0].get("message", "")
        print(f"   ✅ call_endpoint works! Response: {response_text[:100]}...")
except Exception as e:
    print(f"   ❌ call_endpoint failed: {e}")

# 3. Test using the EXISTING _call_endpoint function directly
print("\n3️⃣ Testing with existing _call_endpoint function directly...")
try:
    import httpx
    
    payload = {
        "query": "What is BuyAbility?",
        "interaction_id": "1",
        "conversation_id": "1",
        "user_state": {"zuid": USER_ID, "session_id": "sid", "conversation_id": "cid"},
    }
    
    async with httpx.AsyncClient() as client:
        result = await _call_endpoint(client, payload, LOGIN_MEMENTO)
    
    if "error" in result:
        print(f"   ❌ Error: {result['error']}")
    else:
        response_text = result.get("responses", [{}])[0].get("message", "")
        print(f"   ✅ _call_endpoint works! Response: {response_text[:100]}...")
except Exception as e:
    print(f"   ❌ _call_endpoint failed: {e}")

# 4. Test using requests library (what my fix uses)
print("\n4️⃣ Testing with requests library...")
try:
    import requests
    
    payload = {
        "query": "What is BuyAbility?",
        "interaction_id": "1",
        "conversation_id": "1",
        "user_state": {"zuid": USER_ID, "session_id": "sid", "conversation_id": "cid"},
    }
    
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "x-z-login-memento": f"x-z-login-memento={LOGIN_MEMENTO}",
        "apikey": ZGS_BETH_COPILOT_KONG_KEY,
    }
    
    resp = requests.post(API_URL, json=payload, headers=headers, timeout=60)
    print(f"   Status code: {resp.status_code}")
    
    if resp.status_code == 405:
        print(f"   ❌ 405 Method Not Allowed")
        print(f"   Response: {resp.text[:300]}")
        print("\n   💡 This means the requests library hits a different endpoint behavior!")
        print("   Solution: Use httpx instead of requests")
    elif resp.status_code == 200:
        print(f"   ✅ requests library works!")
    else:
        print(f"   ⚠️ Unexpected status: {resp.status_code}")
        print(f"   Response: {resp.text[:300]}")
        
except Exception as e:
    print(f"   ❌ requests failed: {e}")

print("\n" + "=" * 60)
print("🔍 DEBUG COMPLETE")
print("=" * 60)
