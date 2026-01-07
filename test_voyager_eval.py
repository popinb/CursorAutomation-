# Databricks notebook source
# MAGIC %md
# MAGIC ## Test Voyager-Eval Endpoint

# COMMAND ----------

# ============================================================
# TEST 1: Basic streaming request with requests library
# ============================================================

import requests
import json
import uuid
import time

API_URL = "https://zgs-beth-copilot-stage.int.zgcp-itrc-nonprod-k8s.zg-int.net/voyager-eval"

payload = {
    "prompt": "hi",
    "zuid": "111",
    "conversation_id": str(uuid.uuid4()),
}

print(f"🔍 Testing: {API_URL}")
print(f"📤 Payload: {json.dumps(payload, indent=2)}")
print("=" * 60)

start_time = time.time()

try:
    # Use stream=True for streaming endpoints
    with requests.post(
        API_URL,
        json=payload,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        stream=True,
        timeout=180,  # 3 minute timeout
    ) as resp:
        
        print(f"📥 Status: {resp.status_code}")
        print(f"📥 Headers: {dict(resp.headers)}")
        print("=" * 60)
        
        if resp.status_code != 200:
            print(f"❌ Error: {resp.text}")
        else:
            # Collect streaming response
            chunks = []
            print("📥 Streaming response:")
            
            for chunk in resp.iter_content(chunk_size=1024, decode_unicode=True):
                if chunk:
                    chunks.append(chunk)
                    # Print chunks as they arrive
                    print(chunk, end="", flush=True)
            
            full_response = "".join(chunks)
            elapsed = time.time() - start_time
            
            print(f"\n\n{'=' * 60}")
            print(f"✅ Complete! Received {len(full_response)} chars in {elapsed:.1f}s")
            
            # Try to parse as JSON
            try:
                data = json.loads(full_response)
                print(f"📦 Response is valid JSON with keys: {list(data.keys())}")
            except json.JSONDecodeError:
                # Might be multiple JSON objects (JSONL/SSE)
                print("📦 Response is not single JSON - might be streaming format")
                lines = full_response.strip().split('\n')
                print(f"   Lines received: {len(lines)}")
                
except requests.exceptions.Timeout:
    print(f"❌ Timeout after {time.time() - start_time:.1f}s")
except Exception as e:
    print(f"❌ Error: {type(e).__name__}: {e}")

# COMMAND ----------

# ============================================================
# TEST 2: Using httpx with streaming (async)
# ============================================================

import httpx
import json
import uuid
import asyncio

API_URL = "https://zgs-beth-copilot-stage.int.zgcp-itrc-nonprod-k8s.zg-int.net/voyager-eval"

payload = {
    "prompt": "hi",
    "zuid": "111", 
    "conversation_id": str(uuid.uuid4()),
}

print(f"🔍 Testing with httpx async streaming...")
print(f"📤 Payload: {payload}")
print("=" * 60)

async def test_streaming():
    async with httpx.AsyncClient(timeout=180) as client:
        async with client.stream(
            "POST",
            API_URL,
            json=payload,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
        ) as resp:
            print(f"📥 Status: {resp.status_code}")
            
            chunks = []
            async for chunk in resp.aiter_text():
                chunks.append(chunk)
                print(chunk, end="", flush=True)
            
            return "".join(chunks)

result = await test_streaming()
print(f"\n\n✅ Total length: {len(result)} chars")

# COMMAND ----------

# ============================================================
# TEST 3: Server-Sent Events (SSE) format
# ============================================================

import requests
import json
import uuid

API_URL = "https://zgs-beth-copilot-stage.int.zgcp-itrc-nonprod-k8s.zg-int.net/voyager-eval"

payload = {
    "prompt": "hi",
    "zuid": "111",
    "conversation_id": str(uuid.uuid4()),
}

print(f"🔍 Testing SSE (Server-Sent Events) format...")
print("=" * 60)

try:
    with requests.post(
        API_URL,
        json=payload,
        headers={
            "Accept": "text/event-stream",  # SSE header
            "Content-Type": "application/json",
        },
        stream=True,
        timeout=180,
    ) as resp:
        
        print(f"📥 Status: {resp.status_code}")
        print(f"📥 Content-Type: {resp.headers.get('content-type')}")
        print("=" * 60)
        
        # Parse SSE format: "data: {...}\n\n"
        for line in resp.iter_lines(decode_unicode=True):
            if line:
                print(f"📥 {line}")
                
                # SSE format: "data: {json}"
                if line.startswith("data:"):
                    data_str = line[5:].strip()
                    if data_str and data_str != "[DONE]":
                        try:
                            event_data = json.loads(data_str)
                            print(f"   → Parsed: {list(event_data.keys())}")
                        except:
                            pass

except Exception as e:
    print(f"❌ Error: {e}")

# COMMAND ----------

# ============================================================
# TEST 4: Simple synchronous test with longer timeout
# ============================================================

import requests
import json
import uuid

API_URL = "https://zgs-beth-copilot-stage.int.zgcp-itrc-nonprod-k8s.zg-int.net/voyager-eval"

payload = {
    "prompt": "hi",
    "zuid": "111",
    "conversation_id": str(uuid.uuid4()),
}

print(f"🔍 Simple POST (no streaming, long timeout)...")

try:
    resp = requests.post(
        API_URL,
        json=payload,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        timeout=300,  # 5 minutes
    )
    
    print(f"📥 Status: {resp.status_code}")
    print(f"📥 Response: {resp.text[:2000]}")
    
except requests.exceptions.Timeout:
    print("❌ Timed out after 5 minutes")
except Exception as e:
    print(f"❌ Error: {e}")
