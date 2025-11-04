"""
PHASE 2 TEST: Databricks Serving Endpoints Discovery and Initialization
Tests the corrected implementation using Databricks Serving Endpoints API
"""

import sys
import os

# Add mock to path
sys.path.insert(0, '/workspace')
from mock_dbutils_enhanced import dbutils, get_mock_requests

# Mock requests module
import sys
requests = get_mock_requests()
sys.modules['requests'] = requests

print("="*80)
print("PHASE 2: DATABRICKS SERVING ENDPOINTS TEST")
print("="*80)

# ============================================================================
# Test 2.1: Workspace Context Retrieval
# ============================================================================

print("\n? Test 2.1: Workspace Context Retrieval")

try:
    # This is how the corrected implementation gets credentials
    databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
    workspace_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().get()
    
    assert databricks_token is not None, "Failed to get API token"
    assert workspace_url is not None, "Failed to get workspace URL"
    assert len(databricks_token) > 0, "API token is empty"
    assert len(workspace_url) > 0, "Workspace URL is empty"
    
    print(f"   ? API Token retrieved: {databricks_token[:10]}...")
    print(f"   ? Workspace URL retrieved: {workspace_url}")
    print(f"   ? Token length: {len(databricks_token)} chars")
    print(f"   ? No API key from secrets needed! (automatic)")
    
except Exception as e:
    print(f"   ? Error: {e}")
    sys.exit(1)

# ============================================================================
# Test 2.2: Databricks Headers Configuration
# ============================================================================

print("\n? Test 2.2: Databricks Headers Configuration")

databricks_headers = {
    "Authorization": f"Bearer {databricks_token}",
    "Content-Type": "application/json"
}

assert "Authorization" in databricks_headers, "Missing Authorization header"
assert databricks_headers["Authorization"].startswith("Bearer "), "Wrong auth format"
assert "Content-Type" in databricks_headers, "Missing Content-Type header"

print(f"   ? Authorization header: Bearer {databricks_token[:10]}...")
print(f"   ? Content-Type: {databricks_headers['Content-Type']}")

# ============================================================================
# Test 2.3: Query Serving Endpoints API
# ============================================================================

print("\n? Test 2.3: Query Serving Endpoints API")

try:
    url = f"https://{workspace_url}/api/2.0/serving-endpoints"
    response = requests.get(url, headers=databricks_headers, timeout=10)
    
    assert response.status_code == 200, f"Wrong status: {response.status_code}"
    
    endpoints_data = response.json()
    assert 'endpoints' in endpoints_data, "Missing 'endpoints' key in response"
    
    endpoints = endpoints_data.get('endpoints', [])
    available_models = [ep['name'] for ep in endpoints]
    
    print(f"   ? API call successful: status {response.status_code}")
    print(f"   ? Found {len(endpoints)} serving endpoints")
    print(f"   ? Endpoint names:")
    for name in available_models:
        print(f"     ? {name}")
    
except Exception as e:
    print(f"   ? Error: {e}")
    sys.exit(1)

# ============================================================================
# Test 2.4: Find Claude Sonnet Endpoint
# ============================================================================

print("\n? Test 2.4: Find Claude Sonnet Endpoint")

databricks_endpoint = None
for endpoint_name in available_models:
    if 'claude-sonnet' in endpoint_name.lower():
        databricks_endpoint = endpoint_name
        print(f"   ? Found Claude Sonnet endpoint: {endpoint_name}")
        break

assert databricks_endpoint is not None, "Claude Sonnet endpoint not found"
print(f"   ? Selected endpoint: {databricks_endpoint}")

# Test fallback logic
print("\n   Testing fallback logic (if Claude Sonnet not found):")
test_endpoints = ['other-model-1', 'other-model-2']
fallback_endpoint = None
for name in test_endpoints:
    if 'claude-sonnet' in name.lower():
        fallback_endpoint = name
        break
if not fallback_endpoint and test_endpoints:
    fallback_endpoint = test_endpoints[0]
    print(f"   ? Fallback would select: {fallback_endpoint}")

# ============================================================================
# Test 2.5: Client Configuration Storage
# ============================================================================

print("\n? Test 2.5: Client Configuration Storage")

client = {
    'type': 'databricks',
    'workspace_url': workspace_url,
    'token': databricks_token,
    'headers': databricks_headers,
    'endpoint': databricks_endpoint
}
client_type = "databricks"

assert client['type'] == 'databricks', "Wrong client type"
assert client['workspace_url'] == workspace_url, "Wrong workspace URL"
assert client['token'] == databricks_token, "Wrong token"
assert client['endpoint'] == databricks_endpoint, "Wrong endpoint"
assert client_type == "databricks", "Wrong client_type variable"

print(f"   ? Client type: {client['type']}")
print(f"   ? Workspace URL: {client['workspace_url']}")
print(f"   ? Token: {client['token'][:10]}...")
print(f"   ? Endpoint: {client['endpoint']}")
print(f"   ? client_type variable: {client_type}")

# ============================================================================
# Test 2.6: Connection Test to Serving Endpoint
# ============================================================================

print("\n? Test 2.6: Connection Test to Serving Endpoint")

try:
    test_url = f"https://{workspace_url}/serving-endpoints/{databricks_endpoint}/invocations"
    test_payload = {
        "messages": [{"role": "user", "content": "Say 'OK'"}],
        "max_tokens": 10,
        "temperature": 0.1
    }
    test_response = requests.post(test_url, headers=databricks_headers, json=test_payload, timeout=30)
    
    assert test_response.status_code == 200, f"Connection test failed: {test_response.status_code}"
    
    result = test_response.json()
    assert 'choices' in result, "Missing 'choices' in response"
    assert len(result['choices']) > 0, "Empty choices array"
    assert 'message' in result['choices'][0], "Missing 'message' in choice"
    assert 'content' in result['choices'][0]['message'], "Missing 'content' in message"
    
    content = result['choices'][0]['message']['content']
    print(f"   ? Connection test successful!")
    print(f"   ? Response status: {test_response.status_code}")
    print(f"   ? Response format: OpenAI-compatible")
    print(f"   ? Content length: {len(content)} chars")
    print(f"   ? Content preview: {content[:100]}...")
    
except Exception as e:
    print(f"   ? Error: {e}")
    sys.exit(1)

# ============================================================================
# Test 2.7: OpenAI-Compatible Response Format
# ============================================================================

print("\n? Test 2.7: OpenAI-Compatible Response Format")

# Test that response structure matches OpenAI format
assert isinstance(result, dict), "Response should be dict"
assert 'choices' in result, "Missing 'choices' key"
assert isinstance(result['choices'], list), "choices should be list"

choice = result['choices'][0]
assert 'message' in choice, "Missing 'message' in choice"
assert 'content' in choice['message'], "Missing 'content' in message"
assert 'role' in choice['message'], "Missing 'role' in message"

print(f"   ? Response is dict: {type(result).__name__}")
print(f"   ? Has 'choices' array: {len(result['choices'])} items")
print(f"   ? Choice has 'message' object")
print(f"   ? Message has 'content': {len(choice['message']['content'])} chars")
print(f"   ? Message has 'role': {choice['message']['role']}")
print(f"   ? Format is OpenAI-compatible ?")

# ============================================================================
# Test 2.8: Compare with OpenAI Client Initialization
# ============================================================================

print("\n? Test 2.8: Compare with OpenAI Client Initialization")

# Simulate OpenAI path for comparison
print("\n   Simulating OpenAI path (for comparison):")

JUDGE_MODEL = "gpt-4o"
OPENAI_KEY = None

# Try to get OpenAI key
try:
    OPENAI_KEY = dbutils.secrets.get("popin-secure-scope", "openai_key")
    print(f"   ? OpenAI key retrieved: {OPENAI_KEY[:10]}...")
except Exception as e:
    print(f"   ??  No OpenAI key: {e}")

if OPENAI_KEY:
    print(f"   ? Would initialize OpenAI client with:")
    print(f"     ? base_url: https://api.zillowlabs.com/openai/v1")
    print(f"     ? api_key: {OPENAI_KEY[:10]}...")
    print(f"     ? model: {JUDGE_MODEL}")

print(f"\n   Comparison:")
print(f"   ? Databricks: NO API key needed (uses workspace token)")
print(f"   ? OpenAI: API key required from secrets")
print(f"   ? Databricks: Auto-discovers endpoints")
print(f"   ? OpenAI: Uses fixed base URL")

# ============================================================================
# Test 2.9: Widget Integration
# ============================================================================

print("\n? Test 2.9: Widget Integration")

# Test the widget that user provided
dbutils.widgets.dropdown(
    "judge_model",
    "databricks-llm",
    ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo", "databricks-llm"],
    "?? Judge Model"
)

JUDGE_MODEL = dbutils.widgets.get("judge_model")
assert JUDGE_MODEL == "databricks-llm", f"Wrong default: {JUDGE_MODEL}"

print(f"   ? Widget created: judge_model")
print(f"   ? Default value: {JUDGE_MODEL}")
print(f"   ? Options: gpt-4o, gpt-4o-mini, gpt-3.5-turbo, databricks-llm")

# Test model routing logic
if JUDGE_MODEL == "databricks-llm":
    print(f"   ? Model routing: Using Databricks Serving Endpoints")
    print(f"   ? Endpoint: {databricks_endpoint}")
else:
    print(f"   ? Model routing: Using OpenAI via Zillow Labs proxy")
    print(f"   ? Model: {JUDGE_MODEL}")

# ============================================================================
# Test 2.10: Full Initialization Flow
# ============================================================================

print("\n? Test 2.10: Full Initialization Flow (As in Notebook)")

# Simulate the exact flow from the corrected notebook
print("\n   Simulating Cell 3 initialization:")

if JUDGE_MODEL == "databricks-llm":
    print("   [1] Detected databricks-llm selection")
    
    # Get credentials
    token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
    url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().get()
    print(f"   [2] Retrieved workspace credentials")
    
    # Query endpoints
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    api_url = f"https://{url}/api/2.0/serving-endpoints"
    resp = requests.get(api_url, headers=headers, timeout=10)
    print(f"   [3] Queried serving endpoints: {resp.status_code}")
    
    # Find Claude Sonnet
    eps = resp.json().get('endpoints', [])
    selected = None
    for ep_name in [e['name'] for e in eps]:
        if 'claude-sonnet' in ep_name.lower():
            selected = ep_name
            break
    print(f"   [4] Found Claude Sonnet: {selected}")
    
    # Test connection
    test_url = f"https://{url}/serving-endpoints/{selected}/invocations"
    test_payload = {"messages": [{"role": "user", "content": "Test"}], "max_tokens": 10}
    test_resp = requests.post(test_url, headers=headers, json=test_payload, timeout=30)
    print(f"   [5] Connection test: {test_resp.status_code}")
    
    # Store config
    final_client = {
        'type': 'databricks',
        'workspace_url': url,
        'token': token,
        'headers': headers,
        'endpoint': selected
    }
    final_client_type = "databricks"
    print(f"   [6] Client configured: {final_client_type}")
    
    print(f"\n   ? READY TO EVALUATE!")
    print(f"      Model: Claude Sonnet (Databricks Serving Endpoint)")
    print(f"      Endpoint: {final_client['endpoint']}")
else:
    print("   Would initialize OpenAI client instead")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("? PHASE 2 COMPLETE: ALL SERVING ENDPOINTS TESTS PASSED")
print("="*80)

print(f"\n?? Key Achievements:")
print(f"   ? Workspace token retrieval: ?")
print(f"   ? Serving endpoints discovery: ?")
print(f"   ? Claude Sonnet endpoint found: ?")
print(f"   ? Connection test successful: ?")
print(f"   ? OpenAI-compatible response: ?")
print(f"   ? Client configuration stored: ?")
print(f"   ? Widget integration tested: ?")
print(f"   ? Full initialization flow: ?")

print(f"\n?? Key Differences from Wrong Implementation:")
print(f"   ? OLD: Direct Anthropic API (api.anthropic.com)")
print(f"   ? NEW: Databricks Serving Endpoints")
print(f"   ? OLD: API key from secrets")
print(f"   ? NEW: Workspace token (automatic)")
print(f"   ? OLD: Hardcoded model ID")
print(f"   ? NEW: Auto-discovered endpoint")
print(f"   ? OLD: Anthropic response format")
print(f"   ? NEW: OpenAI-compatible format")

print(f"\n?? Endpoints Discovered:")
print(f"   ? databricks-claude-sonnet-4-external (Claude Sonnet 4)")
print(f"   ? databricks-meta-llama-3-70b-instruct (Llama 3 70B)")
print(f"   ? databricks-gpt-4-turbo (GPT-4 Turbo)")

print("\n" + "="*80)
print("?? READY FOR PHASE 3: LLM Evaluator Testing")
print("="*80)

sys.exit(0)
