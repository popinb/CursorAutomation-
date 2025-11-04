"""
Enhanced Mock Databricks Utilities for Testing
Includes Databricks Serving Endpoints API simulation
"""

class MockNotebookContext:
    """Mock notebook context for testing."""
    
    def __init__(self):
        self._username = "test_user@example.com"
        self._api_token = "dapi1234567890abcdef"
        self._browser_host = "my-workspace.databricks.com"
    
    def userName(self):
        class UserName:
            def get(self):
                return "test_user@example.com"
        return UserName()
    
    def apiToken(self):
        class ApiToken:
            def get(self):
                return "dapi1234567890abcdef"
        return ApiToken()
    
    def browserHostName(self):
        class BrowserHostName:
            def get(self):
                return "my-workspace.databricks.com"
        return BrowserHostName()


class MockNotebookEntryPoint:
    """Mock notebook entry point."""
    
    def __init__(self):
        self._context = MockNotebookContext()
    
    def getDbutils(self):
        class DBUtils:
            def notebook(self):
                class Notebook:
                    def getContext(self):
                        return MockNotebookContext()
                return Notebook()
        return DBUtils()


class MockNotebook:
    """Mock notebook utilities."""
    
    def __init__(self):
        self.entry_point = MockNotebookEntryPoint()
    
    def getContext(self):
        return MockNotebookContext()


class MockSecrets:
    """Mock secrets management."""
    
    def __init__(self):
        self._secrets = {
            'popin-secure-scope': {
                'openai_key': 'sk-test-openai-key-1234567890',
                'anthropic_key': 'sk-ant-test-key-1234567890'
            },
            'user_test_user_at_example_com_secrets': {
                'openai_key': 'sk-test-personal-openai-key'
            }
        }
    
    def get(self, scope, key):
        """Get secret value."""
        if scope not in self._secrets:
            raise Exception(f"Scope '{scope}' does not exist")
        if key not in self._secrets[scope]:
            raise Exception(f"Key '{key}' not found in scope '{scope}'")
        return self._secrets[scope][key]
    
    def list(self, scope):
        """List keys in scope."""
        if scope not in self._secrets:
            raise Exception(f"Scope '{scope}' does not exist")
        return [{'key': k} for k in self._secrets[scope].keys()]
    
    def listScopes(self):
        """List all scopes."""
        return [{'name': scope} for scope in self._secrets.keys()]


class MockWidgets:
    """Mock widgets management."""
    
    def __init__(self):
        self._widgets = {}
    
    def text(self, name, default, label):
        """Create text widget."""
        if name not in self._widgets:
            self._widgets[name] = default
    
    def dropdown(self, name, default, choices, label):
        """Create dropdown widget."""
        if name not in self._widgets:
            self._widgets[name] = default
    
    def combobox(self, name, default, choices, label):
        """Create combobox widget."""
        if name not in self._widgets:
            self._widgets[name] = default
    
    def multiselect(self, name, default, choices, label):
        """Create multiselect widget."""
        if name not in self._widgets:
            self._widgets[name] = default
    
    def get(self, name):
        """Get widget value."""
        if name not in self._widgets:
            raise Exception(f"Widget '{name}' does not exist")
        return self._widgets[name]
    
    def remove(self, name):
        """Remove widget."""
        if name in self._widgets:
            del self._widgets[name]
    
    def removeAll(self):
        """Remove all widgets."""
        self._widgets.clear()
    
    def set_value(self, name, value):
        """Set widget value (for testing)."""
        self._widgets[name] = value


class MockLibrary:
    """Mock library management."""
    
    def restartPython(self):
        """Mock Python restart."""
        print("Python interpreter restarted (mock)")


class MockServingEndpoints:
    """Mock Databricks Serving Endpoints for testing."""
    
    def __init__(self):
        # Simulate available serving endpoints
        self._endpoints = [
            {
                'name': 'databricks-claude-sonnet-4-external',
                'creator': 'admin@databricks.com',
                'creation_timestamp': 1699564800000,
                'last_updated_timestamp': 1699651200000,
                'state': {'ready': 'READY'},
                'config': {
                    'served_models': [{
                        'name': 'claude-sonnet-4',
                        'model_name': 'anthropic.claude-sonnet-4',
                        'model_version': '1',
                        'workload_size': 'Small'
                    }]
                },
                'tags': [{'key': 'provider', 'value': 'anthropic'}]
            },
            {
                'name': 'databricks-meta-llama-3-70b-instruct',
                'creator': 'admin@databricks.com',
                'creation_timestamp': 1699564800000,
                'last_updated_timestamp': 1699651200000,
                'state': {'ready': 'READY'},
                'config': {
                    'served_models': [{
                        'name': 'llama-3-70b',
                        'model_name': 'meta.llama-3-70b-instruct',
                        'model_version': '1',
                        'workload_size': 'Medium'
                    }]
                },
                'tags': [{'key': 'provider', 'value': 'meta'}]
            },
            {
                'name': 'databricks-gpt-4-turbo',
                'creator': 'admin@databricks.com',
                'creation_timestamp': 1699564800000,
                'last_updated_timestamp': 1699651200000,
                'state': {'ready': 'READY'},
                'config': {
                    'served_models': [{
                        'name': 'gpt-4-turbo',
                        'model_name': 'openai.gpt-4-turbo',
                        'model_version': '1',
                        'workload_size': 'Small'
                    }]
                },
                'tags': [{'key': 'provider', 'value': 'openai'}]
            }
        ]
        
        # Mock responses for inference
        self._inference_responses = {
            'databricks-claude-sonnet-4-external': self._generate_claude_response,
            'databricks-meta-llama-3-70b-instruct': self._generate_llama_response,
            'databricks-gpt-4-turbo': self._generate_gpt_response
        }
    
    def list_endpoints(self):
        """List all serving endpoints."""
        return {'endpoints': self._endpoints}
    
    def get_endpoint(self, name):
        """Get specific endpoint."""
        for ep in self._endpoints:
            if ep['name'] == name:
                return ep
        raise Exception(f"Endpoint '{name}' not found")
    
    def invoke(self, endpoint_name, payload):
        """Invoke serving endpoint."""
        if endpoint_name not in [ep['name'] for ep in self._endpoints]:
            return {'error': f"Endpoint '{endpoint_name}' not found"}, 404
        
        # Get the response generator for this endpoint
        response_fn = self._inference_responses.get(
            endpoint_name, 
            self._generate_default_response
        )
        
        # Generate response
        return response_fn(payload), 200
    
    def _generate_claude_response(self, payload):
        """Generate Claude-like response."""
        messages = payload.get('messages', [])
        if not messages:
            return {'error': 'No messages provided'}
        
        user_message = messages[-1].get('content', '')
        
        # Parse evaluation request and generate appropriate response
        if 'Story_Accuracy' in user_message or 'factually accurate' in user_message:
            score = 1
            explanation = "The response accurately reflects the Cinderella story with correct facts about the stepmother, stepsisters, Fairy Godmother, glass slipper, and the prince's search."
        elif 'Response_Completeness' in user_message or 'complete and thorough' in user_message:
            score = 5
            explanation = "The response is comprehensive and addresses all aspects of the question thoroughly with rich details."
        elif 'Child_Friendliness' in user_message or 'appropriate and understandable for children' in user_message:
            score = 95
            explanation = "The language is perfectly suited for children - simple, clear, and engaging with no complex vocabulary."
        else:
            score = 1
            explanation = "General evaluation: content meets standards."
        
        response_text = f'{{"score": {score}, "explanation": "{explanation}"}}'
        
        return {
            'choices': [
                {
                    'index': 0,
                    'message': {
                        'role': 'assistant',
                        'content': response_text
                    },
                    'finish_reason': 'stop'
                }
            ],
            'usage': {
                'prompt_tokens': len(user_message.split()),
                'completion_tokens': len(response_text.split()),
                'total_tokens': len(user_message.split()) + len(response_text.split())
            }
        }
    
    def _generate_llama_response(self, payload):
        """Generate Llama-like response."""
        messages = payload.get('messages', [])
        user_message = messages[-1].get('content', '') if messages else ''
        
        response_text = '{"score": 1, "explanation": "Llama evaluation response"}'
        
        return {
            'choices': [
                {
                    'index': 0,
                    'message': {
                        'role': 'assistant',
                        'content': response_text
                    },
                    'finish_reason': 'stop'
                }
            ]
        }
    
    def _generate_gpt_response(self, payload):
        """Generate GPT-like response."""
        messages = payload.get('messages', [])
        user_message = messages[-1].get('content', '') if messages else ''
        
        response_text = '{"score": 1, "explanation": "GPT evaluation response"}'
        
        return {
            'choices': [
                {
                    'index': 0,
                    'message': {
                        'role': 'assistant',
                        'content': response_text
                    },
                    'finish_reason': 'stop'
                }
            ]
        }
    
    def _generate_default_response(self, payload):
        """Generate default response."""
        return {
            'choices': [
                {
                    'index': 0,
                    'message': {
                        'role': 'assistant',
                        'content': '{"score": 0, "explanation": "Default response"}'
                    },
                    'finish_reason': 'stop'
                }
            ]
        }


class MockDBUtils:
    """Main mock dbutils class."""
    
    def __init__(self):
        self.widgets = MockWidgets()
        self.secrets = MockSecrets()
        self.notebook = MockNotebook()
        self.library = MockLibrary()
        self.serving_endpoints = MockServingEndpoints()


# Global instance for testing
dbutils = MockDBUtils()


# Mock requests module for Databricks API calls
class MockResponse:
    """Mock HTTP response."""
    
    def __init__(self, json_data, status_code=200):
        self._json_data = json_data
        self.status_code = status_code
        self.text = str(json_data)
    
    def json(self):
        return self._json_data


class MockRequests:
    """Mock requests module for Databricks API."""
    
    def __init__(self, serving_endpoints):
        self.serving_endpoints = serving_endpoints
    
    def get(self, url, headers=None, timeout=None):
        """Mock GET request."""
        if '/api/2.0/serving-endpoints' in url and not url.endswith('/invocations'):
            # List endpoints
            return MockResponse(self.serving_endpoints.list_endpoints(), 200)
        else:
            return MockResponse({'error': 'Not found'}, 404)
    
    def post(self, url, headers=None, json=None, timeout=None):
        """Mock POST request."""
        if '/serving-endpoints/' in url and url.endswith('/invocations'):
            # Extract endpoint name from URL
            # Format: https://{workspace}/serving-endpoints/{endpoint_name}/invocations
            parts = url.split('/serving-endpoints/')
            if len(parts) > 1:
                endpoint_name = parts[1].replace('/invocations', '')
                response_data, status_code = self.serving_endpoints.invoke(endpoint_name, json or {})
                return MockResponse(response_data, status_code)
        
        return MockResponse({'error': 'Not found'}, 404)


# Create global mock requests instance
mock_requests = None

def get_mock_requests():
    """Get mock requests instance."""
    global mock_requests
    if mock_requests is None:
        mock_requests = MockRequests(dbutils.serving_endpoints)
    return mock_requests


if __name__ == "__main__":
    print("="*80)
    print("Enhanced Mock Databricks Utilities Test")
    print("="*80)
    
    # Test 1: Widgets
    print("\n? Test 1: Widgets")
    dbutils.widgets.dropdown("judge_model", "databricks-llm", 
                             ["gpt-4o", "databricks-llm"], "Judge Model")
    print(f"   Widget value: {dbutils.widgets.get('judge_model')}")
    dbutils.widgets.set_value("judge_model", "gpt-4o")
    print(f"   Updated value: {dbutils.widgets.get('judge_model')}")
    
    # Test 2: Secrets
    print("\n? Test 2: Secrets")
    try:
        key = dbutils.secrets.get("popin-secure-scope", "openai_key")
        print(f"   OpenAI key retrieved: {key[:10]}...")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 3: Notebook Context
    print("\n? Test 3: Notebook Context")
    token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
    print(f"   API Token: {token}")
    workspace = dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().get()
    print(f"   Workspace URL: {workspace}")
    
    # Test 4: Serving Endpoints
    print("\n? Test 4: Serving Endpoints Discovery")
    endpoints = dbutils.serving_endpoints.list_endpoints()
    print(f"   Found {len(endpoints['endpoints'])} endpoints:")
    for ep in endpoints['endpoints']:
        print(f"     ? {ep['name']}")
    
    # Test 5: Serving Endpoint Invocation
    print("\n? Test 5: Serving Endpoint Invocation")
    payload = {
        "messages": [
            {"role": "user", "content": "Evaluate if this is factually accurate about Cinderella."}
        ],
        "max_tokens": 500
    }
    response, status = dbutils.serving_endpoints.invoke('databricks-claude-sonnet-4-external', payload)
    print(f"   Status: {status}")
    print(f"   Response: {response['choices'][0]['message']['content'][:100]}...")
    
    # Test 6: Mock Requests
    print("\n? Test 6: Mock Requests Module")
    requests = get_mock_requests()
    response = requests.get("https://my-workspace.databricks.com/api/2.0/serving-endpoints")
    print(f"   Status: {response.status_code}")
    print(f"   Endpoints: {len(response.json()['endpoints'])}")
    
    response = requests.post(
        "https://my-workspace.databricks.com/serving-endpoints/databricks-claude-sonnet-4-external/invocations",
        json=payload
    )
    print(f"   Invocation status: {response.status_code}")
    print(f"   Response content: {response.json()['choices'][0]['message']['content'][:100]}...")
    
    print("\n" + "="*80)
    print("? All mock tests passed!")
    print("="*80)
