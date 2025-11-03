"""
Mock Databricks utilities for local testing.
Simulates dbutils.widgets, secrets, and notebook context.
"""

class MockWidgets:
    """Mock Databricks widgets."""
    
    def __init__(self):
        self._widgets = {}
        self._values = {}
    
    def text(self, name, default, label):
        """Create text widget."""
        self._widgets[name] = {'type': 'text', 'default': default, 'label': label}
        if name not in self._values:
            self._values[name] = default
        print(f"? Widget created: {label} = '{default}'")
    
    def dropdown(self, name, default, choices, label):
        """Create dropdown widget."""
        self._widgets[name] = {'type': 'dropdown', 'default': default, 'choices': choices, 'label': label}
        if name not in self._values:
            self._values[name] = default
        print(f"? Widget created: {label} = '{default}' (options: {len(choices)})")
    
    def multiselect(self, name, default, choices, label):
        """Create multiselect widget."""
        self._widgets[name] = {'type': 'multiselect', 'default': default, 'choices': choices, 'label': label}
        if name not in self._values:
            self._values[name] = default
        print(f"? Widget created: {label}")
    
    def get(self, name):
        """Get widget value."""
        return self._values.get(name, '')
    
    def set(self, name, value):
        """Set widget value (for testing)."""
        self._values[name] = value
        print(f"?? Widget '{name}' set to: '{value}'")
    
    def removeAll(self):
        """Remove all widgets."""
        count = len(self._widgets)
        self._widgets.clear()
        self._values.clear()
        print(f"???  Removed {count} widgets")

class MockSecrets:
    """Mock Databricks secrets."""
    
    def __init__(self):
        self._secrets = {
            'popin-secure-scope': {
                'openai_key': 'sk-mock-test-key-for-testing-only'
            }
        }
    
    def get(self, scope, key):
        """Get secret value."""
        if scope in self._secrets and key in self._secrets[scope]:
            return self._secrets[scope][key]
        raise Exception(f"Secret not found: {scope}/{key}")

class MockNotebookContext:
    """Mock notebook context."""
    
    def __init__(self, username="test.user@example.com"):
        self._username = username
        self._token = "mock-databricks-token"
        self._hostname = "mock-workspace.databricks.com"
    
    def userName(self):
        class UserName:
            def __init__(self, name):
                self._name = name
            def get(self):
                return self._name
        return UserName(self._username)
    
    def apiToken(self):
        class ApiToken:
            def __init__(self, token):
                self._token = token
            def get(self):
                return self._token
        return ApiToken(self._token)
    
    def browserHostName(self):
        class HostName:
            def __init__(self, hostname):
                self._hostname = hostname
            def get(self):
                return self._hostname
        return HostName(self._hostname)

class MockNotebookEntry:
    """Mock notebook entry point."""
    
    def __init__(self, username="test.user@example.com"):
        self._context = MockNotebookContext(username)
    
    def getDbutils(self):
        class MockDbutils:
            def notebook(self):
                class MockNotebook:
                    def __init__(self, context):
                        self._context = context
                    def getContext(self):
                        return self._context
                return MockNotebook(self._context)
        return MockDbutils()

class MockLibrary:
    """Mock library utilities."""
    
    def restartPython(self):
        """Mock Python restart."""
        print("?? Python restart (simulated - not actually restarting)")

class MockDbutils:
    """Main mock dbutils class."""
    
    def __init__(self, username="test.user@example.com"):
        self.widgets = MockWidgets()
        self.secrets = MockSecrets()
        self.notebook = MockNotebookEntry(username)
        self.library = MockLibrary()

# Create global dbutils for testing
dbutils = MockDbutils()

print("? Mock Databricks environment initialized")
