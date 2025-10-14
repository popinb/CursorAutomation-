# COMMAND ----------

# Core system classes
import time
import requests
import json

class MetricType(Enum):
    BINARY = "binary"
    SCALE_1_5 = "scale_1_5"
    PERCENTAGE = "percentage"

@dataclass
class MetricConfig:
    """Configuration for a metric."""
    name: str
    description: str
    metric_type: MetricType
    prompt_template: str
    threshold: float

class LLMJudgeEvaluator:
    """Main evaluator class with proper LLM routing."""
    
    def __init__(self, judge_model: str, metrics: List[MetricConfig]):
        self.judge_model = judge_model
        self.metrics = metrics
        self.is_databricks_llm = judge_model == "databricks-llm"
        
        # Initialize the appropriate client
        if self.is_databricks_llm:
            self._initialize_databricks_client()
        else:
            self._initialize_openai_client()
    
    def _initialize_databricks_client(self):
        """Initialize Databricks native LLM client."""
        print("🔧 Initializing Databricks LLM client...")
        
        try:
            # Get Databricks workspace credentials
            self.databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
            self.workspace_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().get()
            
            # Set up headers for Databricks API calls
            self.databricks_headers = {
                "Authorization": f"Bearer {self.databricks_token}",
                "Content-Type": "application/json"
            }
            
            # Test Databricks connection
            self._test_databricks_connection()
            print("✅ Databricks LLM client initialized successfully")
            
        except Exception as e:
            print(f"❌ Failed to initialize Databricks client: {e}")
            raise
    
    def _initialize_openai_client(self):
        """Initialize OpenAI client (existing logic)."""
        print(f"🔧 Initializing OpenAI client for model: {self.judge_model}")
        
        # Use the existing global client from previous cells
        if 'client' not in globals():
            raise ValueError("OpenAI client not initialized. Please run previous cells first.")
        
        self.openai_client = client
        
        # Test OpenAI connection
        self._test_openai_connection()
        print(f"✅ OpenAI client initialized for {self.judge_model}")
    
    def _test_databricks_connection(self):
        """Test Databricks LLM connection."""
        try:
            # Try to list available endpoints first
            url = f"https://{self.workspace_url}/api/2.0/serving-endpoints"
            response = requests.get(url, headers=self.databricks_headers, timeout=10)
            
            if response.status_code == 200:
                endpoints = response.json().get('endpoints', [])
                available_models = [ep['name'] for ep in endpoints if ep.get('state', {}).get('ready', {}).get('update_state') == 'UPDATE_STATE_READY']
                
                if available_models:
                    # Use the first available model or a default one
                    self.databricks_endpoint = available_models[0]
                    print(f"   Available Databricks models: {available_models}")
                    print(f"   Using endpoint: {self.databricks_endpoint}")
                else:
                    # Fallback to common Databricks model names
                    fallback_models = ["databricks-dbrx-instruct", "databricks-llama-2-70b-chat", "databricks-mpt-30b-instruct"]
                    self.databricks_endpoint = fallback_models[0]
                    print(f"   No ready endpoints found, using fallback: {self.databricks_endpoint}")
            else:
                print(f"   Could not list endpoints (status {response.status_code}), using default")
                self.databricks_endpoint = "databricks-dbrx-instruct"
                
        except Exception as e:
            print(f"   Warning: Could not test Databricks connection: {e}")
            self.databricks_endpoint = "databricks-dbrx-instruct"  # Default fallback
    
    def _test_openai_connection(self):
        """Test OpenAI connection."""
        try:
            # Use model-specific parameters for testing
            if "gpt-5" in self.judge_model.lower() or self.judge_model in ["gpt-5-chat-latest"]:
                test_response = self.openai_client.chat.completions.create(
                    model=self.judge_model,
                    messages=[{"role": "user", "content": "Say 'OK'"}],
                    max_completion_tokens=10
                )
            else:
                test_response = self.openai_client.chat.completions.create(
                    model=self.judge_model,
                    messages=[{"role": "user", "content": "Say 'OK'"}],
                    max_tokens=10
                )
            
            if test_response.choices and test_response.choices[0].message.content:
                print(f"   ✅ OpenAI test successful: {test_response.choices[0].message.content}")
            else:
                raise Exception("Empty response from OpenAI")
                
        except Exception as e:
            print(f"   ❌ OpenAI test failed: {e}")
            raise
    
    def _call_databricks_llm(self, prompt: str) -> str:
        """Call Databricks LLM endpoint."""
        try:
            # Databricks serving endpoint URL
            url = f"https://{self.workspace_url}/serving-endpoints/{self.databricks_endpoint}/invocations"
            
            # Prepare the payload for Databricks
            payload = {
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 1000,
                "temperature": 0.1
            }
            
            # Make the API call
            response = requests.post(url, headers=self.databricks_headers, json=payload, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                
                # Handle different response formats
                if 'choices' in result and result['choices']:
                    return result['choices'][0]['message']['content']
                elif 'predictions' in result and result['predictions']:
                    pred = result['predictions'][0]
                    if 'candidates' in pred:
                        return pred['candidates'][0]['message']['content']
                    elif 'generated_text' in pred:
                        return pred['generated_text']
                    else:
                        return str(pred)
                else:
                    return str(result)
            else:
                raise Exception(f"Databricks API error {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"   Error calling Databricks LLM: {e}")
            raise
    
    def _call_openai_llm(self, prompt: str) -> str:
        """Call OpenAI LLM endpoint."""
        try:
            # Use model-specific parameters
            if "gpt-5" in self.judge_model.lower() or self.judge_model in ["gpt-5-chat-latest"]:
                response = self.openai_client.chat.completions.create(
                    model=self.judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=1000,
                    response_format={"type": "json_object"}
                )
            else:
                response = self.openai_client.chat.completions.create(
                    model=self.judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000,
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"   Error calling OpenAI LLM: {e}")
            raise
    
    def evaluate_single(self, prompt: str, response: str, ground_truth: str, metric: MetricConfig) -> dict:
        """Evaluate a single sample with one metric."""
        # Format the evaluation prompt
        eval_prompt = metric.prompt_template.format(
            prompt=prompt,
            response=response,
            ground_truth=ground_truth if ground_truth else "Not provided"
        )
        
        try:
            # Route to appropriate LLM based on model type
            if self.is_databricks_llm:
                print(f"   🏢 Calling Databricks LLM for {metric.name}")
                llm_response = self._call_databricks_llm(eval_prompt)
            else:
                print(f"   🤖 Calling OpenAI ({self.judge_model}) for {metric.name}")
                llm_response = self._call_openai_llm(eval_prompt)
            
            # Check if we got a valid response
            if not llm_response or llm_response.strip() == "":
                print(f"   Warning: Empty response for {metric.name}")
                return {
                    "score": 0,
                    "explanation": "Empty response from LLM",
                    "status": "❌"
                }
            
            # Clean the response content
            content = llm_response.strip()
            
            # Remove markdown code blocks if present
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
            elif content.startswith("```"):
                content = content.replace("```", "").strip()
            
            # Parse JSON response
            try:
                result_json = json.loads(content)
            except json.JSONDecodeError:
                print(f"   Warning: Invalid JSON for {metric.name}")
                print(f"   Raw content: {content[:200]}...")
                return {
                    "score": 0,
                    "explanation": f"Invalid JSON response: {content[:200]}",
                    "status": "❌"
                }
            
            # Extract score with flexible key matching
            score = 0
            explanation = "No explanation provided"
            
            # Clean all keys by removing quotes and normalizing
            cleaned_json = {}
            for key, value in result_json.items():
                clean_key = key.strip('"').strip("'").strip()
                cleaned_json[clean_key] = value
            
            # Method 1: Try exact key match
            score_key = f"{metric.name}_score"
            if score_key in cleaned_json:
                score = cleaned_json[score_key]
            
            # Method 2: Flexible matching
            else:
                for key, value in cleaned_json.items():
                    if (metric.name.lower() in key.lower() and 
                        "score" in key.lower() and 
                        isinstance(value, (int, float))):
                        score = value
                        break
                
                # Method 3: Any "_score" key
                if score == 0:
                    for key, value in cleaned_json.items():
                        if key.endswith("_score") and isinstance(value, (int, float)):
                            score = value
                            break
                
                # Method 4: Just "score"
                if score == 0 and "score" in cleaned_json:
                    if isinstance(cleaned_json["score"], (int, float)):
                        score = cleaned_json["score"]
            
            # Extract explanation
            for key in ["explanation", "Explanation", "reason", "Reason"]:
                if key in cleaned_json:
                    explanation = cleaned_json[key]
                    break
            
            return {
                "score": score,
                "explanation": explanation,
                "status": "✅" if score >= metric.threshold else "❌"
            }
            
        except Exception as e:
            print(f"Error evaluating {metric.name}: {e}")
            return {
                "score": 0,
                "explanation": f"Evaluation error: {str(e)}",
                "status": "❌"
            }
    
    def evaluate_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Evaluate entire dataset."""
        results_df = df.copy()
        
        llm_type = "Databricks LLM" if self.is_databricks_llm else f"OpenAI ({self.judge_model})"
        print(f"\n🚀 Starting evaluation of {len(df)} samples with {len(self.metrics)} metrics...")
        print(f"   Using: {llm_type}")
        
        for metric in self.metrics:
            print(f"\n📊 Evaluating metric: {metric.name}")
            
            scores = []
            explanations = []
            statuses = []
            
            for idx, row in df.iterrows():
                if idx > 0 and idx % 5 == 0:
                    print(f"   Progress: {idx}/{len(df)} samples")
                
                result = self.evaluate_single(
                    prompt=row['prompt'],
                    response=row['response'],
                    ground_truth=row.get('ground_truth', ''),
                    metric=metric
                )
                
                scores.append(result['score'])
                explanations.append(result['explanation'])
                statuses.append(result['status'])
                
                # Add small delay for Databricks to avoid overwhelming the endpoint
                if self.is_databricks_llm:
                    time.sleep(0.5)
            
            # Add results to dataframe
            results_df[f"{metric.name}_score"] = scores
            results_df[f"{metric.name}_explanation"] = explanations
            results_df[f"{metric.name}_status"] = statuses
            
            # Calculate summary statistics
            mean_score = sum(scores) / len(scores) if scores else 0
            pass_rate = sum(1 for s in scores if s >= metric.threshold) / len(scores) if scores else 0
            
            print(f"   ✅ Complete - Mean: {mean_score:.3f}, Pass Rate: {pass_rate:.1%}")
        
        return results_df

# DEFINE HELPER FUNCTIONS
def process_custom_metrics(custom_metrics: list) -> List[MetricConfig]:
    """Convert custom metric definitions to MetricConfig objects."""
    configs = []
    
    for metric in custom_metrics:
        # Map metric type
        if metric['type'] == 'binary':
            metric_type = MetricType.BINARY
        elif metric['type'] == 'scale_1_5':
            metric_type = MetricType.SCALE_1_5
        elif metric['type'] == 'percentage':
            metric_type = MetricType.PERCENTAGE
        else:
            print(f"   Warning: Unknown metric type '{metric['type']}' for {metric.get('name', 'unknown')}")
            continue
        
        # Get threshold from METRIC_THRESHOLDS or use default
        threshold = METRIC_THRESHOLDS.get(metric['name'], 1.0)
        
        config = MetricConfig(
            name=metric['name'],
            description=metric['description'],
            metric_type=metric_type,
            prompt_template=metric['evaluation_prompt'],
            threshold=threshold
        )
        configs.append(config)
    
    return configs

print("✅ Refactored system classes loaded with proper LLM routing")

# EXECUTION LOGIC
print("📊 Using CUSTOM_METRICS from cell 5...")

# Process metrics and create evaluator
metric_configs = process_custom_metrics(CUSTOM_METRICS)

if not metric_configs:
    print("❌ No valid metrics to evaluate!")
    print("Please define metrics in cell 5")
else:
    # Create evaluator with proper routing
    evaluator = LLMJudgeEvaluator(
        judge_model=JUDGE_MODEL,
        metrics=metric_configs
    )
    
    # Run evaluation
    print("="*60)
    print("🚀 STARTING EVALUATION")
    print("="*60)
    
    start_time = time.time()
    results_df = evaluator.evaluate_dataset(EVALUATION_DATA)
    eval_time = time.time() - start_time
    
    print("\n" + "="*60)
    print(f"✅ EVALUATION COMPLETE in {eval_time:.1f} seconds")
    print("="*60)
    
    # Display sample results
    print("\n📊 Sample Results:")
    display_cols = ['prompt', 'response'] + [f"{m.name}_score" for m in metric_configs] + [f"{m.name}_status" for m in metric_configs]
    display_cols = [col for col in display_cols if col in results_df.columns]
    display(results_df[display_cols].head())