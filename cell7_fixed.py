# COMMAND ----------

# Core system classes
import time  # Add missing import

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
    """Main evaluator class."""
    
    def __init__(self, judge_model: str, metrics: List[MetricConfig]):
        self.judge_model = judge_model
        self.metrics = metrics
        
        # Validate client exists
        if 'client' not in globals():
            raise ValueError("OpenAI client not initialized. Please run previous cells first.")
        
        self.client = client  # Store client reference
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the LLM judge."""
        try:
            if self.judge_model == "databricks-llm":
                # Databricks LLM
                self.model = ChatOpenAI(
                    model_name="databricks-llm",
                    temperature=0,
                    model_kwargs={"response_format": {"type": "json_object"}}
                )
            else:
                # Zillow API models using OpenAI client
                def run_chat(messages: list) -> AIMessage:
                    # Extract content from first message
                    content = messages[0].content if messages else ""
                    
                    # Use model-specific parameters
                    if "gpt-5" in self.judge_model.lower() or self.judge_model in ["gpt-5-chat-latest"]:
                        # GPT-5 variants use max_completion_tokens and don't support temperature=0
                        resp = self.client.chat.completions.create(
                            model=self.judge_model,
                            messages=[{"role": "user", "content": content}],
                            max_completion_tokens=1000,
                            response_format={"type": "json_object"}
                        )
                    else:
                        # Other models use max_tokens and temperature=0
                        resp = self.client.chat.completions.create(
                            model=self.judge_model,
                            messages=[{"role": "user", "content": content}],
                            max_tokens=1000,
                            temperature=0.0,
                            response_format={"type": "json_object"}
                        )
                    return AIMessage(content=resp.choices[0].message.content)
                
                self.model = RunnableLambda(run_chat)
        except Exception as e:
            print(f"❌ Error initializing model {self.judge_model}: {e}")
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
            # Get evaluation from LLM
            result = self.model.invoke([HumanMessage(content=eval_prompt)])
            
            # Check if we got a valid response
            if not result or not result.content or result.content.strip() == "":
                print(f"   Warning: Empty response for {metric.name}")
                return {
                    "score": 0,
                    "explanation": "Empty response from LLM",
                    "status": "❌"
                }
            
            # Parse JSON response
            try:
                result_json = json.loads(result.content)
            except json.JSONDecodeError as json_err:
                print(f"   Warning: Invalid JSON for {metric.name}: {result.content[:100]}...")
                return {
                    "score": 0,
                    "explanation": f"Invalid JSON response: {result.content[:200]}",
                    "status": "❌"
                }
            
            # Extract score - NEVER use direct key access that could cause KeyError
            score = 0
            explanation = "No explanation provided"
            
            # Method 1: Try exact key match
            score_key = f"{metric.name}_score"
            if score_key in result_json:
                score = result_json[score_key]
            
            # Method 2: Search for any key containing the metric name and "score"
            elif any(metric.name in key and "score" in key.lower() for key in result_json.keys()):
                for key, value in result_json.items():
                    if metric.name in key and "score" in key.lower() and isinstance(value, (int, float)):
                        score = value
                        break
            
            # Method 3: Search for any key ending with "_score"
            elif any(key.endswith("_score") for key in result_json.keys()):
                for key, value in result_json.items():
                    if key.endswith("_score") and isinstance(value, (int, float)):
                        score = value
                        break
            
            # Method 4: Search for any key containing "score" (last resort)
            else:
                for key, value in result_json.items():
                    if "score" in key.lower() and isinstance(value, (int, float)):
                        score = value
                        break
            
            # Extract explanation safely
            for key in ["explanation", "Explanation", "reason", "Reason"]:
                if key in result_json:
                    explanation = result_json[key]
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
        
        print(f"\n🚀 Starting evaluation of {len(df)} samples with {len(self.metrics)} metrics...")
        print(f"   Using model: {self.judge_model}")
        
        for metric in self.metrics:
            print(f"\n📊 Evaluating metric: {metric.name}")
            
            scores = []
            explanations = []
            statuses = []
            
            for idx, row in df.iterrows():
                if idx > 0 and idx % 10 == 0:
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
            
            # Add results to dataframe
            results_df[f"{metric.name}_score"] = scores
            results_df[f"{metric.name}_explanation"] = explanations
            results_df[f"{metric.name}_status"] = statuses
            
            # Calculate summary statistics
            mean_score = sum(scores) / len(scores) if scores else 0
            pass_rate = sum(1 for s in scores if s >= metric.threshold) / len(scores) if scores else 0
            
            print(f"   ✅ Complete - Mean: {mean_score:.3f}, Pass Rate: {pass_rate:.1%}")
        
        return results_df

# Convert custom metrics to MetricConfig objects
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
            continue  # Skip invalid types
        
        # Safely get threshold with fallback
        threshold = metric.get('threshold', 1.0)
        try:
            threshold = float(threshold)
        except (ValueError, TypeError):
            print(f"   Warning: Invalid threshold '{threshold}' for {metric.get('name', 'unknown')}, using default 1.0")
            threshold = 1.0
        
        config = MetricConfig(
            name=metric['name'],
            description=metric['description'],
            metric_type=metric_type,
            prompt_template=metric['evaluation_prompt'],
            threshold=threshold
        )
        configs.append(config)
    
    return configs

# Load metrics from CSV or use default
def load_metrics_from_config():
    """Load metrics from uploaded CSV or use default metrics."""
    if 'METRICS_CONFIG_DATA' in globals() and METRICS_CONFIG_DATA is not None:
        print("📊 Loading metrics from uploaded CSV...")
        metrics_list = []
        
        for _, row in METRICS_CONFIG_DATA.iterrows():
            metric = {
                'name': row['name'],
                'type': row['type'],
                'description': row['description'],
                'evaluation_prompt': row['evaluation_prompt'],
                'threshold': row['threshold']
            }
            metrics_list.append(metric)
        
        print(f"✅ Loaded {len(metrics_list)} metrics from CSV")
        return metrics_list
    else:
        print("📊 Using default metrics...")
        # Default metrics if no CSV uploaded
        return [
            {
                "name": "accuracy",
                "type": "binary",
                "description": "Checks if the response contains accurate information",
                "evaluation_prompt": """
Evaluate if the response contains accurate information.

User Query: {prompt}
AI Response: {response}
Ground Truth (if available): {ground_truth}

Criteria:
- All facts must be correct
- No misleading information
- Numbers and statistics must be accurate

Return JSON:
{{
    "accuracy_score": 1,
    "explanation": "All information is accurate and verified."
}}
""",
                "threshold": 1.0
            },
            {
                "name": "helpfulness",
                "type": "scale_1_5",
                "description": "Rates how helpful the response is (1-5 scale)",
                "evaluation_prompt": """
Rate the helpfulness of this response from 1 to 5.

User Query: {prompt}
AI Response: {response}
Ground Truth (if available): {ground_truth}

Scale:
5 = Extremely helpful - Comprehensive answer with actionable steps
4 = Very helpful - Good answer with useful information
3 = Moderately helpful - Adequate but could be better
2 = Slightly helpful - Limited value, missing key information
1 = Not helpful - Fails to address the question

Return JSON:
{{
    "helpfulness_score": 4,
    "explanation": "Very helpful response that answers the main question..."
}}
""",
                "threshold": 3.0
            }
        ]

print("✅ System classes loaded")

# COMMAND ----------

# Move the execution logic to after function definitions
# Load metrics - use CUSTOM_METRICS from cell 5 if it exists, otherwise load from config
if 'CUSTOM_METRICS' in globals() and CUSTOM_METRICS:
    print("📊 Using CUSTOM_METRICS from cell 5...")
    metrics_to_use = CUSTOM_METRICS
else:
    print("📊 Loading metrics from configuration...")
    metrics_to_use = load_metrics_from_config()

# Process metrics and create evaluator
metric_configs = process_custom_metrics(metrics_to_use)

if not metric_configs:
    print("❌ No valid metrics to evaluate!")
    print("Please define metrics in cell 5 or upload a metrics configuration CSV file")
else:
    # Validate required variables exist
    required_vars = ['JUDGE_MODEL', 'EVALUATION_DATA']
    missing_vars = [var for var in required_vars if var not in globals()]
    
    if missing_vars:
        print(f"❌ Missing required variables: {missing_vars}")
        print("Please run previous cells first")
    else:
        # Create evaluator
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