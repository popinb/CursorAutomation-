# Databricks notebook source
# MAGIC %md
# MAGIC ## 7. Core Evaluation Framework

# COMMAND ----------

class LLMJudgeEvaluator:
    """Main evaluator class for running LLM-based metrics."""
    
    def __init__(self, config: EvaluationConfig, metrics: List[MetricConfig]):
        self.config = config
        self.metrics = metrics
        self.models = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize LLM models for evaluation."""
        for model_name in self.config.judge_models:
            try:
                self.models[model_name] = ChatOpenAI(
                    model_name=model_name,
                    temperature=0,
                    model_kwargs={"response_format": {"type": "json_object"}}
                )
            except Exception as e:
                print(f"Warning: Could not initialize model {model_name}: {e}")
    
    def _create_evaluator_function(self, metric: MetricConfig):
        """Create an evaluator function for a specific metric."""
        def evaluator(eval_df, builtin_metrics=None):
            results = []
            details = []
            
            for _, row in eval_df.iterrows():
                # Prepare variables for the prompt template
                template_vars = {}
                for var in metric.required_variables:
                    if var == "prompt":
                        template_vars[var] = row.get(self.config.prompt_column, "")
                    elif var == "response":
                        template_vars[var] = row.get(self.config.response_column, "")
                    elif var == "user_profile":
                        template_vars[var] = row.get("user_profile", "")
                    elif var == "context":
                        template_vars[var] = row.get("context", "")
                    else:
                        template_vars[var] = row.get(var, "")
                
                # Format the evaluation prompt
                eval_prompt = metric.prompt_template.format(**template_vars)
                
                # Get evaluation from all judge models
                model_scores = []
                model_details = []
                
                for model_name, model in self.models.items():
                    try:
                        llm_response = model.invoke(eval_prompt)
                        result_json = json.loads(llm_response.content)
                        
                        score_key = f"{metric.name}_score"
                        score = result_json.get(score_key, 0)
                        
                        # Convert to appropriate scale
                        if metric.metric_type == MetricType.CATEGORICAL and metric.scale_max:
                            score = float(score) / metric.scale_max
                        
                        model_scores.append(score)
                        model_details.append({
                            "model": model_name,
                            "score": score,
                            "explanation": result_json.get("explanation", ""),
                            "raw_response": result_json
                        })
                        
                    except Exception as e:
                        print(f"Error evaluating {metric.name} with {model_name}: {e}")
                        model_scores.append(0.0)
                        model_details.append({
                            "model": model_name,
                            "error": str(e)
                        })
                
                # Use majority vote or average for final score
                if len(model_scores) > 1:
                    # Majority vote for binary, average for others
                    if metric.metric_type == MetricType.BINARY:
                        final_score = 1.0 if sum(model_scores) > len(model_scores) / 2 else 0.0
                    else:
                        final_score = sum(model_scores) / len(model_scores)
                else:
                    final_score = model_scores[0] if model_scores else 0.0
                
                results.append(final_score)
                details.append({
                    "final_score": final_score,
                    "model_details": model_details
                })
            
            return {
                f"{metric.name}/mean": sum(results) / len(results) if results else 0.0,
                f"{metric.name}/scores": results,
                f"{metric.name}/details": details,
            }
        
        return evaluator
    
    def run_evaluation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run evaluation on the provided dataframe."""
        print(f"Running evaluation with {len(self.metrics)} metrics...")
        
        # Prepare data for evaluation
        eval_data = df.copy()
        eval_data = eval_data.rename(columns={
            self.config.prompt_column: "inputs",
            self.config.response_column: "predictions"
        })
        
        # Run each metric
        for metric in self.metrics:
            print(f"Evaluating {metric.name}...")
            evaluator_func = self._create_evaluator_function(metric)
            results = evaluator_func(eval_data)
            
            # Add results to dataframe
            df[f"{metric.name}_score"] = results[f"{metric.name}/scores"]
            df[f"{metric.name}_details"] = results[f"{metric.name}/details"]
            
            # Add status based on threshold
            if metric.threshold is not None:
                df[f"{metric.name}_status"] = [
                    "✅" if score >= metric.threshold else "❌" 
                    for score in results[f"{metric.name}/scores"]
                ]
            
            print(f"✅ {metric.name}: {results[f'{metric.name}/mean']:.3f}")
        
        return df

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. MLflow Integration and Visualization

# COMMAND ----------

class MLflowVisualizer:
    """Handles MLflow logging and visualization of evaluation results."""
    
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
    
    def log_evaluation_results(self, df: pd.DataFrame, metrics: List[MetricConfig], run_name: str):
        """Log evaluation results to MLflow."""
        with mlflow.start_run(run_name=run_name):
            # Log overall metrics
            for metric in metrics:
                scores = df[f"{metric.name}_score"]
                mlflow.log_metric(f"{metric.name}_mean", scores.mean())
                mlflow.log_metric(f"{metric.name}_std", scores.std())
                mlflow.log_metric(f"{metric.name}_min", scores.min())
                mlflow.log_metric(f"{metric.name}_max", scores.max())
                
                if metric.threshold is not None:
                    pass_rate = (scores >= metric.threshold).mean()
                    mlflow.log_metric(f"{metric.name}_pass_rate", pass_rate)
            
            # Log sample data
            sample_data = df.head(10).to_dict('records')
            mlflow.log_text(json.dumps(sample_data, indent=2), "sample_data.json")
            
            # Create and log visualizations
            self._create_visualizations(df, metrics)
    
    def _create_visualizations(self, df: pd.DataFrame, metrics: List[MetricConfig]):
        """Create visualization plots for the metrics."""
        # Create subplots for different metric types
        fig = make_subplots(
            rows=len(metrics), 
            cols=2,
            subplot_titles=[f"{m.name} Distribution" for m in metrics] + 
                          [f"{m.name} Over Time" for m in metrics],
            specs=[[{"secondary_y": False}, {"secondary_y": False}] for _ in metrics]
        )
        
        for i, metric in enumerate(metrics, 1):
            scores = df[f"{metric.name}_score"]
            
            # Distribution plot
            fig.add_trace(
                go.Histogram(x=scores, name=f"{metric.name}_dist", nbinsx=20),
                row=i, col=1
            )
            
            # Time series plot (if index represents time)
            fig.add_trace(
                go.Scatter(x=list(range(len(scores))), y=scores, 
                          mode='lines+markers', name=f"{metric.name}_series"),
                row=i, col=2
            )
        
        fig.update_layout(height=300 * len(metrics), showlegend=False)
        
        # Log the plot
        mlflow.log_figure(fig, f"metrics_visualization.html")
        
        # Create summary table
        summary_data = []
        for metric in metrics:
            scores = df[f"{metric.name}_score"]
            summary_data.append({
                "Metric": metric.name,
                "Mean": f"{scores.mean():.3f}",
                "Std": f"{scores.std():.3f}",
                "Min": f"{scores.min():.3f}",
                "Max": f"{scores.max():.3f}",
                "Pass Rate": f"{(scores >= metric.threshold).mean():.1%}" if metric.threshold else "N/A"
            })
        
        summary_df = pd.DataFrame(summary_data)
        mlflow.log_table(summary_df, "metrics_summary.json")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Sample Data Generation

# COMMAND ----------

def create_sample_data() -> pd.DataFrame:
    """Create sample data for testing the evaluation system."""
    sample_data = {
        "prompt": [
            "What's the best way to buy a house in Seattle?",
            "I have a credit score of 750, can I get a mortgage?",
            "How much should I save for a down payment?",
            "What are the current mortgage rates?",
            "Should I buy or rent in this market?"
        ],
        "response": [
            "To buy a house in Seattle, you should first get pre-approved for a mortgage, work with a local real estate agent, and be prepared for a competitive market. Consider your budget, location preferences, and timeline.",
            "With a credit score of 750, you're in excellent position to qualify for a mortgage. You'll likely get the best interest rates available. I recommend getting pre-approved to see your exact loan options.",
            "Aim to save 20% of the home's purchase price for a down payment to avoid PMI. However, many programs allow as little as 3-5% down. Consider your monthly payment comfort level when deciding.",
            "Current mortgage rates vary by loan type and your credit profile. As of today, 30-year fixed rates are around 6.5-7%, but rates change daily. Check with lenders for current rates.",
            "The buy vs rent decision depends on your financial situation, timeline, and local market conditions. Generally, if you plan to stay 5+ years and can afford the monthly payment, buying often makes sense."
        ],
        "user_profile": [
            "Location: Seattle, WA; Income: $120k; Credit Score: 720; Down Payment: $50k",
            "Location: Seattle, WA; Income: $120k; Credit Score: 750; Down Payment: $50k",
            "Location: Seattle, WA; Income: $120k; Credit Score: 720; Down Payment: $50k",
            "Location: Seattle, WA; Income: $120k; Credit Score: 720; Down Payment: $50k",
            "Location: Seattle, WA; Income: $120k; Credit Score: 720; Down Payment: $50k"
        ]
    }
    return pd.DataFrame(sample_data)

# COMMAND ----------