# Example Configuration for PM Metrics Evaluation Framework
# Copy these settings to Section 2 of the main notebook

# =============================================================================
# EXPERIMENT CONFIGURATION - CUSTOMIZE FOR YOUR USE CASE
# =============================================================================

# Experiment Metadata
EXPERIMENT_NAME = "Real_Estate_Content_Quality_Q4_2024"  # Descriptive experiment name
PM_NAME = "Sarah_Johnson"  # Your name for tracking
PROJECT_NAME = "Listing_Description_Optimization"  # Your project name

# Data Configuration
DATA_FILE_PATH = "/databricks/datasets/real_estate_responses.csv"  # Path to your data
PROMPT_COLUMN = "user_query"  # Column with user questions
RESPONSE_COLUMN = "ai_response"  # Column with AI responses
USER_CONTEXT_COLUMN = "user_profile"  # Column with user context/personalization

# Model Configuration  
JUDGE_MODELS = ["gpt-4o"]  # LLM judges (can use multiple for ensemble voting)
RESPONSE_GENERATION_MODEL = "gpt-4o"  # For generating new responses if needed

# Evaluation Settings
USE_EXISTING_RESPONSES = True  # False if you need to generate new responses
MAX_SAMPLES = 100  # Limit samples for testing (None = use all data)
RANDOM_SEED = 42  # For reproducible results

# =============================================================================
# EXAMPLE CUSTOM METRICS
# =============================================================================

# Example: Real Estate Expertise Metric
REAL_ESTATE_EXPERTISE_PROMPT = """
You are an expert real estate professional evaluating AI responses for accuracy and expertise.

### Task
Evaluate whether the AI response demonstrates proper real estate knowledge and provides expert-level guidance.

### Materials
**User Question:**
```
{prompt}
```

**AI Response:**
```
{response}
```

**User Context:**
```
{user_context}
```

### Evaluation Criteria
1. **Technical Accuracy**: Correct use of real estate terminology and concepts
2. **Market Knowledge**: Demonstrates understanding of current market conditions
3. **Legal Compliance**: Follows fair housing and real estate regulations
4. **Professional Guidance**: Provides actionable, expert-level advice

### Scoring Scale (1-5)
- **5 (Expert Level)**: Demonstrates deep expertise, perfect accuracy, comprehensive guidance
- **4 (Professional)**: Good expertise, mostly accurate, solid guidance
- **3 (Competent)**: Adequate knowledge, some accuracy issues, basic guidance
- **2 (Novice)**: Limited expertise, several inaccuracies, poor guidance
- **1 (Inadequate)**: No expertise demonstrated, major errors, unhelpful

### Output Format
Return ONLY this JSON:
```json
{{
  "real_estate_expertise_score": <1-5 integer>,
  "explanation": "<brief explanation focusing on expertise level and accuracy>"
}}
```
"""

# Example: Customer Service Quality Metric
CUSTOMER_SERVICE_PROMPT = """
You are a customer service expert evaluating AI responses for service quality.

### Task
Evaluate the customer service quality of the AI response, focusing on tone, empathy, and helpfulness.

### Materials
**User Question:**
```
{prompt}
```

**AI Response:**
```
{response}
```

### Evaluation Criteria
1. **Professional Tone**: Friendly, respectful, and professional communication
2. **Empathy**: Shows understanding of user needs and concerns
3. **Responsiveness**: Directly addresses the user's question
4. **Proactive Help**: Goes beyond minimum requirements to be helpful

### Scoring
- **1 (Excellent Service)**: Exceptional tone, high empathy, very responsive and helpful
- **0 (Poor Service)**: Unprofessional, lacks empathy, unhelpful or dismissive

### Output Format
Return ONLY this JSON:
```json
{{
  "customer_service_score": <0 or 1>,
  "explanation": "<brief explanation of service quality assessment>"
}}
```
"""

# Add custom metrics to the registry
CUSTOM_METRICS_REGISTRY = {
    "Real_Estate_Expertise": {
        "prompt_template": REAL_ESTATE_EXPERTISE_PROMPT,
        "threshold": 3,  # Competent or better
        "description": "Evaluates real estate knowledge and expertise level",
        "score_type": "scale"
    },
    
    "Customer_Service": {
        "prompt_template": CUSTOMER_SERVICE_PROMPT,
        "threshold": 1,  # Must be excellent
        "description": "Assesses customer service quality and professionalism",
        "score_type": "binary"
    }
}

# Select which metrics to run
EXAMPLE_ACTIVE_METRICS = [
    "Accuracy",  # From built-in examples
    "Helpfulness",  # From built-in examples
    "Real_Estate_Expertise",  # Custom metric
    "Customer_Service",  # Custom metric
]

# =============================================================================
# SAMPLE DATA STRUCTURE
# =============================================================================

# Example of what your CSV data should look like:
SAMPLE_DATA_STRUCTURE = """
user_query,ai_response,user_profile
"What's the best neighborhood for a family with young kids in Seattle?","Based on your family's needs and budget, I'd recommend looking at Ballard or Queen Anne. Both neighborhoods offer excellent schools like Ballard High School and Queen Anne Elementary, numerous parks including Golden Gardens and Kerry Park, and family-friendly amenities. The average home price in Ballard is around $850K, while Queen Anne averages $950K. Both areas have low crime rates and good public transportation access.","Family with 2 young children, household income $150K, budget up to $900K, prefers walkable neighborhoods with good schools"
"How much house can I afford with my current income?","Based on your $100K annual income and $25K down payment, you can comfortably afford a home priced between $350K-$400K. This assumes a debt-to-income ratio of 28% and current mortgage rates around 6.5%. Your estimated monthly payment would be $2,100-$2,400 including taxes and insurance. I recommend getting pre-approved to confirm your exact buying power.","Single professional, $100K annual income, $25K saved for down payment, excellent credit score (750+), minimal existing debt"
"What are the current mortgage rates for first-time buyers?","Current mortgage rates for first-time buyers are averaging 6.75% for a 30-year fixed mortgage. However, you may qualify for special first-time buyer programs that offer rates as low as 6.25%. FHA loans are available at 6.5% with as little as 3.5% down. VA loans (if you qualify) offer rates around 6.0%. I recommend shopping with multiple lenders as rates can vary by 0.25-0.5% between providers.","First-time homebuyer, good credit score, researching mortgage options, no previous home ownership"
"""

print("✅ Example configuration loaded!")
print("📋 Copy the relevant sections to your main notebook configuration.")
print("🎯 Customize the experiment details and metrics for your specific use case.")