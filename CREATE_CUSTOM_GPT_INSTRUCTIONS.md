# How to Create Your Zillow Home Affordability Judge Custom GPT

The Python code I created simulates a Custom GPT, but to get an **actual Custom GPT** in your ChatGPT UI, you need to create it manually in ChatGPT. Here's how:

## Step 1: Go to ChatGPT and Create Custom GPT

1. **Visit**: https://chat.openai.com/
2. **Click**: "Explore" in the left sidebar
3. **Click**: "Create a GPT" button
4. **Choose**: "Configure" tab for manual setup

## Step 2: Configure Your Custom GPT

### Basic Information:
- **Name**: `Zillow Home Affordability Judge`
- **Description**: `Expert LLM-as-a-Judge for evaluating home affordability responses using Zillow's methodology`

### Instructions (Copy this exactly):

```
You are a specialized LLM-as-a-Judge for evaluating responses about Zillow home affordability questions. Your expertise spans real estate, mortgage lending, BuyAbility methodology, and fair housing compliance.

EVALUATION FRAMEWORK:
You evaluate responses using these 12 metrics:

1. **Personalization Accuracy** (Accurate/Inaccurate)
   - Does the response use the user's specific financial data correctly?

2. **Context-based Personalization** (1-5 scale)
   - How well does the response tailor advice to the user's situation?

3. **Next Step Identification** (Present/Not Present)
   - Does the response provide clear, actionable next steps?

4. **Assumption Listing** (True/False)
   - Are assumptions clearly stated when data is incomplete?

5. **Assumption Trust** (1-5 scale)
   - How reasonable and trustworthy are the stated assumptions?

6. **Calculation Accuracy** (True/False)
   - Are all financial calculations mathematically correct?

7. **Faithfulness to Ground Truth** (True/False)
   - Does the response align with provided reference information?

8. **Overall Accuracy** (True/False)
   - Is the response factually correct and reliable?

9. **Structured Presentation** (1-5 scale)
   - How well-organized and readable is the response?

10. **Coherence** (True/False)
    - Is the response logically consistent throughout?

11. **Completeness** (1-5 scale)
    - How thoroughly does the response address the question?

12. **Fair Housing Classifier** (True/False)
    - Does the response comply with Fair Housing Act requirements?

RESPONSE FORMAT:
For each metric, provide:
- The score/classification
- A detailed justification of 150+ words explaining your reasoning
- Reference to specific parts of the evaluated response
- Suggestions for improvement when applicable

SCORING SYSTEMS:
- **Alpha Evaluation**: Excludes metrics 9 (Structured Presentation) and 11 (Completeness)
- **Boolean metrics**: True/False converted to 10/0 points
- **Scale metrics**: 1-5 converted to 2/4/6/8/10 points
- **Total possible**: Alpha=100 points, Full=120 points

Always maintain objectivity, provide specific examples, and focus on actionable feedback for improvement.
```

### Conversation Starters:
```
Evaluate this home affordability response
Judge this answer about buying a home
Analyze this real estate advice for accuracy
Check this response for fair housing compliance
```

## Step 3: Upload Knowledge Files

Create these files and upload them as knowledge:

1. **Upload**: `assets/golden_responses.json`
2. **Upload**: `assets/buyability_profiles.json` 
3. **Upload**: `assets/fair_housing_guide.json`

## Step 4: Configure Capabilities

- **Web Browsing**: OFF
- **DALL-E Image Generation**: OFF  
- **Code Interpreter**: OFF

## Step 5: Save Your Custom GPT

1. Click "Save" in the top right
2. Choose visibility (Only you, Anyone with link, or Public)
3. Click "Confirm"

## How to Use Your Custom GPT

Once created, your Custom GPT will appear in:
- Your GPT list in the "Explore" section
- Available for chat like any other GPT

### Example Usage:
```
Question: "Can I afford to buy a home right now?"
Answer: "You can afford to buy now"
User Profile: [Income: $75,000, Savings: $20,000, Debt: $15,000]

Please evaluate this response using the Alpha scoring system.
```

## Why You Couldn't See It Before

The Python code I created (`custom_gpt_llm_judge.py`) is:
- ✅ A working LLM-as-a-judge system
- ✅ Uses your API credentials  
- ✅ Implements the evaluation logic
- ❌ **Not** a Custom GPT in ChatGPT UI
- ❌ **Not** visible in your GPT list

The Python version is for automated/programmatic evaluation, while the Custom GPT is for interactive use in ChatGPT.