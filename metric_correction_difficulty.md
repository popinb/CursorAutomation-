# Difficulty Analysis: Correcting All 12 Metrics for Accuracy

## 📊 **Overall Assessment: MODERATE to HIGH Difficulty**

**Time Estimate: 2-4 weeks for a skilled engineer**
**Complexity: Some easy fixes, but several require significant domain expertise**

---

## 🟢 **EASY to Fix (1-2 days each)**

### 1. **Calculation Accuracy** 
**Current:** Hardcoded `"True"`
**Fix Difficulty:** ⭐ VERY EASY
```python
# Current (broken):
calc_acc = "True"

# Fix (requires math validation):
def validate_calculations(candidate: str, profile: dict) -> bool:
    # Extract numbers, validate DTI ratios, affordability calculations
    extracted_income = extract_number_after_keyword(candidate, "income")
    extracted_debts = extract_number_after_keyword(candidate, "debt")
    if extracted_income and extracted_debts:
        calculated_dti = (extracted_debts * 12) / extracted_income
        # Validate against mentioned DTI ratios
    return math_checks_pass
```
**Effort:** 1-2 days of regex/NLP work

### 2. **Fair Housing Classifier**
**Current:** Simple regex for protected classes
**Fix Difficulty:** ⭐ EASY  
```python
# Current (too simple):
hits_prot = _has_kw(c, _PROT)

# Fix (add more patterns, edge cases):
def detect_fair_housing_violations(text: str) -> bool:
    # Add context-aware detection
    # Check for implicit bias patterns
    # Handle euphemisms and coded language
```
**Effort:** 1-2 days for better pattern matching

### 3. **Structured Presentation**
**Current:** Arbitrary minimum score of 2
**Fix Difficulty:** ⭐ VERY EASY
```python
# Current (inflated):
structured = str(min(5, max(1, 2 + struc_pts)))

# Fix (remove artificial floor):
structured = str(min(5, max(1, 1 + struc_pts)))
```
**Effort:** 30 minutes

---

## 🟡 **MEDIUM Difficulty (3-5 days each)**

### 4. **Personalization Accuracy**
**Current:** Just checks for keyword presence
**Fix Difficulty:** ⭐⭐ MEDIUM
```python
# Current (too lenient):
personalization = "Accurate" if ((f["has_income"] and f["has_debts"]) or (f["has_down"] and f["has_credit"])) else "Inaccurate"

# Fix (semantic validation needed):
def validate_personalization(candidate: str, profile: dict) -> str:
    # Extract actual numbers from candidate
    # Compare with profile values
    # Check if calculations use correct profile data
    # Validate ranges/tiers match
```
**Challenges:**
- Need robust number extraction from text
- Handle various formatting ("$90k", "90000", "ninety thousand")
- Validate credit score ranges match profile ranges

### 5. **Context-based Personalization** 
**Current:** 80% keyword coverage = perfect score
**Fix Difficulty:** ⭐⭐ MEDIUM
```python
# Fix (stricter thresholds + relevance weighting):
def score_context_personalization(features: dict, question_type: str) -> str:
    # Weight features by relevance to specific question
    # Require 90%+ for perfect score
    # Check quality of feature usage, not just presence
```
**Challenges:**
- Define relevance weights for different question types
- Distinguish between meaningful usage vs. casual mention

### 6. **Assumption Listing**
**Current:** Basic keyword detection
**Fix Difficulty:** ⭐⭐ MEDIUM
```python
# Fix (semantic understanding needed):
def detect_assumptions(candidate: str) -> bool:
    # Parse for explicit assumption statements
    # Identify implicit assumptions in calculations
    # Distinguish assumptions from facts
```
**Challenges:**
- NLP complexity to identify implicit assumptions
- Domain knowledge of what should be assumed vs. stated

---

## 🔴 **HARD to Fix (1-2 weeks each)**

### 7. **Faithfulness to Ground Truth**
**Current:** Checks if DTI/PITI concepts mentioned
**Fix Difficulty:** ⭐⭐⭐ HARD
```python
# Fix (requires semantic matching against ground truth):
def validate_faithfulness(candidate: str, golden_answers: dict, question: str) -> bool:
    # Match question to appropriate golden answer
    # Extract key facts from both candidate and golden
    # Semantic similarity comparison
    # Fact verification against golden truth
```
**Challenges:**
- **Requires LLM or advanced NLP** for semantic comparison
- Complex logic to map questions to golden answers
- Need structured fact extraction from both sources

### 8. **Overall Accuracy**
**Current:** Simple composite of other metrics
**Fix Difficulty:** ⭐⭐⭐ HARD
```python
# Fix (holistic evaluation needed):
def assess_overall_accuracy(candidate: str, profile: dict, golden: dict) -> bool:
    # Comprehensive correctness assessment
    # Cross-validate all components
    # Check logical consistency
    # Verify end-to-end reasoning
```
**Challenges:**
- **Most complex metric** - depends on all others being fixed first
- Requires domain expertise in mortgage/finance logic
- Subjective judgment calls

### 9. **Assumption Trust**
**Current:** Simple counting system
**Fix Difficulty:** ⭐⭐⭐ HARD  
```python
# Fix (requires domain expertise):
def assess_assumption_trustworthiness(candidate: str, context: dict) -> int:
    # Evaluate reasonableness of each assumption
    # Check against industry standards
    # Assess transparency and justification
```
**Challenges:**
- **Requires deep domain knowledge** of mortgage assumptions
- Subjective assessment of "trustworthiness"
- Need database of reasonable assumption ranges

---

## 🟠 **MEDIUM-HARD (1 week each)**

### 10. **Next Step Identification**
**Current:** Simple action keyword detection  
**Fix Difficulty:** ⭐⭐⭐ MEDIUM-HARD
```python
# Fix (context-aware action detection):
def validate_next_steps(candidate: str, question_type: str, user_stage: str) -> str:
    # Identify appropriate next steps for user's situation
    # Validate against business process flow
    # Check for missing critical steps
```
**Challenges:**
- Need business process knowledge
- Context-dependent evaluation

### 11. **Coherence**
**Current:** Checks for causal language keywords
**Fix Difficulty:** ⭐⭐⭐ MEDIUM-HARD
```python
# Fix (logical flow analysis):
def assess_coherence(candidate: str) -> bool:
    # Analyze logical flow of arguments
    # Check for contradictions
    # Evaluate reasoning chains
```
**Challenges:**
- **May require LLM assistance** for complex reasoning evaluation
- Subjective nature of "coherence"

### 12. **Completeness**
**Current:** Same as context-based personalization
**Fix Difficulty:** ⭐⭐ MEDIUM (if different from #5)
- Can reuse improved context-based logic
- Need question-specific completeness criteria

---

## 📋 **Implementation Strategy by Priority**

### **Phase 1: Quick Wins (1 week)**
1. ✅ **Structured Presentation** (30 min)
2. ✅ **Calculation Accuracy** (2 days) 
3. ✅ **Fair Housing** (2 days)
4. ✅ **Adjust thresholds** (1 day)

### **Phase 2: Medium Complexity (2 weeks)** 
5. ✅ **Personalization Accuracy** (5 days)
6. ✅ **Context-based Personalization** (5 days)
7. ✅ **Assumption Listing** (4 days)

### **Phase 3: Hard Problems (2-4 weeks)**
8. ✅ **Next Step Identification** (1 week)
9. ✅ **Coherence** (1 week)  
10. ✅ **Faithfulness to Ground Truth** (2 weeks)
11. ✅ **Assumption Trust** (1 week)
12. ✅ **Overall Accuracy** (1 week - after others fixed)

---

## 🛠 **Required Skills & Resources**

### **Technical Skills Needed:**
- **Regex/NLP expertise** (for number extraction, pattern matching)
- **Domain knowledge** (mortgage/finance business rules)
- **LLM integration** (for complex semantic tasks)
- **Python development** (testing, refactoring)

### **Potential Roadblocks:**
1. **Ground truth data quality** - May need to improve golden answers
2. **Subjective metrics** - Some require business stakeholder input
3. **Performance trade-offs** - More accurate = slower/more expensive
4. **Edge cases** - Real-world text variations are complex

### **Alternative Approach: Hybrid LLM Solution**
Instead of pure rule-based fixes, consider:
```python
def llm_assisted_metric(metric_name: str, candidate: str, context: dict) -> str:
    # Use LLM with strict prompts for complex metrics
    # Combine with rule-based validation
    # More accurate but less deterministic
```

---

## 🎯 **Final Assessment**

**Total Effort: 2-4 weeks** for complete accuracy overhaul

**Complexity Distribution:**
- 🟢 **25% Easy** (3 metrics)
- 🟡 **33% Medium** (4 metrics) 
- 🔴 **42% Hard** (5 metrics)

**Recommended Approach:**
1. **Start with Phase 1** for immediate 50% improvement
2. **Consider LLM assistance** for the hardest metrics
3. **Incremental deployment** with A/B testing
4. **Domain expert consultation** for business rule validation

The good news: **Even fixing just the easy metrics would dramatically improve accuracy** and lower the inflated scores!