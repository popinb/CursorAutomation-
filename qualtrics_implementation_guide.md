# Qualtrics Latency Study Implementation Guide

## Overview
This guide provides complete implementation instructions for a within-subjects study measuring the effect of response latency and streaming on perceived quality and acceptance.

## Study Design
- **Sample Size**: 100 participants
- **Design**: Within-subjects 2 (Modality) × 3 (Latency) factorial
- **Trials per participant**: 5 total
  - 3 Non-streaming trials (Fast 500ms, Medium 2000ms, Slow 6000ms)
  - 2 Streaming trials (Medium 2000ms, Slow 6000ms)
- **Questions per participant**: 9 total

## Implementation Steps

### 1. Create New Survey
1. Log into Qualtrics
2. Create new survey
3. Name it "Response Latency Study"

### 2. Set Up Survey Flow
1. Go to Survey Flow
2. Add Embedded Data variables (see `qualtrics_survey_flow.txt`)
3. Add Randomizer block with 10 conditions for balanced assignment
4. Add question blocks in order

### 3. Create Question Blocks

#### Block 1: Consent & Instructions
- **Question Type**: Text/Graphic
- **Content**: Use text from `qualtrics_question_texts.txt`
- **Follow-up**: Multiple Choice (Yes/No for consent)

#### Blocks 2-6: Trial Pages (1-5)
- **Question Type**: Text/Graphic
- **Content**: Use template from `qualtrics_question_texts.txt`
- **JavaScript**: Add code from `qualtrics_javascript_code.js`
- **Follow-up**: Matrix question (4 rows, 7-point scale)

#### Block 7: Post-Trial Perceptions
- **Question Type**: 5 Slider questions
- **Range**: 0-10 seconds
- **Labels**: Trial A through E with prompt text

#### Block 8: Wrap-up Questions
- **Question 1**: Slider for max acceptable wait (0-10s)
- **Question 2**: Multiple Choice for preference

#### Block 9: Thank You
- **Question Type**: Text/Graphic
- **Content**: Completion message

### 4. Add Custom CSS
1. Go to Look & Feel > Style
2. Click Custom CSS
3. Add CSS from `qualtrics_javascript_code.js`

### 5. Configure Randomization
1. In Survey Flow, add Randomizer element
2. Create 10 conditions to ensure balanced assignment
3. Each condition assigns different prompts to different latency/modality combinations

### 6. Set Up Display Logic
1. For each trial block, add display logic
2. Show block only if corresponding embedded data is set
3. Use piping to display correct prompt text

### 7. Test the Survey
1. Preview the survey
2. Test all conditions
3. Verify timing works correctly
4. Check data collection

## Data Collection

### Embedded Data Variables
- `participant_id`: Qualtrics Response ID
- `trial_X_prompt`: Prompt ID (1-5)
- `trial_X_modality`: "nonstream" or "stream"
- `trial_X_latency`: Latency in milliseconds (500, 2000, 6000)
- `trial_X_tft`: Time to first token (0 for non-stream, 300 for stream)

### Response Variables
- `trial_X_quality`: Quality rating (1-7)
- `trial_X_wait_feel`: Perceived wait length (1-7)
- `trial_X_wait_affect`: Whether wait affected quality judgment (1-7)
- `trial_X_accept`: Acceptance of wait time (1-7)
- `post_trial_X_perceived`: Perceived wait in seconds (0-10)
- `max_acceptable_wait`: Maximum acceptable wait (0-10)
- `preference`: Response style preference

## Analysis Plan

### Primary Analyses
1. **Quality Rating Analysis**
   ```r
   # Mixed-effects regression
   quality ~ latency_level * modality + (1 | participant_id) + (1 | prompt_id)
   ```

2. **Acceptance Analysis**
   ```r
   # Logistic mixed model
   accept ~ latency_sec * modality + (1 | participant) + (1 | prompt)
   # Compute ℓ50 (latency where Pr(accept)=0.5)
   ```

3. **Perceived Latency Analysis**
   ```r
   # Compare perceived vs actual latency
   bias = perceived_sec - actual_sec
   # Test if streaming reduces bias
   ```

### Secondary Analyses
- Mediation analysis (perceived wait → quality)
- Preference distribution
- Maximum acceptable wait thresholds

## Expected Outcomes

### Primary Results
- **Quality drop per latency step**: Expected -0.4 points Medium→Slow
- **Streaming benefit**: 
  - Δ in perceived latency (seconds) at Medium & Slow
  - Δ in ℓ50 (tolerance threshold)
- **Operational max acceptable wait**: Median of wrap-up slider
- **% users preferring streaming**: Distribution of preferences

### Power Analysis
- 100 participants × 5 trials = 500 observations
- Sufficient power for medium effects in within-subjects comparisons
- Expected effect sizes: Cohen's d = 0.3-0.5

## Troubleshooting

### Common Issues
1. **Timing not working**: Check JavaScript console for errors
2. **Piping not displaying**: Verify embedded data variable names
3. **Randomization not balanced**: Check Randomizer conditions
4. **CSS not applying**: Ensure custom CSS is properly saved

### Testing Checklist
- [ ] All 5 trials display correctly
- [ ] Timing works for both modalities
- [ ] Matrix questions appear after response
- [ ] Post-trial sliders work
- [ ] Data is collected properly
- [ ] Randomization is balanced

## File Structure
```
qualtrics_latency_study.html          # Complete standalone implementation
qualtrics_survey_flow.txt             # Survey flow configuration
qualtrics_javascript_code.js          # JavaScript for trial pages
qualtrics_question_texts.txt          # Question text content
qualtrics_implementation_guide.md     # This guide
```

## Next Steps
1. Import survey structure into Qualtrics
2. Add custom CSS and JavaScript
3. Test with pilot participants (n=5-10)
4. Refine based on pilot feedback
5. Launch full study (n=100)
6. Analyze results using provided analysis plan