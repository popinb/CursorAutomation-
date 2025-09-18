# Qualtrics Implementation Guide for Latency Perception Study

## Quick Start Checklist

1. **Create New Survey** in Qualtrics
2. **Set up Survey Flow** (see Section 2)
3. **Add Questions** (copy from HTML file)
4. **Test with Preview**
5. **Launch with N=100**

## 1. Question Setup Details

### Q1: Consent/Instructions
- Type: Text/Graphic
- No validation needed
- Force response: Yes

### Q2-Q6: Trial Questions (5 total)
- Type: Matrix Table
- Scale: 7-point Likert
- Statements: 4 rows (see HTML file)
- Force response: Yes
- Add HTML/JavaScript from the provided code
- Add Timing question (hidden) to track page timing

### Q7: Perception Sliders
- Type: Slider
- Configuration: Multi-slider with 5 sliders
- Range: 0-10 seconds
- Force response: Yes

### Q8: Max Acceptable Wait
- Type: Slider  
- Range: 0-10 seconds (with 0.1 decimal)
- Show value: Yes
- Force response: Yes

### Q9: Preference
- Type: Multiple Choice
- 3 options (see HTML file)
- Force response: Yes

## 2. Survey Flow Setup (CRITICAL)

### Step 1: Add Embedded Data Block
```
participant_id = ${e://Field/ResponseID}
trial1_prompt_id = 
trial1_modality = 
trial1_latency_ms = 
trial1_tft_ms = 
trial2_prompt_id = 
trial2_modality = 
trial2_latency_ms = 
trial2_tft_ms = 
[... continue for trials 3-5]
```

### Step 2: Add Randomizer for Condition Assignment
- Type: Randomizer
- Setting: Evenly Present Elements
- Add 10 elements (A-J) with embedded data assignments from `counterbalancing_setup.js`

### Step 3: Add Trial Order Randomizer
- Type: Randomizer  
- Setting: Randomly Present All
- Contains Q2-Q6 with piped embedded data

### Complete Flow Structure:
1. Embedded Data
2. Condition Assignment Randomizer
3. Q1: Consent
4. Trial Order Randomizer (Q2-Q6)
5. Q7: Perception Sliders
6. Q8: Max Wait
7. Q9: Preference
8. End of Survey

## 3. JavaScript Implementation Notes

### For Each Trial Question (Q2-Q6):
1. Click the question
2. Go to "JavaScript" in the question options
3. Copy the entire JavaScript code from the HTML file
4. The code handles:
   - Loading the correct prompt based on embedded data
   - Displaying response with correct timing/modality
   - Showing/hiding the Next button appropriately

### Key Variables Used:
- `${e://Field/prompt_id}` - Which prompt to show
- `${e://Field/modality}` - "stream" or "nonstream"  
- `${e://Field/latency_ms}` - Total latency in milliseconds
- `${e://Field/tft_ms}` - Time to first token (streaming only)

## 4. Testing Checklist

Before launching:
- [ ] Preview survey and check all 5 trials load
- [ ] Verify streaming animation works smoothly
- [ ] Check non-streaming appears all at once
- [ ] Confirm timing feels correct (0.5s, 2s, 6s)
- [ ] Test on mobile device
- [ ] Verify data exports with all embedded fields
- [ ] Check randomization is working (test 3-5 times)

## 5. Data Export Configuration

Essential fields to include:
- All embedded data fields (trial1_prompt_id, etc.)
- Q2_1 through Q6_4 (matrix responses)
- Q7_1 through Q7_5 (perception sliders)
- Q8 (max acceptable wait)
- Q9 (preference)
- Timing data for Q2-Q6
- ResponseID
- Start/End Date

## 6. Quick Fixes for Common Issues

### "JavaScript not working"
- Make sure you're adding it to the JavaScript section, not HTML
- Check for any console errors in browser dev tools

### "Responses appear too fast/slow"
- Verify embedded data is being set correctly in Survey Flow
- Check that latency_ms values are 500, 2000, or 6000

### "Streaming looks choppy"
- Adjust the character interval calculation in the JavaScript
- Consider reducing interval from 50ms to 30ms

### "Next button appears too early"
- Increase the delay in `showNextButton()` timeout
- Currently set to latency + 500ms

## 7. Analysis Preparation

After data collection:
1. Export data as CSV
2. Run the provided R analysis code
3. Key metrics to report:
   - Quality ratings by condition
   - ℓ50 acceptance thresholds  
   - Streaming perception benefit
   - User preferences

## 8. Attention Check

One prompt includes: "Please select 'Strongly Agree' for statement 2"
- This is embedded in the answer text
- Filter out participants who don't follow this instruction

## 9. Mobile Optimization

The CSS includes mobile-responsive styles. Additional considerations:
- Test that sliders work well on touch devices
- Ensure matrix questions display properly
- Consider adding mobile detection embedded data

## 10. Pilot Testing Recommendations

Before full launch:
1. Run with 5-10 participants
2. Check data quality and completeness
3. Verify timing calculations in analysis
4. Get feedback on instruction clarity
5. Adjust if needed before N=100 launch