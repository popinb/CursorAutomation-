# Complete Qualtrics Setup Guide
## Response Latency Perception Study

This guide provides step-by-step instructions to implement the complete study in Qualtrics.

## 📋 Overview
- **Study Type:** Within-subjects latency perception study
- **Duration:** 5-7 minutes per participant  
- **Sample Size:** N=100 participants
- **Questions per participant:** 9 questions total
- **Conditions:** 2 (Streaming vs Non-streaming) × 3 (Fast/Medium/Slow latency)

---

## 🚀 Quick Setup Checklist

- [ ] Create new Qualtrics survey
- [ ] Set up Survey Flow with embedded data and randomization
- [ ] Create 5 question blocks (consent + 5 trials + post-trial + wrap-up)
- [ ] Add JavaScript code to trial questions
- [ ] Configure matrix tables and sliders
- [ ] Test all conditions
- [ ] Launch survey

---

## 📝 Step-by-Step Implementation

### Step 1: Create New Survey
1. Log into Qualtrics
2. Create new survey: "Response Latency Perception Study"
3. Choose "Get started from scratch"

### Step 2: Set Up Survey Flow

**Go to Survey Flow and set up the following elements in order:**

#### A. Embedded Data Element
Add these embedded data fields at the top of Survey Flow:

```
participant_id = ${q://QID_SYSTEM/ResponseID}
current_trial = 1
trial_1_prompt_id = 
trial_1_modality = 
trial_1_latency_ms = 
trial_1_tft_ms = 300
trial_1_actual_latency_ms = 
trial_2_prompt_id = 
trial_2_modality = 
trial_2_latency_ms = 
trial_2_tft_ms = 300
trial_2_actual_latency_ms = 
trial_3_prompt_id = 
trial_3_modality = 
trial_3_latency_ms = 
trial_3_tft_ms = 300
trial_3_actual_latency_ms = 
trial_4_prompt_id = 
trial_4_modality = 
trial_4_latency_ms = 
trial_4_tft_ms = 300
trial_4_actual_latency_ms = 
trial_5_prompt_id = 
trial_5_modality = 
trial_5_latency_ms = 
trial_5_tft_ms = 300
trial_5_actual_latency_ms = 
```

#### B. Randomizer Element (Condition Assignment)
Create a Randomizer with 5 equally weighted groups (20% each):

**Group 1:**
```
trial_1_prompt_id = 1, trial_1_modality = nonstream, trial_1_latency_ms = 500
trial_2_prompt_id = 2, trial_2_modality = nonstream, trial_2_latency_ms = 2000  
trial_3_prompt_id = 3, trial_3_modality = nonstream, trial_3_latency_ms = 6000
trial_4_prompt_id = 4, trial_4_modality = stream, trial_4_latency_ms = 2000
trial_5_prompt_id = 5, trial_5_modality = stream, trial_5_latency_ms = 6000
```

**Group 2:**
```
trial_1_prompt_id = 2, trial_1_modality = nonstream, trial_1_latency_ms = 500
trial_2_prompt_id = 3, trial_2_modality = nonstream, trial_2_latency_ms = 2000
trial_3_prompt_id = 4, trial_3_modality = nonstream, trial_3_latency_ms = 6000
trial_4_prompt_id = 5, trial_4_modality = stream, trial_4_latency_ms = 2000
trial_5_prompt_id = 1, trial_5_modality = stream, trial_5_latency_ms = 6000
```

**Group 3:**
```
trial_1_prompt_id = 3, trial_1_modality = nonstream, trial_1_latency_ms = 500
trial_2_prompt_id = 4, trial_2_modality = nonstream, trial_2_latency_ms = 2000
trial_3_prompt_id = 5, trial_3_modality = nonstream, trial_3_latency_ms = 6000
trial_4_prompt_id = 1, trial_4_modality = stream, trial_4_latency_ms = 2000
trial_5_prompt_id = 2, trial_5_modality = stream, trial_5_latency_ms = 6000
```

**Group 4:**
```
trial_1_prompt_id = 4, trial_1_modality = nonstream, trial_1_latency_ms = 500
trial_2_prompt_id = 5, trial_2_modality = nonstream, trial_2_latency_ms = 2000
trial_3_prompt_id = 1, trial_3_modality = nonstream, trial_3_latency_ms = 6000
trial_4_prompt_id = 2, trial_4_modality = stream, trial_4_latency_ms = 2000
trial_5_prompt_id = 3, trial_5_modality = stream, trial_5_latency_ms = 6000
```

**Group 5:**
```
trial_1_prompt_id = 5, trial_1_modality = nonstream, trial_1_latency_ms = 500
trial_2_prompt_id = 1, trial_2_modality = nonstream, trial_2_latency_ms = 2000
trial_3_prompt_id = 2, trial_3_modality = nonstream, trial_3_latency_ms = 6000
trial_4_prompt_id = 3, trial_4_modality = stream, trial_4_latency_ms = 2000
trial_5_prompt_id = 4, trial_5_modality = stream, trial_5_latency_ms = 6000
```

#### C. Block Elements (in order)
1. **Block: Consent & Instructions**
2. **Embedded Data:** `current_trial = 1`
3. **Block: Trial 1**
4. **Embedded Data:** `current_trial = 2`  
5. **Block: Trial 2**
6. **Embedded Data:** `current_trial = 3`
7. **Block: Trial 3**
8. **Embedded Data:** `current_trial = 4`
9. **Block: Trial 4**
10. **Embedded Data:** `current_trial = 5`
11. **Block: Trial 5**
12. **Randomizer:** (Randomize Trial 1-5 blocks)
13. **Block: Post-Trial Sliders**
14. **Block: Wrap-Up Questions**

### Step 3: Create Questions

#### Question 1: Consent & Instructions (Text/Graphic)
- Copy content from `consent_instructions.html`
- Set as required question
- No response options needed (just "Next" button)

#### Questions 2-6: Trial Questions (Matrix Table)
Create 5 identical matrix table questions with these settings:

**Question Type:** Matrix Table
**Scale:** 7-point Likert scale

**Statements (rows):**
1. Overall quality of the response
2. The wait felt... (1=very short, 7=very long)  
3. The wait negatively affected my quality judgment
4. I would accept this wait time regularly

**Scale Labels:**
- 1 = Strongly Disagree / Very Short / Very Poor
- 2 = Disagree / Short / Poor
- 3 = Somewhat Disagree / Somewhat Short / Below Average  
- 4 = Neither / Neutral / Average
- 5 = Somewhat Agree / Somewhat Long / Above Average
- 6 = Agree / Long / Good
- 7 = Strongly Agree / Very Long / Excellent

**Question Text:** 
```html
<div id="trial-content">
    <p><strong>Trial ${e://Field/current_trial} of 5</strong></p>
    <p>Please wait for the response to appear, then rate it using the scales below.</p>
</div>
```

**JavaScript:** Copy the entire contents of `trial_javascript.js` into each trial question's JavaScript section.

**Settings:**
- Force Response: Yes
- Request Response: Yes  
- Add Timing: Yes

#### Question 7: Post-Trial Sliders (Slider)
**Question Type:** Slider (Multi-slider)
**Number of sliders:** 5

**Question Text:** Copy from `post_trial_sliders.html`

**Slider Settings:**
- Range: 0 to 10 seconds
- Step: 1 (whole numbers)
- Show value: Yes
- Labels: "Trial A", "Trial B", "Trial C", "Trial D", "Trial E"

**Settings:**
- Force Response: Yes

#### Question 8: Max Acceptable Wait (Slider)  
**Question Type:** Slider (Single slider)

**Question Text:**
```html
<div style="font-family: Arial, sans-serif; line-height: 1.6;">
    <h3>Maximum Acceptable Wait Time</h3>
    <p>Based on your experience with AI responses in this study, what's the <strong>longest wait time</strong> you would regularly accept for similar tasks?</p>
    <p><em>Think about your tolerance for waiting when asking questions that require thoughtful, detailed responses.</em></p>
</div>
```

**Slider Settings:**
- Range: 0 to 10 seconds
- Step: 0.5 seconds  
- Show value: Yes
- Label: "Maximum acceptable wait time (seconds)"

#### Question 9: Streaming Preference (Multiple Choice)
**Question Type:** Multiple Choice (Single Answer)

**Question Text:**
```html
<div style="font-family: Arial, sans-serif; line-height: 1.6;">
    <h3>Response Style Preference</h3>
    <p>During this study, you experienced two types of responses:</p>
    <ul>
        <li><strong>Non-streaming:</strong> The complete response appeared all at once after a wait</li>
        <li><strong>Streaming:</strong> The response appeared gradually, word by word</li>
    </ul>
    <p>For similar AI tasks in the future, which would you prefer?</p>
</div>
```

**Answer Choices:**
1. I prefer streaming responses (text appears gradually)
2. I prefer non-streaming responses (complete text appears at once)  
3. No preference - both are equally acceptable
4. It depends on the situation

### Step 4: Testing

#### Test Each Condition:
1. **Preview Survey** multiple times to see different randomization groups
2. **Check JavaScript timing** - verify responses appear at correct intervals
3. **Test streaming behavior** - ensure text streams smoothly
4. **Verify data collection** - check that embedded data is captured correctly
5. **Mobile testing** - test on mobile devices

#### Key Testing Points:
- ✅ Responses appear after correct delay (0.5s, 2s, 6s)
- ✅ Streaming shows first token at ~0.3s, then streams to completion
- ✅ Non-streaming shows blank then full response
- ✅ Matrix appears only after response is complete
- ✅ All embedded data fields populate correctly
- ✅ Trial randomization works properly

### Step 5: Launch Settings

**Survey Options:**
- Anonymize Response: Yes
- Prevent Ballot Box Stuffing: Yes  
- Save and Continue: No (keep sessions short)

**Response Options:**
- Incomplete Survey Response: Record after 20 minutes
- Survey Protection: Password protect if needed

---

## 🔧 Technical Implementation Notes

### JavaScript Behavior Details

**Non-streaming mode:**
- Shows blank response container
- After `latency_ms`, full response appears instantly
- Matrix table becomes visible

**Streaming mode:**  
- Shows blank response container
- After 300ms (`tft_ms`), starts streaming characters
- Characters appear at calculated intervals to reach total `latency_ms`
- Blinking cursor during streaming
- Matrix table becomes visible when complete

### Data Collection

The survey automatically collects:
- **Trial-level:** Quality ratings, wait perception, acceptance, impact on quality
- **Post-trial:** Perceived wait times in seconds for each trial
- **Individual:** Max acceptable wait, streaming preference
- **Metadata:** Actual display latencies, page timing, response times

### Randomization Balance

Across 100 participants:
- Each prompt appears 20 times in each condition
- 60 non-streaming trials (20 fast, 20 medium, 20 slow)
- 40 streaming trials (20 medium, 20 slow)  
- Perfect counterbalancing across participants

---

## 📊 Expected Data Output

### Key Variables:
- `quality_rating_1` through `quality_rating_5` (1-7 scale)
- `wait_felt_1` through `wait_felt_5` (1-7 scale)  
- `wait_hurt_quality_1` through `wait_hurt_quality_5` (1-7 scale)
- `would_accept_1` through `would_accept_5` (1-7 scale)
- `perceived_wait_A` through `perceived_wait_E` (0-10 seconds)
- `max_acceptable_wait` (0-10 seconds)
- `streaming_preference` (1-4 categorical)

### Analysis-Ready Format:
Each row = 1 participant with 5 trial observations
- 100 rows × 5 trials = 500 total trial observations
- Perfect for mixed-effects modeling
- Built-in counterbalancing across conditions

---

## 🚨 Troubleshooting

### Common Issues:

**JavaScript not working:**
- Ensure JavaScript is enabled in question settings
- Check browser console for errors
- Verify embedded data fields are correctly named

**Timing issues:**
- Test across different browsers and devices
- Check internet connection speed effects
- Verify setTimeout functions work correctly

**Randomization problems:**
- Double-check Survey Flow randomizer weights
- Ensure all embedded data fields are set
- Test multiple preview sessions

**Matrix table issues:**
- Verify all scale points are labeled correctly
- Check mobile responsiveness
- Ensure force response is enabled

### Support Resources:
- Qualtrics Support Documentation
- JavaScript debugging tools
- Survey preview and test modes

---

## ✅ Pre-Launch Checklist

- [ ] All 9 questions created and configured
- [ ] JavaScript added to all 5 trial questions  
- [ ] Survey Flow properly configured with randomization
- [ ] All embedded data fields defined
- [ ] Tested across multiple browsers
- [ ] Mobile compatibility verified
- [ ] Data export format confirmed
- [ ] IRB approval obtained (if required)
- [ ] Participant recruitment plan ready

**Estimated Setup Time:** 2-3 hours for complete implementation and testing

---

*This implementation provides a robust, balanced study design that will generate high-quality data on latency perception and streaming effects with minimal participant burden.*