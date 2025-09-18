# Complete Qualtrics Implementation: Response Latency Perception Study

This repository contains a complete, ready-to-deploy implementation of a response latency perception study for Qualtrics. The study measures how response timing affects perceived quality and whether streaming reduces perceived latency.

## 📁 Files Overview

### Core Implementation Files
- **`QUALTRICS_SETUP_GUIDE.md`** - Complete step-by-step setup instructions
- **`enhanced_trial_javascript.js`** - Production-ready JavaScript for trial questions
- **`trial_javascript.js`** - Basic JavaScript implementation
- **`consent_instructions.html`** - Consent and instructions page content
- **`trial_question_template.html`** - Template for trial questions
- **`post_trial_sliders.html`** - Post-trial perception sliders
- **`wrap_up_questions.html`** - Final wrap-up questions

### Configuration Files
- **`qualtrics_survey_flow.json`** - Survey Flow configuration
- **`matrix_configuration.json`** - Matrix table setup details
- **`randomization_groups.json`** - Balanced randomization scheme

### Analysis
- **`data_analysis_plan.R`** - Complete R analysis script

## 🚀 Quick Start

1. **Read the Setup Guide**: Start with `QUALTRICS_SETUP_GUIDE.md` for detailed instructions
2. **Create Survey**: Set up a new Qualtrics survey
3. **Configure Survey Flow**: Use the randomization groups and embedded data structure
4. **Add Questions**: Copy HTML content and JavaScript code for each question type
5. **Test**: Preview and test all conditions before launching
6. **Analyze**: Use the R script for comprehensive data analysis

## 📊 Study Design

- **Participants**: N=100 (5-7 minutes each)
- **Design**: Within-subjects 2×3 design
  - **Modality**: Non-streaming vs Streaming
  - **Latency**: Fast (0.5s), Medium (2s), Slow (6s)
- **Questions**: 9 questions per participant total
- **Trials**: 5 trials per participant (500 total observations)

## 🎯 Key Features

### Technical Implementation
- ✅ **JavaScript-powered timing control** - Precise latency manipulation
- ✅ **Streaming simulation** - Realistic word-by-word text appearance
- ✅ **Mobile-optimized** - Responsive design for all devices
- ✅ **Balanced randomization** - Every prompt in every condition
- ✅ **Attention checks** - Built-in data quality measures
- ✅ **Error handling** - Robust fallbacks for technical issues

### Data Collection
- ✅ **Quality ratings** - 7-point Likert scales
- ✅ **Wait perception** - Both Likert and slider measures
- ✅ **Acceptance thresholds** - Binary acceptance decisions
- ✅ **Individual differences** - Max acceptable wait, streaming preference
- ✅ **Timing metadata** - Actual display times for validation

### Analysis Ready
- ✅ **Mixed-effects models** - Account for participant and item effects
- ✅ **L50 calculations** - Latency thresholds for 50% acceptance
- ✅ **Mediation analysis** - Test if perceived wait mediates latency→quality
- ✅ **Effect size calculations** - Cohen's d and odds ratios
- ✅ **Visualizations** - Publication-ready plots

## 📈 Expected Outcomes

The study will provide:
1. **Latency tolerance curves** - How acceptance drops with increasing wait time
2. **Streaming benefits** - Quantified reduction in perceived latency
3. **Quality impact** - How wait time affects perceived response quality  
4. **Individual differences** - Variation in latency tolerance
5. **Operational guidelines** - Recommended wait times for AI systems

## 🔧 Technical Requirements

### Qualtrics Features Used
- Survey Flow with embedded data and randomization
- Matrix table questions with custom JavaScript
- Multi-slider questions
- Multiple choice and single slider questions
- Timing data collection

### Browser Compatibility
- Modern browsers with JavaScript enabled
- Mobile-responsive design
- Fallback handling for technical issues

## 📋 Implementation Checklist

- [ ] Create Qualtrics survey
- [ ] Set up Survey Flow with 5 randomization groups
- [ ] Add consent/instructions page
- [ ] Create 5 trial questions with JavaScript
- [ ] Add post-trial slider question
- [ ] Add wrap-up questions
- [ ] Test all conditions and timing
- [ ] Verify data export format
- [ ] Launch survey
- [ ] Monitor data quality
- [ ] Run analysis with provided R script

## 🎯 Study Validation

The implementation includes multiple validation features:
- **Timing accuracy**: JavaScript logs actual vs intended latencies
- **Attention checks**: Subtle instructions embedded in responses
- **Data quality**: Multiple measures of the same constructs
- **Balance checking**: Randomization verification in analysis
- **Mobile testing**: Cross-device compatibility

## 📞 Support

For implementation questions:
1. Check the detailed setup guide first
2. Test in Qualtrics preview mode
3. Verify JavaScript console for errors
4. Use browser developer tools for debugging

## 🏆 Expected Impact

This study design enables robust conclusions about:
- **User experience optimization** for AI systems
- **Interface design decisions** (streaming vs non-streaming)
- **Performance benchmarks** for response latency
- **Individual difference factors** in latency tolerance

The complete implementation provides everything needed to run a publication-quality study on response latency perception with minimal setup time and maximum scientific rigor.

---

*Total estimated setup time: 2-3 hours*  
*Data collection time: ~10 hours for 100 participants*  
*Analysis time: 1-2 hours with provided R script*