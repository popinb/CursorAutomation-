/* 
ENHANCED QUALTRICS JAVASCRIPT FOR TRIAL PAGES
This version includes better error handling, mobile optimization, and debugging
Add this to each trial question's JavaScript section
*/

Qualtrics.SurveyEngine.addOnload(function() {
    // Get current question element and ID
    var qid = this.questionId;
    var question = jQuery("#" + qid);
    
    // Get embedded data for current trial with error handling
    var trialNum = "${e://Field/current_trial}" || "1";
    var promptId = "${e://Field/trial_" + trialNum + "_prompt_id}" || "1";
    var modality = "${e://Field/trial_" + trialNum + "_modality}" || "nonstream";
    var latencyMs = parseInt("${e://Field/trial_" + trialNum + "_latency_ms}") || 2000;
    var tftMs = parseInt("${e://Field/trial_" + trialNum + "_tft_ms}") || 300;
    
    // Debug logging (remove for production)
    console.log("Trial " + trialNum + ": Prompt " + promptId + ", " + modality + ", " + latencyMs + "ms");
    
    // Prompt and answer content
    var prompts = {
        "1": {
            "question": "Define 'compound interest' in one sentence.",
            "answer": "Compound interest is the interest earned on both the original principal amount and previously earned interest, causing money to grow exponentially over time. This powerful financial concept means that the longer you save or invest, the more dramatic your returns become, as each period's interest becomes part of the base for calculating the next period's interest."
        },
        "2": {
            "question": "Give 3 dinner ideas using only rice, eggs, and spinach.",
            "answer": "Here are three delicious dinner options: (1) Spinach Fried Rice - stir-fry cooked rice with scrambled eggs and fresh spinach, seasoned with soy sauce and garlic; (2) Rice Bowl with Poached Egg - serve steamed rice topped with sautéed spinach and a perfectly poached egg; (3) Spanish-Style Rice - cook rice with beaten eggs stirred in at the end, mixed with wilted spinach and seasoned with paprika and salt."
        },
        "3": {
            "question": "Fix this one-line Python bug: pritn('hi')",
            "answer": "The bug is a simple typo in the function name. Change 'pritn' to 'print' to fix it: print('hi'). This is one of the most common beginner mistakes in Python programming. The correct function name is 'print' with all lowercase letters, and it's used to display output to the console or terminal."
        },
        "4": {
            "question": "Summarize this blurb in one sentence: 'Machine learning algorithms require large datasets to train effectively. However, collecting quality data can be expensive and time-consuming. Synthetic data generation offers a promising solution to this challenge.'",
            "answer": "While machine learning needs large, high-quality datasets that are costly and slow to collect, synthetic data generation provides a promising alternative solution to overcome these data collection challenges. This approach allows researchers and companies to create artificial training data that mimics real-world patterns without the traditional barriers of expense and time."
        },
        "5": {
            "question": "Recommend a weekend activity for a rainy day.",
            "answer": "Try hosting a cooking challenge where you create a new dish using only ingredients you already have at home. This activity combines creativity, skill-building, and practical value while keeping you entertained indoors. You can make it more engaging by setting themes like 'fusion cuisine' or 'comfort food makeover,' and even document your creations to share with friends or save for future reference."
        }
    };
    
    var currentPrompt = prompts[promptId] || prompts["1"];
    var responseText = currentPrompt.answer;
    
    // Store timing data
    var startTime = Date.now();
    var actualDisplayTime = null;
    
    // Create response container with loading state
    var responseContainer = jQuery('<div id="response-container" style="margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 8px; min-height: 120px; background-color: #fafafa; box-shadow: 0 2px 4px rgba(0,0,0,0.1);"><div id="response-text" style="font-family: Georgia, serif; line-height: 1.6; font-size: 16px; color: #333;"></div><div id="loading-indicator" style="color: #666; font-style: italic;">Thinking...</div></div>');
    
    // Create prompt display with better styling
    var promptHtml = '<div style="margin-bottom: 25px; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);"><strong style="font-size: 18px;">Question:</strong><div style="margin-top: 10px; font-size: 16px; line-height: 1.5;">' + currentPrompt.question + '</div></div>';
    
    // Insert prompt and response container
    question.find('.QuestionText').after(promptHtml + responseContainer[0].outerHTML);
    
    // Hide matrix initially and add attention check
    question.find('table').hide();
    var attentionCheckAdded = false;
    
    // Enhanced non-streaming response function
    function showNonStreamResponse() {
        setTimeout(function() {
            actualDisplayTime = Date.now() - startTime;
            jQuery('#loading-indicator').fadeOut(200, function() {
                jQuery('#response-text').html(responseText).hide().fadeIn(400);
                addAttentionCheck();
                showMatrix();
            });
        }, latencyMs);
    }
    
    // Enhanced streaming response function
    function showStreamResponse() {
        var words = responseText.split(' ');
        var totalWords = words.length;
        var remainingTime = latencyMs - tftMs;
        var intervalTime = remainingTime / totalWords;
        
        // Ensure minimum interval for readability
        intervalTime = Math.max(intervalTime, 50);
        
        setTimeout(function() {
            jQuery('#loading-indicator').fadeOut(200);
            
            var currentText = '';
            var wordIndex = 0;
            
            function addNextWord() {
                if (wordIndex < totalWords) {
                    currentText += (wordIndex > 0 ? ' ' : '') + words[wordIndex];
                    jQuery('#response-text').html(currentText + '<span class="typing-cursor">|</span>');
                    wordIndex++;
                    setTimeout(addNextWord, intervalTime);
                } else {
                    // Remove cursor and finalize
                    actualDisplayTime = Date.now() - startTime;
                    jQuery('#response-text').html(currentText);
                    addAttentionCheck();
                    showMatrix();
                }
            }
            
            addNextWord();
        }, tftMs);
    }
    
    // Add subtle attention check
    function addAttentionCheck() {
        if (!attentionCheckAdded && Math.random() < 0.2) { // 20% chance
            var attentionText = responseText + ' <span style="color: #d32f2f; font-weight: bold;">[Please select "Somewhat Agree" for the wait perception question to show you are paying attention.]</span>';
            jQuery('#response-text').html(attentionText);
            attentionCheckAdded = true;
        }
    }
    
    // Function to show matrix with animation
    function showMatrix() {
        setTimeout(function() {
            question.find('table').fadeIn(600);
            
            // Record actual latency in embedded data
            Qualtrics.SurveyEngine.setEmbeddedData('trial_' + trialNum + '_actual_latency_ms', actualDisplayTime);
            
            // Scroll to matrix on mobile
            if (window.innerWidth < 768) {
                question.find('table')[0].scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
        }, 300);
    }
    
    // Execute based on modality with error handling
    try {
        if (modality === 'stream') {
            showStreamResponse();
        } else {
            showNonStreamResponse();
        }
    } catch (error) {
        console.error('Error in response display:', error);
        // Fallback to immediate display
        jQuery('#loading-indicator').hide();
        jQuery('#response-text').html(responseText);
        showMatrix();
    }
    
    // Add enhanced CSS
    jQuery('head').append(`
        <style>
            .typing-cursor {
                animation: blink 1s infinite;
                font-weight: bold;
                color: #666;
            }
            
            @keyframes blink {
                0%, 50% { opacity: 1; }
                51%, 100% { opacity: 0; }
            }
            
            /* Mobile optimizations */
            @media (max-width: 768px) {
                #response-container {
                    margin: 15px 0 !important;
                    padding: 15px !important;
                    font-size: 15px !important;
                }
                
                .QuestionText + div {
                    padding: 15px !important;
                    margin-bottom: 20px !important;
                }
            }
            
            /* Improve matrix table appearance */
            .Skin table.ChoiceStructure {
                border-collapse: separate;
                border-spacing: 0;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            
            .Skin .Matrix th {
                background-color: #f5f5f5;
                font-weight: 600;
            }
        </style>
    `);
});

Qualtrics.SurveyEngine.addOnReady(function() {
    // Hide Next button until matrix is shown
    jQuery('#NextButton').hide();
    
    // Show Next button when matrix becomes visible
    var checkMatrix = setInterval(function() {
        if (jQuery('table:visible').length > 0) {
            jQuery('#NextButton').show();
            clearInterval(checkMatrix);
        }
    }, 100);
    
    // Add progress indicator
    var trialNum = "${e://Field/current_trial}" || "1";
    var progressHtml = '<div style="position: fixed; top: 10px; right: 10px; background: rgba(0,0,0,0.8); color: white; padding: 8px 12px; border-radius: 20px; font-size: 12px; z-index: 1000;">Trial ' + trialNum + ' of 5</div>';
    jQuery('body').append(progressHtml);
});

Qualtrics.SurveyEngine.addOnUnload(function() {
    // Clean up any remaining intervals and timeouts
    var highestTimeoutId = setTimeout(";");
    for (var i = 0; i < highestTimeoutId; i++) {
        clearTimeout(i);
    }
    
    var highestIntervalId = setInterval(";");
    for (var i = 0; i < highestIntervalId; i++) {
        clearInterval(i);
    }
});