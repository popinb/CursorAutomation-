/* 
QUALTRICS JAVASCRIPT FOR TRIAL PAGES
Add this to each trial question's JavaScript section
*/

Qualtrics.SurveyEngine.addOnload(function() {
    // Get current question element
    var qid = this.questionId;
    var question = jQuery("#" + qid);
    
    // Get embedded data for current trial
    var trialNum = "${e://Field/current_trial}"; // Set this in Survey Flow for each block
    var promptId = "${e://Field/trial_" + trialNum + "_prompt_id}";
    var modality = "${e://Field/trial_" + trialNum + "_modality}";
    var latencyMs = parseInt("${e://Field/trial_" + trialNum + "_latency_ms}");
    var tftMs = parseInt("${e://Field/trial_" + trialNum + "_tft_ms}");
    
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
    
    var currentPrompt = prompts[promptId];
    var responseText = currentPrompt.answer;
    
    // Store actual timing data
    var startTime = Date.now();
    
    // Create response container
    var responseContainer = jQuery('<div id="response-container" style="margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; min-height: 100px; background-color: #fafafa;"><div id="response-text" style="font-family: Georgia, serif; line-height: 1.5; font-size: 16px;"></div></div>');
    
    // Insert prompt and response container
    var promptHtml = '<div style="margin-bottom: 20px; padding: 15px; background-color: #e3f2fd; border-radius: 5px;"><strong>Question:</strong> ' + currentPrompt.question + '</div>';
    question.find('.QuestionText').after(promptHtml + responseContainer[0].outerHTML);
    
    // Hide matrix initially
    question.find('table').hide();
    
    // Function for non-streaming response
    function showNonStreamResponse() {
        setTimeout(function() {
            jQuery('#response-text').html(responseText);
            showMatrix();
        }, latencyMs);
    }
    
    // Function for streaming response
    function showStreamResponse() {
        var chars = responseText.split('');
        var totalChars = chars.length;
        var remainingTime = latencyMs - tftMs;
        var intervalTime = remainingTime / totalChars;
        
        // Show first token
        setTimeout(function() {
            var currentText = '';
            var charIndex = 0;
            
            function addNextChar() {
                if (charIndex < totalChars) {
                    currentText += chars[charIndex];
                    jQuery('#response-text').html(currentText + '<span style="animation: blink 1s infinite;">|</span>');
                    charIndex++;
                    setTimeout(addNextChar, intervalTime);
                } else {
                    // Remove cursor and show matrix
                    jQuery('#response-text').html(currentText);
                    showMatrix();
                }
            }
            
            addNextChar();
        }, tftMs);
    }
    
    // Function to show matrix after response is complete
    function showMatrix() {
        question.find('table').fadeIn(500);
        // Record actual display time
        var actualLatency = Date.now() - startTime;
        Qualtrics.SurveyEngine.setEmbeddedData('trial_' + trialNum + '_actual_latency_ms', actualLatency);
    }
    
    // Execute based on modality
    if (modality === 'stream') {
        showStreamResponse();
    } else {
        showNonStreamResponse();
    }
    
    // Add CSS for blinking cursor
    jQuery('head').append('<style>.blink { animation: blink 1s infinite; } @keyframes blink { 0%, 50% { opacity: 1; } 51%, 100% { opacity: 0; } }</style>');
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
});

Qualtrics.SurveyEngine.addOnUnload(function() {
    // Clean up any remaining intervals
});