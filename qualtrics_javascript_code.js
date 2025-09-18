// QUALTRICS JAVASCRIPT CODE FOR LATENCY STUDY
// ============================================
// Add this code to the JavaScript section of each trial question

Qualtrics.SurveyEngine.addOnload(function() {
    // Get trial data from embedded data
    const trialNum = this.getQuestionInfo().QuestionID.split('_')[1]; // Extract trial number
    const promptId = "${e://Field/trial_" + trialNum + "_prompt}";
    const modality = "${e://Field/trial_" + trialNum + "_modality}";
    const latency = parseInt("${e://Field/trial_" + trialNum + "_latency}");
    const tft = parseInt("${e://Field/trial_" + trialNum + "_tft}");
    
    // Store trial data globally
    window.currentTrial = {
        num: trialNum,
        promptId: promptId,
        modality: modality,
        latency: latency,
        tft: tft
    };
    
    // Initialize the trial
    initializeTrial();
});

Qualtrics.SurveyEngine.addOnReady(function() {
    // Start the response display
    startResponseDisplay();
});

function initializeTrial() {
    // Get the response container
    const responseContainer = document.getElementById('response-container');
    if (!responseContainer) {
        // Create response container if it doesn't exist
        const questionContainer = document.querySelector('.QuestionOuter');
        const responseDiv = document.createElement('div');
        responseDiv.id = 'response-container';
        responseDiv.className = 'response-container';
        responseDiv.innerHTML = '<div class="loading-indicator">Preparing response...</div>';
        questionContainer.appendChild(responseDiv);
    }
    
    // Hide the matrix question initially
    const matrixQuestion = document.querySelector('.Matrix');
    if (matrixQuestion) {
        matrixQuestion.style.display = 'none';
    }
}

function startResponseDisplay() {
    const trial = window.currentTrial;
    const responseContainer = document.getElementById('response-container');
    const answerText = getAnswerText(trial.promptId);
    
    if (trial.modality === 'nonstream') {
        // Non-streaming: show full answer after delay
        setTimeout(() => {
            responseContainer.innerHTML = '<div class="streaming-text">' + answerText + '</div>';
            showMatrixQuestion();
        }, trial.latency);
    } else {
        // Streaming: start showing characters after TFT, then stream
        setTimeout(() => {
            streamText(responseContainer, answerText, trial.latency - trial.tft, () => {
                showMatrixQuestion();
            });
        }, trial.tft);
    }
}

function streamText(container, text, duration, onComplete) {
    const characters = text.split('');
    const interval = duration / characters.length;
    let currentIndex = 0;
    
    container.innerHTML = '<div class="streaming-text"></div>';
    const textElement = container.querySelector('.streaming-text');
    
    const streamInterval = setInterval(() => {
        if (currentIndex < characters.length) {
            textElement.textContent += characters[currentIndex];
            currentIndex++;
        } else {
            clearInterval(streamInterval);
            if (onComplete) onComplete();
        }
    }, interval);
}

function showMatrixQuestion() {
    const matrixQuestion = document.querySelector('.Matrix');
    if (matrixQuestion) {
        matrixQuestion.style.display = 'block';
    }
}

function getAnswerText(promptId) {
    const answers = {
        '1': "Compound interest is the interest calculated on the initial principal and the accumulated interest from previous periods, meaning you earn interest on your interest, causing your investment to grow exponentially over time.",
        '2': "1) Fried rice with scrambled eggs and wilted spinach, 2) Spinach and egg rice bowl with a soft-boiled egg on top, 3) Rice and egg frittata with fresh spinach mixed in and baked until golden.",
        '3': "The corrected code is: `print('hi')` - the original had a typo 'pritn' instead of 'print', which is the proper Python function for displaying output to the console.",
        '4': "Heavy rain, dropping temperatures, and strong winds are expected throughout the weekend according to the weather forecast.",
        '5': "For a rainy weekend, I recommend indoor activities like reading a good book, cooking a new recipe, doing puzzles or board games, watching movies, or learning something new online through tutorials or courses."
    };
    return answers[promptId] || "Answer not found.";
}

// CSS Styles (add to Look & Feel > Style > Custom CSS)
const css = `
.response-container {
    min-height: 200px;
    border: 1px solid #ddd;
    padding: 15px;
    margin: 10px 0;
    background-color: #f9f9f9;
    border-radius: 5px;
}

.streaming-text {
    font-family: Arial, sans-serif;
    line-height: 1.5;
}

.loading-indicator {
    color: #666;
    font-style: italic;
}

.trial-info {
    background-color: #e8f4f8;
    padding: 10px;
    margin: 10px 0;
    border-radius: 3px;
    font-size: 12px;
}
`;

// Add CSS to page
const style = document.createElement('style');
style.textContent = css;
document.head.appendChild(style);