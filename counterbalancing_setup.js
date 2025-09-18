// Counterbalancing Setup for Qualtrics Survey Flow
// This creates 5 different condition assignments to ensure each prompt appears in each condition

// Configuration for 5 different randomization elements
const conditionAssignments = [
    // Assignment A
    {
        trial1: { prompt_id: 1, modality: "nonstream", latency_ms: 500, tft_ms: 0 },
        trial2: { prompt_id: 2, modality: "nonstream", latency_ms: 2000, tft_ms: 0 },
        trial3: { prompt_id: 3, modality: "nonstream", latency_ms: 6000, tft_ms: 0 },
        trial4: { prompt_id: 4, modality: "stream", latency_ms: 2000, tft_ms: 300 },
        trial5: { prompt_id: 5, modality: "stream", latency_ms: 6000, tft_ms: 300 }
    },
    // Assignment B
    {
        trial1: { prompt_id: 2, modality: "nonstream", latency_ms: 500, tft_ms: 0 },
        trial2: { prompt_id: 3, modality: "nonstream", latency_ms: 2000, tft_ms: 0 },
        trial3: { prompt_id: 4, modality: "nonstream", latency_ms: 6000, tft_ms: 0 },
        trial4: { prompt_id: 5, modality: "stream", latency_ms: 2000, tft_ms: 300 },
        trial5: { prompt_id: 1, modality: "stream", latency_ms: 6000, tft_ms: 300 }
    },
    // Assignment C
    {
        trial1: { prompt_id: 3, modality: "nonstream", latency_ms: 500, tft_ms: 0 },
        trial2: { prompt_id: 4, modality: "nonstream", latency_ms: 2000, tft_ms: 0 },
        trial3: { prompt_id: 5, modality: "nonstream", latency_ms: 6000, tft_ms: 0 },
        trial4: { prompt_id: 1, modality: "stream", latency_ms: 2000, tft_ms: 300 },
        trial5: { prompt_id: 2, modality: "stream", latency_ms: 6000, tft_ms: 300 }
    },
    // Assignment D
    {
        trial1: { prompt_id: 4, modality: "nonstream", latency_ms: 500, tft_ms: 0 },
        trial2: { prompt_id: 5, modality: "nonstream", latency_ms: 2000, tft_ms: 0 },
        trial3: { prompt_id: 1, modality: "nonstream", latency_ms: 6000, tft_ms: 0 },
        trial4: { prompt_id: 2, modality: "stream", latency_ms: 2000, tft_ms: 300 },
        trial5: { prompt_id: 3, modality: "stream", latency_ms: 6000, tft_ms: 300 }
    },
    // Assignment E
    {
        trial1: { prompt_id: 5, modality: "nonstream", latency_ms: 500, tft_ms: 0 },
        trial2: { prompt_id: 1, modality: "nonstream", latency_ms: 2000, tft_ms: 0 },
        trial3: { prompt_id: 2, modality: "nonstream", latency_ms: 6000, tft_ms: 0 },
        trial4: { prompt_id: 3, modality: "stream", latency_ms: 2000, tft_ms: 300 },
        trial5: { prompt_id: 4, modality: "stream", latency_ms: 6000, tft_ms: 300 }
    }
];

// Additional counterbalancing patterns to ensure complete coverage
const additionalAssignments = [
    // Assignment F - Different latency mappings
    {
        trial1: { prompt_id: 1, modality: "nonstream", latency_ms: 2000, tft_ms: 0 },
        trial2: { prompt_id: 2, modality: "nonstream", latency_ms: 6000, tft_ms: 0 },
        trial3: { prompt_id: 3, modality: "stream", latency_ms: 2000, tft_ms: 300 },
        trial4: { prompt_id: 4, modality: "stream", latency_ms: 6000, tft_ms: 300 },
        trial5: { prompt_id: 5, modality: "nonstream", latency_ms: 500, tft_ms: 0 }
    },
    // Assignment G
    {
        trial1: { prompt_id: 2, modality: "nonstream", latency_ms: 6000, tft_ms: 0 },
        trial2: { prompt_id: 3, modality: "stream", latency_ms: 6000, tft_ms: 300 },
        trial3: { prompt_id: 4, modality: "nonstream", latency_ms: 500, tft_ms: 0 },
        trial4: { prompt_id: 5, modality: "nonstream", latency_ms: 2000, tft_ms: 0 },
        trial5: { prompt_id: 1, modality: "stream", latency_ms: 2000, tft_ms: 300 }
    },
    // Assignment H
    {
        trial1: { prompt_id: 3, modality: "stream", latency_ms: 2000, tft_ms: 300 },
        trial2: { prompt_id: 4, modality: "stream", latency_ms: 6000, tft_ms: 300 },
        trial3: { prompt_id: 5, modality: "nonstream", latency_ms: 500, tft_ms: 0 },
        trial4: { prompt_id: 1, modality: "nonstream", latency_ms: 2000, tft_ms: 0 },
        trial5: { prompt_id: 2, modality: "nonstream", latency_ms: 6000, tft_ms: 0 }
    },
    // Assignment I
    {
        trial1: { prompt_id: 4, modality: "nonstream", latency_ms: 500, tft_ms: 0 },
        trial2: { prompt_id: 5, modality: "stream", latency_ms: 2000, tft_ms: 300 },
        trial3: { prompt_id: 1, modality: "stream", latency_ms: 6000, tft_ms: 300 },
        trial4: { prompt_id: 2, modality: "nonstream", latency_ms: 2000, tft_ms: 0 },
        trial5: { prompt_id: 3, modality: "nonstream", latency_ms: 6000, tft_ms: 0 }
    },
    // Assignment J
    {
        trial1: { prompt_id: 5, modality: "stream", latency_ms: 6000, tft_ms: 300 },
        trial2: { prompt_id: 1, modality: "nonstream", latency_ms: 500, tft_ms: 0 },
        trial3: { prompt_id: 2, modality: "stream", latency_ms: 2000, tft_ms: 300 },
        trial4: { prompt_id: 3, modality: "nonstream", latency_ms: 2000, tft_ms: 0 },
        trial5: { prompt_id: 4, modality: "nonstream", latency_ms: 6000, tft_ms: 0 }
    }
];

// Function to generate Qualtrics Embedded Data assignments
function generateEmbeddedDataAssignment(assignment, assignmentLetter) {
    console.log(`\n// Assignment ${assignmentLetter}`);
    console.log("Set Embedded Data:");
    
    for (let i = 1; i <= 5; i++) {
        const trial = assignment[`trial${i}`];
        console.log(`trial${i}_prompt_id = ${trial.prompt_id}`);
        console.log(`trial${i}_modality = ${trial.modality}`);
        console.log(`trial${i}_latency_ms = ${trial.latency_ms}`);
        console.log(`trial${i}_tft_ms = ${trial.tft_ms}`);
    }
}

// Generate all assignments
console.log("=== QUALTRICS SURVEY FLOW RANDOMIZER SETUP ===");
console.log("\nCreate a Randomizer with 'Evenly Present Elements' containing these elements:");

// Generate main assignments
['A', 'B', 'C', 'D', 'E'].forEach((letter, index) => {
    generateEmbeddedDataAssignment(conditionAssignments[index], letter);
});

// Generate additional assignments for better coverage
['F', 'G', 'H', 'I', 'J'].forEach((letter, index) => {
    generateEmbeddedDataAssignment(additionalAssignments[index], letter);
});

// Verification function to check coverage
function verifyCoverage() {
    const coverage = {};
    
    // Initialize coverage tracking
    for (let prompt = 1; prompt <= 5; prompt++) {
        coverage[prompt] = {
            'nonstream-500': 0,
            'nonstream-2000': 0,
            'nonstream-6000': 0,
            'stream-2000': 0,
            'stream-6000': 0
        };
    }
    
    // Count occurrences
    const allAssignments = [...conditionAssignments, ...additionalAssignments];
    allAssignments.forEach(assignment => {
        for (let i = 1; i <= 5; i++) {
            const trial = assignment[`trial${i}`];
            const key = `${trial.modality}-${trial.latency_ms}`;
            coverage[trial.prompt_id][key]++;
        }
    });
    
    // Report coverage
    console.log("\n\n=== COVERAGE VERIFICATION ===");
    console.log("Each cell shows how many times each prompt appears in each condition:");
    console.log("\nPrompt | NS-500 | NS-2000 | NS-6000 | S-2000 | S-6000");
    console.log("-------|---------|----------|----------|---------|--------");
    
    for (let prompt = 1; prompt <= 5; prompt++) {
        const counts = coverage[prompt];
        console.log(`   ${prompt}   |   ${counts['nonstream-500']}    |    ${counts['nonstream-2000']}    |    ${counts['nonstream-6000']}    |   ${counts['stream-2000']}    |   ${counts['stream-6000']}`);
    }
}

// Run verification
verifyCoverage();