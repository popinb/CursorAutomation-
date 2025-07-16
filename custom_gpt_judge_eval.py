#!/usr/bin/env python3
"""
Custom GPT LLM as a Judge Evaluator
===================================

This module converts a Custom GPT evaluation setup into a structured OpenAI evals framework.
It simulates the exact instructions and knowledge sources that would be in a Custom GPT.

Key Components:
1. Custom GPT Instructions (system prompts)
2. Knowledge Sources (simulated as context documents)
3. LLM Judge Evaluation Logic
4. OpenAI Evals Framework Structure
"""

import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
from abc import ABC, abstractmethod
import openai
from openai import OpenAI
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvalSample:
    """Represents a single evaluation sample following OpenAI evals format."""
    input: List[Dict[str, str]]  # Chat format messages
    ideal: Optional[str] = None  # Expected output
    expected: Optional[str] = None  # Alternative to ideal
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EvalResult:
    """Result of evaluating a single sample."""
    sample_id: str
    input: Dict
    output: str
    expected: Optional[str]
    score: float
    passed: bool
    reasoning: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class CustomGPTKnowledgeBase:
    """
    Simulates the knowledge sources that would be uploaded to a Custom GPT.
    In a real Custom GPT, these would be PDF files, documents, or other knowledge sources.
    """
    
    def __init__(self, knowledge_sources: List[str]):
        self.knowledge_sources = knowledge_sources
        self.knowledge_content = self._load_knowledge_sources()
    
    def _load_knowledge_sources(self) -> str:
        """
        In a real implementation, this would load actual files.
        For this example, we'll use simulated knowledge content.
        """
        return """
        EVALUATION CRITERIA KNOWLEDGE BASE
        
        1. ACCURACY ASSESSMENT
        - Check factual correctness
        - Verify logical consistency
        - Assess completeness of information
        
        2. QUALITY STANDARDS
        - Clarity and coherence
        - Appropriate tone and style
        - Grammar and language usage
        
        3. TASK-SPECIFIC CRITERIA
        - Relevance to the question
        - Depth of analysis
        - Use of evidence and examples
        
        4. SCORING RUBRIC
        - Excellent (4-5): Exceeds expectations
        - Good (3-4): Meets expectations
        - Fair (2-3): Partially meets expectations
        - Poor (1-2): Below expectations
        - Unacceptable (0-1): Fails to meet basic requirements
        """
    
    def get_context(self, query: str = "") -> str:
        """Get relevant context from knowledge base."""
        return self.knowledge_content


class CustomGPTJudge:
    """
    Main judge class that simulates a Custom GPT evaluator.
    Contains the custom instructions and evaluation logic.
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4", knowledge_base: Optional[CustomGPTKnowledgeBase] = None):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.knowledge_base = knowledge_base
        
        # Custom GPT Instructions - This would be the "Instructions" field in Custom GPT
        self.custom_instructions = """
        You are an expert AI evaluator designed to assess the quality of responses from AI systems.
        Your role is to act as an impartial judge, providing detailed analysis and scoring.
        
        CORE RESPONSIBILITIES:
        1. Evaluate responses for accuracy, relevance, and quality
        2. Provide detailed reasoning for your assessments
        3. Score responses on a scale of 0-5
        4. Identify strengths and areas for improvement
        
        EVALUATION PROCESS:
        1. Read the original question/prompt carefully
        2. Analyze the response against the evaluation criteria
        3. Consider the knowledge base guidelines
        4. Provide step-by-step reasoning
        5. Assign a numerical score with justification
        
        SCORING CRITERIA:
        - 5: Exceptional - Exceeds all expectations
        - 4: Good - Meets all expectations well
        - 3: Satisfactory - Meets basic expectations
        - 2: Below Average - Partially meets expectations
        - 1: Poor - Significant issues present
        - 0: Unacceptable - Fails to meet basic requirements
        
        Always provide your reasoning before giving the final score.
        Be objective, fair, and constructive in your evaluations.
        """
    
    async def evaluate_response(
        self, 
        original_prompt: str, 
        response_to_evaluate: str, 
        expected_answer: Optional[str] = None,
        evaluation_criteria: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a response using the Custom GPT judge logic.
        """
        
        # Get context from knowledge base
        context = self.knowledge_base.get_context() if self.knowledge_base else ""
        
        # Construct the evaluation prompt
        evaluation_prompt = self._construct_evaluation_prompt(
            original_prompt, 
            response_to_evaluate, 
            expected_answer, 
            evaluation_criteria,
            context
        )
        
        try:
            # Call OpenAI API for evaluation
            response = await self._call_openai_api(evaluation_prompt)
            
            # Parse the response to extract score and reasoning
            parsed_result = self._parse_evaluation_response(response)
            
            return parsed_result
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            return {
                "score": 0.0,
                "reasoning": f"Evaluation failed due to error: {str(e)}",
                "passed": False
            }
    
    def _construct_evaluation_prompt(
        self, 
        original_prompt: str, 
        response: str, 
        expected_answer: Optional[str],
        evaluation_criteria: Optional[str],
        context: str
    ) -> List[Dict[str, str]]:
        """Construct the prompt for evaluation."""
        
        system_message = f"{self.custom_instructions}\n\n"
        
        if context:
            system_message += f"KNOWLEDGE BASE CONTEXT:\n{context}\n\n"
        
        if evaluation_criteria:
            system_message += f"SPECIFIC EVALUATION CRITERIA:\n{evaluation_criteria}\n\n"
        
        user_prompt = f"""
        Please evaluate the following response:

        ORIGINAL PROMPT:
        {original_prompt}

        RESPONSE TO EVALUATE:
        {response}
        """
        
        if expected_answer:
            user_prompt += f"""
        EXPECTED/REFERENCE ANSWER:
        {expected_answer}
        """
        
        user_prompt += """
        
        Please provide your evaluation in the following format:
        
        REASONING:
        [Your step-by-step analysis here]
        
        SCORE: [0-5]
        
        PASSED: [Yes/No]
        """
        
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt}
        ]
    
    async def _call_openai_api(self, messages: List[Dict[str, str]]) -> str:
        """Make an async call to OpenAI API."""
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=messages,
                temperature=0.1,  # Low temperature for consistent evaluations
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API call failed: {str(e)}")
            raise
    
    def _parse_evaluation_response(self, response: str) -> Dict[str, Any]:
        """Parse the evaluation response to extract score and reasoning."""
        try:
            # Extract reasoning
            reasoning_start = response.find("REASONING:")
            score_start = response.find("SCORE:")
            passed_start = response.find("PASSED:")
            
            reasoning = ""
            score = 0.0
            passed = False
            
            if reasoning_start != -1 and score_start != -1:
                reasoning = response[reasoning_start + 10:score_start].strip()
            
            if score_start != -1:
                score_section = response[score_start + 6:]
                if passed_start != -1:
                    score_text = score_section[:passed_start - score_start - 6].strip()
                else:
                    score_text = score_section.split('\n')[0].strip()
                
                # Extract numeric score
                import re
                score_match = re.search(r'(\d+(?:\.\d+)?)', score_text)
                if score_match:
                    score = float(score_match.group(1))
            
            if passed_start != -1:
                passed_text = response[passed_start + 7:].split('\n')[0].strip().lower()
                passed = passed_text.startswith('yes')
            else:
                # If no explicit passed/failed, use score threshold
                passed = score >= 3.0
            
            return {
                "score": score,
                "reasoning": reasoning,
                "passed": passed,
                "raw_response": response
            }
            
        except Exception as e:
            logger.error(f"Error parsing evaluation response: {str(e)}")
            return {
                "score": 0.0,
                "reasoning": f"Failed to parse evaluation: {str(e)}",
                "passed": False,
                "raw_response": response
            }


class CustomGPTEvaluator:
    """
    OpenAI Evals-compatible evaluator class.
    This follows the OpenAI evals framework structure.
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4", knowledge_sources: Optional[List[str]] = None):
        # Initialize knowledge base
        self.knowledge_base = CustomGPTKnowledgeBase(knowledge_sources or [])
        
        # Initialize the judge
        self.judge = CustomGPTJudge(api_key, model, self.knowledge_base)
        
        # Evaluation metadata
        self.eval_name = "custom-gpt-judge"
        self.eval_description = "Custom GPT LLM as a Judge Evaluator"
        self.eval_version = "1.0.0"
    
    async def evaluate_single_sample(self, sample: EvalSample) -> EvalResult:
        """Evaluate a single sample."""
        
        # Extract the user prompt from the input messages
        user_prompt = ""
        for message in sample.input:
            if message["role"] == "user":
                user_prompt = message["content"]
                break
        
        # For this example, we'll simulate getting a response to evaluate
        # In a real scenario, this would be the actual model response being evaluated
        response_to_evaluate = sample.metadata.get("response_to_evaluate", "") if sample.metadata else ""
        
        if not response_to_evaluate:
            # If no response provided, we can't evaluate
            return EvalResult(
                sample_id=f"sample_{int(time.time())}",
                input={"prompt": user_prompt},
                output="No response to evaluate",
                expected=sample.ideal or sample.expected,
                score=0.0,
                passed=False,
                reasoning="No response provided for evaluation"
            )
        
        # Perform the evaluation
        eval_result = await self.judge.evaluate_response(
            original_prompt=user_prompt,
            response_to_evaluate=response_to_evaluate,
            expected_answer=sample.ideal or sample.expected,
            evaluation_criteria=sample.metadata.get("evaluation_criteria") if sample.metadata else None
        )
        
        return EvalResult(
            sample_id=f"sample_{int(time.time())}",
            input={"prompt": user_prompt},
            output=response_to_evaluate,
            expected=sample.ideal or sample.expected,
            score=eval_result["score"],
            passed=eval_result["passed"],
            reasoning=eval_result["reasoning"],
            metadata={"raw_evaluation": eval_result}
        )
    
    async def evaluate_samples(self, samples: List[EvalSample]) -> List[EvalResult]:
        """Evaluate multiple samples."""
        results = []
        
        for i, sample in enumerate(samples):
            logger.info(f"Evaluating sample {i+1}/{len(samples)}")
            result = await self.evaluate_single_sample(sample)
            results.append(result)
            
            # Add small delay to avoid rate limiting
            await asyncio.sleep(0.1)
        
        return results
    
    def generate_report(self, results: List[EvalResult]) -> Dict[str, Any]:
        """Generate evaluation report in OpenAI evals format."""
        
        total_samples = len(results)
        passed_samples = sum(1 for r in results if r.passed)
        total_score = sum(r.score for r in results)
        
        report = {
            "eval_name": self.eval_name,
            "eval_description": self.eval_description,
            "eval_version": self.eval_version,
            "total_samples": total_samples,
            "passed_samples": passed_samples,
            "failed_samples": total_samples - passed_samples,
            "pass_rate": passed_samples / total_samples if total_samples > 0 else 0,
            "average_score": total_score / total_samples if total_samples > 0 else 0,
            "score_distribution": self._calculate_score_distribution(results),
            "detailed_results": [
                {
                    "sample_id": r.sample_id,
                    "score": r.score,
                    "passed": r.passed,
                    "reasoning": r.reasoning
                }
                for r in results
            ]
        }
        
        return report
    
    def _calculate_score_distribution(self, results: List[EvalResult]) -> Dict[str, int]:
        """Calculate score distribution."""
        distribution = {"0-1": 0, "1-2": 0, "2-3": 0, "3-4": 0, "4-5": 0}
        
        for result in results:
            score = result.score
            if score < 1:
                distribution["0-1"] += 1
            elif score < 2:
                distribution["1-2"] += 1
            elif score < 3:
                distribution["2-3"] += 1
            elif score < 4:
                distribution["3-4"] += 1
            else:
                distribution["4-5"] += 1
        
        return distribution


# Example usage and demo functions
class EvalDataLoader:
    """Utility class to load evaluation data in OpenAI evals format."""
    
    @staticmethod
    def load_from_jsonl(file_path: str) -> List[EvalSample]:
        """Load samples from JSONL file."""
        samples = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    sample = EvalSample(
                        input=data["input"],
                        ideal=data.get("ideal"),
                        expected=data.get("expected"),
                        metadata=data.get("metadata")
                    )
                    samples.append(sample)
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
        except Exception as e:
            logger.error(f"Error loading samples: {str(e)}")
        
        return samples
    
    @staticmethod
    def create_sample_data() -> List[EvalSample]:
        """Create sample evaluation data for demonstration."""
        return [
            EvalSample(
                input=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions accurately."},
                    {"role": "user", "content": "What is the capital of France?"}
                ],
                ideal="Paris",
                metadata={
                    "response_to_evaluate": "The capital of France is Paris, which is located in the north-central part of the country.",
                    "evaluation_criteria": "Check for factual accuracy and completeness."
                }
            ),
            EvalSample(
                input=[
                    {"role": "system", "content": "You are a math tutor."},
                    {"role": "user", "content": "What is 15 + 27?"}
                ],
                ideal="42",
                metadata={
                    "response_to_evaluate": "15 + 27 = 42",
                    "evaluation_criteria": "Check mathematical accuracy."
                }
            ),
            EvalSample(
                input=[
                    {"role": "system", "content": "You are a creative writing assistant."},
                    {"role": "user", "content": "Write a short poem about the ocean."}
                ],
                ideal=None,  # No single correct answer for creative tasks
                metadata={
                    "response_to_evaluate": "Waves crash upon the shore so blue,\nWith foam as white as morning dew.\nThe ocean vast and deep and wide,\nWhere secrets of the earth reside.",
                    "evaluation_criteria": "Assess creativity, imagery, and poetic structure."
                }
            )
        ]


async def main():
    """Main function to demonstrate the Custom GPT Judge Evaluator."""
    
    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: Please set OPENAI_API_KEY environment variable")
        return
    
    # Initialize the evaluator
    knowledge_sources = ["evaluation_guidelines.pdf", "quality_standards.txt", "scoring_rubric.md"]
    evaluator = CustomGPTEvaluator(api_key=api_key, knowledge_sources=knowledge_sources)
    
    # Load or create sample data
    print("Loading evaluation samples...")
    samples = EvalDataLoader.create_sample_data()
    
    # Run evaluation
    print(f"Running evaluation on {len(samples)} samples...")
    results = await evaluator.evaluate_samples(samples)
    
    # Generate report
    print("Generating evaluation report...")
    report = evaluator.generate_report(results)
    
    # Display results
    print("\n" + "="*50)
    print("CUSTOM GPT JUDGE EVALUATION REPORT")
    print("="*50)
    print(f"Eval Name: {report['eval_name']}")
    print(f"Total Samples: {report['total_samples']}")
    print(f"Passed: {report['passed_samples']}")
    print(f"Failed: {report['failed_samples']}")
    print(f"Pass Rate: {report['pass_rate']:.2%}")
    print(f"Average Score: {report['average_score']:.2f}/5.0")
    
    print("\nScore Distribution:")
    for range_name, count in report['score_distribution'].items():
        print(f"  {range_name}: {count} samples")
    
    print("\nDetailed Results:")
    for result in report['detailed_results']:
        print(f"\nSample ID: {result['sample_id']}")
        print(f"Score: {result['score']}/5.0")
        print(f"Passed: {result['passed']}")
        print(f"Reasoning: {result['reasoning'][:200]}...")
    
    # Save report to file
    output_file = "custom_gpt_evaluation_report.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\nFull report saved to: {output_file}")


if __name__ == "__main__":
    # Run the evaluation
    asyncio.run(main())