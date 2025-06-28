import re
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

from taskflow.llm import LLMClient


@dataclass
class EvaluationResult:
    """Represents the result of a task evaluation."""
    is_fulfilled: bool
    score: int  # 1-5 scale
    feedback_message: str
    raw_evaluation: str


class TaskEvaluator:
    """
    Handles task evaluation logic, including prompting the LLM for evaluation,
    parsing responses, and determining task fulfillment.
    """
    
    def __init__(self, llm_client: LLMClient):
        """
        Initialize the TaskEvaluator.
        
        Args:
            llm_client: LLM client for evaluation prompts
        """
        self.llm_client = llm_client
        
        # Score patterns for parsing LLM responses
        self.score_patterns = [
            r"(?:score|rating):\s*([1-5])",
            r"([1-5])/5",
            r"score\s+([1-5])",
            r"rating\s+([1-5])",
            r"^([1-5])\s*-",
            r"^([1-5])\.",
        ]
        
        # Default score threshold for considering a task fulfilled
        self.fulfillment_threshold = 4
    
    def evaluate(self, original_task: str, agent_response: str, 
                feedback_context: str = "") -> EvaluationResult:
        """
        Evaluates if the original task was fulfilled by the agent response.
        
        Args:
            original_task: The original user task/prompt
            agent_response: The response from the agent that handled the task
            feedback_context: Any previous feedback that should be considered
            
        Returns:
            EvaluationResult containing the evaluation outcome
        """
        evaluation_prompt = self._build_evaluation_prompt(
            original_task, agent_response, feedback_context
        )
        
        print("Evaluating if task was fulfilled...")
        
        try:
            response = self.llm_client.chat(prompt=evaluation_prompt)
            raw_evaluation = response.content.strip()
            print(f"Evaluation result: {raw_evaluation}")
            
            # Parse the score from the evaluation result
            score = self._parse_score(raw_evaluation)
            
            # Determine if task is fulfilled based on score
            is_fulfilled = score >= self.fulfillment_threshold
            
            # Create feedback message
            feedback_message = f"Score: {score}/5 - {raw_evaluation}"
            
            result = EvaluationResult(
                is_fulfilled=is_fulfilled,
                score=score,
                feedback_message=feedback_message,
                raw_evaluation=raw_evaluation
            )
            
            return result
            
        except Exception as e:
            print(f"Error during task evaluation: {e}")
            error_result = EvaluationResult(
                is_fulfilled=False,
                score=1,
                feedback_message=f"Error during evaluation: {e}",
                raw_evaluation=f"Evaluation failed: {e}"
            )
            
            return error_result
    
    def evaluate_with_custom_criteria(self, original_task: str, agent_response: str,
                                    custom_criteria: str, feedback_context: str = "") -> EvaluationResult:
        """
        Evaluates task fulfillment using custom evaluation criteria.
        
        Args:
            original_task: The original user task/prompt
            agent_response: The response from the agent that handled the task
            custom_criteria: Custom criteria for evaluation
            feedback_context: Any previous feedback that should be considered
            
        Returns:
            EvaluationResult containing the evaluation outcome
        """
        evaluation_prompt = self._build_custom_evaluation_prompt(
            original_task, agent_response, custom_criteria, feedback_context
        )
        
        print("Evaluating task with custom criteria...")
        
        try:
            response = self.llm_client.chat(prompt=evaluation_prompt)
            raw_evaluation = response.content.strip()
            print(f"Custom evaluation result: {raw_evaluation}")
            
            score = self._parse_score(raw_evaluation)
            is_fulfilled = score >= self.fulfillment_threshold
            feedback_message = f"Score: {score}/5 - {raw_evaluation}"
            
            result = EvaluationResult(
                is_fulfilled=is_fulfilled,
                score=score,
                feedback_message=feedback_message,
                raw_evaluation=raw_evaluation
            )
            return result
            
        except Exception as e:
            print(f"Error during custom task evaluation: {e}")
            return EvaluationResult(
                is_fulfilled=False,
                score=1,
                feedback_message=f"Error during custom evaluation: {e}",
                raw_evaluation=f"Custom evaluation failed: {e}"
            )
    
    def parse_evaluator_agent_response(self, evaluator_response: str) -> EvaluationResult:
        """
        Parses a response from an Evaluator agent to extract evaluation results.
        
        Args:
            evaluator_response: The response from an Evaluator agent
            
        Returns:
            EvaluationResult containing the parsed evaluation
        """
        # Extract score from the evaluator response
        score_match = re.search(r"Evaluation score:\s*([1-5])", evaluator_response)
        score = int(score_match.group(1)) if score_match else 1
        
        is_fulfilled = score >= self.fulfillment_threshold
        feedback_message = f"Evaluator score: {score}/5"
        
        result = EvaluationResult(
            is_fulfilled=is_fulfilled,
            score=score,
            feedback_message=feedback_message,
            raw_evaluation=evaluator_response
        )
        return result
    
    def set_fulfillment_threshold(self, threshold: int):
        """
        Set the score threshold for considering a task fulfilled.
        
        Args:
            threshold: Score threshold (1-5)
        """
        if 1 <= threshold <= 5:
            self.fulfillment_threshold = threshold
        else:
            raise ValueError("Threshold must be between 1 and 5")
    
    def _build_evaluation_prompt(self, original_task: str, agent_response: str, 
                               feedback_context: str = "") -> str:
        """Build the evaluation prompt for the LLM."""
        evaluation_prompt = f"""Evaluate if the following user task was properly fulfilled by the agent response:

Original Task:
{original_task}

Agent Response:
{agent_response}"""

        if feedback_context:
            evaluation_prompt += f"""

Previous Feedback Context:
{feedback_context}

Please consider the previous feedback when evaluating if the agent response addresses the concerns raised."""
        
        evaluation_prompt += """

Please evaluate if the agent response adequately fulfills the user's original task. 

IMPORTANT EVALUATION CRITERIA:
- If the user requested to generate a commit message, then only generating the message is sufficient.
- If the user requested to "commit the changes", the task is only complete if the response shows that changes were actually committed (e.g., contains "Successfully committed with hash:" or similar confirmation).
- If the user requested a review, the task is complete if a review was provided.
- If the user requested a diff, the task is complete if the diff output was provided.

Rate the response on a scale of 1-5:
1 = Poor: The response completely fails to address the user's request
2 = Below Average: The response partially addresses the request but has significant gaps
3 = Average: The response addresses most of the request but lacks some important elements
4 = Good: The response addresses the request well with minor issues
5 = Excellent: The response completely and accurately fulfills the user's request

Respond with:
Score: [1-5]
[Detailed explanation of your evaluation]"""

        return evaluation_prompt
    
    def _build_custom_evaluation_prompt(self, original_task: str, agent_response: str,
                                      custom_criteria: str, feedback_context: str = "") -> str:
        """Build a custom evaluation prompt with specific criteria."""
        evaluation_prompt = f"""Evaluate if the following user task was properly fulfilled by the agent response using the specified criteria:

Original Task:
{original_task}

Agent Response:
{agent_response}

Custom Evaluation Criteria:
{custom_criteria}"""

        if feedback_context:
            evaluation_prompt += f"""

Previous Feedback Context:
{feedback_context}

Please consider the previous feedback when evaluating if the agent response addresses the concerns raised."""
        
        evaluation_prompt += """

Please evaluate if the agent response adequately fulfills the user's original task according to the custom criteria provided.

Rate the response on a scale of 1-5:
1 = Poor: The response completely fails to meet the criteria
2 = Below Average: The response partially meets the criteria but has significant gaps
3 = Average: The response meets most of the criteria but lacks some important elements
4 = Good: The response meets the criteria well with minor issues
5 = Excellent: The response completely and accurately meets all criteria

Respond with:
Score: [1-5]
[Detailed explanation of your evaluation based on the custom criteria]"""

        return evaluation_prompt
    
    def _parse_score(self, evaluation_result: str) -> int:
        """Parse the score from the LLM evaluation result."""
        for pattern in self.score_patterns:
            match = re.search(pattern, evaluation_result, re.IGNORECASE | re.MULTILINE)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    continue
        
        # Default score if no pattern matches
        return 1
