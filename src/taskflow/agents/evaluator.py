from typing import Optional, Dict, List, Callable

from taskflow.util import logger
from taskflow.llm import LLMClient
from taskflow.agents import Agent


class Evaluator(Agent):
    """
    A general-purpose agent responsible for evaluating whether a previous agent's response
    fulfills the user's original request with a numerical score from 1-5.
    """
    def __init__(self, model: LLMClient, system_prompt: str, available_tools: Optional[Dict[str, Callable]] = None):
        super().__init__("Evaluator", model, "Evaluates previous agent responses with a score from 1-5.", system_prompt, available_tools)

    def _get_tool_schemas(self) -> List[Dict]:
        """Returns the tool schemas available to the evaluator agent."""
        # The evaluator might need access to tools like diff_tool depending on the evaluation context
        return []

    def _extract_user_request_and_response(self, prompt: str) -> tuple[str, str]:
        """
        Extracts the original user request and the previous agent's response from the prompt.
        
        Parameters:
            prompt: The prompt containing both the user request and the response to evaluate.
            
        Returns:
            A tuple of (user_request, agent_response)
        """
        # Look for common patterns that separate user request from agent response
        separators = [
            "Agent Response:",
            "Previous Step Result:",
            "Step Result:",
            "Response:",
            "Output:",
            "Result:",
        ]
        
        user_request = ""
        agent_response = ""
        
        # Try to find the original task/request
        if "Original Task:" in prompt:
            start_idx = prompt.find("Original Task:") + len("Original Task:")
            # Find the end of the original task (usually before agent response)
            end_markers = separators + ["Step Context:", "Previous Step Results:"]
            end_idx = len(prompt)
            
            for marker in end_markers:
                marker_idx = prompt.find(marker, start_idx)
                if marker_idx != -1 and marker_idx < end_idx:
                    end_idx = marker_idx
            
            user_request = prompt[start_idx:end_idx].strip()
        
        # Try to find the agent response
        for separator in separators:
            if separator in prompt:
                start_idx = prompt.find(separator) + len(separator)
                # Take everything after the separator as the response
                agent_response = prompt[start_idx:].strip()
                break
        
        # If no clear separation found, try to parse JSON responses
        if not agent_response:
            try:
                import re
                
                # Look for JSON-like structures that might be agent responses
                json_pattern = r'\{[^{}]*"message"[^{}]*\}'
                json_matches = re.findall(json_pattern, prompt, re.DOTALL)
                
                if json_matches:
                    agent_response = json_matches[-1]  # Take the last JSON match
                
                # Also look for larger JSON blocks
                json_block_pattern = r'\{.*?\}'
                json_blocks = re.findall(json_block_pattern, prompt, re.DOTALL)
                
                if json_blocks and not agent_response:
                    agent_response = json_blocks[-1]  # Take the last JSON block
                    
            except Exception:
                pass
        
        # If still no response found, take the latter part of the prompt
        if not agent_response and user_request:
            # Everything after the user request could be the response
            remaining = prompt.replace(f"Original Task: {user_request}", "").strip()
            if remaining:
                agent_response = remaining
        elif not user_request and not agent_response:
            # Fallback: split the prompt in half
            lines = prompt.split('\n')
            mid_point = len(lines) // 2
            user_request = '\n'.join(lines[:mid_point]).strip()
            agent_response = '\n'.join(lines[mid_point:]).strip()
        
        return user_request, agent_response

    def _determine_evaluation_context(self, user_request: str, agent_response: str) -> str:
        """
        Determines what type of evaluation is needed based on the user request.
        
        Parameters:
            user_request: The original user request.
            agent_response: The agent's response to evaluate.
            
        Returns:
            A string indicating the evaluation context or empty string if no special context needed.
        """
        request_lower = user_request.lower()
        
        # Check if this involves code changes and might need diff context
        if any(keyword in request_lower for keyword in [
            'commit', 'diff', 'changes', 'staged', 'git', 'repository', 'code review'
        ]):
            project_dir = self._extract_project_dir(user_request + " " + agent_response)
            if project_dir:
                return f"code_changes:{project_dir}"
        
        return ""

    def _get_additional_context(self, evaluation_context: str) -> str:
        """
        Gets additional context needed for evaluation (e.g., git diff for code-related tasks).
        
        Parameters:
            evaluation_context: The context type and parameters.
            
        Returns:
            Additional context string or empty string if none needed.
        """
        if evaluation_context.startswith("code_changes:"):
            project_dir = evaluation_context.split(":", 1)[1]
            
            # Get the diff to provide context for code-related evaluations
            diff_prompt = f"Get the diff of staged changes for the project directory: {project_dir}"
            
            try:
                tools = self._get_tool_schemas()
                resp = self.model.chat(prompt=diff_prompt, system_prompt=self.system_prompt, tools=tools)
                
                if resp.function_call and resp.function_call.name == "diff_tool":
                    diff_result = self._execute_function_call(resp.function_call)
                    return f"\nGit Diff Context:\n```diff\n{diff_result}\n```"
                
            except Exception as e:
                print(f"Warning: Could not get diff context: {e}")
        
        return ""

    def _parse_evaluation_score(self, evaluation_text: str) -> tuple[int, str, str]:
        """
        Parses the evaluation response to extract score, explanation, and determines if fulfilled.
        
        Parameters:
            evaluation_text: The LLM's evaluation response
            
        Returns:
            A tuple of (score, explanation, fulfillment_status)
        """
        import re
        
        # Look for score patterns
        score_patterns = [
            r"(?:score|rating):\s*([1-5])",
            r"([1-5])/5",
            r"score\s+([1-5])",
            r"rating\s+([1-5])",
            r"^([1-5])\s*-",
            r"^([1-5])\.",
        ]
        
        score = 1  # Default score
        for pattern in score_patterns:
            match = re.search(pattern, evaluation_text, re.IGNORECASE | re.MULTILINE)
            if match:
                try:
                    score = int(match.group(1))
                    break
                except ValueError:
                    continue
        
        # Extract explanation (everything after score or the whole text)
        explanation = evaluation_text.strip()
        
        # Remove score line if it exists at the beginning
        explanation_lines = explanation.split('\n')
        if explanation_lines and any(str(i) in explanation_lines[0] for i in range(1, 6)):
            explanation = '\n'.join(explanation_lines[1:]).strip()
        
        # Determine fulfillment status based on score
        if score >= 4:
            fulfillment_status = "FULFILLED"
        else:
            fulfillment_status = "NOT_FULFILLED"
        
        return score, explanation, fulfillment_status

    def run(self, prompt: str, **kwargs) -> str:
        """
        Evaluates whether a previous agent's response fulfills the user's original request.
        Returns a score from 1-5 with explanation.

        Parameters:
            prompt: The prompt containing the user request and the previous agent's response.
            **kwargs: Additional keyword arguments (for compatibility).

        Returns:
            A formatted string with score, explanation, and the original agent response.
        """
        print(f"Evaluator agent running with prompt: {prompt[:100]}...")

        # Extract the user request and agent response from the prompt
        user_request, agent_response = self._extract_user_request_and_response(prompt)
        
        if not user_request:
            return "Error: Could not extract user request from prompt."
        
        if not agent_response:
            return "Error: Could not extract agent response from prompt."
        
        print(f"Extracted user request: {user_request[:100]}...")
        print(f"Extracted agent response: {agent_response[:100]}...")
        
        # Determine if we need additional context for evaluation
        evaluation_context = self._determine_evaluation_context(user_request, agent_response)
        additional_context = self._get_additional_context(evaluation_context)
        
        # Create the evaluation prompt
        eval_prompt = f"""You are evaluating whether an agent's response answers the user request. Rate the response on a scale of 1-5:

1 = Poor: The response completely fails to address the user's request
2 = Below Average: The response partially addresses the request but has significant gaps
3 = Average: The response addresses most of the request but lacks some important elements
4 = Good: The response addresses the request well with minor issues
5 = Excellent: The response completely and accurately fulfills the user's request

Original User Request:
```
{user_request}
```

Agent's Response:
```
{agent_response}
```
{additional_context}

EVALUATION CRITERIA:
- Does the response directly address what the user asked for?
- If the user requested an action (like committing changes), was the action actually performed?
- If the user requested information (like a review or diff), was the information provided?
- If the user requested generation of content (like a commit message), was the content generated?
- Are there any obvious gaps between what was requested and what was delivered?
- Quality and completeness of the response

Please respond with:
Score: [1-5]
[Detailed explanation of why you gave this score, including what was done well and what could be improved]

Be specific about your reasoning."""

        logger.info("-"*80)
        logger.info(eval_prompt)
        logger.info("-"*80)

        try:
            # Get the evaluation from the LLM
            eval_resp = self.model.chat(prompt=eval_prompt, system_prompt=self.system_prompt)
            evaluation_result = eval_resp.content.strip()
            
            print(f"Evaluation result: {evaluation_result}")
            
            # Parse the score and explanation
            score, explanation, fulfillment_status = self._parse_evaluation_score(evaluation_result)
            
            # Format the final response
            formatted_response = f"""----------------
Evaluation score: {score}
{explanation}
-----------------
FINAL RESULT
------------------
{agent_response}"""
            
            return formatted_response
            
        except Exception as e:
            print(f"Error during Evaluator LLM interaction: {e}")
            return f"Error: LLM interaction failed during evaluation: {e}"
