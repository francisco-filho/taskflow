from typing import Optional, Dict, List, Any, Callable
from abc import ABC, abstractmethod

from taskflow.util import logger
from taskflow.llm import LLMClient
from taskflow.models import UserNotApprovedException
from taskflow.tools import DIFF_TOOL_SCHEMA, COMMIT_TOOL_SCHEMA
from taskflow.exceptions import NoChangesStaged

class Agent(ABC):
    """
    Abstract base class for AI agents.
    """
    def __init__(self, name: str, model: LLMClient, description: str, system_prompt: str, available_tools: Optional[Dict[str, Callable]] = None):
        """
        Initializes an agent.

        Parameters:
            name: The name of the agent.
            model: An instance of LLMClient (e.g., GeminiClient) for LLM interactions.
            description: A brief description of what the agent does.
            system_prompt: The system-level prompt to guide the agent's LLM behavior.
            available_tools: A dictionary mapping tool names to their functions.
        """
        self.name = name
        self.model = model
        self.description = description
        self.system_prompt = system_prompt
        self.available_tools = available_tools if available_tools is not None else {}
        print(f"Agent '{self.name}' initialized with {len(self.available_tools)} available tools.")

    @abstractmethod
    def run(self, prompt: str, **kwargs) -> Any:
        """
        Abstract method to run the agent's specific logic.
        Each concrete agent must implement this.
        """
        pass

    def _execute_function_call(self, function_call):
        """
        Executes a function call returned by the LLM.

        Parameters:
            function_call: The function call object from the LLM response.

        Returns:
            The result of the function execution as a string.
        """
        function_name = function_call.name
        function_args = function_call.args

        if function_name not in self.available_tools:
            return f"Error: Function '{function_name}' not available for agent '{self.name}'."

        try:
            print(f"Executing function: {function_name} with args: {function_args}")
            result = self.available_tools[function_name](**function_args)
            return result
        except NoChangesStaged as e:
            raise
        except Exception as e:
            return f"Error executing function '{function_name}': {e}"

    def _get_tool_schemas(self) -> List[Dict]:
        """
        Returns the tool schemas for function calling.
        This should be overridden by subclasses to provide specific tool schemas.
        """
        return []

    def _extract_project_dir(self, prompt: str) -> str:
        """
        Helper method to extract project directory from prompt.
        This can be overridden by subclasses for different extraction logic.
        """
        project_dir_match = prompt.split("project '")
        if len(project_dir_match) > 1:
            return project_dir_match[1].split("'")[0]
        
        # Alternative patterns to match
        if "project directory:" in prompt.lower():
            parts = prompt.lower().split("project directory:")
            if len(parts) > 1:
                return parts[1].strip().split()[0]
        
        if "in " in prompt and ("/" in prompt or "\\" in prompt):
            # Try to find path-like strings
            words = prompt.split()
            for word in words:
                if "/" in word or "\\" in word:
                    return word.strip(".,!?")
        
        return ""


class Commiter(Agent):
    """
    An agent responsible for committing changes to a git repository.
    This agent expects to receive a commit message and performs the actual commit operation.
    """

    def __init__(self, model: LLMClient, system_prompt: str, available_tools: Optional[Dict[str, Callable]] = None):
        super().__init__("Commiter", model, "Commits staged changes to git repository using provided commit message.", system_prompt, available_tools)

    def _get_tool_schemas(self) -> List[Dict]:
        """Returns the tool schemas available to the commiter agent."""
        return [COMMIT_TOOL_SCHEMA]

    def _extract_commit_message(self, prompt: str) -> str:
        """
        Extracts the commit message from the prompt, including both message and details.
        
        Parameters:
            prompt: The prompt containing the commit message.
            
        Returns:
            The extracted commit message formatted with details.
        """
        # First try to parse as JSON (from DiffMessager output)
        try:
            import json
            import re
            
            # Look for JSON-like structure in the prompt
            json_pattern = r'\{[^{}]*"message"[^{}]*"details"[^{}]*\}'
            json_matches = re.findall(json_pattern, prompt, re.DOTALL)
            
            for json_str in json_matches:
                try:
                    commit_data = json.loads(json_str)
                    if isinstance(commit_data, dict) and "message" in commit_data:
                        message = commit_data["message"]
                        details = commit_data.get("details", [])
                        
                        if details:
                            # Format message with details
                            formatted_message = message
                            if not message.endswith('\n'):
                                formatted_message += '\n'
                            formatted_message += '\n' + '\n'.join([f"{detail}" for detail in details])
                            return formatted_message
                        else:
                            return message
                except json.JSONDecodeError:
                    continue
                    
            # If we find a larger JSON block, try to parse it
            json_block_pattern = r'\{.*?"message".*?"details".*?\}'
            json_blocks = re.findall(json_block_pattern, prompt, re.DOTALL)
            
            for json_block in json_blocks:
                try:
                    commit_data = json.loads(json_block)
                    if isinstance(commit_data, dict) and "message" in commit_data:
                        message = commit_data["message"]
                        details = commit_data.get("details", [])
                        
                        if details:
                            # Format message with details
                            formatted_message = message
                            if not message.endswith('\n'):
                                formatted_message += '\n'
                            formatted_message += '\n' + '\n'.join([f"• {detail}" for detail in details])
                            return formatted_message
                        else:
                            return message
                except json.JSONDecodeError:
                    continue
                    
        except Exception:
            pass
        
        # Fallback to original extraction patterns for simple messages
        patterns = [
            'commit message: "',
            'commit message:"',
            'message: "',
            'message:"',
            '"message":',
            'commit with message "',
            'commit: "',
        ]
        
        prompt_lower = prompt.lower()
        
        for pattern in patterns:
            if pattern in prompt_lower:
                start_idx = prompt_lower.find(pattern) + len(pattern)
                if pattern.endswith('"'):
                    # Find the closing quote
                    end_idx = prompt.find('"', start_idx)
                    if end_idx != -1:
                        return prompt[start_idx:end_idx].strip()
                else:
                    # Extract until end of line or sentence
                    rest = prompt[start_idx:].strip()
                    if rest.startswith('"'):
                        # Remove opening quote and find closing quote
                        rest = rest[1:]
                        end_idx = rest.find('"')
                        if end_idx != -1:
                            return rest[:end_idx].strip()
                    else:
                        # Take until newline or period
                        for end_char in ['\n', '.', '!', '?']:
                            end_idx = rest.find(end_char)
                            if end_idx != -1:
                                return rest[:end_idx].strip()
                        return rest.strip()
        
        # If no specific pattern found, try to extract JSON-like message
        try:
            if '"message"' in prompt:
                start = prompt.find('"message"')
                colon_idx = prompt.find(':', start)
                if colon_idx != -1:
                    rest = prompt[colon_idx + 1:].strip()
                    if rest.startswith('"'):
                        rest = rest[1:]
                        end_idx = rest.find('"')
                        if end_idx != -1:
                            return rest[:end_idx].strip()
        except:
            pass
        
        return ""

    def run(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Commits changes using the provided commit message.
        
        The agent will:
        1. Extract the commit message from the prompt
        2. Extract the project directory
        3. Perform the commit using the commit tool
        
        Parameters:
            prompt: The user prompt containing the commit message and project information.
            **kwargs: Additional keyword arguments (for compatibility).

        Returns:
            A dictionary containing the commit result.
        """
        print(f"Commiter agent running with prompt: {prompt[:100]}...")

        project_dir = self._extract_project_dir(prompt)
        if not project_dir:
            return {
                "message": "Error: Could not extract project directory from prompt", 
                "details": ["Please specify the project directory in your prompt"],
                "error": True
            }

        commit_message = self._extract_commit_message(prompt)
        if not commit_message:
            return {
                "message": "Error: Could not extract commit message from prompt", 
                "details": ["Please provide a commit message in your prompt"],
                "error": True
            }

        print(f"Extracted commit message: {commit_message}")
        print(f"Project directory: {project_dir}")

        try:
            # Perform the commit
            # Try to generalize this to all agents
            approval = input(f"\n{'-'*80}\n{commit_message}\n{'-'*80}\n\nCan i commit the stagged changes with this message? [y/N] ")
            if "y" == approval.strip():
                commit_result = self._perform_commit(project_dir, commit_message)
            else:
                raise UserNotApprovedException("User did not approve the message")
            
            if "Successfully committed" in commit_result:
                print("✓ Changes committed successfully!")
                return {
                    "message": commit_message,
                    "commit_result": commit_result,
                    "committed": True,
                    "error": False
                }
            else:
                print("✗ Commit failed")
                return {
                    "message": commit_message,
                    "commit_result": commit_result,
                    "committed": False,
                    "error": True
                }

        except Exception as e:
            print(f"Error during Commiter execution: {e}")
            return {
                "message": f"Execution failed: {e}", 
                "details": [],
                "error": True
            }

    def _perform_commit(self, project_dir: str, commit_message: str) -> str:
        """
        Performs the actual commit using the commit tool.
        
        Parameters:
            project_dir: The project directory path.
            commit_message: The commit message to use.
            
        Returns:
            The result of the commit operation.
        """
        commit_prompt = f"""Commit the staged changes in project directory '{project_dir}' with the following commit message:

"{commit_message}"

Use the commit_tool to perform the actual commit."""

        try:
            tools = self._get_tool_schemas()
            resp = self.model.chat(prompt=commit_prompt, system_prompt=self.system_prompt, tools=tools)
            logger.info("-"*50)
            logger.info(resp)
            logger.info("-"*50)
            
            if resp.function_call and resp.function_call.name == "commit_tool":
                commit_result = self._execute_function_call(resp.function_call)
                return commit_result
            else:
                return "Error: Failed to commit - no function call made"
                
        except Exception as e:
            return f"Error: Failed to commit - {e}"

class Evaluator(Agent):
    """
    A general-purpose agent responsible for evaluating whether a previous agent's response
    fulfills the user's original request.
    """
    def __init__(self, model: LLMClient, system_prompt: str, available_tools: Optional[Dict[str, Callable]] = None):
        super().__init__("Evaluator", model, "Evaluates whether previous agent responses fulfill user requests.", system_prompt, available_tools)

    def _get_tool_schemas(self) -> List[Dict]:
        """Returns the tool schemas available to the evaluator agent."""
        # The evaluator might need access to tools like diff_tool depending on the evaluation context
        return [DIFF_TOOL_SCHEMA]

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
                import json
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

    def run(self, prompt: str, **kwargs) -> str:
        """
        Evaluates whether a previous agent's response fulfills the user's original request.
        The prompt should contain both the user request and the response to be evaluated.

        Parameters:
            prompt: The prompt containing the user request and the previous agent's response.
            **kwargs: Additional keyword arguments (for compatibility).

        Returns:
            A string indicating if the request was fulfilled or not, with reasoning.
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
        eval_prompt = f"""You are evaluating whether an agent's response fulfills a user's request.

Original User Request:
```
{user_request}
```

Agent's Response:
```
{agent_response}
```
{additional_context}

Evaluate whether the agent's response adequately fulfills the user's original request the only exception is if the user 'asks for evaluation', because this is your job

IMPORTANT EVALUATION CRITERIA:
- Does the response directly address what the user asked for?
- If the user requested an action (like committing changes), was the action actually performed?
- If the user requested information (like a review or diff), was the information provided?
- If the user requested generation of content (like a commit message), was the content generated?
- Are there any obvious gaps between what was requested and what was delivered?
- You should not reject a response because a lack of the previous agent evaluation, YOU will do the evaluation
- The evaluation is your job

Respond with either:
1. "REQUEST FULFILLED\n\n{agent_response}" if the agent's response adequately addresses the user's request
2. "REQUEST NOT FULFILLED: [specific reason]" if the request was not adequately fulfilled

Be specific about what is missing or what needs to be done if the request was not fulfilled."""

        try:
            # Get the evaluation from the LLM
            eval_resp = self.model.chat(prompt=eval_prompt, system_prompt=self.system_prompt)
            evaluation_result = eval_resp.content.strip()
            
            print(f"Evaluation result: {evaluation_result}")
            return evaluation_result
            
        except Exception as e:
            print(f"Error during Evaluator LLM interaction: {e}")
            return f"Error: LLM interaction failed during evaluation: {e}"

class Reviewer(Agent):
    """
    An agent responsible for generating a concise review of project changes.
    """
    def __init__(self, model: LLMClient, system_prompt: str, available_tools: Optional[Dict[str, Callable]] = None):
        super().__init__("Reviewer", model, "Generates a concise review of code changes.", system_prompt, available_tools)

    def _get_tool_schemas(self) -> List[Dict]:
        """Returns the tool schemas available to the reviewer agent."""
        return [DIFF_TOOL_SCHEMA]

    def run(self, prompt: str, **kwargs) -> str:
        """
        Generates a review of project changes using function calling.

        Parameters:
            prompt: The user prompt containing the request and project information.
            **kwargs: Additional keyword arguments (for compatibility).

        Returns:
            A string representing the review, or an error message.
        """
        print(f"Reviewer agent running with prompt: {prompt[:100]}...")

        # Extract project directory from prompt
        project_dir = self._extract_project_dir(prompt)
        if not project_dir:
            return "Error: Could not extract project directory from prompt. Please specify the project directory."

        # Create a focused prompt for getting the diff
        diff_prompt = f"Get the diff of staged changes for the project directory: {project_dir}"

        try:
            # First call with function calling enabled to get the diff
            tools = self._get_tool_schemas()
            resp = self.model.chat(prompt=diff_prompt, system_prompt=self.system_prompt, tools=tools)

            # Check if the model wants to call a function
            if resp.function_call:
                function_result = self._execute_function_call(resp.function_call)

                # Now ask for the review with the diff result and original context
                review_prompt = f"""User request: "{prompt}"

Git diff:
```diff
{function_result}
```

Generate a concise review of the changes based on the user's request and the diff shown above."""

                # Second call to get the actual review
                review_resp = self.model.chat(prompt=review_prompt, system_prompt=self.system_prompt)
                return review_resp.content
            else:
                # If no function call, treat as direct response
                return resp.content

        except Exception as e:
            print(f"Error during Reviewer LLM interaction: {e}")
            return f"Error: LLM interaction failed during review generation: {e}"