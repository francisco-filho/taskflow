import json
from typing import Optional, Dict, List, Any, Callable
from abc import ABC, abstractmethod

from pydantic import BaseModel

from taskflow.util import logger
from taskflow.llm import LLMClient
from taskflow.models import CommitMessage
from taskflow.tools import DIFF_TOOL_SCHEMA, COMMIT_TOOL_SCHEMA

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

    def _execute_function_call(self, function_call) -> str:
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
            return str(result)
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

class DiffMessager(Agent):
    """
    An agent responsible for generating commit messages based on project changes.
    This agent analyzes diffs and creates appropriate commit messages.
    """

    def __init__(self, model: LLMClient, system_prompt: str, available_tools: Optional[Dict[str, Callable]] = None):
        super().__init__("DiffMessager", model, "Generates commit messages from code diffs by analyzing staged changes.", system_prompt, available_tools)

    def _get_tool_schemas(self) -> List[Dict]:
        """Returns the tool schemas available to the diff messager agent."""
        return [DIFF_TOOL_SCHEMA]

    def run(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generates a commit message based on staged changes.
        
        The agent will:
        1. Get the diff of staged changes
        2. Generate an appropriate commit message based on the changes
        
        Parameters:
            prompt: The user prompt containing the request and project information.
            **kwargs: Additional keyword arguments (for compatibility).

        Returns:
            A dictionary containing the generated commit message and details.
        """
        print(f"DiffMessager agent running with prompt: {prompt[:100]}...")

        project_dir = self._extract_project_dir(prompt)
        if not project_dir:
            return {
                "message": "Error: Could not extract project directory from prompt", 
                "details": ["Please specify the project directory in your prompt"],
                "error": True
            }

        try:
            # Step 1: Get the diff of staged changes
            diff_result = self._get_diff(project_dir)
            if "Error" in diff_result:
                return {
                    "message": "Error getting diff", 
                    "details": [diff_result],
                    "error": True
                }

            # Step 2: Generate commit message based on diff
            commit_message_data = self._generate_commit_message(prompt, diff_result)
            if commit_message_data.get("error"):
                return commit_message_data

            print("✓ Commit message generated successfully!")
            return commit_message_data

        except Exception as e:
            print(f"Error during DiffMessager execution: {e}")
            return {
                "message": f"Execution failed: {e}", 
                "details": [],
                "error": True
            }

    def _get_diff(self, project_dir: str) -> str:
        """
        Gets the diff of staged changes for the project.
        
        Parameters:
            project_dir: The project directory path.
            
        Returns:
            The diff result as a string.
        """
        diff_prompt = f"Get the diff of staged changes for the project directory: {project_dir}"
        
        try:
            tools = self._get_tool_schemas()
            resp = self.model.chat(prompt=diff_prompt, system_prompt=self.system_prompt, tools=tools)
            
            if resp.function_call and resp.function_call.name == "diff_tool":
                diff_result = self._execute_function_call(resp.function_call)
                return diff_result
            else:
                return "Error: Failed to get diff - no function call made"
                
        except Exception as e:
            return f"Error: Failed to get diff - {e}"

    def _generate_commit_message(self, original_prompt: str, diff_result: str) -> Dict[str, Any]:
        """
        Generates a commit message based on the diff result.
        
        Parameters:
            original_prompt: The original user prompt.
            diff_result: The git diff output.
            
        Returns:
            A dictionary with the commit message and details.
        """
        commit_prompt = f"""Based on the user request: "{original_prompt}"

And the following git diff:

```diff
{diff_result}
```

Generate a commit message in the specified JSON format with a message and a detailed list of changes."""

        try:
            # Generate commit message
            commit_resp = self.model.chat(prompt=commit_prompt, system_prompt=self.system_prompt, output=CommitMessage)
            message_content = commit_resp.content

            try:
                parsed_json = json.loads(message_content)
                if isinstance(parsed_json, dict) and "message" in parsed_json and "details" in parsed_json:
                    return parsed_json
                else:
                    print(f"Warning: LLM response was not in expected JSON format. Raw: {message_content}")
                    return {
                        "message": "Invalid format from LLM", 
                        "details": [message_content],
                        "error": True
                    }
            except json.JSONDecodeError:
                print(f"Warning: LLM response was not valid JSON. Raw: {message_content}")
                return {
                    "message": "Invalid JSON response from LLM", 
                    "details": [message_content],
                    "error": True
                }
                
        except Exception as e:
            return {
                "message": f"Failed to generate commit message: {e}", 
                "details": [],
                "error": True
            }

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
                            formatted_message += '\n' + '\n'.join([f"• {detail}" for detail in details])
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
            commit_result = self._perform_commit(project_dir, commit_message)
            
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
    An agent responsible for evaluating the quality of commit messages.
    """
    def __init__(self, model: LLMClient, system_prompt: str, available_tools: Optional[Dict[str, Callable]] = None):
        super().__init__("Evaluator", model, "Evaluates the quality of commit messages.", system_prompt, available_tools)

    def _get_tool_schemas(self) -> List[Dict]:
        """Returns the tool schemas available to the evaluator agent."""
        return [DIFF_TOOL_SCHEMA]

    def run(self, prompt: str, commit_message: Optional[Dict[str, Any]] = None, **kwargs) -> str:
        """
        Evaluates a given commit message against project changes using function calling.

        Parameters:
            prompt: The user prompt containing the request and project information.
            commit_message: The commit message (as a dictionary) to evaluate. If not provided, 
                          it will be extracted from the prompt.
            **kwargs: Additional keyword arguments (for compatibility).

        Returns:
            A string indicating if the commit message is accepted or rejected with a reason.
        """
        print(f"Evaluator agent running with prompt: {prompt[:100]}...")

        project_dir = self._extract_project_dir(prompt)
        if not project_dir:
            return "Error: Could not extract project directory from prompt. Please specify the project directory."

        if commit_message is None:
            return "Error: No commit message provided for evaluation."

        message_str = f"Message: {commit_message.get('message', '')}\nDetails:\n" + \
                      "\n".join([f"- {d}" for d in commit_message.get('details', [])])

        # Create a focused prompt for getting the diff
        diff_prompt = f"Get the diff of staged changes for the project directory: {project_dir}"

        try:
            # First call with function calling enabled to get the diff
            tools = self._get_tool_schemas()
            resp = self.model.chat(prompt=diff_prompt, system_prompt=self.system_prompt, tools=tools)

            # Check if the model wants to call a function
            if resp.function_call:
                function_result = self._execute_function_call(resp.function_call)

                # Now ask for the evaluation with the diff result and original context
                eval_prompt = f"""User request: "{prompt}"

Commit Message to evaluate:
```
{message_str}
```

Git Diff:
```diff
{function_result}
```

Evaluate the quality of the commit message against the changes shown in the diff.
If your evaluation is positive, respond with 'Commit message accepted'.
If the commit message has any problems, respond with 'Bad commit message', two new lines, and the reason."""

                # Second call to get the actual evaluation
                eval_resp = self.model.chat(prompt=eval_prompt, system_prompt=self.system_prompt)
                return eval_resp.content
            else:
                # If no function call, treat as direct response
                return resp.content

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