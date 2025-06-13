import json
from typing import Optional, Dict, List, Any, Callable
from abc import ABC, abstractmethod

from pydantic import BaseModel

from taskflow.llm import LLMClient
from taskflow.tools import DIFF_TOOL_SCHEMA

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

class CommitMessage(BaseModel):
    message: str
    details: list[str]

class Commiter(Agent):
    """
    An agent responsible for generating commit messages based on project changes.
    """

    def __init__(self, model: LLMClient, system_prompt: str, available_tools: Optional[Dict[str, Callable]] = None):
        super().__init__("Commiter", model, "Generates commit messages from code diffs.", system_prompt, available_tools)

    def _get_tool_schemas(self) -> List[Dict]:
        """Returns the tool schemas available to the commiter agent."""
        return [DIFF_TOOL_SCHEMA]

    def run(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generates a commit message using function calling.

        Parameters:
            prompt: The user prompt containing the request and project information.
            **kwargs: Additional keyword arguments (for compatibility).

        Returns:
            A dictionary containing the generated commit message and details,
            or an error message if the process fails.
        """
        print(f"Commiter agent running with prompt: {prompt[:100]}...")

        project_dir = self._extract_project_dir(prompt)
        if not project_dir:
            return {"message": "Error: Could not extract project directory from prompt", "details": ["Please specify the project directory in your prompt"]}

        diff_prompt = f"Get the diff of staged changes for the project directory: {project_dir}"

        try:
            # First call with function calling enabled to get the diff
            tools = self._get_tool_schemas()
            resp = self.model.chat(prompt=diff_prompt, system_prompt=self.system_prompt, tools=tools)

            # Check if the model wants to call a function
            if resp.candidates[0].content.parts and hasattr(resp.candidates[0].content.parts[0], 'function_call'):
                function_call = resp.candidates[0].content.parts[0].function_call
                function_result = self._execute_function_call(function_call)

                # Now ask for the commit message with the diff result and original prompt context
                commit_prompt = f"""Based on the user request: "{prompt}"

And the following git diff:

```diff
{function_result}
```

Generate a commit message in the specified JSON format with a concise message and detailed bullet points."""

                # Second call to get the actual commit message
                commit_resp = self.model.chat(prompt=commit_prompt, system_prompt=self.system_prompt, output=CommitMessage)
                message_content = commit_resp.candidates[0].content.parts[0].text

                try:
                    parsed_json = json.loads(message_content)
                    if isinstance(parsed_json, dict) and "message" in parsed_json and "details" in parsed_json:
                        return parsed_json
                    else:
                        print(f"Warning: LLM response for Commiter was not in expected JSON format. Raw: {message_content}")
                        return {"message": "Invalid format from LLM", "details": [message_content]}
                except json.JSONDecodeError:
                    print(f"Warning: LLM response for Commiter was not valid JSON. Raw: {message_content}")
                    return {"message": "Invalid JSON response from LLM", "details": [message_content]}
            else:
                # Direct response
                message_content = resp.candidates[0].content.parts[0].text
                return {"message": "No function call made", "details": [message_content]}

        except Exception as e:
            print(f"Error during Commiter LLM interaction: {e}")
            return {"message": f"LLM interaction failed: {e}", "details": []}

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
            if resp.candidates[0].content.parts and hasattr(resp.candidates[0].content.parts[0], 'function_call'):
                function_call = resp.candidates[0].content.parts[0].function_call
                function_result = self._execute_function_call(function_call)

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
                return eval_resp.candidates[0].content.parts[0].text
            else:
                # If no function call, treat as direct response
                return resp.candidates[0].content.parts[0].text

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
            if resp.candidates[0].content.parts and hasattr(resp.candidates[0].content.parts[0], 'function_call'):
                function_call = resp.candidates[0].content.parts[0].function_call
                function_result = self._execute_function_call(function_call)

                # Now ask for the review with the diff result and original context
                review_prompt = f"""User request: "{prompt}"

Git diff:
```diff
{function_result}
```

Generate a concise review of the changes based on the user's request and the diff shown above."""

                # Second call to get the actual review
                review_resp = self.model.chat(prompt=review_prompt, system_prompt=self.system_prompt)
                return review_resp.candidates[0].content.parts[0].text
            else:
                # If no function call, treat as direct response
                return resp.candidates[0].content.parts[0].text

        except Exception as e:
            print(f"Error during Reviewer LLM interaction: {e}")
            return f"Error: LLM interaction failed during review generation: {e}"