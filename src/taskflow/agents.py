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
    def run(self, **kwargs) -> Any:
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

    def run(self, project_dir: str) -> Dict[str, Any]:
        """
        Generates a commit message using function calling.

        Parameters:
            project_dir: The directory of the project to generate a diff from.

        Returns:
            A dictionary containing the generated commit message and details,
            or an error message if the process fails.
        """
        print(f"Commuter agent running for project: {project_dir}")

        full_prompt = f"Generate a commit message for the staged changes in the project directory: {project_dir}. First, get the diff of the staged changes, then create a commit message based on the diff."

        try:
            # First call with function calling enabled
            tools = self._get_tool_schemas()
            resp = self.model.chat(prompt=full_prompt, system_prompt=self.system_prompt, tools=tools)

            # Check if the model wants to call a function
            if resp.candidates[0].content.parts and hasattr(resp.candidates[0].content.parts[0], 'function_call'):
                function_call = resp.candidates[0].content.parts[0].function_call
                function_result = self._execute_function_call(function_call)

                # Now ask for the commit message with the diff result
                commit_prompt = f"Based on the following git diff, generate a commit message in the specified JSON format:\n\n```diff\n{function_result}\n```"

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
                # If no function call, treat as direct response
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

    def run(self, commit_message: Dict[str, Any], project_dir: str) -> str:
        """
        Evaluates a given commit message against project changes using function calling.

        Parameters:
            commit_message: The commit message (as a dictionary) to evaluate.
            project_dir: The directory of the project to generate a diff from.

        Returns:
            A string indicating if the commit message is accepted or rejected with a reason.
        """
        print(f"Evaluator agent running for commit message: {commit_message.get('message', 'N/A')}")

        message_str = f"Message: {commit_message.get('message', '')}\nDetails:\n" + \
                      "\n".join([f"- {d}" for d in commit_message.get('details', [])])

        full_prompt = f"Evaluate the quality of this commit message by first getting the diff of staged changes from project directory {project_dir}, then comparing it with the commit message:\n\nCommit Message:\n```\n{message_str}\n```"

        try:
            # First call with function calling enabled
            tools = self._get_tool_schemas()
            resp = self.model.chat(prompt=full_prompt, system_prompt=self.system_prompt, tools=tools)

            # Check if the model wants to call a function
            if resp.candidates[0].content.parts and hasattr(resp.candidates[0].content.parts[0], 'function_call'):
                function_call = resp.candidates[0].content.parts[0].function_call
                function_result = self._execute_function_call(function_call)

                # Now ask for the evaluation with the diff result
                eval_prompt = (
                    f"Given the following commit message and git diff, evaluate the quality of the commit message.\n"
                    f"Commit Message:\n```\n{message_str}\n```\n\n"
                    f"Git Diff:\n```diff\n{function_result}\n```\n\n"
                    f"If your evaluation is positive, respond with 'Commit message accepted'. "
                    f"If the commit message has any problems, respond with 'Bad commit message', "
                    f"two new lines, and the motive."
                )

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

    def run(self, project_dir: str) -> str:
        """
        Generates a review of project changes using function calling.

        Parameters:
            project_dir: The directory of the project to generate a diff from.

        Returns:
            A string representing the review, or an error message.
        """
        print(f"Reviewer agent running for project: {project_dir}")

        full_prompt = f"Generate a concise review of the staged changes in the project directory: {project_dir}. First, get the diff of the staged changes, then provide a review based on the diff."

        try:
            # First call with function calling enabled
            tools = self._get_tool_schemas()
            resp = self.model.chat(prompt=full_prompt, system_prompt=self.system_prompt, tools=tools)

            # Check if the model wants to call a function
            if resp.candidates[0].content.parts and hasattr(resp.candidates[0].content.parts[0], 'function_call'):
                function_call = resp.candidates[0].content.parts[0].function_call
                function_result = self._execute_function_call(function_call)

                # Now ask for the review with the diff result
                review_prompt = f"Given the following git diff, generate a concise review of the changes:\n\n```diff\n{function_result}\n```"

                # Second call to get the actual review
                review_resp = self.model.chat(prompt=review_prompt, system_prompt=self.system_prompt)
                return review_resp.candidates[0].content.parts[0].text
            else:
                # If no function call, treat as direct response
                return resp.candidates[0].content.parts[0].text

        except Exception as e:
            print(f"Error during Reviewer LLM interaction: {e}")
            return f"Error: LLM interaction failed during review generation: {e}"
