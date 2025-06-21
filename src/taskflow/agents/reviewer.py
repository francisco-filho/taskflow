from typing import Optional, Dict, List, Callable

from taskflow.llm import LLMClient
from taskflow.tools import DIFF_TOOL_SCHEMA
from taskflow.agents import Agent

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
