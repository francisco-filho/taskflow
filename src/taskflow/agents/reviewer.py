from typing import Optional, Dict, List, Callable
from pydantic import BaseModel

from taskflow.llm import LLMClient
from taskflow.agents import Agent


class Reviewer(Agent):
    """
    An agent responsible for generating a concise review of project changes.
    Works with any available diff tools (local or remote).
    """
    def __init__(self, model: LLMClient, system_prompt: str, available_tools: Optional[Dict[str, Callable]] = None):
        super().__init__("Reviewer", model, "Generates a concise review of code changes using available diff tools.", system_prompt, available_tools)

    def _get_tool_schemas(self) -> List[Dict]:
        """Returns the tool schemas available to the reviewer agent."""
        if not self.available_tools:
            return []
        return [tool.get_schema() for tool in self.available_tools.values()]

    def run(self, prompt: str, **kwargs) -> str:
        """
        Generates a review of project changes, it can use:
        - The 'diff' in the prompt, if available
        - The available diff tools when the prompt does not contain diff changes.
        Delegates the prompt directly to the LLM to choose the appropriate tool and arguments.

        Parameters:
            prompt: The user prompt containing the request, project information or diff changes.
            **kwargs: Additional keyword arguments (for compatibility).

        Returns:
            A string representing the review, or an error message.
        """
        print(f"Reviewer agent running with prompt: {prompt[:100]}...")

        # Check if we have any tools available
        if not self.available_tools:
            return "Error: No diff tools available. Please provide diff tools to the reviewer."

        try:
            # First call with function calling enabled to get the diff
            tools = self._get_tool_schemas()
            resp = self.model.chat(prompt=prompt, system_prompt=self.system_prompt, tools=tools)

            # Check if the model wants to call a function
            if resp.function_call:
                function_result = self._execute_function_call(resp.function_call)
                
                # Check if the result is an error
                if function_result.startswith("Error:") or function_result.startswith("Warning:"):
                    return f"Failed to get diff: {function_result}"

                # Now ask for the review with the diff result and original context
                review_prompt = f"""User request: "{prompt}"

Diff content:
```diff
{function_result}
```

Generate a concise review of the changes based on the user's request and the diff shown above."""

                # Second call to get the actual review
                review_resp = self.model.chat(prompt=review_prompt, system_prompt=self.system_prompt)
                return review_resp.content
            else:
                # If no function call, the LLM might have provided a direct response or explanation
                return resp.content

        except Exception as e:
            print(f"Error during Reviewer execution: {e}")
            return f"Error: Review generation failed: {e}"
