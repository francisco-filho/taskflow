from typing import Optional, Dict, List, Any, Callable

from taskflow.util import logger
from taskflow.llm import LLMClient
from taskflow.agents import Agent, Tool
from taskflow.exceptions import NoChangesStaged, ToolExecutionNotAuthorized


def _result(commit_message: str, error=False) -> Dict[str, Any]:
    return {
        "message": commit_message,
        "error": error}


class DiffMessager(Agent):
    """
    An agent responsible for generating commit messages based on project changes.
    This agent analyzes diffs and creates appropriate commit messages.
    """

    def __init__(self, model: LLMClient, system_prompt: str, available_tools: Optional[Dict[str, Tool]] = None):
        super().__init__("DiffMessager", model, "Generates commit messages from code diffs by analyzing staged changes.", system_prompt, available_tools)

    def _get_tool_schemas(self) -> List[Dict]:
        """Returns the tool schemas available to the diff messager agent."""
        if not self.available_tools:
            return []
        schemas = []
        for tool in self.available_tools.values():
            if hasattr(tool.fn, 'get_schema'):
                schemas.append(tool.fn.get_schema())
        return schemas

    def run(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generates a commit message based on diff changes.
        
        The agent will:
        1. Check if diff is already provided in the prompt
        2. If not, get the diff of staged changes using available tools
        3. If the user asks to create message for remote repo, use available tools
        4. Generate an appropriate commit message based on the changes
        
        Parameters:
            prompt: The user prompt containing the request and project information.
            **kwargs: Additional keyword arguments (for compatibility).

        Returns:
            A dictionary containing the generated commit message and details.
        """
        print(f"DiffMessager agent running with prompt: {prompt[:100]}...")

        try:
            # Check if diff is already provided in the prompt
            if "```diff" in prompt.lower() or "diff --git" in prompt:
                print("Diff found in prompt, skipping tool call...")
                cm = self._generate_commit_message_from_prompt(prompt)
                return _result(cm)

            # Step 1: Get the diff of staged changes using LLM tool calling
            diff_result = self._get_diff(prompt)
            if diff_result.startswith("Error: ") or diff_result.startswith("Warning: "):
                return _result(f"Error getting diff: => {diff_result}", True)
            # Step 2: Generate commit message based on diff
            commit_message_data = self._generate_commit_message(prompt, diff_result)

            print("✓ Commit message generated successfully!")
            return _result(commit_message_data)

        except NoChangesStaged:
            raise
        except ToolExecutionNotAuthorized as e:
            raise
        except Exception as e:
            print(f"Error during DiffMessager execution: {e}")
            return _result(f"Execution failed: {e}", True)

    def _get_diff(self, prompt: str) -> str:
        """
        Gets the diff of staged changes by delegating to the LLM with tool calling.
        
        Parameters:
            prompt: The original user prompt containing project information.
            
        Returns:
            The diff result as a string.
        """
        try:
            tools = self._get_tool_schemas()
            resp = self.model.chat(prompt=prompt, system_prompt=self.system_prompt, tools=tools)
            
            if resp.function_call:
                diff_result = self._execute_function_call(resp.function_call)
                return diff_result
            else:
                return "Error: Failed to get diff - no function call made"
                
        except NoChangesStaged as e:
            logger.error(e)
            raise
        except Exception as e:
            logger.error(e)
            raise

    def _generate_commit_message_from_prompt(self, prompt: str) -> str:
        """
        Generates a commit message when diff is already provided in the prompt.
        
        Parameters:
            prompt: The user prompt containing both the request and the diff.
            
        Returns:
            A dictionary with the commit message and details.
        """
        commit_prompt = f"""Based on the user request and diff provided in the following prompt:

"{prompt}"

Generate a commit message in the specified format with a message and a detailed list of changes"""

        try:
            # Generate commit message
            commit_resp = self.model.chat(prompt=commit_prompt, system_prompt=self.system_prompt)
            message_content = commit_resp.content
            return message_content

            # try:
            #     parsed_json = json.loads(message_content)
            #     if isinstance(parsed_json, dict) and "message" in parsed_json and "details" in parsed_json:
            #         return parsed_json
            #     else:
            #         print(f"Warning: LLM response was not in expected JSON format. Raw: {message_content}")
            #         return {
            #             "message": "Invalid format from LLM", 
            #             "details": [message_content],
            #             "error": True
            #         }
            # except json.JSONDecodeError:
            #     print(f"Warning: LLM response was not valid JSON. Raw: {message_content}")
            #     return {
            #         "message": "Invalid JSON response from LLM", 
            #         "details": [message_content],
            #         "error": True
            #     }
                
        except Exception as e:
            return _result(f"Failed to generate commit message: {e}", True)

    def _generate_commit_message(self, original_prompt: str, diff_result: str) -> str:
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

Generate a commit message in the following format, text only, be very succint:

{{Commit message, focusing in the overall changes}}

- {{Detail 1 about the changes}}
- {{Detail 2 about the changes}}
... repeate if necessary

It could have as much detail lines as seen necessary

"""

        try:
            # Generate commit message
            commit_resp = self.model.chat(prompt=commit_prompt, system_prompt=self.system_prompt)
            message_content = commit_resp.content

            return message_content

            # try:
            #     parsed_json = json.loads(message_content)
            #     if isinstance(parsed_json, dict) and "message" in parsed_json and "details" in parsed_json:
            #         return parsed_json
            #     else:
            #         print(f"Warning: LLM response was not in expected JSON format. Raw: {message_content}")
            #         return {
            #             "message": "Invalid format from LLM", 
            #             "details": [message_content],
            #             "error": True
            #         }
            # except json.JSONDecodeError:
            #     print(f"Warning: LLM response was not valid JSON. Raw: {message_content}")
            #     return {
            #         "message": "Invalid JSON response from LLM", 
            #         "details": [message_content],
            #         "error": True
            #     }
                
        except Exception as e:
            return f"Failed to generate commit message: {e}"
