import json
from typing import Optional, Dict, List, Any, Callable
from abc import ABC, abstractmethod

from taskflow.util import logger
from taskflow.llm import LLMClient
from taskflow.models import CommitMessage
from taskflow.tools import DIFF_TOOL_SCHEMA
from taskflow.agents import Agent
from taskflow.exceptions import NoChangesStaged

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
            if diff_result.startswith("Error: "):
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

        except NoChangesStaged:
            raise
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
                
        except NoChangesStaged as e:
            logger.error(e)
            raise
        except Exception as e:
            logger.error(e)
            raise

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

Generate a commit message in the specified JSON format with a message and a detailed list of changes"""

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