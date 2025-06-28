from typing import Optional, Dict, List, Any, Callable
from abc import ABC, abstractmethod

from taskflow.util import logger
from taskflow.llm import LLMClient
from taskflow.models import UserNotApprovedException
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
        if not self.available_tools:
            return []
        return [tool.get_schema() for tool in self.available_tools.values()]

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
    This agent uses the LLM to decide what action to take based on the user prompt.
    """

    def __init__(self, model: LLMClient, system_prompt: str, available_tools: Optional[Dict[str, Callable]] = None):
        super().__init__("Commiter", model, "Commits staged changes to git repository using provided commit message.", system_prompt, available_tools)

    def _get_tool_schemas(self) -> List[Dict]:
        """Returns the tool schemas available to the commiter agent."""
        if not self.available_tools:
            return []
        return [tool.get_schema() for tool in self.available_tools.values()]

    def run(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Processes the user prompt and decides what action to take using the LLM.
        
        The agent will:
        1. Pass the user prompt to the LLM
        2. Let the LLM decide what tool to call (if any)
        3. Execute the tool if a function call is made
        4. Return the result
        
        Parameters:
            prompt: The user prompt that may contain commit instructions.
            **kwargs: Additional keyword arguments (for compatibility).

        Returns:
            A dictionary containing the result of the operation.
        """
        print(f"Commiter agent running with prompt: {prompt[:100]}...")

        try:
            # Let the LLM decide what to do with the prompt
            tools = self._get_tool_schemas()
            resp = self.model.chat(prompt=prompt, system_prompt=self.system_prompt, tools=tools)
            
            logger.info("-"*50)
            logger.info(f"LLM Response: {resp}")
            logger.info("-"*50)
            
            # If the LLM decided to call a function, execute it
            if resp.function_call:
                print(f"LLM decided to call function: {resp.function_call.name}")
                
                # Handle commit operations with user approval
                if resp.function_call.name == "commit_tool":
                    # Extract commit message from function arguments for approval
                    commit_message = resp.function_call.args.get("message", "No commit message provided")
                    
                    # Ask for user approval
                    approval = input(f"\n{'-'*80}\n{commit_message}\n{'-'*80}\n\nCan I commit the staged changes with this message? [y/N] ")
                    if approval.strip().lower() != "y":
                        raise UserNotApprovedException("User did not approve the commit")
                
                # Execute the function call
                result = self._execute_function_call(resp.function_call)
                
                # Format the response based on the function called
                if resp.function_call.name == "commit_tool":
                    if "Successfully committed" in str(result):
                        print("✓ Changes committed successfully!")
                        return {
                            "message": resp.function_call.args.get("message", ""),
                            "commit_result": result,
                            "committed": True,
                            "error": False
                        }
                    else:
                        print("✗ Commit failed")
                        return {
                            "message": resp.function_call.args.get("message", ""),
                            "commit_result": result,
                            "committed": False,
                            "error": True
                        }
                else:
                    # For other tools, return the result directly
                    return {
                        "message": f"Executed {resp.function_call.name}",
                        "result": result,
                        "error": False
                    }
            
            # If no function call was made, return the LLM's text response
            else:
                print("LLM provided a text response without function calls")
                return {
                    "message": resp.content,
                    "error": False
                }

        except UserNotApprovedException as e:
            print(f"Operation cancelled by user: {e}")
            return {
                "message": f"Operation cancelled: {e}",
                "error": True
            }
        except NoChangesStaged as e:
            print(f"No changes staged: {e}")
            raise  # Re-raise this specific exception
        except Exception as e:
            print(f"Error during Commiter execution: {e}")
            return {
                "message": f"Execution failed: {e}",
                "error": True
            }

