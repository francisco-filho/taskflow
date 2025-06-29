from typing import Optional, Dict, List, Any, Callable

from taskflow.util import logger
from taskflow.llm import LLMClient
from taskflow.models import UserNotApprovedException
from taskflow.agents import Agent
from taskflow.exceptions import NoChangesStaged

class Commiter(Agent):
    """
    An agent responsible for committing changes to a git repository.
    This agent uses the LLM to decide what action to take based on the user prompt.
    """

    def __init__(self, model: LLMClient, system_prompt: str, available_tools: Optional[Dict[str, Callable]] = None):
        sysp = system_prompt + "\nIf a project directory was not informed use the current directory as base for the git repository"
        super().__init__("Commiter", model, "Commits staged changes to git repository using provided commit message.", sysp, available_tools)

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
            2.1 - If the prompt contains the 'commit message' then format and do the commit
            2.2 - If the prompt does not conatins a commit message but contains a 'diff' generate the message base on the diff
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

