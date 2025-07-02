import json
import logging
import tempfile
import subprocess
from typing import Optional, Dict, List, Any, Callable
from abc import ABC, abstractmethod

from taskflow.util import printc
from taskflow.llm import LLMClient
from taskflow.exceptions import NoChangesStaged


class ToolExecutionNotAuthorized(Exception):
    """Exception raised when tool execution is not authorized by the user."""
    def __init__(self, tool_name: str, params: Dict[str, Any]):
        self.tool_name = tool_name
        self.params = params
        super().__init__(f"Tool '{tool_name}' execution not authorized. Params: {params}")


class Tool():
    name: str
    fn: Callable
    needs_approval: bool

    def __init__(self, name: str, fn: Callable, needs_approval=True):
        self.name = name
        self.fn = fn
        self.needs_approval = needs_approval

    def _edit_parameters(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Opens vim to edit parameters and returns the modified parameters.
        """
        try:
            # Create a temporary file with the current parameters
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
                json.dump(kwargs, temp_file, indent=2, ensure_ascii=False)
                temp_file_path = temp_file.name

            # Open vim with the temporary file
            subprocess.run(['vim', temp_file_path], check=True)

            # Read the modified content
            with open(temp_file_path, 'r') as temp_file:
                edited_content = temp_file.read()
                edited_kwargs = json.loads(edited_content)

            # Clean up the temporary file
            import os
            os.unlink(temp_file_path)

            return edited_kwargs

        except subprocess.CalledProcessError:
            print("Error: Could not open vim editor.")
            return kwargs
        except json.JSONDecodeError:
            print("Error: Invalid JSON format after editing. Using original parameters.")
            return kwargs
        except Exception as e:
            print(f"Error during editing: {e}. Using original parameters.")
            return kwargs

    def __call__(self, **kwargs):
        if self.needs_approval:
            while True:
                print("-"*80)
                printc(f"Tool: [blue]{self.name}[/blue]")
                print("Parameters:")
                #printc(kwargs)
                for param, value in kwargs.items():
                    printc(f"{param}: [green]{value}[/green]")
                print("-"*80)
                printc("[red]Do you approve the execution of the tool above?[/red] (y/n/e): ", end="")
                
                try:
                    user_input = input().strip().lower()
                    if user_input in ['y', 'yes']:
                        break
                    elif user_input in ['e', 'edit']:
                        kwargs = self._edit_parameters(kwargs)
                        continue
                    else:
                        raise ToolExecutionNotAuthorized(self.name, kwargs)
                    # user_input = input().strip().lower()
                    # if user_input not in ['y', 'yes']:
                    #     raise ToolExecutionNotAuthorized(self.name, kwargs)
                except (EOFError, KeyboardInterrupt):
                    # Handle cases where input might not be available
                    raise ToolExecutionNotAuthorized(self.name, kwargs)
                except ToolExecutionNotAuthorized as e:
                    raise
                    
            
            print("-" * 80)

        return self.fn(**kwargs)


class Agent(ABC):
    """
    Abstract base class for AI agents.
    """
    def __init__(self, name: str, model: LLMClient, description: str, system_prompt: str, available_tools: Optional[Dict[str, Tool]] = None):
        """
        Initializes an agent.

        Parameters:
            name: The name of the agent.
            model: An instance of LLMClient (e.g., GeminiClient) for LLM interactions.
            description: A brief description of what the agent does.
            system_prompt: The system-level prompt to guide the agent's LLM behavior.
            available_tools: A dictionary mapping tool names to Tool instances.
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
            tool = self.available_tools[function_name]
            result = tool(**function_args)
            return result
        except NoChangesStaged as e:
            raise
        except ToolExecutionNotAuthorized as e:
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
        schemas = []
        for tool in self.available_tools.values():
            if hasattr(tool.fn, 'get_schema'):
                schemas.append(tool.fn.get_schema())
        return schemas

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
