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

class Evaluator(Agent):
    """
    A general-purpose agent responsible for evaluating whether a previous agent's response
    fulfills the user's original request.
    """
    def __init__(self, model: LLMClient, system_prompt: str, available_tools: Optional[Dict[str, Callable]] = None):
        super().__init__("Evaluator", model, "Evaluates whether previous agent responses fulfill user requests.", system_prompt, available_tools)

    def _get_tool_schemas(self) -> List[Dict]:
        """Returns the tool schemas available to the evaluator agent."""
        # The evaluator might need access to tools like diff_tool depending on the evaluation context
        return []

    def _extract_user_request_and_response(self, prompt: str) -> tuple[str, str]:
        """
        Extracts the original user request and the previous agent's response from the prompt.
        
        Parameters:
            prompt: The prompt containing both the user request and the response to evaluate.
            
        Returns:
            A tuple of (user_request, agent_response)
        """
        # Look for common patterns that separate user request from agent response
        separators = [
            "Agent Response:",
            "Previous Step Result:",
            "Step Result:",
            "Response:",
            "Output:",
            "Result:",
        ]
        
        user_request = ""
        agent_response = ""
        
        # Try to find the original task/request
        if "Original Task:" in prompt:
            start_idx = prompt.find("Original Task:") + len("Original Task:")
            # Find the end of the original task (usually before agent response)
            end_markers = separators + ["Step Context:", "Previous Step Results:"]
            end_idx = len(prompt)
            
            for marker in end_markers:
                marker_idx = prompt.find(marker, start_idx)
                if marker_idx != -1 and marker_idx < end_idx:
                    end_idx = marker_idx
            
            user_request = prompt[start_idx:end_idx].strip()
        
        # Try to find the agent response
        for separator in separators:
            if separator in prompt:
                start_idx = prompt.find(separator) + len(separator)
                # Take everything after the separator as the response
                agent_response = prompt[start_idx:].strip()
                break
        
        # If no clear separation found, try to parse JSON responses
        if not agent_response:
            try:
                import json
                import re
                
                # Look for JSON-like structures that might be agent responses
                json_pattern = r'\{[^{}]*"message"[^{}]*\}'
                json_matches = re.findall(json_pattern, prompt, re.DOTALL)
                
                if json_matches:
                    agent_response = json_matches[-1]  # Take the last JSON match
                
                # Also look for larger JSON blocks
                json_block_pattern = r'\{.*?\}'
                json_blocks = re.findall(json_block_pattern, prompt, re.DOTALL)
                
                if json_blocks and not agent_response:
                    agent_response = json_blocks[-1]  # Take the last JSON block
                    
            except Exception:
                pass
        
        # If still no response found, take the latter part of the prompt
        if not agent_response and user_request:
            # Everything after the user request could be the response
            remaining = prompt.replace(f"Original Task: {user_request}", "").strip()
            if remaining:
                agent_response = remaining
        elif not user_request and not agent_response:
            # Fallback: split the prompt in half
            lines = prompt.split('\n')
            mid_point = len(lines) // 2
            user_request = '\n'.join(lines[:mid_point]).strip()
            agent_response = '\n'.join(lines[mid_point:]).strip()
        
        return user_request, agent_response

    def _determine_evaluation_context(self, user_request: str, agent_response: str) -> str:
        """
        Determines what type of evaluation is needed based on the user request.
        
        Parameters:
            user_request: The original user request.
            agent_response: The agent's response to evaluate.
            
        Returns:
            A string indicating the evaluation context or empty string if no special context needed.
        """
        request_lower = user_request.lower()
        
        # Check if this involves code changes and might need diff context
        if any(keyword in request_lower for keyword in [
            'commit', 'diff', 'changes', 'staged', 'git', 'repository', 'code review'
        ]):
            project_dir = self._extract_project_dir(user_request + " " + agent_response)
            if project_dir:
                return f"code_changes:{project_dir}"
        
        return ""

    def _get_additional_context(self, evaluation_context: str) -> str:
        """
        Gets additional context needed for evaluation (e.g., git diff for code-related tasks).
        
        Parameters:
            evaluation_context: The context type and parameters.
            
        Returns:
            Additional context string or empty string if none needed.
        """
        if evaluation_context.startswith("code_changes:"):
            project_dir = evaluation_context.split(":", 1)[1]
            
            # Get the diff to provide context for code-related evaluations
            diff_prompt = f"Get the diff of staged changes for the project directory: {project_dir}"
            
            try:
                tools = self._get_tool_schemas()
                resp = self.model.chat(prompt=diff_prompt, system_prompt=self.system_prompt, tools=tools)
                
                if resp.function_call and resp.function_call.name == "diff_tool":
                    diff_result = self._execute_function_call(resp.function_call)
                    return f"\nGit Diff Context:\n```diff\n{diff_result}\n```"
                
            except Exception as e:
                print(f"Warning: Could not get diff context: {e}")
        
        return ""

    def run(self, prompt: str, **kwargs) -> str:
        """
        Evaluates whether a previous agent's response fulfills the user's original request.
        The prompt should contain both the user request and the response to be evaluated.

        Parameters:
            prompt: The prompt containing the user request and the previous agent's response.
            **kwargs: Additional keyword arguments (for compatibility).

        Returns:
            A string indicating if the request was fulfilled or not, with reasoning.
        """
        print(f"Evaluator agent running with prompt: {prompt[:100]}...")

        # Extract the user request and agent response from the prompt
        user_request, agent_response = self._extract_user_request_and_response(prompt)
        
        if not user_request:
            return "Error: Could not extract user request from prompt."
        
        if not agent_response:
            return "Error: Could not extract agent response from prompt."
        
        print(f"Extracted user request: {user_request[:100]}...")
        print(f"Extracted agent response: {agent_response[:100]}...")
        
        # Determine if we need additional context for evaluation
        evaluation_context = self._determine_evaluation_context(user_request, agent_response)
        additional_context = self._get_additional_context(evaluation_context)
        
        # Create the evaluation prompt
        eval_prompt = f"""You are evaluating whether an agent's response answers the user request.

Original User Request:
```
{user_request}
```

Agent's Response:
```
{agent_response}
```
{additional_context}

Evaluate whether the agent's response adequately fulfills the user's original request. Do not take in consideration the user's request for evaluation.

IMPORTANT EVALUATION CRITERIA:
- If the only problem with the previous agent's response is that it does not include evaluation, accept as fulfilled
- If the user requested a evaluation, do the evaluation NOW, do not reject the 'agent_response' if it does not include a evaluation
- You should not reject a response because a lack of the previous agent evaluation, YOU will do the evaluation
- Does the response directly address what the user asked for?
- If the user requested an action (like committing changes), was the action actually performed?
- If the user requested information (like a review or diff), was the information provided?
- If the user requested generation of content (like a commit message), was the content generated?
- Are there any obvious gaps between what was requested and what was delivered?

Respond with either:
1. "REQUEST FULFILLED\n\n{agent_response}" if the agent's response adequately addresses the user's request
2. "REQUEST NOT FULFILLED: [specific reason]" if the request was not adequately fulfilled (ignore evaluation)

Be specific about what is missing or what needs to be done if the request was not fulfilled."""
        logger.info("-"*80)
        logger.info(eval_prompt)
        logger.info("-"*80)

        try:
            # Get the evaluation from the LLM
            eval_resp = self.model.chat(prompt=eval_prompt, system_prompt=self.system_prompt)
            evaluation_result = eval_resp.content.strip()
            
            print(f"Evaluation result: {evaluation_result}")
            return evaluation_result
            
        except Exception as e:
            print(f"Error during Evaluator LLM interaction: {e}")
            return f"Error: LLM interaction failed during evaluation: {e}"

