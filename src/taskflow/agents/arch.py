from typing import Optional, Dict, List, Callable
import json
from pydantic import BaseModel

from taskflow.llm import LLMClient
from taskflow.agents import Agent


class ArchitectThinking(BaseModel):
    """
    Data model for tracking the architect's thinking process during plan creation.
    
    Attributes:
        done: Whether the architect has completed the planning process
        message: The current message or plan content
        error: Whether an error occurred during processing
    """
    done: bool = False
    message: str
    error: bool = False


class Architect(Agent):
    """
    Software architect agent that analyzes user requirements and creates detailed development plans.
    
    This agent specializes in breaking down complex software development requests into specific,
    actionable tasks that can be executed by coding agents. It uses available tools to gather
    information about existing project structure and files, then generates step-by-step instructions
    following predefined task formats.
    
    The architect focuses on planning and task decomposition rather than writing code directly.
    Each generated task specifies exactly what a coder should do, including which files to read,
    modify, or create.
    """
    def __init__(self, model: LLMClient, system_prompt: str, available_tools: Optional[Dict[str, Callable]] = None):
        """
        Initialize the Architect agent with LLM client and configuration.
        
        Args:
            model: LLM client instance for processing requests and generating responses
            system_prompt: Additional system prompt to append to the base architect prompt
            available_tools: Dictionary of tools the architect can use to gather project information
        """
        sp = f"""
You are a Software Architect and understand about development, design and architecture of complex systems. You
can use tools to understand the directory/file structure of the projects and read files to understand
how to implement the requirements of the user.
You create detailed plans step-by-step for the coders to implement. You do not need to write code, just the
instructions, files and say what the developers should do.

When you need more information about the project structure or file contents, use the available tools.
When you have enough information, create a detailed development plan with specific tasks.
         {system_prompt}"""
        super().__init__("Architect", model, """
        Do design and architecture of the software systems.  Understand what needs to be changed in a complex system.
        It Generate step by step instructions for the coders, including the filenames, classes and what should be
        implemented, but it does not write code. The archtecth can be only selected ONCE for user request, do not archtet more than one time.
        """, sp, available_tools)

    def run(self, prompt: str, **kwargs) -> Dict[str, any]:
        """
        Execute the architect's planning process for a given user request.
        
        This method implements an iterative approach where the architect:
        1. Analyzes the user request
        2. Uses available tools to gather necessary project information
        3. Creates a detailed development plan with specific tasks
        
        The process continues until either sufficient information is gathered to create
        a complete plan, or the maximum iteration limit is reached.
        
        Args:
            prompt: The user's development request or requirement
            **kwargs: Additional keyword arguments (currently unused)
            
        Returns:
            A dictionary containing:
            - message: The step-by-step development plan with specific tasks for coders
            - replan: Boolean indicating if planning was successful (True) or failed (False)
            
        Raises:
            Exception: If the planning process fails due to tool errors or LLM issues
        """
        print(f"Architect agent running with prompt: {prompt[:100]}...")

        try:
            tools = self._get_tool_schemas()
            done = False
            context = ""  # Store information gathered from tools
            max_iterations = 10  # Prevent infinite loops
            iteration = 0

            while not done and iteration < max_iterations:
                iteration += 1
                
                # Create the current prompt with context
                current_prompt = f"""
User Request: {prompt}

Context gathered so far:
{context}

IMPORTANT: Follow these exact task formats in your final response:

Example 1: Move a Method
User Request: Move the method 'calculate_total' from 'cart.py' to 'pricing.py'.
Agent Response:
* **Task 1:** Read the `calculate_total` method from `cart.py` and append its content to the `pricing.py` file.
* **Task 2:** Read `cart.py` and delete the `calculate_total` method from it.

Example 2: Add a New Function with an Import
User Request: Create a new function 'get_user_agent' in 'utils.py' that uses the 'request' object from Flask. Make sure the 'request' object is imported.
Agent Response:
* **Task 1:** Read `utils.py`, add `from flask import request` to the top of the file, and save the changes.
* **Task 2:** Read `utils.py` again, append the new function `def get_user_agent(): return request.headers.get('User-Agent')`, and save the file.

Instructions:
- If you need more information about the project structure or file contents, make a function call to the appropriate tool
- If you have enough information to create a detailed development plan, respond with a JSON object:
  {{"done": true, "message": "your detailed step-by-step plan with specific tasks for coders following the exact format above"}}

Do not explain the implementation or describe good practices, just do the plan with the tasks.

Your response should be either a function call OR the JSON object with your final plan.
"""

                print(f"Iteration {iteration}: Calling LLM...")
                resp = self.model.chat(prompt=current_prompt, system_prompt=self.system_prompt, tools=tools)
                
                # Check if LLM wants to call a function
                if hasattr(resp, 'function_call') and resp.function_call:
                    print(f"Executing function: {resp.function_call.name}")
                    function_result = self._execute_function_call(resp.function_call)
                    
                    #if function_result.startswith("Error:") or function_result.startswith("Warning:"):
                    #    return {"message": f"Failed to call function: {function_result}", "replan": False}
                    
                    # Add the function result to context
                    context += f"\n\nFunction {resp.function_call.name} result:\n{function_result}"
                    continue
                
                # Check if the response contains JSON indicating completion
                response_content = resp.content if hasattr(resp, 'content') else str(resp)
                
                # Try to parse JSON from response
                try:
                    # Look for JSON in the response
                    if '{' in response_content and '}' in response_content:
                        start = response_content.find('{')
                        end = response_content.rfind('}') + 1
                        json_str = response_content[start:end]
                        parsed_response = json.loads(json_str)
                        
                        if parsed_response.get('done', False):
                            done = True
                            return {"message": parsed_response.get('message', response_content), "replan": True}
                except json.JSONDecodeError:
                    pass
                
                # If no JSON found, check if response seems like a final plan
                if any(keyword in response_content.lower() for keyword in ['task 1:', 'step 1:', 'task:', 'step:']):
                    return {"message": response_content, "replan": True}
                
                # If response doesn't seem to be a function call or final plan, add to context
                context += f"\n\nLLM Response: {response_content}"
                
                # Ask for clarification if we're not making progress
                if iteration >= 3:
                    context += f"\n\nNote: Please either use a tool to gather more information or provide your final development plan."

            # If we exit the loop without completion
            if iteration >= max_iterations:
                return {"message": f"Maximum iterations reached. Last response: {response_content}", "replan": False}
            
            return {"message": "No valid response received from LLM.", "replan": False}

        except Exception as e:
            print(f"Error during Architect execution: {e}")
            return {"message": f"Error: Architecture planning failed: {e}", "replan": False}
