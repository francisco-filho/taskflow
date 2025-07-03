from typing import Optional, Dict, List, Callable

from taskflow.util import logger
from taskflow.llm import LLMClient
from taskflow.agents import Agent


class Evaluator(Agent):
    """
    A simplified agent responsible for evaluating whether a previous agent's response
    fulfills the user's original request with a numerical score from 1-5.
    """
    def __init__(self, model: LLMClient, system_prompt: str, available_tools: Optional[Dict[str, Callable]] = None):
        super().__init__("Evaluator", model, "Evaluates previous agent responses with a score from 1-5.", system_prompt, available_tools)

    def run(self, prompt: str, **kwargs) -> str:
        """
        Evaluates whether a previous agent's response fulfills the user's original request.
        Returns a score from 1-5 with explanation.

        Parameters:
            prompt: The prompt containing the evaluation request.
            **kwargs: Additional keyword arguments (for compatibility).

        Returns:
            A formatted string with the evaluation result.
        """
        print(f"Evaluator agent running with prompt: {prompt[:100]}...")

        # Create the evaluation system prompt
        evaluation_system_prompt = f"""{self.system_prompt}

You are an evaluation agent that scores responses on a scale of 1-5:

1 = Poor: The response completely fails to address the user's request
2 = Below Average: The response partially addresses the request but has significant gaps
3 = Average: The response addresses most of the request but lacks some important elements
4 = Good: The response addresses the request well with minor issues
5 = Excellent: The response completely and accurately fulfills the user's request

EVALUATION PROCESS:
1. Analyze the prompt to understand what needs to be evaluated
2. If you need additional information to make a proper evaluation, use available tools
3. Continue gathering information until you have enough context
4. Provide your final evaluation with a score and detailed explanation

EVALUATION CRITERIA:
- Does the response directly address what the user asked for?
- If an action was requested, was it actually performed?
- If information was requested, was it provided accurately and completely?
- If content generation was requested, was quality content generated?
- Are there any gaps between what was requested and what was delivered?

Always respond with your final evaluation in this format:
Score: [1-5]
[Detailed explanation of your reasoning]"""

        try:
            # Start the evaluation process
            tools = self._get_tool_schemas() if hasattr(self, '_get_tool_schemas') else []
            
            # The LLM will decide if it needs more information and use tools accordingly
            current_prompt = prompt
            max_iterations = 10  # Prevent infinite loops
            iteration = 0
            
            while iteration < max_iterations:
                iteration += 1
                print(f"Evaluation iteration {iteration}")
                
                # Get response from LLM
                resp = self.model.chat(
                    prompt=current_prompt, 
                    system_prompt=evaluation_system_prompt,
                    tools=tools
                )
                
                # If LLM wants to use a tool, execute it and continue
                if resp.function_call:
                    print(f"LLM wants to use tool: {resp.function_call.name}")
                    
                    # Execute the function call
                    tool_result = self._execute_function_call(resp.function_call)
                    
                    # Add tool result to the conversation context
                    current_prompt += f"\n\nTool Result ({resp.function_call.name}):\n{tool_result}\n\nNow please continue your evaluation based on this additional information."
                    
                    continue
                
                # If no function call, we have the final evaluation
                evaluation_result = resp.content.strip()
                print(f"Final evaluation: {evaluation_result}")
                
                return evaluation_result
            
            # If we hit max iterations, return what we have
            return "Error: Maximum evaluation iterations reached. Could not complete evaluation."
            
        except Exception as e:
            print(f"Error during Evaluator execution: {e}")
            return f"Error: Evaluation failed: {e}"
