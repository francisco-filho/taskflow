import json
from typing import Optional, Dict, Any, Callable

from taskflow.util import logger
from taskflow.agents import Agent
from taskflow.llm import LLMClient
from taskflow.plan import Planner, PlanStep, ExecutionPlan
from taskflow.models import Task
from taskflow.exceptions import NoChangesStaged
from taskflow.taskeval import TaskEvaluator


class PlanExecutor:
    """
    Responsible for executing plan steps and managing step results.
    Handles the granular execution loop and error management.
    """
    
    def __init__(self, available_agents: list[Agent]):
        """
        Initialize PlanExecutor.
        
        Args:
            available_agents: List of available agents for execution
        """
        self.available_agents = available_agents
        self.step_results: Dict[int, Any] = {}
        self.memory = []
    
    def _get_agent_by_name(self, agent_name: str) -> Optional[Agent]:
        """Get an agent by its name"""
        for agent in self.available_agents:
            if agent.name.lower() == agent_name.lower():
                return agent
        return None
    
    def execute_plan(self, 
                    plan: ExecutionPlan, 
                    context_builder: Callable[[PlanStep, str], str],
                    task_prompt: str,
                    max_attempts: int = 10) -> Dict[int, Any]:
        """
        Execute the given execution plan.
        
        Args:
            plan: The execution plan to run
            context_builder: Function to build context for each step
            task_prompt: The original task prompt
            max_attempts: Maximum number of attempts for overall execution
            
        Returns:
            Dictionary of step results indexed by step number
        """
        self.step_results.clear()  # Reset results for new execution
        
        overall_attempt = 0
        last_agent_response = "\n"
        while not plan.is_complete() and overall_attempt < max_attempts:
            overall_attempt += 1
            current_step = plan.get_current_step()
            
            if not current_step:
                break
                
            print(f"\n--- Executing Step {current_step.step_number}: {current_step.agent_name} ---")
            print(f"Description: {current_step.description}")
            
            # Get the agent for this step
            self.memory.append([s for s in plan.steps])
            agent = self._get_agent_by_name(current_step.agent_name)
            if not agent:
                print(f"Agent '{current_step.agent_name}' not found. Skipping step.")
                plan.advance_step()
                continue
            self.memory.append("\n---Agent---")
            self.memory.append(agent.name)
            
            # Build context for this step
            step_context = context_builder(current_step, task_prompt)
            print(f"Step context length: {len(step_context)} characters")
            
            try:
                # Execute the agent
                self.memory.append(f"\n--- prompt ----\n{step_context}")
                result = ""
                agent_resp = agent.run(prompt=step_context)
                # print("*"*80)
                # print(agent_resp)
                # print("*"*80)
                if isinstance(agent_resp, str):
                    result = agent_resp
                else:
                    # TODO: chek for errors
                    result = agent_resp['message']

                last_agent_response = result
                self.memory.append(result)
                
                # Store the result
                self.step_results[current_step.step_number] = f"\n{result}\n"
                
                result_str = json.dumps(result, indent=2) if isinstance(result, dict) else str(result)
                
                # Handle Evaluator agent output specially - print the formatted response
                if current_step.agent_name.lower() == "evaluator":
                    print(f"\n{result_str}")  # Print the formatted evaluation response
                else:
                    print(f"Step {current_step.step_number} completed. Result:\n{result_str}")
                
                # Move to next step
                plan.advance_step()
                
            except NoChangesStaged as e:
                logger.error(e)
                raise  # Re-raise this specific exception to be handled at higher level
            except Exception as e:
                print(f"Error executing step {current_step.step_number}: {e}")
                plan.advance_step()
        
        return self.step_results.copy()
    
    def get_step_results(self) -> Dict[int, Any]:
        """Get the current step results"""
        return self.step_results.copy()


class TaskCompletionHandler:
    """
    Responsible for handling task completion evaluation and user approval.
    Encapsulates all logic related to determining task fulfillment.
    """
    
    def __init__(self, evaluator: TaskEvaluator):
        """
        Initialize TaskCompletionHandler.
        
        Args:
            evaluator: TaskEvaluator instance for evaluation logic
        """
        self.evaluator = evaluator
    

    def handle_completion(self, 
                     task: Task, 
                     plan: ExecutionPlan, 
                     step_results: Dict[int, Any]) -> tuple[bool, Any]:
        """
        Handle task completion evaluation and approval.
        
        Args:
            task: The task that was executed
            plan: The execution plan that was completed
            step_results: Results from all executed steps
            
        Returns:
            Tuple of (success: bool, final_response: Any)
        """
        if not plan.is_complete():
            print(f"\n--- Plan execution incomplete ---")
            return False, None
            
        print("\n--- Plan Execution Complete ---")
        
        # Get the final result (from the last step)
        if not step_results:
            print("No step results available.")
            return False, None
            
        final_step_num = max(step_results.keys())
        final_response = step_results[final_step_num]
        
        # Find the last step to check if it was an Evaluator
        last_step = self._find_step_by_number(plan, final_step_num)
        
        # Determine completion success based on evaluation
        success = self._evaluate_completion(task, plan, final_step_num, final_response, step_results)
        
        if success:
            # Handle user approval if required
            success = self._handle_user_approval(task)
        
        # If the last step was an Evaluator, combine the actual result with the evaluation
        if last_step and last_step.agent_name.lower() == "evaluator" and len(step_results) > 1:
            # Get the second-to-last step result (the actual work result)
            sorted_steps = sorted(step_results.keys())
            if len(sorted_steps) >= 2:
                actual_result_step = sorted_steps[-2]  # Second to last
                actual_result = step_results[actual_result_step]
                evaluation_result = final_response
                
                # Combine them in the desired format
                combined_result = f"{actual_result}\n{'-'*50}\n{evaluation_result}"
                return success, combined_result
        
        return success, final_response

    def _evaluate_completion(self, 
                           task: Task, 
                           plan: ExecutionPlan, 
                           final_step_num: int, 
                           final_response: Any,
                           step_results: Dict[int, Any]) -> bool:
        """
        Evaluate whether the task was completed successfully.
        
        Args:
            task: The task that was executed
            plan: The execution plan
            final_step_num: Step number of the final step
            final_response: Response from the final step
            step_results: All step results
            
        Returns:
            True if task is considered successfully completed
        """
        # Find the last step to check if it was an Evaluator
        last_step = self._find_step_by_number(plan, final_step_num)
        
        # Handle evaluation based on whether the last step was an Evaluator agent
        if last_step and last_step.agent_name.lower() == "evaluator":
            return self._handle_evaluator_agent_completion(final_response)
        else:
            return self._handle_standard_completion(task, final_response)
    
    def _find_step_by_number(self, plan: ExecutionPlan, step_number: int) -> Optional[PlanStep]:
        """Find a step in the plan by its step number"""
        for step in plan.steps:
            if step.step_number == step_number:
                return step
        return None
    
    def _handle_evaluator_agent_completion(self, final_response: Any) -> bool:
        """Handle completion when the last step was an Evaluator agent"""
        evaluator_response = str(final_response)
        evaluation_result = self.evaluator.parse_evaluator_agent_response(evaluator_response)
        
        print(f"✓ Evaluator completed with score: {evaluation_result.score}/5")
        
        if evaluation_result.is_fulfilled:
            print("✓ Overall task was fulfilled successfully!")
            return True
        else:
            print(f"✗ Overall task evaluation score too low: {evaluation_result.score}/5")
            return False
    
    def _handle_standard_completion(self, task: Task, final_response: Any) -> bool:
        """Handle completion using standard TaskEvaluator"""
        if task.needs_eval:
            final_result_str = json.dumps(final_response, indent=2) if isinstance(final_response, dict) else str(final_response)
            evaluation_result = self.evaluator.evaluate(task.prompt, final_result_str)
            
            if evaluation_result.is_fulfilled:
                print("✓ Overall task was fulfilled successfully!")
                return True
            else:
                print(f"✗ Overall task not fulfilled: {evaluation_result.feedback_message}")
                return False
        else:
            print("Task execution completed (no evaluation requested).")
            return True
    
    def _handle_user_approval(self, task: Task) -> bool:
        """
        Handle user approval logic.
        
        Args:
            task: The task being executed
            
        Returns:
            True if approved or no approval needed, False otherwise
        """
        if task.needs_approval:
            user_feedback = input("\nTask completed! Do you approve this result? (yes/no): ").lower().strip()
            if user_feedback == "yes":
                print("User approved. Task completed.")
                return True
            else:
                print("User did not approve. Task marked as incomplete.")
                return False
        else:
            print("Task completed successfully.")
            return True


class TaskFlow:
    """
    Responsible for coordinating and executing tasks using agents.
    Focuses on high-level orchestration and task lifecycle management.
    """
    
    def __init__(self, model: LLMClient):
        """
        Initialize TaskFlow
        
        Args:
            model: LLM client for orchestration decisions
        """
        self.orchestrator_model = model  # This LLM is used by TaskFlow for evaluation
        self.available_agents = []
        self.planner = Planner(model, self.available_agents)
        self.final_response = None  # Store the final response
        self.current_plan: Optional[ExecutionPlan] = None
        
        # Initialize the task evaluator and completion handler
        self.evaluator = TaskEvaluator(model)
        self.completion_handler = TaskCompletionHandler(self.evaluator)
        
        # Initialize the plan executor
        self.plan_executor = PlanExecutor(self.available_agents)
        
        print("TaskFlow initialized.")

    def add(self, agent: Agent):
        """
        Adds an agent to the list of available agents.
        """
        self.available_agents.append(agent)
        self.planner.update_available_agents(self.available_agents)
        
        # Update the plan executor with the new agents list
        self.plan_executor = PlanExecutor(self.available_agents)
        
        print(f"Agent '{agent.name}' added to TaskFlow.")

    def _build_step_context(self, step: PlanStep, task_prompt: str) -> str:
        """
        Build the context for a specific step, including results from previous steps.
        This method remains in TaskFlow as it depends on the overall execution state.
        """
        context_parts = []
        
        # Add the original task
        context_parts.append(f"Original Task: {task_prompt}")
        
        # Add the specific step context
        if step.input_context:
            context_parts.append(f"Step Context: {step.input_context}")
        
        # Get current step results from the executor
        step_results = self.plan_executor.get_step_results()
        
        # Add results from dependent steps
        if step.depends_on:
            context_parts.append("Previous Step Results:")
            for dep_step_num in step.depends_on:
                if dep_step_num in step_results:
                    result = step_results[dep_step_num]
                    result_str = json.dumps(result, indent=2) if isinstance(result, dict) else str(result)
                    context_parts.append(f"  Step {dep_step_num} Result: {result_str}\n")
        
        # If no specific dependencies, but this isn't the first step, include the most recent result
        elif step.step_number > 1 and step_results:
            latest_step = max(step_results.keys())
            if latest_step < step.step_number:
                result = step_results[latest_step]
                result_str = json.dumps(result, indent=2) if isinstance(result, dict) else str(result)
                context_parts.append(f"Previous Step Result: {result_str}")
        
        return "\n\n".join(context_parts)

    def run(self, task: Task, max_attempts: int = 10):
        """
        Executes the task by creating and following an execution plan.
        Now focuses on orchestration while delegating execution and completion handling.
        """
        print(f"\n--- TaskFlow: Running task ---")
        print(f"{task.prompt}")

        # Step 1: Create execution plan using the Planner
        if task.needs_plan or self.planner.should_create_detailed_plan(task.prompt):
            print("\n--- Creating Detailed Execution Plan ---")
            self.current_plan = self.planner.create_execution_plan(task.prompt)
            
            # Log the plan
            if self.current_plan.steps:
                plan_summary = self.current_plan.get_plan_summary()
                print(f"\n{plan_summary}")
        else:
            # Create a simple single-step plan using the Planner
            print("\n--- Creating Simple Plan ---")
            self.current_plan = self.planner.create_execution_plan(task.prompt)
            
            if self.current_plan.steps:
                selected_agent_name = self.current_plan.steps[0].agent_name
                print(f"Created simple plan with 1 step: {selected_agent_name}")
            else:
                print("Created simple plan with 0 steps")

        if not self.current_plan or not self.current_plan.steps:
            print("Failed to create execution plan. No suitable agents found.")
            return

        # Step 2: Execute the plan using PlanExecutor
        try:
            step_results = self.plan_executor.execute_plan(
                plan=self.current_plan,
                context_builder=self._build_step_context,
                task_prompt=task.prompt,
                max_attempts=max_attempts
            )
            
            # Step 3: Handle completion using TaskCompletionHandler
            success, final_response = self.completion_handler.handle_completion(
                task=task,
                plan=self.current_plan,
                step_results=step_results
            )

            # print("*"*80)
            # for l in self.plan_executor.memory:
            #     print(l)
            # print("*"*80)
            
            if success:
                self.final_response = final_response
        except NoChangesStaged as e:
            logger.error(e)
            exit(1)

    def get_final_response(self):
        """Get the final response from the successfully completed task."""
        return self.final_response
    
    def get_plan_status(self) -> str:
        """Get a summary of the current plan status"""
        if not self.current_plan:
            return "No execution plan created yet."
        return self.current_plan.get_plan_summary()
    
    def get_evaluator(self) -> TaskEvaluator:
        """Get the task evaluator instance for advanced evaluation features."""
        return self.evaluator
    
    def set_evaluation_threshold(self, threshold: int):
        """Set the evaluation threshold for task fulfillment."""
        self.evaluator.set_fulfillment_threshold(threshold)
