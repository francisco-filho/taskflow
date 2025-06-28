import json
from typing import Optional, Dict, Any

from taskflow.util import logger
from taskflow.agents import Agent
from taskflow.llm import LLMClient
from taskflow.plan import Planner, PlanStep, ExecutionPlan
from taskflow.models import Task
from taskflow.exceptions import NoChangesStaged
from taskflow.taskeval import TaskEvaluator


class TaskFlow:
    """
    Responsible for coordinating and executing tasks using agents
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
        self.step_results: Dict[int, Any] = {}  # Store results from each step
        
        # Initialize the task evaluator
        self.evaluator = TaskEvaluator(model)
        
        print("TaskFlow initialized.")


    def add(self, agent: Agent):
        """
        Adds an agent to the list of available agents.
        """
        self.available_agents.append(agent)
        self.planner.update_available_agents(self.available_agents)
        print(f"Agent '{agent.name}' added to TaskFlow.")

    def _get_agent_by_name(self, agent_name: str) -> Optional[Agent]:
        """Get an agent by its name"""
        for agent in self.available_agents:
            if agent.name.lower() == agent_name.lower():
                return agent
        return None

    def _build_step_context(self, step: PlanStep, task_prompt: str) -> str:
        """Build the context for a specific step, including results from previous steps"""
        context_parts = []
        
        # Add the original task
        context_parts.append(f"Original Task: {task_prompt}")
        
        # Add the specific step context
        if step.input_context:
            context_parts.append(f"Step Context: {step.input_context}")
        
        # Add results from dependent steps
        if step.depends_on:
            context_parts.append("Previous Step Results:")
            for dep_step_num in step.depends_on:
                if dep_step_num in self.step_results:
                    result = self.step_results[dep_step_num]
                    result_str = json.dumps(result, indent=2) if isinstance(result, dict) else str(result)
                    context_parts.append(f"  Step {dep_step_num} Result: {result_str}")
        
        # If no specific dependencies, but this isn't the first step, include the most recent result
        elif step.step_number > 1 and self.step_results:
            latest_step = max(self.step_results.keys())
            if latest_step < step.step_number:
                result = self.step_results[latest_step]
                result_str = json.dumps(result, indent=2) if isinstance(result, dict) else str(result)
                context_parts.append(f"Previous Step Result: {result_str}")
        
        return "\n\n".join(context_parts)

    def _continue_execution(self, task: Task, max_attempts: int = 10) -> bool:
        """
        Continue execution from the current step in the plan.
        Used both for normal execution and resumption.
        Updated to use the TaskEvaluator for all evaluation logic.
        
        Returns:
            True if execution completed successfully, False otherwise
        """
        if not self.current_plan or not self.current_plan.steps:
            print("No execution plan available.")
            return False
        
        overall_attempt = 0
        while not self.current_plan.is_complete() and overall_attempt < max_attempts:
            overall_attempt += 1
            current_step = self.current_plan.get_current_step()
            
            if not current_step:
                break
                
            print(f"\n--- Executing Step {current_step.step_number}: {current_step.agent_name} ---")
            print(f"Description: {current_step.description}")
            
            # Get the agent for this step
            agent = self._get_agent_by_name(current_step.agent_name)
            if not agent:
                print(f"Agent '{current_step.agent_name}' not found. Skipping step.")
                self.current_plan.advance_step()
                continue
            
            # Build context for this step
            step_context = self._build_step_context(current_step, task.prompt)
            print(f"Step context length: {len(step_context)} characters")
            
            try:
                # Execute the agent
                result = agent.run(prompt=step_context)
                
                # Store the result
                self.step_results[current_step.step_number] = result
                
                result_str = json.dumps(result, indent=2) if isinstance(result, dict) else str(result)
                
                # Handle Evaluator agent output specially - print the formatted response
                if current_step.agent_name.lower() == "evaluator":
                    print(f"\n{result_str}")  # Print the formatted evaluation response
                else:
                    print(f"Step {current_step.step_number} completed. Result:\n{result_str}")
                
                # Move to next step
                self.current_plan.advance_step()
                
            except NoChangesStaged as e:
                logger.error(e)
                exit(1)
            except Exception as e:
                print(f"Error executing step {current_step.step_number}: {e}")
                self.current_plan.advance_step()

        # Evaluate overall completion using the TaskEvaluator
        if self.current_plan.is_complete():
            print("\n--- Plan Execution Complete ---")
            
            # Get the final result (from the last step)
            if self.step_results:
                final_step_num = max(self.step_results.keys())
                self.final_response = self.step_results[final_step_num]
                
                # Check if the last step was an Evaluator
                last_step = None
                for step in self.current_plan.steps:
                    if step.step_number == final_step_num:
                        last_step = step
                        break
                
                # Handle evaluation based on whether the last step was an Evaluator agent
                if last_step and last_step.agent_name.lower() == "evaluator":
                    # Parse the Evaluator agent response
                    evaluator_response = str(self.final_response)
                    evaluation_result = self.evaluator.parse_evaluator_agent_response(evaluator_response)
                    
                    print(f"✓ Evaluator completed with score: {evaluation_result.score}/5")
                    
                    if evaluation_result.is_fulfilled:
                        print("✓ Overall task was fulfilled successfully!")
                        return self._handle_user_approval(task)
                    else:
                        print(f"✗ Overall task evaluation score too low: {evaluation_result.score}/5")
                        return False
                else:
                    # Use TaskEvaluator for standard evaluation
                    if task.needs_eval:
                        final_result_str = json.dumps(self.final_response, indent=2) if isinstance(self.final_response, dict) else str(self.final_response)
                        evaluation_result = self.evaluator.evaluate(task.prompt, final_result_str)
                        
                        if evaluation_result.is_fulfilled:
                            print("✓ Overall task was fulfilled successfully!")
                            return self._handle_user_approval(task)
                        else:
                            print(f"✗ Overall task not fulfilled: {evaluation_result.feedback_message}")
                            return False
                    else:
                        print("Task execution completed (no evaluation requested).")
                        return self._handle_user_approval(task, default_complete=True)
        else:
            print(f"\n--- Plan execution incomplete after {max_attempts} attempts ---")
            return False

    def _handle_user_approval(self, task: Task, default_complete: bool = False) -> bool:
        """
        Handle user approval logic.
        
        Args:
            task: The task being executed
            default_complete: Whether to return True if no approval is needed
            
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
            if default_complete:
                print("Task completed successfully.")
            return True

    def run(self, task: Task, max_attempts: int = 10):
        """
        Executes the task by creating and following an execution plan.
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

        # Step 2: Execute the plan
        self._continue_execution(task, max_attempts)

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
