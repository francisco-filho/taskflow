import json
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from pydantic import BaseModel, Field

from taskflow.agents import Agent
from taskflow.memory import PersistentMemory, EventType
from taskflow.llm import LLMClient
from taskflow.plan import Planner, PlanStep, ExecutionPlan


class Task(BaseModel):
    """
    Represents a task requested by the user.
    """
    prompt: str = Field(..., description="The original user prompt for the task.")
    needs_plan: bool = Field(False, description="True if the task requires an explicit plan from an LLM.")
    needs_approval: bool = Field(True, description="True if the final result of the task needs user approval.")
    needs_eval: bool = Field(True, description="True if you want a LLM call to evaluate if the user request was fulfilled.")


class TaskFlow:
    """
    Responsible for coordinating and executing tasks using agents,
    maintaining its memory with persistence and resume capability.
    """
    
    def __init__(self, model: LLMClient, memory_file_path: Optional[str] = None):
        """
        Initialize TaskFlow with optional memory file for resumption capability.
        
        Args:
            model: LLM client for orchestration decisions
            memory_file_path: Path to memory file. If provided and exists, will attempt to resume.
                            If None, starts fresh. If provided but doesn't exist, creates new file.
        """
        self.memory = PersistentMemory(memory_file_path=memory_file_path, max_interaction_size=24)
        self.orchestrator_model = model  # This LLM is used by TaskFlow for evaluation
        self.available_agents = []
        self.planner = Planner(model, self.available_agents)
        self.final_response = None  # Store the final response
        self.current_plan: Optional[ExecutionPlan] = None
        self.step_results: Dict[int, Any] = {}  # Store results from each step
        
        # Check if we can resume from existing memory
        if self.memory.can_resume():
            print("Found existing incomplete task. Use check_resume_status() to see details.")
        
        print("TaskFlow initialized.")

    def check_resume_status(self) -> Optional[Dict[str, Any]]:
        """
        Check if there's a task that can be resumed and return its status.
        
        Returns:
            Dictionary with resume information or None if nothing to resume
        """
        return self.memory.get_resume_info()

    def resume_task(self, max_attempts: int = 10) -> bool:
        """
        Resume an incomplete task from memory.
        
        Args:
            max_attempts: Maximum attempts for task completion
            
        Returns:
            True if successfully resumed, False if no valid state to resume
        """
        if not self.memory.can_resume():
            print("No valid task state to resume from.")
            return False
        
        print("\n--- Resuming Task from Memory ---")
        execution_state = self.memory.get_execution_state()
        
        # Restore the execution state
        self.step_results = execution_state.step_results
        self.final_response = execution_state.final_response
        
        # Recreate the execution plan
        self.current_plan = ExecutionPlan.from_dict_list(execution_state.plan_steps)
        self.current_plan.current_step = execution_state.current_step
        
        print(f"Resuming task: {execution_state.user_prompt}")
        print(f"Completed steps: {list(self.step_results.keys())}")
        print(f"Resuming from step: {execution_state.current_step + 1}")
        
        # Record resume event
        self.memory.record_event(
            EventType.SYSTEM_EVENT,
            message=f"Resuming task from step {execution_state.current_step + 1}"
        )
        
        # Create a task object for the resumed execution
        task = Task(
            prompt=execution_state.user_prompt,
            needs_plan=True,  # Already have a plan
            needs_approval=True,
            needs_eval=True
        )
        
        # Continue execution from current step
        return self._continue_execution(task, max_attempts)

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

    def _is_task_fulfilled(self, original_task: str, agent_response: str, feedback_context: str = "") -> tuple[bool, str]:
        """
        Evaluates if the original task was fulfilled by the agent response.
        
        Parameters:
            original_task: The original user task/prompt
            agent_response: The response from the agent that handled the task
            feedback_context: Any previous feedback that should be considered
            
        Returns:
            A tuple of (is_fulfilled: bool, feedback_message: str)
        """
        evaluation_prompt = f"""Evaluate if the following user task was properly fulfilled by the agent response:

Original Task:
{original_task}

Agent Response:
{agent_response}"""

        if feedback_context:
            evaluation_prompt += f"""

Previous Feedback Context:
{feedback_context}

Please consider the previous feedback when evaluating if the agent response addresses the concerns raised."""
        
        evaluation_prompt += """

Please evaluate if the agent response adequately fulfills the user's original task. 

IMPORTANT EVALUATION CRITERIA:
- If the user requested to "commit changes" or "commit the changes", the task is only complete if the response shows that changes were actually committed (e.g., contains "Successfully committed with hash:" or similar confirmation).
- If the user requested to generate a commit message only, then generating the message is sufficient.
- If the user requested a review, the task is complete if a review was provided.
- If the user requested a diff, the task is complete if the diff output was provided.

Respond with either:
1. "FULFILLED" if the task was completed successfully
2. "NOT_FULFILLED: [specific reason why it wasn't fulfilled]" if the task was not completed

Be specific about what is missing or what needs to be done."""

        print("Evaluating if task was fulfilled...")
        try:
            response = self.orchestrator_model.chat(prompt=evaluation_prompt)
            evaluation_result = response.content.strip()
            print(f"Evaluation result: {evaluation_result}")
            
            # Record evaluation
            self.memory.record_event(
                EventType.EVALUATION,
                data={'evaluation_result': evaluation_result},
                message=evaluation_result
            )
            
            if evaluation_result.startswith("FULFILLED"):
                return True, evaluation_result
            else:
                # Extract the reason from "NOT_FULFILLED: reason"
                if "NOT_FULFILLED:" in evaluation_result:
                    reason = evaluation_result.split("NOT_FULFILLED:", 1)[1].strip()
                else:
                    reason = evaluation_result
                return False, reason
                
        except Exception as e:
            print(f"Error during task evaluation: {e}")
            return False, f"Error during evaluation: {e}"

    def _continue_execution(self, task: Task, max_attempts: int = 10) -> bool:
        """
        Continue execution from the current step in the plan.
        Used both for normal execution and resumption.
        
        Returns:
            True if execution completed successfully, False otherwise
        """
        if not self.current_plan or not self.current_plan.steps:
            print("No execution plan available.")
            return False

        # Update memory with current execution state
        self._update_memory_state(task.prompt)
        
        overall_attempt = 0
        while not self.current_plan.is_complete() and overall_attempt < max_attempts:
            overall_attempt += 1
            current_step = self.current_plan.get_current_step()
            
            if not current_step:
                break
                
            print(f"\n--- Executing Step {current_step.step_number}: {current_step.agent_name} ---")
            print(f"Description: {current_step.description}")
            
            # Record step start
            self.memory.record_event(
                EventType.STEP_STARTED,
                step_number=current_step.step_number,
                agent_name=current_step.agent_name,
                message=current_step.description
            )
            
            # Get the agent for this step
            agent = self._get_agent_by_name(current_step.agent_name)
            if not agent:
                print(f"Agent '{current_step.agent_name}' not found. Skipping step.")
                self.memory.record_event(
                    EventType.STEP_FAILED,
                    step_number=current_step.step_number,
                    agent_name=current_step.agent_name,
                    message=f"Agent '{current_step.agent_name}' not found"
                )
                self.current_plan.advance_step()
                self._update_memory_state(task.prompt)
                continue
            
            # Build context for this step
            step_context = self._build_step_context(current_step, task.prompt)
            print(f"Step context length: {len(step_context)} characters")
            
            # Record agent input
            self.memory.record_event(
                EventType.AGENT_INPUT,
                step_number=current_step.step_number,
                agent_name=current_step.agent_name,
                data={'context': step_context},
                message=f"Sending context to {current_step.agent_name}"
            )
            
            try:
                # Execute the agent
                result = agent.run(prompt=step_context)
                
                # Store the result
                self.step_results[current_step.step_number] = result
                
                result_str = json.dumps(result, indent=2) if isinstance(result, dict) else str(result)
                print(f"Step {current_step.step_number} completed. Result:\n{result_str}")
                
                # Record agent output and step completion
                self.memory.record_event(
                    EventType.AGENT_OUTPUT,
                    step_number=current_step.step_number,
                    agent_name=current_step.agent_name,
                    data={'result': result},
                    message=f"Agent {current_step.agent_name} completed step {current_step.step_number}"
                )
                
                self.memory.record_event(
                    EventType.STEP_COMPLETED,
                    step_number=current_step.step_number,
                    agent_name=current_step.agent_name,
                    message=f"Step {current_step.step_number} completed successfully"
                )
                
                # Move to next step
                self.current_plan.advance_step()
                self._update_memory_state(task.prompt)
                
            except Exception as e:
                print(f"Error executing step {current_step.step_number}: {e}")
                
                # Record step failure
                self.memory.record_event(
                    EventType.STEP_FAILED,
                    step_number=current_step.step_number,
                    agent_name=current_step.agent_name,
                    data={'error': str(e)},
                    message=f"Step {current_step.step_number} failed: {e}"
                )
                
                # For now, skip failed steps. Could implement retry logic here.
                self.current_plan.advance_step()
                self._update_memory_state(task.prompt)

        # Evaluate overall completion
        if self.current_plan.is_complete():
            print("\n--- Plan Execution Complete ---")
            
            # Get the final result (from the last step)
            if self.step_results:
                final_step_num = max(self.step_results.keys())
                self.final_response = self.step_results[final_step_num]
                
                # Update memory with final state
                self._update_memory_state(task.prompt, is_complete=True)
                
                # Evaluate if the overall task was fulfilled
                if task.needs_eval:
                    final_result_str = json.dumps(self.final_response, indent=2) if isinstance(self.final_response, dict) else str(self.final_response)
                    is_fulfilled, evaluation_message = self._is_task_fulfilled(task.prompt, final_result_str)
                    
                    if is_fulfilled:
                        print("✓ Overall task was fulfilled successfully!")
                        self.memory.record_event(
                            EventType.TASK_COMPLETED,
                            message=f"Task fulfilled: {evaluation_message}"
                        )
                        
                        if task.needs_approval:
                            user_feedback = input("\nTask completed! Do you approve this result? (yes/no): ").lower().strip()
                            self.memory.record_event(
                                EventType.USER_APPROVAL,
                                data={'approved': user_feedback == 'yes'},
                                message=f"User approval: {user_feedback}"
                            )
                            if user_feedback == "yes":
                                print("User approved. Task completed.")
                                return True
                            else:
                                print("User did not approve. Task marked as incomplete.")
                                return False
                        else:
                            print("Task completed successfully.")
                            return True
                    else:
                        print(f"✗ Overall task not fulfilled: {evaluation_message}")
                        self.memory.record_event(
                            EventType.TASK_FAILED,
                            message=f"Task not fulfilled: {evaluation_message}"
                        )
                        return False
                else:
                    print("Task execution completed (no evaluation requested).")
                    self.memory.record_event(
                        EventType.TASK_COMPLETED,
                        message="Task execution completed without evaluation"
                    )
                    if task.needs_approval:
                        user_feedback = input("\nDo you approve this result? (yes/no): ").lower().strip()
                        self.memory.record_event(
                            EventType.USER_APPROVAL,
                            data={'approved': user_feedback == 'yes'},
                            message=f"User approval: {user_feedback}"
                        )
                        if user_feedback == "yes":
                            print("User approved. Task completed.")
                            return True
                        else:
                            print("User did not approve.")
                            return False
                    return True
        else:
            print(f"\n--- Plan execution incomplete after {max_attempts} attempts ---")
            self.memory.record_event(
                EventType.TASK_FAILED,
                message=f"Plan execution incomplete after {max_attempts} attempts"
            )
            return False

    def _update_memory_state(self, user_prompt: str, is_complete: bool = False):
        """Update the execution state in memory"""
        if self.current_plan:
            self.memory.update_execution_state(
                user_prompt=user_prompt,
                plan_steps=self.current_plan.to_dict_list(),
                current_step=self.current_plan.current_step,
                step_results=self.step_results,
                is_complete=is_complete,
                final_response=self.final_response
            )

    def run(self, task: Task, max_attempts: int = 10):
        """
        Executes the task by creating and following an execution plan.
        """
        # Record the start of a new task
        self.memory.record_event(
            EventType.USER_PROMPT,
            data={'prompt': task.prompt, 'needs_plan': task.needs_plan, 'needs_approval': task.needs_approval, 'needs_eval': task.needs_eval},
            message=f"Starting new task: {task.prompt}"
        )
        
        print(f"\n--- TaskFlow: Running task ---")

        # Step 1: Create execution plan using the Planner
        if task.needs_plan or self.planner.should_create_detailed_plan(task.prompt):
            print("\n--- Creating Detailed Execution Plan ---")
            self.current_plan = self.planner.create_execution_plan(task.prompt)
            
            # Record plan creation
            self.memory.record_event(
                EventType.PLAN_CREATED,
                data={
                    'plan_steps': self.current_plan.to_dict_list(),
                    'reasoning': 'Detailed plan created by Planner'
                },
                message=f"Planner created plan with {len(self.current_plan.steps)} steps"
            )
            
            # Log the plan
            if self.current_plan.steps:
                plan_summary = self.current_plan.get_plan_summary()
                print(f"\n{plan_summary}")
        else:
            # Create a simple single-step plan using the Planner
            print("\n--- Creating Simple Plan ---")
            self.current_plan = self.planner.create_execution_plan(task.prompt)
            
            # Record simple plan creation
            self.memory.record_event(
                EventType.PLAN_CREATED,
                data={'plan_steps': self.current_plan.to_dict_list()},
                message=f"Planner created simple plan with {len(self.current_plan.steps)} steps"
            )
            
            if self.current_plan.steps:
                selected_agent_name = self.current_plan.steps[0].agent_name
                print(f"Created simple plan with 1 step: {selected_agent_name}")
            else:
                print("Created simple plan with 0 steps")

        if not self.current_plan or not self.current_plan.steps:
            print("Failed to create execution plan. No suitable agents found.")
            self.memory.record_event(
                EventType.TASK_FAILED,
                message="Failed to create execution plan. No suitable agents found."
            )
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
    
    def get_memory_summary(self) -> str:
        """Get a summary of the memory state"""
        return self.memory.get_summary()
    
    def clear_memory(self):
        """Clear all memory (start fresh)"""
        self.memory.clear_memory()
        self.current_plan = None
        self.step_results = {}
        self.final_response = None
        print("TaskFlow memory cleared. Ready for new task.")