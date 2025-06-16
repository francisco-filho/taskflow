import json
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from pydantic import BaseModel, Field

from taskflow.agents import Agent, Commiter
from taskflow.memory import Memory
from taskflow.llm import LLMClient

@dataclass
class PlanStep:
    """Represents a single step in the execution plan"""
    step_number: int
    agent_name: str
    description: str
    input_context: str = ""  # Context to pass to this step
    depends_on: List[int] = None  # List of step numbers this step depends on
    
    def __post_init__(self):
        if self.depends_on is None:
            self.depends_on = []

class ExecutionPlan:
    """Represents the complete execution plan for a task"""
    def __init__(self):
        self.steps: List[PlanStep] = []
        self.current_step = 0
        
    def add_step(self, step: PlanStep):
        self.steps.append(step)
        
    def get_current_step(self) -> Optional[PlanStep]:
        if self.current_step < len(self.steps):
            return self.steps[self.current_step]
        return None
        
    def advance_step(self):
        self.current_step += 1
        
    def is_complete(self) -> bool:
        return self.current_step >= len(self.steps)
        
    def get_plan_summary(self) -> str:
        summary = "Execution Plan:\n"
        for i, step in enumerate(self.steps, 1):
            status = "✓" if i <= self.current_step else "○"
            summary += f"{status} Step {step.step_number}: {step.agent_name} - {step.description}\n"
        return summary

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
    Responsible for selecting and orchestrating the execution of tasks,
    maintaining its memory.
    """
    available_agents: List[Agent] = []

    def __init__(self, model: LLMClient):
        self.memory = Memory(user_prompt="Initial prompt", max_interaction_size=24)
        self.orchestrator_model = model # This LLM is used by OrchestratorAI itself for agent selection
        self.available_agents = []
        self.final_response = None  # Store the final response
        self.current_plan: Optional[ExecutionPlan] = None
        self.step_results: Dict[int, Any] = {}  # Store results from each step
        print("OrchestratorAI initialized.")

    def add(self, agent: Agent):
        """
        Adds an agent to the list of available agents.
        """
        self.available_agents.append(agent)
        print(f"Agent '{agent.name}' added to OrchestratorAI.")

    def _create_execution_plan(self, task_prompt: str) -> ExecutionPlan:
        """
        Creates a detailed execution plan for complex tasks that require multiple steps or agents.
        """
        if not self.available_agents:
            print("No agents available to create plan.")
            return ExecutionPlan()

        agent_descriptions = "\n".join([f"- {a.name}: {a.description}" for a in self.available_agents])
        
        planning_prompt = f"""Given the user's task: '{task_prompt}', create a detailed execution plan.

Available Agents:
{agent_descriptions}

Analyze the task and determine if it requires multiple steps or agents. If so, break it down into sequential steps.

Please respond with a JSON object in this format:
{{
    "requires_planning": true/false,
    "reasoning": "explanation of why planning is or isn't needed",
    "steps": [
        {{
            "step_number": 1,
            "agent_name": "AgentName",
            "description": "What this step will accomplish",
            "input_context": "What context/input this step needs",
            "depends_on": [list of step numbers this depends on, empty if none]
        }}
    ]
}}

Guidelines:
- If the task is simple and can be handled by one agent, set requires_planning to false and provide a single step
- If the task requires multiple operations, reviews, or outputs from one agent feeding into another, set requires_planning to true
- Each step should have a clear purpose and specify what context it needs from previous steps
- Consider dependencies between steps (e.g., you need to generate code before you can commit it)
- Be specific about what each agent should do and what input it needs

Examples of tasks that need planning:
- "Generate a commit message and then commit the changes" (2 steps: generate message, then commit)
- "Review the code, make changes, then commit" (3 steps: review, modify, commit)
- "Analyze the diff and create a detailed report" (might need 1 or 2 steps depending on complexity)

Respond ONLY with valid JSON."""

        print("Creating execution plan...")
        try:
            response = self.orchestrator_model.chat(prompt=planning_prompt)
            plan_data = json.loads(response.content.strip())
            
            print(f"Planning analysis: {plan_data.get('reasoning', 'No reasoning provided')}")
            
            execution_plan = ExecutionPlan()
            
            if plan_data.get("requires_planning", False) and plan_data.get("steps"):
                for step_data in plan_data["steps"]:
                    step = PlanStep(
                        step_number=step_data["step_number"],
                        agent_name=step_data["agent_name"],
                        description=step_data["description"],
                        input_context=step_data.get("input_context", ""),
                        depends_on=step_data.get("depends_on", [])
                    )
                    execution_plan.add_step(step)
                print(f"Created execution plan with {len(execution_plan.steps)} steps")
            else:
                # Single step plan - select the best agent for the entire task
                selected_agent = self._select_agent(task_prompt)
                if selected_agent:
                    step = PlanStep(
                        step_number=1,
                        agent_name=selected_agent.name,
                        description=f"Handle the complete task: {task_prompt}",
                        input_context=task_prompt
                    )
                    execution_plan.add_step(step)
                    print("Created single-step execution plan")
                
            return execution_plan
            
        except Exception as e:
            print(f"Error during plan creation: {e}")
            # Fallback to single agent selection
            selected_agent = self._select_agent(task_prompt)
            execution_plan = ExecutionPlan()
            if selected_agent:
                step = PlanStep(
                    step_number=1,
                    agent_name=selected_agent.name,
                    description=f"Handle the complete task: {task_prompt}",
                    input_context=task_prompt
                )
                execution_plan.add_step(step)
            return execution_plan

    def _select_agent(self, task_prompt: str) -> Optional[Agent]:
        """
        Selects the most appropriate agent for the given task prompt using an LLM.
        """
        if not self.available_agents:
            print("No agents available to select from.")
            return None

        agent_descriptions = "\n".join([f"- {a.name}: {a.description}" for a in self.available_agents])
        selection_prompt = (
            f"Given the user's task: '{task_prompt}', which of the following agents is most suitable to handle it?\n"
            f"Available Agents:\n{agent_descriptions}\n\n"
            f"Respond ONLY with the name of the most suitable agent (e.g., 'Commiter') or 'None' if no agent is suitable. "
            f"Do not include any other text."
        )

        print("OrchestratorAI is selecting an agent...")
        try:
            response = self.orchestrator_model.chat(prompt=selection_prompt)
            selected_agent_name = response.content.strip()
            print(f"LLM selected agent: '{selected_agent_name}'")

            for agent in self.available_agents:
                if agent.name.lower() == selected_agent_name.lower():
                    return agent
            print(f"Selected agent '{selected_agent_name}' not found in available agents list.")
            return None
        except Exception as e:
            print(f"Error during agent selection: {e}")
            return None

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

    def run(self, task: Task, max_attempts: int = 3):
        """
        Executes the task by creating and following an execution plan.
        """
        self.memory.user_prompt = task.prompt
        self.memory.append("system", f"Starting task: '{task.prompt}'")
        print(f"\n--- TaskFlow: Running task ---")

        # Step 1: Create execution plan if needed
        if task.needs_plan or self._should_create_plan(task.prompt):
            print("\n--- Creating Execution Plan ---")
            self.current_plan = self._create_execution_plan(task.prompt)
            self.memory.append("system", f"Created execution plan with {len(self.current_plan.steps)} steps")
            
            # Log the plan
            plan_summary = self.current_plan.get_plan_summary()
            print(f"\n{plan_summary}")
            self.memory.append("system", f"Execution plan:\n{plan_summary}")
        else:
            # Create a simple single-step plan
            print("\n--- Creating Simple Plan ---")
            self.current_plan = ExecutionPlan()
            selected_agent = self._select_agent(task.prompt)
            if selected_agent:
                step = PlanStep(
                    step_number=1,
                    agent_name=selected_agent.name,
                    description=f"Handle task: {task.prompt}",
                    input_context=task.prompt
                )
                self.current_plan.add_step(step)
            print(f"Created simple plan with 1 step: {selected_agent.name if selected_agent else 'No agent'}")

        if not self.current_plan.steps:
            print("Failed to create execution plan. No suitable agents found.")
            return

        # Step 2: Execute the plan
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
                print(f"Step {current_step.step_number} completed. Result:\n{result_str}")
                self.memory.append("model", f"Step {current_step.step_number} result: {result_str}")
                
                # Move to next step
                self.current_plan.advance_step()
                
            except Exception as e:
                print(f"Error executing step {current_step.step_number}: {e}")
                self.memory.append("system", f"Error in step {current_step.step_number}: {e}")
                
                # For now, skip failed steps. Could implement retry logic here.
                self.current_plan.advance_step()

        # Step 3: Evaluate overall completion
        if self.current_plan.is_complete():
            print("\n--- Plan Execution Complete ---")
            
            # Get the final result (from the last step)
            if self.step_results:
                final_step_num = max(self.step_results.keys())
                self.final_response = self.step_results[final_step_num]
                
                # Evaluate if the overall task was fulfilled
                if task.needs_eval:
                    final_result_str = json.dumps(self.final_response, indent=2) if isinstance(self.final_response, dict) else str(self.final_response)
                    is_fulfilled, evaluation_message = self._is_task_fulfilled(task.prompt, final_result_str)
                    
                    if is_fulfilled:
                        print("✓ Overall task was fulfilled successfully!")
                        self.memory.append("system", f"Task fulfilled: {evaluation_message}")
                        
                        if task.needs_approval:
                            user_feedback = input("\nTask completed! Do you approve this result? (yes/no): ").lower().strip()
                            if user_feedback == "yes":
                                print("User approved. Task completed.")
                                return
                            else:
                                print("User did not approve. Task marked as incomplete.")
                                return
                        else:
                            print("Task completed successfully.")
                            return
                    else:
                        print(f"✗ Overall task not fulfilled: {evaluation_message}")
                        self.memory.append("system", f"Task not fulfilled: {evaluation_message}")
                else:
                    print("Task execution completed (no evaluation requested).")
                    if task.needs_approval:
                        user_feedback = input("\nDo you approve this result? (yes/no): ").lower().strip()
                        if user_feedback == "yes":
                            print("User approved. Task completed.")
                        else:
                            print("User did not approve.")
        else:
            print(f"\n--- Plan execution incomplete after {max_attempts} attempts ---")
            self.memory.append("system", f"Plan execution incomplete after {max_attempts} attempts")

    def _should_create_plan(self, task_prompt: str) -> bool:
        """
        Determine if a task should have a detailed execution plan created.
        This is a heuristic that can be overridden by the task.needs_plan flag.
        """
        # Keywords that suggest multi-step operations
        multi_step_keywords = [
            "and then", "after", "first", "then", "finally", "review and", 
            "generate and", "create and", "analyze and", "commit", "both"
        ]
        
        task_lower = task_prompt.lower()
        return any(keyword in task_lower for keyword in multi_step_keywords)

    def get_final_response(self):
        """Get the final response from the successfully completed task."""
        return self.final_response
    
    def get_plan_status(self) -> str:
        """Get a summary of the current plan status"""
        if not self.current_plan:
            return "No execution plan created yet."
        return self.current_plan.get_plan_summary()