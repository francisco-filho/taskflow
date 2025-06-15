import json
from typing import Optional, List

from pydantic import BaseModel, Field

from taskflow.agents import Agent, Commiter
from taskflow.memory import Memory
from taskflow.llm import LLMClient

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
        print("OrchestratorAI initialized.")

    def add(self, agent: Agent):
        """
        Adds an agent to the list of available agents.
        """
        self.available_agents.append(agent)
        print(f"Agent '{agent.name}' added to OrchestratorAI.")

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

    def _evaluate_task_completion(self, user_request: str, task_response: str, feedback_context: str = "") -> str:
        """
        Evaluates if the user request was fulfilled by the task response using the orchestrator LLM.
        
        Parameters:
            user_request: The original user request/prompt
            task_response: The response from the agent that handled the task
            feedback_context: Any previous feedback that should be considered in evaluation
            
        Returns:
            A string indicating if the task was completed successfully or not
        """
        # Build evaluation prompt with feedback context if available
        evaluation_prompt = f"""Evaluate if the following user request was properly fulfilled by the task response:

User Request:
{user_request}

Task Response:
{task_response}"""

        if feedback_context:
            evaluation_prompt += f"""

Previous Feedback Context:
{feedback_context}

Please consider the previous feedback when evaluating if the task response addresses the concerns raised."""
        
        evaluation_prompt += """

Please evaluate if the task response adequately fulfills the user's request. 
If the request was fulfilled, respond with 'Task completed successfully'.
If the request was not fulfilled or there are issues, respond with 'Task not completed', followed by two new lines, and then explain the specific reasons why it wasn't fulfilled."""

        print("Evaluating task completion...")
        try:
            response = self.orchestrator_model.chat(prompt=evaluation_prompt)
            evaluation_result = response.content.strip()
            print(f"Evaluation result: {evaluation_result}")
            return evaluation_result
        except Exception as e:
            print(f"Error during task evaluation: {e}")
            return f"Error during evaluation: {e}"

    def run(self, task: Task, max_attempts: int = 3):
        """
        Executes the task by selecting and running the most appropriate agent.
        It also handles evaluation and user approval based on task configuration.

        Parameters:
            task: The task to be executed.
            max_attempts: The maximum number of attempts for an agent to complete the task.
        """
        self.memory.user_prompt = task.prompt
        self.memory.append("system", f"Starting task: '{task.prompt}'")
        print(f"\n--- OrchestratorAI: Running task ---")

        attempt = 0
        # Keep track of all feedback accumulated over iterations
        feedback_history = []
        last_agent_response = None # To store the output of the primary agent

        while attempt < max_attempts:
            attempt += 1
            print(f"\n--- Attempt {attempt}/{max_attempts} for task ---")
            self.memory.append("system", f"Attempt {attempt} for task.")

            # Build the current task prompt with accumulated feedback
            if feedback_history:
                feedback_text = "\n\n".join(feedback_history)
                current_task_prompt = f"{task.prompt}\n\n--- Previous Feedback ---\n{feedback_text}"
            else:
                current_task_prompt = task.prompt

            # Select the primary agent for the task
            selected_agent = self._select_agent(current_task_prompt)

            if not selected_agent:
                print("OrchestratorAI could not identify a suitable agent for this task. Task aborted.")
                self.memory.append("system", "Task aborted: No suitable agent found.")
                return

            print(f"Selected Agent: {selected_agent.name}")
            self.memory.append("system", f"Selected agent: {selected_agent.name}")

            try:
                # All agents now accept a prompt parameter as their first argument
                result = selected_agent.run(prompt=current_task_prompt)
                last_agent_response = result # Store the primary agent's output

                agent_response_str = json.dumps(result, indent=2) if isinstance(result, dict) else str(result)
                print(f"Agent '{selected_agent.name}' responded:\n{agent_response_str}")
                self.memory.append("model", agent_response_str)

                # Evaluation logic based on task configuration
                if task.needs_eval:
                    print(f"\n--- Evaluating task completion ---")
                    # Pass feedback context to evaluation
                    #feedback_context = "\n\n".join(feedback_history) if feedback_history else ""
                    feedback_context = feedback_history[-1] if feedback_history else ""
                    evaluation_result = self._evaluate_task_completion(task.prompt, agent_response_str, feedback_context)
                    self.memory.append("system", f"Task evaluation: {evaluation_result}")

                    if "Task completed successfully" in evaluation_result:
                        print("Task evaluation: Request was fulfilled successfully.")
                        
                        # If evaluation passed, check if user approval is needed
                        if task.needs_approval:
                            user_feedback = input("\nDo you approve this result? (yes/no/retry with feedback): ").lower().strip()
                            self.memory.append("user", f"User feedback: {user_feedback}")

                            if user_feedback == "yes":
                                print("User approved. Task completed.")
                                self.memory.append("system", "Task completed: User approved.")
                                self.final_response = last_agent_response  # Store final response
                                return
                            elif user_feedback == "no":
                                feedback = input("Please provide feedback for the agent (what needs to be improved): ")
                                feedback_history.append(f"User feedback (attempt {attempt}): {feedback}")
                                self.memory.append("system", f"User rejected. Feedback added to history.")
                                print("User rejected. Retrying with feedback.")
                            elif user_feedback == "retry with feedback":
                                feedback = input("Please provide specific feedback for the agent (what should be changed): ")
                                feedback_history.append(f"User retry feedback (attempt {attempt}): {feedback}")
                                self.memory.append("system", f"User requested retry. Feedback added to history.")
                                print("User requested retry with feedback.")
                            else:
                                print("Invalid input. Assuming 'no' and retrying.")
                                feedback_history.append(f"User rejected with invalid input (attempt {attempt}). Please be more precise.")
                                self.memory.append("system", "Invalid user input, assuming rejection. Retrying.")
                        else:
                            print("Task completed: Evaluation passed and no user approval needed.")
                            self.memory.append("system", "Task completed: Evaluation passed and no approval needed.")
                            self.final_response = last_agent_response  # Store final response
                            return
                    else:
                        print("Task evaluation: Request was not fulfilled properly. Retrying with feedback.")
                        feedback_history.append(f"Evaluation feedback (attempt {attempt}): {evaluation_result}")
                        self.memory.append("system", f"Evaluation failed. Feedback added to history.")
                        
                elif task.needs_approval: # If no evaluation but approval needed
                    user_feedback = input("\nDo you approve this result? (yes/no/retry with feedback): ").lower().strip()
                    self.memory.append("user", f"User feedback: {user_feedback}")

                    if user_feedback == "yes":
                        print("User approved. Task completed.")
                        self.memory.append("system", "Task completed: User approved.")
                        self.final_response = last_agent_response  # Store final response
                        return
                    elif user_feedback == "no":
                        feedback = input("Please provide feedback for the agent (what needs to be improved): ")
                        feedback_history.append(f"User feedback (attempt {attempt}): {feedback}")
                        self.memory.append("system", f"User rejected. Feedback added to history.")
                        print("User rejected. Retrying with feedback.")
                    elif user_feedback == "retry with feedback":
                        feedback = input("Please provide specific feedback for the agent (what should be changed): ")
                        feedback_history.append(f"User retry feedback (attempt {attempt}): {feedback}")
                        self.memory.append("system", f"User requested retry. Feedback added to history.")
                        print("User requested retry with feedback.")
                    else:
                        print("Invalid input. Assuming 'no' and retrying.")
                        feedback_history.append(f"User rejected with invalid input (attempt {attempt}). Please be more precise.")
                        self.memory.append("system", "Invalid user input, assuming rejection. Retrying.")
                else:
                    print(f"Task completed: Agent '{selected_agent.name}' finished and no evaluation/approval needed.")
                    self.memory.append("system", "Task completed: No evaluation/approval needed.")
                    self.final_response = last_agent_response  # Store final response
                    return

            except Exception as e:
                print(f"An error occurred during agent '{selected_agent.name}' execution: {e}")
                feedback_history.append(f"Error (attempt {attempt}): {e}. Please try again.")
                self.memory.append("system", f"Error during agent execution: {e}. Retrying.")

        print(f"\n--- Max attempts ({max_attempts}) reached for task. Task incomplete. ---")
        self.memory.append("system", f"Task incomplete: Max attempts ({max_attempts}) reached.")
        # Even if max attempts reached, store the last response if available
        if last_agent_response:
            self.final_response = last_agent_response

    def get_final_response(self):
        """Get the final response from the successfully completed task."""
        return self.final_response