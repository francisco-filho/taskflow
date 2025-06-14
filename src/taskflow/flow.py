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

    def run(self, task: Task, max_attempts: int = 3):
        """
        Executes the task by selecting and running the most appropriate agent.
        It also handles evaluation by an Evaluator agent and user approval.

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
        last_agent_response = None # To store the output of the primary agent (e.g., Commiter)
        evaluator_agent = next((agent for agent in self.available_agents if agent.name == "Evaluator"), None)

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

                # Evaluation and approval logic
                if evaluator_agent and isinstance(selected_agent, Commiter):
                    print(f"\n--- Delegating to Evaluator agent ---")
                    # Pass the commit message to the evaluator along with the prompt
                    evaluation_result = evaluator_agent.run(
                        prompt=current_task_prompt, 
                        commit_message=last_agent_response
                    )
                    print(f"Evaluator responded:\n{evaluation_result}")
                    self.memory.append("system", f"Evaluator feedback: {evaluation_result}")

                    if "Commit message accepted" in evaluation_result:
                        print("Evaluator accepted the commit message.")
                        if task.needs_approval:
                            user_feedback = input("\nDo you approve this result? (yes/no/retry with feedback): ").lower().strip()
                            self.memory.append("user", f"User feedback: {user_feedback}")

                            if user_feedback == "yes":
                                print("User approved. Task completed.")
                                self.memory.append("system", "Task completed: User approved.")
                                return
                            elif user_feedback == "no":
                                feedback = input("Please provide feedback for the agent (e.g., 'The message is too long.'): ")
                                feedback_history.append(f"User feedback (attempt {attempt}): {feedback}")
                                self.memory.append("system", f"User rejected. Feedback added to history.")
                                print("User rejected. Retrying with feedback.")
                            elif user_feedback == "retry with feedback":
                                feedback = input("Please provide specific feedback for the agent: ")
                                feedback_history.append(f"User retry feedback (attempt {attempt}): {feedback}")
                                self.memory.append("system", f"User requested retry. Feedback added to history.")
                                print("User requested retry with feedback.")
                            else:
                                print("Invalid input. Assuming 'no' and retrying.")
                                feedback_history.append(f"User rejected with invalid input (attempt {attempt}). Please be more precise.")
                                self.memory.append("system", "Invalid user input, assuming rejection. Retrying.")
                        else:
                            print("Task completed: No user approval needed and evaluator accepted.")
                            self.memory.append("system", "Task completed: Evaluator accepted and no approval needed.")
                            return
                    else:
                        print("Evaluator rejected the commit message. Retrying with feedback.")
                        feedback_history.append(f"Evaluator feedback (attempt {attempt}): {evaluation_result}")
                        self.memory.append("system", f"Evaluator rejected. Feedback added to history.")
                elif not evaluator_agent and task.needs_approval: # If no evaluator but approval needed
                    user_feedback = input("\nDo you approve this result? (yes/no/retry with feedback): ").lower().strip()
                    self.memory.append("user", f"User feedback: {user_feedback}")

                    if user_feedback == "yes":
                        print("User approved. Task completed.")
                        self.memory.append("system", "Task completed: User approved.")
                        return
                    elif user_feedback == "no":
                        feedback = input("Please provide feedback for the agent (e.g., 'The message is too long.'): ")
                        feedback_history.append(f"User feedback (attempt {attempt}): {feedback}")
                        self.memory.append("system", f"User rejected. Feedback added to history.")
                        print("User rejected. Retrying with feedback.")
                    elif user_feedback == "retry with feedback":
                        feedback = input("Please provide specific feedback for the agent: ")
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
                    return

            except Exception as e:
                print(f"An error occurred during agent '{selected_agent.name}' execution: {e}")
                feedback_history.append(f"Error (attempt {attempt}): {e}. Please try again.")
                self.memory.append("system", f"Error during agent execution: {e}. Retrying.")

        print(f"\n--- Max attempts ({max_attempts}) reached for task. Task incomplete. ---")
        self.memory.append("system", f"Task incomplete: Max attempts ({max_attempts}) reached.")