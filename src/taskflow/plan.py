import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from taskflow.agents import Agent
from taskflow.llm import LLMClient

from taskflow.models import PlanningResponse

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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'step_number': self.step_number,
            'agent_name': self.agent_name,
            'description': self.description,
            'input_context': self.input_context,
            'depends_on': self.depends_on
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PlanStep':
        """Create from dictionary"""
        return cls(
            step_number=data['step_number'],
            agent_name=data['agent_name'],
            description=data['description'],
            input_context=data.get('input_context', ''),
            depends_on=data.get('depends_on', [])
        )

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
    
    def to_dict_list(self) -> List[Dict[str, Any]]:
        """Convert steps to list of dictionaries for serialization"""
        return [step.to_dict() for step in self.steps]
    
    @classmethod
    def from_dict_list(cls, steps_data: List[Dict[str, Any]]) -> 'ExecutionPlan':
        """Create ExecutionPlan from list of step dictionaries"""
        plan = cls()
        for step_data in steps_data:
            plan.add_step(PlanStep.from_dict(step_data))
        return plan


class Planner:
    """
    Responsible for analyzing tasks and creating execution plans.
    """
    
    def __init__(self, model: LLMClient, available_agents: List[Agent]):
        """
        Initialize the Planner.
        
        Args:
            model: LLM client for planning decisions
            available_agents: List of available agents to create plans with
        """
        self.model = model
        self.available_agents = available_agents
    
    def update_available_agents(self, agents: List[Agent]):
        """Update the list of available agents."""
        self.available_agents = agents
    
    def should_create_detailed_plan(self, task_prompt: str) -> bool:
        """
        Determine if a task should have a detailed execution plan created.
        """
        # Keywords that suggest multi-step operations
        multi_step_keywords = [
            "and then", "after", "first", "then", "finally", "review and", 
            "generate and", "create and", "analyze and", "commit", "both"
        ]
        
        task_lower = task_prompt.lower()
        return any(keyword in task_lower for keyword in multi_step_keywords)
    
    def create_execution_plan(self, task_prompt: str) -> ExecutionPlan:
        """
        Creates a detailed execution plan for tasks.
        
        Args:
            task_prompt: The user's task prompt
            
        Returns:
            ExecutionPlan with the steps needed to complete the task
        """
        if not self.available_agents:
            print("No agents available to create plan.")
            return ExecutionPlan()

        agent_descriptions = "\n".join([f"- {a.name}: {a.description}" for a in self.available_agents])
        
        planning_prompt = f"""Given the user's task: '{task_prompt}', create a detailed execution plan.

Available Agents:
{agent_descriptions}

Analyze the task and determine if it requires multiple steps or agents. If so, break it down into sequential steps.

Please respond with a JSON object in this format, without markdown quotation marks:
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
- Do not add agents that do not contribute with the user goal

Examples of tasks that need planning:
- "Propose a commit message" (1 step: Generate commit message)
- "Generate a commit message and then commit the changes" (2 steps: generate message, then commit)
- "Review the code, make changes, then commit" (3 steps: review, modify, commit)
- "Analyze the diff and create a detailed report" (might need 1 or 2 steps depending on complexity)

Do not use Markdown. Respond as JSON"""

        print("Creating execution plan...")
        
        try:
            response = self.model.chat(prompt=planning_prompt, output=PlanningResponse)
            print(f"Planning response: {response.content}")
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
                selected_agent = self._select_best_agent(task_prompt)
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
            selected_agent = self._select_best_agent(task_prompt)
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
    
    def _select_best_agent(self, task_prompt: str) -> Optional[Agent]:
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

        print("Planner is selecting best agent...")
        try:
            response = self.model.chat(prompt=selection_prompt)
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