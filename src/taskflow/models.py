from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# Tasks
class Request(BaseModel):
    """
    Represents a request from the user.
    """
    prompt: str = Field(..., description="The original user prompt for the task.")
    needs_plan: bool = Field(False, description="True if the task requires an explicit plan from an LLM.")
    needs_approval: bool = Field(True, description="True if the final result of the task needs user approval.")
    needs_eval: bool = Field(True, description="True if you want a LLM call to evaluate if the user request was fulfilled.")

# Planner
class PlanStepModel(BaseModel):
    """Pydantic model for individual plan steps"""
    step_number: int = Field(..., description="Sequential number of the step")
    agent_name: str = Field(..., description="Name of the agent that will execute this step")
    description: str = Field(..., description="Description of what this step will accomplish")
    input_context: str = Field(default="", description="Context or input this step needs")
    depends_on: List[int] = Field(default_factory=list, description="List of step numbers this step depends on")

class PlanningResponse(BaseModel):
    """Pydantic model for the planning prompt response"""
    requires_planning: bool = Field(..., description="Whether the task requires multiple steps or agents")
    reasoning: str = Field(..., description="Explanation of why planning is or isn't needed")
    steps: List[PlanStepModel] = Field(default_factory=list, description="List of execution steps")

# Commiter agent
class CommitMessage(BaseModel):
    subject: str
    body: str

class UserNotApprovedException(Exception):
    pass
