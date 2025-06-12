from collections import deque
from typing import Optional, List, Dict

class Memory:
    """
    Responsible for storing interactions (prompts, responses) relative to the task.
    It evicts older messages if the maximum number of interactions is exceeded.
    """
    user_prompt: str
    plan: Optional[str]
    interactions: deque
    max_interactions: int

    def __init__(self, user_prompt: str, max_interaction_size: int = 24):
        self.user_prompt = user_prompt
        self.plan = None
        self.interactions = deque(maxlen=max_interaction_size)
        self.max_interactions = max_interaction_size

    def append(self, role: str, content: str):
        """
        Appends an interaction to the memory.

        Parameters:
            role: The role of the interaction (e.g., "user", "model", "system").
            content: The content of the interaction.
        """
        self.interactions.append({"role": role, "content": content})
        print(f"Memory: Appended {role} interaction. Current size: {len(self.interactions)}")

    def get_history(self) -> List[Dict[str, str]]:
        """
        Returns the current interaction history.
        """
        return list(self.interactions)

    def set_plan(self, plan: str):
        """
        Sets the plan for the task.
        """
        self.plan = plan