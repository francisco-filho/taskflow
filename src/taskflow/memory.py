import json
import os
from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum

class EventType(Enum):
    USER_PROMPT = "user_prompt"
    PLAN_CREATED = "plan_created"
    STEP_STARTED = "step_started"
    AGENT_INPUT = "agent_input"
    AGENT_OUTPUT = "agent_output"
    STEP_COMPLETED = "step_completed"
    STEP_FAILED = "step_failed"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    EVALUATION = "evaluation"
    USER_APPROVAL = "user_approval"
    SYSTEM_EVENT = "system_event"

@dataclass
class MemoryEvent:
    """Represents a single event in the task execution history"""
    timestamp: str
    event_type: EventType
    step_number: Optional[int] = None
    agent_name: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'timestamp': self.timestamp,
            'event_type': self.event_type.value,
            'step_number': self.step_number,
            'agent_name': self.agent_name,
            'data': self.data,
            'message': self.message
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEvent':
        """Create from dictionary loaded from JSON"""
        return cls(
            timestamp=data['timestamp'],
            event_type=EventType(data['event_type']),
            step_number=data.get('step_number'),
            agent_name=data.get('agent_name'),
            data=data.get('data'),
            message=data.get('message')
        )

@dataclass
class ExecutionState:
    """Represents the current state of task execution"""
    user_prompt: str
    plan_steps: List[Dict[str, Any]]
    current_step: int
    step_results: Dict[int, Any]
    is_complete: bool
    final_response: Optional[Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExecutionState':
        return cls(**data)

class PersistentMemory:
    """
    Enhanced memory system that persists execution state to disk,
    allowing for task resumption after crashes or interruptions.
    """
    
    def __init__(self, memory_file_path: Optional[str] = None, max_interaction_size: int = 24):
        self.memory_file_path = memory_file_path
        self.max_interaction_size = max_interaction_size
        self.events: List[MemoryEvent] = []
        self.execution_state: Optional[ExecutionState] = None
        
        # If memory file path is provided, try to load existing state
        if self.memory_file_path and os.path.exists(self.memory_file_path):
            self._load_from_file()
            print(f"Loaded existing memory from: {self.memory_file_path}")
        else:
            print("Starting with fresh memory")
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string"""
        return datetime.now().isoformat()
    
    def _save_to_file(self):
        """Save current state to memory file"""
        if not self.memory_file_path:
            return
        
        try:
            memory_data = {
                'events': [event.to_dict() for event in self.events],
                'execution_state': self.execution_state.to_dict() if self.execution_state else None,
                'last_updated': self._get_timestamp()
            }
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.memory_file_path), exist_ok=True)
            
            with open(self.memory_file_path, 'w', encoding='utf-8') as f:
                json.dump(memory_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Error saving memory to file: {e}")
    
    def _load_from_file(self):
        """Load state from memory file"""
        if not self.memory_file_path or not os.path.exists(self.memory_file_path):
            return
        
        try:
            with open(self.memory_file_path, 'r', encoding='utf-8') as f:
                memory_data = json.load(f)
            
            # Load events
            self.events = [MemoryEvent.from_dict(event_data) 
                          for event_data in memory_data.get('events', [])]
            
            # Load execution state
            if memory_data.get('execution_state'):
                self.execution_state = ExecutionState.from_dict(memory_data['execution_state'])
                
            print(f"Loaded {len(self.events)} events from memory file")
            
        except Exception as e:
            print(f"Error loading memory from file: {e}")
    
    def record_event(self, event_type: EventType, step_number: Optional[int] = None, 
                    agent_name: Optional[str] = None, data: Optional[Dict[str, Any]] = None,
                    message: Optional[str] = None):
        """Record a new event in memory"""
        event = MemoryEvent(
            timestamp=self._get_timestamp(),
            event_type=event_type,
            step_number=step_number,
            agent_name=agent_name,
            data=data,
            message=message
        )
        
        self.events.append(event)
        
        # Keep only recent events to prevent memory bloat
        if len(self.events) > self.max_interaction_size:
            self.events = self.events[-self.max_interaction_size:]
        
        # Save to file after each event
        self._save_to_file()
        
        print(f"Recorded event: {event_type.value}" + 
              (f" (Step {step_number})" if step_number else "") +
              (f" - {agent_name}" if agent_name else ""))
    
    def update_execution_state(self, user_prompt: str, plan_steps: List[Dict[str, Any]], 
                             current_step: int, step_results: Dict[int, Any], 
                             is_complete: bool, final_response: Optional[Any] = None):
        """Update the current execution state"""
        self.execution_state = ExecutionState(
            user_prompt=user_prompt,
            plan_steps=plan_steps,
            current_step=current_step,
            step_results=step_results,
            is_complete=is_complete,
            final_response=final_response
        )
        self._save_to_file()
    
    def get_execution_state(self) -> Optional[ExecutionState]:
        """Get the current execution state"""
        return self.execution_state
    
    def can_resume(self) -> bool:
        """Check if there's a valid state to resume from"""
        return (self.execution_state is not None and 
                not self.execution_state.is_complete and
                len(self.execution_state.plan_steps) > 0)
    
    def get_resume_info(self) -> Optional[Dict[str, Any]]:
        """Get information about what can be resumed"""
        if not self.can_resume():
            return None
        
        state = self.execution_state
        completed_steps = list(state.step_results.keys())
        remaining_steps = [step for step in state.plan_steps 
                          if step['step_number'] > state.current_step]
        
        return {
            'user_prompt': state.user_prompt,
            'total_steps': len(state.plan_steps),
            'completed_steps': completed_steps,
            'current_step': state.current_step,
            'remaining_steps': len(remaining_steps),
            'step_results': state.step_results
        }
    
    def get_events_by_type(self, event_type: EventType) -> List[MemoryEvent]:
        """Get all events of a specific type"""
        return [event for event in self.events if event.event_type == event_type]
    
    def get_events_by_step(self, step_number: int) -> List[MemoryEvent]:
        """Get all events for a specific step"""
        return [event for event in self.events if event.step_number == step_number]
    
    def get_latest_event(self, event_type: EventType) -> Optional[MemoryEvent]:
        """Get the most recent event of a specific type"""
        events = self.get_events_by_type(event_type)
        return events[-1] if events else None
    
    def clear_memory(self):
        """Clear all memory (useful for starting fresh)"""
        self.events.clear()
        self.execution_state = None
        if self.memory_file_path and os.path.exists(self.memory_file_path):
            os.remove(self.memory_file_path)
        print("Memory cleared")
    
    def get_summary(self) -> str:
        """Get a human-readable summary of the memory state"""
        summary = f"Memory Summary (Total events: {len(self.events)})\n"
        summary += "=" * 50 + "\n"
        
        if self.execution_state:
            state = self.execution_state
            summary += f"Task: {state.user_prompt}\n"
            summary += f"Total Steps: {len(state.plan_steps)}\n"
            summary += f"Current Step: {state.current_step}\n"
            summary += f"Completed Steps: {list(state.step_results.keys())}\n"
            summary += f"Status: {'Complete' if state.is_complete else 'In Progress'}\n\n"
        
        # Show recent events
        recent_events = self.events[-10:] if len(self.events) > 10 else self.events
        summary += "Recent Events:\n"
        summary += "-" * 20 + "\n"
        
        for event in recent_events:
            summary += f"[{event.timestamp}] {event.event_type.value}"
            if event.step_number:
                summary += f" (Step {event.step_number})"
            if event.agent_name:
                summary += f" - {event.agent_name}"
            if event.message:
                summary += f": {event.message}"
            summary += "\n"
        
        return summary
    
    # Legacy methods for backward compatibility
    def append(self, role: str, content: str):
        """Legacy method for backward compatibility"""
        if role == "system":
            self.record_event(EventType.SYSTEM_EVENT, message=content)
        else:
            self.record_event(EventType.SYSTEM_EVENT, message=f"{role}: {content}")
    
    @property
    def user_prompt(self) -> str:
        """Get user prompt for backward compatibility"""
        if self.execution_state:
            return self.execution_state.user_prompt
        return ""
    
    @user_prompt.setter
    def user_prompt(self, value: str):
        """Set user prompt for backward compatibility"""
        if not self.execution_state:
            self.execution_state = ExecutionState(
                user_prompt=value,
                plan_steps=[],
                current_step=0,
                step_results={},
                is_complete=False
            )
        else:
            self.execution_state.user_prompt = value
        self._save_to_file()

# For backward compatibility
Memory = PersistentMemory