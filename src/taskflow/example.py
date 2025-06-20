#!/usr/bin/env python3
"""
Example usage of the enhanced TaskFlow with memory persistence and resume capability.
"""

import os
import sys
from taskflow.flow import TaskFlow, Task
from taskflow.llm import LLMClient
from taskflow.agents import Agent

# Example agent implementations (you'll need to replace with your actual agents)
class DiffAgent(Agent):
    def __init__(self):
        super().__init__(name="DiffAgent", description="Generate and analyze git diffs")
    
    def run(self, prompt: str):
        # Your diff implementation here
        return {"diff_output": "example diff content", "files_changed": 3}

class CommitAgent(Agent):
    def __init__(self):
        super().__init__(name="CommitAgent", description="Commit changes to git repository")
    
    def run(self, prompt: str):
        # Your commit implementation here
        return {"commit_hash": "abc123", "message": "Auto-generated commit"}

class ReviewAgent(Agent):
    def __init__(self):
        super().__init__(name="ReviewAgent", description="Review code changes and provide feedback")
    
    def run(self, prompt: str):
        # Your review implementation here
        return {"review": "Code looks good", "issues": [], "approved": True}

def main():
    """Main execution function"""
    
    # Memory file path - this is where the execution state will be saved
    memory_file = "./taskflow_memory.json"
    
    # Initialize LLM client (replace with your actual implementation)
    llm_client = LLMClient()  # Your LLM client initialization
    
    # Initialize TaskFlow with memory file
    print("Initializing TaskFlow...")
    taskflow = TaskFlow(model=llm_client, memory_file_path=memory_file)
    
    # Add available agents
    taskflow.add(DiffAgent())
    taskflow.add(CommitAgent())
    taskflow.add(ReviewAgent())
    
    # Check if there's a task to resume
    resume_info = taskflow.check_resume_status()
    if resume_info:
        print("\n=== RESUME AVAILABLE ===")
        print(f"Previous task: {resume_info['user_prompt']}")
        print(f"Total steps: {resume_info['total_steps']}")
        print(f"Completed steps: {resume_info['completed_steps']}")
        print(f"Remaining steps: {resume_info['remaining_steps']}")
        print("\nStep results so far:")
        for step_num, result in resume_info['step_results'].items():
            print(f"  Step {step_num}: {result}")
        
        choice = input("\nDo you want to resume the previous task? (yes/no): ").lower().strip()
        if choice == 'yes':
            print("Resuming previous task...")
            success = taskflow.resume_task(max_attempts=10)
            if success:
                print("Task resumed and completed successfully!")
            else:
                print("Task resumption failed or was not completed.")
            return
        else:
            print("Starting fresh task (previous task will remain in memory)...")
    
    # Define a new task
    user_prompt = input("Enter your task (or press Enter for example): ").strip()
    if not user_prompt:
        user_prompt = "Generate a diff of the changes and then commit them with an appropriate message"
    
    # Create task object
    task = Task(
        prompt=user_prompt,
        needs_plan=True,  # Let TaskFlow decide based on complexity
        needs_approval=True,  # Ask for user approval
        needs_eval=True   # Evaluate if task was fulfilled
    )
    
    print(f"\nExecuting task: {task.prompt}")
    
    try:
        # Run the task
        taskflow.run(task, max_attempts=10)
        
        # Get final response
        final_response = taskflow.get_final_response()
        if final_response:
            print(f"\nFinal Response: {final_response}")
        
        # Show memory summary
        print("\n=== MEMORY SUMMARY ===")
        print(taskflow.get_memory_summary())
        
    except KeyboardInterrupt:
        print("\n\nTask interrupted! You can resume later by running this script again.")
        print(f"Memory saved to: {memory_file}")
        
    except Exception as e:
        print(f"\nError during task execution: {e}")
        print(f"Memory saved to: {memory_file}")
        print("You can try to resume by running the script again.")

def clear_memory():
    """Utility function to clear memory"""
    memory_file = "./taskflow_memory.json"
    if os.path.exists(memory_file):
        os.remove(memory_file)
        print(f"Cleared memory file: {memory_file}")
    else:
        print("No memory file found.")

def show_memory_status():
    """Utility function to show current memory status"""
    memory_file = "./taskflow_memory.json"
    llm_client = LLMClient()  # Your LLM client
    taskflow = TaskFlow(model=llm_client, memory_file_path=memory_file)
    
    resume_info = taskflow.check_resume_status()
    if resume_info:
        print("=== CURRENT MEMORY STATUS ===")
        print(f"Task: {resume_info['user_prompt']}")
        print(f"Total steps: {resume_info['total_steps']}")
        print(f"Completed: {resume_info['completed_steps']}")
        print(f"Remaining: {resume_info['remaining_steps']}")
        print("\nFull memory summary:")
        print(taskflow.get_memory_summary())
    else:
        print("No incomplete task found in memory.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "clear":
            clear_memory()
        elif sys.argv[1] == "status":
            show_memory_status()
        else:
            print("Usage: python taskflow_example.py [clear|status]")
            print("  clear  - Clear the memory file")
            print("  status - Show current memory status")
            print("  (no args) - Run TaskFlow normally")
    else:
        main()