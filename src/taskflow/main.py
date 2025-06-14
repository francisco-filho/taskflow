import os
import argparse
from pathlib import Path
import json

from dotenv import load_dotenv

from taskflow.llm import get_client
from taskflow.flow import Task, TaskFlow
from taskflow.agents import Commiter, Evaluator, Reviewer
from taskflow.tools import diff_tool
from taskflow.mock import create_temp_git_repo

# TODO: improve the memory, how to reproduce after a failure?
# TODO: extract the prompts from the code
# TODO: add costs/token usage


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="TaskFlow AI - Automated git commit and review system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --project /path/to/project --task commit
  %(prog)s --project /path/to/project --task review
  %(prog)s --project /path/to/project --task commit --model gemini-2.5-flash-preview-05-20
  %(prog)s --create-temp-repo --task commit
        """
    )
    
    parser.add_argument(
        "--project", "-p",
        type=str,
        help="Path to the project directory (default: current directory)"
    )
    
    parser.add_argument(
        "--task", "-t",
        choices=["commit", "review"],
        default="review",
        help="Task to perform: 'commit' or 'review' (default: review)"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        help="AI model to use (overrides DEFAULT_MODEL env var)"
    )
    
    parser.add_argument(
        "--create-temp-repo",
        action="store_true",
        help="Create a temporary git repository for testing"
    )
    
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum number of attempts for task execution (default: 3)"
    )
    
    parser.add_argument(
        "--needs-approval",
        action="store_true",
        help="Require approval for the task execution"
    )

    parser.add_argument(
        "--needs-eval",
        action="store_true",
        help="Enable LLM evaluation to check if the user request was fulfilled (default: True)"
    )
    
    return parser.parse_args()

def validate_project_directory(project_dir):
    """Validate that the project directory exists and is a git repository."""
    project_path = Path(project_dir)
    
    if not project_path.exists():
        raise ValueError(f"Project directory does not exist: {project_dir}")
    
    if not project_path.is_dir():
        raise ValueError(f"Path is not a directory: {project_dir}")
    
    # Check if it's a git repository
    git_dir = project_path / ".git"
    if not git_dir.exists():
        raise ValueError(f"Directory is not a git repository: {project_dir}")
    
    return project_path.resolve()

def create_task(task_type, project_dir, needs_approval=False, needs_eval=False):
    """Create a task based on the task type."""
    if task_type == "commit":
        return Task(
            prompt=f"""
            Generate a commit message for the staged changes in the project '{project_dir}'.
            """,
            needs_approval=needs_approval,
            needs_eval=needs_eval
        )
    elif task_type == "review":
        return Task(
            prompt=f"""
            Generate a concise REVIEW about changes in the project '{project_dir}'.
            """,
            needs_approval=needs_approval,
            needs_eval=needs_eval
        )
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def initialize_agents(client):
    """Initialize all agents with their respective configurations."""
    commiter_agent = Commiter(
        model=client,
        system_prompt="""
You are a senior programmer that explains hard concepts clearly and are very succinctly in your messages. You can evaluate changes in a project just by reading the diff output from git.

You MUST use the `diff_tool` to get the changes in the project.

Respond ONLY in the JSON format (example):

{"message": "Refactor GitReviewer for improved LLM integration and REPL functionality", "details": ["Introduced a `_get_config` method in `LLMGoogle` to centralize LLM calls.", "Refactored `main.py` to use a new `init_repl` function, streamlining the application's entry point and focusing on a REPL interface.", "Moved the `Message` Pydantic model to a dedicated `models.py`"]}
""",
        available_tools={'diff_tool': diff_tool}
    )

    evaluator_agent = Evaluator(
        model=client,
        system_prompt="""
You are a senior programmer that has attention to details and likes very clear texts. You made code reviews and evaluate the quality of the commit messages based on the diff changes.
You MUST use the `diff_tool` to get the changes in the project.
If your evaluation is positive, just respond with 'Commit message accepted', but
if the commit message has any problems respond with 'Bad commit message', two new lines and the motive.
""",
        available_tools={'diff_tool': diff_tool}
    )

    reviewer_agent = Reviewer(
        model=client,
        system_prompt="""
You are a meticulous code reviewer. Your task is to provide a concise and constructive review of the given code changes, focusing on clarity, potential issues, and adherence to best practices. Summarize the key changes and any recommendations.
You MUST use the `diff_tool` to get the staged changes in the project.
""",
        available_tools={'diff_tool': diff_tool}
    )
    
    return commiter_agent, evaluator_agent, reviewer_agent

def format_final_response(response):
    """Format the final response for clean output."""
    if isinstance(response, dict):
        return json.dumps(response, indent=2)
    return str(response)

def main():
    """Main function to run the TaskFlow CLI."""
    args = parse_arguments()
    
    # Load environment variables
    load_dotenv()
    
    print("Initializing TaskFlow AI system...")
    
    # Determine the model to use
    model_name = args.model or os.getenv("DEFAULT_MODEL", "gemini-2.5-flash-preview-05-20")
    print(f"Using model: {model_name}")
    
    # Initialize the client
    client = get_client(model_name)
    
    # Determine project directory
    if args.create_temp_repo:
        project_dir = os.path.abspath(os.path.join(os.getcwd(), "tmp_test_project"))
        print(f"Creating temporary git repository at: {project_dir}")
        create_temp_git_repo(project_dir)
    else:
        project_dir = args.project or os.getcwd()
        try:
            project_dir = validate_project_directory(project_dir)
            print(f"Using project directory: {project_dir}")
        except ValueError as e:
            print(f"Error: {e}")
            return 1
    
    # Create the task
    try:
        task = create_task(args.task, project_dir, args.needs_approval, args.needs_eval)
        print(f"Created task: {args.task}")
    except ValueError as e:
        print(f"Error creating task: {e}")
        return 1
    
    # Initialize agents
    commiter_agent, evaluator_agent, reviewer_agent = initialize_agents(client)
    
    # Create and configure the flow
    flow = TaskFlow(model=client)
    flow.add(commiter_agent)
    flow.add(evaluator_agent)
    flow.add(reviewer_agent)
    
    # Run the task
    try:
        print(f"\nRunning {args.task} task (max attempts: {args.max_attempts})...")
        flow.run(task, max_attempts=args.max_attempts)
        
        # Print the final response as the last thing
        final_response = flow.get_final_response()
        if final_response:
            print("\n" + "="*50)
            print("FINAL RESULT:")
            print("="*50)
            print(format_final_response(final_response))
        else:
            print("\n" + "="*50)
            print("No final response available - task may have failed")
            print("="*50)
            
    except Exception as e:
        print(f"Error during task execution: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())