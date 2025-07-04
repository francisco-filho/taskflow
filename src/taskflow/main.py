import os
from pathlib import Path
import json

from dotenv import load_dotenv

from taskflow.llm import get_client
from taskflow.flow import TaskFlow
from taskflow.agents import Tool
from taskflow.agents.commiter import Commiter
from taskflow.agents.reviewer import Reviewer
from taskflow.agents.diff import DiffMessager
from taskflow.agents.evaluator import Evaluator
from taskflow.agents.techwritter import TechnicalWriter
from taskflow.tools import ListFilesTool, diff_tool, commit_tool, list_files_tool, read_file_tool
from taskflow.mock import create_temp_git_repo
from taskflow.tool.github_diff import GithubPullRequestDiffTool
from taskflow.tool.gitlab_diff import GitlabMergeRequestDiffTool
from taskflow.cli import create_task, parse_arguments, initialize_agents

# github_tool = GithubPullRequestDiffTool()
# gitlab_tool = GitlabMergeRequestDiffTool()
# list_tool = ListFilesTool()


def validate_project_directory(project_dir, require_git=True):
    """Validate that the project directory exists and optionally is a git repository."""
    project_path = Path(project_dir)
    
    if not project_path.exists():
        raise ValueError(f"Project directory does not exist: {project_dir}")
    
    if not project_path.is_dir():
        raise ValueError(f"Path is not a directory: {project_dir}")
    
    # Check if it's a git repository (only for git-related tasks)
    if require_git:
        git_dir = project_path / ".git"
        if not git_dir.exists():
            raise ValueError(f"Directory is not a git repository: {project_dir}")
    
    return project_path.resolve()



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
        project_dir = "/tmp/tmp_test_project"
        print(f"Creating temporary git repository at: {project_dir}")
        create_temp_git_repo(project_dir)
    else:
        project_dir = args.project or os.getcwd()
        try:
            # Documentation task doesn't require git repository
            require_git = args.task != "doc"
            project_dir = validate_project_directory(project_dir, require_git=require_git)
            print(f"Using project directory: {project_dir}")
        except ValueError as e:
            print(f"Error: {e}")
            return 1
    
    # Create the task
    try:
        task = create_task(
            args.task, 
            project_dir, 
            args.needs_approval, 
            args.needs_eval,
            args.file_name,
            args.file_ext,
            args.prompt
        )
        print(f"Created task: {args.task}")
    except ValueError as e:
        print(f"Error creating task: {e}")
        return 1
    
    # Initialize agents
    diff_messager_agent, commiter_agent, evaluator_agent, reviewer_agent, technical_writer_agent, arch_agent = initialize_agents(client)
    
    # Create and configure the flow
    flow = TaskFlow(model=client)
    flow.add(diff_messager_agent)
    flow.add(commiter_agent)
    flow.add(evaluator_agent)
    flow.add(reviewer_agent)
    flow.add(technical_writer_agent)
    flow.add(arch_agent)
    
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
