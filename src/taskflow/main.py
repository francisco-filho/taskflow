import os
import argparse
from pathlib import Path
import json

from dotenv import load_dotenv

from taskflow.llm import get_client
from taskflow.flow import Task, TaskFlow
from taskflow.agents import Commiter, Evaluator
from taskflow.agents.reviewer import Reviewer
from taskflow.agents.diff import DiffMessager
from taskflow.agents.techwritter import TechnicalWriter
from taskflow.tools import diff_tool, commit_tool, list_files_tool, read_file_tool
from taskflow.mock import create_temp_git_repo
from taskflow.tool.github_diff import GithubPullRequestDiffTool

github_tool = GithubPullRequestDiffTool()

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="TaskFlow AI - Automated git commit, review, and documentation system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --project /path/to/project --task diff
  %(prog)s --project /path/to/project --task review
  %(prog)s --project /path/to/project --task commit
  %(prog)s --project /path/to/project --task doc
  %(prog)s --project /path/to/project --task diff --model gemini-2.5-flash-preview-05-20
  %(prog)s --create-temp-repo --task diff
        """
    )
    
    parser.add_argument(
        "--project", "-p",
        type=str,
        help="Path to the project directory (default: current directory)"
    )
    
    parser.add_argument(
        "--task", "-t",
        choices=["diff", "review", "commit", "doc"],
        default="review",
        help="Task to perform: 'diff', 'review', 'commit', or 'doc' (default: review)"
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
        default=13,
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
    
    # Documentation-specific arguments
    parser.add_argument(
        "--file-name",
        type=str,
        help="Specific filename to document (for doc task)"
    )
    
    parser.add_argument(
        "--file-ext",
        type=str,
        help="File extension to document (for doc task, e.g., 'py', 'js')"
    )
    
    return parser.parse_args()

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

def create_task(task_type, project_dir, needs_approval=False, needs_eval=False, file_name=None, file_ext=None):
    """Create a task based on the task type."""
    if task_type == "diff":
        return Task(
            prompt=f"""
Propose a commit message for the staged changes in the project '{project_dir}' with a evaluation
            """,
#             prompt=f"""
# Propose a commit message for the staged changes in the project 'https://github.com/francisco-filho/taskflow/pull/1' them make a evaluation of the commit message
#             """,
            needs_approval=needs_approval,
            needs_eval=needs_eval,
            needs_plan=True
        )
    elif task_type == "review":
        return Task(
            prompt=f"""
            Generate a concise REVIEW about changes in the project '{project_dir}'.
            """,
#             prompt=f"""
# Generate a concise review about changes informed in the pull request 'https://github.com/francisco-filho/taskflow/pull/1'
# """,
            needs_approval=needs_approval,
            needs_eval=needs_eval,
            needs_plan=True
        )
    elif task_type == "commit":
        return Task(
            prompt=f"""
            Generate a commit message for the staged changes in the project '{project_dir}' and commit the changes. Do the commit.
            """,
            needs_approval=needs_approval,
            needs_eval=False,
            needs_plan=True
        )
    elif task_type == "doc":
        # Build documentation prompt based on file specifications
        doc_prompt = f"Generate technical documentation for the code files in the project '{project_dir}'. The file can be located in any sub-directory of the project, so do not assume is in the root of the project. "
        
        if file_name:
            doc_prompt += f" focusing on the following files ['{file_name}']"
        elif file_ext:
            doc_prompt += f" focusing on files with extension '{file_ext}'"
        else:
            doc_prompt += " focusing on Python files"
        
        doc_prompt += ". Explain what the code does, its architecture, and key components for developers."
        print("-"*50)
        print(doc_prompt)
        print("-"*50)
        
        return Task(
            prompt=doc_prompt,
            needs_approval=needs_approval,
            needs_eval=needs_eval,
            needs_plan=True
        )
    else:
        raise ValueError(f"Unknown task type: {task_type}")

def initialize_agents(client):
    """Initialize all agents with their respective configurations."""
    diff_messager_agent = DiffMessager(
        model=client,
        system_prompt="""
You are a senior programmer that explains hard concepts clearly and are very succinct in your messages. You can evaluate changes in a project just by reading the diff output from git.

CAPABILITIES:
1. Use the `diff_tool` to get the changes in the project
2. Generate commit messages based on the diff

INSTRUCTIONS:
- Use diff_tool to analyze changes, then generate a commit message
- Analyze the changes thoroughly to create meaningful commit messages
- Focus on the purpose and impact of the changes

For commit message generation, respond ONLY in the JSON format (example):
{"message": "Refactor GitReviewer for improved LLM integration and REPL functionality", "details": ["Introduced a `_get_config` method in `LLMGoogle` to centralize LLM calls.", "Refactored `main.py` to use a new `init_repl` function, streamlining the application's entry point and focusing on a REPL interface.", "Moved the `Message` Pydantic model to a dedicated `models.py`"]}
""",
        available_tools={'diff_tool': diff_tool, 'github_pull_request_diff_tool': github_tool}
    )

    commiter_agent = Commiter(
        model=client,
        system_prompt="""
You are a git commit specialist. Your only responsibility is to commit staged changes using the provided commit message.

CAPABILITIES:
1. Use the `commit_tool` to commit changes to the repository

INSTRUCTIONS:
- Extract the commit message from the user's prompt
- Extract the project directory from the user's prompt
- Use commit_tool to perform the actual commit operation
- Confirm successful commit completion

You will receive prompts containing:
- A commit message (in various formats like 'commit message: "..."' or JSON format)
- A project directory path

Your job is to execute the commit using the provided information.
""",
        available_tools={'commit_tool': commit_tool}
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
If the diff was not provided by the user you MUST use a diff tool to get the staged changes in the project.
""",
        available_tools={'diff_tool': diff_tool, 'github_pull_request_diff_tool': github_tool}
    )

    technical_writer_agent = TechnicalWriter(
        model=client,
        system_prompt="""
You are a technical writer specializing in software documentation. Your expertise lies in creating clear, comprehensive documentation for developers.

CAPABILITIES:
1. Use the `list_files_tool` to find files in the project based on name or extension
2. Use the `read_file_tool` to read the content of code files
3. Generate technical documentation that explains code functionality and architecture

INSTRUCTIONS:
- Always use list_files_tool first to find the relevant files
- Read the content of identified files using read_file_tool
- Generate documentation that explains:
  * What each file/module does and its purpose
  * Key classes, functions, and their responsibilities  
  * Overall architecture and component interactions
  * Important design patterns and techniques
  * Context for design decisions (when apparent)
- Write for developers who need to understand and work with the code
- Focus on explanation, not judgment - describe what the code does and why
- Structure documentation clearly with headings and sections
- Include code examples when helpful for understanding
""",
        available_tools={
            'list_files_tool': list_files_tool,
            'read_file_tool': read_file_tool
        }
    )
    
    return diff_messager_agent, commiter_agent, evaluator_agent, reviewer_agent, technical_writer_agent

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
            args.file_ext
        )
        print(f"Created task: {args.task}")
    except ValueError as e:
        print(f"Error creating task: {e}")
        return 1
    
    # Initialize agents
    diff_messager_agent, commiter_agent, evaluator_agent, reviewer_agent, technical_writer_agent = initialize_agents(client)
    
    # Create and configure the flow
    memory_file = "./taskflow_memory.json"
    flow = TaskFlow(model=client, memory_file_path=memory_file)
    flow.add(diff_messager_agent)
    flow.add(commiter_agent)
    flow.add(evaluator_agent)
    flow.add(reviewer_agent)
    flow.add(technical_writer_agent)
    
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
