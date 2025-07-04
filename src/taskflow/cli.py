import argparse

from taskflow.models import Request
from taskflow.agents.commiter import Commiter
from taskflow.agents.reviewer import Reviewer
from taskflow.agents.diff import DiffMessager
from taskflow.agents.evaluator import Evaluator
from taskflow.agents.arch import Architect
from taskflow.agents.techwritter import TechnicalWriter
from taskflow.tools import ListFilesTool, diff_tool, commit_tool, list_files_tool, read_file_tool
from taskflow.agents import Tool
from taskflow.tool.github_diff import GithubPullRequestDiffTool
from taskflow.tool.gitlab_diff import GitlabMergeRequestDiffTool

github_tool = GithubPullRequestDiffTool()
gitlab_tool = GitlabMergeRequestDiffTool()
list_tool = ListFilesTool()

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="TaskFlow AI - Automated git commit, review, and documentation system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --project $path --task diff
  %(prog)s --project $path --task review
  %(prog)s --project $path --task commit
  %(prog)s --project $path --task doc
  %(prog)s --project $path --task diff --model gemini-2.5-flash-preview-05-20
  %(prog)s --create-temp-repo --task diff
        """
    )
    
    parser.add_argument(
        "--project", "-d",
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

    parser.add_argument(
        "--prompt", "-p",
        type=str,
        help="Custom prompt to send to the agents for the task"
    )
    
    return parser.parse_args()

def create_task(task_type, project_dir, needs_approval=False, needs_eval=False, file_name=None, file_ext=None, custom_prompt=None):
    """Create a task based on the task type."""
    if custom_prompt:
        return Request(
            prompt=custom_prompt,
            needs_approval=needs_approval,
            needs_eval=needs_eval, # Assuming custom prompts might need evaluation
            needs_plan=True # Assuming custom prompts might need planning
        )
    elif task_type == "diff":
        return Request(
            prompt=f"""
Propose a commit message for the staged changes in the project 'https://github.com/francisco-filho/taskflow/pull/1'""",
#             prompt=f"""
# Propose a commit message for the staged changes in the project 'https://gitlab.com/francisco-filho/test1/-/merge_requests/1'""",
#             prompt=f"""
# Propose a commit message for the staged changes in the project 'https://gitlab.com/testgenai/ai/test1/-/commit/8ca41ebb29d64ad1e0da2c6a6bfdab3377d00e0d'""",
            needs_approval=needs_approval,
            needs_eval=True,
            needs_plan=True
        )
    elif task_type == "review":
        return Request(
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
        return Request(
            prompt=f"""
            Generate a commit message for the staged changes in the project '{project_dir}' and commit the changes. Commit the changes in the repository.
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
        
        return Request(
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
1. Receive diff in prompt and generate messages
2. Use the `diff_tool` to get the changes in the project when the diff is not in the prompt
3. Generate commit messages based on the diff

INSTRUCTIONS:
- If the user provides the 'diff' in the prompt, use it
- If the user did not provide de 'diff' in the prompt Use diff_tool to analyze changes, then generate a com<Find>mit message
- Analyze the changes thoroughly to create meaningful commit messages
- Focus on the purpose and impact of the changes

For commit message generation, respond ONLY in the text format below:

Write the Commit message here, focusing in the overall changes

- {{Detail 1 about the changes}}
- {{Detail 2 about the changes}}
... repeate if necessary

""",
        available_tools={
            'diff_tool': Tool('diff_tool', diff_tool, needs_approval=True),
            'github_pull_request_diff_tool': Tool('github_pull_request_diff_tool', github_tool, needs_approval=False),
            'gitlab_merge_request_diff_tool': Tool('gitlab_merge_request_diff_tool', gitlab_tool, needs_approval=False),
        }
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
        available_tools={'commit_tool': Tool('commit_tool', commit_tool, needs_approval=True)}
    )

    evaluator_agent = Evaluator(
        model=client,
        system_prompt="""
You are a senior programmer that has attention to details and likes very clear texts. You made code reviews and evaluate the quality of the commit messages based on the diff changes.
If your evaluation is positive, just respond with 'Commit message accepted', but
""",
        available_tools={'diff_tool': Tool('diff_tool', diff_tool, needs_approval=False)}
    )

    reviewer_agent = Reviewer(
        model=client,
        system_prompt="""
You are a meticulous code reviewer. Your task is to provide a concise and constructive review of the given code changes, focusing on clarity, potential issues, and adherence to best practices. Summarize the key changes and any recommendations.
If the diff was not provided by the user you MUST use a diff tool to get the staged changes in the project.
""",
        available_tools={
            'diff_tool': Tool('diff_tool', diff_tool, needs_approval=False), 
            'github_pull_request_diff_tool': Tool('github_pull_request_diff_tool', github_tool, needs_approval=False),
            'gitlab_merge_request_diff_tool': Tool('gitlab_merge_request_diff_tool', gitlab_tool, needs_approval=False),
        }
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
            'list_files_tool': Tool('list_files_tool', list_files_tool, needs_approval=False),
            'read_file_tool': Tool('read_file_tool', read_file_tool, needs_approval=False)
        }
    )

    arch_agent = Architect(
        model=client,
        system_prompt="",
        available_tools={
            'list_files_tool': Tool('list_files_tool', list_files_tool, needs_approval=False),
            'read_file_tool': Tool('read_file_tool', read_file_tool, needs_approval=False)
        }
    )
    
    return diff_messager_agent, commiter_agent, evaluator_agent, reviewer_agent, technical_writer_agent, arch_agent
