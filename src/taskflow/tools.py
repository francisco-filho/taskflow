import git 

from taskflow.util import logger
from taskflow.models import CommitMessage

def diff_tool(project_dir: str, only_staged: bool = True) -> str:
    """
    Executes a diff in the changes made in the repository.

    Parameters:
        project_dir: Directory of the project with a git repo.
        only_staged: If true, diff only staged files.

    Returns:
        A string representing the git diff output, or an error message if the
        directory is not a git repository.
    """
    try:
        repo = git.Repo(project_dir)

        if repo.is_dirty() or repo.untracked_files:
            if only_staged:
                diff_output = repo.git.diff('--cached')
            else:
                diff_output = repo.git.diff()
            return diff_output
        else:
            return "No changes detected in the repository."
    except git.InvalidGitRepositoryError:
        return f"Error: '{project_dir}' is not a valid Git repository."
    except Exception as e:
        return f"An unexpected error occurred while generating diff: {e}"

def commit_tool(project_dir: str, message: CommitMessage) -> str:
    """
    Commits staged changes with the provided commit message.

    Parameters:
        project_dir: Directory of the project with a git repo.
        message: CommitMessage object containing message and details.

    Returns:
        A string indicating success or error message.
    """
    try:
        repo = git.Repo(project_dir)
        
        #TODO: Check if there are staged changes
        
        # Format the commit message
        m = CommitMessage(**message)
        formatted_message = m.message
        if m.details:
            formatted_message += "\n\n"
            for detail in m.details:
                formatted_message += f"- {detail}\n"
        
        # Create the commit
        #commit = repo.index.commit(formatted_message)
    
        commit = "a"
        logger.info(f"Comitando: {formatted_message}")
        
        return f"Successfully committed with hash: {commit.hexsha[:8]}"
        
    except git.InvalidGitRepositoryError:
        return f"Error: '{project_dir}' is not a valid Git repository."
    except git.GitCommandError as e:
        return f"Git command error: {e}"
    except Exception as e:
        return f"An unexpected error occurred while committing: {e}"

# Function schema for Gemini function calling
DIFF_TOOL_SCHEMA = {
    "name": "diff_tool",
    "description": "Executes a diff in the changes made in the repository",
    "parameters": {
        "type": "object",
        "properties": {
            "project_dir": {
                "type": "string",
                "description": "Directory of the project with a git repo"
            },
            "only_staged": {
                "type": "boolean",
                "description": "If true, diff only staged files",
                "default": True
            }
        },
        "required": ["project_dir"]
    }
}

COMMIT_TOOL_SCHEMA = {
    "name": "commit_tool",
    "description": "Commits staged changes with the provided commit message",
    "parameters": {
        "type": "object",
        "properties": {
            "project_dir": {
                "type": "string",
                "description": "Directory of the project with a git repo"
            },
            "message": {
                "type": "object",
                "description": "CommitMessage object with message and details",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Main commit message"
                    },
                    "details": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of detailed changes"
                    }
                },
                "required": ["message", "details"]
            }
        },
        "required": ["project_dir", "message"]
    }
}