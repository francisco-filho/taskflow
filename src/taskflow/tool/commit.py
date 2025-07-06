import git 
from typing import Dict

from taskflow.util import logger
from taskflow.models import CommitMessage
from taskflow.exceptions import FileReadPermissionException, BinaryFileException, FileDecodingException, FileDoesNotExistException, NoChangesStaged


class CommitTool:
    """
    A tool class for committing staged changes with a commit message.
    """
    
    def __call__(self, project_dir: str, message: CommitMessage | dict) -> str:
        """
        Commits staged changes with the provided commit message.

        Parameters:
            project_dir: Directory of the project with a git repo.
            message: CommitMessage object containing the commit message and details.

        Returns:
            A string indicating success or error message.
        """
        try:
            if isinstance(message, dict):
                message = CommitMessage(**message)
            repo = git.Repo(project_dir)
            
            # Build the full commit message with details
            full_message = message.message
            if message.details:
                full_message += "\n"
                for detail in message.details:
                    full_message += f"- {detail}\n"
                # Remove the trailing newline
                full_message = full_message.rstrip()
            
            commit = repo.index.commit(full_message)
        
            return f"Successfully committed with hash: {commit.hexsha[:8]}\n{full_message}"
            
        except git.InvalidGitRepositoryError:
            return f"Error: '{project_dir}' is not a valid Git repository."
        except git.GitCommandError as e:
            return f"Git command error: {e}"
        except Exception as e:
            logger.error(e)
            return f"An unexpected error occurred while committing: {e}"
    
    def get_schema(self) -> dict:
        """
        Returns the function schema for this tool.
        
        Returns:
            Dictionary containing the tool schema for function calling.
        """
        return {
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
                        "description": "CommitMessage object containing the commit message and details",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "The main commit message"
                            },
                            "details": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of detailed changes or notes"
                            }
                        },
                        "required": ["message"]
                    }
                },
                "required": ["project_dir", "message"]
            }
        }
