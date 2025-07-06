import git 
from typing import Dict

from taskflow.util import logger
from taskflow.models import CommitMessage
from taskflow.exceptions import FileReadPermissionException, BinaryFileException, FileDecodingException, FileDoesNotExistException, NoChangesStaged


class CommitTool:
    """
    A tool class for committing staged changes with a commit message.
    """
    
    def __call__(self, project_dir: str, commit_message: CommitMessage | dict) -> str:
        """
        Commits staged changes with the provided commit message.

        Parameters:
            project_dir: Directory of the project with a git repo.
            commit_message: CommitMessage object containing the commit message and details.

        Returns:
            A string indicating success or error message.
        """
        try:
            if isinstance(commit_message, dict):
                message = CommitMessage(**commit_message)
            repo = git.Repo(project_dir)
            
            # Build the full commit message with details
            full_message = message.subject
            if message.body:
                #full_message += "\n"
                #for detail in message.details:
                #    full_message += f"- {detail}\n"
                # Remove the trailing newline
                full_message = f"{full_message}\n\n{message.body}"
            
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
                    "commit_message": {
                        "type": "object",
                        "description": "CommitMessage object containing the subject and description of the chages",
                        "properties": {
                            "subject": {
                                "type": "string",
                                "description": "The subject of the commit message"
                            },
                            "body": {
                                "type": "string",
                                "description": "The body of the message containing the description and notes of the changes made by the user"
                            }
                        },
                        "required": ["subject"]
                    }
                },
                "required": ["project_dir", "commit_message"]
            }
        }
