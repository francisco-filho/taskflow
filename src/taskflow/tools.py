import git 
from pathlib import Path
from typing import List, Dict

from taskflow.util import logger
from taskflow.models import CommitMessage
from taskflow.exceptions import FileReadPermissionException, BinaryFileException, FileDecodingException, FileDoesNotExistException, NoChangesStaged


class DiffTool:
    """
    A tool class for executing git diff operations in a repository.
    """
    
    def __call__(self, project_dir: str, only_staged: bool = True) -> str:
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
                    diff_output = repo.git.diff('--cached', '-U10')
                else:
                    diff_output = repo.git.diff()


                if not diff_output.strip():
                    raise NoChangesStaged(">>> No changes detected in the repository.")
                return diff_output
        except git.InvalidGitRepositoryError:
            return f"Error: '{project_dir}' is not a valid Git repository."
    
    def get_schema(self) -> dict:
        """
        Returns the function schema for this tool.
        
        Returns:
            Dictionary containing the tool schema for function calling.
        """
        return {
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


class CommitTool:
    """
    A tool class for committing staged changes with a commit message.
    """
    
    def __call__(self, project_dir: str, message: str) -> str:
        """
        Commits staged changes with the provided commit message.

        Parameters:
            project_dir: Directory of the project with a git repo.
            message: Commit message string.

        Returns:
            A string indicating success or error message.
        """
        try:
            repo = git.Repo(project_dir)
            
            # Create the commit with the simple string message
            commit = repo.index.commit(message)
        
            #commit = "a"
            #logger.info(f"Comitando: {message}")
            
            return f"Successfully committed with hash: {commit.hexsha[:8]}"
            
        except git.InvalidGitRepositoryError:
            return f"Error: '{project_dir}' is not a valid Git repository."
        except git.GitCommandError as e:
            return f"Git command error: {e}"
        except Exception as e:
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
                        "type": "string",
                        "description": "Commit message string"
                    }
                },
                "required": ["project_dir", "message"]
            }
        }


class ReadFileTool:
    """
    A tool class for reading text/code files safely, avoiding binary files.
    Supports reading single files or multiple files at once.
    """
    
    # Common binary file extensions to avoid
    BINARY_EXTENSIONS = {
        '.exe', '.dll', '.so', '.dylib', '.bin', '.dat', '.db', '.sqlite',
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.ico', '.webp',
        '.mp3', '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.wav',
        '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
        '.zip', '.rar', '.7z', '.tar', '.gz', '.bz2',
        '.pyc', '.pyo', '.class', '.o', '.obj'
    }
    
    def __call__(self, file_paths: List[str]) -> Dict[str, str]:
        """
        Reads one or more text/code files and returns their content as a dictionary.

        Parameters:
            file_paths: List of absolute paths to files to read.

        Returns:
            A dictionary with filename as key and file content as value.
            
        Raises:
            FileDoesNotExistException: When a file does not exist.
            BinaryFileException: When a file appears to be binary.
            FileReadPermissionException: When permission is denied.
            FileDecodingException: When a file cannot be decoded as text.
        """
        if not isinstance(file_paths, list):
            raise ValueError("file_paths must be a list")
        
        if not file_paths:
            raise ValueError("file_paths cannot be empty")
        
        result = {}
        
        for file_path in file_paths:
            try:
                content = self._read_single_file(file_path)
                filename = Path(file_path).name
                result[filename] = content
            except Exception as e:
                # Include error information in the result
                filename = Path(file_path).name
                result[filename] = f"Error reading file: {e}"
        
        return result
    
    def _read_single_file(self, file_path: str) -> str:
        """
        Reads a single file and returns its content.
        
        Parameters:
            file_path: Absolute path to the file to read.
            
        Returns:
            The file content as a string.
        """
        try:
            file_path_obj = Path(file_path)
            
            # Check if file exists
            if not file_path_obj.exists():
                raise FileDoesNotExistException(f"File '{file_path}' does not exist.")
            
            # Check if it's actually a file (not a directory)
            if not file_path_obj.is_file():
                raise FileDoesNotExistException(f"'{file_path}' is not a file.")
            
            # Check if it's likely a binary file based on extension
            if file_path_obj.suffix.lower() in self.BINARY_EXTENSIONS:
                raise BinaryFileException(f"'{file_path}' appears to be a binary file and cannot be read as text.")
            
            # Try to read the file with different encodings
            encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        content = file.read()
                        
                        # Additional check: if content contains many null bytes, it's likely binary
                        if '\x00' in content and content.count('\x00') > len(content) * 0.01:
                            raise BinaryFileException(f"'{file_path}' appears to contain binary data.")
                        
                        return content
                        
                except UnicodeDecodeError:
                    continue
            
            # If all encodings failed, it's likely a binary file
            raise FileDecodingException(f"'{file_path}' could not be decoded as text (likely binary file).")
            
        except PermissionError:
            raise FileReadPermissionException(f"Permission denied when trying to read '{file_path}'.")
        except (FileDoesNotExistException, BinaryFileException, FileReadPermissionException, FileDecodingException):
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            raise Exception(f"An unexpected error occurred while reading '{file_path}': {e}")
    
    def _read_multiple_files(self, file_paths: List[str]) -> str:
        """
        Reads multiple files and returns their content in the specified format.
        
        Parameters:
            file_paths: List of absolute paths to files to read.
            
        Returns:
            The formatted string containing all file contents.
        """
        result_parts = ["# =================================", "<files>"]
        
        for file_path in file_paths:
            try:
                content = self._read_single_file(file_path)
                result_parts.append(f'<file path="{file_path}">')
                result_parts.append(content)
                result_parts.append("</file>")
                result_parts.append("")  # Empty line between files
                
            except Exception as e:
                # Include error information in the output
                result_parts.append(f'<file path="{file_path}">')
                result_parts.append(f"Error reading file: {e}")
                result_parts.append("</file>")
                result_parts.append("")  # Empty line between files
        
        result_parts.append("</files>")
        result_parts.append("# =================================")
        
        return "\n".join(result_parts)
    
    @staticmethod
    def get_schema() -> dict:
        """
        Returns the function schema for this tool.
        
        Returns:
            Dictionary containing the tool schema for function calling.
        """
        return {
            "name": "read_file_tool",
            "description": "Reads one or more text/code files and returns their content as a dictionary with filename as key and content as value. Avoids reading binary files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of absolute paths to files to read"
                    }
                },
                "required": ["file_paths"]
            }
        }


class ListFilesTool():
    """
    A tool class for listing files in a project directory.
    """
    
    def __call__(self, project_dir: str, file_names: List[str]) -> List[str]:
        """
        Lists files in the project directory based on the provided file names.

        Parameters:
            project_dir: Directory of the project.
            file_names: List of filenames to search for.

        Returns:
            A list of absolute file paths found for the specified filenames.
            Returns empty list if no files found or on error.
        """
        try:
            project_path = Path(project_dir).resolve()
            
            if not project_path.exists() or not project_path.is_dir():
                return []
            
            if not isinstance(file_names, list):
                raise ValueError("file_names must be a list")
            
            if not file_names:
                raise ValueError("file_names cannot be empty")
            
            result = []
            
            # Search for each filename in the list
            for name in file_names:
                for file_path in project_path.rglob(name):
                    if ".venv" in str(file_path):
                        continue
                    if file_path.is_file():
                        result.append(str(file_path.absolute()))
            
            return result
            
        except Exception as e:
            # Log error but return empty list to maintain consistent return type
            return []
    
    def get_schema(self) -> dict:
        """
        Returns the function schema for this tool.
        
        Returns:
            Dictionary containing the tool schema for function calling.
        """
        return {
            "name": "list_files_tool",
            "description": "Lists files in the project directory based on the provided file names. Returns absolute paths of found files. This tool cannot be used in remote projects (gitlab, github)",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_dir": {
                        "type": "string",
                        "description": "Directory of the project"
                    },
                    "file_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of filenames to search for"
                    }
                },
                "required": ["project_dir", "file_names"]
            }
        }

class FinalAnswerTool:
    """
    A tool class that shows the final answer to the user.
    This tool is used to present the final result or conclusion of a task.
    """
    
    def __call__(self, answer: str) -> str:
        """
        Presents the final answer to the user.

        Parameters:
            answer: The final answer or result to show to the user.

        Returns:
            A formatted string containing the final answer.
        """
        if not answer or not answer.strip():
            return "Error: Answer cannot be empty."
        
        # Format the answer with a clear header
        formatted_answer = f"=== FINAL ANSWER ===\n{answer.strip()}\n===================="
        
        # Log the final answer for debugging purposes
        logger.info(f"Final answer provided: {answer[:100]}...")
        
        return formatted_answer
    
    def get_schema(self) -> dict:
        """
        Returns the function schema for this tool.
        
        Returns:
            Dictionary containing the tool schema for function calling.
        """
        return {
            "name": "final_answer_tool",
            "description": "Shows the final answer to the user. Use this tool to present the final result or conclusion of a task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The final answer or result to show to the user"
                    }
                },
                "required": ["answer"]
            }
        }


# For backward compatibility, you can create instances
diff_tool = DiffTool()
commit_tool = CommitTool()
list_files_tool = ListFilesTool()
read_file_tool = ReadFileTool()

# The old schema constants are now available through the class methods
DIFF_TOOL_SCHEMA = diff_tool.get_schema()
COMMIT_TOOL_SCHEMA = commit_tool.get_schema()
LIST_FILES_TOOL_SCHEMA = list_files_tool.get_schema()
READ_FILE_TOOL_SCHEMA = ReadFileTool.get_schema()
