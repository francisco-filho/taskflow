import git 
from pathlib import Path
from typing import List, Dict, Optional

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
                    diff_output = repo.git.diff('--cached')
                else:
                    diff_output = repo.git.diff()


                if not diff_output.strip():
                    raise NoChangesStaged(">>> No changes detected in the repository.")
                return diff_output
        except git.InvalidGitRepositoryError:
            return f"Error: '{project_dir}' is not a valid Git repository."
    
    @staticmethod
    def get_schema() -> dict:
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
    
    def __call__(self, project_dir: str, message: CommitMessage) -> str:
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
            # Format the commit message
            m = CommitMessage(**message)
            formatted_message = m.message
            if m.details:
                formatted_message += "\n\n"
                for detail in m.details:
                    formatted_message += f"- {detail}\n"
            
            # Create the commit
            commit = repo.index.commit(formatted_message)
        
            #commit = "a"
            #logger.info(f"Comitando: {formatted_message}")
            
            return f"Successfully committed with hash: {commit.hexsha[:8]}"
            
        except git.InvalidGitRepositoryError:
            return f"Error: '{project_dir}' is not a valid Git repository."
        except git.GitCommandError as e:
            return f"Git command error: {e}"
        except Exception as e:
            return f"An unexpected error occurred while committing: {e}"
    
    @staticmethod
    def get_schema() -> dict:
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


class ReadFileTool:
    """
    A tool class for reading text/code files safely, avoiding binary files.
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
    
    def __call__(self, file_path: str) -> str:
        """
        Reads a text/code file and returns its content as a string.

        Parameters:
            file_path: Absolute path to the file to read.

        Returns:
            The file content as a string.
            
        Raises:
            FileDoesNotExistException: When the file does not exist.
            BinaryFileException: When the file appears to be binary.
            FileReadPermissionException: When permission is denied.
            FileDecodingException: When the file cannot be decoded as text.
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
    
    @staticmethod
    def get_schema() -> dict:
        """
        Returns the function schema for this tool.
        
        Returns:
            Dictionary containing the tool schema for function calling.
        """
        return {
            "name": "read_file_tool",
            "description": "Reads text/code files and returns their content as a string. Avoids reading binary files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the file to read"
                    }
                },
                "required": ["file_path"]
            }
        }


class ListFilesTool():
    """
    A tool class for listing files in a project directory.
    """
    
    def __call__(self, project_dir: str, name: Optional[str] = None, ext: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Lists files in the project directory based on name or extension.

        Parameters:
            project_dir: Directory of the project.
            name: Optional filename to search for (returns first occurrence).
            ext: Optional file extension to filter by (without the dot).

        Returns:
            A list of dictionaries with filename as key and absolute path as value.
            Returns empty list if no files found or on error.
        """
        try:
            project_path = Path(project_dir).resolve()
            
            if not project_path.exists() or not project_path.is_dir():
                return []
            
            result = []
            
            if name:
                # Search for specific filename (first occurrence)
                for file_path in project_path.rglob(name):
                    if ".venv" in str(file_path):
                        continue
                    if file_path.is_file():
                        print("*"*80)
                        print(file_path)
                        print("*"*80)
                        
                        result.append({'filename': str(file_path.name), 'path': str(file_path.absolute())})
                        break  # Return only first occurrence
            elif ext:
                # Search for files with specific extension
                pattern = f"*.{ext}" if not ext.startswith('.') else f"*{ext}"
                for file_path in project_path.rglob(pattern):
                    if ".venv" in str(file_path):
                        continue
                    if file_path.is_file():
                        #result.append({file_path.name: str(file_path)})
                        result.append({'filename': str(file_path.name), 'path': str(file_path)})
            else:
                # List all files if no filter specified
                for file_path in project_path.rglob("*"):
                    if ".venv" in str(file_path):
                        continue
                    if file_path.is_file():
                        result.append({'filename': str(file_path.name), 'path': str(file_path)})
                        #result.append({file_path.name: str(file_path)})
            
            return result
            
        except Exception as e:
            # Log error but return empty list to maintain consistent return type
            return []
    
    @staticmethod
    def get_schema() -> dict:
        """
        Returns the function schema for this tool.
        
        Returns:
            Dictionary containing the tool schema for function calling.
        """
        return {
            "name": "list_files_tool",
            "description": "Lists files in the project directory, optionally filtered by name or extension",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_dir": {
                        "type": "string",
                        "description": "Directory of the project"
                    },
                    "name": {
                        "type": "string",
                        "description": "Optional filename to search for (returns first occurrence)",
                        "default": None
                    },
                    "ext": {
                        "type": "string", 
                        "description": "Optional file extension to filter by (without the dot)",
                        "default": None
                    }
                },
                "required": ["project_dir"]
            }
        }


# For backward compatibility, you can create instances
diff_tool = DiffTool()
commit_tool = CommitTool()
list_files_tool = ListFilesTool()
read_file_tool = ReadFileTool()

# The old schema constants are now available through the class methods
DIFF_TOOL_SCHEMA = DiffTool.get_schema()
COMMIT_TOOL_SCHEMA = CommitTool.get_schema()
LIST_FILES_TOOL_SCHEMA = ListFilesTool.get_schema()
READ_FILE_TOOL_SCHEMA = ReadFileTool.get_schema()
