from typing import Dict, Any

# Custom exceptions for ReadFileTool
class FileDoesNotExistException(Exception):
    """Raised when a file does not exist."""
    pass

class BinaryFileException(Exception):
    """Raised when attempting to read a binary file as text."""
    pass

class FileReadPermissionException(Exception):
    """Raised when permission is denied to read a file."""
    pass

class FileDecodingException(Exception):
    """Raised when a file cannot be decoded as text."""
    pass

class NoChangesStaged(Exception):
    """Raised when there is no changes in project and the application try to diff-staged"""

class ToolExecutionNotAuthorized(Exception):
    """Exception raised when tool execution is not authorized by the user."""
    def __init__(self, tool_name: str, params: Dict[str, Any]):
        self.tool_name = tool_name
        self.params = params
        super().__init__(f"Tool '{tool_name}' execution not authorized. Params: {params}")
