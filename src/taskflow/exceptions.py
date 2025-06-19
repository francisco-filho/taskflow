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
