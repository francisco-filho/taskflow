from pathlib import Path

class WriteFileTool:
    """
    A tool class for writing code/text files safely with proper error handling.
    """
    
    def __call__(self, file_path: str | Path, content: str, action: str) -> str:
        """
        Writes content to a file at the specified path.

        Parameters:
            file_path: Absolute path where the file should be written (str or Path).
            content: The content to write to the file.
            action: Action to perform - "create" or "update".

        Returns:
            A string indicating success or error message.
        """
        try:
            file_path_obj = Path(file_path)
            
            # Validate action parameter
            if action not in ["create", "update"]:
                return f"Error: Invalid action '{action}'. Must be 'create' or 'update'."
            
            # Check if file exists for create/update logic
            file_exists = file_path_obj.exists()
            
            if action == "create" and file_exists:
                return f"Error: File '{file_path}' already exists. Use 'update' action to modify existing files."
            
            if action == "update" and not file_exists:
                return f"Error: File '{file_path}' does not exist. Use 'create' action to create new files."
            
            # Always create parent directories if they don't exist
            if not file_path_obj.parent.exists():
                file_path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            # Check if parent is actually a directory
            if not file_path_obj.parent.is_dir():
                return f"Error: Parent path '{file_path_obj.parent}' is not a directory."
            
            # Write the file with UTF-8 encoding
            with open(file_path_obj, 'w', encoding='utf-8') as file:
                file.write(content)
            
            action_past_tense = "created" if action == "create" else "updated"
            return f"Successfully {action_past_tense} file: {file_path}"
            
        except PermissionError:
            return f"Error: Permission denied when trying to write to '{file_path}'."
        except IsADirectoryError:
            return f"Error: '{file_path}' is a directory, not a file."
        except OSError as e:
            return f"Error: OS error occurred while writing to '{file_path}': {e}"
        except Exception as e:
            return f"Error: An unexpected error occurred while writing to '{file_path}': {e}"
    
    def get_schema(self) -> dict:
        """
        Returns the function schema for this tool.
        
        Returns:
            Dictionary containing the tool schema for function calling.
        """
        return {
            "name": "write_file_tool",
            "description": "Writes content to a file at the specified path. Always creates parent directories if needed. Use 'create' for new files or 'update' for existing files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path where the file should be written"
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file"
                    },
                    "action": {
                        "type": "string",
                        "enum": ["create", "update"],
                        "description": "Action to perform - 'create' for new files or 'update' for existing files"
                    }
                },
                "required": ["file_path", "content", "action"]
            }
        }

