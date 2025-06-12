import git 

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
        # Initialize a Git repository object
        repo = git.Repo(project_dir)

        # Check if there are any changes in the repository
        if repo.is_dirty() or repo.untracked_files:
            if only_staged:
                # Get the diff of staged files
                # This shows changes in the index (staged changes) compared to HEAD
                diff_output = repo.git.diff('--cached')
                print("-"*50)
                print(diff_output)
                print("-"*50)
            else:
                # Get the diff of all changes (staged and unstaged)
                # This shows changes in the working directory compared to HEAD
                diff_output = repo.git.diff()
            return diff_output
        else:
            return "No changes detected in the repository."
    except git.InvalidGitRepositoryError:
        return f"Error: '{project_dir}' is not a valid Git repository."
    except Exception as e:
        return f"An unexpected error occurred while generating diff: {e}"

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