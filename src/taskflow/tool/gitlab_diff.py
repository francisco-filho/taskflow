import os
import httpx
from taskflow.util import logger

class TokenNotProvidedException(Exception):
    pass

class GitlabMergeRequestDiffTool:
    """
    A tool class for fetching the diff of a GitLab merge request.
    """
    
    def __call__(self, project: str, merge_request_id: int, gitlab_url: str = "https://gitlab.com") -> str:
        """
        Fetches the diff of a GitLab merge request.

        Parameters:
            project: Project name in the format 'owner/project' or project ID.
            merge_request_id: The ID of the merge request.
            gitlab_url: The GitLab instance URL (defaults to gitlab.com).

        Returns:
            A string containing the full diff of all commits in the merge request,
            or an error message if the request fails.
        """
        try:
            # Validate project format (can be owner/project or numeric ID)
            if isinstance(project, str) and '/' in project and project.count('/') != 1:
                return f"Error: Invalid project format. Expected 'owner/project' or project ID, got '{project}'"
            
            # URL encode the project name for API usage
            if isinstance(project, str) and '/' in project:
                project_encoded = project.replace('/', '%2F')
            else:
                project_encoded = str(project)
            
            # GitLab API URL for merge request diff
            url = f"{gitlab_url.rstrip('/')}/api/v4/projects/{project_encoded}/merge_requests/{merge_request_id}/diffs"
            
            # Set up headers
            headers = {
                "User-Agent": "TaskFlow-Bot/1.0"
            }
            
            # Add authentication if token is available in environment
            token = os.getenv("GITLAB_TOKEN")
            if not token:
                raise TokenNotProvidedException("GITLAB_TOKEN")
            if token:
                headers["Authorization"] = f"Bearer {token}"
            
            # Make the HTTP request
            with httpx.Client(timeout=30.0) as client:
                response = client.get(url, headers=headers)
                
                if response.status_code == 200:
                    diffs_data = response.json()
                    
                    if not diffs_data:
                        return f"Warning: Merge request #{merge_request_id} in {project} has no diff content (possibly already merged or empty)"
                    
                    # Convert GitLab diff format to unified diff format
                    diff_content = self._format_gitlab_diffs(diffs_data)
                    return diff_content
                
                elif response.status_code == 404:
                    return f"Error: Merge request #{merge_request_id} not found in project {project}"
                
                elif response.status_code == 403:
                    token_msg = "No GITLAB_TOKEN environment variable found. " if not os.getenv("GITLAB_TOKEN") else ""
                    return f"Error: Access forbidden. {token_msg}Project {project} may be private or you don't have permission"
                
                elif response.status_code == 401:
                    return f"Error: Unauthorized access. Please check your GITLAB_TOKEN environment variable"
                
                else:
                    return f"Error: GitLab API request failed with status {response.status_code}: {response.text}"
                    
        except TokenNotProvidedException:
            raise
        except httpx.TimeoutException:
            return f"Error: Request timeout while fetching merge request #{merge_request_id} from {project}"
        
        except httpx.RequestError as e:
            return f"Error: Network error while fetching merge request diff: {e}"
        
        except Exception as e:
            logger.error(f"Unexpected error in GitlabMergeRequestDiffTool: {e}")
            return f"Error: An unexpected error occurred while fetching merge request diff: {e}"
    
    def _format_gitlab_diffs(self, diffs_data: list) -> str:
        """
        Converts GitLab API diff format to unified diff format.
        
        Parameters:
            diffs_data: List of diff objects from GitLab API.
            
        Returns:
            Formatted unified diff string.
        """
        formatted_diff = ""
        
        for diff_item in diffs_data:
            old_path = diff_item.get('old_path', '')
            new_path = diff_item.get('new_path', '')
            diff_content = diff_item.get('diff', '')
            
            # Handle file operations
            if diff_item.get('new_file', False):
                formatted_diff += f"--- /dev/null\n+++ b/{new_path}\n"
            elif diff_item.get('deleted_file', False):
                formatted_diff += f"--- a/{old_path}\n+++ /dev/null\n"
            elif diff_item.get('renamed_file', False):
                formatted_diff += f"--- a/{old_path}\n+++ b/{new_path}\n"
            else:
                formatted_diff += f"--- a/{old_path}\n+++ b/{new_path}\n"
            
            # Add the actual diff content
            formatted_diff += diff_content
            
            # Add separator between files
            if diff_content and not diff_content.endswith('\n'):
                formatted_diff += '\n'
        
        return formatted_diff
    
    def get_schema(self) -> dict:
        """
        Returns the function schema for this tool.
        
        Returns:
            Dictionary containing the tool schema for function calling.
        """
        return {
            "name": "gitlab_merge_request_diff_tool",
            "description": "Fetches the diff of a GitLab merge request containing all commits",
            "parameters": {
                "type": "object",
                "properties": {
                    "project": {
                        "type": "string",
                        "description": "Project name in the format 'owner/project' or numeric project ID"
                    },
                    "merge_request_id": {
                        "type": "integer",
                        "description": "The ID of the merge request"
                    },
                    "gitlab_url": {
                        "type": "string",
                        "description": "The GitLab instance URL (defaults to https://gitlab.com)",
                        "default": "https://gitlab.com"
                    }
                },
                "required": ["project", "merge_request_id"]
            }
        }


# For backward compatibility, create an instance
#gitlab_merge_request_diff_tool = GitlabMergeRequestDiffTool()

# Schema constant for easy import
#GITLAB_MERGE_REQUEST_DIFF_TOOL_SCHEMA = GitlabMergeRequestDiffTool().get_schema()
