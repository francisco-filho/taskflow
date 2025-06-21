import os
import httpx
from taskflow.util import logger

class TokenNotProvidedException(Exception):
    pass

class GithubPullRequestDiffTool:
    """
    A tool class for fetching the diff of a GitHub pull request.
    """
    
    def __call__(self, repo: str, pull_request_id: int) -> str:
        """
        Fetches the diff of a GitHub pull request.

        Parameters:
            repo: Repository name in the format 'owner/repo'.
            pull_request_id: The ID of the pull request.

        Returns:
            A string containing the full diff of all commits in the pull request,
            or an error message if the request fails.
        """
        try:
            # Validate repo format
            if '/' not in repo or repo.count('/') != 1:
                return f"Error: Invalid repository format. Expected 'owner/repo', got '{repo}'"
            
            # GitHub API URL for pull request diff
            url = f"https://api.github.com/repos/{repo}/pulls/{pull_request_id}"
            
            # Set up headers
            headers = {
                "Accept": "application/vnd.github.diff",
                "User-Agent": "TaskFlow-Bot/1.0"
            }
            
            # Add authentication if token is available in environment
            token = os.getenv("GITHUB_TOKEN")
            if not token:
                raise TokenNotProvidedException("GITHUB_TOKEN")
            if token:
                headers["Authorization"] = f"token {token}"
            
            # Make the HTTP request
            with httpx.Client(timeout=30.0) as client:
                response = client.get(url, headers=headers)
                
                if response.status_code == 200:
                    diff_content = response.text
                    if not diff_content.strip():
                        return f"Warning: Pull request #{pull_request_id} in {repo} has no diff content (possibly already merged or empty)"
                    return diff_content
                
                elif response.status_code == 404:
                    return f"Error: Pull request #{pull_request_id} not found in repository {repo}"
                
                elif response.status_code == 403:
                    token_msg = "No GITHUB_TOKEN environment variable found. " if not os.getenv("GITHUB_TOKEN") else ""
                    return f"Error: Access forbidden. {token_msg}Repository {repo} may be private or rate limit exceeded"
                
                elif response.status_code == 401:
                    return f"Error: Unauthorized access. Please check your GITHUB_TOKEN environment variable"
                
                else:
                    return f"Error: GitHub API request failed with status {response.status_code}: {response.text}"
                    
        except TokenNotProvidedException:
            raise
        except httpx.TimeoutException:
            return f"Error: Request timeout while fetching pull request #{pull_request_id} from {repo}"
        
        except httpx.RequestError as e:
            return f"Error: Network error while fetching pull request diff: {e}"
        
        except Exception as e:
            logger.error(f"Unexpected error in GithubPullRequestDiffTool: {e}")
            return f"Error: An unexpected error occurred while fetching pull request diff: {e}"
    
    def get_schema(self) -> dict:
        """
        Returns the function schema for this tool.
        
        Returns:
            Dictionary containing the tool schema for function calling.
        """
        return {
            "name": "github_pull_request_diff_tool",
            "description": "Fetches the diff of a GitHub pull request containing all commits",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {
                        "type": "string",
                        "description": "Repository name in the format 'owner/repo'"
                    },
                    "pull_request_id": {
                        "type": "integer",
                        "description": "The ID of the pull request"
                    }
                },
                "required": ["repo", "pull_request_id"]
            }
        }


# For backward compatibility, create an instance
#github_pull_request_diff_tool = GithubPullRequestDiffTool()

# Schema constant for easy import
#GITHUB_PULL_REQUEST_DIFF_TOOL_SCHEMA = GithubPullRequestDiffTool.get_schema()
