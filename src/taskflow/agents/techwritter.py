from typing import Optional, Dict, List, Callable
from pathlib import Path

from taskflow.util import logger
from taskflow.llm import LLMClient
from taskflow.tools import LIST_FILES_TOOL_SCHEMA, READ_FILE_TOOL_SCHEMA
from taskflow.agents import Agent


class TechnicalWriter(Agent):
    """
    An agent responsible for generating technical documentation for code files.
    This agent uses the LLM to analyze user requests and decide what files to read
    before generating comprehensive documentation.
    """

    def __init__(self, model: LLMClient, system_prompt: str, available_tools: Optional[Dict[str, Callable]] = None):
        super().__init__(
            "TechnicalWriter",
            model,
            "Generates technical documentation for code files by analyzing their content and structure.",
            f"You work with absolute file paths. If the user do not provide them, you should try to get using the available tools\n\n{system_prompt}",
            available_tools
        )

    def run(self, prompt: str, **kwargs) -> str:
        """
        Generates technical documentation based on the user's request.
        
        The agent uses the LLM to:
        1. Understand what the user wants documented
        2. Decide which tools to call to gather the necessary files
        3. Read file contents as needed
        4. Generate comprehensive documentation
        
        Parameters:
            prompt: The user prompt containing the documentation request.
            **kwargs: Additional keyword arguments (for compatibility).

        Returns:
            The generated documentation as a string.
        """
        print(f"TechnicalWriter agent running with prompt: {prompt[:100]}...")

        try:
            tools = self._get_tool_schemas()
            file_contents = {}
            available_files = []
            
            # Let the LLM handle the entire process
            current_prompt = prompt
            max_iterations = 10  # Prevent infinite loops
            iteration = 0
            
            while iteration < max_iterations:
                iteration += 1
                print(f"Iteration {iteration}: Calling LLM to process request...")
                
                resp = self.model.chat(prompt=current_prompt, system_prompt=self.system_prompt, tools=tools)
                
                logger.info("-"*50)
                logger.info(f"LLM Response (iteration {iteration}): {resp}")
                logger.info("-"*50)
                
                if resp.function_call:
                    print(f"LLM decided to call function: {resp.function_call.name}")
                    
                    # Execute the function call
                    result = self._execute_function_call(resp.function_call)
                    
                    if resp.function_call.name == "list_files_tool":
                        # Handle list files result - now returns List[str] of absolute paths
                        if isinstance(result, list) and result:
                            available_files = result
                            print(f"✓ Found {len(available_files)} files")
                            
                            # Create a summary for the LLM
                            files_summary = "\n".join([f"- {Path(fp).name} (path: {fp})" for fp in available_files])
                            
                            current_prompt = f"""Files found ({len(available_files)} total):
{files_summary}

Original request: {prompt}

Now you have the list of available files. Please use read_file_tool to read the files you want to include in the documentation. 
Note: read_file_tool accepts a list of file paths, so you can read multiple files in one call if needed."""
                            
                        else:
                            current_prompt = f"""No files were found or there was an error: {result}

Original request: {prompt}

Please try a different approach or provide an error message if no files can be found."""
                    
                    elif resp.function_call.name == "read_file_tool":
                        # Handle read file result - now returns Dict[str, str]
                        file_paths = resp.function_call.args.get("file_paths", [])
                        
                        # Ensure file_paths is a list (the tool expects a list)
                        if not isinstance(file_paths, list):
                            file_paths = [file_paths] if file_paths else []
                        
                        if isinstance(result, dict) and result:
                            # Successfully read files - result is now a dict with filename -> content
                            file_contents.update(result)
                            
                            # Check for any errors in the results
                            success_files = []
                            error_files = []
                            
                            for filename, content in result.items():
                                if content.startswith("Error reading file:"):
                                    error_files.append(filename)
                                else:
                                    success_files.append(filename)
                            
                            if success_files:
                                print(f"✓ Successfully read {len(success_files)} files: {', '.join(success_files)}")
                            if error_files:
                                print(f"✗ Failed to read {len(error_files)} files: {', '.join(error_files)}")
                            
                            if success_files:
                                # Generate documentation prompt WITH ACTUAL FILE CONTENTS
                                file_contents_section = ""
                                for filename in success_files:
                                    content = file_contents[filename]
                                    file_contents_section += f"\n--- FILE: {filename} ---\n{content}\n--- END OF {filename} ---\n"
                                
                                current_prompt = f"""Files have been read successfully.

Successfully read files: {', '.join(success_files)} ({len(success_files)} total)
{f"Failed to read files: {', '.join(error_files)}" if error_files else ""}

Original request: {prompt}

FILE CONTENTS:
{file_contents_section}

You now have the file contents above. Please generate comprehensive technical documentation in MARKDOWN format that includes:

1. A brief usage section explaining how to use the functionality
2. Explanation of what each file does and its purpose  
3. Description of main classes, functions, and their responsibilities
4. Overall architecture and component interactions
5. Important design patterns or techniques used
6. Context for design decisions (when apparent from code)
7. Written for developers who need to understand and work with this code

Generate the documentation now based on the file contents you have read."""
                            else:
                                # All files failed to read
                                current_prompt = f"""All files failed to read:
{chr(10).join([f"- {fname}: {content}" for fname, content in result.items()])}

Available files: {[Path(fp).name for fp in available_files] if available_files else 'None found'}

Original request: {prompt}

Please try reading different files from the available list, or provide an error message if no files can be accessed."""
                            
                        else:
                            # Failed to read files or got unexpected result
                            failed_files = [Path(fp).name for fp in file_paths]
                            print(f"✗ Failed to read files: {', '.join(failed_files)} - {result}")
                            
                            current_prompt = f"""Failed to read files '{', '.join(failed_files)}': {result}

Available files: {[Path(fp).name for fp in available_files] if available_files else 'None found'}

Original request: {prompt}

Please try reading different files from the available list, or provide an error message if no files can be accessed."""
                    
                    else:
                        return f"Error: LLM called unexpected function: {resp.function_call.name}"
                
                else:
                    # LLM provided a text response - this should be the final documentation
                    print("LLM provided final documentation response")
                    
                    content = resp.content.strip()
                    
                    # Check if this looks like proper documentation
                    if file_contents and len(content) > 200:  # Reasonable documentation length
                        print(f"✓ Documentation generated successfully for {len(file_contents)} file group(s)!")
                        return content
                    elif not file_contents and not available_files:
                        # No files were processed at all - might be a valid response or error
                        if "error" in content.lower() or "cannot" in content.lower() or "unable" in content.lower():
                            return content  # Return error message as-is
                        else:
                            # Might need to search for files first
                            current_prompt = f"""Your response: {content}

Original request: {prompt}

It seems you haven't found or read any files yet. Please use the list_files_tool first to find relevant files, then read them with read_file_tool before generating documentation."""
                    elif available_files and not file_contents:
                        # Files were found but not read yet
                        current_prompt = f"""Your response: {content}

You found {len(available_files)} files but haven't read them yet:
{chr(10).join([f"- {Path(fp).name}" for fp in available_files])}

Original request: {prompt}

Please use read_file_tool to read the relevant files before generating documentation."""
                    else:
                        # Files were read but response seems incomplete
                        # Include file contents in the retry prompt
                        file_contents_section = ""
                        for filename, content in file_contents.items():
                            file_contents_section += f"\n--- FILE: {filename} ---\n{content}\n--- END OF {filename} ---\n"
                        
                        current_prompt = f"""You have read files but your response seems incomplete or too brief: {content}

Files available: {list(file_contents.keys())}

Original request: {prompt}

FILE CONTENTS:
{file_contents_section}

Please generate comprehensive technical documentation in MARKDOWN format for these files based on their actual contents shown above."""
            
            # If we've reached max iterations without a final response
            if file_contents:
                return f"Error: Maximum iterations reached. Successfully read files ({list(file_contents.keys())}) but could not generate final documentation."
            else:
                return "Error: Maximum iterations reached without successfully reading any files or generating documentation."

        except Exception as e:
            print(f"Error during TechnicalWriter execution: {e}")
            return f"Execution failed: {e}"
