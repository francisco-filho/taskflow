from typing import Optional, Dict, List, Callable
from pathlib import Path

from taskflow.util import logger
from taskflow.llm import LLMClient
from taskflow.tools import LIST_FILES_TOOL_SCHEMA, READ_FILE_TOOL_SCHEMA
from taskflow.agents import Agent


class TechnicalWriter(Agent):
    """
    An agent responsible for generating technical documentation for code files.
    This agent analyzes code files and creates comprehensive documentation explaining
    what the code does and why it does it, targeted at developers.
    """

    def __init__(self, model: LLMClient, system_prompt: str, available_tools: Optional[Dict[str, Callable]] = None):
        super().__init__(
            "TechnicalWriter",
            model,
            "Generates technical documentation for code files by analyzing their content and structure.",
            system_prompt,
            available_tools
        )

    def _get_tool_schemas(self) -> List[Dict]:
        """Returns the tool schemas available to the technical writer agent."""
        return [LIST_FILES_TOOL_SCHEMA, READ_FILE_TOOL_SCHEMA]

    def _generate_documentation(self, original_prompt: str, file_contents: Dict[str, str]) -> str:
        """
        Generates technical documentation based on the file contents.
        
        Parameters:
            original_prompt: The original user prompt.
            file_contents: Dictionary mapping filename to content.
            
        Returns:
            The generated documentation as a string.
        """
        # Prepare the content for documentation generation
        files_info = []
        for filename, content in file_contents.items():
            files_info.append(f"## File: {filename}\n\n```\n{content}\n```")
        
        combined_content = "\n\n".join(files_info)
        
        doc_prompt = f"""Based on the user request: "{original_prompt}"

Please generate comprehensive technical documentation for the following code files. The documentation should:

1. Explain what each file does and its purpose
2. Describe the main classes, functions, and their responsibilities
3. Explain the overall architecture and how components interact
4. Highlight important design patterns or techniques used
5. Provide context for why certain decisions were made (when apparent from the code)
6. Be written for developers who need to understand and work with this code

Files to document:

{combined_content}

Generate clear, well-structured documentation that helps developers understand the codebase."""

        try:
            doc_resp = self.model.chat(prompt=doc_prompt, system_prompt=self.system_prompt)
            return doc_resp.content
        except Exception as e:
            return f"Failed to generate documentation: {e}"

    def run(self, prompt: str, **kwargs) -> str:
        """
        Generates technical documentation based on specified files or file patterns.
        
        The agent will now always use the LLM to decide which tool to call
        (list_files_tool or read_file_tool) based on the user's prompt.
        
        Parameters:
            prompt: The user prompt containing the documentation request and file specifications.
            **kwargs: Additional keyword arguments (for compatibility).

        Returns:
            A dictionary containing the generated documentation.
        """
        print(f"TechnicalWriter agent running with prompt: {prompt[:100]}...")

        project_dir = self._extract_project_dir(prompt)
        if not project_dir:
            return {
                "documentation": "Error: Could not extract project directory from prompt. Please specify the project directory.",
                "files_processed": [],
                "error": True
            }

        tools = self._get_tool_schemas()
        file_contents = {}
        files_to_process = []

        try:
            # First, ask the LLM to identify what needs to be done (list or read a specific file)
            initial_llm_prompt = (
                f"The user wants a MARKDOWN documentation for code. The project directory is '{project_dir}'. "
                f"Based on the prompt: '{prompt}', should I list files to find relevant ones, or is there a specific "
                f"absolute file path mentioned that I should try to read directly? "
                f"If you need to list files, use 'list_files_tool' with appropriate parameters (project_dir, name, or ext). "
                f"If a specific absolute file path is given, use 'read_file_tool' for that path. "
                f"Be explicit about the file path if using 'read_file_tool'."
            )
            print(f"Calling LLM for initial file identification: {initial_llm_prompt[:100]}...")
            
            resp = self.model.chat(prompt=initial_llm_prompt, system_prompt=self.system_prompt, tools=tools)

            if resp.function_call:
                if resp.function_call.name == "list_files_tool":
                    print("LLM decided to call list_files_tool.")
                    # LLM wants to list files
                    list_result = self._execute_function_call(resp.function_call)
                    print("*"*80)
                    print(list_result, isinstance(list_result, list))
                    print("*"*80)
                    
                    if not isinstance(list_result, list) or not list_result:
                        return {
                            "documentation": "Error: LLM called list_files_tool but no files were returned or result was invalid.",
                            "files_processed": [],
                            "error": True
                        }
                    
                    # list_result is a list of dictionaries, e.g., [{'filename': 'main.py', 'path': '/path/to/main.py'}]
                    for file_info in list_result:
                        # For each file, ask the LLM to read it
                        filename = file_info.get('filename')
                        filepath = file_info.get('path')
                        if filename and filepath:
                            files_to_process.append({'filename': filename, 'path': filepath})

                elif resp.function_call.name == "read_file_tool":
                    print("LLM decided to call read_file_tool directly.")
                    # LLM wants to read a specific file directly
                    filepath = resp.function_call.args.get("file_path")
                    if filepath:
                        files_to_process.append({'filename': Path(filepath).name, 'path': filepath})
                    else:
                        return {
                            "documentation": "Error: LLM called read_file_tool but no file_path was provided.",
                            "files_processed": [],
                            "error": True
                        }
                else:
                    return {
                        "documentation": f"Error: LLM called an unexpected tool: {resp.function_call.name}",
                        "files_processed": [],
                        "error": True
                    }
            else:
                return {
                    "documentation": f"Error: LLM did not provide a function call to identify files. Response: {resp.content}",
                    "files_processed": [],
                    "error": True
                }

            if not files_to_process:
                return {
                    "documentation": "Error: No files were identified for processing after initial LLM interaction.",
                    "files_processed": [],
                    "error": True
                }

            # Now, for each file identified (either from list or direct read),
            # ask the LLM to read its content using read_file_tool
            for file_entry in files_to_process:
                filename = file_entry['filename']
                filepath = file_entry['path']
                
                read_file_llm_prompt = (
                    f"The file '{filename}' located at '{filepath}' needs to be read for documentation. "
                    f"Please use the 'read_file_tool' to get its content."
                )
                print(f"Calling LLM to read file '{filename}': {read_file_llm_prompt[:100]}...")
                
                read_resp = self.model.chat(prompt=read_file_llm_prompt, system_prompt=self.system_prompt, tools=tools)

                if read_resp.function_call and read_resp.function_call.name == "read_file_tool":
                    read_content = self._execute_function_call(read_resp.function_call)
                    if not read_content.startswith("Error:") and not read_content.startswith("An unexpected error"):
                        file_contents[filename] = read_content
                    else:
                        logger.warning(f"Could not read {filename}: {read_content}")
                else:
                    logger.warning(f"LLM did not call read_file_tool for {filename}. Response: {read_resp.content}")
            
            if not file_contents:
                return {
                    "documentation": "Error: No file contents could be retrieved for documentation.",
                    "files_processed": [f['filename'] for f in files_to_process],
                    "error": True
                }

            # Step 4: Generate documentation
            documentation = self._generate_documentation(prompt, file_contents)
            
            print(f"✓ Documentation generated successfully for {len(file_contents)} files!")
            return documentation
            # return {
            #     "documentation": documentation,
            #     "files_processed": list(file_contents.keys()),
            #     "error": False
            # }

        except Exception as e:
            print(f"Error during TechnicalWriter execution: {e}")
            return {
                "documentation": f"Execution failed: {e}",
                "files_processed": [],
                "error": True
            }
