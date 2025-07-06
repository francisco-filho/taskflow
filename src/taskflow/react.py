import os
import argparse
import logging
from typing import Dict, List

from dotenv import load_dotenv
from taskflow.agents import Tool
from taskflow.llm import LLMClient, get_client
from taskflow.exceptions import NoChangesStaged, ToolExecutionNotAuthorized
from taskflow.tools import diff_tool, commit_tool, list_files_tool, read_file_tool, FinalAnswerTool
from taskflow.tool.write_file import WriteFileTool
from taskflow.util import DEFAULT_MODEL, printc

write_file_tool = WriteFileTool()
final_answer_tool = FinalAnswerTool()

logging.getLogger("google_genai.models").setLevel(logging.ERROR)

load_dotenv()

REACT_PROMPT = """
You work in a loop in the following way, you analise the user request, them you should have a though,
and choose the action you should take. You will have tools at your disposal to get information,
interact with the enviroment, and execute actions that you cannot do as a llm.

Your loop will be like this (until you detect the Final Answer:
- Thougth is your thinking process
- Action is something executed by a tool or by yoursel when possible
- Observation is the result of the action


Example:
User task: Rename the class Cat() present in animal.py to Doc()

Thougth: I Should find the full path of the animal.py to be able to read it.
Action: use tool list_files_by_name(animal.py)
Observation: {{/path/to/file/animal.py}}

Thougth: I have the path of the file, i should read it
Action: use tool read_file({{path/to/file/animal.py}})
Observation: ```import os\nclass Cat(): pass```

Thougth: I will change the file to ```import os\nclass Dog(): pass```
Action: use tool write_file({{path/to/file/animal.py}})
Observation: file {{file}} saved with success

Thougth: I fullfilled the user request. Do not need to call tools.
Action: call tool to deliver the final answer

End of example.


User request:
{user_prompt}
"""


class ReactAgent():
    def __init__(self, client: LLMClient, available_tools, verbose: bool = False):
        self.available_tools = available_tools
        self.logger = logging.getLogger("taskflow")
        self.logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        print(f"Logging verbose: {verbose}")

    def run(self, prompt: str):
        p = REACT_PROMPT.format(user_prompt=prompt)
        max_iterations = 10
        iter = 0
        tools = self._get_tool_schemas()

        while iter <= max_iterations:
            resp = llm.chat(p, tools=tools)

            if resp.function_call:
                action = f"\nAction: {str(resp.function_call)}" 
                self.logger.debug(action)
                p = p + action 
                fn_result = self._execute_function_call(resp.function_call)

                if resp.function_call.name == "final_answer_tool":
                    printc(f"[blue]{fn_result}")
                    break
                observation = f"\nObservation: {str(fn_result)}"
                self.logger.debug(observation)
                p = p + observation 
            else:
                thougth =f"\n{str(resp.content)}" 
                self.logger.debug(thougth)
                p = p + thougth 
                final_resp = thougth

            iter = iter + 1
        #logging.info(final_resp)
            

    def _execute_function_call(self, function_call):
        """
        Executes a function call returned by the LLM.

        Parameters:
            function_call: The function call object from the LLM response.

        Returns:
            The result of the function execution as a string.
        """
        function_name = function_call.name
        function_args = function_call.args

        if function_name not in self.available_tools:
            return f"Error: Function '{function_name}' not available for agent '{self.name}'."

        try:
            print(f"Executing function: {function_name} with args: {function_args}")
            tool = self.available_tools[function_name]
            result = tool(**function_args)
            return result
        except NoChangesStaged as e:
            raise
        except ToolExecutionNotAuthorized as e:
            raise
        except Exception as e:
            return f"Error executing function '{function_name}': {e}"

    def _get_tool_schemas(self) -> List[Dict]:
        """
        Returns the tool schemas for function calling.
        This should be overridden by subclasses to provide specific tool schemas.
        """
        if not self.available_tools:
            return []
        schemas = []
        for tool in self.available_tools.values():
            if hasattr(tool.fn, 'get_schema'):
                schemas.append(tool.fn.get_schema())
        return schemas

if __name__ == '__main__':
    model_name = os.getenv("DEFAULT_MODEL", "gemini-2.5-flash-preview-05-20")
    print(f"Using model: {model_name}")


    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default=model_name, help="Model to use")
    args = parser.parse_args()

    # Initialize the client
    llm = get_client(args.model)

    agent = ReactAgent(
        llm,
        available_tools={  
            'list_files_tool': Tool('list_files_tool', list_files_tool, needs_approval=False),
            'read_file_tool': Tool('read_file_tool', read_file_tool, needs_approval=False),
            'diff_tool': Tool('diff_tool', diff_tool, needs_approval=False), 
            'write_file_tool': Tool('write_file_tool', write_file_tool, needs_approval=True), 
            'commit_tool': Tool('commit_tool', commit_tool, needs_approval=True),
            'final_answer_tool': Tool('final_answer_tool', final_answer_tool, needs_approval=False),
        },
        verbose=False,
    )

    #agent.run("""update models.py renaming the Request() class to UserRequest()""")
    agent.run("""Generate a commit message of the staged changes in the current directory.
    Use this format to the commit message:
MAIN TOPIC OF THE CHANGES
- DETAIL1
- OTHER DETAILS IF NECESSARY

Example:
Create upload file
- Add method upload()
- Catch erros and respond to them

When you get the message them DO commit the changes in the repo
    """)


