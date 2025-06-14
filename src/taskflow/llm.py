import json
from typing import Optional, List, Dict, Any
from abc import ABC, abstractmethod

from pydantic import BaseModel
from google import genai
import ollama

from taskflow.util import logger

def get_client(model: str = ""):
    """Get a client implementation based on the model name."""

    if model == "default":
        return OllamaClient()
    elif model in ["deepseek-r1:8b", "qwen2.5-coder:14b", "devstral:24b"]:
        return OllamaClient(model_name=model)
    elif model.startswith("gemini"):
        return GeminiClient(model)
    else:
        return OllamaClient()

class FunctionCall(BaseModel):
    name: str
    args: Dict[str, Any]

class ChatResponse(BaseModel):
    content: str
    function_call: Optional[FunctionCall] = None
    #input_tokens: int | None
    #output_tokens: int | None
    #duration: int

class LLMClient(ABC):
    """
    Abstract base class for LLM clients.
    """
    @abstractmethod
    def chat(self, prompt: str, system_prompt: str = "", output=None, tools: Optional[List[Dict]] = None) -> ChatResponse:
        """
        Sends a chat prompt to the LLM and returns the response.
        """
        pass

class GeminiClient(LLMClient):
    """
    Implementation of LLMClient for Google Gemini models.
    """
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        """
        Initializes the Gemini client.

        Parameters:
            model_name: The name of the Gemini model to use (e.g., "gemini-2.0-flash").
        """
        self.model_name = model_name
        self.model = genai.Client()
        print(f"GeminiClient initialized with model: {model_name}")

    def chat(self, prompt: str, system_prompt: str = "", output=None, tools: Optional[List[Dict]] = None) -> ChatResponse:
        """
        Sends a chat prompt to the Gemini model.

        Parameters:
            prompt: The user's prompt.
            system_prompt: An optional system-level instruction for the model.
            output: Optional output schema for structured responses.
            tools: Optional list of tool schemas for function calling.

        Returns:
            A ChatResponse object containing the content and optional function call.
        """
        contents = []
        if system_prompt:
            contents.append({"role": "user", "parts": [{ "text": f"System Instruction:\n{system_prompt}\n\nUser Prompt:\n{prompt}" }]})
        else:
            contents.append({"role": "user", "parts": [{ "text": prompt }]})

        try:
            config = genai.types.GenerateContentConfig(
                thinking_config=genai.types.ThinkingConfig(thinking_budget=0)
            )

            if output:
                config.response_mime_type = "application/json"
                config.response_schema = output

            if tools:
                config.tools = [genai.types.Tool(function_declarations=tools)]

            logger.debug(f"--- llm call ---\n{contents}")
            response = self.model.models.generate_content(contents=contents, model=self.model_name, config=config)
            
            # Extract content and function call from the Gemini response
            content = ""
            function_call = None
            
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, 'text') and part.text:
                            content += part.text
                        elif hasattr(part, 'function_call') and part.function_call:
                            # Extract function call information
                            function_call = FunctionCall(
                                name=part.function_call.name,
                                args=dict(part.function_call.args) if part.function_call.args else {}
                            )
            
            return ChatResponse(content=content, function_call=function_call)
            
        except Exception as e:
            logger.error(f"Error during Gemini chat: {e}")
            return ChatResponse(content=f"Error: {e}", function_call=None)


class OllamaClient(LLMClient):
    """
    Implementation of LLMClient for Ollama models.
    """
    def __init__(self, model_name: str = "qwen2.5-coder:14b", host: str = "http://127.0.0.1:11434"):
        """
        Initializes the Ollama client.

        Parameters:
            model_name: The name of the Ollama model to use (e.g., "llama3.1", "mistral", "codellama").
            host: The Ollama server host URL.
        """
        self.model_name = model_name
        self.host = host
        self.client = ollama.Client(host=host)
        print(f"OllamaClient initialized with model: {model_name} at {host}")

    def chat(self, prompt: str, system_prompt: str = "", output=None, tools: Optional[List[Dict]] = None) -> ChatResponse:
        """
        Sends a chat prompt to the Ollama model.

        Parameters:
            prompt: The user's prompt.
            system_prompt: An optional system-level instruction for the model.
            output: Optional output schema for structured responses.
            tools: Optional list of tool schemas for function calling.

        Returns:
            A ChatResponse object containing the content and optional function call.
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})

        try:
            options = {}
            
            # function calling
            if tools and len(tools) > 0:
                tool_descriptions = []
                for tool in tools:
                    tool_desc = f"Function: {tool['name']}\nDescription: {tool.get('description', '')}\nParameters: {json.dumps(tool.get('parameters', {}), indent=2)}"
                    tool_descriptions.append(tool_desc)
                
                function_instruction = f"""
You have access to the following functions:

{chr(10).join(tool_descriptions)}

If you need to call a function, respond with JSON in this exact format:
{{
    "function_call": {{
        "name": "function_name",
        "args": {{
            "param1": "value1",
            "param2": "value2"
        }}
    }}
}}
If you need a function call only respond with the json above and nothing else.

If you don't need to call a function, respond normally with your answer.
"""
                messages[-1]["content"] += function_instruction

            logger.debug(f"--- ollama call ---\n{messages}")
            
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                options=options,
                think=0,
                format=output.model_json_schema() if output else None
            )
            
            content = response['message']['content'] if 'message' in response and 'content' in response['message'] else ""
            function_call = None
            
            # Try to parse function call from response
            if tools and content.strip():
                try:
                    # Try to parse as JSON to check for function calls
                    parsed_response = json.loads(content.strip())
                    if isinstance(parsed_response, dict) and "function_call" in parsed_response:
                        func_call_data = parsed_response["function_call"]
                        if "name" in func_call_data and "args" in func_call_data:
                            function_call = FunctionCall(
                                name=func_call_data["name"],
                                args=func_call_data["args"]
                            )
                            content = ""
                except (json.JSONDecodeError, KeyError, TypeError):
                    # Not a function call, treat as regular content
                    pass
            
            return ChatResponse(content=content, function_call=function_call)
            
        except Exception as e:
            print(f"Error during Ollama chat: {e}")
            return ChatResponse(content=f"Error: {e}", function_call=None)