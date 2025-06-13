import os
from typing import Optional, List, Dict
from abc import ABC, abstractmethod
from google import genai
from google.genai import types

from taskflow.util import logger

class LLMClient(ABC):
    """
    Abstract base class for LLM clients.
    """
    @abstractmethod
    def chat(self, prompt: str, system_prompt: str = "", output=None, tools: Optional[List[Dict]] = None) -> types.GenerateContentResponse:
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

    def chat(self, prompt: str, system_prompt: str = "", output=None, tools: Optional[List[Dict]] = None) -> types.GenerateContentResponse:
        """
        Sends a chat prompt to the Gemini model.

        Parameters:
            prompt: The user's prompt.
            system_prompt: An optional system-level instruction for the model.
            output: Optional output schema for structured responses.
            tools: Optional list of tool schemas for function calling.

        Returns:
            A GenerateContentResponse object from the Gemini API.
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
            return response
        except Exception as e:
            print(f"Error during Gemini chat: {e}")
            # Create a mock response for error handling to avoid breaking downstream logic
            mock_response = types.GenerateContentResponse()
            mock_response._chunks = [types.GenerateContentResponse.Candidate(
                content=types.Content(parts=[types.Part(text=f"Error: {e}")]),
                finish_reason=types.HarmCategory.HARM_CATEGORY_UNSPECIFIED
            )]
            return mock_response