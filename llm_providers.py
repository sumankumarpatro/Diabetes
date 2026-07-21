from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import json
import ollama
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from config import config

class LLMProvider(ABC):
    """
    Abstract Base Class for LLM providers.
    """
    @abstractmethod
    def generate_structured_json(self, prompt: str, schema: Dict[str, Any]) -> Optional[str]:
        """
        Returns the raw string response from the LLM.
        """
        pass

class OllamaProvider(LLMProvider):
    """
    Implementation of LLMProvider using Ollama with retry logic.
    """
    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.LLM_MODEL_NAME
        logger.info(f"[OllamaProvider] Initialized with model: {self.model_name}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((RuntimeError, ConnectionError)), 
        reraise=True
    )
    def generate_structured_json(self, prompt: str, schema: Dict[str, Any]) -> Optional[str]:
        system_prompt = (
            "You are a clinical information extraction specialist. "
            "Your task is to extract specific clinical features from the provided text. "
            f"You must respond ONLY with a valid JSON object that follows this schema: {json.dumps(schema)}. "
            "Do not include any conversational text or markdown formatting."
        )

        full_prompt = f"System: {system_prompt}\n\nUser: {prompt}\n\nJSON Output:"

        try:
            logger.debug(f"[OllamaProvider] Sending prompt to Ollama: {prompt[:100]}...")
            # Using format='json' tells Ollama to ensure the output is valid JSON.
            # We still use the parser in LLMInterface as a safety net.
            response = ollama.generate(
                model=self.model_name,
                prompt=full_prompt,
                format='json'
            )
            
            return response['response']
        except (ConnectionError, TimeoutError, RuntimeError) as e:
            logger.warning(f"[OllamaProvider] Attempt failed due to error: {suppress_error_msg(e)}. Retrying...")
            raise e

def suppress_error_msg(e):
    return str(e)[:100]
