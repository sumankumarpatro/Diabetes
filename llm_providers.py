import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import ollama
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from config import config

class LLMProvider(ABC):
    @abstractmethod
    async def generate_structured_json(self, prompt: str, schema: Dict[str, Any]) -> Optional[str]:
        pass

    @abstractmethod
    async def generate_text(self, prompt: str) -> Optional[str]:
        pass

class OllamaProvider(LLMProvider):
    def __init__(self, model_name: str = None, base_url: str = None):
        self.model_name = model_name or config.LLM_MODEL_NAME
        self.base_url = base_url or config.OLLAMA_BASE_URL
        self.client = ollama.AsyncClient(host=self.base_url)
        logger.info(f"[OllamaProvider] Initialized model: {self.model_name} at {self.base_url}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=6),
        retry=retry_if_exception_type((RuntimeError, ConnectionError, TimeoutError, OSError)),
        reraise=True
    )
    async def generate_structured_json(self, prompt: str, schema: Dict[str, Any]) -> Optional[str]:
        system_prompt = (
            "You are a clinical information extraction specialist. "
            "Extract clinical entities strictly from the provided text. "
            f"You must respond ONLY with a valid JSON object matching this schema: {json.dumps(schema)}. "
            "Do not include any conversational markdown, preambles, or postscripts."
        )
        full_prompt = f"System: {system_prompt}\n\nUser: {prompt}\n\nJSON Output:"

        try:
            response = await self.client.generate(
                model=self.model_name,
                prompt=full_prompt,
                format='json',
                options={
                    "temperature": 0.0,
                    "num_ctx": 2048,
                    "num_predict": 256
                }
            )
            return response.get('response', '')
        except (ConnectionError, TimeoutError, RuntimeError, OSError) as e:
            logger.warning(f"[OllamaProvider] JSON generation failed ({str(e)[:80]}). Retrying...")
            raise e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=6),
        retry=retry_if_exception_type((RuntimeError, ConnectionError, TimeoutError, OSError)),
        reraise=True
    )
    async def generate_text(self, prompt: str) -> Optional[str]:
        try:
            response = await self.client.generate(
                model=self.model_name,
                prompt=prompt
            )
            return response.get('response', '')
        except (ConnectionError, TimeoutError, RuntimeError) as e:
            logger.warning(f"[OllamaProvider] Text generation failed ({str(e)[:80]}). Retrying...")
            raise e

    async def close(self):
        if hasattr(self.client, 'close'):
            await self.client.close()