import json
import re
from typing import Any, Dict, Optional
from loguru import logger
from llm_providers import LLMProvider, OllamaProvider

class LLMInterface:
    """
    An asynchronous wrapper around LLM providers to provide a consistent,
    non-blocking interface across your parsing frameworks.
    """
    def __init__(self, provider: LLMProvider):
        self.provider = provider
        logger.info(f"[LLMInterface] Initialized with provider: {type(self.provider).__name__}")

    async def generate_structured_json(self, prompt: str, schema: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Asynchronously delegates JSON generation to the underlying provider 
        and performs a clean, robust extraction.
        """
        try:
            # Awaiting the asynchronous provider call natively
            raw_response = await self.provider.generate_structured_json(prompt, schema)
            if not raw_response:
                return None
            
            # If the async provider returned a raw string, parse it using our robust engine
            if isinstance(raw_response, str):
                return self._robust_json_parse(raw_response)
            return raw_response
        except Exception as e:
            logger.error(f"[LLMInterface] Error during JSON generation: {e}")
            return None

    async def generate_text(self, prompt: str) -> Optional[str]:
        """
        Asynchronously delegates plain text generation to the underlying provider.
        """
        try:
            # Awaiting the asynchronous provider call natively
            return await self.provider.generate_text(prompt)
        except Exception as e:
            logger.error(f"[LLMInterface] Error during text generation: {e}")
            return None

    def _robust_json_parse(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Uses regex to locate and parse the primary JSON object inside a payload string.
        Safely extracts JSON when wrapped inside markdown code blocks.
        """
        try:
            logger.debug(f"[LLMInterface] Attempting to parse text: {text[:500]}...")
            
            # Clean out markdown code boundaries exactly as your original code did
            text = re.sub(r'```json\s*|```', '', text).strip()
            
            # Identify first opening brace and trailing closing brace characters
            match = re.search(r'(\{.*\})', text, re.DOTALL)
            if match:
                json_str = match.group(1)
                return json.loads(json_str)
            else:
                logger.error(f"[LLMInterface] No JSON object found in response text: {text[:100]}...")
                return None
        except json.JSONDecodeError as e:
            logger.error(f"[LLMInterface] JSON decode error: {e}")
            return None
        except Exception as e:
            logger.error(f"[LLMInterface] Failed to parse JSON from text: {e}")
            return None
