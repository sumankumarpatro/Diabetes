from loguru import logger
import json
import re
from typing import Any, Dict, Optional
from llm_providers import LLMProvider, OllamaProvider

class LLMInterface:
    """
    A wrapper around LLM providers to provide a consistent interface.
    """
    def __init__(self, provider: LLMProvider):
        self.provider = provider
        logger.info(f"[LLMInterface] Initialized with provider: {type(self.provider).__name__}")

    def generate_structured_json(self, prompt: str, schema: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Delegates the JSON generation to the underlying provider and performs robust extraction.
        """
        try:
            raw_response = self.provider.generate_structured_json(prompt, schema)
            if not raw_response:
                return None

            # If the provider returned a string instead of a dict, try to parse it
            if isinstance(raw_response, str):
                return self._parse_json_payload(raw_response)
            
            return raw_response
        except Exception as e:
            logger.error(f"[LLMInterface] Error during JSON generation: {e}")
            return None

    def _parse_json_payload(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Uses regex to find and parse the first JSON object in a string.
        Handles potential markdown code blocks.
        """
        try:
            # Debug: print the text that we are trying to parse
            logger.debug(f"[LLMInterface] Attempting to parse text: {text[:500]}...")
            
            # Remove markdown code block markers if present
            text = re.sub(r'```json\s*|```', '', text).strip()
            
            # Regex to find content between the first '{' and the last '}'
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
