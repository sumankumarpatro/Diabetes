import json
import re
from typing import Any, Dict, Optional
from loguru import logger
from llm_providers import LLMProvider

class LLMInterface:
    def __init__(self, provider: LLMProvider):
        self.provider = provider

    async def generate_structured_json(self, prompt: str, schema: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            raw_response = await self.provider.generate_structured_json(prompt, schema)
            if not raw_response:
                return None
            if isinstance(raw_response, str):
                return self._parse_json_payload(raw_response)
            return raw_response
        except Exception as e:
            logger.error(f"[LLMInterface] JSON generation error: {e}")
            return None

    async def generate_text(self, prompt: str) -> Optional[str]:
        try:
            return await self.provider.generate_text(prompt)
        except Exception as e:
            logger.error(f"[LLMInterface] Text generation error: {e}")
            return None

    def _parse_json_payload(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Robust JSON parser: cleans markdown, repairs missing commas,
        single-quotes, and unescaped characters automatically.
        """
        try:
            cleaned = re.sub(r'```json\s*|```', '', text).strip()
            start = cleaned.find('{')
            end = cleaned.rfind('}')
            
            if start == -1 or end == -1 or end <= start:
                return None

            json_str = cleaned[start:end+1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
            repaired = re.sub(r"(?<=\{|\s|,)(['\"])?([a-zA-Z0-9_]+)\1(?=\s*:)", r'"\2"', json_str)
            repaired = re.sub(r":\s*'([^']*)'", r': "\1"', repaired)
            repaired = re.sub(r'([0-9]|"|true|false|null)\s*\n\s*"', r'\1,\n"', repaired)
            repaired = re.sub(r'}\s*\n\s*{', r'},\n{', repaired)
            repaired = re.sub(r']\s*\n\s*"', r'],\n"', repaired)
            repaired = re.sub(r',\s*([}\]])', r'\1', repaired)

            return json.loads(repaired)

        except Exception as e:
            logger.debug(f"[LLMInterface] Final JSON repair failed: {e}")
            return None