"""
Thin Gemini client used by every chain (google-genai SDK).

All provider-specific code lives here so the chains only depend on one method:

    client.generate(prompt, system=..., images=None, json_output=False) -> str

Any failure (missing key, timeout, provider error, blocked/empty response) is
raised as LLMError with a readable message. Tests pass their own object with a
compatible `generate` method instead of a real client.
"""

import os
from typing import Any, List, Optional

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from src.utils.security import SYSTEM_PROMPT

load_dotenv()

DEFAULT_MODEL = "gemini-3.8-flash"  # override with LLM_MODEL in .env
DEFAULT_TIMEOUT_S = 60.0


class LLMError(RuntimeError):
    """Raised when the language model cannot produce a usable response."""


def _to_part(image: Any) -> Any:
    """Accept PIL images / types.Part as-is; convert {"mime_type", "data"} dicts to Parts."""
    if isinstance(image, dict):
        try:
            return types.Part.from_bytes(data=image["data"], mime_type=image["mime_type"])
        except KeyError as e:
            raise LLMError(f"Image dict is missing the {e} key.") from e
    return image


def _is_placeholder(key: str) -> bool:
    """True for template values such as 'your-gemini-api-key-here' or 'Paste-your-...-here'."""
    k = key.strip().strip('"').lower()
    return k.startswith(("your-", "your_", "paste")) or k.endswith(("-here", "_here"))


class GeminiClient:
    """Minimal wrapper around google-genai's Client.models.generate_content."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_s: Optional[float] = None,
        temperature: float = 0.2,
    ):
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key or _is_placeholder(api_key):
            raise LLMError(
                "GEMINI_API_KEY is not set. Add it to your .env file (see .env.example)."
            )
        self.model = model or os.getenv("LLM_MODEL", DEFAULT_MODEL)
        self.timeout_s = timeout_s or float(os.getenv("LLM_TIMEOUT", DEFAULT_TIMEOUT_S))
        self.temperature = temperature
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=int(self.timeout_s * 1000)),  # milliseconds
        )

    def generate(
        self,
        prompt: str,
        system: str = SYSTEM_PROMPT,
        images: Optional[List[Any]] = None,
        json_output: bool = False,
    ) -> str:
        """
        Send one prompt (plus optional images) and return the response text.

        `images` may contain PIL images, google.genai `types.Part` objects, or
        {"mime_type": ..., "data": bytes} dicts.
        """
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=self.temperature,
            response_mime_type="application/json" if json_output else "text/plain",
        )
        contents: List[Any] = [*(_to_part(img) for img in (images or [])), prompt]

        try:
            response = self._client.models.generate_content(
                model=self.model, contents=contents, config=config
            )
        except httpx.TimeoutException as e:
            raise LLMError(f"The model request timed out after {self.timeout_s:.0f}s.") from e
        except genai_errors.APIError as e:
            raise LLMError(self._describe_api_error(e)) from e
        except httpx.HTTPError as e:
            raise LLMError(f"Could not reach Gemini: {e}") from e

        text = response.text
        if not text or not text.strip():
            feedback = getattr(response, "prompt_feedback", None)
            block = getattr(feedback, "block_reason", None)
            finish = None
            if response.candidates:
                finish = getattr(response.candidates[0], "finish_reason", None)
            detail = f"block reason: {block}" if block else f"finish reason: {finish}"
            raise LLMError(f"Gemini returned no usable text ({detail}).")
        return text

    def _describe_api_error(self, e: "genai_errors.APIError") -> str:
        message = e.message or str(e)
        if e.code in (401, 403) or (e.code == 400 and "API key" in message):
            return f"Gemini rejected the API key: {message}"
        if e.code == 404:
            return f"Model '{self.model}' is not available: {message}"
        if e.code == 429:
            return f"Gemini quota or rate limit exceeded: {message}"
        if e.code in (408, 504):
            return f"The model request timed out: {message}"
        return f"Gemini request failed ({e.code} {e.status}): {message}"


_default_client: Optional[GeminiClient] = None


def get_client() -> GeminiClient:
    """Return a shared GeminiClient, created on first use."""
    global _default_client
    if _default_client is None:
        _default_client = GeminiClient()
    return _default_client
