"""Provider-independent, server-side LLM client.

Speaks the OpenAI-compatible ``POST {LLM_BASE_URL}/chat/completions`` format,
which is served by Ollama, vLLM, LM Studio, many hosted open-weight model
providers, and other OpenAI-compatible gateways. The provider, model, URL,
and key all come from environment variables; none are hard-coded here.

Error messages produced here never include the API key, request headers,
or the prompt.
"""

from __future__ import annotations

import logging

import httpx

from moviematch.config import LlmSettings

logger = logging.getLogger("moviematch.llm")


class LlmError(Exception):
    """A transport-level LLM failure with a message that is safe to show."""

    def __init__(self, kind: str, safe_message: str):
        super().__init__(safe_message)
        self.kind = kind
        self.safe_message = safe_message


class OpenAICompatibleClient:
    def __init__(self, settings: LlmSettings, transport: httpx.BaseTransport | None = None):
        if not settings.configured:
            raise ValueError("LLM is not configured")
        self._settings = settings
        self._url = settings.base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if settings.api_key:
            headers["Authorization"] = f"Bearer {settings.api_key}"
        self._client = httpx.Client(
            headers=headers, timeout=settings.timeout_seconds, transport=transport
        )

    @property
    def model(self) -> str:
        return self._settings.model

    def complete(self, messages: list[dict[str, str]], response_format: dict | None = None) -> str:
        body: dict[str, object] = {
            "model": self._settings.model,
            "messages": messages,
            "temperature": self._settings.temperature,
            "stream": False,
        }
        if self._settings.structured_output and response_format is not None:
            body["response_format"] = response_format
        elif self._settings.json_mode:
            body["response_format"] = {"type": "json_object"}
        try:
            response = self._client.post(self._url, json=body)
            if response.status_code == 400 and "response_format" in body and "context length" not in response.text:
                # Some providers reject response_format; the prompt already asks for
                # bare JSON and the validator checks it, so retry once without it.
                logger.warning("LLM rejected response_format (HTTP 400); retrying without it")
                body.pop("response_format")
                response = self._client.post(self._url, json=body)
        except httpx.TimeoutException:
            logger.warning("LLM request timed out after %ss", self._settings.timeout_seconds)
            raise LlmError(
                "timeout",
                f"The LLM did not respond within {self._settings.timeout_seconds:g} seconds.",
            ) from None
        except httpx.HTTPError as exc:
            logger.warning("LLM connection failed: %s", type(exc).__name__)
            raise LlmError("connection", "Could not connect to the LLM service.") from None

        if response.status_code >= 400:
            logger.warning("LLM service returned HTTP %s", response.status_code)
            hint = ""
            if response.status_code in (401, 403):
                hint = " Check LLM_API_KEY."
            elif response.status_code == 404:
                hint = " Check LLM_BASE_URL and LLM_MODEL."
            elif response.status_code == 400 and self._settings.structured_output:
                hint = " If the provider does not support structured output, set LLM_STRUCTURED_OUTPUT=false."
            elif response.status_code == 400 and self._settings.json_mode:
                hint = " If the provider does not support JSON mode, set LLM_JSON_MODE=false."
            raise LlmError(
                "http", f"The LLM service returned HTTP {response.status_code}.{hint}"
            )

        try:
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError):
            logger.warning("LLM response did not follow the chat-completions format")
            raise LlmError(
                "bad_response", "The LLM service returned an unexpected response format."
            ) from None
        if not isinstance(content, str) or not content.strip():
            raise LlmError("bad_response", "The LLM service returned an empty message.")
        return content

    def close(self) -> None:
        self._client.close()
