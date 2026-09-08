from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any
import urllib.error
import urllib.request

from tabulus.reference_resolution.llm_contract import (
    LLMAdjudicationCase,
    LLMDecision,
    parse_llm_decision,
)


DEFAULT_LLM_TIMEOUT_SECONDS = 60.0
DEFAULT_LLM_MAX_TOKENS = 512
DEFAULT_LLM_MAX_TRUNCATION_TOKENS = 2048
DEFAULT_LLM_TEMPERATURE = 0.0
DEFAULT_LLM_TOP_P = 1.0
DEFAULT_LLM_PRESENCE_PENALTY = 0.0
class OpenAICompatibleLLMError(RuntimeError):
    """Raised when an OpenAI-compatible endpoint returns unusable output."""


@dataclass(frozen=True)
class LLMUsage:
    """Token usage reported by an OpenAI-compatible chat endpoint."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True)
class LLMAdjudicationResponse:
    """Validated model response plus reproducibility metadata."""

    decision: LLMDecision
    model: str
    response_id: str
    finish_reason: str
    usage: LLMUsage
    provider: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "decision": self.decision.to_dict(),
            "model": self.model,
            "response_id": self.response_id,
            "finish_reason": self.finish_reason,
            "usage": self.usage.to_dict(),
        }

        if self.provider:
            payload["provider"] = self.provider

        return payload


def _optional_int(
    value: Any,
) -> int | None:
    if isinstance(value, bool):
        return None

    if isinstance(value, int):
        return value

    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None

    return None


def _usage_from_payload(
    payload: Any,
) -> LLMUsage:
    if not isinstance(payload, dict):
        return LLMUsage()

    return LLMUsage(
        prompt_tokens=_optional_int(
            payload.get("prompt_tokens")
        ),
        completion_tokens=_optional_int(
            payload.get("completion_tokens")
        ),
        total_tokens=_optional_int(
            payload.get("total_tokens")
        ),
    )


def _system_prompt() -> str:
    return (
        "You adjudicate bibliographic identity for Tabulus. "
        "You must use only the supplied citation evidence and candidate "
        "records. Never invent a DOI, title, author, venue, or publication. "
        "Return exactly one JSON object with these allowed fields only: "
        "decision, candidate_id, search_query, confidence, evidence. "
        "Use exactly these ASCII field names and never translate or rename them. "
        "The evidence field should be a JSON array of short strings. "
        "If confidence is supplied, it must be a JSON number from 0 to 1, "
        "not a quoted string. "
        "The decision must be one of select_candidate, retry_search, or "
        "reject_all. For select_candidate, choose only a candidate_id "
        "explicitly supplied by Tabulus. For retry_search, provide a concise "
        "bibliographic search query justified by the supplied citation. "
        "For reject_all, do not supply a candidate_id or search_query. "
        "Do not include markdown or prose outside the JSON object."
    )


def _user_prompt(
    case: LLMAdjudicationCase,
) -> str:
    return (
        "Adjudicate this unresolved bibliographic reference.\n\n"
        + json.dumps(
            case.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


LOGGER = logging.getLogger(__name__)

DEFAULT_LLM_MAX_ATTEMPTS = 4
DEFAULT_LLM_BACKOFF_SECONDS = 1.0
DEFAULT_LLM_MAX_RETRY_DELAY_SECONDS = 30.0

LLM_RETRYABLE_HTTP_STATUSES = {
    429,
    500,
    502,
    503,
    504,
}


def _request_json(
    url: str,
    *,
    api_key: str,
    payload: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    body = json.dumps(
        payload,
        ensure_ascii=False,
    ).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "tabulus/0.1",
        },
        method="POST",
    )

    response_body = b""

    for attempt in range(
        1,
        DEFAULT_LLM_MAX_ATTEMPTS + 1,
    ):
        try:
            with urllib.request.urlopen(
                request,
                timeout=timeout_seconds,
            ) as response:
                response_body = response.read()

            break

        except urllib.error.HTTPError as error:
            retryable = (
                error.code
                in LLM_RETRYABLE_HTTP_STATUSES
            )

            if (
                retryable
                and attempt
                < DEFAULT_LLM_MAX_ATTEMPTS
            ):
                delay = (
                    DEFAULT_LLM_BACKOFF_SECONDS
                    * (2 ** (attempt - 1))
                )

                retry_after = (
                    error.headers.get(
                        "Retry-After"
                    )
                    if error.headers is not None
                    else None
                )

                if retry_after:
                    try:
                        parsed_retry_after = float(
                            retry_after
                        )

                        if parsed_retry_after >= 0:
                            delay = parsed_retry_after

                    except ValueError:
                        pass

                delay = min(
                    delay,
                    DEFAULT_LLM_MAX_RETRY_DELAY_SECONDS,
                )

                LOGGER.warning(
                    "[LLM] transient HTTP %s; "
                    "retrying in %.1fs "
                    "(attempt %d/%d)",
                    error.code,
                    delay,
                    attempt + 1,
                    DEFAULT_LLM_MAX_ATTEMPTS,
                )

                time.sleep(
                    delay
                )

                continue

            raise OpenAICompatibleLLMError(
                "LLM endpoint returned HTTP "
                f"{error.code} after "
                f"{attempt} attempt(s)."
            ) from error

        except TimeoutError as error:
            if attempt < DEFAULT_LLM_MAX_ATTEMPTS:
                delay = min(
                    DEFAULT_LLM_BACKOFF_SECONDS
                    * (2 ** (attempt - 1)),
                    DEFAULT_LLM_MAX_RETRY_DELAY_SECONDS,
                )

                LOGGER.warning(
                    "[LLM] request timed out; "
                    "retrying in %.1fs "
                    "(attempt %d/%d)",
                    delay,
                    attempt + 1,
                    DEFAULT_LLM_MAX_ATTEMPTS,
                )

                time.sleep(delay)
                continue

            raise OpenAICompatibleLLMError(
                "LLM request timed out after "
                f"{attempt} attempt(s)."
            ) from error

        except urllib.error.URLError as error:
            raise OpenAICompatibleLLMError(
                f"Could not reach LLM endpoint: {error.reason}"
            ) from error

    if not response_body:
        raise OpenAICompatibleLLMError(
            "LLM endpoint returned an empty response."
        )

    try:
        value = json.loads(
            response_body.decode("utf-8")
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as error:
        raise OpenAICompatibleLLMError(
            "LLM endpoint returned invalid JSON."
        ) from error

    if not isinstance(value, dict):
        raise OpenAICompatibleLLMError(
            "LLM response must be a JSON object."
        )

    return value


@dataclass(frozen=True)
class OpenAICompatibleLLMClient:
    """Dependency-free client for bounded Stage 6 LLM adjudication.

    The client is intentionally provider-neutral. Any endpoint implementing
    the OpenAI-compatible ``chat/completions`` contract may be used by
    supplying its base URL, API key, and model name.

    Thinking is disabled by default because Stage 6 is a bounded candidate
    adjudication task. Sampling is also minimized by default so repeated
    resolution runs are suitable for controlled ablation experiments.
    Strict validation remains in Tabulus regardless of what the model returns.
    """

    base_url: str
    api_key: str
    model: str
    timeout_seconds: float = DEFAULT_LLM_TIMEOUT_SECONDS
    max_tokens: int = DEFAULT_LLM_MAX_TOKENS
    temperature: float = DEFAULT_LLM_TEMPERATURE
    top_p: float = DEFAULT_LLM_TOP_P
    presence_penalty: float = (
        DEFAULT_LLM_PRESENCE_PENALTY
    )
    enable_thinking: bool = False
    provider_name: str = "openai-compatible"
    include_chat_template_kwargs: bool = True
    reasoning_effort: str | None = None

    def __post_init__(self) -> None:
        if not self.base_url.strip():
            raise ValueError(
                "LLM base_url must be provided."
            )

        if not self.api_key.strip():
            raise ValueError(
                "LLM api_key must be provided."
            )

        if not self.model.strip():
            raise ValueError(
                "LLM model must be provided."
            )

        if not self.provider_name.strip():
            raise ValueError(
                "LLM provider_name must be provided."
            )

        if self.timeout_seconds <= 0:
            raise ValueError(
                "LLM timeout_seconds must be greater than zero."
            )

        if self.max_tokens <= 0:
            raise ValueError(
                "LLM max_tokens must be greater than zero."
            )

        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(
                "LLM temperature must be between 0 and 2."
            )

        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError(
                "LLM top_p must be between 0 and 1."
            )

    @property
    def chat_completions_url(self) -> str:
        return (
            f"{self.base_url.rstrip('/')}"
            "/chat/completions"
        )

    def adjudicate(
        self,
        case: LLMAdjudicationCase,
    ) -> LLMAdjudicationResponse:
        """Run one bounded LLM adjudication and strictly validate its output."""

        payload = {
            "model": self.model.strip(),
            "messages": [
                {
                    "role": "system",
                    "content": _system_prompt(),
                },
                {
                    "role": "user",
                    "content": _user_prompt(case),
                },
            ],
            "response_format": {
                "type": "json_object",
            },
            "temperature": self.temperature,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
            "max_tokens": self.max_tokens,
        }

        if self.include_chat_template_kwargs:
            payload["chat_template_kwargs"] = {
                "enable_thinking": self.enable_thinking,
            }

        if self.reasoning_effort is not None:
            payload["reasoning"] = {
                "effort": self.reasoning_effort,
            }

        for attempt in range(
            1,
            DEFAULT_LLM_MAX_ATTEMPTS + 1,
        ):
            response = _request_json(
                self.chat_completions_url,
                api_key=self.api_key.strip(),
                payload=payload,
                timeout_seconds=self.timeout_seconds,
            )

            choices = response.get("choices")

            if (
                not isinstance(choices, list)
                or not choices
                or not isinstance(choices[0], dict)
            ):
                raise OpenAICompatibleLLMError(
                    "LLM response has no usable choices."
                )

            first_choice = choices[0]

            message = first_choice.get("message")

            if not isinstance(message, dict):
                raise OpenAICompatibleLLMError(
                    "LLM response choice has no message object."
                )

            content = message.get("content")

            if not isinstance(content, str) or not content.strip():
                raise OpenAICompatibleLLMError(
                    "LLM response message has no textual content."
                )

            try:
                decision_payload = json.loads(content)
            except json.JSONDecodeError as error:
                finish_reason = str(
                    first_choice.get("finish_reason")
                    or ""
                ).strip()

                if finish_reason == "length":
                    current_max_tokens = int(
                        payload["max_tokens"]
                    )

                    truncation_ceiling = max(
                        self.max_tokens,
                        DEFAULT_LLM_MAX_TRUNCATION_TOKENS,
                    )

                    next_max_tokens = min(
                        current_max_tokens * 2,
                        truncation_ceiling,
                    )

                    if (
                        attempt < DEFAULT_LLM_MAX_ATTEMPTS
                        and next_max_tokens
                        > current_max_tokens
                    ):
                        delay = min(
                            DEFAULT_LLM_BACKOFF_SECONDS
                            * (2 ** (attempt - 1)),
                            DEFAULT_LLM_MAX_RETRY_DELAY_SECONDS,
                        )

                        payload["max_tokens"] = (
                            next_max_tokens
                        )

                        LOGGER.warning(
                            "[LLM] message content was truncated "
                            "at max_tokens=%d; retrying with "
                            "max_tokens=%d in %.1fs "
                            "(attempt %d/%d)",
                            current_max_tokens,
                            next_max_tokens,
                            delay,
                            attempt + 1,
                            DEFAULT_LLM_MAX_ATTEMPTS,
                        )

                        time.sleep(delay)
                        continue

                    raise OpenAICompatibleLLMError(
                        "LLM message content remained truncated "
                        "at the bounded max_tokens ceiling "
                        f"({current_max_tokens})."
                    ) from error

                if attempt < DEFAULT_LLM_MAX_ATTEMPTS:
                    delay = min(
                        DEFAULT_LLM_BACKOFF_SECONDS
                        * (2 ** (attempt - 1)),
                        DEFAULT_LLM_MAX_RETRY_DELAY_SECONDS,
                    )

                    LOGGER.warning(
                        "[LLM] message content was invalid JSON; "
                        "retrying adjudication in %.1fs "
                        "(attempt %d/%d)",
                        delay,
                        attempt + 1,
                        DEFAULT_LLM_MAX_ATTEMPTS,
                    )

                    time.sleep(delay)
                    continue

                raise OpenAICompatibleLLMError(
                    "LLM message content is not valid JSON "
                    f"after {attempt} attempt(s)."
                ) from error

            try:
                decision = parse_llm_decision(
                    decision_payload,
                    case,
                )
            except ValueError as error:
                raise OpenAICompatibleLLMError(
                    f"LLM decision failed Tabulus validation: {error}"
                ) from error

            break

        return LLMAdjudicationResponse(
            decision=decision,
            model=str(
                response.get("model")
                or self.model
            ).strip(),
            response_id=str(
                response.get("id") or ""
            ).strip(),
            finish_reason=str(
                first_choice.get("finish_reason")
                or ""
            ).strip(),
            usage=_usage_from_payload(
                response.get("usage")
            ),
            provider=self.provider_name.strip(),
        )


@dataclass
class FailoverLLMClient:
    """Prefer one LLM provider and fall back only on operational failure.

    Every adjudication starts with the primary provider. If its bounded
    retries or output validation are exhausted, the identical adjudication
    is attempted through the fallback provider. The next adjudication probes
    the primary again, so recovery is detected automatically.
    """

    primary_client: Any
    fallback_client: Any
    primary_provider: str = "kisski"
    fallback_provider: str = "openrouter"

    def adjudicate(
        self,
        case: LLMAdjudicationCase,
    ) -> LLMAdjudicationResponse:
        try:
            return self.primary_client.adjudicate(
                case
            )
        except OpenAICompatibleLLMError as primary_error:
            LOGGER.warning(
                "[LLM] primary provider %s failed after "
                "bounded attempts; falling back to %s "
                "for this adjudication: %s",
                self.primary_provider,
                self.fallback_provider,
                primary_error,
            )

        try:
            return self.fallback_client.adjudicate(
                case
            )
        except OpenAICompatibleLLMError as fallback_error:
            raise OpenAICompatibleLLMError(
                "Both LLM providers failed for the same "
                "adjudication: "
                f"primary={self.primary_provider}, "
                f"fallback={self.fallback_provider}."
            ) from fallback_error

