from __future__ import annotations

import json
import urllib.error
import urllib.parse

import pytest

import tabulus.reference_resolution.llm_client as llm_module
from tabulus.reference_resolution import (
    CandidateScore,
    CoreAssessment,
    CoreAssessmentStatus,
    CrossrefAssessment,
    CrossrefAssessmentStatus,
    LLMDecisionType,
    OpenAICompatibleLLMClient,
    OpenAICompatibleLLMError,
    RankedCandidate,
    ReferenceEvidence,
    ResolutionCandidate,
    ScholarlyResolution,
    ScholarlyResolutionStatus,
    build_llm_adjudication_case,
)


class _FakeResponse:
    def __init__(
        self,
        payload: dict,
    ):
        self.payload = json.dumps(
            payload
        ).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc,
        tb,
    ):
        return False

    def read(self):
        return self.payload


def _score() -> CandidateScore:
    return CandidateScore(
        score=0.72,
        comparable_weight=0.75,
        field_scores=(
            ("authors", 1.0),
            ("year", 1.0),
        ),
        sufficient_evidence=False,
        strong_match=False,
    )


def _case():
    evidence = ReferenceEvidence(
        reference_index=7,
        raw_reference="Smith J. Example citation. 2020.",
        authors=("J. Smith",),
        year=2020,
    )

    crossref_candidate = ResolutionCandidate(
        source="crossref",
        source_id="10.1000/a",
        doi="10.1000/a",
        title="Candidate A",
        authors=("Jane Smith",),
        year=2020,
    )

    core_candidate = ResolutionCandidate(
        source="core",
        source_id="123",
        doi="10.1000/b",
        title="Candidate B",
        authors=("Smith, J.",),
        year=2020,
    )

    crossref = CrossrefAssessment(
        reference_index=7,
        status=CrossrefAssessmentStatus.NEEDS_CORE,
        selected_candidate=None,
        selected_score=None,
        ranked_candidates=(
            RankedCandidate(
                crossref_candidate,
                _score(),
            ),
        ),
        reason="Ambiguous.",
    )

    core = CoreAssessment(
        reference_index=7,
        status=CoreAssessmentStatus.NEEDS_LLM,
        selected_candidate=None,
        selected_score=None,
        ranked_candidates=(
            RankedCandidate(
                core_candidate,
                _score(),
            ),
        ),
        reason="Ambiguous.",
    )

    resolution = ScholarlyResolution(
        reference_index=7,
        raw_reference=evidence.raw_reference,
        status=ScholarlyResolutionStatus.NEEDS_LLM,
        selected_candidate=None,
        selected_score=None,
        crossref_assessment=crossref,
        core_assessment=core,
        reason="Needs LLM.",
    )

    return build_llm_adjudication_case(
        evidence,
        resolution,
    )


def test_client_requires_configuration() -> None:
    with pytest.raises(
        ValueError,
        match="base_url",
    ):
        OpenAICompatibleLLMClient(
            base_url="",
            api_key="secret",
            model="model",
        )

    with pytest.raises(
        ValueError,
        match="api_key",
    ):
        OpenAICompatibleLLMClient(
            base_url="https://example.org/v1",
            api_key="",
            model="model",
        )

    with pytest.raises(
        ValueError,
        match="model",
    ):
        OpenAICompatibleLLMClient(
            base_url="https://example.org/v1",
            api_key="secret",
            model="",
        )


def test_client_builds_verified_kisski_style_request(
    monkeypatch,
) -> None:
    seen = {}

    def fake_urlopen(
        request,
        timeout,
    ):
        seen["url"] = request.full_url
        seen["authorization"] = request.get_header(
            "Authorization"
        )
        seen["timeout"] = timeout
        seen["payload"] = json.loads(
            request.data.decode("utf-8")
        )

        return _FakeResponse(
            {
                "id": "chatcmpl-test",
                "model": "qwen3.6-35b-a3b",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": (
                                '{"decision":"select_candidate",'
                                '"candidate_id":"core:1",'
                                '"confidence":0.9,'
                                '"evidence":["supported"]}'
                            ),
                            "reasoning": None,
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 52,
                    "completion_tokens": 29,
                    "total_tokens": 81,
                },
            }
        )

    monkeypatch.setattr(
        llm_module.urllib.request,
        "urlopen",
        fake_urlopen,
    )

    client = OpenAICompatibleLLMClient(
        base_url=(
            "https://chat-ai.academiccloud.de/v1"
        ),
        api_key="secret",
        model="qwen3.6-35b-a3b",
        timeout_seconds=20.0,
    )

    result = client.adjudicate(
        _case()
    )

    assert seen["url"] == (
        "https://chat-ai.academiccloud.de/"
        "v1/chat/completions"
    )
    assert seen["authorization"] == (
        "Bearer secret"
    )
    assert seen["timeout"] == 20.0

    payload = seen["payload"]

    assert payload["model"] == (
        "qwen3.6-35b-a3b"
    )
    assert payload["response_format"] == {
        "type": "json_object"
    }
    assert payload["chat_template_kwargs"] == {
        "enable_thinking": False
    }
    assert payload["temperature"] == 0.0
    assert payload["top_p"] == 1.0
    assert payload["presence_penalty"] == 0.0
    assert payload["max_tokens"] == 512

    assert result.decision.decision == (
        LLMDecisionType.SELECT_CANDIDATE
    )
    assert result.decision.candidate_id == (
        "core:1"
    )
    assert result.model == (
        "qwen3.6-35b-a3b"
    )
    assert result.finish_reason == "stop"
    assert result.usage.prompt_tokens == 52
    assert result.usage.completion_tokens == 29
    assert result.usage.total_tokens == 81


def test_client_rejects_invented_candidate(
    monkeypatch,
) -> None:
    def fake_urlopen(
        request,
        timeout,
    ):
        return _FakeResponse(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": (
                                '{"decision":"select_candidate",'
                                '"candidate_id":"core:999"}'
                            )
                        },
                    }
                ]
            }
        )

    monkeypatch.setattr(
        llm_module.urllib.request,
        "urlopen",
        fake_urlopen,
    )

    with pytest.raises(
        OpenAICompatibleLLMError,
        match="not supplied",
    ):
        OpenAICompatibleLLMClient(
            base_url="https://example.org/v1",
            api_key="secret",
            model="model",
        ).adjudicate(
            _case()
        )


def test_client_rejects_invented_doi_field(
    monkeypatch,
) -> None:
    def fake_urlopen(
        request,
        timeout,
    ):
        return _FakeResponse(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": (
                                '{"decision":"select_candidate",'
                                '"candidate_id":"core:1",'
                                '"doi":"10.9999/fake"}'
                            )
                        },
                    }
                ]
            }
        )

    monkeypatch.setattr(
        llm_module.urllib.request,
        "urlopen",
        fake_urlopen,
    )

    with pytest.raises(
        OpenAICompatibleLLMError,
        match="unsupported fields",
    ):
        OpenAICompatibleLLMClient(
            base_url="https://example.org/v1",
            api_key="secret",
            model="model",
        ).adjudicate(
            _case()
        )


def test_client_rejects_missing_content(
    monkeypatch,
) -> None:
    def fake_urlopen(
        request,
        timeout,
    ):
        return _FakeResponse(
            {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {
                            "content": None,
                            "reasoning": (
                                "unfinished reasoning"
                            ),
                        },
                    }
                ]
            }
        )

    monkeypatch.setattr(
        llm_module.urllib.request,
        "urlopen",
        fake_urlopen,
    )

    with pytest.raises(
        OpenAICompatibleLLMError,
        match="no textual content",
    ):
        OpenAICompatibleLLMClient(
            base_url="https://example.org/v1",
            api_key="secret",
            model="model",
        ).adjudicate(
            _case()
        )


def test_client_rejects_non_json_message_content(
    monkeypatch,
) -> None:
    def fake_urlopen(
        request,
        timeout,
    ):
        return _FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": "Candidate core:1 is best."
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr(
        llm_module.urllib.request,
        "urlopen",
        fake_urlopen,
    )

    with pytest.raises(
        OpenAICompatibleLLMError,
        match="not valid JSON",
    ):
        OpenAICompatibleLLMClient(
            base_url="https://example.org/v1",
            api_key="secret",
            model="model",
        ).adjudicate(
            _case()
        )


def test_client_http_error_is_explicit(
    monkeypatch,
) -> None:
    def fake_urlopen(
        request,
        timeout,
    ):
        raise urllib.error.HTTPError(
            request.full_url,
            429,
            "Too Many Requests",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr(
        llm_module.urllib.request,
        "urlopen",
        fake_urlopen,
    )

    with pytest.raises(
        OpenAICompatibleLLMError,
        match="HTTP 429",
    ):
        OpenAICompatibleLLMClient(
            base_url="https://example.org/v1",
            api_key="secret",
            model="model",
        ).adjudicate(
            _case()
        )


def test_client_url_error_is_explicit(
    monkeypatch,
) -> None:
    def fake_urlopen(
        request,
        timeout,
    ):
        raise urllib.error.URLError(
            "connection refused"
        )

    monkeypatch.setattr(
        llm_module.urllib.request,
        "urlopen",
        fake_urlopen,
    )

    with pytest.raises(
        OpenAICompatibleLLMError,
        match="connection refused",
    ):
        OpenAICompatibleLLMClient(
            base_url="https://example.org/v1",
            api_key="secret",
            model="model",
        ).adjudicate(
            _case()
        )



def test_request_json_retries_transient_http_500(
    monkeypatch,
) -> None:
    import io
    import urllib.error

    calls = []
    sleeps = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(
            self,
            exc_type,
            exc_value,
            traceback,
        ):
            return False

        def read(self):
            return b'{"ok": true}'

    def fake_urlopen(
        request,
        timeout,
    ):
        calls.append(
            request
        )

        if len(calls) == 1:
            raise urllib.error.HTTPError(
                request.full_url,
                500,
                "Internal Server Error",
                {},
                io.BytesIO(),
            )

        return FakeResponse()

    monkeypatch.setattr(
        llm_module.urllib.request,
        "urlopen",
        fake_urlopen,
    )

    monkeypatch.setattr(
        llm_module.time,
        "sleep",
        lambda seconds: sleeps.append(
            seconds
        ),
    )

    result = llm_module._request_json(
        "https://llm.example/v1/chat/completions",
        api_key="secret",
        payload={
            "model": "test"
        },
        timeout_seconds=10.0,
    )

    assert result == {
        "ok": True
    }

    assert len(calls) == 2
    assert sleeps == [1.0]


def test_client_expands_token_budget_after_truncation(
    monkeypatch,
) -> None:
    seen_max_tokens = []

    responses = iter(
        (
            {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {
                            "content": (
                                '{"decision":"select_candidate",'
                                '"candidate_id":"core:1",'
                                '"evidence":["unterminated'
                            )
                        },
                    }
                ]
            },
            {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {
                            "content": (
                                '{"decision":"select_candidate",'
                                '"candidate_id":"core:1",'
                                '"evidence":["still unterminated'
                            )
                        },
                    }
                ]
            },
            {
                "id": "chatcmpl-expanded",
                "model": "model",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": (
                                '{"decision":"select_candidate",'
                                '"candidate_id":"core:1",'
                                '"confidence":0.9,'
                                '"evidence":["supported"]}'
                            )
                        },
                    }
                ],
            },
        )
    )

    def fake_urlopen(
        request,
        timeout,
    ):
        payload = json.loads(
            request.data.decode("utf-8")
        )
        seen_max_tokens.append(
            payload["max_tokens"]
        )
        return _FakeResponse(next(responses))

    monkeypatch.setattr(
        llm_module.urllib.request,
        "urlopen",
        fake_urlopen,
    )
    monkeypatch.setattr(
        llm_module.time,
        "sleep",
        lambda seconds: None,
    )

    result = OpenAICompatibleLLMClient(
        base_url="https://example.org/v1",
        api_key="secret",
        model="model",
    ).adjudicate(_case())

    assert seen_max_tokens == [
        512,
        1024,
        2048,
    ]
    assert result.decision.decision == (
        LLMDecisionType.SELECT_CANDIDATE
    )
    assert result.decision.candidate_id == "core:1"
    assert result.finish_reason == "stop"


def test_client_fails_when_truncation_reaches_bounded_ceiling(
    monkeypatch,
) -> None:
    seen_max_tokens = []

    def fake_urlopen(
        request,
        timeout,
    ):
        payload = json.loads(
            request.data.decode("utf-8")
        )
        seen_max_tokens.append(
            payload["max_tokens"]
        )

        return _FakeResponse(
            {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {
                            "content": (
                                '{"decision":"select_candidate",'
                                '"candidate_id":"core:1",'
                                '"evidence":["unterminated'
                            )
                        },
                    }
                ]
            }
        )

    monkeypatch.setattr(
        llm_module.urllib.request,
        "urlopen",
        fake_urlopen,
    )
    monkeypatch.setattr(
        llm_module.time,
        "sleep",
        lambda seconds: None,
    )

    with pytest.raises(
        OpenAICompatibleLLMError,
        match=(
            "remained truncated.*"
            "bounded max_tokens ceiling.*2048"
        ),
    ):
        OpenAICompatibleLLMClient(
            base_url="https://example.org/v1",
            api_key="secret",
            model="model",
        ).adjudicate(_case())

    assert seen_max_tokens == [
        512,
        1024,
        2048,
    ]


def test_client_records_provider_and_can_omit_chat_template_kwargs(
    monkeypatch,
) -> None:
    seen = {}

    def fake_urlopen(
        request,
        timeout,
    ):
        seen["payload"] = json.loads(
            request.data.decode("utf-8")
        )

        return _FakeResponse(
            {
                "model": "qwen/qwen3.6-35b-a3b",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": (
                                '{"decision":"reject_all",'
                                '"confidence":0.9,'
                                '"evidence":["unsupported"]}'
                            )
                        },
                    }
                ],
            }
        )

    monkeypatch.setattr(
        llm_module.urllib.request,
        "urlopen",
        fake_urlopen,
    )

    result = OpenAICompatibleLLMClient(
        base_url="https://openrouter.ai/api/v1",
        api_key="secret",
        model="qwen/qwen3.6-35b-a3b",
        provider_name="openrouter",
        include_chat_template_kwargs=False,
        reasoning_effort="none",
    ).adjudicate(
        _case()
    )

    assert (
        "chat_template_kwargs"
        not in seen["payload"]
    )
    assert seen["payload"]["reasoning"] == {
        "effort": "none",
    }
    assert result.provider == "openrouter"
    assert result.to_dict()["provider"] == "openrouter"


def test_failover_client_uses_primary_when_healthy() -> None:
    class Primary:
        def adjudicate(self, case):
            return "primary-result"

    class Fallback:
        def adjudicate(self, case):
            raise AssertionError(
                "Fallback must not be called."
            )

    client = llm_module.FailoverLLMClient(
        primary_client=Primary(),
        fallback_client=Fallback(),
    )

    assert client.adjudicate(_case()) == (
        "primary-result"
    )


def test_failover_client_uses_fallback_after_primary_failure() -> None:
    calls = []

    class Primary:
        def adjudicate(self, case):
            calls.append("primary")
            raise OpenAICompatibleLLMError(
                "HTTP 500 after bounded attempts"
            )

    class Fallback:
        def adjudicate(self, case):
            calls.append("fallback")
            return "fallback-result"

    client = llm_module.FailoverLLMClient(
        primary_client=Primary(),
        fallback_client=Fallback(),
    )

    assert client.adjudicate(_case()) == (
        "fallback-result"
    )
    assert calls == [
        "primary",
        "fallback",
    ]


def test_failover_client_probes_primary_again_on_next_adjudication() -> None:
    calls = []
    primary_attempts = 0

    class Primary:
        def adjudicate(self, case):
            nonlocal primary_attempts

            primary_attempts += 1
            calls.append("primary")

            if primary_attempts == 1:
                raise OpenAICompatibleLLMError(
                    "temporary outage"
                )

            return "primary-recovered"

    class Fallback:
        def adjudicate(self, case):
            calls.append("fallback")
            return "fallback-result"

    client = llm_module.FailoverLLMClient(
        primary_client=Primary(),
        fallback_client=Fallback(),
    )

    assert client.adjudicate(_case()) == (
        "fallback-result"
    )

    assert client.adjudicate(_case()) == (
        "primary-recovered"
    )

    assert calls == [
        "primary",
        "fallback",
        "primary",
    ]


def test_failover_client_raises_when_both_providers_fail() -> None:
    class Primary:
        def adjudicate(self, case):
            raise OpenAICompatibleLLMError(
                "primary unavailable"
            )

    class Fallback:
        def adjudicate(self, case):
            raise OpenAICompatibleLLMError(
                "fallback unavailable"
            )

    client = llm_module.FailoverLLMClient(
        primary_client=Primary(),
        fallback_client=Fallback(),
    )

    with pytest.raises(
        OpenAICompatibleLLMError,
        match="Both LLM providers failed",
    ):
        client.adjudicate(
            _case()
        )
