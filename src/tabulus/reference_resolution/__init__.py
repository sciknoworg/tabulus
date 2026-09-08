from tabulus.reference_resolution.fallback import (
    CoreAssessment,
    CoreAssessmentStatus,
    ScholarlyResolution,
    ScholarlyResolutionStatus,
    assess_core,
    resolve_crossref_then_core,
)
from tabulus.reference_resolution.llm_contract import (
    DEFAULT_LLM_CANDIDATES_PER_SOURCE,
    LLMAdjudicationCase,
    LLMCandidate,
    LLMDecision,
    LLMDecisionType,
    build_llm_adjudication_case,
    parse_llm_decision,
)
from tabulus.reference_resolution.orchestrator import (
    ReferenceResolutionRunResult,
    resolve_reference_artifact,
    resolve_reference_artifact_with_clients,
)
from tabulus.reference_resolution.artifact import (
    REFERENCE_RESOLUTION_NAME,
    REFERENCE_RESOLUTION_SCHEMA_VERSION,
    build_reference_resolution_payload,
    default_reference_resolution_path,
    write_reference_resolution_artifact,
    write_reference_resolution_json,
)
from tabulus.reference_resolution.finalize import (
    ReferenceResolutionTrace,
    finalize_reference_resolution,
)
from tabulus.reference_resolution.llm_client import (
    LLMAdjudicationResponse,
    LLMUsage,
    OpenAICompatibleLLMClient,
    OpenAICompatibleLLMError,
)
from tabulus.reference_resolution.models import (
    CandidateScore,
    ReferenceEvidence,
    ReferenceResolution,
    ResolutionCandidate,
    ResolutionStatus,
)
from tabulus.reference_resolution.assessment import (
    CrossrefAssessment,
    CrossrefAssessmentStatus,
    RankedCandidate,
    assess_crossref,
    assess_crossref_retrievals,
    rank_candidates,
)
from tabulus.reference_resolution.core import (
    CoreClient,
    CoreError,
    CoreRateLimit,
    CoreSearchResponse,
    candidate_from_core_work,
)
from tabulus.reference_resolution.crossref import (
    CrossrefClient,
    CrossrefError,
    candidate_from_crossref_item,
)
from tabulus.reference_resolution.pipeline import (
    CrossrefRetrieval,
    collect_resolution_targets,
    retrieve_crossref_evidence,
)
from tabulus.reference_resolution.scoring import (
    FIELD_WEIGHTS,
    STRONG_MATCH_THRESHOLD,
    normalize_doi,
    score_candidate,
)

__all__ = [
    "ReferenceResolutionRunResult",
    "REFERENCE_RESOLUTION_NAME",
    "REFERENCE_RESOLUTION_SCHEMA_VERSION",
    "ReferenceResolutionTrace",
    "LLMAdjudicationResponse",
    "LLMUsage",
    "OpenAICompatibleLLMClient",
    "OpenAICompatibleLLMError",
    "LLMAdjudicationCase",
    "LLMCandidate",
    "LLMDecision",
    "LLMDecisionType",
    "CoreAssessment",
    "CoreAssessmentStatus",
    "ScholarlyResolution",
    "ScholarlyResolutionStatus",
    "CoreClient",
    "CoreError",
    "CoreRateLimit",
    "CoreSearchResponse",
    "CrossrefAssessment",
    "CrossrefAssessmentStatus",
    "RankedCandidate",
    "CrossrefRetrieval",
    "CrossrefClient",
    "CrossrefError",
    "CandidateScore",
    "FIELD_WEIGHTS",
    "ReferenceEvidence",
    "ReferenceResolution",
    "ResolutionCandidate",
    "ResolutionStatus",
    "STRONG_MATCH_THRESHOLD",
    "assess_core",
    "assess_crossref",
    "assess_crossref_retrievals",
    "collect_resolution_targets",
    "build_reference_resolution_payload",
    "build_llm_adjudication_case",
    "candidate_from_core_work",
    "candidate_from_crossref_item",
    "default_reference_resolution_path",
    "finalize_reference_resolution",
    "parse_llm_decision",
    "rank_candidates",
    "resolve_reference_artifact",
    "resolve_reference_artifact_with_clients",
    "resolve_crossref_then_core",
    "write_reference_resolution_artifact",
    "write_reference_resolution_json",
    "retrieve_crossref_evidence",
    "normalize_doi",
    "score_candidate",
]
