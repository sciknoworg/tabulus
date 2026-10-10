"""Evaluation metrics and Step-level scoring APIs for Tabulus."""

from tabulus.evaluation.bibliography import BibliographyEvaluation, evaluate_bibliography
from tabulus.evaluation.reference_matching import ReferenceMatchingEvaluation, evaluate_reference_matching
from tabulus.evaluation.reference_resolution import ReferenceResolutionEvaluation, evaluate_reference_resolution
from tabulus.evaluation.reference_table_classification import (
    ReferenceTableClassificationEvaluation,
    evaluate_reference_table_classification,
)
from tabulus.evaluation.rms import (
    DEFAULT_NUMBER_THRESHOLD,
    DEFAULT_TEXT_THRESHOLD,
    RMSScores,
    relative_mapping_similarity,
)
from tabulus.evaluation.table_localization import TableLocalizationEvaluation, evaluate_table_localization
from tabulus.evaluation.table_reconstruction import (
    SUPPORTED_TABLE_RECONSTRUCTION_METRICS,
    TableReconstructionEvaluation,
    evaluate_table_reconstruction,
)

__all__ = [
    "BibliographyEvaluation",
    "DEFAULT_NUMBER_THRESHOLD",
    "DEFAULT_TEXT_THRESHOLD",
    "RMSScores",
    "ReferenceMatchingEvaluation",
    "ReferenceResolutionEvaluation",
    "ReferenceTableClassificationEvaluation",
    "SUPPORTED_TABLE_RECONSTRUCTION_METRICS",
    "TableLocalizationEvaluation",
    "TableReconstructionEvaluation",
    "evaluate_bibliography",
    "evaluate_reference_matching",
    "evaluate_reference_resolution",
    "evaluate_reference_table_classification",
    "evaluate_table_localization",
    "evaluate_table_reconstruction",
    "relative_mapping_similarity",
]
