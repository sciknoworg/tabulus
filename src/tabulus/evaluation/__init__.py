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
    exact_mapping_match,
    MappingStatistics,
    evaluate_mapping_statistics,
    prepare_mappings,
    relative_mapping_similarity,
)
from tabulus.evaluation.grid_scoring import (
    PreparedGrid, prepare_grid, score_table_grids, aggregate_grid_results,
)
from tabulus.evaluation.grid_inputs import GridInput, FragmentPair, load_grid, load_fragment_manifest
from tabulus.evaluation.batch import aggregate_evaluation_runs, evaluate_pairs, summarize_results
from tabulus.evaluation.step7_physical import prepare_step7_pairs
from tabulus.evaluation.table_localization import TableLocalizationEvaluation, evaluate_table_localization
from tabulus.evaluation.step7_physical import evaluate_step7_physical
from tabulus.evaluation.table_reconstruction import (
    ExactCellCounts,
    SUPPORTED_TABLE_RECONSTRUCTION_METRICS,
    TableReconstructionEvaluation,
    count_exact_cells,
    evaluate_table_reconstruction,
)

__all__ = [
    "BibliographyEvaluation",
    "DEFAULT_NUMBER_THRESHOLD",
    "DEFAULT_TEXT_THRESHOLD",
    "ExactCellCounts",
    "RMSScores",
    "ReferenceMatchingEvaluation",
    "ReferenceResolutionEvaluation",
    "ReferenceTableClassificationEvaluation",
    "SUPPORTED_TABLE_RECONSTRUCTION_METRICS",
    "TableLocalizationEvaluation",
    "TableReconstructionEvaluation",
    "count_exact_cells",
    "exact_mapping_match",
    "MappingStatistics",
    "evaluate_mapping_statistics",
    "prepare_mappings",
    "PreparedGrid",
    "prepare_grid",
    "score_table_grids",
    "aggregate_grid_results",
    "GridInput",
    "FragmentPair",
    "load_grid",
    "load_fragment_manifest",
    "evaluate_pairs",
    "summarize_results",
    "aggregate_evaluation_runs",
    "prepare_step7_pairs",
    "evaluate_bibliography",
    "evaluate_reference_matching",
    "evaluate_reference_resolution",
    "evaluate_reference_table_classification",
    "evaluate_step7_physical",
    "evaluate_table_localization",
    "evaluate_table_reconstruction",
    "relative_mapping_similarity",
]
