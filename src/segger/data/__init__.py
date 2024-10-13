"""
Data module for Segger.

Contains utilities for handling and processing spatial transcriptomics data.
"""

__all__ = [
    "XeniumSample",
    "MerscopeSample",
    "SpatialTranscriptomicsDataset",
    "filter_transcripts",
    "create_anndata",
    "compute_transcript_metrics",
    "SpatialTranscriptomicsSample",
    "calculate_gene_celltype_abundance_embedding",
    "get_edge_index",
]

from segger.data.utils import (
    filter_transcripts,
    create_anndata,
    compute_transcript_metrics,
    get_edge_index,
    calculate_gene_celltype_abundance_embedding,
    SpatialTranscriptomicsDataset,
)

from segger.data.io import (
    XeniumSample,
    MerscopeSample,
    SpatialTranscriptomicsSample,
)

from segger.data.constants import SpatialTranscriptomicsKeys, XeniumKeys, MerscopeKeys
