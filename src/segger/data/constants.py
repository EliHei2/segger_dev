from enum import Enum, auto


class SpatialTranscriptomicsKeys(Enum):
    """Unified keys for spatial transcriptomics data, supporting multiple platforms."""

    # Files and directories
    TRANSCRIPTS_FILE = auto()
    BOUNDARIES_FILE = auto()
    CELL_METADATA_FILE = auto()

    # Cell identifiers
    CELL_ID = auto()
    TRANSCRIPTS_ID = auto()

    # Coordinates and locations
    TRANSCRIPTS_X = auto()
    TRANSCRIPTS_Y = auto()
    BOUNDARIES_VERTEX_X = auto()
    BOUNDARIES_VERTEX_Y = auto()
    GLOBAL_X = auto()
    GLOBAL_Y = auto()

    # Metadata
    METADATA_CELL_KEY = auto()
    COUNTS_CELL_KEY = auto()
    CELL_X = auto()
    CELL_Y = auto()
    FEATURE_NAME = auto()
    QUALITY_VALUE = auto()
    OVERLAPS_BOUNDARY = auto()


class XeniumKeys(Enum):
    """Keys for *10X Genomics Xenium* formatted dataset."""

    # File mappings
    TRANSCRIPTS_FILE = "transcripts.parquet"
    BOUNDARIES_FILE = "nucleus_boundaries.parquet"
    CELL_METADATA_FILE = None  # Not applicable for Xenium

    # Cell identifiers
    CELL_ID = "cell_id"
    TRANSCRIPTS_ID = "transcript_id"

    # Coordinates and locations
    TRANSCRIPTS_X = "x_location"
    TRANSCRIPTS_Y = "y_location"
    BOUNDARIES_VERTEX_X = "vertex_x"
    BOUNDARIES_VERTEX_Y = "vertex_y"

    # Metadata
    FEATURE_NAME = "feature_name"
    QUALITY_VALUE = "qv"
    OVERLAPS_BOUNDARY = "overlaps_nucleus"
    METADATA_CELL_KEY = None  # Not applicable for Xenium
    COUNTS_CELL_KEY = None  # Not applicable for Xenium
    CELL_X = None  # Not applicable for Xenium
    CELL_Y = None  # Not applicable for Xenium


class MerscopeKeys(Enum):
    """Keys for *MERSCOPE* data (Vizgen platform)."""

    # File mappings
    TRANSCRIPTS_FILE = "detected_transcripts.csv"
    BOUNDARIES_FILE = "cell_boundaries.parquet"
    CELL_METADATA_FILE = "cell_metadata.csv"

    # Cell identifiers
    CELL_ID = "EntityID"
    TRANSCRIPTS_ID = "transcript_id"

    # Coordinates and locations
    TRANSCRIPTS_X = "global_x"
    TRANSCRIPTS_Y = "global_y"
    BOUNDARIES_VERTEX_X = "center_x"
    BOUNDARIES_VERTEX_Y = "center_y"

    # Metadata
    FEATURE_NAME = "gene"
    QUALITY_VALUE = None  # Not applicable for Merscope
    OVERLAPS_BOUNDARY = None  # Not applicable for Merscope
    METADATA_CELL_KEY = "EntityID"
    COUNTS_CELL_KEY = "cell"
    CELL_X = "center_x"
    CELL_Y = "center_y"


class SpatialDataKeys(Enum):
    """Keys for *MERSCOPE* data (Vizgen platform)."""

    # File mappings
    TRANSCRIPTS_FILE = "detected_transcripts.csv"
    BOUNDARIES_FILE = "cell_boundaries.parquet"
    CELL_METADATA_FILE = "cell_metadata.csv"

    # Cell identifiers
    CELL_ID = "cell_id"
    TRANSCRIPTS_ID = "transcript_id"

    # Coordinates and locations
    TRANSCRIPTS_X = "x"
    TRANSCRIPTS_Y = "y"
    BOUNDARIES_VERTEX_X = "center_x"
    BOUNDARIES_VERTEX_Y = "center_y"

    # Metadata
    FEATURE_NAME = "<placeholder>"
    QUALITY_VALUE = "qv"
    OVERLAPS_BOUNDARY = "overlaps_nucleus"
    METADATA_CELL_KEY = None
    COUNTS_CELL_KEY = None
    CELL_X = None
    CELL_Y = None
