import os
import shapely
from pyarrow import parquet as pq, compute as pc
import numpy as np
import pandas as pd
from pathlib import Path
import geopandas as gpd
from segger.data.parquet import _utils as utils
from scipy.spatial import KDTree, Rectangle
from segger.data.parquet._ndtree import NDTree
from functools import cached_property
from typing import List, Optional
import logging
from itertools import compress
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit
import torch
from pqdm.threads import pqdm
import random
from segger.data.parquet.transcript_embedding import TranscriptEmbedding


# TODO: Add documentation for settings
class STSampleParquet:
    """
    A class to manage spatial transcriptomics data stored in parquet files.

    This class provides methods for loading, processing, and saving data related
    to ST samples. It supports parallel processing and efficient handling of
    transcript and boundary data.
    """

    def __init__(
        self,
        base_dir: os.PathLike,
        n_workers: Optional[int] = 1,
        sample_type: str = None,
        weights: pd.DataFrame = None,
    ):
        """
        Initializes the STSampleParquet instance.

        Parameters
        ----------
        base_dir : os.PathLike
            The base directory containing the ST data.
        n_workers : Optional[int], default 1
            The number of workers for parallel processing.
        sample_type : Optional[str], default None
            The sample type of the raw data, e.g., 'xenium' or 'merscope'

        Raises
        ------
        FileNotFoundError
            If the base directory does not exist or the required files are
            missing.
        """
        # Setup paths and resource constraints
        self._base_dir = Path(base_dir)
        self.settings = utils.load_settings(sample_type)
        transcripts_fn = self.settings.transcripts.filename
        self._transcripts_filepath = self._base_dir / transcripts_fn
        boundaries_fn = self.settings.boundaries.filename
        self._boundaries_filepath = self._base_dir / boundaries_fn
        self.n_workers = n_workers

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.Logger(f"STSample@{base_dir}")

        # Internal caches
        self._extents = None
        self._transcripts_metadata = None
        self._boundaries_metadata = None

        # Setup default embedding for transcripts
        self._emb_genes = None
        if weights is not None:
            self._emb_genes = weights.index.to_list()
        classes = self.transcripts_metadata["feature_names"]
        self._transcript_embedding = TranscriptEmbedding(np.array(classes), weights)

    @classmethod
    def _get_parquet_metadata(
        cls,
        filepath: os.PathLike,
        columns: Optional[List[str]] = None,
    ) -> dict:
        """
        Reads and returns metadata from the parquet file.

        Parameters
        ----------
        filepath : os.PathLike
            The path to the parquet file.
        columns : Optional[List[str]], default None
            List of columns to extract metadata for. If None, all columns
            are used.

        Returns
        -------
        dict
            A dictionary containing metadata such as the number of rows,
            number of columns, and column sizes.

        Raises
        ------
        FileNotFoundError
            If the parquet file does not exist at the specified path.
        KeyError
            If any of the requested columns are not found in the parquet file.
        """
        # Size in bytes of field dtypes
        size_map = {
            "BOOLEAN": 1,
            "INT32": 4,
            "FLOAT": 4,
            "INT64": 8,
            "DOUBLE": 8,
            "BYTE_ARRAY": 8,
            "INT96": 12,
        }

        # Read in metadata
        metadata = pq.read_metadata(filepath)
        if columns is None:
            columns = metadata.schema.names
        missing = set(columns) - set(metadata.schema.names)
        if len(missing) > 0:
            msg = f"Columns {', '.join(missing)} not found in schema."
            raise KeyError(msg)

        # Grab important fields from metadata
        summary = dict()
        summary["n_rows"] = metadata.num_rows
        summary["n_columns"] = len(columns)
        summary["column_sizes"] = dict()
        for c in columns:
            # Error where 10X saved BOOLEAN field as INT32 in schema
            if c == "overlaps_nucleus":
                dtype = "BOOLEAN"
            else:
                i = metadata.schema.names.index(c)
                dtype = metadata.schema[i].physical_type
            summary["column_sizes"][c] = size_map[dtype]

        return summary

    @cached_property
    def transcripts_metadata(self) -> dict:
        """
        Retrieves metadata for the transcripts stored in the sample.

        Returns
        -------
        dict
            Metadata dictionary for transcripts including column sizes and
            feature names.

        Raises
        ------
        FileNotFoundError
            If the transcript parquet file does not exist.
        """
        if self._transcripts_metadata is None:
            # Base metadata
            metadata = STSampleParquet._get_parquet_metadata(
                self._transcripts_filepath,
                self.settings.transcripts.columns,
            )
            # Get filtered unique feature names
            table = pq.read_table(self._transcripts_filepath)
            names = pc.unique(table[self.settings.transcripts.label])
            if self._emb_genes is not None:
                # Filter substring is extended with the genes missing in the embedding
                names_str = [x.decode("utf-8") if isinstance(x, bytes) else x for x in names.to_pylist()]
                missing_genes = list(set(names_str) - set(self._emb_genes))
                logging.warning(f"Number of missing genes: {len(missing_genes)}")
                self.settings.transcripts.filter_substrings.extend(missing_genes)
            pattern = "|".join(self.settings.transcripts.filter_substrings)
            mask = pc.invert(pc.match_substring_regex(names, pattern))
            filtered_names = pc.filter(names, mask).to_pylist()
            metadata["feature_names"] = [x.decode("utf-8") if isinstance(x, bytes) else x for x in filtered_names]
            self._transcripts_metadata = metadata
        return self._transcripts_metadata

    @cached_property
    def boundaries_metadata(self) -> dict:
        """
        Retrieves metadata for the boundaries stored in the sample.

        Returns
        -------
        dict
            Metadata dictionary for boundaries including column sizes.

        Raises
        ------
        FileNotFoundError
            If the boundaries parquet file does not exist.
        """
        if self._boundaries_metadata is None:
            metadata = STSampleParquet._get_parquet_metadata(
                self._boundaries_filepath,
                self.settings.boundaries.columns,
            )
            self._boundaries_metadata = metadata
        return self._boundaries_metadata

    @property
    def n_transcripts(self) -> int:
        """
        The total number of transcripts in the sample.

        Returns
        -------
        int
            The number of transcripts.
        """
        return self.transcripts_metadata["n_rows"]

    @cached_property
    def extents(self) -> shapely.Polygon:
        """
        The combined extents (bounding box) of the transcripts and boundaries.

        Returns
        -------
        shapely.Polygon
            The bounding box covering all transcripts and boundaries.
        """
        if self._extents is None:
            # Get individual extents
            xy = self.settings.transcripts.xy
            tx_extents = utils.get_xy_extents(self._transcripts_filepath, *xy)
            xy = self.settings.boundaries.xy
            bd_extents = utils.get_xy_extents(self._boundaries_filepath, *xy)

            # Combine extents and get bounding box
            extents = tx_extents.union(bd_extents)
            self._extents = shapely.box(*extents.bounds)

        return self._extents

    def _get_balanced_regions(
        self,
    ) -> List[shapely.Polygon]:
        """
        Splits the sample extents into balanced regions for parallel processing.
        See NDTree documentation for more information.

        Returns
        -------
        List[shapely.Polygon]
            A list of polygons representing the regions.
        """
        # If no. workers is 1, return full extents
        if self.n_workers == 1:
            return [self.extents]

        # Otherwise, split based on boundary distribution which is much smaller
        # than transcripts DataFrame.
        # Note: Assumes boundaries are distributed similarly to transcripts at
        # a coarse level.
        data = pd.read_parquet(
            self._boundaries_filepath,
            columns=self.settings.boundaries.xy,
        ).values
        ndtree = NDTree(data, self.n_workers)

        return ndtree.boxes

    @staticmethod
    def _setup_directory(
        data_dir: os.PathLike,
    ):
        """
        Sets up the directory structure for saving processed tiles.

        Ensures that the necessary subdirectories for 'train', 'test', and
        'val' are created under the provided base directory. If any of these
        subdirectories already exist and are not empty, an error is raised.

        Directory structure created:
        ----------------------------
        data_dir/
            ├── train_tiles/
            │   └── processed/
            ├── test_tiles/
            │   └── processed/
            └── val_tiles/
                └── processed/

        Parameters
        ----------
        data_dir : os.PathLike
            The path to the base directory where the data should be stored.

        Raises
        ------
        AssertionError
            If any of the 'processed' directories already contain files.
        """
        data_dir = Path(data_dir)  # by default, convert to Path object
        for tile_type in ["train_tiles", "test_tiles", "val_tiles"]:
            for stage in ["raw", "processed"]:
                tile_dir = data_dir / tile_type / stage
                tile_dir.mkdir(parents=True, exist_ok=True)
                if os.listdir(tile_dir):
                    msg = f"Directory '{tile_dir}' must be empty."
                    raise AssertionError(msg)

    def set_transcript_embedding(self, weights: pd.DataFrame):
        """
        Sets the transcript embedding for the sample.

        Parameters
        ----------
        weights : pd.DataFrame
            A DataFrame containing the weights for each transcript.

        Raises
        ------
        ValueError
            If the provided weights do not match the number of transcript
            features.
        """
        classes = self._transcripts_metadata["feature_names"]
        self._transcript_embedding = TranscriptEmbedding(classes, weights)

    def save(
        self,
        data_dir: os.PathLike,
        k_bd: int = 3,
        dist_bd: float = 15.0,
        k_tx: int = 3,
        dist_tx: float = 5.0,
        tile_size: Optional[int] = None,
        tile_width: Optional[float] = None,
        tile_height: Optional[float] = None,
        neg_sampling_ratio: float = 5.0,
        frac: float = 1.0,
        val_prob: float = 0.1,
        test_prob: float = 0.2,
    ):
        """
        Saves the tiles of the sample as PyTorch geometric datasets. See
        documentation for 'STTile' for more information on dataset contents.

        Note: This function requires either 'tile_size' OR both 'tile_width' and
        'tile_height' to be provided.

        Parameters
        ----------
        data_dir : os.PathLike
            The directory where the dataset should be saved.
        k_bd : int, optional, default 3
            Number of nearest neighbors for boundary nodes.
        dist_bd : float, optional, default 15.0
            Maximum distance for boundary neighbors.
        k_tx : int, optional, default 3
            Number of nearest neighbors for transcript nodes.
        dist_tx : float, optional, default 5.0
            Maximum distance for transcript neighbors.
        tile_size : int, optional
            If provided, specifies the size of the tile. Overrides `tile_width`
            and `tile_height`.
        tile_width : int, optional
            Width of the tiles in pixels. Ignored if `tile_size` is provided.
        tile_height : int, optional
            Height of the tiles in pixels. Ignored if `tile_size` is provided.
        neg_sampling_ratio : float, optional, default 5.0
            Ratio of negative samples.
        frac : float, optional, default 1.0
            Fraction of the dataset to process.
        val_prob: float, optional, default 0.1
            Proportion of data for use for validation split.
        test_prob: float, optional, default 0.2
            Proportion of data for use for test split.

        Raises
        ------
        ValueError
            If the 'frac' parameter is greater than 1.0 or if the calculated
            number of tiles is zero.
        AssertionError
            If the specified directory structure is not properly set up.
        """
        # Check inputs
        try:
            if frac > 1:
                msg = f"Arg 'frac' should be <= 1.0, but got {frac}."
                raise ValueError(msg)
            if tile_size is not None:
                n_tiles = self.n_transcripts / tile_size / self.n_workers * frac
                if int(n_tiles) == 0:
                    msg = f"Sampling parameters would yield 0 total tiles."
                    raise ValueError(msg)
        # Propagate errors to logging
        except Exception as e:
            self.logger.error(str(e), exc_info=True)
            raise e

        # Setup directory structure to save tiles
        data_dir = Path(data_dir)
        STSampleParquet._setup_directory(data_dir)

        # Function to parallelize over workers
        def func(region):
            xm = STInMemoryDataset(sample=self, extents=region)
            tiles = xm._tile(tile_width, tile_height, tile_size)
            if frac < 1:
                tiles = random.sample(tiles, int(len(tiles) * frac))
            for tile in tiles:
                # Choose training, test, or validation datasets
                data_type = np.random.choice(
                    a=["train_tiles", "test_tiles", "val_tiles"],
                    p=[1 - (test_prob + val_prob), test_prob, val_prob],
                )
                xt = STTile(dataset=xm, extents=tile)
                pyg_data = xt.to_pyg_dataset(
                    k_bd=k_bd,
                    dist_bd=dist_bd,
                    k_tx=k_tx,
                    dist_tx=dist_tx,
                    neg_sampling_ratio=neg_sampling_ratio,
                )
                if pyg_data is not None:
                    if pyg_data["tx", "belongs", "bd"].edge_index.numel() == 0:
                        # this tile is only for testing
                        data_type = "test_tiles"
                    filepath = data_dir / data_type / "processed" / f"{xt.uid}.pt"
                    torch.save(pyg_data, filepath)

        # TODO: Add Dask backend
        regions = self._get_balanced_regions()
        pqdm(regions, func, n_jobs=self.n_workers)


# TODO: Add documentation for settings
class STInMemoryDataset:
    """
    A class for handling in-memory representations of ST data.

    This class is used to load and manage ST sample data from parquet files,
    filter boundaries and transcripts, and provide spatial tiling for further
    analysis. The class also pre-loads KDTrees for efficient spatial queries.

    Parameters
    ----------
    sample : STSampleParquet
        The ST sample containing paths to the data files.
    extents : shapely.Polygon
        The polygon defining the spatial extents for the dataset.
    margin : int, optional, default 10
        The margin to buffer around the extents when filtering data.

    Attributes
    ----------
    sample : STSampleParquet
        The ST sample from which the data is loaded.
    extents : shapely.Polygon
        The spatial extents of the dataset.
    margin : int
        The buffer margin around the extents for filtering.
    transcripts : pd.DataFrame
        The filtered transcripts within the dataset extents.
    boundaries : pd.DataFrame
        The filtered boundaries within the dataset extents.
    kdtree_tx : KDTree
        The KDTree for fast spatial queries on the transcripts.

    Raises
    ------
    ValueError
        If the transcripts or boundaries could not be loaded or filtered.
    """

    def __init__(
        self,
        sample: STSampleParquet,
        extents: shapely.Polygon,
        margin: int = 10,
    ):
        """
        Initializes the STInMemoryDataset instance by loading transcripts
        and boundaries from parquet files and pre-loading a KDTree for fast
        spatial queries.

        Parameters
        ----------
        sample : STSampleParquet
            The ST sample containing paths to the data files.
        extents : shapely.Polygon
            The polygon defining the spatial extents for the dataset.
        margin : int, optional, default 10
            The margin to buffer around the extents when filtering data.
        """
        # Set properties
        self.sample = sample
        self.extents = extents
        self.margin = margin
        self.settings = self.sample.settings

        # Load data from parquet files
        self._load_transcripts(self.sample._transcripts_filepath)
        self._load_boundaries(self.sample._boundaries_filepath)

        # Pre-load KDTrees
        self.kdtree_tx = KDTree(self.transcripts[self.settings.transcripts.xy], leafsize=100)

    def _load_transcripts(self, path: os.PathLike, min_qv: float = 30.0):
        """
        Loads and filters the transcripts dataframe for the dataset.

        Parameters
        ----------
        path : os.PathLike
            The file path to the transcripts parquet file.
        min_qv : float, optional, default 30.0
            The minimum quality value (QV) for filtering transcripts.

        Raises
        ------
        ValueError
            If the transcripts dataframe cannot be loaded or filtered.
        """
        # Load and filter transcripts dataframe
        bounds = self.extents.buffer(self.margin, join_style="mitre")
        transcripts = utils.read_parquet_region(
            path,
            x=self.settings.transcripts.x,
            y=self.settings.transcripts.y,
            bounds=bounds,
            extra_columns=self.settings.transcripts.columns,
        )
        transcripts[self.settings.transcripts.label] = transcripts[self.settings.transcripts.label].apply(
            lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
        )
        transcripts = utils.filter_transcripts(
            transcripts,
            self.settings.transcripts.label,
            self.settings.transcripts.filter_substrings,
            min_qv,
        )

        # Only set object properties once everything finishes successfully
        self.transcripts = transcripts

    def _load_boundaries(self, path: os.PathLike):
        """
        Loads and filters the boundaries dataframe for the dataset.

        Parameters
        ----------
        path : os.PathLike
            The file path to the boundaries parquet file.

        Raises
        ------
        ValueError
            If the boundaries dataframe cannot be loaded or filtered.
        """
        # Load and filter boundaries dataframe
        outset = self.extents.buffer(self.margin, join_style="mitre")
        boundaries = utils.read_parquet_region(
            path,
            x=self.settings.boundaries.x,
            y=self.settings.boundaries.y,
            bounds=outset,
            extra_columns=self.settings.boundaries.columns,
        )
        boundaries = utils.filter_boundaries(
            boundaries,
            inset=self.extents,
            outset=outset,
            x=self.settings.boundaries.x,
            y=self.settings.boundaries.y,
            label=self.settings.boundaries.label,
        )
        self.boundaries = boundaries

    def _get_rectangular_tile_bounds(
        self,
        tile_width: float,
        tile_height: float,
    ) -> List[shapely.Polygon]:
        """
        Generates rectangular tiles for the dataset based on the extents.

        Parameters
        ----------
        tile_width : float
            The width of each tile.
        tile_height : float
            The height of each tile.

        Returns
        -------
        List[shapely.Polygon]
            A list of polygons representing the rectangular tiles.
        """
        # Generate the x and y coordinates for the tile boundaries
        x_min, y_min, x_max, y_max = self.extents.bounds
        x_coords = np.arange(x_min, x_max, tile_width)
        x_coords = np.append(x_coords, x_max)
        y_coords = np.arange(y_min, y_max, tile_height)
        y_coords = np.append(y_coords, y_max)

        # Generate tiles from grid points
        tiles = []
        for x_min, x_max in zip(x_coords[:-1], x_coords[1:]):
            for y_min, y_max in zip(y_coords[:-1], y_coords[1:]):
                tiles.append(shapely.box(x_min, y_min, x_max, y_max))

        return tiles

    def _get_balanced_tile_bounds(
        self,
        max_size: Optional[int],
    ) -> List[shapely.Polygon]:
        """
        Generates spatially balanced tiles based on KDTree partitioning.

        Parameters
        ----------
        max_size : Optional[int]
            The maximum number of points in each tile.

        Returns
        -------
        List[shapely.Polygon]
            A list of polygons representing balanced tile bounds.

        Raises
        ------
        ValueError
            If `max_size` is smaller than the KDTree's leaf size.
        """
        # Can only request up to brute force resolution of KDTree
        leafsize = self.kdtree_tx.leafsize
        if max_size < leafsize:
            msg = f"Arg 'max_size' less than KDTree 'leafsize', {leafsize}."
            raise ValueError(msg)

        # DFS search to construct tile bounds
        def recurse(node, bounds):
            if node.children <= max_size:
                bounds = shapely.box(*bounds.mins, *bounds.maxes)
                return [bounds]
            lb, gb = bounds.split(node.split_dim, node.split)
            return recurse(node.less, lb) + recurse(node.greater, gb)

        node = self.kdtree_tx.tree
        bounds = Rectangle(self.kdtree_tx.mins, self.kdtree_tx.maxes)
        return recurse(node, bounds)

    def _tile(
        self,
        width: Optional[float] = None,
        height: Optional[float] = None,
        max_size: Optional[int] = None,
    ) -> List[shapely.Polygon]:
        """
        Generates tiles based on either fixed dimensions or balanced
        partitioning.

        Parameters
        ----------
        width : Optional[float]
            The width of each tile. Required if `max_size` is not provided.
        height : Optional[float]
            The height of each tile. Required if `max_size` is not provided.
        max_size : Optional[int]
            The maximum number of points in each tile. Required if `width` and
            `height` are not provided.

        Returns
        -------
        List[shapely.Polygon]
            A list of polygons representing the tiles.

        Raises
        ------
        ValueError
            If both `width`/`height` and `max_size` are provided or none are
            provided.
        """
        # Square tiling kwargs provided
        if not max_size and (width and height):
            return self._get_rectangular_tile_bounds(width, height)
        # Balanced tiling kwargs provided or None
        elif not (width or height):
            return self._get_balanced_tile_bounds(max_size)
        # Bad set of kwargs
        else:
            args = list(compress(locals().keys(), locals().values()))
            args.remove("self")
            msg = "Function requires either 'max_size' or both " f"'width' and 'height'. Found: {', '.join(args)}."
            logging.error(msg)
            raise ValueError


# TODO: Add documentation for settings
class STTile:
    """
    A class representing a tile of a ST sample.

    Attributes
    ----------
    dataset : STInMemoryDataset
        The ST dataset containing data.
    extents : shapely.Polygon
        The extents of the tile in the sample.
    boundaries : pd.DataFrame
        Filtered boundaries within the tile extents.
    transcripts : pd.DataFrame
        Filtered transcripts within the tile extents.
    """

    def __init__(
        self,
        dataset: STInMemoryDataset,
        extents: shapely.Polygon,
    ):
        """
        Initializes a STTile instance.

        Parameters
        ----------
        dataset : STInMemoryDataset
            The ST dataset containing data.
        extents : shapely.Polygon
            The extents of the tile in the sample.

        Notes
        -----
        The `boundaries` and `transcripts` attributes are cached to avoid the
        overhead of filtering when tiles are instantiated. This is particularly
        useful in multiprocessing settings where generating tiles in parallel
        could lead to high overhead.

        Internal Attributes
        --------------------
        _boundaries : pd.DataFrame, optional
            Cached DataFrame of filtered boundaries. Initially set to None.
        _transcripts : pd.DataFrame, optional
            Cached DataFrame of filtered transcripts. Initially set to None.
        """
        self.dataset = dataset
        self.extents = extents
        self.margin = dataset.margin
        self.settings = self.dataset.settings

        # Internal caches for filtered data
        self._boundaries = None
        self._transcripts = None

    @property
    def uid(self) -> str:
        """
        Generates a unique identifier for the tile based on its extents. This
        UID is particularly useful for saving or indexing tiles in distributed
        processing environments.

        The UID is constructed using the minimum and maximum x and y coordinates
        of the tile's bounding box, representing its position and size in the
        sample.

        Returns
        -------
        str
            A unique identifier string in the format
            'x=<x_min>_y=<y_min>_w=<width>_h=<height>' where:
            - `<x_min>`: Minimum x-coordinate of the tile's extents.
            - `<y_min>`: Minimum y-coordinate of the tile's extents.
            - `<width>`: Width of the tile.
            - `<height>`: Height of the tile.

        Example
        -------
        If the tile's extents are bounded by (x_min, y_min) = (100, 200) and
        (x_max, y_max) = (150, 250), the generated UID would be:
        'x=100_y=200_w=50_h=50'
        """
        x_min, y_min, x_max, y_max = map(int, self.extents.bounds)
        uid = f"tiles_x={x_min}_y={y_min}_w={x_max-x_min}_h={y_max-y_min}"
        return uid

    @cached_property
    def boundaries(self) -> pd.DataFrame:
        """
        Returns the filtered boundaries within the tile extents, cached for
        efficiency.

        The boundaries are computed only once and cached. If the boundaries
        have not been computed yet, they are computed using
        `get_filtered_boundaries()`.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the filtered boundaries within the tile
            extents.
        """
        if self._boundaries is None:
            self._boundaries = self.get_filtered_boundaries()
        return self._boundaries

    @cached_property
    def transcripts(self) -> pd.DataFrame:
        """
        Returns the filtered transcripts within the tile extents, cached for
        efficiency.

        The transcripts are computed only once and cached. If the transcripts
        have not been computed yet, they are computed using
        `get_filtered_transcripts()`.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the filtered transcripts within the tile
            extents.
        """
        if self._transcripts is None:
            self._transcripts = self.get_filtered_transcripts()
        return self._transcripts

    def get_filtered_boundaries(self) -> pd.DataFrame:
        """
        Filters the boundaries in the sample to include only those within
        the specified tile extents.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the filtered boundaries within the tile
            extents.
        """
        filtered_boundaries = utils.filter_boundaries(
            boundaries=self.dataset.boundaries,
            inset=self.extents,
            outset=self.extents.buffer(self.margin, join_style="mitre"),
            x=self.settings.boundaries.x,
            y=self.settings.boundaries.y,
            label=self.settings.boundaries.label,
        )
        return filtered_boundaries

    def get_filtered_transcripts(self) -> pd.DataFrame:
        """
        Filters the transcripts in the sample to include only those within
        the specified tile extents.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the filtered transcripts within the tile
            extents.
        """

        # Buffer tile bounds to include transcripts around boundary
        outset = self.extents.buffer(self.margin, join_style="mitre")
        xmin, ymin, xmax, ymax = outset.bounds

        # Get transcripts inside buffered region
        x, y = self.settings.transcripts.xy
        mask = self.dataset.transcripts[x].between(xmin, xmax)
        mask &= self.dataset.transcripts[y].between(ymin, ymax)
        filtered_transcripts = self.dataset.transcripts[mask]

        return filtered_transcripts

    def get_transcript_props(self) -> torch.Tensor:
        """
        Encodes transcript features in a sparse format.

        Returns
        -------
        props : torch.Tensor
            A sparse tensor containing the encoded transcript features.

        Notes
        -----
        The intention is for this function to simplify testing new strategies
        for 'tx' node representations. For example, the encoder can be any type
        of encoder that transforms the transcript labels into a numerical
        matrix (in sparse format).
        """
        # Encode transcript features in sparse format
        embedding = self.dataset.sample._transcript_embedding
        label = self.settings.transcripts.label
        props = embedding.embed(self.transcripts[label])

        return props

    @staticmethod
    def get_polygon_props(
        polygons: gpd.GeoSeries,
        area: bool = True,
        convexity: bool = True,
        elongation: bool = True,
        circularity: bool = True,
    ) -> pd.DataFrame:
        """
        Computes geometric properties of polygons.

        Parameters
        ----------
        polygons : gpd.GeoSeries
            A GeoSeries containing polygon geometries.
        area : bool, optional
            If True, compute the area of each polygon (default is True).
        convexity : bool, optional
            If True, compute the convexity of each polygon (default is True).
        elongation : bool, optional
            If True, compute the elongation of each polygon (default is True).
        circularity : bool, optional
            If True, compute the circularity of each polygon (default is True).

        Returns
        -------
        props : pd.DataFrame
            A DataFrame containing the computed properties for each polygon.
        """
        props = pd.DataFrame(index=polygons.index, dtype=float)
        if area:
            props["area"] = polygons.area
        if convexity:
            props["convexity"] = polygons.convex_hull.area / polygons.area
        if elongation:
            rects = polygons.minimum_rotated_rectangle()
            props["elongation"] = rects.area / polygons.envelope.area
        if circularity:
            r = polygons.minimum_bounding_radius()
            props["circularity"] = polygons.area / r**2

        return props

    @staticmethod
    def get_kdtree_edge_index(
        index_coords: np.ndarray,
        query_coords: np.ndarray,
        k: int,
        max_distance: float,
    ):
        """
        Computes the k-nearest neighbor edge indices using a KDTree.

        Parameters
        ----------
        index_coords : np.ndarray
            An array of shape (n_samples, n_features) representing the
            coordinates of the points to be indexed.
        query_coords : np.ndarray
            An array of shape (m_samples, n_features) representing the
            coordinates of the query points.
        k : int
            The number of nearest neighbors to find for each query point.
        max_distance : float
            The maximum distance to consider for neighbors.

        Returns
        -------
        torch.Tensor
            An array of shape (2, n_edges) containing the edge indices. Each
            column represents an edge between two points, where the first row
            contains the source indices and the second row contains the target
            indices.
        """
        # KDTree search
        tree = KDTree(index_coords)
        dist, idx = tree.query(query_coords, k, max_distance)

        # To sparse adjacency
        edge_index = np.argwhere(dist != np.inf).T
        edge_index[1] = idx[dist != np.inf]
        edge_index = torch.tensor(edge_index, dtype=torch.long).contiguous()

        return edge_index

    def get_boundary_props(
        self,
        area: bool = True,
        convexity: bool = True,
        elongation: bool = True,
        circularity: bool = True,
    ) -> torch.Tensor:
        """
        Computes geometric properties of boundary polygons.

        Parameters
        ----------
        area : bool, optional
            If True, compute the area of each boundary polygon (default is
            True).
        convexity : bool, optional
            If True, compute the convexity of each boundary polygon (default is
            True).
        elongation : bool, optional
            If True, compute the elongation of each boundary polygon (default is
              True).
        circularity : bool, optional
            If True, compute the circularity of each boundary polygon (default
            is True).

        Returns
        -------
        torch.Tensor
            A tensor containing the computed properties for each boundary
            polygon.

        Notes
        -----
        The intention is for this function to simplify testing new strategies
        for 'bd' node representations. You can just change the function body to
        return another torch.Tensor without worrying about changes to the rest
        of the code.
        """
        # Get polygons from coordinates
        polygons = utils.get_polygons_from_xy(
            self.boundaries,
            x=self.settings.boundaries.x,
            y=self.settings.boundaries.y,
            label=self.settings.boundaries.label,
        )
        # Geometric properties of polygons
        props = self.get_polygon_props(polygons)
        props = torch.as_tensor(props.values).float()

        return props

    def to_pyg_dataset(
        self,
        # train: bool,
        neg_sampling_ratio: float = 5,
        k_bd: int = 3,
        dist_bd: float = 15,
        k_tx: int = 3,
        dist_tx: float = 5,
        area: bool = True,
        convexity: bool = True,
        elongation: bool = True,
        circularity: bool = True,
    ) -> HeteroData:
        """
        Converts the sample data to a PyG HeteroData object (more information
        on the structure of the object below).

        Parameters
        ----------
        train: bool
            Whether a sample is part of the training dataset. If True, add
            negative edges to dataset.
        k_bd : int, optional
            The number of nearest neighbors for the 'bd' nodes (default is 4).
        dist_bd : float, optional
            The maximum distance for neighbors of 'bd' nodes (default is 20).
        k_tx : int, optional
            The number of nearest neighbors for the 'tx' nodes (default is 4).
        dist_tx : float, optional
            The maximum distance for neighbors of 'tx' nodes (default is 20).
        area : bool, optional
            If True, compute the area of each polygon (default is True).
        convexity : bool, optional
            If True, compute the convexity of each polygon (default is True).
        elongation : bool, optional
            If True, compute the elongation of each polygon (default is True).
        circularity : bool, optional
            If True, compute the circularity of each polygon (default is True).

        Returns
        -------
        data : HeteroData
            A PyG HeteroData object containing the sample data.

        Segger PyG HeteroData Spec
        --------------------------
        A heterogenous graph with two node types and two edge types.

        Node Types
        ----------
        1. Boundary ("bd")
            Represents boundaries (typically cells) in the ST dataset.

            Attributes
            ----------
            id : str
                Cell ID originating from the ST sample.
            pos : np.ndarray
                X, Y coordinates of the centroid of the polygon boundary.
            x : torch.tensor
                May include area, convexity, elongation, and circularity
                of the polygon boundary (user-specified).

        2. Transcript ("tx")
            Represents transcripts in the ST dataset.

            Attributes
            ----------
            id : int64
                Transcript ID originating from ST sample.
            pos : np.ndarray
                X, Y, Z coordinates of the transcript.
            x : torch.tensor
                Sparse one-hot encoding of the transcript gene name.

        Edge Types
        ----------
        1. ("tx", "belongs", "bd")
            Represents the relationship where a transcript is contained within
            a boundary.

            Attributes
            ----------
            edge_index : torch.Tensor
                Edge indices in COO format between transcripts and nuclei

        2. ("tx", "neighbors", "bd")
            Represents the relationship where a transcript is nearby but not
            within a boundary.

            Attributes
            ----------
            edge_index : torch.Tensor
                Edge indices in COO format between transcripts and boundaries

        3. ("tx", "neighbors", "tx")
            Represents the relationship where a transcript is nearby another
            transcript.

            Attributes
            ----------
            edge_index : torch.Tensor
                Edge indices in COO format between transcripts and transcripts
        """
        # Initialize an empty HeteroData object
        pyg_data = HeteroData()

        # Set up Transcript nodes
        pyg_data["tx"].id = torch.tensor(
            self.transcripts[self.settings.transcripts.id].values.astype(int),
            dtype=torch.long,
        )
        pyg_data["tx"].pos = torch.tensor(
            self.transcripts[self.settings.transcripts.xyz].values,
            dtype=torch.float32,
        )
        pyg_data["tx"].x = self.get_transcript_props()

        # Set up Transcript-Transcript neighbor edges
        nbrs_edge_idx = self.get_kdtree_edge_index(
            self.transcripts[self.settings.transcripts.xy],
            self.transcripts[self.settings.transcripts.xy],
            k=k_tx,
            max_distance=dist_tx,
        )

        # If there are no tx-neighbors-tx edges, skip saving tile
        if nbrs_edge_idx.shape[1] == 0:
            logging.warning(f"No tx-neighbors-tx edges found in tile {self.uid}.")
            return None

        pyg_data["tx", "neighbors", "tx"].edge_index = nbrs_edge_idx

        # Set up Boundary nodes
        polygons = utils.get_polygons_from_xy(
            self.boundaries,
            self.settings.boundaries.x,
            self.settings.boundaries.y,
            self.settings.boundaries.label,
        )
        centroids = polygons.centroid.get_coordinates()
        pyg_data["bd"].id = polygons.index.to_numpy()
        pyg_data["bd"].pos = torch.tensor(centroids.values, dtype=torch.float32)
        pyg_data["bd"].x = self.get_boundary_props(area, convexity, elongation, circularity)

        # Set up Boundary-Transcript neighbor edges
        dist = np.sqrt(polygons.area.max()) * 10  # heuristic distance
        nbrs_edge_idx = self.get_kdtree_edge_index(
            centroids,
            self.transcripts[self.settings.transcripts.xy],
            k=k_bd,
            max_distance=dist,
        )
        pyg_data["tx", "neighbors", "bd"].edge_index = nbrs_edge_idx

        # If there are no tx-neighbors-bd edges, we put the tile automatically in test set
        if nbrs_edge_idx.numel() == 0:
            logging.warning(f"No tx-neighbors-bd edges found in tile {self.uid}.")
            pyg_data["tx", "belongs", "bd"].edge_index = torch.tensor([], dtype=torch.long)
            return pyg_data

        # Now we identify and split the tx-belongs-bd edges
        edge_type = ("tx", "belongs", "bd")

        # Find nuclear transcripts
        tx_cell_ids = self.transcripts[self.settings.boundaries.id]
        cell_ids_map = {idx: i for (i, idx) in enumerate(polygons.index)}
        is_nuclear = self.transcripts[self.settings.transcripts.nuclear].astype(bool)
        is_nuclear &= tx_cell_ids.isin(polygons.index)

        # Set up overlap edges
        row_idx = np.where(is_nuclear)[0]
        col_idx = tx_cell_ids.iloc[row_idx].map(cell_ids_map)
        blng_edge_idx = torch.tensor(np.stack([row_idx, col_idx])).long()
        pyg_data[edge_type].edge_index = blng_edge_idx

        # If there are no tx-belongs-bd edges, flag tile as test only (cannot be used for training)
        if blng_edge_idx.numel() == 0:
            logging.warning(f"No tx-belongs-bd edges found in tile {self.uid}.")
            return pyg_data

        # If there are tx-bd edges, add negative edges for training
        # Need more time-efficient solution than this
        transform = RandomLinkSplit(
            num_val=0,
            num_test=0,
            is_undirected=True,
            edge_types=[edge_type],
            neg_sampling_ratio=neg_sampling_ratio,
        )
        pyg_data, _, _ = transform(pyg_data)

        # Refilter negative edges to include only transcripts in the
        # original positive edges (still need a memory-efficient solution)
        edges = pyg_data[edge_type]
        mask = edges.edge_label_index[0].unsqueeze(1) == edges.edge_index[0].unsqueeze(0)
        mask = torch.nonzero(torch.any(mask, 1)).squeeze()
        edges.edge_label_index = edges.edge_label_index[:, mask]
        edges.edge_label = edges.edge_label[mask]

        return pyg_data
