import os
import sys
import gc
import shapely
from pyarrow import parquet as pq
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
import yaml
import geopandas as gpd
from segger.data.xenium_utils import *
from scipy.spatial import KDTree, Rectangle
from segger.data._ndtree import NDTree
from multiprocessing import Pool
from functools import cached_property
from typing import List, Tuple, Optional, Callable, TYPE_CHECKING
import inspect
import logging
import json
from itertools import compress
import psutil
from pqdm.processes import pqdm
from itertools import repeat
from joblib import Parallel, delayed
import glob
from torch_geometric.data import HeteroData, InMemoryDataset, Data
from torch_geometric.transforms import RandomLinkSplit
import torch
from pqdm.threads import pqdm
import random


# TODO: Add documentation
class XeniumFilename:
    transcripts = "transcripts.parquet"
    boundaries = "nucleus_boundaries.parquet"


class XeniumSampleParquet:
    # TODO: Add documentation

    def __init__(
        self,
        base_dir: os.PathLike,
        n_workers: Optional[int] = 1,
    ):
        # Setup paths and resource constraints
        self._base_dir = Path(base_dir)  # TODO: check that directory is valid
        self._transcripts_filepath = base_dir / XeniumFilename.transcripts
        self._boundaries_filepath = base_dir / XeniumFilename.boundaries
        self.n_workers = n_workers

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.Logger(f'XeniumSample@{base_dir}')

        # Internal caches
        self._extents = None
        self._transcripts_metadata = None
        self._boundaries_metadata = None


    # TODO: Add documentation
    @classmethod
    def _get_parquet_metadata(
        cls,
        filepath: os.PathLike,
        columns: Optional[List[str]] = None,
    ) -> dict:
        # Size in bytes of field dtypes
        size_map = {
            'BOOLEAN': 1, 
            'INT32': 4,
            'FLOAT': 4,
            'INT64': 8,
            'DOUBLE': 8,
            'BYTE_ARRAY': 8,
            'INT96': 12,
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
        summary['n_rows'] = metadata.num_rows
        summary['n_columns'] = len(columns)
        summary['column_sizes'] = dict()
        for c in columns:
            # Error where 10X saved BOOLEAN field as INT32 in schema
            if c == 'overlaps_nucleus':
                dtype = 'BOOLEAN'
            else:
                i = metadata.schema.names.index(c)
                dtype = metadata.schema[i].physical_type
            summary['column_sizes'][c] = size_map[dtype]

        return summary


    # TODO: Add documentation
    @cached_property
    def transcripts_metadata(self) -> dict:
        if self._transcripts_metadata is None:
            # Base metadata
            metadata = XeniumSampleParquet._get_parquet_metadata(
                self._transcripts_filepath,
                TranscriptColumns.columns,
            )
            # Gene panel names
            with open(self._base_dir / 'gene_panel.json', 'r') as file:
                targets = json.load(file)['payload']['targets']
            filter_names = tuple(e.value for e in XeniumFilterCodewords)
            names = [t['type']['data']['name'] for t in targets]
            names = filter(lambda n: not n.startswith(filter_names), names)
            metadata['feature_names'] = list(names)
            self._transcripts_metadata = metadata

        return self._transcripts_metadata


    # TODO: Add documentation
    @cached_property
    def boundaries_metadata(self) -> dict:
        if self._boundaries_metadata is None:
            metadata = XeniumSampleParquet._get_parquet_metadata(
                self._boundaries_filepath,
                BoundaryColumns.columns,
            )
            self._boundaries_metadata = metadata
        return self._boundaries_metadata

    # TODO: Add documentation
    @property
    def n_transcripts(self) -> int:
        return self.transcripts_metadata['n_rows']


    # TODO: Add documentation
    @cached_property
    def extents(self):
        if self._extents is None:
            # Get individual extents
            xy = TranscriptColumns.xy
            tx_extents = get_xy_extents(self._transcripts_filepath, *xy)
            xy = BoundaryColumns.xy
            bd_extents = get_xy_extents(self._boundaries_filepath, *xy)

            # Combine extents and get bounding box
            extents = tx_extents.union(bd_extents)
            self._extents = shapely.box(*extents.bounds)

        return self._extents


    # TODO: Add documentation
    def _get_balanced_regions(
        self,
    ) -> List[shapely.Polygon]:

        # If no. workers is 1, return full extents
        if self.n_workers == 1:
            return [self.extents]
        
        # Otherwise, split based on boundary distribution which is much smaller
        # than transcripts DataFrame.
        # Note: Assumes boundaries are distributed similarly to transcripts at 
        # a coarse level.
        data = pd.read_parquet(
            self._boundaries_filepath,
            columns=BoundaryColumns.xy,
        ).values
        ndtree = NDTree(data, self.n_workers)

        return ndtree.boxes


    # TODO: Add documentation
    @staticmethod
    def _setup_directory(
        data_dir: os.PathLike,
    ):
        data_dir = Path(data_dir)  # by default, convert to Path object
        for dt in ['train', 'test', 'val']:
            tile_dir = data_dir / dt / 'processed'
            tile_dir.mkdir(parents=True, exist_ok=True)
            if os.listdir(tile_dir):
                msg = f"Directory '{tile_dir}' must be empty."
                raise AssertionError(msg)


    # TODO: Add documentation
    def save(
        self,
        data_dir: os.PathLike,
        max_size: int = 1e5,
        k_bd: int = 3,
        dist_bd: float = 15.,
        k_tx: int = 3,
        dist_tx: float = 5.,
        neg_sampling_ratio: float = 5.,
        frac: float = 1.,
    ):
        # Check inputs
        try:
            if frac > 1:
                msg = f"Arg 'frac' should be <= 1.0, but got {frac}."
                raise ValueError(msg)
            n_tiles = self.n_transcripts / max_size / self.n_workers * frac
            if int(n_tiles) == 0:
                msg = f"Sampling parameters would yield 0 total tiles."
                raise ValueError(msg)
        # Propagate errors to logging
        except Exception as e:
            self.logger.error(str(e), exc_info=True)
            raise e

        # Setup directory structure to save tiles
        data_dir = Path(data_dir)
        XeniumSampleParquet._setup_directory(data_dir)

        # Function to parallelize over workers
        def func(region):
            xm = XeniumInMemoryDataset(sample=self, extents=region)
            tiles = xm._tile(max_size=max_size)
            if frac < 1:
                tiles = random.sample(tiles, int(len(tiles) * frac))
            for tile in tiles:
                # Choose training, test, or validation datasets
                data_type = np.random.choice(
                    a=['train', 'test', 'val'],
                    p=[0.7, 0.2, 0.1],  # hard-coded for now
                )
                xt = XeniumTile(dataset=xm, extents=tile)
                pyg_data = xt.to_pyg_dataset(
                    k_bd=k_bd,
                    dist_bd=dist_bd,
                    k_tx=k_tx,
                    dist_tx=dist_tx,
                    neg_sampling_ratio=neg_sampling_ratio,
                )
                filepath = data_dir / data_type / 'processed' / f'{xt.uid}.pt'
                torch.save(pyg_data, filepath)

        # TODO: Add Dask backend
        regions = self._get_balanced_regions()
        outs = pqdm(regions, func, n_jobs=self.n_workers)


class XeniumInMemoryDataset():
    # TODO: Add documentation

    def __init__(
        self,
        sample: XeniumSampleParquet,
        extents: shapely.Polygon,
        margin: int = 10,
    ):
        # Set properties
        self.sample = sample
        self.extents = extents
        self.margin = margin

        # Load data from parquet files
        self._load_transcripts(sample._transcripts_filepath)
        self._load_boundaries(sample._boundaries_filepath)

        # Pre-load KDTrees
        enums = TranscriptColumns
        self.kdtree_tx = KDTree(self.transcripts[enums.xy], leafsize=100)


    # TODO: Add documentation
    def _load_transcripts(self, path: os.PathLike, min_qv: float = 30.0):
        # Load and filter transcripts dataframe
        enums = TranscriptColumns
        bounds = self.extents.buffer(self.margin, join_style='mitre')
        transcripts = read_parquet_region(
            path,
            x=enums.x,
            y=enums.y,
            bounds=bounds,
            extra_columns=enums.columns,
        )
        transcripts = filter_transcripts(transcripts, min_qv)

        # Transcript latent is one hot encoding of names
        genes = transcripts[[enums.label]]
        categories = self.sample.transcripts_metadata['feature_names']
        encoder = OneHotEncoder(
            categories=[categories],
            sparse_output=False,
        )
        encoder.fit(genes)
        
        # Only set object properties once everything finishes successfully
        self.transcripts = transcripts
        self.tx_encoder = encoder


    # TODO: Add documentation
    def _load_boundaries(self, path: os.PathLike):
        # Load and filter boundaries dataframe
        enums = BoundaryColumns
        outset = self.extents.buffer(self.margin, join_style='mitre')
        boundaries = read_parquet_region(
            path,
            x=enums.x,
            y=enums.y,
            bounds=outset,
            extra_columns=enums.columns
        )
        boundaries = filter_boundaries(boundaries, self.extents, outset)
        self.boundaries = boundaries


    # TODO: Add documentation
    def _get_rectangular_tile_bounds(
        self,
        tile_width: float,
        tile_height: float,
    ) -> List[shapely.Polygon]:
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


    # TODO: Add documentation
    def _get_balanced_tile_bounds(
        self,
        max_size: Optional[int],
    ) -> List[shapely.Polygon]:
        
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


    # TODO: Add documentation
    def _tile(self,
        width: Optional[float] = None,
        height: Optional[float] = None,
        max_size: Optional[int] = None,
     ) -> List[shapely.Polygon]:
        # Square tiling kwargs provided
        if not max_size and (width and height):
            return self._get_rectangular_tile_bounds(width, height)
        # Balanced tiling kwargs provided or None
        elif not (width or height):
            return self._get_balanced_tile_bounds(max_size)
        # Bad set of kwargs
        else:
            args = list(compress(locals().keys(), locals().values()))
            args.remove('self')
            msg = (
                "Function requires either 'max_size' or both "
                f"'width' and 'height'. Found: {', '.join(args)}."
            )
            logging.error(msg)
            raise ValueError


class XeniumTile:
    """
    A class representing a tile of a Xenium sample.

    Attributes
    ----------
    dataset : XeniumInMemoryDataset
        The Xenium dataset containing data.
    extents : shapely.Polygon
        The extents of the tile in the sample.
    boundaries : pd.DataFrame
        Filtered boundaries within the tile extents.
    transcripts : pd.DataFrame
        Filtered transcripts within the tile extents.
    """

    def __init__(
        self,
        dataset: XeniumInMemoryDataset,
        extents: shapely.Polygon,
    ):
        """
        Initializes a XeniumTile instance.

        Parameters
        ----------
        dataset : XeniumInMemoryDataset
            The Xenium dataset containing data.
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

        # Internal caches for filtered data
        self._boundaries = None
        self._transcripts = None


    # TODO: Add documentation
    @property
    def uid(self):
        x_min, y_min, x_max, y_max = map(int, self.extents.bounds)
        uid = f'x={x_min}_y={y_min}_w={x_max-x_min}_h={y_max-y_min}'
        return uid


    @cached_property
    def boundaries(self):
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
    def transcripts(self):
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
        filtered_boundaries = filter_boundaries(
            boundaries=self.dataset.boundaries,
            inset=self.extents,
            outset=self.extents.buffer(self.margin, join_style='mitre')
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
        outset = self.extents.buffer(self.margin, join_style='mitre')
        xmin, ymin, xmax, ymax =  outset.bounds

        # Get transcripts inside buffered region
        enums = TranscriptColumns
        mask = self.dataset.transcripts[enums.x].between(xmin, xmax)
        mask &= self.dataset.transcripts[enums.y].between(ymin, ymax)
        filtered_transcripts = self.dataset.transcripts[mask]

        return filtered_transcripts


    def get_transcript_props(
        self,
    ):
        """
        Encodes transcript features in a sparse format.

        Returns
        -------
        props : torch.sparse.FloatTensor
            A sparse tensor containing the encoded transcript features.

        Notes
        -----
        The intention is for this function to simplify testing new strategies 
        for 'tx' node representations. For example, the encoder can be any type
        of encoder that transforms the transcript labels into a numerical 
        matrix (in sparse format).
        """
        # Encode transcript features in sparse format
        enums = TranscriptColumns
        encoder = self.dataset.tx_encoder  # typically, a one-hot encoder
        encoding = encoder.transform(self.transcripts[[enums.label]])
        props = torch.as_tensor(encoding).float().to_sparse()

        return props


    @staticmethod
    def get_polygon_props(
        polygons: gpd.GeoSeries,
        area: bool = True,
        convexity: bool = True,
        elongation: bool = True,
        circularity: bool = True,
    ):
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
            props['area'] = polygons.area
        if convexity:
            props['convexity'] = polygons.convex_hull.area / polygons.area
        if elongation:
            rects = polygons.minimum_rotated_rectangle()
            props['elongation'] = rects.area / polygons.envelope.area
        if circularity:
            r = polygons.minimum_bounding_radius()
            props["circularity"] = polygons.area / r ** 2
        
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
        polygons = get_polygons_from_xy(self.boundaries)

        # Geometric properties of polygons
        props = self.get_polygon_props(polygons)
        props = torch.as_tensor(props.values).float()

        return props


    def to_pyg_dataset(
        self,
        #train: bool,
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
            Represents boundaries (typically cells) in the Xenium dataset.

            Attributes
            ----------
            id : str
                Cell ID originating from the Xenium sample.
            pos : np.ndarray
                X, Y coordinates of the centroid of the polygon boundary.
            x : torch.tensor
                May include area, convexity, elongation, and circularity
                of the polygon boundary (user-specified).

        2. Transcript ("tx")
            Represents transcripts in the Xenium dataset.

            Attributes
            ----------
            id : int64
                Transcript ID originating from Xenium sample.
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

        # Set up Boundary nodes
        polygons = get_polygons_from_xy(self.boundaries)
        centroids = polygons.centroid.get_coordinates()
        pyg_data['bd'].id = polygons.index.to_numpy().reshape(-1, 1)
        pyg_data['bd'].pos = centroids.values
        pyg_data['bd'].x = self.get_boundary_props(
            area, convexity, elongation, circularity
        )

        # Set up Transcript nodes
        enums = TranscriptColumns
        pyg_data['tx'].id = self.transcripts[[enums.id]].values
        pyg_data['tx'].pos = self.transcripts[enums.xyz].values
        pyg_data['tx'].x = self.get_transcript_props()

        # Set up Boundary-Transcript neighbor edges
        dist = np.sqrt(polygons.area.max()) * 10  # heuristic distance
        nbrs_edge_idx = self.get_kdtree_edge_index(
            centroids,
            self.transcripts[enums.xy],
            k=k_bd,
            max_distance=dist,
        )
        pyg_data["tx", "neighbors", "bd"].edge_index = nbrs_edge_idx

        # Set up Transcript-Transcript neighbor edges
        nbrs_edge_idx = self.get_kdtree_edge_index(
            self.transcripts[enums.xy],
            self.transcripts[enums.xy],
            k=k_tx,
            max_distance=dist_tx,
        )
        pyg_data["tx", "neighbors", "tx"].edge_index = nbrs_edge_idx

        # Find nuclear transcripts
        tx_cell_ids = self.transcripts[BoundaryColumns.id]
        cell_ids_map = {idx: i for (i, idx) in enumerate(polygons.index)}
        is_nuclear = self.transcripts[TranscriptColumns.nuclear].astype(bool) 
        is_nuclear &= tx_cell_ids.isin(polygons.index)

        # Set up overlap edges
        row_idx = np.where(is_nuclear)[0]
        col_idx = tx_cell_ids.iloc[row_idx].map(cell_ids_map)
        blng_edge_idx = torch.tensor(np.stack([row_idx, col_idx])).long()
        pyg_data["tx", "belongs", "bd"].edge_index = blng_edge_idx

        # Add negative edges for training
        # Need more time-efficient solution than this
        edge_type = ('tx', 'belongs', 'bd')
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
        mask = edges.edge_label_index[0].unsqueeze(1) == \
                edges.edge_index[0].unsqueeze(0)
        mask = torch.nonzero(torch.any(mask, 1)).squeeze()
        edges.edge_label_index = edges.edge_label_index[:, mask]
        edges.edge_label = edges.edge_label[mask]

        return pyg_data


class XeniumPyGDataset(InMemoryDataset):
    """
    A dataset class for handling SpatialTranscriptomics spatial transcriptomics 
    data.

    Attributes
    ----------
    root : str
        The root directory where the dataset is stored.
    transform : callable
        A function/transform that takes in a Data object and returns a 
        transformed version.
    pre_transform : callable
        A function/transform that takes in a Data object and returns a 
        transformed version.
    pre_filter : callable
        A function that takes in a Data object and returns a boolean indicating 
        whether to keep it.
    """
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None
    ):
        super().__init__(root, transform, pre_transform, pre_filter)
        os.makedirs(os.path.join(self.processed_dir, 'raw'), exist_ok=True)

    @property
    def raw_file_names(self) -> List[str]:
        """
        Return a list of raw file names in the raw directory.

        Returns:
            List[str]: List of raw file names.
        """
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self) -> List[str]:
        """
        Return a list of processed file names in the processed directory.

        Returns:
            List[str]: List of processed file names.
        """
        paths = glob.glob(f'{self.processed_dir}/*.pt')
        file_names = list(map(os.path.basename, paths))
        return file_names

    def len(self) -> int:
        """
        Return the number of processed files.

        Returns:
            int: Number of processed files.
        """
        return len(self.processed_file_names)

    def get(self, idx: int) -> Data:
        """
        Get a processed data object.

        Args:
            idx (int): Index of the data object to retrieve.

        Returns:
            Data: The processed data object.
        """
        filepath = Path(self.processed_dir) / self.processed_file_names[idx]
        data = torch.load(filepath)
        data['tx'].x = data['tx'].x.to_dense()
        return data