from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData
from sklearn.preprocessing import OneHotEncoder
from scipy.spatial import KDTree, ConvexHull
from shapely.geometry import Polygon, Point
import geopandas as gpd
import random
from itertools import product
from multiprocessing import Pool
from tqdm import tqdm
from segger.data.utils import *
from segger.data.constants import *
from enum import Enum
from scipy.spatial import KDTree
import warnings
from shapely.affinity import scale
import dask_geopandas as dgpd
import dask.dataframe as dd
from dask import delayed

class SpatialTranscriptomicsSample(ABC):
    def __init__(
        self,
        transcripts_df: pd.DataFrame = None,
        transcripts_radius: int = 10,
        boundaries_graph: bool = False,
        keys: Dict = None
    ):
        """
        Initialize the SpatialTranscriptomicsSample class.

        Parameters:
        -----------
        transcripts_df : pd.DataFrame, optional
            A DataFrame containing transcript data.
        transcripts_radius : int, optional
            Radius for transcripts in the analysis.
        boundaries_graph : bool, optional
            Whether to include boundaries (e.g., nucleus, cell) graph information.
        keys : Enum, optional
            The enum class containing key mappings specific to the dataset.
        """
        self.transcripts_df = transcripts_df
        self.transcripts_radius = transcripts_radius
        self.boundaries_graph = boundaries_graph
        self.keys = keys
        self.embeddings_dict = {}
        self.tx_tree = None
        self.bd_tree = None
        self.tx_tx_edge_index = None

        if self.transcripts_df is not None:
            self._set_bounds()

    def _set_bounds(self) -> None:
        """
        Set the bounding box limits based on the current transcripts dataframe using lazy evaluation with Dask.
        """
        self.x_min = self.transcripts_df[self.keys.TRANSCRIPTS_X.value].min()  # Lazy execution, no compute yet
        self.x_max = self.transcripts_df[self.keys.TRANSCRIPTS_X.value].max()  # Lazy execution, no compute yet
        self.y_min = self.transcripts_df[self.keys.TRANSCRIPTS_Y.value].min()  # Lazy execution, no compute yet
        self.y_max = self.transcripts_df[self.keys.TRANSCRIPTS_Y.value].max()  # Lazy execution, no compute yet

    @abstractmethod
    def filter_transcripts(self, transcripts_df: pd.DataFrame, min_qv: float = 20.0) -> pd.DataFrame:
        """
        Abstract method to filter transcripts based on dataset-specific criteria.

        Parameters:
        -----------
        transcripts_df : pd.DataFrame
            The dataframe containing transcript data.
        min_qv : float, optional
            The minimum quality value threshold for filtering transcripts.

        Returns:
        --------
        pd.DataFrame
            The filtered dataframe.
        """
        pass


    def load_transcripts(
        self,
        base_path: Path = None,
        sample: str = None,
        transcripts_filename: str = None,
        path: Path = None,
        min_qv: int = 20,
        file_format: str = "parquet",
        additional_embeddings: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> None:
        """
        Load transcripts from a Parquet file using Dask for chunked processing.

        Parameters:
        -----------
        base_path : Path, optional
            The base directory path where samples are stored.
        sample : str, optional
            The sample name or identifier.
        transcripts_filename : str, optional
            The filename of the transcripts file (default is derived from the dataset keys).
        path : Path, optional
            Specific path to the transcripts file.
        min_qv : int, optional
            Minimum quality value to filter transcripts (default is 20).
        file_format : str, optional
            Format of the file to load. Only 'parquet' supported in this refactor.
        additional_embeddings : Dict[str, pd.DataFrame], optional
            A dictionary of additional embeddings for genes.

        Returns:
        --------
        None
        """
        if file_format != "parquet":
            raise ValueError("This version only supports parquet files with Dask.")

        # Load transcripts as a Dask DataFrame for efficient chunked loading
        transcripts_filename = transcripts_filename or self.keys.TRANSCRIPTS_FILE.value
        file_path = path or (base_path / sample / transcripts_filename)
        self.transcripts_df = dd.read_parquet(file_path, columns=[
            self.keys.TRANSCRIPTS_X.value,
            self.keys.TRANSCRIPTS_Y.value,
            self.keys.FEATURE_NAME.value,
            self.keys.QUALITY_VALUE.value
        ])
        
        # Filter transcripts using Dask (lazily)
        self.transcripts_df = self.transcripts_df[self.transcripts_df[self.keys.QUALITY_VALUE.value] >= min_qv]

        if additional_embeddings:
            for key, embedding_df in additional_embeddings.items():
                valid_genes = embedding_df.index
                initial_count = len(self.transcripts_df)
                self.transcripts_df = self.transcripts_df[
                    self.transcripts_df[self.keys.FEATURE_NAME.value].isin(valid_genes)
                ]
                print(
                    f"Dropped transcripts not found in {key} embedding."
                )

        self._set_bounds()
        
        # Encode genes as one-hot by default (lazy execution with Dask)
        genes = self.transcripts_df[[self.keys.FEATURE_NAME.value]]
        self.tx_encoder = OneHotEncoder(sparse_output=False)
        self.tx_encoder.fit(genes.compute())  # Fit on the actual data (loaded lazily)
        self.embeddings_dict['one_hot'] = delayed(self.tx_encoder.transform)(genes)
        self.current_embedding = 'one_hot'
    
    def _set_bounds(self) -> None:
        """Set the bounding box limits based on the current transcripts dataframe."""
        self.x_min = self.transcripts_df[self.keys.TRANSCRIPTS_X.value].min().compute()
        self.x_max = self.transcripts_df[self.keys.TRANSCRIPTS_X.value].max().compute()
        self.y_min = self.transcripts_df[self.keys.TRANSCRIPTS_Y.value].min().compute()
        self.y_max = self.transcripts_df[self.keys.TRANSCRIPTS_Y.value].max().compute()

    def load_boundaries(self, path: Path, file_format: str = "parquet") -> None:
        """
        Load boundaries data lazily using Dask.

        Parameters:
        -----------
        path : Path
            Path to the boundaries file.
        file_format : str, optional
            Format of the file to load. Only 'parquet' supported in this refactor.

        Returns:
        --------
        None
        """
        if file_format != "parquet":
            raise ValueError(f"Unsupported file format: {file_format}")

        # Load boundaries lazily using Dask DataFrame
        self.boundaries_df = dd.read_parquet(path, columns=[
            self.keys.BOUNDARIES_VERTEX_X.value,
            self.keys.BOUNDARIES_VERTEX_Y.value,
            self.keys.CELL_ID.value
        ])
        
    def set_embedding(self, embedding_name: str) -> None:
        """
        Set the current embedding type for the transcripts.

        Parameters:
        -----------
        embedding_name : str
            The name of the embedding to use.

        Returns:
        --------
        None
        """
        if embedding_name in self.embeddings_dict:
            self.current_embedding = embedding_name
        else:
            raise ValueError(f"Embedding {embedding_name} not found in embeddings_dict.")
        
    def get_tile_data(self, x_min: float, y_min: float, x_size: float, y_size: float) -> Tuple[dd.DataFrame, dd.DataFrame]:
        """
        Load the necessary data for a given tile from the transcripts and boundaries.
        Uses Dask's filtering for chunked processing.
        Parameters:
        -----------
        x_min : float
            Minimum x-coordinate of the tile.
        y_min : float
            Minimum y-coordinate of the tile.
        x_size : float
            Width of the tile.
        y_size : float
            Height of the tile.
        Returns:
        --------
        Tuple[dd.DataFrame, dd.DataFrame]
            Transcripts and boundaries data for the tile.
        """
        x_max = x_min + x_size
        y_max = y_min + y_size
        # Use Dask DataFrame filtering to get only the required data
        tile_transcripts = self.transcripts_df[
            (self.transcripts_df[self.keys.TRANSCRIPTS_X.value] >= x_min) &
            (self.transcripts_df[self.keys.TRANSCRIPTS_X.value] <= x_max) &
            (self.transcripts_df[self.keys.TRANSCRIPTS_Y.value] >= y_min) &
            (self.transcripts_df[self.keys.TRANSCRIPTS_Y.value] <= y_max)
        ]
        tile_boundaries = self.boundaries_df[
            (self.boundaries_df[self.keys.BOUNDARIES_VERTEX_X.value] >= x_min) &
            (self.boundaries_df[self.keys.BOUNDARIES_VERTEX_X.value] <= x_max) &
            (self.boundaries_df[self.keys.BOUNDARIES_VERTEX_Y.value] >= y_min) &
            (self.boundaries_df[self.keys.BOUNDARIES_VERTEX_Y.value] <= y_max)
        ]
        return tile_transcripts, tile_boundaries
    
    def get_bounding_box(
        self,
        x_min: float = None,
        y_min: float = None,
        x_max: float = None,
        y_max: float = None,
        in_place: bool = True,
    ) -> Optional['SpatialTranscriptomicsSample']:
        """
        Subsets the transcripts_df and boundaries_df within the specified bounding box using Dask.
        Parameters:
        -----------
        x_min : float, optional
            The minimum x-coordinate of the bounding box.
        y_min : float, optional
            The minimum y-coordinate of the bounding box.
        x_max : float, optional
            The maximum x-coordinate of the bounding box.
        y_max : float, optional
            The maximum y-coordinate of the bounding box.
        in_place : bool, optional
            If True, modifies the current instance. If False, returns a new instance with the subsetted data.
        Returns:
        --------
        Optional[SpatialTranscriptomicsSample]
            If in_place is True, returns None after modifying the existing instance.
            If in_place is False, returns a new SpatialTranscriptomicsSample instance with the subsetted data.
        """
        # Validate the bounding box coordinates
        x_min, y_min, x_max, y_max = self._validate_bbox(x_min, y_min, x_max, y_max)
        # Use Dask DataFrame to subset transcripts within the bounding box
        subset_transcripts_df = self.transcripts_df[
            (self.transcripts_df[self.keys.TRANSCRIPTS_X.value] >= x_min)
            & (self.transcripts_df[self.keys.TRANSCRIPTS_X.value] <= x_max)
            & (self.transcripts_df[self.keys.TRANSCRIPTS_Y.value] >= y_min)
            & (self.transcripts_df[self.keys.TRANSCRIPTS_Y.value] <= y_max)
        ]
        # Subset boundaries_df if it exists using Dask
        subset_boundaries_df = None
        if self.boundaries_df is not None:
            subset_boundaries_df = self.boundaries_df[
                (self.boundaries_df[self.keys.BOUNDARIES_VERTEX_X.value] >= x_min)
                & (self.boundaries_df[self.keys.BOUNDARIES_VERTEX_X.value] <= x_max)
                & (self.boundaries_df[self.keys.BOUNDARIES_VERTEX_Y.value] >= y_min)
                & (self.boundaries_df[self.keys.BOUNDARIES_VERTEX_Y.value] <= y_max)
            ]
        # Reset the indices lazily using Dask's `reset_index`
        subset_transcripts_df = subset_transcripts_df.reset_index(drop=True)
        if subset_boundaries_df is not None:
            subset_boundaries_df = subset_boundaries_df.reset_index(drop=True)
        if in_place:
            # Update the current instance with the subsetted data
            self.transcripts_df = subset_transcripts_df
            self.boundaries_df = subset_boundaries_df
            self.x_min, self.y_min, self.x_max, self.y_max = x_min, y_min, x_max, y_max
            # No need to return a new instance
            print(f"Subset completed in-place.")
            return None
        else:
            # Create a new instance of SpatialTranscriptomicsSample with the subsetted data
            new_instance = self.__class__(subset_transcripts_df, self.transcripts_radius, self.boundaries_graph, self.keys)
            new_instance.boundaries_df = subset_boundaries_df
            new_instance.x_min, new_instance.y_min, new_instance.x_max, new_instance.y_max = x_min, y_min, x_max, y_max
            print(f"Subset completed with new instance.")
        return new_instance
        

    def generate_and_scale_polygons(
        self,
        boundaries_df: dd.DataFrame,
        scale_factor: float = 1.0
    ) -> dgpd.GeoDataFrame:
        """
        Generate polygons from boundary coordinates, scale them, and add centroids using Dask.

        Parameters:
        -----------
        boundaries_df : dask.dataframe.DataFrame
            DataFrame containing boundary coordinates.
        scale_factor : float
            The factor by which to scale the polygons.

        Returns:
        --------
        dask_geopandas.GeoDataFrame
            A GeoDataFrame containing scaled Polygon objects and their centroids.
        """
        def create_scaled_polygon(group_data, scale_factor):
            x_coords = group_data[self.keys.BOUNDARIES_VERTEX_X.value]
            y_coords = group_data[self.keys.BOUNDARIES_VERTEX_Y.value]
            
            if len(x_coords) >= 3:  # Ensure there are at least 3 points to form a polygon
                polygon = Polygon(list(zip(x_coords, y_coords)))
                if polygon.is_valid and not polygon.is_empty:  # Check for valid, non-degenerate polygon
                    scaled_polygon = scale(polygon, xfact=scale_factor, yfact=scale_factor, origin='centroid')
                    if scaled_polygon.is_valid and not scaled_polygon.is_empty:
                        return scaled_polygon
            return None
        
        # Create polygons lazily using delayed
        delayed_polygons = boundaries_df.groupby(self.keys.CELL_ID.value).apply(
            lambda group: delayed(create_scaled_polygon)(group, scale_factor)
        ).compute()

        # Filter out None results (invalid polygons)
        valid_polygons = [p for p in delayed_polygons if p is not None]

        # Create a GeoDataFrame from valid polygons and their centroids
        polygons_gdf = dgpd.from_geopandas(
            gpd.GeoDataFrame(geometry=valid_polygons, crs='EPSG:4326'),
            npartitions=boundaries_df.npartitions
        )

        # Add centroids lazily
        polygons_gdf['centroid_x'] = polygons_gdf.geometry.centroid.x
        polygons_gdf['centroid_y'] = polygons_gdf.geometry.centroid.y

        return polygons_gdf

    

    def compute_transcript_overlap_with_boundaries(
        self,
        transcripts_df: dd.DataFrame,
        boundaries_df: dd.DataFrame = None,
        polygons_gdf: dgpd.GeoDataFrame = None,
        scale_factor: float = 1.0
    ) -> dd.DataFrame:
        """
        Computes the overlap of transcript locations with scaled boundary polygons
        and assigns corresponding cell IDs to the transcripts using Dask.

        Parameters:
        -----------
        transcripts_df : dask.dataframe.DataFrame
            Dask DataFrame containing transcript data.
        boundaries_df : dask.dataframe.DataFrame, optional
            Dask DataFrame containing boundary data. Required if polygons_gdf is not provided.
        polygons_gdf : dask_geopandas.GeoDataFrame, optional
            Precomputed Dask GeoDataFrame containing boundary polygons. If None, will compute from boundaries_df.
        scale_factor : float, optional
            The factor by which to scale the boundary polygons. Default is 1.0.

        Returns:
        --------
        dask.dataframe.DataFrame
            The updated DataFrame with overlap information (True for overlap, False for no overlap)
            and assigned cell IDs.
        """
        # Check if polygons_gdf is provided, otherwise compute from boundaries_df
        if polygons_gdf is None:
            if boundaries_df is None:
                raise ValueError("Both boundaries_df and polygons_gdf cannot be None. Provide at least one.")
            
            # Generate polygons from boundaries_df if polygons_gdf is None
            print(f"No precomputed polygons provided. Computing polygons from boundaries with a scale factor of {scale_factor}.")
            polygons_gdf = self.generate_and_scale_polygons(boundaries_df, scale_factor)
        
        if polygons_gdf.empty().compute():
            raise ValueError("No valid polygons were generated from the boundaries.")
        else:
            print(f"Polygons are available. Proceeding with overlap computation.")

        # Create a delayed function to check if a point is within any polygon
        def check_overlap(transcript, polygons_gdf):
            x = transcript[self.keys.TRANSCRIPTS_X.value]
            y = transcript[self.keys.TRANSCRIPTS_Y.value]
            point = Point(x, y)

            overlap = False
            cell_id = None

            # Check for point containment lazily within polygons
            for _, polygon in polygons_gdf.iterrows():
                if polygon.geometry.contains(point):
                    overlap = True
                    cell_id = polygon[self.keys.CELL_ID.value]
                    break

            return overlap, cell_id

        # Apply the check_overlap function in parallel to each row using Dask's map_partitions
        print(f"Starting overlap computation for transcripts with the boundary polygons.")
        
        transcripts_df = transcripts_df.map_partitions(
            lambda df: df.assign(
                **{
                    self.keys.OVERLAPS_BOUNDARY.value: df.apply(lambda row: delayed(check_overlap)(row, polygons_gdf)[0], axis=1),
                    self.keys.CELL_ID.value: df.apply(lambda row: delayed(check_overlap)(row, polygons_gdf)[1], axis=1),
                }
            )
        )

        return transcripts_df




    def compute_boundaries_geometries(
        self,
        boundaries_df: dd.DataFrame = None,
        polygons_gdf: dgpd.GeoDataFrame = None,
        scale_factor: float = 1.0,
        area: bool = True,
        convexity: bool = True,
        elongation: bool = True,
        circularity: bool = True,
    ) -> dgpd.GeoDataFrame:
        """
        Computes geometries for boundaries (e.g., nuclei, cells) from the dataframe using Dask.

        Parameters:
        -----------
        boundaries_df : dask.dataframe.DataFrame, optional
            The dataframe containing boundaries data. Required if polygons_gdf is not provided.
        polygons_gdf : dask_geopandas.GeoDataFrame, optional
            Precomputed Dask GeoDataFrame containing boundary polygons. If None, will compute from boundaries_df.
        scale_factor : float, optional
            The factor by which to scale the polygons (default is 1.0, no scaling).
        area : bool, optional
            Whether to compute area.
        convexity : bool, optional
            Whether to compute convexity.
        elongation : bool, optional
            Whether to compute elongation.
        circularity : bool, optional
            Whether to compute circularity.

        Returns:
        --------
        dask_geopandas.GeoDataFrame
            A GeoDataFrame containing computed geometries.
        """
        # Check if polygons_gdf is provided, otherwise compute from boundaries_df
        if polygons_gdf is None:
            if boundaries_df is None:
                raise ValueError("Both boundaries_df and polygons_gdf cannot be None. Provide at least one.")
            
            # Generate polygons from boundaries_df if polygons_gdf is None
            print(f"No precomputed polygons provided. Computing polygons from boundaries with a scale factor of {scale_factor}.")
            polygons_gdf = self.generate_and_scale_polygons(boundaries_df, scale_factor)
        
        # Check if the generated polygons_gdf is empty
        if polygons_gdf.empty().compute():
            raise ValueError("No valid polygons were generated from the boundaries.")
        else:
            print(f"Polygons are available. Proceeding with geometrical computations.")
        
        # Compute additional geometrical properties
        polygons = polygons_gdf.geometry

        # Compute additional geometrical properties
        if area:
            print("Computing area...")
            polygons_gdf['area'] = polygons.area
        if convexity:
            print("Computing convexity...")
            polygons_gdf['convexity'] = polygons.convex_hull.area / polygons.area
        if elongation:
            print("Computing elongation...")
            r = polygons.minimum_rotated_rectangle
            polygons_gdf['elongation'] = (r.length * r.length) / r.area
        if circularity:
            print("Computing circularity...")
            r = polygons_gdf.minimum_bounding_radius()
            polygons_gdf['circularity'] = polygons.area / (r * r)

        print("Geometrical computations completed.")

        return polygons_gdf

    def _validate_bbox(self, x_min: float, y_min: float, x_max: float, y_max: float) -> Tuple[float, float, float, float]:
        """Validates and sets default values for bounding box coordinates."""
        if x_min is None:
            x_min = self.x_min
        if y_min is None:
            y_min = self.y_min
        if x_max is None:
            x_max = self.x_max
        if y_max is None:
            y_max = self.y_max
        return x_min, y_min, x_max, y_max

    def save_dataset_for_segger(
        self,
        processed_dir: Path,
        x_size: float = 1000,
        y_size: float = 1000,
        d_x: float = 900,
        d_y: float = 900,
        margin_x: float = None,
        margin_y: float = None,
        compute_labels: bool = True,
        r_tx: float = 5,
        k_tx: int = 3,
        val_prob: float = 0.1,
        test_prob: float = 0.2,
        neg_sampling_ratio_approx: float = 5,
        sampling_rate: float = 1,
        num_workers: int = 1,
        receptive_field: Dict[str, float] = {
            "k_bd": 4,
            "dist_bd": 20,
            "k_tx": 5,
            "dist_tx": 10,
        },
        method: str = 'kd_tree',
        gpu: bool = False,
        workers: int = 1
    ) -> None:
        """
        Saves the dataset for Segger in a processed format using Dask for parallel and lazy processing.

        Parameters:
        -----------
        processed_dir : Path
            Directory to save the processed dataset.
        x_size : float, optional
            Width of each tile.
        y_size : float, optional
            Height of each tile.
        d_x : float, optional
            Step size in the x direction for tiles.
        d_y : float, optional
            Step size in the y direction for tiles.
        margin_x : float, optional
            Margin in the x direction to include transcripts.
        margin_y : float, optional
            Margin in the y direction to include transcripts.
        compute_labels : bool, optional
            Whether to compute edge labels for tx_belongs_bd edges.
        r_tx : float, optional
            Radius for building the transcript-to-transcript graph.
        k_tx : int, optional
            Number of nearest neighbors for the tx-tx graph.
        val_prob : float, optional
            Probability of assigning a tile to the validation set.
        test_prob : float, optional
            Probability of assigning a tile to the test set.
        neg_sampling_ratio_approx : float, optional
            Approximate ratio of negative samples.
        sampling_rate : float, optional
            Rate of sampling tiles.
        num_workers : int, optional
            Number of workers to use for parallel processing.
        receptive_field : dict, optional
            Dictionary containing the values for 'k_bd', 'dist_bd', 'k_tx', and 'dist_tx'.
        method : str, optional
            Method for computing edge indices (e.g., 'kd_tree', 'faiss').
        gpu : bool, optional
            Whether to use GPU acceleration for edge index computation.
        workers : int, optional
            Number of workers to use to compute the neighborhood graph (per tile).

        Returns:
        --------
        None
        """
        # Prepare directories for storing processed tiles
        self._prepare_directories(processed_dir)
        
        # Get x and y coordinate ranges for tiling
        x_range, y_range = self._get_ranges(d_x, d_y)
        
        # Generate parameters for each tile
        tile_params = self._generate_tile_params(
            x_range, y_range, x_size, y_size, margin_x, margin_y, compute_labels, 
            r_tx, k_tx, val_prob, test_prob, neg_sampling_ratio_approx, sampling_rate, 
            processed_dir, receptive_field, method, gpu, workers
        )

        # Process each tile using Dask to parallelize the task
        print("Starting tile processing...")
        tasks = [delayed(self._process_tile)(params) for params in tile_params]
        
        # Use Dask to process all tiles in parallel
        dask.compute(*tasks, num_workers=num_workers)
        print("Tile processing completed.")


    def _prepare_directories(self, processed_dir: Path) -> None:
        """Prepares directories for saving tiles."""
        processed_dir.mkdir(parents=True, exist_ok=True)
        (processed_dir / 'train_tiles/processed').mkdir(parents=True, exist_ok=True)
        (processed_dir / 'test_tiles/processed').mkdir(parents=True, exist_ok=True)
        (processed_dir / 'val_tiles/processed').mkdir(parents=True, exist_ok=True)

    def _get_ranges(self, d_x: float, d_y: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generates ranges for tiling."""
        x_range = np.arange(self.x_min // 1000 * 1000, self.x_max, d_x)
        y_range = np.arange(self.y_min // 1000 * 1000, self.y_max, d_y)
        return x_range, y_range
    
    def _generate_tile_params(
        self,
        x_range: np.ndarray,
        y_range: np.ndarray,
        x_size: float,
        y_size: float,
        margin_x: float,
        margin_y: float,
        compute_labels: bool,
        r_tx: float,
        k_tx: int,
        val_prob: float,
        test_prob: float,
        neg_sampling_ratio_approx: float,
        sampling_rate: float,
        processed_dir: Path,
        receptive_field: Dict[str, float],
        method: str,
        gpu: bool,
        workers: int,
        use_precomputed: bool  # New argument
    ) -> List[Tuple]:
        """
        Generates parameters for processing tiles.

        Parameters:
        -----------
        x_range : np.ndarray
            Array of x-coordinate ranges for the tiles.
        y_range : np.ndarray
            Array of y-coordinate ranges for the tiles.
        x_size : float
            Width of each tile.
        y_size : float
            Height of each tile.
        margin_x : float
            Margin in the x direction to include transcripts.
        margin_y : float
            Margin in the y direction to include transcripts.
        compute_labels : bool
            Whether to compute edge labels for tx_belongs_bd edges.
        r_tx : float
            Radius for building the transcript-to-transcript graph.
        k_tx : int
            Number of nearest neighbors for the tx-tx graph.
        val_prob : float
            Probability of assigning a tile to the validation set.
        test_prob : float
            Probability of assigning a tile to the test set.
        neg_sampling_ratio_approx : float
            Approximate ratio of negative samples.
        sampling_rate : float
            Rate of sampling tiles.
        processed_dir : Path
            Directory to save the processed tiles.
        receptive_field : Dict[str, float]
            Dictionary containing the values for 'k_bd', 'dist_bd', 'k_tx', and 'dist_tx'.
        method : str
            Method for computing edge indices (e.g., 'kd_tree', 'faiss').
        gpu : bool
            Whether to use GPU acceleration for edge index computation.
        workers : int
            Number of workers to use for parallel processing.
        use_precomputed : bool
            Whether to use precomputed graphs for tx-tx edges.

        Returns:
        --------
        List[Tuple]
            List of parameters for processing each tile.
        """
        margin_x = margin_x if margin_x is not None else x_size // 10
        margin_y = margin_y if margin_y is not None else y_size // 10

        x_masks_bd = [
            (self.boundaries_df[self.keys.BOUNDARIES_VERTEX_X.value] > x) & 
            (self.boundaries_df[self.keys.BOUNDARIES_VERTEX_X.value] < x + x_size)
            for x in x_range
        ]
        y_masks_bd = [
            (self.boundaries_df[self.keys.BOUNDARIES_VERTEX_Y.value] > y) & 
            (self.boundaries_df[self.keys.BOUNDARIES_VERTEX_Y.value] < y + y_size)
            for y in y_range
        ]
        x_masks_tx = [
            (self.transcripts_df[self.keys.TRANSCRIPTS_X.value] > (x - margin_x))
            & (self.transcripts_df[self.keys.TRANSCRIPTS_X.value] < (x + x_size + margin_x))
            for x in x_range
        ]
        y_masks_tx = [
            (self.transcripts_df[self.keys.TRANSCRIPTS_Y.value] > (y - margin_y))
            & (self.transcripts_df[self.keys.TRANSCRIPTS_Y.value] < (y + y_size + margin_y))
            for y in y_range
        ]

        tile_params = [
            (
                i, j, x_masks_bd, y_masks_bd, x_masks_tx, y_masks_tx, x_size, y_size, x_range[i], y_range[j], 
                compute_labels, r_tx, k_tx, neg_sampling_ratio_approx, val_prob, test_prob, 
                processed_dir, receptive_field, sampling_rate, method, gpu, workers, use_precomputed
            )
            for i, j in product(range(len(x_range)), range(len(y_range)))
        ]
        return tile_params


    def _process_tiles(self, tile_params: List[Tuple], num_workers: int) -> None:
        """Processes the tiles using the specified number of workers."""
        if num_workers > 1:
            with Pool(processes=num_workers) as pool:
                for _ in tqdm(pool.imap_unordered(self._process_tile, tile_params), total=len(tile_params)):
                    pass
        else:
            for params in tqdm(tile_params):
                self._process_tile(params)


    def _process_tile(self, tile_params: Tuple) -> None:
        """
        Process a single tile using Dask for parallelism and lazy evaluation, and save the data.

        Parameters:
        -----------
        tile_params : tuple
            Parameters for the tile processing.

        Returns:
        --------
        None
        """
        (
            i, j, x_masks_bd, y_masks_bd, x_masks_tx, y_masks_tx, x_size, y_size, x_loc, y_loc, compute_labels, 
            r_tx, k_tx, neg_sampling_ratio_approx, val_prob, test_prob, processed_dir, receptive_field, 
            sampling_rate, method, gpu, workers
        ) = tile_params

        # Log the tile processing start with bounding box coordinates
        print(f"Processing tile at location (x_min: {x_loc}, y_min: {y_loc}), size (width: {x_size}, height: {y_size})")

        # Sampling rate to decide if the tile should be processed
        if random.random() > sampling_rate:
            print(f"Skipping tile at (x_min: {x_loc}, y_min: {y_loc}) due to sampling rate.")
            return

        # Apply lazy filtering using Dask
        x_mask = x_masks_bd[i]
        y_mask = y_masks_bd[j]
        mask = x_mask & y_mask

        # If the mask returns no data, skip the tile
        if mask.sum().compute() == 0:
            print(f"No data found in the boundaries for tile at (x_min: {x_loc}, y_min: {y_loc}). Skipping.")
            return

        # Lazy loading of boundary and transcript data for the tile using Dask
        print(f"Loading boundary and transcript data for tile at (x_min: {x_loc}, y_min: {y_loc})...")
        bd_df = self.boundaries_df[mask].persist()  # Use Dask's persist to load necessary partitions
        bd_df = bd_df.loc[bd_df[self.keys.CELL_ID.value].isin(bd_df[self.keys.CELL_ID.value])]

        tx_mask = x_masks_tx[i] & y_masks_tx[j]
        transcripts_df = self.transcripts_df[tx_mask].persist()

        # If no data is found in transcripts or boundaries, skip the tile
        if transcripts_df.shape[0].compute() == 0 or bd_df.shape[0].compute() == 0:
            print(f"No transcripts or boundaries data found for tile at (x_min: {x_loc}, y_min: {y_loc}). Skipping.")
            return

        # Build PyG data structure from tile-specific data
        print(f"Building PyG data for tile at (x_min: {x_loc}, y_min: {y_loc})...")
        data = self.build_pyg_data_from_tile(
            bd_df, transcripts_df, r_tx=r_tx, k_tx=k_tx, method=method, gpu=gpu, workers=workers
        )

        try:
            # probability to assign to train-val-test split
            prob = random.random() 
            if compute_labels and (prob > val_prob + test_prob):
                print(f"Computing labels for tile at (x_min: {x_loc}, y_min: {y_loc})...")
                transform = RandomLinkSplit(
                    num_val=0, num_test=0, is_undirected=True, edge_types=[('tx', 'belongs', 'bd')],
                    neg_sampling_ratio=neg_sampling_ratio_approx * 2,
                )
                data, _, _ = transform(data)
                edge_index = data[('tx', 'belongs', 'bd')].edge_index
                if edge_index.shape[1] < 10:
                    print(f"Insufficient edge data for tile at (x_min: {x_loc}, y_min: {y_loc}). Skipping.")
                    return
                edge_label_index = data[('tx', 'belongs', 'bd')].edge_label_index
                data[('tx', 'belongs', 'bd')].edge_label_index = edge_label_index[
                    :, torch.nonzero(
                        torch.any(
                            edge_label_index[0].unsqueeze(1) == edge_index[0].unsqueeze(0),
                            dim=1,
                        )
                    ).squeeze()
                ]
                data[('tx', 'belongs', 'bd')].edge_label = data[('tx', 'belongs', 'bd')].edge_label[
                    torch.nonzero(
                        torch.any(
                            edge_label_index[0].unsqueeze(1) == edge_index[0].unsqueeze(0),
                            dim=1,
                        )
                    ).squeeze()
                ]

            # Compute coordinates for boundary and transcript nodes
            coords_bd = data['bd'].pos
            coords_tx = data['tx'].pos[:, :2]

            # Commented out the following lines for computing tx-tx and bd-tx edges
            # print(f"Computing tx-tx edges for tile at (x_min: {x_loc}, y_min: {y_loc})...")
            # data['tx'].tx_field = tx_edges.compute()  # Trigger the computation

            # print(f"Computing bd-tx edges for tile at (x_min: {x_loc}, y_min: {y_loc})...")
            # data['tx'].bd_field = bd_tx_edges.compute()

            # Save the tile data to the appropriate directory based on split
            print(f"Saving data for tile at (x_min: {x_loc}, y_min: {y_loc})...")
            filename = f"tiles_x{x_loc}_y{y_loc}_{x_size}_{y_size}.pt"
            if prob > val_prob + test_prob:
                save_task = delayed(torch.save)(data, processed_dir / 'train_tiles' / 'processed' / filename)
            elif prob > val_prob:
                save_task = delayed(torch.save)(data, processed_dir / 'test_tiles' / 'processed' / filename)
            else:
                save_task = delayed(torch.save)(data, processed_dir / 'val_tiles' / 'processed' / filename)
            
            # Use Dask to save the file in parallel
            save_task.compute()

            print(f"Tile at (x_min: {x_loc}, y_min: {y_loc}) processed and saved successfully.")

        except Exception as e:
            print(f"Error processing tile at (x_min: {x_loc}, y_min: {y_loc}): {e}")


    def build_pyg_data_from_tile(
        self, boundaries_df: dd.DataFrame, transcripts_df: dd.DataFrame, r_tx: float = 5.0, k_tx: int = 3, method: str = 'kd_tree', gpu: bool = False, workers: int = 1
    ) -> HeteroData:
        """
        Builds PyG data from a tile of boundaries and transcripts data using Dask utilities for efficient processing.

        Parameters:
        -----------
        boundaries_df : dd.DataFrame
            Dask DataFrame containing boundaries data (e.g., nucleus, cell).
        transcripts_df : dd.DataFrame
            Dask DataFrame containing transcripts data.
        r_tx : float
            Radius for building the transcript-to-transcript graph.
        k_tx : int
            Number of nearest neighbors for the tx-tx graph.
        method : str, optional
            Method for computing edge indices (e.g., 'kd_tree', 'faiss').
        gpu : bool, optional
            Whether to use GPU acceleration for edge index computation.

        Returns:
        --------
        HeteroData
            PyG Heterogeneous Data object.
        """
        # Initialize the PyG HeteroData object
        data = HeteroData()
        # Lazily compute boundaries geometries using Dask
        print("Computing boundaries geometries...")
        bd_gdf = self.compute_boundaries_geometries(boundaries_df).compute()  # Trigger computation
        # Get the maximum area for any boundary (useful for additional processing, but not required here)
        max_area = bd_gdf['area'].max()
        # Prepare transcript features and positions lazily using Dask
        print("Preparing transcript features and positions...")
        x_xyz = torch.as_tensor(transcripts_df[[self.keys.TRANSCRIPTS_X.value, self.keys.TRANSCRIPTS_Y.value, self.keys.TRANSCRIPTS_Y.value]].compute().values).float()
        data['tx'].id = transcripts_df.index.compute().values
        data['tx'].pos = x_xyz
        # Prepare transcript embeddings (this part is eagerly loaded as it depends on embeddings_dict)
        x_features = torch.as_tensor(self.embeddings_dict[self.current_embedding]).float()
        data['tx'].x = x_features.to_sparse()
        # Check if the overlap column exists, if not, compute it lazily using Dask
        if self.keys.OVERLAPS_BOUNDARY.value not in transcripts_df.columns:
            print(f"Computing overlaps for transcripts...")
            transcripts_df = delayed(self.compute_transcript_overlap_with_boundaries)(
                transcripts_df, boundaries_df, scale_factor=1.0
            ).compute()  # Trigger computation of overlaps
        # Connect transcripts with their corresponding boundaries (e.g., nuclei, cells)
        print("Connecting transcripts with boundaries...")
        ind = np.where(
            (transcripts_df[self.keys.OVERLAPS_BOUNDARY.value].compute()) &
            (transcripts_df[self.keys.CELL_ID.value].isin(bd_gdf[self.keys.CELL_ID.value]))
        )[0]
        tx_bd_edge_index = np.column_stack(
            (ind, np.searchsorted(bd_gdf[self.keys.CELL_ID.value].values, transcripts_df.iloc[ind][self.keys.CELL_ID.value].values))
        )
        # Add boundary node data to PyG HeteroData
        data['bd'].id = bd_gdf[[self.keys.CELL_ID.value]].values
        data['bd'].pos = bd_gdf[['centroid_x', 'centroid_y']].values
        bd_x = bd_gdf.iloc[:, 4:]
        data['bd'].x = torch.as_tensor(bd_x.to_numpy()).float()
        # Add transcript-boundary edge index to PyG HeteroData
        data['tx', 'belongs', 'bd'].edge_index = torch.as_tensor(tx_bd_edge_index.T).long()
        # We omit computing tx-tx edges here, as instructed to drop the precompute option
        print("Finished building PyG data for the tile.")
        return data





class XeniumSample(SpatialTranscriptomicsSample):
    def __init__(self, transcripts_df: dd.DataFrame = None, transcripts_radius: int = 10, boundaries_graph: bool = False):
        super().__init__(transcripts_df, transcripts_radius, boundaries_graph, XeniumKeys)

    def filter_transcripts(self, transcripts_df: dd.DataFrame, min_qv: float = 20.0) -> dd.DataFrame:
        """
        Filters transcripts based on quality value and removes unwanted transcripts for Xenium using Dask.

        Parameters:
        -----------
        transcripts_df : dd.DataFrame
            The Dask DataFrame containing transcript data.
        min_qv : float, optional
            The minimum quality value threshold for filtering transcripts.

        Returns:
        --------
        dd.DataFrame
            The filtered Dask DataFrame.
        """
        filter_codewords = (
            "NegControlProbe_",
            "antisense_",
            "NegControlCodeword_",
            "BLANK_",
            "DeprecatedCodeword_",
        )

        # Ensure FEATURE_NAME is a string type for proper filtering (compatible with Dask)
        # Handle potential bytes to string conversion for Dask DataFrame
        if transcripts_df[self.keys.FEATURE_NAME.value].dtype == 'object':
            transcripts_df[self.keys.FEATURE_NAME.value] = transcripts_df[self.keys.FEATURE_NAME.value].apply(
                lambda x: x.decode('utf-8') if isinstance(x, bytes) else x, meta=('x', 'str')
            )

        # Apply the quality value filter using Dask
        mask_quality = transcripts_df[self.keys.QUALITY_VALUE.value] >= min_qv

        # Apply the filter for unwanted codewords (works efficiently with Dask)
        mask_codewords = ~transcripts_df[self.keys.FEATURE_NAME.value].str.startswith(filter_codewords)

        # Combine the filters and return the filtered Dask DataFrame
        mask = mask_quality & mask_codewords
        return transcripts_df[mask]

class MerscopeSample(SpatialTranscriptomicsSample):
    def __init__(self, transcripts_df: dd.DataFrame = None, transcripts_radius: int = 10, boundaries_graph: bool = False):
        super().__init__(transcripts_df, transcripts_radius, boundaries_graph, MerscopeKeys)

    def filter_transcripts(self, transcripts_df: dd.DataFrame, min_qv: float = 20.0) -> dd.DataFrame:
        """
        Filters transcripts based on specific criteria for Merscope using Dask.

        Parameters:
        -----------
        transcripts_df : dd.DataFrame
            The Dask DataFrame containing transcript data.
        min_qv : float, optional
            The minimum quality value threshold for filtering transcripts.

        Returns:
        --------
        dd.DataFrame
            The filtered Dask DataFrame.
        """
        # Add custom Merscope-specific filtering logic if needed
        # For now, apply only the quality value filter
        return transcripts_df[transcripts_df[self.keys.QUALITY_VALUE.value] >= min_qv]


