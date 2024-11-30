from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Union
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
import pyarrow.parquet as pq
import dask
from sklearn.preprocessing import LabelEncoder
from torch_geometric.transforms import RandomLinkSplit
from dask.diagnostics import ProgressBar
import logging
import warnings

for msg in [r".*Geometry is in a geographic CRS.*", r".*You did not provide metadata.*"]:
    warnings.filterwarnings("ignore", category=UserWarning, message=msg)


class SpatialTranscriptomicsSample(ABC):
    def __init__(
        self,
        transcripts_df: pd.DataFrame = None,
        transcripts_radius: int = 10,
        boundaries_graph: bool = False,
        embedding_df: pd.DataFrame = None,
        keys: Dict = None,
        verbose: bool = True,
    ):
        """Initialize the SpatialTranscriptomicsSample class.

        Args:
            transcripts_df (pd.DataFrame, optional): A DataFrame containing transcript data.
            transcripts_radius (int, optional): Radius for transcripts in the analysis.
            boundaries_graph (bool, optional): Whether to include boundaries (e.g., nucleus, cell) graph information.
            keys (Dict, optional): The enum class containing key mappings specific to the dataset.
        """
        self.transcripts_df = transcripts_df
        self.transcripts_radius = transcripts_radius
        self.boundaries_graph = boundaries_graph
        self.keys = keys
        self.embedding_df = embedding_df
        self.current_embedding = "token"
        self.verbose = verbose

    @abstractmethod
    def filter_transcripts(self, transcripts_df: pd.DataFrame, min_qv: float = 20.0) -> pd.DataFrame:
        """
        Abstract method to filter transcripts based on dataset-specific criteria.

        Parameters:
            transcripts_df (pd.DataFrame): The dataframe containing transcript data.
            min_qv (float, optional): The minimum quality value threshold for filtering transcripts.

        Returns:
            pd.DataFrame: The filtered dataframe.
        """
        pass

    def set_file_paths(self, transcripts_path: Path, boundaries_path: Path) -> None:
        """
        Set the paths for the transcript and boundary files.

        Parameters:
            transcripts_path (Path): Path to the Parquet file containing transcripts data.
            boundaries_path (Path): Path to the Parquet file containing boundaries data.
        """
        self.transcripts_path = transcripts_path
        self.boundaries_path = boundaries_path

        if self.verbose:
            print(f"Set transcripts file path to {transcripts_path}")
        if self.verbose:
            print(f"Set boundaries file path to {boundaries_path}")

    def load_transcripts(
        self,
        base_path: Path = None,
        sample: str = None,
        transcripts_filename: str = None,
        path: Path = None,
        file_format: str = "parquet",
        x_min: float = None,
        x_max: float = None,
        y_min: float = None,
        y_max: float = None,
        # additional_embeddings: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> dd.DataFrame:
        """
        Load transcripts from a Parquet file using Dask for efficient chunked processing,
        only within the specified bounding box, and return the filtered DataFrame with integer token embeddings.

        Parameters:
            base_path (Path, optional): The base directory path where samples are stored.
            sample (str, optional): The sample name or identifier.
            transcripts_filename (str, optional): The filename of the transcripts file (default is derived from the dataset keys).
            path (Path, optional): Specific path to the transcripts file.
            file_format (str, optional): Format of the file to load (default is 'parquet').
            x_min (float, optional): Minimum X-coordinate for the bounding box.
            x_max (float, optional): Maximum X-coordinate for the bounding box.
            y_min (float, optional): Minimum Y-coordinate for the bounding box.
            y_max (float, optional): Maximum Y-coordinate for the bounding box.

        Returns:
            dd.DataFrame: The filtered transcripts DataFrame.
        """
        if file_format != "parquet":
            raise ValueError("This version only supports parquet files with Dask.")

        # Set the file path for transcripts
        transcripts_filename = transcripts_filename or self.keys.TRANSCRIPTS_FILE.value
        file_path = path or (base_path / sample / transcripts_filename)
        self.transcripts_path = file_path

        # Set metadata
        # self.set_metadata()

        # Use bounding box values from set_metadata if not explicitly provided
        x_min = x_min or self.x_min
        x_max = x_max or self.x_max
        y_min = y_min or self.y_min
        y_max = y_max or self.y_max

        # Check for available columns in the file's metadata (without loading the data)
        parquet_metadata = dd.read_parquet(file_path, meta_only=True)
        available_columns = parquet_metadata.columns

        # Define the list of columns to read
        columns_to_read = [
            self.keys.TRANSCRIPTS_ID.value,
            self.keys.TRANSCRIPTS_X.value,
            self.keys.TRANSCRIPTS_Y.value,
            self.keys.FEATURE_NAME.value,
            self.keys.CELL_ID.value,
        ]

        # Check if the QUALITY_VALUE key exists in the dataset, and add it to the columns list if present
        if self.keys.QUALITY_VALUE.value in available_columns:
            columns_to_read.append(self.keys.QUALITY_VALUE.value)

        if self.keys.OVERLAPS_BOUNDARY.value in available_columns:
            columns_to_read.append(self.keys.OVERLAPS_BOUNDARY.value)

        # Use filters to only load data within the specified bounding box (x_min, x_max, y_min, y_max)
        filters = [
            (self.keys.TRANSCRIPTS_X.value, ">=", x_min),
            (self.keys.TRANSCRIPTS_X.value, "<=", x_max),
            (self.keys.TRANSCRIPTS_Y.value, ">=", y_min),
            (self.keys.TRANSCRIPTS_Y.value, "<=", y_max),
        ]

        # Load the dataset lazily with filters applied for the bounding box
        columns = set(dd.read_parquet(file_path).columns)
        transcripts_df = dd.read_parquet(file_path, columns=columns_to_read, filters=filters).compute()

        # Convert transcript and cell IDs to strings lazily
        transcripts_df[self.keys.TRANSCRIPTS_ID.value] = transcripts_df[self.keys.TRANSCRIPTS_ID.value].apply(
            lambda x: str(x) if pd.notnull(x) else None,
        )
        transcripts_df[self.keys.CELL_ID.value] = transcripts_df[self.keys.CELL_ID.value].apply(
            lambda x: str(x) if pd.notnull(x) else None,
        )

        # Convert feature names from bytes to strings if necessary
        if pd.api.types.is_object_dtype(transcripts_df[self.keys.FEATURE_NAME.value]):
            transcripts_df[self.keys.FEATURE_NAME.value] = transcripts_df[self.keys.FEATURE_NAME.value].astype(str)

        # Apply dataset-specific filtering (e.g., quality filtering for Xenium)
        transcripts_df = self.filter_transcripts(transcripts_df)

        # Handle additional embeddings if provided
        if self.embedding_df is not None and not self.embedding_df.empty:
            valid_genes = self.embedding_df.index
            # Lazily count the number of rows in the DataFrame before filtering
            initial_count = delayed(lambda df: df.shape[0])(transcripts_df)
            # Filter the DataFrame lazily based on valid genes from embeddings
            transcripts_df = transcripts_df[transcripts_df[self.keys.FEATURE_NAME.value].isin(valid_genes)]
            final_count = delayed(lambda df: df.shape[0])(transcripts_df)
            if self.verbose:
                print(f"Dropped {initial_count - final_count} transcripts not found in embedding.")

        # Ensure that the 'OVERLAPS_BOUNDARY' column is boolean if it exists
        if self.keys.OVERLAPS_BOUNDARY.value in transcripts_df.columns:
            transcripts_df[self.keys.OVERLAPS_BOUNDARY.value] = transcripts_df[
                self.keys.OVERLAPS_BOUNDARY.value
            ].astype(bool)

        return transcripts_df

    def load_boundaries(
        self,
        path: Path,
        file_format: str = "parquet",
        x_min: float = None,
        x_max: float = None,
        y_min: float = None,
        y_max: float = None,
    ) -> dd.DataFrame:
        """
        Load boundaries data lazily using Dask, filtering by the specified bounding box.

        Parameters:
            path (Path): Path to the boundaries file.
            file_format (str, optional): Format of the file to load. Only 'parquet' is supported in this refactor.
            x_min (float, optional): Minimum X-coordinate for the bounding box.
            x_max (float, optional): Maximum X-coordinate for the bounding box.
            y_min (float, optional): Minimum Y-coordinate for the bounding box.
            y_max (float, optional): Maximum Y-coordinate for the bounding box.

        Returns:
            dd.DataFrame: The filtered boundaries DataFrame.
        """
        if file_format != "parquet":
            raise ValueError(f"Unsupported file format: {file_format}")

        self.boundaries_path = path

        # Use bounding box values from set_metadata if not explicitly provided
        x_min = x_min or self.x_min
        x_max = x_max or self.x_max
        y_min = y_min or self.y_min
        y_max = y_max or self.y_max

        # Define the list of columns to read
        columns_to_read = [
            self.keys.BOUNDARIES_VERTEX_X.value,
            self.keys.BOUNDARIES_VERTEX_Y.value,
            self.keys.CELL_ID.value,
        ]

        # Use filters to only load data within the specified bounding box (x_min, x_max, y_min, y_max)
        filters = [
            (self.keys.BOUNDARIES_VERTEX_X.value, ">=", x_min),
            (self.keys.BOUNDARIES_VERTEX_X.value, "<=", x_max),
            (self.keys.BOUNDARIES_VERTEX_Y.value, ">=", y_min),
            (self.keys.BOUNDARIES_VERTEX_Y.value, "<=", y_max),
        ]

        # Load the dataset lazily with filters applied for the bounding box
        columns = set(dd.read_parquet(path).columns)
        if "geometry" in columns:
            bbox = (x_min, y_min, x_max, y_max)
            # TODO: check that SpatialData objects write the "bbox covering metadata" to the parquet file
            gdf = dgpd.read_parquet(path, bbox=bbox)
            id_col, x_col, y_col = (
                self.keys.CELL_ID.value,
                self.keys.BOUNDARIES_VERTEX_X.value,
                self.keys.BOUNDARIES_VERTEX_Y.value,
            )

            # Function to expand each polygon into a list of vertices
            def expand_polygon(row):
                expanded_data = []
                polygon = row["geometry"]
                if polygon.geom_type == "Polygon":
                    exterior_coords = polygon.exterior.coords
                    for x, y in exterior_coords:
                        expanded_data.append({id_col: row.name, x_col: x, y_col: y})
                else:
                    # Instead of expanding the gdf and then having code later to recreate it (when computing the pyg graph)
                    # we could directly have this function returning a Dask GeoDataFrame. This means that we don't need
                    # to implement this else black
                    raise ValueError(f"Unsupported geometry type: {polygon.geom_type}")
                return expanded_data

            # Apply the function to each partition and collect results
            def process_partition(df):
                expanded_data = [expand_polygon(row) for _, row in df.iterrows()]
                # Flatten the list of lists
                flattened_data = [item for sublist in expanded_data for item in sublist]
                return pd.DataFrame(flattened_data)

            # Use map_partitions to apply the function and convert it into a Dask DataFrame
            boundaries_df = gdf.map_partitions(process_partition, meta={id_col: str, x_col: float, y_col: float})
        else:
            boundaries_df = dd.read_parquet(path, columns=columns_to_read, filters=filters)

            # Convert the cell IDs to strings lazily
            boundaries_df[self.keys.CELL_ID.value] = boundaries_df[self.keys.CELL_ID.value].apply(
                lambda x: str(x) if pd.notnull(x) else None, meta=("cell_id", "object")
            )

        if self.verbose:
            print(f"Loaded boundaries from '{path}' within bounding box ({x_min}, {x_max}, {y_min}, {y_max}).")

        return boundaries_df

    def set_metadata(self) -> None:
        """
        Set metadata for the transcript dataset, including bounding box limits and unique gene names,
        without reading the entire Parquet file. Additionally, return integer tokens for unique gene names
        instead of one-hot encodings and store the lookup table for later mapping.
        """
        # Load the Parquet file metadata
        parquet_file = pq.read_table(self.transcripts_path)

        # Get the column names for X, Y, and feature names from the class's keys
        x_col = self.keys.TRANSCRIPTS_X.value
        y_col = self.keys.TRANSCRIPTS_Y.value
        feature_col = self.keys.FEATURE_NAME.value

        # Initialize variables to track min/max values for X and Y
        x_min, x_max, y_min, y_max = float("inf"), float("-inf"), float("inf"), float("-inf")

        # Extract unique gene names and ensure they're strings
        gene_set = set()

        # Define the filter for unwanted codewords
        filter_codewords = (
            "NegControlProbe_",
            "antisense_",
            "NegControlCodeword_",
            "BLANK_",
            "DeprecatedCodeword_",
            "UnassignedCodeword_",
        )

        row_group_size = 4_000_000
        start = 0
        n = len(parquet_file)
        while start < n:
            chunk = parquet_file.slice(start, start + row_group_size)
            start += row_group_size

            # Update the bounding box values (min/max)
            x_values = chunk[x_col].to_pandas()
            y_values = chunk[y_col].to_pandas()

            x_min = min(x_min, x_values.min())
            x_max = max(x_max, x_values.max())
            y_min = min(y_min, y_values.min())
            y_max = max(y_max, y_values.max())

            # Convert feature values (gene names) to strings and filter out unwanted codewords
            feature_values = (
                chunk[feature_col]
                .to_pandas()
                .apply(
                    lambda x: x.decode("utf-8") if isinstance(x, bytes) else str(x),
                )
            )

            # Filter out unwanted codewords
            filtered_genes = feature_values[~feature_values.str.startswith(filter_codewords)]

            # Update the unique gene set
            gene_set.update(filtered_genes.unique())

        # Set bounding box limits
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

        if self.verbose:
            print(
                f"Bounding box limits set: x_min={self.x_min}, x_max={self.x_max}, y_min={self.y_min}, y_max={self.y_max}"
            )

        # Convert the set of unique genes into a sorted list for consistent ordering
        self.unique_genes = sorted(gene_set)
        if self.verbose:
            print(f"Extracted {len(self.unique_genes)} unique gene names for integer tokenization.")

        # Initialize a LabelEncoder to convert unique genes into integer tokens
        self.tx_encoder = LabelEncoder()

        # Fit the LabelEncoder on the unique genes
        self.tx_encoder.fit(self.unique_genes)

        # Store the integer tokens mapping to gene names
        self.gene_to_token_map = dict(
            zip(self.tx_encoder.classes_, self.tx_encoder.transform(self.tx_encoder.classes_))
        )

        if self.verbose:
            print("Integer tokens have been computed and stored based on unique gene names.")

        # Optional: Create a reverse mapping for lookup purposes (token to gene)
        self.token_to_gene_map = {v: k for k, v in self.gene_to_token_map.items()}

        if self.verbose:
            print("Lookup tables (gene_to_token_map and token_to_gene_map) have been created.")

    def set_embedding(self, embedding_name: str) -> None:
        """
        Set the current embedding type for the transcripts.

        Parameters:
            embedding_name : str
                The name of the embedding to use.

        """
        if embedding_name in self.embeddings_dict:
            self.current_embedding = embedding_name
        else:
            raise ValueError(f"Embedding {embedding_name} not found in embeddings_dict.")

    @staticmethod
    def create_scaled_polygon(group: pd.DataFrame, scale_factor: float, keys) -> gpd.GeoDataFrame:
        """
        Static method to create and scale a polygon from boundary vertices and return a GeoDataFrame.

        Parameters:
            group (pd.DataFrame): Group of boundary coordinates (for a specific cell).
            scale_factor (float): The factor by which to scale the polygons.
            keys (Dict or Enum): A collection of keys to access column names for 'cell_id', 'vertex_x', and 'vertex_y'.

        Returns:
            gpd.GeoDataFrame: A GeoDataFrame containing the scaled Polygon and cell_id.
        """
        # Extract coordinates and cell ID from the group using keys
        x_coords = group[keys["vertex_x"]]
        y_coords = group[keys["vertex_y"]]
        cell_id = group[keys["cell_id"]].iloc[0]

        # Ensure there are at least 3 points to form a polygon
        if len(x_coords) >= 3:

            polygon = Polygon(zip(x_coords, y_coords))
            if polygon.is_valid and not polygon.is_empty:
                # Scale the polygon by the provided factor
                scaled_polygon = polygon.buffer(scale_factor)
                if scaled_polygon.is_valid and not scaled_polygon.is_empty:
                    return gpd.GeoDataFrame(
                        {"geometry": [scaled_polygon], keys["cell_id"]: [cell_id]}, geometry="geometry", crs="EPSG:4326"
                    )
        # Return an empty GeoDataFrame if no valid polygon is created
        return gpd.GeoDataFrame({"geometry": [None], keys["cell_id"]: [cell_id]}, geometry="geometry", crs="EPSG:4326")

    def generate_and_scale_polygons(self, boundaries_df: dd.DataFrame, scale_factor: float = 1.0) -> dgpd.GeoDataFrame:
        """
        Generate and scale polygons from boundary coordinates using Dask.
        Keeps class structure intact by using static method for the core polygon generation.

        Parameters:
            boundaries_df (dd.DataFrame): DataFrame containing boundary coordinates.
            scale_factor (float, optional): The factor by which to scale the polygons (default is 1.0).

        Returns:
            dgpd.GeoDataFrame: A GeoDataFrame containing scaled Polygon objects and their centroids.
        """
        # if self.verbose: print(f"No precomputed polygons provided. Computing polygons from boundaries with a scale factor of {scale_factor}.")

        # Extract required columns from self.keys
        cell_id_column = self.keys.CELL_ID.value
        vertex_x_column = self.keys.BOUNDARIES_VERTEX_X.value
        vertex_y_column = self.keys.BOUNDARIES_VERTEX_Y.value

        create_polygon = self.create_scaled_polygon
        # Use a lambda to wrap the static method call and avoid passing the function object directly to Dask
        polygons_ddf = boundaries_df.groupby(cell_id_column).apply(
            lambda group: create_polygon(
                group=group,
                scale_factor=scale_factor,
                keys={  # Pass keys as a dict for the lambda function
                    "vertex_x": vertex_x_column,
                    "vertex_y": vertex_y_column,
                    "cell_id": cell_id_column,
                },
            )
        )

        # Lazily compute centroids for each polygon
        if self.verbose:
            print("Adding centroids to the polygons...")
        polygons_ddf["centroid_x"] = polygons_ddf.geometry.centroid.x
        polygons_ddf["centroid_y"] = polygons_ddf.geometry.centroid.y

        polygons_ddf = polygons_ddf.drop_duplicates()
        # polygons_ddf = polygons_ddf.to_crs("EPSG:3857")

        return polygons_ddf

    def compute_transcript_overlap_with_boundaries(
        self,
        transcripts_df: dd.DataFrame,
        boundaries_df: dd.DataFrame = None,
        polygons_gdf: dgpd.GeoDataFrame = None,
        scale_factor: float = 1.0,
    ) -> dd.DataFrame:
        """
        Computes the overlap of transcript locations with scaled boundary polygons
        and assigns corresponding cell IDs to the transcripts using Dask.

        Parameters:
            transcripts_df (dd.DataFrame): Dask DataFrame containing transcript data.
            boundaries_df (dd.DataFrame, optional): Dask DataFrame containing boundary data. Required if polygons_gdf is not provided.
            polygons_gdf (dgpd.GeoDataFrame, optional): Precomputed Dask GeoDataFrame containing boundary polygons. If None, will compute from boundaries_df.
            scale_factor (float, optional): The factor by which to scale the boundary polygons. Default is 1.0.

        Returns:
            dd.DataFrame: The updated DataFrame with overlap information and assigned cell IDs.
        """
        # Check if polygons_gdf is provided, otherwise compute from boundaries_df
        if polygons_gdf is None:
            if boundaries_df is None:
                raise ValueError("Both boundaries_df and polygons_gdf cannot be None. Provide at least one.")

            # Generate polygons from boundaries_df if polygons_gdf is None
            # if self.verbose: print(f"No precomputed polygons provided. Computing polygons from boundaries with a scale factor of {scale_factor}.")
            polygons_gdf = self.generate_and_scale_polygons(boundaries_df, scale_factor)

        if polygons_gdf.empty:
            raise ValueError("No valid polygons were generated from the boundaries.")
        else:
            if self.verbose:
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
        if self.verbose:
            print(f"Starting overlap computation for transcripts with the boundary polygons.")
        if isinstance(transcripts_df, pd.DataFrame):
            # luca: I found this bug here
            warnings.warn("BUG! This function expects Dask DataFrames, not Pandas DataFrames.")
            # if we want to really have the below working in parallel, we need to add n_partitions>1 here
            transcripts_df = dd.from_pandas(transcripts_df, npartitions=1)
            transcripts_df.compute().columns
        transcripts_df = transcripts_df.map_partitions(
            lambda df: df.assign(
                **{
                    self.keys.OVERLAPS_BOUNDARY.value: df.apply(
                        lambda row: delayed(check_overlap)(row, polygons_gdf)[0], axis=1
                    ),
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
            boundaries_df (dd.DataFrame, optional): The dataframe containing boundaries data. Required if polygons_gdf is not provided.
            polygons_gdf (dgpd.GeoDataFrame, optional): Precomputed Dask GeoDataFrame containing boundary polygons. If None, will compute from boundaries_df.
            scale_factor (float, optional): The factor by which to scale the polygons (default is 1.0).
            area (bool, optional): Whether to compute area.
            convexity (bool, optional): Whether to compute convexity.
            elongation (bool, optional): Whether to compute elongation.
            circularity (bool, optional): Whether to compute circularity.

        Returns:
            dgpd.GeoDataFrame: A GeoDataFrame containing computed geometries.
        """
        # Check if polygons_gdf is provided, otherwise compute from boundaries_df
        if polygons_gdf is None:
            if boundaries_df is None:
                raise ValueError("Both boundaries_df and polygons_gdf cannot be None. Provide at least one.")

            # Generate polygons from boundaries_df if polygons_gdf is None
            if self.verbose:
                print(
                    f"No precomputed polygons provided. Computing polygons from boundaries with a scale factor of {scale_factor}."
                )
            polygons_gdf = self.generate_and_scale_polygons(boundaries_df, scale_factor)

        # Check if the generated polygons_gdf is empty
        if polygons_gdf.shape[0] == 0:
            raise ValueError("No valid polygons were generated from the boundaries.")
        else:
            if self.verbose:
                print(f"Polygons are available. Proceeding with geometrical computations.")

        # Compute additional geometrical properties
        polygons = polygons_gdf.geometry

        # Compute additional geometrical properties
        if area:
            if self.verbose:
                print("Computing area...")
            polygons_gdf["area"] = polygons.area
        if convexity:
            if self.verbose:
                print("Computing convexity...")
            polygons_gdf["convexity"] = polygons.convex_hull.area / polygons.area
        if elongation:
            if self.verbose:
                print("Computing elongation...")
            r = polygons.minimum_rotated_rectangle()
            polygons_gdf["elongation"] = (r.length * r.length) / r.area
        if circularity:
            if self.verbose:
                print("Computing circularity...")
            r = polygons_gdf.minimum_bounding_radius()
            polygons_gdf["circularity"] = polygons.area / (r * r)

        if self.verbose:
            print("Geometrical computations completed.")

        return polygons_gdf.reset_index(drop=True)

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
        scale_boundaries: float = 1.0,
        method: str = "kd_tree",
        gpu: bool = False,
        workers: int = 1,
    ) -> None:
        """
        Saves the dataset for Segger in a processed format using Dask for parallel and lazy processing.

        Parameters:
            processed_dir (Path): Directory to save the processed dataset.
            x_size (float, optional): Width of each tile.
            y_size (float, optional): Height of each tile.
            d_x (float, optional): Step size in the x direction for tiles.
            d_y (float, optional): Step size in the y direction for tiles.
            margin_x (float, optional): Margin in the x direction to include transcripts.
            margin_y (float, optional): Margin in the y direction to include transcripts.
            compute_labels (bool, optional): Whether to compute edge labels for tx_belongs_bd edges.
            r_tx (float, optional): Radius for building the transcript-to-transcript graph.
            k_tx (int, optional): Number of nearest neighbors for the tx-tx graph.
            val_prob (float, optional): Probability of assigning a tile to the validation set.
            test_prob (float, optional): Probability of assigning a tile to the test set.
            neg_sampling_ratio_approx (float, optional): Approximate ratio of negative samples.
            sampling_rate (float, optional): Rate of sampling tiles.
            num_workers (int, optional): Number of workers to use for parallel processing.
            scale_boundaries (float, optional): The factor by which to scale the boundary polygons. Default is 1.0.
            method (str, optional): Method for computing edge indices (e.g., 'kd_tree', 'faiss').
            gpu (bool, optional): Whether to use GPU acceleration for edge index computation.
            workers (int, optional): Number of workers to use to compute the neighborhood graph (per tile).

        """
        # Prepare directories for storing processed tiles
        self._prepare_directories(processed_dir)

        # Get x and y coordinate ranges for tiling
        x_range, y_range = self._get_ranges(d_x, d_y)

        # Generate parameters for each tile
        tile_params = self._generate_tile_params(
            x_range,
            y_range,
            x_size,
            y_size,
            margin_x,
            margin_y,
            compute_labels,
            r_tx,
            k_tx,
            val_prob,
            test_prob,
            neg_sampling_ratio_approx,
            sampling_rate,
            processed_dir,
            scale_boundaries,
            method,
            gpu,
            workers,
        )

        # Process each tile using Dask to parallelize the task
        if self.verbose:
            print("Starting tile processing...")
        tasks = [delayed(self._process_tile)(params) for params in tile_params]

        with ProgressBar():
            # Use Dask to process all tiles in parallel
            dask.compute(*tasks, num_workers=num_workers)
        if self.verbose:
            print("Tile processing completed.")

    def _prepare_directories(self, processed_dir: Path) -> None:
        """Prepares directories for saving tiles."""
        processed_dir = Path(processed_dir)  # by default, convert to Path object
        for data_type in ["train", "test", "val"]:
            for data_stage in ["raw", "processed"]:
                tile_dir = processed_dir / f"{data_type}_tiles" / data_stage
                tile_dir.mkdir(parents=True, exist_ok=True)
                if os.listdir(tile_dir):
                    msg = f"Directory '{tile_dir}' must be empty."
                    raise AssertionError(msg)

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
        scale_boundaries: float,
        method: str,
        gpu: bool,
        workers: int,
    ) -> List[Tuple]:
        """
        Generates parameters for processing tiles using the bounding box approach.
        This version eliminates masks and directly uses the tile ranges and margins.

        Parameters are the same as the previous version.
        """
        margin_x = margin_x if margin_x is not None else x_size // 10
        margin_y = margin_y if margin_y is not None else y_size // 10

        # Generate tile parameters based on ranges and margins
        tile_params = [
            (
                i,
                j,
                x_size,
                y_size,
                x_range[i],
                y_range[j],
                margin_x,
                margin_y,
                compute_labels,
                r_tx,
                k_tx,
                neg_sampling_ratio_approx,
                val_prob,
                test_prob,
                processed_dir,
                scale_boundaries,
                sampling_rate,
                method,
                gpu,
                workers,
            )
            for i in range(len(x_range))
            for j in range(len(y_range))
        ]
        return tile_params

    # def _process_tiles(self, tile_params: List[Tuple], num_workers: int) -> None:
    #     """
    #     Processes the tiles using Dask's parallelization utilities.

    #     Parameters:
    #     -----------
    #     tile_params : List[Tuple]
    #         List of parameters for each tile to be processed.
    #     num_workers : int
    #         Number of workers to use for parallel processing.
    #     """
    #     if self.verbose: print("Starting parallel tile processing...")

    #     # Create a list of delayed tasks for each tile
    #     tasks = [delayed(self._process_tile)(params) for params in tile_params]

    #     # Execute tasks using Dask's compute with specified number of workers
    #     with ProgressBar():
    #         dask.compute(*tasks, num_workers=num_workers)

    #     if self.verbose: print("Tile processing completed.")

    def _process_tile(self, tile_params: Tuple) -> None:
        """
        Process a single tile using Dask for parallelism and lazy evaluation, and save the data.

        Parameters:
            tile_params : tuple
                Parameters for the tile processing.
        """
        (
            i,
            j,
            x_size,
            y_size,
            x_loc,
            y_loc,
            margin_x,
            margin_y,
            compute_labels,
            r_tx,
            k_tx,
            neg_sampling_ratio_approx,
            val_prob,
            test_prob,
            processed_dir,
            scale_boundaries,
            sampling_rate,
            method,
            gpu,
            workers,
        ) = tile_params

        if self.verbose:
            print(
                f"Processing tile at location (x_min: {x_loc}, y_min: {y_loc}), size (width: {x_size}, height: {y_size})"
            )

        # Sampling rate to decide if the tile should be processed
        if random.random() > sampling_rate:
            if self.verbose:
                print(f"Skipping tile at (x_min: {x_loc}, y_min: {y_loc}) due to sampling rate.")
            return

        # Read only the required boundaries and transcripts for this tile using delayed loading
        boundaries_df = delayed(self.load_boundaries)(
            path=self.boundaries_path,
            x_min=x_loc - margin_x,
            x_max=x_loc + x_size + margin_x,
            y_min=y_loc - margin_y,
            y_max=y_loc + y_size + margin_y,
        ).compute()

        transcripts_df = delayed(self.load_transcripts)(
            path=self.transcripts_path,
            x_min=x_loc - margin_x,
            x_max=x_loc + x_size,
            y_min=y_loc - margin_y,
            y_max=y_loc + y_size,
        ).compute()

        # If no data is found in transcripts or boundaries, skip the tile
        boundaries_df_count = boundaries_df.compute().shape[0]
        transcripts_df_count = transcripts_df.shape[0]
        # if self.verbose: print(boundaries_df_count)
        # if self.verbose: print(transcripts_df_count)

        # If the number of transcripts is less than 20 or the number of nuclei is less than 2, skip the tile
        if transcripts_df_count < 20 or boundaries_df_count < 2:
            if self.verbose:
                print(
                    f"Dropping tile (x_min: {x_loc}, y_min: {y_loc}) due to insufficient data (transcripts: {transcripts_df_count}, boundaries: {boundaries_df_count})."
                )
            return

        # Build PyG data structure from tile-specific data
        if self.verbose:
            print(f"Building PyG data for tile at (x_min: {x_loc}, y_min: {y_loc})...")
        data = delayed(self.build_pyg_data_from_tile)(
            boundaries_df,
            transcripts_df,
            r_tx=r_tx,
            k_tx=k_tx,
            method=method,
            gpu=gpu,
            workers=workers,
            scale_boundaries=scale_boundaries,
        )

        data = data.compute()
        if self.verbose:
            print(data)

        try:
            # Probability to assign to train-val-test split
            prob = random.random()
            if compute_labels and (prob > test_prob):
                if self.verbose:
                    print(f"Computing labels for tile at (x_min: {x_loc}, y_min: {y_loc})...")
                transform = RandomLinkSplit(
                    num_val=0,
                    num_test=0,
                    is_undirected=True,
                    edge_types=[("tx", "belongs", "bd")],
                    neg_sampling_ratio=neg_sampling_ratio_approx * 2,
                )
                data = delayed(transform)(data).compute()[0]

            # if self.verbose: print(data)

            # Save the tile data to the appropriate directory based on split
            if self.verbose:
                print(f"Saving data for tile at (x_min: {x_loc}, y_min: {y_loc})...")
            filename = f"tiles_x={x_loc}_y={y_loc}_w={x_size}_h={y_size}.pt"
            if prob > val_prob + test_prob:
                torch.save(data, processed_dir / "train_tiles" / "processed" / filename)
            elif prob > test_prob:
                torch.save(data, processed_dir / "val_tiles" / "processed" / filename)
            else:
                torch.save(data, processed_dir / "test_tiles" / "processed" / filename)

            # Use Dask to save the file in parallel
            # save_task.compute()

            if self.verbose:
                print(f"Tile at (x_min: {x_loc}, y_min: {y_loc}) processed and saved successfully.")

        except Exception as e:
            if self.verbose:
                print(f"Error processing tile at (x_min: {x_loc}, y_min: {y_loc}): {e}")

    def build_pyg_data_from_tile(
        self,
        boundaries_df: dd.DataFrame,
        transcripts_df: dd.DataFrame,
        r_tx: float = 5.0,
        k_tx: int = 3,
        method: str = "kd_tree",
        gpu: bool = False,
        workers: int = 1,
        scale_boundaries: float = 1.0,
    ) -> HeteroData:
        """
        Builds PyG data from a tile of boundaries and transcripts data using Dask utilities for efficient processing.

        Parameters:
            boundaries_df (dd.DataFrame): Dask DataFrame containing boundaries data (e.g., nucleus, cell).
            transcripts_df (dd.DataFrame): Dask DataFrame containing transcripts data.
            r_tx (float): Radius for building the transcript-to-transcript graph.
            k_tx (int): Number of nearest neighbors for the tx-tx graph.
            method (str, optional): Method for computing edge indices (e.g., 'kd_tree', 'faiss').
            gpu (bool, optional): Whether to use GPU acceleration for edge index computation.
            workers (int, optional): Number of workers to use for parallel processing.
            scale_boundaries (float, optional): The factor by which to scale the boundary polygons. Default is 1.0.

        Returns:
            HeteroData: PyG Heterogeneous Data object.
        """
        # Initialize the PyG HeteroData object
        data = HeteroData()

        # Lazily compute boundaries geometries using Dask
        if self.verbose:
            print("Computing boundaries geometries...")
        bd_gdf = self.compute_boundaries_geometries(boundaries_df, scale_factor=scale_boundaries)
        bd_gdf = bd_gdf[bd_gdf["geometry"].notnull()]

        # Add boundary node data to PyG HeteroData lazily
        data["bd"].id = bd_gdf[self.keys.CELL_ID.value].values
        data["bd"].pos = torch.as_tensor(bd_gdf[["centroid_x", "centroid_y"]].values.astype(float))

        if data["bd"].pos.isnan().any():
            raise ValueError(data["bd"].id[data["bd"].pos.isnan().any(1)])

        bd_x = bd_gdf.iloc[:, 4:]
        data["bd"].x = torch.as_tensor(bd_x.to_numpy(), dtype=torch.float32)

        # Extract the transcript coordinates lazily
        if self.verbose:
            print("Preparing transcript features and positions...")
        x_xyz = transcripts_df[[self.keys.TRANSCRIPTS_X.value, self.keys.TRANSCRIPTS_Y.value]].to_numpy()
        data["tx"].id = torch.as_tensor(transcripts_df[self.keys.TRANSCRIPTS_ID.value].values.astype(int))
        data["tx"].pos = torch.tensor(x_xyz, dtype=torch.float32)

        # Lazily prepare transcript embeddings (if available)
        if self.verbose:
            print("Preparing transcript embeddings..")
        token_encoding = self.tx_encoder.transform(transcripts_df[self.keys.FEATURE_NAME.value])
        transcripts_df["token"] = token_encoding  # Store the integer tokens in the 'token' column
        data["tx"].token = torch.as_tensor(token_encoding).int()
        # Handle additional embeddings lazily as well
        if self.embedding_df is not None and not self.embedding_df.empty:
            embeddings = delayed(lambda df: self.embedding_df.loc[df[self.keys.FEATURE_NAME.value].values].values)(
                transcripts_df
            )
        else:
            embeddings = token_encoding
        if hasattr(embeddings, "compute"):
            embeddings = embeddings.compute()
        x_features = torch.as_tensor(embeddings).int()
        data["tx"].x = x_features

        # Check if the overlap column exists, if not, compute it lazily using Dask
        if self.keys.OVERLAPS_BOUNDARY.value not in transcripts_df.columns:
            if self.verbose:
                print(f"Computing overlaps for transcripts...")
            transcripts_df = self.compute_transcript_overlap_with_boundaries(
                transcripts_df, polygons_gdf=bd_gdf, scale_factor=1.0
            )

        # Connect transcripts with their corresponding boundaries (e.g., nuclei, cells)
        if self.verbose:
            print("Connecting transcripts with boundaries...")
        overlaps = transcripts_df[self.keys.OVERLAPS_BOUNDARY.value].values
        valid_cell_ids = bd_gdf[self.keys.CELL_ID.value].values
        ind = np.where(overlaps & transcripts_df[self.keys.CELL_ID.value].isin(valid_cell_ids))[0]
        tx_bd_edge_index = np.column_stack(
            (ind, np.searchsorted(valid_cell_ids, transcripts_df.iloc[ind][self.keys.CELL_ID.value]))
        )

        # Add transcript-boundary edge index to PyG HeteroData
        data["tx", "belongs", "bd"].edge_index = torch.as_tensor(tx_bd_edge_index.T, dtype=torch.long)

        # Compute transcript-to-transcript (tx-tx) edges using Dask (lazy computation)
        if self.verbose:
            print("Computing tx-tx edges...")
        tx_positions = transcripts_df[[self.keys.TRANSCRIPTS_X.value, self.keys.TRANSCRIPTS_Y.value]].values
        delayed_tx_edge_index = delayed(get_edge_index)(
            tx_positions, tx_positions, k=k_tx, dist=r_tx, method=method, gpu=gpu, workers=workers
        )
        tx_edge_index = delayed_tx_edge_index.compute()

        # Add the tx-tx edge index to the PyG HeteroData object
        data["tx", "neighbors", "tx"].edge_index = torch.as_tensor(tx_edge_index.T, dtype=torch.long)

        if self.verbose:
            print("Finished building PyG data for the tile.")
        return data


class XeniumSample(SpatialTranscriptomicsSample):
    def __init__(
        self,
        transcripts_df: dd.DataFrame = None,
        transcripts_radius: int = 10,
        boundaries_graph: bool = False,
        embedding_df: pd.DataFrame = None,
        verbose: bool = True,
    ):
        super().__init__(
            transcripts_df, transcripts_radius, boundaries_graph, embedding_df, XeniumKeys, verbose=verbose
        )

    def filter_transcripts(self, transcripts_df: dd.DataFrame, min_qv: float = 20.0) -> dd.DataFrame:
        """
        Filters transcripts based on quality value and removes unwanted transcripts for Xenium using Dask.

        Parameters:
            transcripts_df (dd.DataFrame): The Dask DataFrame containing transcript data.
            min_qv (float, optional): The minimum quality value threshold for filtering transcripts.

        Returns:
            dd.DataFrame: The filtered Dask DataFrame.
        """
        filter_codewords = (
            "NegControlProbe_",
            "antisense_",
            "NegControlCodeword_",
            "BLANK_",
            "DeprecatedCodeword_",
            "UnassignedCodeword_",
        )

        # Ensure FEATURE_NAME is a string type for proper filtering (compatible with Dask)
        # Handle potential bytes to string conversion for Dask DataFrame
        if pd.api.types.is_object_dtype(transcripts_df[self.keys.FEATURE_NAME.value]):
            transcripts_df[self.keys.FEATURE_NAME.value] = transcripts_df[self.keys.FEATURE_NAME.value].apply(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
            )

        # Apply the quality value filter using Dask
        mask_quality = transcripts_df[self.keys.QUALITY_VALUE.value] >= min_qv

        # Apply the filter for unwanted codewords using Dask string functions
        mask_codewords = ~transcripts_df[self.keys.FEATURE_NAME.value].str.startswith(filter_codewords)

        # Combine the filters and return the filtered Dask DataFrame
        mask = mask_quality & mask_codewords

        # Return the filtered DataFrame lazily
        return transcripts_df[mask]


class MerscopeSample(SpatialTranscriptomicsSample):
    def __init__(
        self,
        transcripts_df: dd.DataFrame = None,
        transcripts_radius: int = 10,
        boundaries_graph: bool = False,
        embedding_df: pd.DataFrame = None,
        verbose: bool = True,
    ):
        super().__init__(transcripts_df, transcripts_radius, boundaries_graph, embedding_df, MerscopeKeys)

    def filter_transcripts(self, transcripts_df: dd.DataFrame, min_qv: float = 20.0) -> dd.DataFrame:
        """
        Filters transcripts based on specific criteria for Merscope using Dask.

        Parameters:
            transcripts_df : dd.DataFrame
                The Dask DataFrame containing transcript data.
            min_qv : float, optional
                The minimum quality value threshold for filtering transcripts.

        Returns:
            dd.DataFrame
                The filtered Dask DataFrame.
        """
        # Add custom Merscope-specific filtering logic if needed
        # For now, apply only the quality value filter
        return transcripts_df[transcripts_df[self.keys.QUALITY_VALUE.value] >= min_qv]


class SpatialDataSample(SpatialTranscriptomicsSample):
    def __init__(
        self,
        transcripts_df: dd.DataFrame = None,
        transcripts_radius: int = 10,
        boundaries_graph: bool = False,
        embedding_df: pd.DataFrame = None,
        feature_name: str | None = None,
        verbose: bool = True,
    ):
        if feature_name is not None:
            # luca: just a quick hack for now, I propose to use dataclasses instead of enums to address this
            SpatialDataKeys.FEATURE_NAME._value_ = feature_name
        else:
            raise ValueError(
                "the automatic determination of a feature_name from a SpatialData object is not enabled yet"
            )

        super().__init__(
            transcripts_df, transcripts_radius, boundaries_graph, embedding_df, SpatialDataKeys, verbose=verbose
        )

    def filter_transcripts(self, transcripts_df: dd.DataFrame, min_qv: float = 20.0) -> dd.DataFrame:
        """
        Filters transcripts based on quality value and removes unwanted transcripts for Xenium using Dask.

        Parameters:
            transcripts_df (dd.DataFrame): The Dask DataFrame containing transcript data.
            min_qv (float, optional): The minimum quality value threshold for filtering transcripts.

        Returns:
            dd.DataFrame: The filtered Dask DataFrame.
        """
        filter_codewords = (
            "NegControlProbe_",
            "antisense_",
            "NegControlCodeword_",
            "BLANK_",
            "DeprecatedCodeword_",
            "UnassignedCodeword_",
        )

        # Ensure FEATURE_NAME is a string type for proper filtering (compatible with Dask)
        # Handle potential bytes to string conversion for Dask DataFrame
        if pd.api.types.is_object_dtype(transcripts_df[self.keys.FEATURE_NAME.value]):
            transcripts_df[self.keys.FEATURE_NAME.value] = transcripts_df[self.keys.FEATURE_NAME.value].apply(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
            )

        # Apply the quality value filter using Dask
        mask_quality = transcripts_df[self.keys.QUALITY_VALUE.value] >= min_qv

        # Apply the filter for unwanted codewords using Dask string functions
        mask_codewords = ~transcripts_df[self.keys.FEATURE_NAME.value].str.startswith(filter_codewords)

        # Combine the filters and return the filtered Dask DataFrame
        mask = mask_quality & mask_codewords

        # Return the filtered DataFrame lazily
        return transcripts_df[mask]
