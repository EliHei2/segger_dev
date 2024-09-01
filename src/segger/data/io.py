from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData
from sklearn.preprocessing import OneHotEncoder
from scipy.spatial import KDTree, ConvexHull
from shapely.geometry import Polygon
import geopandas as gpd
import random
from itertools import product
from multiprocessing import Pool
from tqdm import tqdm
from segger.data.utils import *
from segger.data.constants import *
from enum import Enum

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

        if self.transcripts_df is not None:
            self._set_bounds()

    def _set_bounds(self) -> None:
        """Set the bounding box limits based on the current transcripts dataframe."""
        self.x_max = self.transcripts_df[self.keys.TRANSCRIPTS_X.value].max()
        self.y_max = self.transcripts_df[self.keys.TRANSCRIPTS_Y.value].max()
        self.x_min = self.transcripts_df[self.keys.TRANSCRIPTS_X.value].min()
        self.y_min = self.transcripts_df[self.keys.TRANSCRIPTS_Y.value].min()

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
        file_format: str = "csv",
        additional_embeddings: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> None:
        """
        Load transcripts from a file, supporting both gzip-compressed CSV and Parquet formats,
        and allows the inclusion of additional gene embeddings.

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
            Format of the file to load. Options are 'csv' or 'parquet' (default is 'csv').
        additional_embeddings : Dict[str, pd.DataFrame], optional
            A dictionary of additional embeddings for genes.

        Returns:
        --------
        None
        """
        transcripts_filename = transcripts_filename or self.keys.TRANSCRIPTS_FILE.value
        file_path = path or (base_path / sample / transcripts_filename)
        if file_format == "csv":
            self.transcripts_df = pd.read_csv(io.BytesIO(file.read()))
        elif file_format == "parquet":
            self.transcripts_df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        print(f"Loaded {len(self.transcripts_df)} transcripts for sample '{sample}'.")
        self.transcripts_df = self.filter_transcripts(self.transcripts_df, min_qv=min_qv)
        
        if additional_embeddings:
            for key, embedding_df in additional_embeddings.items():
                valid_genes = embedding_df.index
                initial_count = len(self.transcripts_df)
                self.transcripts_df = self.transcripts_df[
                    self.transcripts_df[self.keys.FEATURE_NAME.value].isin(valid_genes)
                ]
                final_count = len(self.transcripts_df)
                print(
                    f"Dropped {initial_count - final_count} transcripts not found in {key} embedding."
                )


        self._set_bounds()

        # Encode genes as one-hot by default
        genes = self.transcripts_df[[self.keys.FEATURE_NAME.value]]
        self.tx_encoder = OneHotEncoder(sparse_output=False)
        self.tx_encoder.fit(genes)
        self.embeddings_dict['one_hot'] = self.tx_encoder.transform(genes)
        self.current_embedding = 'one_hot'

        # Add additional embeddings if provided
        if additional_embeddings:
            for key, embedding_df in additional_embeddings.items():
                self.embeddings_dict[key] = embedding_df.loc[
                    self.transcripts_df[self.keys.FEATURE_NAME.value].values
                ].values
                
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


    def load_boundaries(self, path: Path, file_format: str = "csv") -> 'SpatialTranscriptomicsSample':
        """
        Load boundaries data from a file, supporting both gzip-compressed CSV and Parquet formats.

        Parameters:
        -----------
        path : Path
            Path to the boundaries file.
        file_format : str, optional
            Format of the file to load. Options are 'csv' or 'parquet' (default is 'csv').

        Returns:
        --------
        SpatialTranscriptomicsSample
            The updated instance with loaded boundaries data.
        """
        if file_format == "csv":
            self.boundaries_df = pd.read_csv(path)
        elif file_format == "parquet":
            self.boundaries_df = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        

    # def crop_transcripts(
    #     self, x: float, y: float, x_size: int = 1000, y_size: int = 1000
    # ) -> 'SpatialTranscriptomicsSample':
    #     """
    #     Crop the transcripts to a specific area.

    #     Parameters:
    #     -----------
    #     x : float
    #         The x-coordinate of the top-left corner of the crop.
    #     y : float
    #         The y-coordinate of the top-left corner of the crop.
    #     x_size : int, optional
    #         The width of the crop area.
    #     y_size : int, optional
    #         The height of the crop area.

    #     Returns:
    #     --------
    #     SpatialTranscriptomicsSample
    #         A new instance of the class with the cropped transcripts.
    #     """
    #     cropped_transcripts_df = self.transcripts_df[
    #         (self.transcripts_df[self.keys.TRANSCRIPTS_X.value] > x)
    #         & (self.transcripts_df[self.keys.TRANSCRIPTS_X.value] < x + x_size)
    #         & (self.transcripts_df[self.keys.TRANSCRIPTS_Y.value] > y)
    #         & (self.transcripts_df[self.keys.TRANSCRIPTS_Y.value] < y + y_size)
    #     ]
    # .__class__(cropped_transcripts_df, self.transcripts_radius, self.boundaries_graph, self.keys)


    def get_bounding_box(
        self,
        x_min: float = None,
        y_min: float = None,
        x_max: float = None,
        y_max: float = None,
        in_place: bool = True,
    ) -> Optional['SpatialTranscriptomicsSample']:
        """
        Subsets the transcripts_df and boundaries_df within the specified bounding box.

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

        # Print initial counts
        initial_transcripts_count = len(self.transcripts_df)
        initial_boundaries_count = len(self.boundaries_df) if self.boundaries_df is not None else 0

        x_min, y_min, x_max, y_max = self._validate_bbox(x_min, y_min, x_max, y_max)

        # Subset transcripts_df
        subset_transcripts_df = self.transcripts_df[
            (self.transcripts_df[self.keys.TRANSCRIPTS_X.value] >= x_min)
            & (self.transcripts_df[self.keys.TRANSCRIPTS_X.value] <= x_max)
            & (self.transcripts_df[self.keys.TRANSCRIPTS_Y.value] >= y_min)
            & (self.transcripts_df[self.keys.TRANSCRIPTS_Y.value] <= y_max)
        ]

        # Subset boundaries_df if it exists
        subset_boundaries_df = None
        if self.boundaries_df is not None:
            subset_boundaries_df = self.boundaries_df[
                (self.boundaries_df[self.keys.BOUNDARIES_VERTEX_X.value] >= x_min)
                & (self.boundaries_df[self.keys.BOUNDARIES_VERTEX_X.value] <= x_max)
                & (self.boundaries_df[self.keys.BOUNDARIES_VERTEX_Y.value] >= y_min)
                & (self.boundaries_df[self.keys.BOUNDARIES_VERTEX_Y.value] <= y_max)
            ]

        # Calculate new counts
        final_transcripts_count = len(subset_transcripts_df)
        final_boundaries_count = len(subset_boundaries_df) if subset_boundaries_df is not None else 0

        if in_place:
            self.transcripts_df = subset_transcripts_df
            self.boundaries_df = subset_boundaries_df
            self.x_min, self.y_min, self.x_max, self.y_max = x_min, y_min, x_max, y_max

            # Print summary
            print(f"Subset completed in-place: {final_transcripts_count} transcripts out of {initial_transcripts_count} and "
                f"{final_boundaries_count} boundaries (cells or nuclei) out of {initial_boundaries_count}.")
        else:
            subset_sample = self.__class__(subset_transcripts_df, self.transcripts_radius, self.boundaries_graph, self.keys)
            subset_sample.boundaries_df = subset_boundaries_df
            subset_sample.x_min, subset_sample.y_min, subset_sample.x_max, subset_sample.y_max = x_min, y_min, x_max, y_max

            # Print summary
            print(f"Subset completed with new instance: {final_transcripts_count} transcripts out of {initial_transcripts_count} and "
                f"{final_boundaries_count} boundaries (cells or nuclei) out of {initial_boundaries_count}.")
            
            return subset_sample

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
        Saves the dataset for Segger in a processed format.

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
        self._prepare_directories(processed_dir)
        x_range, y_range = self._get_ranges(d_x, d_y)

        tile_params = self._generate_tile_params(
            x_range, y_range, x_size, y_size, margin_x, margin_y, compute_labels, r_tx, val_prob, 
            test_prob, neg_sampling_ratio_approx, sampling_rate, processed_dir, receptive_field, method, gpu, workers
        )
        
        self._process_tiles(tile_params, num_workers)

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
        val_prob: float,
        test_prob: float,
        neg_sampling_ratio_approx: float,
        sampling_rate: float,
        processed_dir: Path,
        receptive_field: Dict[str, float],
        method: str,
        gpu: bool,
        workers: int
    ) -> List[Tuple]:
        """Generates parameters for processing tiles."""
        margin_x = margin_x if margin_x is not None else d_x // 10
        margin_y = margin_y if margin_y is not None else d_y // 10

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
                compute_labels, r_tx, neg_sampling_ratio_approx, val_prob, test_prob, 
                processed_dir, receptive_field, sampling_rate, method, gpu, workers
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
        Process a single tile and save the data.

        Parameters:
        -----------
        tile_params : tuple
            Parameters for the tile processing.

        Returns:
        --------
        None
        """
        (
            i, j, x_masks_bd, y_masks_bd, x_masks_tx, y_masks_tx, x_size, y_size, x_loc, y_loc, compute_labels, r_tx, 
            neg_sampling_ratio_approx, val_prob, test_prob, processed_dir, receptive_field, sampling_rate, method, gpu, workers
        ) = tile_params

        if random.random() > sampling_rate:
            return

        x_mask = x_masks_bd[i]
        y_mask = y_masks_bd[j]
        mask = x_mask & y_mask

        if mask.sum() == 0:
            return

        bd_df = self.boundaries_df[mask]
        bd_df = self.boundaries_df.loc[self.boundaries_df.cell_id.isin(bd_df.cell_id), :]
        tx_mask = x_masks_tx[i] & y_masks_tx[j]
        transcripts_df = self.transcripts_df[tx_mask]

        if transcripts_df.shape[0] == 0 or bd_df.shape[0] == 0:
            return

        data = self.build_pyg_data_from_tile(
            bd_df, transcripts_df, compute_labels=compute_labels, method=method, gpu=gpu,
            workers=workers
        )
        data = BuildTxGraph(r=r_tx)(data)

        try:
            # probability to assign to train-val-test split
            prob = random.random() 
            if compute_labels and (prob > val_prob + test_prob):
                transform = RandomLinkSplit(
                    num_val=0, num_test=0, is_undirected=True, edge_types=[('tx', 'belongs', 'bd')],
                    neg_sampling_ratio=neg_sampling_ratio_approx * 2,
                )
                data, _, _ = transform(data)
                edge_index = data[('tx', 'belongs', 'bd')].edge_index
                if edge_index.shape[1] < 10:
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

            coords_bd = data['bd'].pos
            coords_tx = data['tx'].pos[:, :2]
            data['tx'].tx_field = get_edge_index(
                coords_tx, coords_tx, k=receptive_field["k_tx"], dist=receptive_field["dist_tx"], method=method,
                gpu=gpu,
                workers=workers
            )
            data['tx'].bd_field = get_edge_index(
                coords_bd, coords_tx, k=receptive_field["k_bd"], dist=receptive_field["dist_bd"], method=method,
                gpu=gpu,
                workers=workers
            )

            filename = f"tiles_x{int(x_loc)}_y{int(y_loc)}_{x_size}_{y_size}.pt"
            # prob = random.random()
            if prob > val_prob + test_prob:
                torch.save(data, processed_dir / 'train_tiles' / 'processed' / filename)
            elif prob > val_prob:
                torch.save(data, processed_dir / 'test_tiles' / 'processed' / filename)
            else:
                torch.save(data, processed_dir / 'val_tiles' / 'processed' / filename)
        except Exception as e:
            print(f"Error processing tile {i}, {j}: {e}")

    def compute_boundaries_geometries(
        self,
        boundaries_df: pd.DataFrame,
        area: bool = True,
        convexity: bool = True,
        elongation: bool = True,
        circularity: bool = True,
    ) -> gpd.GeoDataFrame:
        """
        Computes geometries for boundaries (e.g., nuclei, cells) from the dataframe.

        Parameters:
        -----------
        boundaries_df : pd.DataFrame
            The dataframe containing boundaries data.
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
        gpd.GeoDataFrame
            A geodataframe containing computed geometries.
        """
        grouped = boundaries_df.groupby(self.keys.CELL_ID.value)
        # Counter for skipped cells
        skipped_count = 0

        # List to store valid polygons
        polygons = []
        valid_cells = []

        for cell_id, group_data in grouped:
            x_coords = group_data[self.keys.BOUNDARIES_VERTEX_X.value]
            y_coords = group_data[self.keys.BOUNDARIES_VERTEX_Y.value]
            
            if len(x_coords) >= 4 and len(x_coords) == len(y_coords):
                coords = list(zip(x_coords, y_coords))
                # Ensure the first and last coordinates are the same
                if coords[0] != coords[-1]:
                    coords.append(coords[0])
                polygons.append(Polygon(coords))
                valid_cells.append(cell_id)
            else:
                skipped_count += 1

        # Print the number of skipped cells
        print(f"Skipped {skipped_count} out of {len(grouped)} cells due to insufficient or invalid coordinates, probably due to edge cases.")
        polygons = gpd.GeoSeries(polygons)
        gdf = gpd.GeoDataFrame(geometry=polygons, crs='EPSG:4326')
        gdf[self.keys.CELL_ID.value] = valid_cells
        gdf[['x', 'y']] = gdf.centroid.get_coordinates()
        if area:
            gdf['area'] = polygons.area
        if convexity:
            gdf['convexity'] = polygons.convex_hull.area / polygons.area
        if elongation:
            r = polygons.minimum_rotated_rectangle()
            gdf['elongation'] = (r.length * r.length) / r.area
        if circularity:
            r = gdf.minimum_bounding_radius()
            gdf['circularity'] = polygons.area / (r * r)
        return gdf

    def build_pyg_data_from_tile(
        self, boundaries_df: pd.DataFrame, transcripts_df: pd.DataFrame, compute_labels: bool = True, method: str = 'kd_tree', gpu: bool = False, workers: int = 1
    ) -> HeteroData:
        """
        Builds PyG data from a tile of boundaries and transcripts data.

        Parameters:
        -----------
        boundaries_df : pd.DataFrame
            Dataframe containing boundaries data (e.g., nucleus, cell).
        transcripts_df : pd.DataFrame
            Dataframe containing transcripts data.
        compute_labels : bool, optional
            Whether to compute labels.
        method : str, optional
            Method for computing edge indices (e.g., 'kd_tree', 'faiss').
        gpu : bool, optional
            Whether to use GPU acceleration for edge index computation.

        Returns:
        --------
        HeteroData
            PyG Heterogeneous Data object.
        """
        data = HeteroData()

        # Compute boundaries geometries
        bd_gdf = self.compute_boundaries_geometries(boundaries_df)
        max_area = bd_gdf['area'].max()

        # Prepare transcript features and positions
        x_xyz = torch.as_tensor(transcripts_df[[self.keys.TRANSCRIPTS_X.value, self.keys.TRANSCRIPTS_Y.value, self.keys.TRANSCRIPTS_Y.value]].values).float()
        data['tx'].id = transcripts_df.index.values
        data['tx'].pos = x_xyz

        x_features = torch.as_tensor(self.embeddings_dict[self.current_embedding]).float()
        data['tx'].x = x_features.to_sparse()

        # Compute edge indices using the chosen method
        tx_edge_index = get_edge_index(
            bd_gdf[['x', 'y']].values,
            transcripts_df[[self.keys.TRANSCRIPTS_X.value, self.keys.TRANSCRIPTS_Y.value]].values,
            k=3,
            dist=np.sqrt(max_area) * 10,
            method=method,
            gpu=gpu,
            workers=workers
        )
        data['tx', 'neighbors', 'bd'].edge_index = tx_edge_index

        # Connect transcripts with their corresponding boundaries (e.g., nuclei, cells)
        ind = np.where(
            (transcripts_df[self.keys.OVERLAPS_BOUNDARY.value] == 1) & (transcripts_df[self.keys.CELL_ID.value].isin(bd_gdf[self.keys.CELL_ID.value]))
        )[0]
        tx_bd_edge_index = np.column_stack(
            (ind, np.searchsorted(bd_gdf[self.keys.CELL_ID.value].values, transcripts_df.iloc[ind][self.keys.CELL_ID.value].values))
        )
        data['bd'].id = bd_gdf[[self.keys.CELL_ID.value]].values
        data['bd'].pos = bd_gdf[['x', 'y']].values
        bd_x = bd_gdf.iloc[:, 4:]
        data['bd'].x = torch.as_tensor(bd_x.to_numpy()).float()
        data['tx', 'belongs', 'bd'].edge_index = torch.as_tensor(tx_bd_edge_index.T).long()

        return data


class XeniumSample(SpatialTranscriptomicsSample):
    def __init__(self, transcripts_df: pd.DataFrame = None, transcripts_radius: int = 10, boundaries_graph: bool = False):
        super().__init__(transcripts_df, transcripts_radius, boundaries_graph, XeniumKeys)

    def filter_transcripts(self, transcripts_df: pd.DataFrame, min_qv: float = 20.0) -> pd.DataFrame:
        """
        Filters transcripts based on quality value and removes unwanted transcripts for Xenium.

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
        filter_codewords = (
            "NegControlProbe_",
            "antisense_",
            "NegControlCodeword_",
            "BLANK_",
            "DeprecatedCodeword_",
        )

        # Ensure FEATURE_NAME is a string type for proper filtering
        if transcripts_df[self.keys.FEATURE_NAME.value].dtype == 'object':
            # Convert from bytes to string if necessary
            if transcripts_df[self.keys.FEATURE_NAME.value].apply(lambda x: isinstance(x, bytes)).any():
                transcripts_df[self.keys.FEATURE_NAME.value] = transcripts_df[self.keys.FEATURE_NAME.value].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

        # Apply quality value filter
        mask_quality = transcripts_df[self.keys.QUALITY_VALUE.value].ge(min_qv)

        # Apply filter for unwanted codewords
        mask_codewords = ~transcripts_df[self.keys.FEATURE_NAME.value].str.startswith(filter_codewords)

        # Combine masks and return filtered dataframe
        mask = mask_quality & mask_codewords
        return transcripts_df[mask]


class MerscopeSample(SpatialTranscriptomicsSample):
    def __init__(self, transcripts_df: pd.DataFrame = None, transcripts_radius: int = 10, boundaries_graph: bool = False):
        super().__init__(transcripts_df, transcripts_radius, boundaries_graph, MerscopeKeys)

    def filter_transcripts(self, transcripts_df: pd.DataFrame, min_qv: float = 20.0) -> pd.DataFrame:
        """
        Filters transcripts based on specific criteria for Merscope.

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
        # Assuming Merscope filtering is done differently, you would add custom logic here.
        return transcripts_df


