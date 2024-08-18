import os
import sys
import gc
import shapely
import pyarrow
import numpy as np
import pandas as pd
from pathlib import Path
from segger.data.utils import filter_transcripts
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.data import HeteroData
import yaml
import torch
import geopandas as gpd
from segger.data.xenium_utils import (
    TranscriptColumns,
    BoundaryColumns,
    get_polygons_from_xy,
)
from scipy.spatial import KDTree

class XeniumFilename:
    transcripts = "transcripts.parquet",
    boundaries = "nucleus_boundaries.parquet"


class XeniumSample:


    def __init__(self):
        pass


    @staticmethod
    def _load_dataframe(path: os.PathLike):
        # Load dataframe
        path = Path(path)  # make sure input is Path type
        if '.csv' in path.suffixes:
            df = pd.read_csv(path)
        elif 'parquet' in path.suffixes:
            df = pd.read_parquet(path)
        else:
            raise ValueError("Unsupported file format")
        return df


    def load_transcripts(self, path: os.PathLike, min_qv: int = 30):
        # Load transcripts dataframe
        tx_df = self._load_dataframe(path)

        # Load spatial coordinates
        column_enum = TranscriptColumns
        tx_df = filter_transcripts(tx_df, min_qv)
        x_min, y_min = tx_df[column_enum.xy].min()
        x_max, y_max = tx_df[column_enum.xy].max()

        # Transcript latent is one hot encoding of names
        genes = tx_df[[column_enum.label]]
        encoder = OneHotEncoder()
        encoder.fit(genes)
        
        # Only set object properties once everything finishes successfully
        self.transcripts_df = tx_df
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.tx_encoder = encoder


    def load_nuclei(self, path: os.PathLike):
        self.nuclei_df = self._load_dataframe(path)


    def to_pyg_dataset(
        self,
        output_dir: Path,
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
        k_nc: int = 4,
        dist_nc: float = 20,
        k_tx: int = 4,
        dist_tx: float = 20,
    ) -> None:
        """
        Saves the dataset for Segger in a processed format.

        Parameters:
        processed_dir (Path): Directory to save the processed dataset.
        x_size (float): Width of each tile. This is important to determine the number of tiles and to ensure tiles fit into GPU memory for prediction. Adjust based on GPU memory size and desired number of tiles.
        y_size (float): Height of each tile. This is important to determine the number of tiles and to ensure tiles fit into GPU memory for prediction. Adjust based on GPU memory size and desired number of tiles.
        d_x (float): Step size in the x direction for tiles. Determines the overlap between tiles and the total number of tiles.
        d_y (float): Step size in the y direction for tiles. Determines the overlap between tiles and the total number of tiles.
        margin_x (float): Margin in the x direction to include transcripts.
        margin_y (float): Margin in the y direction to include transcripts.
        compute_labels (bool): Whether to compute edge labels for tx_belongs_nc edges, used in training. Not necessary for prediction only.
        r_tx (float): Radius for building the transcript-to-transcript graph.
        val_prob (float): Probability of assigning a tile to the validation set.
        test_prob (float): Probability of assigning a tile to the test set.
        neg_sampling_ratio_approx (float): Approximate ratio of negative samples.
        sampling_rate (float): Rate of sampling tiles.
        num_workers (int): Number of workers to use for parallel processing.
        receptive_field (dict): Dictionary containing the values for 'k_nc', 'dist_nc', 'k_tx', and 'dist_tx', which are used to restrict the receptive field of transcripts for nuclei and transcripts.

        k_nc and dist_nc will determine the size of the cells with nucleus and k_tx and dist_tx will determine the size of the nucleus-less cells, implicitly.
        """
        # Hyperparameters for constructing dataset
        hparams = dict(
            x_size=x_size,
            y_size=y_size,
            d_x=d_x,
            d_y=d_y,
            margin_x=margin_x,
            margin_y=margin_y,
            r_tx=r_tx,
            k_nc=k_nc,
            dist_nc=dist_nc,
            k_tx=k_tx,
            dist_tx=dist_tx,
        )

        # Filesystem setup
        processed_dir = Path(processed_dir)
        processed_dir.mkdir(parents=True, exist_ok=True)
        for data_type in ['train', 'test', 'val']:
            tile_dir = processed_dir / f'{data_type}_tiles'
            pro_dir = tile_dir / 'processed'
            pro_dir.mkdir(parents=True, exist_ok=True)
            raw_dir = tile_dir / 'raw'
            raw_dir.mkdir(parents=True, exist_ok=True)
            with open(tile_dir / 'hparams.yaml', 'w') as file:
                yaml.dump(hparams, file)

        if margin_x is None:
            margin_x = d_x // 10
        if margin_y is None:
            margin_y = d_y // 10

        x_range = np.arange(self.x_min // 1000 * 1000, self.x_max, d_x)
        y_range = np.arange(self.y_min // 1000 * 1000, self.y_max, d_y)

        x_masks_nc = [
            (self.nuclei_df["vertex_x"] > x)
            & (self.nuclei_df["vertex_x"] < x + x_size)
            for x in x_range
        ]
        y_masks_nc = [
            (self.nuclei_df["vertex_y"] > y)
            & (self.nuclei_df["vertex_y"] < y + y_size)
            for y in y_range
        ]
        x_masks_tx = [
            (self.transcripts_df["x_location"] > (x - margin_x))
            & (self.transcripts_df["x_location"] < (x + x_size + margin_x))
            for x in x_range
        ]
        y_masks_tx = [
            (self.transcripts_df["y_location"] > (y - margin_y))
            & (self.transcripts_df["y_location"] < (y + y_size + margin_y))
            for y in y_range
        ]

        tile_params = [
            (
                i,
                j,
                x_masks_nc,
                y_masks_nc,
                x_masks_tx,
                y_masks_tx,
                x_size,
                y_size,
                compute_labels,
                r_tx,
                neg_sampling_ratio_approx,
                val_prob,
                test_prob,
                processed_dir,
                receptive_field,
                sampling_rate,
            )
            for i, j in product(range(len(x_range)), range(len(y_range)))
        ]

        if num_workers > 1:
            with Pool(processes=num_workers) as pool:
                for _ in tqdm(
                    pool.imap_unordered(self._process_tile, tile_params),
                    total=len(tile_params),
                ):
                    pass
        else:
            for params in tqdm(tile_params):
                self._process_tile(params)

    def _process_tile(self, tile_params):
        """
        Process a single tile and save the data.

        Parameters:
        tile_params (tuple): Parameters for the tile processing.
        """
        (
            i, j, x_masks_nc, y_masks_nc, x_masks_tx, y_masks_tx, x_size, y_size,
            compute_labels, r_tx, neg_sampling_ratio_approx, val_prob, test_prob,
            processed_dir, receptive_field, sampling_rate
        ) = tile_params

        # Generate a random probability to decide whether to process the tile
        prob = random.random()
        if prob > sampling_rate:
            return

        # Create masks for nuclei based on x and y coordinates
        x_mask = x_masks_nc[i]
        y_mask = y_masks_nc[j]
        mask = x_mask & y_mask

        # If the combined mask is empty, return early
        if mask.sum() == 0:
            return

        # Filter nuclei dataframe based on the combined mask
        nc_df = self.nuclei_df[mask]
        # Further filter nuclei dataframe to include only rows with cell_ids present in the filtered dataframe
        nc_df = self.nuclei_df.loc[self.nuclei_df.cell_id.isin(nc_df.cell_id), :]

        # Create a mask for transcripts based on x and y masks
        tx_mask = x_masks_tx[i] & y_masks_tx[j]
        # Filter transcripts dataframe based on the created mask
        transcripts_df = self.transcripts_df[tx_mask]

        # If either the filtered transcripts or nuclei dataframes are empty, return early
        if transcripts_df.shape[0] == 0 or nc_df.shape[0] == 0:
            return

        # Build PyG (PyTorch Geometric) data from the filtered nuclei and transcripts dataframes
        data = self.build_pyg_data_from_tile(nc_df, transcripts_df, compute_labels=compute_labels)
        # Build a graph from the data using the BuildTxGraph class
        data = BuildTxGraph(r=r_tx)(data)

        try:
            if compute_labels:
                # If labels need to be computed, apply a random link split transformation
                transform = RandomLinkSplit(
                    num_val=0,
                    num_test=0,
                    is_undirected=True,
                    edge_types=[("tx", "belongs", "nc")],
                    neg_sampling_ratio=neg_sampling_ratio_approx * 2,
                )
                data, _, _ = transform(data)
                # Get the edge index for the "tx belongs to nc" relationship
                edge_index = data[("tx", "belongs", "nc")].edge_index
                # If there are fewer than 10 edges, return early
                if edge_index.shape[1] < 10:
                    return
                # Get the edge label index for the "tx belongs to nc" relationship
                edge_label_index = data[("tx", "belongs", "nc")].edge_label_index
                # Filter the edge label index to include only those edges that are present in the edge index
                data[("tx", "belongs", "nc")].edge_label_index = (
                    edge_label_index[
                        :,
                        torch.nonzero(
                            torch.any(
                                edge_label_index[0].unsqueeze(1)
                                == edge_index[0].unsqueeze(0),
                                dim=1,
                            )
                        ).squeeze(),
                    ]
                )
                # Filter the edge labels to include only those edges that are present in the edge index
                data[("tx", "belongs", "nc")].edge_label = data[
                    ("tx", "belongs", "nc")
                ].edge_label[
                    torch.nonzero(
                        torch.any(
                            edge_label_index[0].unsqueeze(1)
                            == edge_index[0].unsqueeze(0),
                            dim=1,
                        )
                    ).squeeze()
                ]

            # Get the coordinates of nuclei and transcripts
            coords_nc = data["nc"].pos
            coords_tx = data["tx"].pos[:, :2]
            # Compute the edge index for the transcript field using a k-d tree
            data["tx"].tx_field = self.get_edge_index(
                coords_tx,
                coords_tx,
                k=receptive_field["k_tx"],
                dist=receptive_field["dist_tx"],
                type="kd_tree",
            )
            # Compute the edge index for the nuclei field using a k-d tree
            data["tx"].nc_field = self.get_edge_index(
                coords_nc,
                coords_tx,
                k=receptive_field["k_nc"],
                dist=receptive_field["dist_nc"],
                type="kd_tree",
            )

            # Generate a filename for the tile
            filename = f"tiles_x{int(x_size)}_y{int(y_size)}_{i}_{j}.pt"
            # Generate a random probability to decide the dataset split
            prob = random.random()
            if prob > val_prob + test_prob:
                # Save the data to the training set
                torch.save(
                    data, processed_dir / "train_tiles" / "processed" / filename
                )
            elif prob > val_prob:
                # Save the data to the test set
                torch.save(
                    data, processed_dir / "test_tiles" / "processed" / filename
                )
            else:
                # Save the data to the validation set
                torch.save(
                    data, processed_dir / "val_tiles" / "processed" / filename
                )
        except Exception as e:
            # Print an error message if an exception occurs
            print(f"Error processing tile {i}, {j}: {e}")



class XeniumTile:

    def __init__(
        self,
        sample: XeniumSample,
        bounds: shapely.Polygon,
    ):
        self.sample = sample
        self.bounds = bounds

        # Filter sample-level data by provided tile bounds
        self.boundaries = self.get_filtered_boundaries()
        self.transcripts = self.get_filtered_transcripts()


    def get_filtered_boundaries(self) -> pd.DataFrame:
        pass


    def get_filtered_transcripts(self) -> pd.DataFrame:
        pass


    def get_transcript_props(
        self,
    ):
        """
        Encodes transcript features in a sparse format.

        This method uses the transcript encoder to transform the transcript
        labels into an encoded format and then converts the encoding to a sparse
        tensor.

        Returns
        -------
        props : torch.sparse.FloatTensor
            A sparse tensor containing the encoded transcript features.

        Notes
        -----
        The encoder can be any type of encoder that transforms the transcript
        labels into a numerical matrix. The resulting matrix is then converted
        to a sparse tensor for efficient storage and computation.
        """
        # Encode transcript features in sparse format
        enums = TranscriptColumns
        encoder = self.sample.tx_encoder  # typically, a one-hot encoder
        encoding = encoder.transform(self.transcripts[[enums.label]]).toarray()
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

        Notes
        -----
        The function calculates various geometric properties of the polygons
        based on the specified parameters. The properties include area,
        convexity, elongation, and circularity. The results are stored in a
        DataFrame with each property as a column.
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
        edge_index : torch.Tensor
            An array of shape (2, n_edges) containing the edge indices. Each
            column represents an edge between two points, where the first row
            contains the source indices and the second row contains the target
            indices.

        Notes
        -----
        This function constructs a KDTree from the index points and queries the
        tree to find the k-nearest neighbors for each query point within the
        specified maximum distance. The resulting edge indices are returned as
        a 2D array.
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
        boundaries : gpd.GeoSeries
            A GeoSeries containing boundary polygon geometries.
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
        props : pd.DataFrame
            A DataFrame containing the computed properties for each boundary
            polygon.

        Notes
        -----
        The function calculates various geometric properties of the boundary
        polygons based on the specified parameters. The results are stored in a
        DataFrame with each property as a column.
        """

        # Get polygons from coordinates
        polygons = get_polygons_from_xy(self.boundaries)

        # Geometric properties of polygons
        props = self.get_polygon_props(polygons)
        props = torch.as_tensor(props.values).float()

        return props


    def to_pyg_dataset(
        self,
        k_nc: int = 4,
        dist_nc: float = 20,
        k_tx: int = 4,
        dist_tx: float = 20,
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
        k_nc : int, optional
            The number of nearest neighbors for the 'nc' nodes (default is 4).
        dist_nc : float, optional
            The maximum distance for neighbors of 'nc' nodes (default is 20).
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
        1. Nucleus ("nc")
            Represents nuclei in the Xenium dataset.

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
        1. ("tx", "belongs", "nc")
            Represents the relationship where a transcript is contained within
            a nucleus.

            Attributes
            ----------
            edge_index : torch.Tensor
                Edge indices in COO format between transcripts and nuclei

        2. ("tx", "neighbors", "nc")
            Represents the relationship where a transcripts is nearby but not
            within a nucleus

            Attributes
            ----------
            edge_index : torch.Tensor
                Edge indices in COO format between transcripts and nuclei
        """
        # Initialize an empty HeteroData object
        pyg_data = HeteroData()

        # Set up Nucleus nodes
        polygons = get_polygons_from_xy(self.boundaries)
        centroids = polygons.centroid.get_coordinates()
        pyg_data['nc'].id = polygons.index.to_numpy()
        pyg_data['nc'].pos = centroids.values
        pyg_data['nc'].x = self.get_boundary_props(
            area, convexity, elongation, circularity
        )

        # Set up Transcript nodes
        enums = TranscriptColumns
        pyg_data['tx'].id = self.transcripts[enums.id].values
        pyg_data['tx'].pos = self.transcripts[enums.xyz].values
        pyg_data['tx'].x = self.get_transcript_props()

        # Set up neighbor edges
        dist = np.sqrt(polygons.area.max()) * 10  # heuristic distance
        nbrs_edge_idx = self.get_kdtree_edge_index(
            centroids,
            self.transcripts[enums.xy],
            k=k_nc,
            max_distance=dist,
        )
        pyg_data["tx", "neighbors", "nc"].edge_index = nbrs_edge_idx

        # Find nuclear transcripts
        tx_cell_ids = self.transcripts[BoundaryColumns.id]
        cell_ids_map = {idx: i for (i, idx) in enumerate(polygons.index)}
        is_nuclear = self.transcripts[TranscriptColumns.nuclear].astype(bool) 
        is_nuclear &= tx_cell_ids.isin(polygons.index)

        # Set up overlap edges
        row_idx = np.where(is_nuclear)
        col_idx = tx_cell_ids.iloc[row_idx].map(cell_ids_map)
        blng_edge_idx = torch.tensor(np.stack([row_idx, col_idx])).long()
        pyg_data["tx", "belongs", "nc"].edge_index = blng_edge_idx

        return pyg_data
