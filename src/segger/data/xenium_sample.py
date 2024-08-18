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

shapely.Polygon().boundary

class XeniumFilename:
    transcripts = "transcripts.parquet",
    boundaries = "nucleus_boundaries.parquet"


class XeniumSample:


    def __init__(
        self,
    ):
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


    @staticmethod
    def compute_nuclei_geometries(
        nuclei_df: pd.DataFrame,
        area: bool = True,
        convexity: bool = True,
        elongation: bool = True,
        circularity: bool = True,
    ) -> gpd.GeoDataFrame:
        """
        Computes geometries for nuclei from the dataframe.

        Parameters:
        nuclei_df (pd.DataFrame): The dataframe containing nuclei data.
        area (bool): Whether to compute area.
        convexity (bool): Whether to compute convexity.
        elongation (bool): Whether to compute elongation.
        circularity (bool): Whether to compute circularity.

        Returns:
        gpd.GeoDataFrame: A geodataframe containing computed geometries.
        """
        grouped = nuclei_df.groupby("cell_id")
        polygons = [
            Polygon(list(zip(group_data["vertex_x"], group_data["vertex_y"])))
            for _, group_data in grouped
        ]
        polygons = gpd.GeoSeries(polygons)
        gdf = gpd.GeoDataFrame(geometry=polygons)
        gdf["cell_id"] = list(grouped.groups.keys())
        gdf[["x", "y"]] = gdf.centroid.get_coordinates()
        if area:
            gdf["area"] = polygons.area
        if convexity:
            gdf["convexity"] = polygons.convex_hull.area / polygons.area
        if elongation:
            try:
                r = polygons.minimum_rotated_rectangle()
                gdf["elongation"] = r.area / (r.length * r.width)
            except:
                gdf["elongation"] = 1
        if circularity:
            r = gdf.minimum_bounding_radius()
            gdf["circularity"] = polygons.area / (r * r)
        return gdf

    @staticmethod
    def get_edge_index(
        coords_1: np.ndarray,
        coords_2: np.ndarray,
        k: int = 100,
        dist: int = 10,
        type: str = "edge_index",
    ) -> torch.Tensor:
        """
        Computes edge indices using KDTree for given coordinates.

        Parameters:
        coords_1 (np.ndarray): First set of coordinates.
        coords_2 (np.ndarray): Second set of coordinates.
        k (int): Number of nearest neighbors.
        dist (int): Distance threshold.
        type (str): Type of edge index to return.

        Returns:
        torch.Tensor: Edge indices.
        """
        tree = KDTree(coords_1)
        d_kdtree, idx_out = tree.query(coords_2, k=k, distance_upper_bound=dist)
        if type == "kd_tree":
            return torch.tensor(idx_out, dtype=torch.long)
        edge_index = XeniumSample.kd_to_edge_index_(
            (idx_out.shape[0], coords_1.shape[0]), idx_out
        )
        if type == "edge_index":
            return edge_index
        if type == "adj":
            return torch_geometric.utils.to_dense_adj(edge_index)[0].to_sparse()

    @staticmethod
    def kd_to_edge_index_(shape: tuple, idx_out: np.ndarray) -> torch.Tensor:
        """
        Converts KDTree indices to edge indices.

        Parameters:
        shape (tuple): Shape of the original matrix.
        idx_out (np.ndarray): Indices from KDTree query.

        Returns:
        torch.Tensor: Edge indices.
        """
        idx = np.column_stack((np.arange(shape[0]), idx_out))
        idc = (
            pd.DataFrame(idx).melt(0).loc[lambda df: df["value"] != shape[1], :]
        )
        edge_index = (
            torch.tensor(
                idc[["variable", "value"]].to_numpy(), dtype=torch.long
            )
            .t()
            .contiguous()
        )
        return edge_index

    def build_pyg_data_from_tile(
        self,
        nuclei_df: pd.DataFrame,
        transcripts_df: pd.DataFrame,
        compute_labels: bool = True,
        **kwargs,
    ) -> HeteroData:
        """
        Builds PyG data from a tile of nuclei and transcripts data.

        Parameters:
        nuclei_df (pd.DataFrame): Dataframe containing nuclei data.
        transcripts_df (pd.DataFrame): Dataframe containing transcripts data.
        compute_labels (bool): Whether to compute labels.
        **kwargs: Additional arguments for geometry computation.

        Returns:
        HeteroData: PyG Heterogeneous Data object.
        """
        data = HeteroData()  # Initialize an empty HeteroData object

        # Extract relevant kwargs for nuclei computation
        nc_args = list(inspect.signature(self.compute_nuclei_geometries).parameters)
        nc_dict = {k: kwargs.pop(k) for k in kwargs if k in nc_args}
        
        # Compute nuclei geometries
        nc_gdf = self.compute_nuclei_geometries(nuclei_df, **nc_dict)
        max_area = nc_gdf["area"].max()  # Find the maximum area of nuclei

        # Extract relevant kwargs for edge index computation
        tx_args = list(inspect.signature(self.get_edge_index).parameters)
        tx_dict = {k: kwargs.pop(k) for k in kwargs if k in tx_args}

        # Convert transcript locations to a tensor
        x_xyz = torch.as_tensor(
            transcripts_df[["x_location", "y_location", "z_location"]].values
        ).float()
        data["tx"].id = transcripts_df[["transcript_id"]].values  # Assign transcript IDs to the data object
        data["tx"].pos = x_xyz  # Assign positions to the data object

        # Encode transcript features and convert to a tensor
        x_features = torch.as_tensor(
            self.tx_encoder.transform(
                transcripts_df[["feature_name"]]
            ).toarray()
        ).float()
        data["tx"].x = x_features.to_sparse()  # Assign features to the data object in sparse format

        # Compute edge indices for the graph
        tx_edge_index = self.get_edge_index(
            nc_gdf[["x", "y"]].values,
            transcripts_df[["x_location", "y_location"]].values,
            k=3,
            dist=np.sqrt(max_area) * 10,
        )
        data["tx", "neighbors", "nc"].edge_index = tx_edge_index  # Assign edge indices to the data object

        # Find indices of transcripts that overlap with nuclei
        ind = np.where(
            (transcripts_df.overlaps_nucleus == 1)
            & (transcripts_df.cell_id.isin(nc_gdf["cell_id"]))
        )[0]

        # Create edge indices for transcripts belonging to nuclei
        tx_nc_edge_index = np.column_stack(
            (
                ind,
                np.searchsorted(
                    nc_gdf["cell_id"].values,
                    transcripts_df.iloc[ind]["cell_id"].values,
                ),
            )
        )

        data["nc"].id = nc_gdf[["cell_id"]].values  # Assign cell IDs to the data object
        data["nc"].pos = nc_gdf[["x", "y"]].values  # Assign positions to the data object

        # Extract features from the nuclei dataframe
        nc_x = nc_gdf.iloc[:, 4:]
        data["nc"].x = torch.as_tensor(nc_x.to_numpy()).float()  # Convert nuclei features to a tensor

        # Assign edge indices for transcripts belonging to nuclei
        data["tx", "belongs", "nc"].edge_index = torch.as_tensor(
            tx_nc_edge_index.T
        ).long()

        return data  # Return the constructed HeteroData object


class XeniumTile:

    def __init__(
        self,
        sample: XeniumSample,
        bounds: shapely.Polygon,
    ):
        self.sample = sample
        self.bounds = bounds
        self.boundaries = self.get_filtered_boundaries()
        self.transcripts = self.get_filtered_transcripts()


    def get_filtered_boundaries(self) -> pd.DataFrame:
        pass


    def get_filtered_transcripts(self) -> pd.DataFrame:
        pass


    def get_transcript_props(
        self,
    ):
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


    def get_boundary_props(
        self,
        area: bool = True,
        convexity: bool = True,
        elongation: bool = True,
        circularity: bool = True,
    ) -> torch.Tensor:

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
    ):
        '''
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
                Edge indices #TODO

        2. ("tx", "neighbors", "nc")
            Represents the relationship where a transcripts is nearby but not
            within a nucleus

            Attributes
            ----------
            edge_index : torch.Tensor
                Edge indices #TODO
        '''
        # Initialize an empty HeteroData object
        pyg_data = HeteroData()

        # Setup Nucleus nodes
        polygons = get_polygons_from_xy(self.boundaries)
        pyg_data['nc'].id = polygons.index.to_numpy()
        pyg_data['nc'].pos = polygons.centroid.get_coordinates().values
        pyg_data['nc'].x = self.get_boundary_props(
            area, convexity, elongation, circularity
        )

        # Setup Transcript Nodes
        enums = TranscriptColumns
        pyg_data['tx'].id = self.transcripts[enums.id].values
        pyg_data['tx'].pos = self.transcripts[enums.xyz].values
        pyg_data['tx'].x = self.get_transcript_props()

        # Setup Edges
        