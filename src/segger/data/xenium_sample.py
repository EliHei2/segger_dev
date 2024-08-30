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
    filter_boundaries,
)
from scipy.spatial import KDTree
from multiprocessing import Pool
from functools import cached_property
from typing import List
import inspect
import logging
from itertools import compress


class XeniumFilename:
    transcripts = "transcripts.parquet"
    boundaries = "nucleus_boundaries.parquet"


class XeniumSample:


    def __init__(self, base_dir: os.PathLike):
        # Load nuclei and transcripts
        base_dir = Path(base_dir)  # TODO: check that Xenium directory is valid
        self.load_transcripts(base_dir / XeniumFilename.transcripts)
        self.load_nuclei(base_dir / XeniumFilename.boundaries)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.Logger(f'XeniumSample@{base_dir}')


    @staticmethod
    def _load_dataframe(path: os.PathLike):
        # Load dataframe
        path = Path(path)  # make sure input is Path type
        if '.csv' in path.suffixes:
            df = pd.read_csv(path)
        elif '.parquet' in path.suffixes:
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


    def _get_balanced_tile_bounds(
        self,
        tile_max_size: int,
    ) -> List[shapely.Polygon]:
        pass


    def _get_square_tile_bounds(
        self,
        tile_width: float,
        tile_height: float,
    ) -> List[shapely.Polygon]:
    
        # Generate the x and y coordinates for the tile boundaries
        x_coords = np.arange(self.x_min, self.x_max, tile_width)
        x_coords = np.append(x_coords, self.x_max)
        y_coords = np.arange(self.y_min, self.y_max, tile_height)
        y_coords = np.append(y_coords, self.y_max)
        
        # Generate tiles from grid points
        tiles = []
        for x_min, x_max in zip(x_coords[:-1], x_coords[1:]):
            for y_min, y_max in zip(y_coords[:-1], y_coords[1:]):
                tiles.append(shapely.box(x_min, y_min, x_max, y_max))

        return tiles


    def tile(
        self,
        tile_width: float = None,
        tile_height: float = None,
        tile_max_size: int = None,
     ) -> List[shapely.Polygon]:
        
        # Balanced tiling kwargs provided
        if tile_max_size and not (tile_width or tile_height):
            return self._get_balanced_tile_bounds(tile_max_size)

        # Square tiling kwargs provided
        elif not tile_max_size and (tile_width and tile_height):
            return self._get_square_tile_bounds(tile_width, tile_height)

        # Bad set of kwargs
        else:
            args = list(compress(locals().keys(), locals().values()))
            args.remove('self')
            msg = (
                "Function requires either 'tile_max_size' or both "
                f"'tile_width' and 'tile_height'. Found: {', '.join(args)}."
            )
            logging.error(msg)
            raise ValueError


    def to_pyg_dataset(
        self,
        output_dir: Path,
        tile_mode: str,
        tile_width: float = None,
        tile_height: float = None,
        tile_max_size: int = None,
        tile_margin: float = 15,
        r_tx: float = 5,
        k_nc: int = 4,
        dist_nc: float = 20,
        k_tx: int = 4,
        dist_tx: float = 20,
        workers: int = -1,
    ):

        # Divide Xenium samples into non-overlapping spatial regions
        tile_bounds = self.tile(tile_width, tile_height, tile_max_size)
        tiles = [XeniumTile(bounds=t, margin=tile_margin) for t in tile_bounds]

        # Process each region
        if workers > 1:
            with Pool(processes=workers) as pool:
                
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
    """
    A class representing a tile of a Xenium sample.

    Attributes
    ----------
    sample : XeniumSample
        The Xenium sample containing data.
    bounds : shapely.Polygon
        The bounds of the tile in the sample.
    margin : int, optional
        The margin around the bounds to include additional data (default is 10).
    boundaries : pd.DataFrame
        Filtered boundaries within the tile bounds.
    transcripts : pd.DataFrame
        Filtered transcripts within the tile bounds.
    """

    def __init__(
        self,
        sample: XeniumSample,
        bounds: shapely.Polygon,
        margin: int = 10,
    ):
        """
        Initializes a XeniumTile instance.

        Parameters
        ----------
        sample : XeniumSample
            The Xenium sample containing data.
        bounds : shapely.Polygon
            The bounds of the tile in the sample.
        margin : int, optional
            The margin around the bounds to include additional data (default 
            is 10).

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
        self.sample = sample
        self.bounds = bounds
        self.margin = margin

        # Internal caches for filtered data
        self._boundaries = None
        self._transcripts = None


    @cached_property
    def boundaries(self):
        """
        Returns the filtered boundaries within the tile bounds, cached for
        efficiency.

        The boundaries are computed only once and cached. If the boundaries 
        have not been computed yet, they are computed using 
        `get_filtered_boundaries()`.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the filtered boundaries within the tile 
            bounds.
        """
        if self._boundaries is None:
            self._boundaries = self.get_filtered_boundaries()
            return self._boundaries
        else:
            return self._boundaries


    @cached_property
    def transcripts(self):
        """
        Returns the filtered transcripts within the tile bounds, cached for
        efficiency.

        The transcripts are computed only once and cached. If the transcripts 
        have not been computed yet, they are computed using 
        `get_filtered_transcripts()`.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the filtered transcripts within the tile 
            bounds.
        """
        if self._transcripts is None:
            self._transcripts = self.get_filtered_transcripts()
            return self._transcripts
        else:
            return self._transcripts


    def get_filtered_boundaries(self) -> pd.DataFrame:
        """
        Filters the boundaries in the sample to include only those within
        the specified tile bounds.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the filtered boundaries within the tile 
            bounds.
        """
        filtered_boundaries = filter_boundaries(
            boundaries=self.sample.nuclei_df,
            inset=self.bounds,
            outset=self.bounds.buffer(self.margin, join_style='mitre')
        )
        return filtered_boundaries


    def get_filtered_transcripts(self) -> pd.DataFrame:
        """
        Filters the transcripts in the sample to include only those within
        the specified tile bounds.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the filtered transcripts within the tile 
            bounds.
        """

        # Buffer tile bounds to include transcripts around boundary
        outset = self.bounds.buffer(self.margin, join_style='mitre')
        xmin, ymin, xmax, ymax =  outset.bounds

        # Get transcripts inside buffered region
        enums = TranscriptColumns
        mask = self.sample.transcripts_df[enums.x].between(xmin, xmax)
        mask &= self.sample.transcripts_df[enums.y].between(ymin, ymax)
        filtered_transcripts = self.sample.transcripts_df[mask]

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
        for 'nc' node representations. You can just change the function body to
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
        #TODO: Add pyg_data['tx'].nc_field
        #TODO: Add pyg_data['tx'].tx_field

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
        row_idx = np.where(is_nuclear)[0]
        col_idx = tx_cell_ids.iloc[row_idx].map(cell_ids_map)
        blng_edge_idx = torch.tensor(np.stack([row_idx, col_idx])).long()
        pyg_data["tx", "belongs", "nc"].edge_index = blng_edge_idx

        return pyg_data
