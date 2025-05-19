from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import HeteroData
from functools import cached_property
from torch import LongTensor
import geopandas as gpd
import pandas as pd
import numpy as np
import shapely
import torch

from ...config import SeggerConfig
from . import _utils as utils
from ._enum import DIRNAMES


class ISTTile:
    #TODO: Add documentation
    def __init__(
        self,
        config: SeggerConfig,
        extents: shapely.Polygon,
        encoder: LabelEncoder,
    ):
        #TODO: Add documentation
        self.config = config
        self.extents = extents
        self.tx_encoder = encoder

    def to_pyg(self, flush: bool = True):
        #TODO: Add documentation
        pyg_data = HeteroData()
        # Transcript nodes
        pyg_data['tx'].id = torch.tensor(
            self.tx.index.astype(int),
            dtype=torch.long,
        )
        pyg_data['tx'].pos = torch.tensor(
            self.tx[[self.config.data.tx_x, self.config.data.tx_y]].values,
            dtype=torch.float32,
        )
        pyg_data['tx'].x = self.tx_props
        # Boundary nodes
        pyg_data['bd'].id = self.bd.index.values
        pyg_data['bd'].pos = torch.tensor(
            self.bd.centroid.get_coordinates().values,
            dtype=torch.float32,
        )
        pyg_data['bd'].x = self.bd_props
        # Heterogeneous graphs
        pyg_data['tx', 'belongs', 'bd'].edge_index = self.tx_belongs_bd
        pyg_data['tx', 'neighbors', 'tx'].edge_index = self.tx_neighbors_tx
        filepath = (
            self.config.data.save_dir / 
            self.dirname / 
            'processed' /
            f'{self.uid}.pt'
        )
        torch.save(pyg_data, filepath)
        if flush: self.flush()

    def flush(self):
        #TODO: Add documentation
        cached_props = [
            'tx',
            'bd',
            'tx_props',
            'bd_props',
            'tx_belongs_bd',
            'tx_neighbors_tx',
        ]
        for prop in cached_props:
            if hasattr(self, prop): delattr(self, prop)

    @property
    def uid(self) -> str:
        #TODO: Add documentation
        x_min, y_min, x_max, y_max = map(int, self.extents.bounds)
        return f"tiles_x={x_min}_y={y_min}_w={x_max-x_min}_h={y_max-y_min}"

    @cached_property
    def tx(self) -> pd.DataFrame:
        #TODO: Add documentation
        margin = self.config.data.tile_margin
        xmin, ymin, xmax, ymax = self.extents.buffer(margin).bounds
        columns = [
            self.config.data.tx_feature_name,
            self.config.data.tx_cell_id,
            self.config.data.tx_x,
            self.config.data.tx_y,
        ]
        filters = [
            (self.config.data.tx_x, '>=', xmin),
            (self.config.data.tx_x, '<', xmax),
            (self.config.data.tx_y, '>=', ymin),
            (self.config.data.tx_y, '<', ymax),
        ]
        return pd.read_parquet(
            self.config.data.tx_path,
            filters=filters,
            columns=columns,
        )
    
    @cached_property
    def tx_props(self):
        #TODO: Add documentation
        labels = self.tx[self.config.data.tx_feature_name]
        return LongTensor(self.tx_encoder.transform(labels))
    
    @cached_property
    def bd(self) -> gpd.GeoDataFrame:
        #TODO: Add documentation
        return gpd.read_parquet(
            self.config.data.bd_path,
            bbox=self.extents.bounds,
        )
    
    @cached_property
    def bd_props(self):
        #TODO: Add documentation
        props = utils.get_polygon_props(self.bd)
        return torch.as_tensor(props.values).float()
    
    @cached_property
    def tx_neighbors_tx(self) -> torch.tensor:
        #TODO: Add documentation
        xy = [self.config.data.tx_x, self.config.data.tx_y]
        return utils.get_kdtree_edge_index(
            index_coords=self.tx[xy],
            query_coords=self.tx[xy],
            k=self.config.data.tx_k,
            max_distance=self.config.data.tx_dist,
        )

    @cached_property
    def tx_belongs_bd(self) -> torch.tensor:
        #TODO: Add documentation
        ids = self.tx[self.config.data.tx_cell_id]
        ids_map = dict(zip(*np.unique(self.bd.index, return_inverse=True)))
        mask = ids.isin(ids_map)
        row_idx = np.where(mask)[0]
        col_idx = ids[mask].map(ids_map).to_numpy()
        return torch.tensor(np.stack([row_idx, col_idx])).long()
    
    @cached_property
    def dirname(self) -> str:
        probs = [
            self.config.data.frac_train,
            self.config.data.frac_test,
            self.config.data.frac_val,
        ]
        dirnames = [
            DIRNAMES.train.value,
            DIRNAMES.test.value,
            DIRNAMES.val.value,
        ]
        return np.random.choice(dirnames, 1, p=probs)[0]