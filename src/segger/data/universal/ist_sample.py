from concurrent.futures import ThreadPoolExecutor
from pyarrow import parquet as pq, compute as pc
from sklearn.preprocessing import LabelEncoder
from functools import cached_property
from dataclasses import dataclass
from tqdm.notebook import tqdm
import geopandas as gpd
from typing import List
import pandas as pd
import logging
import shapely
import yaml
import os

from ._enum import DIRNAMES, FEATURE_INDEX_FILE
from ...config import SeggerConfig
from .ist_tile import ISTTile
from . import _utils as utils
from ._ndtree import NDTree


@dataclass
class ISTSample:
    #TODO: Add documentation.
    save_dir: os.PathLike
    tx_path: os.PathLike
    bd_path: os.PathLike
    tx_feature_name: str
    tx_cell_id: str
    tx_x: str
    tx_y: str
    tx_k: int
    tx_dist: float
    bd_k: int
    bd_dist: float
    tile_margin: float
    max_cells_per_tile: int = None
    max_transcripts_per_tile: int = None
    frac_train: float = 0.7
    frac_test: float = 0.2
    frac_val: float = 0.1
    n_workers: int = -1

    def __post_init__(self):
        #TODO: Add documentation
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.Logger(f"ISTSample")

    def save(self, pbar: bool = True):
        #TODO: Add documentation
        self._setup_data_directory()
        if self.n_workers > 1:
            self._save_parallel(pbar=pbar)
        else:
            self._save_serial(pbar=pbar)
        self._save_feature_index()

    def _setup_data_directory(self):
        #TODO: Add documentations
        dirnames = [
            DIRNAMES.train.value,
            DIRNAMES.test.value,
            DIRNAMES.val.value,
        ]
        for tile_type in dirnames:
            for stage in ["raw", "processed"]:
                tile_dir = self.save_dir / tile_type / stage
                tile_dir.mkdir(parents=True, exist_ok=True)
                if os.listdir(tile_dir):
                    msg = f"Directory '{tile_dir}' must be empty."
                    raise FileExistsError(msg)

    def _save_parallel(self, pbar: bool = True):
        #TODO: Add documentation
        n_threads = self.n_workers * 5
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            func = lambda t: t.to_pyg(flush=True)
            futures = []
            for tile in self.tiles:
                futures.append(executor.submit(func, tile))
            for f in tqdm(futures, disable=not pbar):
                f.result()

    def _save_serial(self, pbar: bool = True):
        #TODO: Add documentation
        for tile in tqdm(self.tiles, disable=not pbar):
            tile.to_pyg(flush=True)

    def _save_feature_index(self):
        features = pd.Series(self.tx_encoder.classes_)
        features.to_csv(
            self.save_dir / FEATURE_INDEX_FILE, 
            index=False,
        )

    @cached_property   
    def tiles(self) -> List[ISTTile]:
        #TODO: Add documentation
        if self.max_cells_per_tile:
            bd = gpd.read_parquet(self.bd_path)
            coords = bd.centroid.get_coordinates()
            tree = NDTree(coords, self.max_cells_per_tile)
            del bd, coords
        else:
            tx = pd.read_parquet(self.tx_path, columns=[self.tx_x, self.tx_y])
            coords = tx.values
            tree = NDTree(coords, self.max_cells_per_tile)
            del tx, coords
        return [
            ISTTile(
                save_dir=self.save_dir,
                extents=leaf,
                tx_encoder=self.tx_encoder,
                tx_path=self.tx_path,
                bd_path=self.bd_path,
                tx_feature_name=self.tx_feature_name,
                tx_cell_id=self.tx_cell_id,
                tx_x=self.tx_x,
                tx_y=self.tx_y,
                tx_k=self.tx_k,
                tx_dist=self.tx_dist,
                bd_k=self.bd_k,
                bd_dist=self.bd_dist,
                tile_margin=self.tile_margin,
                frac_train=self.frac_train,
                frac_test=self.frac_test,
                frac_val=self.frac_val,
            )
            for leaf in tree.leaves
        ]

    @cached_property
    def bd_extents(self) -> shapely.Polygon:
        #TODO: Add documentation
        return utils.get_xy_extents_geoparquet(self.bd_path)
    
    @cached_property
    def tx_extents(self) -> shapely.Polygon:
        #TODO: Add documentation
        return utils.get_xy_extents_parquet(
            self.tx_path,
            x=self.tx_x,
            y=self.tx_y,
        )
    
    @cached_property
    def tx_encoder(self) -> LabelEncoder:
        feature_name = self.tx_feature_name
        table = pq.read_table(self.tx_path, columns=[feature_name])
        classes = pc.unique(table[feature_name])
        return LabelEncoder().fit(classes)
