from concurrent.futures import ThreadPoolExecutor
from pyarrow import parquet as pq, compute as pc
from sklearn.preprocessing import LabelEncoder
from functools import cached_property
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


class ISTSample:
    #TODO: Add documentation.
    def __init__(
        self,
        config_path: os.PathLike,
    ):
        #TODO: Add documentation
        # Setup and validate configuration
        self.config_path = config_path
        self._preflight_checks()

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.Logger(f"ISTSample")

    def _preflight_checks(self):
        #TODO: Add documentation.
        # Placeholder - all input validation currently handled by Pydantic.
        self.config = SeggerConfig.from_yaml(self.config_path)

    def save(self):
        #TODO: Add documentation
        self._setup_data_directory()
        if self.config.data.n_workers > 1:
            self._save_parallel()
        else:
            self._save_serial()
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
                tile_dir = self.config.data.save_dir / tile_type / stage
                tile_dir.mkdir(parents=True, exist_ok=True)
                if os.listdir(tile_dir):
                    msg = f"Directory '{tile_dir}' must be empty."
                    raise FileExistsError(msg)

    def _save_parallel(self):
        #TODO: Add documentation
        n_threads = self.config.data.n_workers * 5
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            func = lambda t: t.to_pyg(flush=True)
            futures = []
            for tile in self.tiles:
                futures.append(executor.submit(func, tile))
            for f in tqdm(futures):
                f.result()

    def _save_serial(self):
        #TODO: Add documentation
        for tile in tqdm(self.tiles):
            tile.to_pyg(flush=True)

    def _save_feature_index(self):
        features = pd.Series(self.tx_encoder.classes_)
        features.to_csv(
            self.config.data.save_dir / FEATURE_INDEX_FILE, 
            index=False,
        )

    @cached_property   
    def tiles(self) -> List[ISTTile]:
        #TODO: Add documentation
        bd = gpd.read_parquet(self.config.data.bd_path)
        coords = bd.centroid.get_coordinates()
        tree = NDTree(coords, self.config.data.max_cells_per_tile)
        return [
            ISTTile(self.config, leaf, self.tx_encoder) for leaf in tree.leaves
        ]

    @cached_property
    def bd_extents(self) -> shapely.Polygon:
        #TODO: Add documentation
        return utils.get_xy_extents_geoparquet(self.config.data.bd_path)
    
    @cached_property
    def tx_extents(self) -> shapely.Polygon:
        #TODO: Add documentation
        return utils.get_xy_extents_parquet(
            self.config.data.tx_path,
            x=self.config.data.tx_x,
            y=self.config.data.tx_y,
        )
    
    @cached_property
    def tx_encoder(self) -> LabelEncoder:
        feature_name = self.config.data.tx_feature_name
        table = pq.read_table(self.config.data.tx_path, columns=[feature_name])
        classes = pc.unique(table[feature_name])
        return LabelEncoder().fit(classes)
