from torch_geometric.data import InMemoryDataset, Data
from lightning.pytorch import LightningDataModule
from torch_geometric.loader import DataLoader
from typing import List, Optional, Callable
from torchvision.transforms import Compose
from typing import Optional
from pathlib import Path
import torch
import glob
import os

from ._transforms import MaskEdgeIndex, NegativeSampling
from ..data.universal._enum import DIRNAMES
from ..config import SeggerConfig


class ISTPyGDataset(InMemoryDataset):
    """
    An in-memory dataset class for handling training using iST datasets from 
    segger.
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        super().__init__(root, transform, pre_transform, pre_filter)

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
        paths = glob.glob(f"{self.processed_dir}/tiles_x*_y*_*_*.pt")
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
        return data


class SeggerDataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule for loading and transforming segger data tiles.

    This module wraps adds optional graph-level transforms such as kNN masking 
    or negative edge sampling. It assumes preprocessed `.pt` files organized 
    under `train_tiles/`, `val_tiles/`, and `test_tiles/`.

    Parameters
    ----------
    data_dir : os.PathLike
        Root directory containing tile subfolders for each split.
    batch_size : int, default=4
        Number of samples per batch.
    num_workers : int, default=1
        Number of workers used for data loading.
    negative_sampling_ratio : float, default=1.0
        Ratio of negative edges to sample for supervision during training.
        Set to 0 to disable.
    k_tx : int, optional
        Max number of neighbors per transcript node for kNN masking.
    dist_tx : float, optional
        Max spatial distance for transcript neighbor filtering.
    """

    def __init__(
        self,
        data_dir: os.PathLike,
        batch_size: int = 4,
        n_workers: int = 1,
        k_tx_max: Optional[int] = None,
        dist_tx_max: Optional[float] = None,
        negative_edge_sampling_ratio: float = 1.,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.n_workers = n_workers

        # Add graph transforms
        self.transforms = []

        if k_tx_max or dist_tx_max:
            edge_type = 'tx', 'neighbors', 'tx'
            tm = MaskEdgeIndex(edge_type, k_tx_max, dist_tx_max)
            self.transforms.append(tm)
        if negative_edge_sampling_ratio > 0:
            edge_type = 'tx', 'belongs', 'bd'
            tm = NegativeSampling(edge_type, negative_edge_sampling_ratio)
            self.transforms.append(tm)

        self.transform = Compose(self.transforms)

    def setup(self, stage=None):
        """
        Load datasets for training, validation, and test stages.

        Parameters
        ----------
        stage : str, optional
            Stage identifier used by Lightning. Ignored in this implementation.
        """
        self.train = ISTPyGDataset(
            root=self.data_dir / DIRNAMES.train.value,
            transform=self.transform,
        )
        self.test = ISTPyGDataset(
            root=self.data_dir / DIRNAMES.test.value,
            transform=self.transform,
        )
        self.val = ISTPyGDataset(
            root=self.data_dir / DIRNAMES.val.value,
            transform=self.transform,
        )
        self.loader_kwargs = dict(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
        )

    def train_dataloader(self):
        return DataLoader(self.train, shuffle=True, **self.loader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.test, shuffle=False, **self.loader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.val, shuffle=False, **self.loader_kwargs)
