from lightning.pytorch import LightningDataModule
from torch_geometric.loader import DataLoader
import os
from typing import Optional
from pathlib import Path
from torchvision.transforms import Compose, Lambda
from segger.data.parquet.pyg_dataset import STPyGDataset
from segger.training._transforms import MaskEdgeIndex, NegativeSampling


# TODO: Add documentation
class SeggerDataModule(LightningDataModule):

    def __init__(
        self,
        data_dir: os.PathLike,
        batch_size: int = 4,
        num_workers: int = 1,
        negative_sampling_ratio: float = 1.,
        k_tx: Optional[int] = None,
        dist_tx: Optional[float] = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Add graph transforms
        self.transforms = []

        if k_tx or dist_tx:
            edge_type = 'tx', 'neighbors', 'tx'
            tm = MaskEdgeIndex(edge_type, k_tx, dist_tx)
            self.transforms.append(tm)
        if negative_sampling_ratio > 0:
            edge_type = 'tx', 'belongs', 'bd'
            tm = NegativeSampling(edge_type, negative_sampling_ratio)
            self.transforms.append(tm)

        self.transform = Compose(self.transforms)


    # TODO: Add documentation
    def setup(self, stage=None):
        self.train = STPyGDataset(
            root=self.data_dir / 'train_tiles',
            transform=self.transform,
        )
        self.test = STPyGDataset(
            root=self.data_dir / 'test_tiles',
            transform=self.transform,
        )
        self.val = STPyGDataset(
            root=self.data_dir / 'val_tiles',
            transform=self.transform,
        )
        self.loader_kwargs = dict(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
        )

    # TODO: Add documentation
    def train_dataloader(self):
        return DataLoader(self.train, shuffle=True, **self.loader_kwargs)

    # TODO: Add documentation
    def test_dataloader(self):
        return DataLoader(self.test, shuffle=False, **self.loader_kwargs)

    # TODO: Add documentation
    def val_dataloader(self):
        return DataLoader(self.val, shuffle=False, **self.loader_kwargs)
