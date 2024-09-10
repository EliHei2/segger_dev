from pytorch_lightning import LightningDataModule
from torch_geometric.loader import DataLoader
import os
from pathlib import Path
from segger.data.xenium_sample_parquet import XeniumPyGDataset


# TODO: Add documentation
class SeggerDataModule(LightningDataModule):

    def __init__(
        self,
        data_dir: os.PathLike,
        batch_size: int = 4,
        num_workers: int = 1,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers

    # TODO: Add documentation
    def setup(self, stage=None):
        self.train = XeniumPyGDataset(root=self.data_dir / 'train')
        self.test = XeniumPyGDataset(root=self.data_dir / 'test')
        self.val = XeniumPyGDataset(root=self.data_dir / 'val')
        self.loader_kwargs = dict(
            batch_size=self.batch_size,
            num_workers=0,
            pin_memory=True,
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
