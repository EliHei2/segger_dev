import pytest
import shutil
from pathlib import Path
from typing import Iterator
from lightning import Trainer
from torch_geometric.nn import to_hetero
from segger.training.train import LitSegger
from segger.models.segger_model import Segger
from lightning.pytorch.loggers import CSVLogger
from segger.data.parquet.sample import STSampleParquet
from segger.prediction.predict import load_model, predict
from segger.training.segger_data_module import SeggerDataModule


class Workspace:
    def __init__(self, root: Path):
        self.root = root
        self.dataset = root / "dataset"
        self.model = root / "model"
        self.prediction = root / "prediction"

        for d in [self.dataset, self.model, self.prediction]:
            d.mkdir()

    def cleanup(self):
        """Remove subdirectories to save space im `/tmp` after testing."""
        for d in [self.dataset, self.model, self.prediction]:
            shutil.rmtree(d)


@pytest.fixture(scope="module")
def workspace(tmp_path_factory) -> Iterator[Workspace]:
    """
    Create a shared workspace with subdirectories for each pipeline step for
    segger.
    """
    ws = Workspace(tmp_path_factory.mktemp("workspace"))
    yield ws
    ws.cleanup()


@pytest.mark.dependency()
def test_create_dataset(workspace: Workspace, xenium_human_lung_10k: Path):
    """Run segger dataset creation"""
    sample = STSampleParquet(
        xenium_human_lung_10k,
        n_workers=1,
        sample_type="xenium",
    )
    sample.save(workspace.dataset, tile_size=500)
    # Make sure tile directories exist and contain more than 1 tile
    for dirname in ["train_tiles", "test_tiles", "val_tiles"]:
        tile_dir = workspace.dataset / dirname / "processed"
        assert tile_dir.exists()
        assert tile_dir.glob("*.pt")


@pytest.mark.dependency(depends=["test_create_dataset"])
def test_train(workspace: Workspace):
    """Run segger model training"""
    dm = SeggerDataModule(workspace.dataset, batch_size=1, num_workers=1)
    model = Segger(
        num_tx_tokens=500,
        hidden_channels=2,
        num_mid_layers=1,
        init_emb=2,
        out_channels=2,
        heads=1,
    )
    md = (["tx", "bd"], [("tx", "belongs", "bd"), ("tx", "neighbors", "tx")])
    ls = LitSegger(to_hetero(model, metadata=md, aggr="sum"))
    logger = CSVLogger(workspace.model)
    trainer = Trainer(
        default_root_dir=workspace.model, logger=logger, min_epochs=5, max_epochs=5
    )
    trainer.fit(model=ls, datamodule=dm)
    # Make sure usual outputs are there
    assert workspace.model.glob("*/*/metrics.csv")
    assert workspace.model.glob("*/*/checkpoints/*.ckpt")


@pytest.mark.dependency(depends=["test_train"])
def test_predict(workspace: Workspace):
    """Run segger model inference"""
    dm = SeggerDataModule(workspace.dataset, batch_size=1, num_workers=1)
    checkpoint_path = list(workspace.model.glob("*/*/checkpoints/*.ckpt"))[0]
    ls = load_model(checkpoint_path)
    receptive_field = {"k_bd": 2, "dist_bd": 10, "k_tx": 2, "dist_tx": 10}
    predict(workspace.prediction, ls, dm, 0.5, receptive_field)
    # Make sure predictions file exists
    filename = "segger_labeled_transcripts.parquet"
    assert (workspace.prediction / filename).exists()
