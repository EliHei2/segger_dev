import argparse
from pathlib import Path
from segger.training.segger_data_module import SeggerDataModule
from segger.models.segger_model import Segger
from segger.training.train import LitSegger
from torch_geometric.nn import to_hetero
from lightning.pytorch.loggers import CSVLogger
from lightning import Trainer

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=Path, required=True)
args = parser.parse_args()

segger_data_dir = args.data_dir
models_dir = Path("./models") / segger_data_dir.relative_to("data_tidy/pyg_datasets")

dm = SeggerDataModule(
    data_dir=segger_data_dir,
    batch_size=2,
    num_workers=2,
)
dm.setup()

is_token_based = False
num_tx_tokens = dm.train[0].x_dict["tx"].shape[1]

model = Segger(
    num_tx_tokens=num_tx_tokens,
    init_emb=8,
    hidden_channels=64,
    out_channels=16,
    heads=4,
    num_mid_layers=2,
)
model = to_hetero(model, (["tx", "bd"], [("tx", "belongs", "bd"), ("tx", "neighbors", "tx")]), aggr="sum")

ls = LitSegger(model=model)

trainer = Trainer(
    accelerator="gpu",
    strategy="auto",
    precision="16-mixed",
    devices=4,
    max_epochs=250,
    default_root_dir=models_dir,
    logger=CSVLogger(models_dir),
)

trainer.fit(ls, datamodule=dm)