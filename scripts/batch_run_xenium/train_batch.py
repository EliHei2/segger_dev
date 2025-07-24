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
parser.add_argument('--models_dir', type=Path, required=True)
args = parser.parse_args()

segger_data_dir = args.data_dir
print(segger_data_dir)
models_dir = args.models_dir

dm = SeggerDataModule(
    data_dir=segger_data_dir,
    batch_size=1,
    num_workers=1,
)
dm.setup()

is_token_based = False
num_tx_tokens = dm.train[0].x_dict["tx"].shape[1]

model = Segger(
    num_tx_tokens=num_tx_tokens,
    init_emb=8,
    hidden_channels=32,
    out_channels=16,
    heads=4,
    num_mid_layers=3,
)
model = to_hetero(model, (["tx", "bd"], [("tx", "belongs", "bd"), ("tx", "neighbors", "tx")]), aggr="sum")

batch = dm.train[0]
model.forward(batch.x_dict, batch.edge_index_dict)

ls = LitSegger(model=model)

trainer = Trainer(
    accelerator="gpu",
    strategy="auto",
    precision="16-mixed", # for 5k change to "32"
    devices=4,
    max_epochs=200, # check models/project_tag/sample/lightning_logs/metrics.csv to check the performance on the last epoch
    default_root_dir=models_dir,
    logger=CSVLogger(models_dir),
)

trainer.fit(ls, datamodule=dm)