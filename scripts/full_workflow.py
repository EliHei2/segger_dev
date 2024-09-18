from segger.data.io import XeniumSample
from segger.training.train import LitSegger
from segger.training.segger_data_module import SeggerDataModule
from segger.prediction.predict import predict, load_model
from lightning.pytorch.loggers import CSVLogger
from pytorch_lightning import Trainer
from pathlib import Path
from lightning.pytorch.plugins.environments import LightningEnvironment
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd


# xenium_data_dir = Path('./data_tidy/pyg_datasets/fixed_0911/xenium_pancreas_cancer')
segger_data_dir = Path('./data_tidy/pyg_datasets/fixed_0911')

# Base directory to store Pytorch Lightning models
models_dir = Path('./models/fixed_0911')

# Initialize the Lightning model
metadata = (["tx", "bd"], [("tx", "belongs", "bd"), ("tx", "neighbors", "tx")])
ls = LitSegger(
    num_tx_tokens=500,
    init_emb=8,
    hidden_channels=32,
    out_channels=8,
    heads=2,
    num_mid_layers=2,
    aggr='sum',
    metadata=metadata,
)

# Initialize the Lightning data module
dm = SeggerDataModule(
    data_dir=segger_data_dir,
    batch_size=4,  
    num_workers=2,  
)

# Initialize the Lightning trainer
trainer = Trainer(
    accelerator='cuda',  
    strategy='auto',
    precision='16-mixed',
    devices=2,  
    max_epochs=100,  
    default_root_dir=models_dir,
    logger=CSVLogger(models_dir),
)


trainer.fit(
    model=ls,
    datamodule=dm
)

model_version = 0  # 'v_num' from training output above
model_path = models_dir / 'lightning_logs' / f'version_{model_version}'
metrics = pd.read_csv(model_path / 'metrics.csv', index_col=1)

fig, ax = plt.subplots(1,1, figsize=(2,2))

for col in metrics.columns.difference(['epoch']):
    metric = metrics[col].dropna()
    ax.plot(metric.index, metric.values, label=col)

ax.legend(loc=(1, 0.33))
ax.set_ylim(0, 1)
ax.set_xlabel('Step')

model_version = 0

# Load in latest checkpoint
model_path = models_dir / 'lightning_logs' / f'version_{model_version}'
model = load_model(model_path / 'checkpoints')
dm.setup()

receptive_field = {'k_bd': 3, 'dist_bd': 20,'k_tx': 30, 'dist_tx': 5}

# Perform segmentation (predictions)
segmentation = predict(
    model,
    dm.train_dataloader(),
    score_cut=0.2,  
    receptive_field=receptive_field,
    use_cc=False,
)