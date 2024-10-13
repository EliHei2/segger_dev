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

# import pandas as pd
from segger.data.utils import calculate_gene_celltype_abundance_embedding
import scanpy as sc
import os

# import Dask.DataFrame as dd

os.environ["DASK_DAEMON"] = "False"

xenium_data_dir = Path(
    "/omics/odcf/analysis/OE0606_projects/oncolgy_data_exchange/20230831-pan-cns-TMA-Xenium/output-XETG00078__0010722__TMA_AKSI__20230831__151713/"
)
segger_data_dir = Path("./data_tidy/pyg_datasets/pan_cns_AKSI")
# models_dir = Path('./models/bc_embedding_1001')

# scRNAseq_path = '/omics/groups/OE0606/internal/tangy/tasks/schier/data/atals_filtered.h5ad'

# scRNAseq = sc.read(scRNAseq_path)

# sc.pp.subsample(scRNAseq, 0.1)

# # Step 1: Calculate the gene cell type abundance embedding
# celltype_column = 'celltype_minor'
# gene_celltype_abundance_embedding = calculate_gene_celltype_abundance_embedding(scRNAseq, celltype_column)


# Setup Xenium sample to create dataset
xs = XeniumSample(verbose=False)  # , embedding_df=gene_celltype_abundance_embedding)
xs.set_file_paths(
    transcripts_path=xenium_data_dir / "transcripts.parquet",
    boundaries_path=xenium_data_dir / "nucleus_boundaries.parquet",
)

# dd.read_parquet(transcripts_path[0])
xs.set_metadata()
# xs.x_max = 1000
# xs.y_max = 1000

try:
    xs.save_dataset_for_segger(
        processed_dir=segger_data_dir,
        x_size=220,
        y_size=220,
        d_x=200,
        d_y=200,
        margin_x=10,
        margin_y=10,
        compute_labels=True,  # Set to True if you need to compute labels
        r_tx=5,
        k_tx=5,
        val_prob=0.3,
        test_prob=0.1,
        num_workers=6,
    )
except AssertionError as err:
    print(f"Dataset already exists at {segger_data_dir}")
