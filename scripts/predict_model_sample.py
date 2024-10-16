from segger.training.segger_data_module import SeggerDataModule
from segger.prediction.predict_parquet import segment, load_model
from pathlib import Path
from matplotlib import pyplot as plt
import seaborn as sns
import scanpy as sc
import os
import dask.dataframe as dd
import pandas as pd
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import cupy as cp
from dask.distributed import Client, LocalCluster
from dask_cuda import LocalCUDACluster
import dask.dataframe as dd


segger_data_dir = Path("./data_tidy/pyg_datasets/bc_embedding_1001")
models_dir = Path("./models/bc_embedding_1001_small")
benchmarks_dir = Path("/dkfz/cluster/gpu/data/OE0606/elihei/segger_experiments/data_tidy/benchmarks/xe_rep1_bc")
transcripts_file = "data_raw/xenium/Xenium_FFPE_Human_Breast_Cancer_Rep1/transcripts.parquet"
# Initialize the Lightning data module
dm = SeggerDataModule(
    data_dir=segger_data_dir,
    batch_size=1,
    num_workers=0,
)

dm.setup()


model_version = 0

# Load in latest checkpoint
model_path = models_dir / "lightning_logs" / f"version_{model_version}"
model = load_model(model_path / "checkpoints")

receptive_field = {"k_bd": 4, "dist_bd": 20, "k_tx": 5, "dist_tx": 3}

segment(
    model,
    dm,
    save_dir=benchmarks_dir,
    seg_tag="parquet_test_big",
    transcript_file=transcripts_file,
    # file_format='anndata',
    receptive_field=receptive_field,
    min_transcripts=5,
    score_cut=0.5,
    # max_transcripts=1500,
    cell_id_col="segger_cell_id",
    use_cc=True,
    knn_method="cuda",
    verbose=True,
    gpu_ids=["0"],
    # client=client
)


# if __name__ == "__main__":
#     cluster = LocalCUDACluster(
#         # CUDA_VISIBLE_DEVICES="0",
#         n_workers=1,
#         dashboard_address=":8080",
#         memory_limit='30GB',  # Adjust based on system memory
#         lifetime="2 hours",  # Increase worker lifetime
#         lifetime_stagger="75 minutes",
#         local_directory='.', # Stagger worker restarts
#         lifetime_restart=True  # Automatically restart workers
#     )
#     client = Client(cluster)

# segment(
#     model,
#     dm,
#     save_dir=benchmarks_dir,
#     seg_tag='segger_embedding_0926_mega_0.5_20',
#     transcript_file=transcripts_file,
#     file_format='anndata',
#     receptive_field = receptive_field,
#     min_transcripts=5,
#     score_cut=0.5,
#     # max_transcripts=1500,
#     cell_id_col='segger_cell_id',
#     use_cc=False,
#     knn_method='cuda',
#     # client=client
# )

#     client.close()
#     cluster.close()
