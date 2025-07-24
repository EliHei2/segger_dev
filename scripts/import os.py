import os
import torch
import cupy as cp
import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch._dynamo
import gc
import re
import glob
from pathlib import Path
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from segger.data.utils import (
    get_edge_index,
    format_time,
    create_anndata,
    coo_to_dense_adj,
    filter_transcripts,
)

import anndata as ad

from segger.training.train import LitSegger
from segger.training.segger_data_module import SeggerDataModule
from segger.prediction.boundary import generate_boundaries

from scipy.sparse.csgraph import connected_components as cc
from typing import Union, Dict, Tuple, Optional
import dask.dataframe as dd
from dask import delayed
from dask.diagnostics import ProgressBar
import time
import dask

# from rmm.allocators.cupy import rmm_cupy_allocator
from cupyx.scipy.sparse import coo_matrix
from torch.utils.dlpack import to_dlpack, from_dlpack

from dask.distributed import Client, LocalCluster
import cupy as cp
import numpy as np
import pandas as pd
from cupyx.scipy.sparse import coo_matrix
from cupyx.scipy.sparse import find  # To find non-zero elements in sparse matrix
from scipy.sparse.csgraph import connected_components as cc
from scipy.sparse import coo_matrix as scipy_coo_matrix
from dask.distributed import get_client
from pqdm.processes import pqdm
from tqdm import tqdm
import json
from datetime import datetime
import dask_geopandas as dgpd  # Assuming dask-geopandas is installed

# import cudf
# import dask_cudf
import cupy as cp
import cupyx
import warnings
import shutil
from time import time
from cupyx.scipy.sparse import coo_matrix as cp_coo_matrix
from cupyx.scipy.sparse.csgraph import connected_components as cp_cc
import random

# Setup Dask cluster with 3 workers

verbose=True

save_dir = Path('/dkfz/cluster/gpu/data/OE0606/elihei/segger_experiments/data_tidy/benchmarks/output-XETG00078__0041722__Region_1__20241203__142052/output-XETG00078__0041722__Region_1__20241203__142052_0.75_False_4_10_5_3_20250616')

seg_final_filtered = pd.read_parquet(save_dir / 'transcripts_df.parquet')


seg_final_filtered = seg_final_filtered.sort_values(
    "score", ascending=False
).drop_duplicates(subset="transcript_id", keep="first")


transcript_file = Path('/omics/odcf/analysis/OE0606_projects/xenium_projects/temp/20241209_Xenium5k_CNSL_BrM/20241209_Xenium5k_CNSL_BrM/output-XETG00078__0041722__Region_1__20241203__142052//transcripts.parquet')


transcripts_df = pd.read_parquet(transcript_file)
transcripts_df["transcript_id"] = transcripts_df["transcript_id"].astype(str)

step_start_time = time()
if verbose:
    print(f"Merging segmentation results with transcripts...")

# Outer merge to include all transcripts, even those without assigned cell ids
transcripts_df_filtered = transcripts_df.merge(
    seg_final_filtered, on="transcript_id", how="outer"
)

if verbose:
    elapsed_time = time() - step_start_time
    print(
        f"Merged segmentation results with transcripts in {elapsed_time:.2f} seconds."
    )



# Step 5: Save the merged results based on options
transcripts_df_filtered["segger_cell_id"] = transcripts_df_filtered[
    "segger_cell_id"
].fillna("UNASSIGNED")
# transcripts_df_filtered = filter_transcripts(transcripts_df_filtered, qv=qv)


if verbose:
    step_start_time = time()
    print(f"Saving transcripts.parquet...")
transcripts_save_path = save_dir / "segger_transcripts.parquet"
# transcripts_df_filtered = transcripts_df_filtered.repartition(npartitions=100)
transcripts_df_filtered.to_parquet(
    transcripts_save_path,
    engine="pyarrow",  # PyArrow is faster and recommended
    compression="snappy",  # Use snappy compression for speed
    # write_index=False,  # Skip writing index if not needed
    # append=False,  # Set to True if you're appending to an existing Parquet file
    # overwrite=True,
)  # Dask handles Parquet well
if verbose:
    elapsed_time = time() - step_start_time
    print(f"Saved trasncripts.parquet in {elapsed_time:.2f} seconds.")


if verbose:
    step_start_time = time()
    print(f"Saving anndata object...")


def create_anndata2(
    df: pd.DataFrame,
    panel_df: Optional[pd.DataFrame] = None,
    min_transcripts: int = 5,
    cell_id_col: str = "cell_id",
    qv_threshold: float = 30,
    min_cell_area: float = 10.0,
    max_cell_area: float = 1000.0,
) -> ad.AnnData:
    """
    Generates an AnnData object from a dataframe of segmented transcriptomics data.

    Parameters:
        df (pd.DataFrame): The dataframe containing segmented transcriptomics data.
        panel_df (Optional[pd.DataFrame]): The dataframe containing panel information.
        min_transcripts (int): The minimum number of transcripts required for a cell to be included.
        cell_id_col (str): The column name representing the cell ID in the input dataframe.
        qv_threshold (float): The quality value threshold for filtering transcripts.
        min_cell_area (float): The minimum cell area to include a cell.
        max_cell_area (float): The maximum cell area to include a cell.

    Returns:
        ad.AnnData: The generated AnnData object containing the transcriptomics data and metadata.
    """
    # Filter out unassigned cells
    df_filtered = df[df[cell_id_col].astype(str) != "UNASSIGNED"]
    # Create pivot table for gene expression counts per cell
    pivot_df = df_filtered.rename(
        columns={cell_id_col: "cell", "feature_name": "gene"}
    )[["cell", "gene"]].pivot_table(
        index="cell", columns="gene", aggfunc="size", fill_value=0
    )
    pivot_df = pivot_df[pivot_df.sum(axis=1) >= min_transcripts]
    # Summarize cell metrics
    cell_summary = []
    for cell_id, cell_data in df_filtered.groupby(cell_id_col):
        if len(cell_data) < min_transcripts:
            continue
        cell_convex_hull = ConvexHull(
            cell_data[["x_location", "y_location"]], qhull_options="QJ"
        )
        cell_area = cell_convex_hull.area
        if cell_area < min_cell_area or cell_area > max_cell_area:
            continue
        cell_summary.append(
            {
                "cell": cell_id,
                "cell_centroid_x": cell_data["x_location"].mean(),
                "cell_centroid_y": cell_data["y_location"].mean(),
                "cell_area": cell_area,
            }
        )
    cell_summary = pd.DataFrame(cell_summary).set_index("cell")
    # Add genes from panel_df (if provided) to the pivot table
    if panel_df is not None:
        panel_df = panel_df.sort_values("gene")
        genes = panel_df["gene"].values
        for gene in genes:
            if gene not in pivot_df:
                pivot_df[gene] = 0
        pivot_df = pivot_df[genes.tolist()]
    # Create var DataFrame
    if panel_df is None:
        var_df = pd.DataFrame(
            [
                {"gene": gene, "feature_types": "Gene Expression", "genome": "Unknown"}
                for gene in np.unique(pivot_df.columns.values)
            ]
        ).set_index("gene")
    else:
        var_df = panel_df[["gene", "ensembl"]].rename(columns={"ensembl": "gene_ids"})
        var_df["feature_types"] = "Gene Expression"
        var_df["genome"] = "Unknown"
        var_df = var_df.set_index("gene")
    # Compute total assigned and unassigned transcript counts for each gene
    assigned_counts = df_filtered.groupby("feature_name")["feature_name"].count()
    unassigned_counts = (
        df[df[cell_id_col].astype(str) == "UNASSIGNED"]
        .groupby("feature_name")["feature_name"]
        .count()
    )
    # var_df["total_assigned"] = var_df.index.map(assigned_counts).fillna(0).astype(int)
    # var_df["total_unassigned"] = (
    #     var_df.index.map(unassigned_counts).fillna(0).astype(int)
    # )
    # Filter cells and create the AnnData object
    cells = list(set(pivot_df.index) & set(cell_summary.index))
    pivot_df = pivot_df.loc[cells, :]
    cell_summary = cell_summary.loc[cells, :]
    adata = ad.AnnData(pivot_df.values)
    adata.var = var_df
    adata.obs["transcripts"] = pivot_df.sum(axis=1).values
    adata.obs["unique_transcripts"] = (pivot_df > 0).sum(axis=1).values
    adata.obs_names = pivot_df.index.values.tolist()
    adata.obs = pd.merge(
        adata.obs,
        cell_summary.loc[adata.obs_names, :],
        left_index=True,
        right_index=True,
    )
    return adata

transcripts_df_filtered = pd.read_parquet(transcripts_save_path)
anndata_save_path = save_dir / "segger_adata.h5ad"
segger_adata = create_anndata(
    transcripts_df_filtered, min_transcripts=5, cell_id_col='segger_cell_id' #**anndata_kwargs
)  # Compute for AnnData
segger_adata.write(anndata_save_path)
if verbose:
    elapsed_time = time() - step_start_time
    print(f"Saved anndata object in {elapsed_time:.2f} seconds.")

if verbose:
elapsed_time = time() - step_start_time
print(f"Results saved in {elapsed_time:.2f} seconds at {save_dir}.")

# Step 6: Save segmentation parameters as a JSON log
log_data = {
    "seg_tag": seg_tag,
    "score_cut": score_cut,
    "use_cc": use_cc,
    "receptive_field": receptive_field,
    "knn_method": knn_method,
    "save_transcripts": save_transcripts,
    "save_anndata": save_anndata,
    "save_cell_masks": save_cell_masks,
    "timestamp": datetime.now().isoformat(),
}

log_path = save_dir / "segmentation_log.json"
with open(log_path, "w") as log_file:
    json.dump(log_data, log_file, indent=4)

# Step 7: Garbage collection and memory cleanup
torch.cuda.empty_cache()
gc.collect()

# Total time taken for the segmentation process
if verbose:
    total_time = time() - start_time
    print(f"Total segmentation process completed in {total_time:.2f} seconds.")
