import os
import torch
import cupy as cp
import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch._dynamo
import gc

# import rmm
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

from segger.training.train import LitSegger
from segger.training.segger_data_module import SeggerDataModule
from segger.prediction.boundary import generate_boundaries

from scipy.sparse.csgraph import connected_components as cc
from typing import Union, Dict, Tuple
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


# CONFIG
torch._dynamo.config.suppress_errors = True
os.environ["PYTORCH_USE_CUDA_DSA"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


# Function to zero out diagonal of sparse COO matrix
def zero_out_diagonal_gpu(sparse_matrix):
    """
    Zero out the diagonal elements of a sparse CuPy COO matrix while keeping it sparse on the GPU.

    Args:
        sparse_matrix (cupyx.scipy.sparse.coo_matrix): Input sparse matrix.

    Returns:
        cupyx.scipy.sparse.coo_matrix: Matrix with diagonal elements zeroed out.
    """
    # Filter out the diagonal (where row == col)
    non_diagonal_mask = sparse_matrix.row != sparse_matrix.col

    # Create a new sparse matrix without diagonal elements
    sparse_matrix_no_diag = cupyx.scipy.sparse.coo_matrix(
        (
            sparse_matrix.data[non_diagonal_mask],
            (
                sparse_matrix.row[non_diagonal_mask],
                sparse_matrix.col[non_diagonal_mask],
            ),
        ),
        shape=sparse_matrix.shape,
    )

    return sparse_matrix_no_diag


# Function to subset rows and columns of a sparse matrix
def subset_sparse_matrix(sparse_matrix, row_idx, col_idx):
    """
    Subset a sparse matrix using row and column indices.

    Parameters:
    sparse_matrix (cupyx.scipy.sparse.spmatrix): The input sparse matrix in COO, CSR, or CSC format.
    row_idx (cupy.ndarray): Row indices to keep in the subset.
    col_idx (cupy.ndarray): Column indices to keep in the subset.

    Returns:
    cupyx.scipy.sparse.spmatrix: A new sparse matrix that is a subset of the input matrix.
    """

    # Convert indices to CuPy arrays if not already
    row_idx = cp.asarray(row_idx)
    col_idx = cp.asarray(col_idx)

    # Ensure sparse matrix is in COO format for easy indexing (you can use CSR/CSC if more optimal)
    sparse_matrix = sparse_matrix.tocoo()

    # Create boolean masks for the row and column indices
    row_mask = cp.isin(sparse_matrix.row, row_idx)
    col_mask = cp.isin(sparse_matrix.col, col_idx)

    # Apply masks to filter the data, row, and column arrays
    mask = row_mask & col_mask
    row_filtered = sparse_matrix.row[mask]
    col_filtered = sparse_matrix.col[mask]
    data_filtered = sparse_matrix.data[mask]

    # Map the row and col indices to the new submatrix indices
    row_mapped = cp.searchsorted(row_idx, row_filtered)
    col_mapped = cp.searchsorted(col_idx, col_filtered)

    # Return the new subset sparse matrix
    return coo_matrix(
        (data_filtered, (row_mapped, col_mapped)), shape=(len(row_idx), len(col_idx))
    )


def load_model(checkpoint_path: str) -> LitSegger:
    """
    Load a LitSegger model from a checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Specific checkpoint file to load, or directory where the model checkpoints are stored.
        If directory, the latest checkpoint is loaded.

    Returns
    -------
    LitSegger
        The loaded LitSegger model.

    Raises
    ------
    FileNotFoundError
        If the specified checkpoint file does not exist.
    """
    checkpoint_path = Path(checkpoint_path)
    msg = f"No checkpoint found at {checkpoint_path}. Please make sure you've provided the correct path."

    # Get last checkpoint if directory is provided
    if os.path.isdir(checkpoint_path):
        checkpoints = glob.glob(str(checkpoint_path / "*.ckpt"))
        if len(checkpoints) == 0:
            raise FileNotFoundError(msg)

        # Sort checkpoints by epoch and step
        def sort_order(c):
            match = re.match(r".*epoch=(\d+)-step=(\d+).ckpt", c)
            return int(match[1]), int(match[2])

        checkpoint_path = Path(sorted(checkpoints, key=sort_order)[-1])
    elif not checkpoint_path.exists():
        raise FileExistsError(msg)

    # Load model from checkpoint
    lit_segger = LitSegger.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
    )

    return lit_segger




def get_similarity_scores(
    model: torch.nn.Module,
    batch: Batch,
    from_type: str,
    to_type: str,
    receptive_field: dict,
    compute_sigmoid: bool = True,
    knn_method: str = "kd_tree",
    gpu_id: int = 0,
) -> torch.Tensor:
    """
    Compute similarity scores between embeddings for 'from_type' and 'to_type' nodes
    using sparse matrix multiplication with PyTorch only.

    Returns:
        torch.Tensor: A sparse torch.tensor (COO format) with similarity scores.
    """

    device = torch.device(f"cuda:{gpu_id}")
    batch = batch.to(device)

    # Step 1: Compute shape and get spatial coordinates
    num_from = batch[from_type].x.shape[0]
    num_to = batch[to_type].x.shape[0]
    shape = (num_from, num_to)

    if from_type == to_type:
        coords_1 = coords_2 = batch[to_type].pos
    else:
        coords_1 = batch[to_type].pos[:, :2]
        coords_2 = batch[from_type].pos[:, :2]

    # Get kNN edge indices
    edge_index = get_edge_index(
        coords_1.cpu() if knn_method == "kd_tree" else coords_1,
        coords_2.cpu() if knn_method == "kd_tree" else coords_2,
        k=receptive_field[f"k_{to_type}"],
        dist=receptive_field[f"dist_{to_type}"],
        method=knn_method,
    ).to(device)

    # Convert to padded dense edge_index [num_from, k]
    dense_index = coo_to_dense_adj(
        edge_index.T, num_nodes=num_from, num_nbrs=receptive_field[f"k_{to_type}"]
    ).to(device)

    def get_normalized_embedding(x: torch.Tensor, key: str) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0)
        is_1d = x.ndim == 1
        if is_1d:
            x = x.unsqueeze(1)
        emb = (
            model.tx_embedding[key]((x.sum(-1).int())) if is_1d
            else model.lin0[key](x.float())
        )
        return F.normalize(emb, p=2, dim=1)

    with torch.no_grad():
        if from_type != to_type:
            embeddings = model(batch.x_dict, batch.edge_index_dict)
        else:
            embeddings = {
                key: get_normalized_embedding(x, key)
                for key, x in batch.x_dict.items()
            }

    def sparse_multiply(
        embeddings: Dict[str, torch.Tensor],
        edge_index: torch.Tensor,  # [num_from, k]
        shape: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Compute sparse similarity scores using torch only (no cupy).
        """
        padded_emb = F.pad(embeddings[to_type], (0, 0, 0, 1))  # Add dummy row for -1 padding
        neighbor_embs = padded_emb[edge_index]  # [num_from, k, dim]
        source_embs = embeddings[from_type].unsqueeze(1)  # [num_from, 1, dim]

        similarity = (neighbor_embs * source_embs).sum(dim=-1)  # [num_from, k]

        valid_mask = edge_index != -1
        row_idx = torch.arange(edge_index.size(0), device=device).unsqueeze(1).expand_as(edge_index)
        row_valid = row_idx[valid_mask]
        col_valid = edge_index[valid_mask]
        val_valid = similarity[valid_mask]

        if compute_sigmoid:
            val_valid = torch.sigmoid(val_valid)

        indices = torch.stack([row_valid, col_valid], dim=0)
        return torch.sparse_coo_tensor(indices, val_valid, shape, device=device).coalesce()

    return sparse_multiply(embeddings, dense_index, shape)


def predict_batch(
    lit_segger: torch.nn.Module,
    batch: Batch,
    score_cut: float,
    receptive_field: Dict[str, float],
    use_cc: bool = True,
    knn_method: str = "cuda",
    edge_index_save_path: Union[str, Path] = None,
    output_ddf_save_path: Union[str, Path] = None,
    gpu_id: int = 0,
):
    def _get_id():
        return "".join(np.random.choice(list("abcdefghijklmnopqrstuvwxyz"), 8)) + "-nx"

    device = f"cuda:{gpu_id}"
    batch = batch.to(device)
    lit_segger.model = lit_segger.model.to(device)

    transcript_id = batch["tx"].id.cpu().numpy().astype("str")
    assignments = {"transcript_id": transcript_id}

    if len(batch["bd"].pos) >= 10 and len(batch["tx"].pos) >= 1000:
        # Step 1: Compute similarity scores
        scores = get_similarity_scores(
            lit_segger.model,
            batch,
            "tx",
            "bd",
            receptive_field,
            knn_method=knn_method,
            gpu_id=gpu_id,
        )
        torch.cuda.empty_cache()

        # Convert sparse torch.tensor to dense
        dense_scores = scores.to_dense().cpu().numpy()
        del scores

        belongs = dense_scores.max(axis=1)
        assignments["score"] = belongs

        mask = belongs >= score_cut
        all_ids = np.concatenate(batch["bd"].id)
        max_idx = dense_scores.argmax(axis=1)
        assigned_ids = np.where(mask, all_ids[max_idx], None)
        assignments["segger_cell_id"] = assigned_ids
        assignments["bound"] = mask.astype(np.int8)

        del dense_scores
        torch.cuda.empty_cache()

        if use_cc:
            scores_tx = get_similarity_scores(
                lit_segger.model,
                batch,
                "tx",
                "tx",
                receptive_field,
                compute_sigmoid=False,
                knn_method=knn_method,
                gpu_id=gpu_id,
            )
            values = scores_tx.values()
            threshold = values.median().item()
            values[values < threshold] = 0
            scores_tx = torch.sparse_coo_tensor(
                scores_tx.indices(), values, scores_tx.shape, device=device
            ).coalesce()

            # zero-out diagonal
            indices = scores_tx.indices()
            mask = indices[0] != indices[1]
            filtered_indices = indices[:, mask]
            filtered_values = values[mask]
            scores_tx = torch.sparse_coo_tensor(
                filtered_indices, filtered_values, scores_tx.shape, device=device
            ).coalesce()

            # Unassigned transcripts
            segger_cell_id_arr = assignments["segger_cell_id"]
            no_id_mask = np.array([v is None for v in segger_cell_id_arr])
            no_id = np.where(no_id_mask)[0]

            if len(no_id) > 0:
                # Filter submatrix for unassigned transcripts
                keep = torch.tensor(no_id, device=device)
                id_map = {int(i): idx for idx, i in enumerate(no_id)}
                edge_i = scores_tx.indices()
                valid_mask = torch.isin(edge_i[0], keep) & torch.isin(edge_i[1], keep)

                sub_i = edge_i[:, valid_mask]
                sub_v = scores_tx.values()[valid_mask]
                sub_v[sub_v < score_cut] = 0
                valid = sub_v > 0
                sub_i = sub_i[:, valid]
                sub_v = sub_v[valid]

                if sub_i.numel() > 0:
                    edge_df = pd.DataFrame({
                        "source": [transcript_id[i.item()] for i in sub_i[0]],
                        "target": [transcript_id[i.item()] for i in sub_i[1]],
                    })

                    edge_index_ddf = delayed(dd.from_pandas)(edge_df, npartitions=1)
                    delayed_write_edge_index = delayed(edge_index_ddf.to_parquet)(
                        edge_index_save_path, append=True, ignore_divisions=True
                    )
                    delayed_write_edge_index.persist()

        # Final cleanup
        assignments = {
            "transcript_id": assignments["transcript_id"].astype("str"),
            "score": assignments["score"].astype("float32"),
            "segger_cell_id": assignments["segger_cell_id"].astype("str"),
            "bound": assignments["bound"].astype("int8"),
        }

        assignments = pd.DataFrame(assignments)
        assignments = assignments[assignments["bound"] == 1]
        batch_ddf = delayed(dd.from_pandas)(assignments, npartitions=1)
        delayed_write_output_ddf = delayed(batch_ddf.to_parquet)(
            output_ddf_save_path, append=True, ignore_divisions=True
        )
        delayed_write_output_ddf.persist()

        torch.cuda.empty_cache()


def segment(
    model: LitSegger,
    dm: SeggerDataModule,
    save_dir: Union[str, Path],
    seg_tag: str,
    transcript_file: Union[str, Path],
    score_cut: float = 0.5,
    use_cc: bool = True,
    file_format: str = "",
    save_transcripts: bool = True,
    save_anndata: bool = True,
    save_cell_masks: bool = False,  # Placeholder for future implementation
    receptive_field: dict = {"k_bd": 4, "dist_bd": 10, "k_tx": 5, "dist_tx": 3},
    knn_method: str = "cuda",
    verbose: bool = False,
    gpu_ids: list = ["0"],
    **anndata_kwargs,
) -> None:
    """
    Perform segmentation using the model, save transcripts, AnnData, and cell masks as needed,
    and log the parameters used during segmentation.

    Args:
        model (LitSegger): The trained segmentation model.
        dm (SeggerDataModule): The SeggerDataModule instance for data loading.
        save_dir (Union[str, Path]): Directory to save the final segmentation results.
        seg_tag (str): Tag to include in the saved filename.
        transcript_file (Union[str, Path]): Path to the transcripts Parquet file.
        score_cut (float, optional): The threshold for assigning transcripts to cells based on
                                     similarity scores. Defaults to 0.5.
        use_cc (bool, optional): If True, perform connected components analysis for unassigned
                                 transcripts. Defaults to True.
        save_transcripts (bool, optional): Whether to save the transcripts as Parquet. Defaults to True.
        save_anndata (bool, optional): Whether to save the results in AnnData format. Defaults to True.
        save_cell_masks (bool, optional): Save cell masks as Dask Geopandas Parquet. Defaults to False.
        receptive_field (dict, optional): Defines the receptive field for transcript-cell and
                                          transcript-transcript relations. Defaults to
                                          {'k_bd': 4, 'dist_bd': 10, 'k_tx': 5, 'dist_tx': 3}.
        knn_method (str, optional): The method to use for nearest neighbors ('cuda' or 'kd_tree').
                                    Defaults to 'cuda'.
        verbose (bool, optional): Whether to print verbose status updates. Defaults to False.
        **anndata_kwargs: Additional keyword arguments passed to the `create_anndata` function.

    Returns:
        None. Saves the result to disk in various formats and logs the parameter choices.
    """

    start_time = time()

    # Create a subdirectory with important parameter info (receptive field values)
    sub_dir_name = f"{seg_tag}_{score_cut}_{use_cc}_{receptive_field['k_bd']}_{receptive_field['dist_bd']}_{receptive_field['k_tx']}_{receptive_field['dist_tx']}_{datetime.now().strftime('%Y%m%d')}"
    save_dir = Path(save_dir) / sub_dir_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Paths for saving the output_ddf and edge_index Parquet files
    output_ddf_save_path = save_dir / "transcripts_df.parquet"
    edge_index_save_path = save_dir / "edge_index.parquet"

    if output_ddf_save_path.exists():
        warnings.warn(f"Removing existing file: {output_ddf_save_path}")
        shutil.rmtree(output_ddf_save_path)

    if use_cc:
        if edge_index_save_path.exists():
            warnings.warn(f"Removing existing file: {edge_index_save_path}")
            shutil.rmtree(edge_index_save_path)

    if verbose:
        print(f"Starting segmentation for {seg_tag}...")

    # Step 1: Load the data loaders from the SeggerDataModule
    step_start_time = time()
    train_dataloader = dm.train_dataloader()
    val_dataloader = dm.val_dataloader()
    test_dataloader = dm.test_dataloader()

    # Loop through the data loaders (train, val, and test)
    for loader_name, loader in zip(
        ["Train", "Validation", "Test"],
        [train_dataloader, val_dataloader, test_dataloader],
    ):
        # for loader_name, loader in zip(['Test'], [test_dataloader]):
        if verbose:
            print(f"Processing {loader_name} data...")

        for batch in tqdm(loader, desc=f"Processing {loader_name} batches"):
            gpu_id = random.choice(gpu_ids)
            # Call predict_batch for each batch
            predict_batch(
                model,
                batch,
                score_cut,
                receptive_field,
                use_cc=use_cc,
                knn_method=knn_method,
                edge_index_save_path=edge_index_save_path,
                output_ddf_save_path=output_ddf_save_path,
                gpu_id=gpu_id,
            )

    if verbose:
        elapsed_time = time() - step_start_time
        print(f"Batch processing completed in {elapsed_time:.2f} seconds.")

    seg_final_dd = pd.read_parquet(output_ddf_save_path)

    step_start_time = time()
    if verbose:
        print(f"Applying max score selection logic...")
    output_ddf_save_path = save_dir / "transcripts_df.parquet"

    seg_final_dd = pd.read_parquet(output_ddf_save_path)

    seg_final_filtered = seg_final_dd.sort_values(
        "score", ascending=False
    ).drop_duplicates(subset="transcript_id", keep="first")

    if verbose:
        elapsed_time = time() - step_start_time
        print(f"Max score selection completed in {elapsed_time:.2f} seconds.")

    # Step 3: Load the transcripts DataFrame and merge results

    if verbose:
        print(f"Loading transcripts from {transcript_file}...")

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

    if use_cc:
        step_start_time = time()
        if verbose:
            print(f"Computing connected components for unassigned transcripts...")

        edge_index_dd = pd.read_parquet(edge_index_save_path)

        transcript_ids_in_edges = pd.concat(
            [edge_index_dd["source"], edge_index_dd["target"]]
        ).unique()

        lookup_table = pd.Series(
            data=np.arange(len(transcript_ids_in_edges)),
            index=transcript_ids_in_edges,
        ).to_dict()

        edge_index_dd["index_source"] = edge_index_dd["source"].map(lookup_table)
        edge_index_dd["index_target"] = edge_index_dd["target"].map(lookup_table)

        source_indices = edge_index_dd["index_source"].to_numpy()
        target_indices = edge_index_dd["index_target"].to_numpy()
        data = np.ones(len(source_indices), dtype=np.uint8)

        coo_matrix_cpu = scipy_coo_matrix(
            (data, (source_indices, target_indices)),
            shape=(len(transcript_ids_in_edges), len(transcript_ids_in_edges)),
        )

        n, comps = cc(coo_matrix_cpu, directed=True, connection="strong")

        if verbose:
            elapsed_time = time() - step_start_time
            print(
                f"Computed connected components for unassigned transcripts in {elapsed_time:.2f} seconds."
            )

        step_start_time = time()
        if verbose:
            print(f"Mapping component labels...")

        def _get_id():
            return "".join(np.random.choice(list("abcdefghijklmnopqrstuvwxyz"), 8)) + "-nx"

        new_ids = np.array([_get_id() for _ in range(n)])
        comp_labels = new_ids[comps]
        comp_labels = pd.Series(comp_labels, index=transcript_ids_in_edges)

        unassigned_mask = transcripts_df_filtered["segger_cell_id"].isna()
        unassigned_transcripts_df = transcripts_df_filtered.loc[
            unassigned_mask, ["transcript_id"]
        ]
        new_segger_cell_ids = unassigned_transcripts_df["transcript_id"].map(comp_labels)
        unassigned_transcripts_df = unassigned_transcripts_df.assign(
            segger_cell_id=new_segger_cell_ids
        )

        transcripts_df_filtered = transcripts_df_filtered.merge(
            unassigned_transcripts_df[["transcript_id", "segger_cell_id"]],
            on="transcript_id",
            how="left",
            suffixes=("", "_new"),
        )
        transcripts_df_filtered["segger_cell_id"] = transcripts_df_filtered[
            "segger_cell_id"
        ].fillna(transcripts_df_filtered["segger_cell_id_new"])
        transcripts_df_filtered.drop(columns=["segger_cell_id_new"], inplace=True)

        if verbose:
            elapsed_time = time() - step_start_time
            print(f"Component labels merged in {elapsed_time:.2f} seconds.")

    # Step 5: Save the merged results based on options
    transcripts_df_filtered["segger_cell_id"] = transcripts_df_filtered[
        "segger_cell_id"
    ].fillna("UNASSIGNED")
    # transcripts_df_filtered = filter_transcripts(transcripts_df_filtered, qv=qv)

    if save_transcripts:
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

    if save_anndata:
        if verbose:
            step_start_time = time()
            print(f"Saving anndata object...")
        anndata_save_path = save_dir / "segger_adata.h5ad"
        segger_adata = create_anndata(
            transcripts_df_filtered, **anndata_kwargs
        )  # Compute for AnnData
        segger_adata.write(anndata_save_path)
        if verbose:
            elapsed_time = time() - step_start_time
            print(f"Saved anndata object in {elapsed_time:.2f} seconds.")

    if save_cell_masks:
        if verbose:
            step_start_time = time()
            print(f"Computing and saving cell masks anndata object...")
        # Placeholder for future cell masks implementation as Dask Geopandas Parquet
        boundaries_gdf = generate_boundaries(transcripts_df_filtered)
        cell_masks_save_path = save_dir / "segger_cell_boundaries.parquet"

        boundaries_gdf.to_parquet(cell_masks_save_path)
        if verbose:
            elapsed_time = time() - step_start_time
            print(f"Saved cell masks in {elapsed_time:.2f} seconds.")

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
