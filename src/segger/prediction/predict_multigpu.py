import os
import torch
import cupy as cp
import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch._dynamo
import gc
import rmm
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
)
from segger.training.train import LitSegger
from segger.training.segger_data_module import SeggerDataModule
from scipy.sparse.csgraph import connected_components as cc
from typing import Union, Dict
import dask.dataframe as dd
from dask import delayed
from dask.diagnostics import ProgressBar
import time
import dask
from rmm.allocators.cupy import rmm_cupy_allocator
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
import cudf
import dask_cudf
import cupy as cp
import cupyx
import warnings
import shutil
from time import time
from cupyx.scipy.sparse import coo_matrix as cp_coo_matrix
from cupyx.scipy.sparse.csgraph import connected_components as cp_cc
from dask.distributed import LocalCluster, Client, progress
import dask
import random
from dask_cuda import LocalCUDACluster

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
            (sparse_matrix.row[non_diagonal_mask], sparse_matrix.col[non_diagonal_mask]),
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
    return coo_matrix((data_filtered, (row_mapped, col_mapped)), shape=(len(row_idx), len(col_idx)))


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
    knn_method: str = "cuda",
    gpu_id: int = 0,  # Added argument for GPU ID
) -> coo_matrix:
    """
    Compute similarity scores between embeddings for 'from_type' and 'to_type' nodes
    using sparse matrix multiplication with CuPy and the 'sees' edge relation.

    Args:
        model (torch.nn.Module): The segmentation model used to generate embeddings.
        batch (Batch): A batch of data containing input features and edge indices.
        from_type (str): The type of node from which the similarity is computed.
        to_type (str): The type of node to which the similarity is computed.
        knn_method (str, optional): The method to use for nearest neighbors. Defaults to 'cuda'.
        gpu_id (int, optional): The GPU ID to use for the computations. Defaults to 0.

    Returns:
        coo_matrix: A sparse matrix containing the similarity scores between
                    'from_type' and 'to_type' nodes.
    """

    with cp.cuda.Device(gpu_id):
        # Move the batch to the specified GPU
        batch = batch.to(f"cuda:{gpu_id}")

        # Step 1: Get embeddings from the model (on GPU)
        shape = batch[from_type].x.shape[0], batch[to_type].x.shape[0]
        # Compute edge indices using knn method (still on GPU)
        edge_index = get_edge_index(
            batch[to_type].pos[:, :2].cpu(),  # 'tx' positions
            batch[from_type].pos[:, :2].cpu(),  # 'bd' positions
            k=receptive_field[f"k_{to_type}"],
            dist=receptive_field[f"dist_{to_type}"],
            method=knn_method,
        )

        # Convert to dense adjacency matrix (on GPU)
        edge_index = coo_to_dense_adj(edge_index.T, num_nodes=shape[0], num_nbrs=receptive_field[f"k_{to_type}"])

        with torch.no_grad():
            embeddings = model(batch.x_dict, batch.edge_index_dict)

        def sparse_multiply(embeddings, edge_index, shape) -> coo_matrix:
            m = torch.nn.ZeroPad2d((0, 0, 0, 1))  # pad bottom with zeros

            similarity = torch.bmm(
                m(embeddings[to_type])[edge_index],  # 'to' x 'from' neighbors x embed
                embeddings[from_type].unsqueeze(-1),  # 'to' x embed x 1
            )  # -> 'to' x 'from' neighbors x 1
            del embeddings
            # Sigmoid to get most similar 'to_type' neighbor
            similarity[similarity == 0] = -torch.inf  # ensure zero stays zero
            similarity = F.sigmoid(similarity)
            # Neighbor-filtered similarity scores
            # shape = batch[from_type].x.shape[0], batch[to_type].x.shape[0]
            indices = torch.argwhere(edge_index != -1).T
            indices[1] = edge_index[edge_index != -1]
            rows = cp.fromDlpack(to_dlpack(indices[0, :].to("cuda")))
            columns = cp.fromDlpack(to_dlpack(indices[1, :].to("cuda")))
            # print(rows)
            del indices
            values = similarity[edge_index != -1].flatten()
            sparse_result = coo_matrix((cp.fromDlpack(to_dlpack(values)), (rows, columns)), shape=shape)
            return sparse_result
            # Free GPU memory after computation

        # Call the sparse multiply function
        sparse_similarity = sparse_multiply(embeddings, edge_index, shape)

    return sparse_similarity


def predict_batch(
    lit_segger: torch.nn.Module,
    batch: Batch,
    score_cut: float,
    receptive_field: Dict[str, float],
    use_cc: bool = True,
    knn_method: str = "cuda",
    edge_index_save_path: Union[str, Path] = None,
    output_ddf_save_path: Union[str, Path] = None,
    gpu_id: int = 0,  # Added argument for GPU ID
):
    """
    Predict cell assignments for a batch of transcript data using a segmentation model.
    Writes both the assignments and edge_index directly into Parquet files incrementally.

    Args:
        lit_segger (torch.nn.Module): The lightning module wrapping the segmentation model.
        batch (Batch): A batch of transcript and cell data.
        score_cut (float): The threshold for assigning transcripts to cells based on similarity scores.
        receptive_field (Dict[str, float]): Dictionary defining the receptive field for transcript-cell
                                            and transcript-transcript relations.
        use_cc (bool, optional): If True, perform connected components analysis for unassigned transcripts.
                                 Defaults to True.
        knn_method (str, optional): The method to use for nearest neighbors. Defaults to 'cuda'.
        edge_index_save_path (str, optional): Path to the Parquet file where edge indices are saved incrementally.
        output_ddf_save_path (str, optional): Path to the Parquet file where transcript assignments (`output_ddf`)
                                              are saved incrementally.
        gpu_id (int, optional): The GPU ID to use for the computations. Defaults to 0.
    """

    def _get_id():
        """Generate a random Xenium-style ID."""
        return "".join(np.random.choice(list("abcdefghijklmnopqrstuvwxyz"), 8)) + "-nx"

    print(gpu_id)
    with cp.cuda.Device(gpu_id):
        # Move the batch to the specified GPU
        batch = batch.to(f"cuda:{gpu_id}")
        lit_segger.model = lit_segger.model.to(f"cuda:{gpu_id}")

        # Extract transcript IDs and initialize a dictionary for assignments
        transcript_id = batch["tx"].id.cpu().numpy().astype("str")
        assignments = {"transcript_id": transcript_id}

        if len(batch["bd"].pos) >= 10:
            while True:
                try:
                    # Step 1: Compute similarity scores between 'tx' (transcripts) and 'bd' (boundaries)
                    scores = get_similarity_scores(
                        lit_segger.model, batch, "tx", "bd", receptive_field, knn_method=knn_method, gpu_id=gpu_id
                    )
                    break
                except Exception as e:
                    print(f"This weird error: {e}.")
            torch.cuda.empty_cache()

            # Convert sparse matrix to dense format (on GPU)
            dense_scores = scores.toarray()  # Convert to dense NumPy array
            del scores  # Remove from memory
            cp.get_default_memory_pool().free_all_blocks()  # Free CuPy memory

            # Step 2: Maximize score and assign transcripts based on score threshold
            belongs = cp.max(dense_scores, axis=1)  # Max score per transcript
            assignments["score"] = cp.asnumpy(belongs)  # Move back to CPU

            mask = assignments["score"] >= score_cut  # Mask for assigned transcripts
            all_ids = batch["bd"].id  # Boundary IDs as NumPy array
            assignments["segger_cell_id"] = np.where(mask, all_ids[cp.argmax(dense_scores, axis=1).get()], None)
            # Clear memory after score processing
            del dense_scores
            cp.get_default_memory_pool().free_all_blocks()  # Free CuPy memory
            torch.cuda.empty_cache()

            assignments["bound"] = np.where(mask, 1, 0)  # Mark as 'bound' (1 if assigned, 0 if unassigned)

            # Step 3: Handle unassigned transcripts with connected components (if use_cc=True)
            if use_cc:
                scores_tx = get_similarity_scores(
                    lit_segger.model, batch, "tx", "tx", receptive_field, knn_method=knn_method, gpu_id=gpu_id
                )

                # Stay on GPU and use CuPy sparse matrices
                no_id_scores = cupyx.scipy.sparse.coo_matrix(
                    (scores_tx.data, (scores_tx.row, scores_tx.col)), shape=scores_tx.shape
                )

                # Apply threshold on GPU
                no_id_scores.data[no_id_scores.data < score_cut] = 0  # Apply threshold

                # Zero out the diagonal on GPU
                no_id_scores = zero_out_diagonal_gpu(no_id_scores)
                no_id_scores.eliminate_zeros()  # Remove zero entries to keep the matrix sparse

                # Find unassigned transcripts (those with no segger_cell_id)
                no_id = cp.where(cp.asarray(assignments["segger_cell_id"] == None))[
                    0
                ]  # Using CuPy to handle None values

                if len(no_id) > 0:  # Only compute if there are unassigned transcripts
                    # Apply score cut-off to unassigned transcripts
                    no_id_scores = subset_sparse_matrix(no_id_scores, no_id, no_id)
                    no_id_scores.data[no_id_scores.data < score_cut] = 0  # Apply threshold
                    no_id_scores.eliminate_zeros()  # Clean up zeros

                    # Find the non-zero entries in the no_id_scores to construct edge_index
                    non_zero_rows, non_zero_cols, _ = find(no_id_scores)
                    unassigned_ids = transcript_id[no_id.get()]  # Unassigned transcript IDs

                    # Construct edge index (source, target) based on non-zero connections in no_id_scores
                    source_nodes = unassigned_ids[non_zero_rows.get()]
                    target_nodes = unassigned_ids[non_zero_cols.get()]
                    if len(source_nodes) > 10:
                        # while True:
                        try:
                            # # Save edge_index using CuDF and Dask-CuDF for GPU acceleration
                            edge_index_ddf = delayed(dd.from_pandas)(
                                pd.DataFrame({"source": source_nodes, "target": target_nodes}), npartitions=1
                            )
                            # Use delayed for asynchronous disk writing of edge_index in Dask DataFrame
                            delayed_write_edge_index = delayed(edge_index_ddf.to_parquet)(
                                edge_index_save_path, append=True, ignore_divisions=True
                            )
                            delayed_write_edge_index.compute()  # Schedule writing
                            # break
                        except Exception as e:
                            print(f"resource access fault, trying again.")

            assignments = {
                "transcript_id": assignments["transcript_id"].astype("str"),
                "score": assignments["score"].astype("float32"),
                "segger_cell_id": assignments["segger_cell_id"].astype("str"),  # Ensure 'string' dtype
                "bound": assignments["bound"].astype("int8"),  # Ensure 'int64' dtype
            }
            # Step 4: Convert assignments to Dask-CuDF DataFrame for this batch
            # batch_ddf = dask_cudf.from_cudf(cudf.DataFrame(assignments), npartitions=1)
            if len(assignments["transcript_id"]) > 10:
                # while True:
                try:
                    batch_ddf = delayed(dd.from_pandas)(pd.DataFrame(assignments), npartitions=1)

                    # Save the updated `output_ddf` asynchronously using Dask delayed
                    delayed_write_output_ddf = delayed(batch_ddf.to_parquet)(
                        output_ddf_save_path, append=True, ignore_divisions=True
                    )
                    delayed_write_output_ddf.compute()  # Schedule writing
                    # Free memory after computation
                    cp.get_default_memory_pool().free_all_blocks()  # Free CuPy memory
                    torch.cuda.empty_cache()
                    # break
                except Exception as e:
                    print(f"resource access fault, trying again.")


def segment(
    model: LitSegger,
    dm: SeggerDataModule,
    save_dir: Union[str, Path],
    seg_tag: str,
    transcript_file: Union[str, Path],
    score_cut: float = 0.5,
    use_cc: bool = True,
    save_transcripts: bool = True,
    save_anndata: bool = True,
    save_cell_masks: bool = False,  # Placeholder for future implementation
    receptive_field: dict = {"k_bd": 4, "dist_bd": 10, "k_tx": 5, "dist_tx": 3},
    knn_method: str = "cuda",
    verbose: bool = False,
    gpu_ids: list = ["0"],
    cleanup: bool = True,
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
                                          `{'k_bd': 4, 'dist_bd': 10, 'k_tx': 5, 'dist_tx': 3}`.
        knn_method (str, optional): The method to use for nearest neighbors (`'cuda'` or `'kd_tree'`).
                                    Defaults to 'cuda'.
        verbose (bool, optional): Whether to print verbose status updates. Defaults to False.
        cleanup (bool, optional): To clean up intermediate files, i.e., `transcripts_df.parqeut` and `edge_index.parquet`. Defaults to True.
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
        print(f"Assigning transcripts to nuclei for {seg_tag}...")

    # Step 1: Load the data loaders from the SeggerDataModule
    step_start_time = time()
    # train_dataloader = dm.train_dataloader()
    # val_dataloader = dm.val_dataloader()
    # test_dataloader = dm.test_dataloader()

    # # Initialize Dask DataFrame for assignments
    # output_ddf = None
    cluster = LocalCUDACluster(
        CUDA_VISIBLE_DEVICES=gpu_ids,
        n_workers=len(gpu_ids),  # Number of workers to use (adjust based on your available CPUs/GPUs)
        threads_per_worker=1,  # Each worker will use 1 thread
    )

    # # Connect the client to the cluster
    client = Client(cluster)

    @dask.delayed
    def process_batch(batch, gpu_id):
        # Assume you're using CuPy, and you need to use a specific GPU
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

    # this is to off-load some tasks from the computation graph
    for j in range(len(dm.train) // 100):
        print(len(dm.train))
        ind_min = j * 100
        ind_max = min((j + 1) * 100, len(dm.train))
        delayed_tasks = [
            process_batch(batch, gpu_ids[i % len(gpu_ids)]) for i, batch in enumerate(dm.train[ind_min:ind_max])
        ]
        future = dask.persist(*delayed_tasks)
        progress(future)
        dask.compute(*future)
        torch.cuda.empty_cache()
        cp.get_default_memory_pool().free_all_blocks()

    for j in range(len(dm.val) // 50):
        ind_min = j * 50
        ind_max = min((j + 1) * 50, len(dm.val))
        delayed_tasks = [
            process_batch(batch, gpu_ids[i % len(gpu_ids)]) for i, batch in enumerate(dm.val[ind_min:ind_max])
        ]
        future = dask.persist(*delayed_tasks)
        progress(future)
        dask.compute(*future)
        torch.cuda.empty_cache()
        cp.get_default_memory_pool().free_all_blocks()

    for j in range(len(dm.test) // 50):
        ind_min = j * 50
        ind_max = min((j + 1) * 50, len(dm.test))
        delayed_tasks = [
            process_batch(batch, gpu_ids[i % len(gpu_ids)]) for i, batch in enumerate(dm.test[ind_min:ind_max])
        ]
        future = dask.persist(*delayed_tasks)
        progress(future)
        dask.compute(*future)
        torch.cuda.empty_cache()
        cp.get_default_memory_pool().free_all_blocks()

    client.close()

    # pqdm(delayed_tasks, n_jobs=len(gpu_ids), argument_type='delayed', progress_bar=True)
    # dask.compute(*delayed_tasks)
    # delayed_tasks = [process_batch(batch, gpu_ids[i % len(gpu_ids)]) for i, batch in enumerate(batches)]

    # # Use tqdm for progress bar
    # with ProgressBar():
    #     # Execute the delayed tasks with a Dask compute call
    #     dask.compute(*delayed_tasks)

    # Loop through the data loaders (train, val, and test)
    # for loader_name, loader in zip(
    #     ["Train", "Validation", "Test"], [train_dataloader, val_dataloader, test_dataloader]
    # ):
    #     # for loader_name, loader in zip(['Test'], [test_dataloader]):
    #     if verbose:
    #         print(f"Processing {loader_name} data...")

    #     for batch in tqdm(loader, desc=f'Processing {loader_name} batches'):
    #         gpu_id = random.choice(gpu_ids)
    #         # Call predict_batch for each batch
    #         predict_batch(
    #             model,
    #             batch,
    #             score_cut,
    #             receptive_field,
    #             use_cc=use_cc,
    #             knn_method=knn_method,
    #             edge_index_save_path=edge_index_save_path,
    #             output_ddf_save_path=output_ddf_save_path,
    #             gpu_id=gpu_id
    #         )

    # if verbose:
    #     elapsed_time = time() - step_start_time
    #     print(f"Batch processing completed in {elapsed_time:.2f} seconds.")
    if verbose:
        elapsed_time = time() - step_start_time
        print(f"Assigned transcripts to nuclei in {elapsed_time:.2f} seconds.")
    # Load the full saved segmentation results
    seg_final_dd = dd.read_parquet(output_ddf_save_path, ignore_metadata_file=True)
    seg_final_dd = seg_final_dd.set_index("transcript_id", sorted=False)

    step_start_time = time()
    if verbose:
        print(f"Applying max score selection logic...")

    # Step 1: Find max bound indices (bound == 1) and max unbound indices (bound == 0)
    max_bound_idx = seg_final_dd[seg_final_dd["bound"] == 1].groupby("transcript_id")["score"].idxmax()
    max_unbound_idx = seg_final_dd[seg_final_dd["bound"] == 0].groupby("transcript_id")["score"].idxmax()

    # Step 2: Combine indices, prioritizing bound=1 scores
    final_idx = max_bound_idx.combine_first(max_unbound_idx).compute()

    # Step 3: Use the computed final_idx to select the best assignments
    # Make sure you are using the divisions and set the index correctly before loc
    # seg_final_dd = seg_final_dd.set_index('transcript_id', sorted=True)
    seg_final_filtered = seg_final_dd.loc[final_idx].compute()

    if verbose:
        elapsed_time = time() - step_start_time
        print(f"Max score selection completed in {elapsed_time:.2f} seconds.")

    # Step 3: Load the transcripts DataFrame and merge results

    if verbose:
        print(f"Loading transcripts from {transcript_file}...")

    transcripts_df = dd.read_parquet(transcript_file)
    transcripts_df["transcript_id"] = transcripts_df["transcript_id"].astype(str)

    step_start_time = time()
    if verbose:
        print(f"Merging segmentation results with transcripts...")

    # Outer merge to include all transcripts, even those without assigned cell ids
    transcripts_df_filtered = transcripts_df.merge(seg_final_filtered, on="transcript_id", how="outer")

    if verbose:
        elapsed_time = time() - step_start_time
        print(f"Merging segmentation results with transcripts completed in {elapsed_time:.2f} seconds.")

    # Step 4: Handle unassigned transcripts using connected components (if use_cc=True)
    if use_cc:

        step_start_time = time()
        if verbose:
            print(f"Regrouping the unassigned transcripts...")
        # Load edge indices from saved Parquet
        edge_index_dd = dd.read_parquet(edge_index_save_path)

        # Step 2: Get unique transcript_ids from edge_index_dd and their positional indices
        transcript_ids_in_edges = dd.concat([edge_index_dd["source"], edge_index_dd["target"]]).unique().compute()

        # Create a lookup table with unique indices
        lookup_table = pd.Series(data=range(len(transcript_ids_in_edges)), index=transcript_ids_in_edges).to_dict()

        # Map source and target to positional indices
        edge_index_dd["index_source"] = edge_index_dd["source"].map(lookup_table)
        edge_index_dd["index_target"] = edge_index_dd["target"].map(lookup_table)
        # Step 3: Compute connected components for transcripts involved in edges
        source_indices = np.asarray(edge_index_dd["index_source"].compute())
        target_indices = np.asarray(edge_index_dd["index_target"].compute())
        data_cp = np.ones(len(source_indices), dtype=cp.float32)

        # Create the sparse COO matrix
        coo_cp_matrix = scipy_coo_matrix(
            (data_cp, (source_indices, target_indices)),
            shape=(len(transcript_ids_in_edges), len(transcript_ids_in_edges)),
        )

        # Use CuPy's connected components algorithm to compute components
        n, comps = cc(coo_cp_matrix, directed=True, connection="weak")

        # Step 4: Map back the component labels to the original transcript_ids
        comp_labels = pd.Series(comps, index=transcript_ids_in_edges)
        # Step 5: Handle only unassigned transcripts in transcripts_df_filtered
        unassigned_mask = transcripts_df_filtered["segger_cell_id"].isna()

        unassigned_transcripts_df = transcripts_df_filtered.loc[unassigned_mask, ["transcript_id"]]

        # Step 6: Map component labels only to unassigned transcript_ids
        new_segger_cell_ids = unassigned_transcripts_df["transcript_id"].map(comp_labels)

        # Step 7: Create a DataFrame with updated 'segger_cell_id' for unassigned transcripts
        unassigned_transcripts_df = unassigned_transcripts_df.assign(segger_cell_id=new_segger_cell_ids)

        # Step 8: Merge this DataFrame back into the original to update only the unassigned segger_cell_id
        # We perform a left join so that only the rows in unassigned_transcripts_df are updated
        # transcripts_df_filtered = transcripts_df_filtered.drop(columns='segger_cell_id')

        # Merging the updates back to the original DataFrame
        transcripts_df_filtered = transcripts_df_filtered.merge(
            unassigned_transcripts_df[["transcript_id", "segger_cell_id"]],
            on="transcript_id",
            how="left",  # Perform a left join to only update the unassigned rows
            suffixes=("", "_new"),  # Suffix for new column to avoid overwriting
        )

        # Step 9: Fill missing segger_cell_id values with the updated values from the merge
        transcripts_df_filtered["segger_cell_id"] = transcripts_df_filtered["segger_cell_id"].fillna(
            transcripts_df_filtered["segger_cell_id_new"]
        )

        # Step 10: Clean up by dropping the temporary 'segger_cell_id_new' column
        transcripts_df_filtered = transcripts_df_filtered.drop(columns=["segger_cell_id_new"])

        # Fill the NaN values in segger_cell_id with the already existing (assigned) values
        # transcripts_df_filtered['segger_cell_id'] = transcripts_df_filtered['segger_cell_id'].fillna(transcripts_df_filtered['segger_cell_id_target'])

        # Drop any temporary columns used during the merge
        # transcripts_df_filtered = transcripts_df_filtered.drop(columns=['segger_cell_id_target'])

        if verbose:
            elapsed_time = time() - step_start_time
            print(f"Regrouped the unassigned transcripts in {elapsed_time:.2f} seconds.")

    # Step 5: Save the merged results based on options

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
            write_index=False,  # Skip writing index if not needed
            append=False,  # Set to True if you're appending to an existing Parquet file
            overwrite=True,
        )  # Dask handles Parquet well
        if verbose:
            elapsed_time = time() - step_start_time
            print(f"Saved trasncripts.parquet in {elapsed_time:.2f} seconds.")

    if save_anndata:
        if verbose:
            step_start_time = time()
            print(f"Saving anndata object...")
        anndata_save_path = save_dir / "segger_adata.h5ad"
        segger_adata = create_anndata(transcripts_df_filtered.compute(), **anndata_kwargs)  # Compute for AnnData
        segger_adata.write(anndata_save_path)
        if verbose:
            elapsed_time = time() - step_start_time
            print(f"Saved anndata object in {elapsed_time:.2f} seconds.")

    if save_cell_masks:
        if verbose:
            step_start_time = time()
            print(f"Computing and saving cell masks anndata object...")
        # Placeholder for future cell masks implementation as Dask Geopandas Parquet
        cell_masks_save_path = save_dir / "segger_cell_boundaries.parquet"
        if verbose:
            elapsed_time = time() - step_start_time
            print(f"Saved cell masks in {elapsed_time:.2f} seconds.")

    if cleanup:
        if verbose:
            step_start_time = time()
            print(f"Cleaning up intermediate files...")
        shutil.rmtree(output_ddf_save_path)
        shutil.rmtree(edge_index_save_path)

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
