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
    get_edge_index_cuda,
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

# Setup Dask cluster with 3 workers


# CONFIG
torch._dynamo.config.suppress_errors = True
os.environ["PYTORCH_USE_CUDA_DSA"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


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
    knn_method: str,
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

    Returns:
        coo_matrix: A sparse matrix containing the similarity scores between
                    'from_type' and 'to_type' nodes.
    """

    # Keep everything on GPU until final results
    batch = batch.to("cuda")

    # Step 1: Get embeddings from the model (on GPU)
    shape = batch[from_type].x.shape[0], batch[to_type].x.shape[0]

    # Compute edge indices using knn method (still on GPU)
    edge_index = get_edge_index(
        batch[to_type].pos[:, :2],  # 'tx' positions
        batch[from_type].pos[:, :2],  # 'bd' positions
        k=receptive_field[f"k_{to_type}"],
        dist=receptive_field[f"dist_{to_type}"],
        method=knn_method,
    )

    # Convert to dense adjacency matrix (on GPU)
    edge_index = coo_to_dense_adj(
        edge_index.T,
        num_nodes=shape[0],
        num_nbrs=receptive_field[f"k_{to_type}"],
    )

    with torch.no_grad():
        embeddings = model(batch.x_dict, batch.edge_index_dict)

    del batch

    # print(edge_index)
    # print(embeddings)

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
) -> pd.DataFrame:
    """
    Predict cell assignments for a batch of transcript data using a segmentation model.
    Adds a 'bound' column to indicate if the transcript is assigned to a cell (bound=1)
    or unassigned (bound=0). Unassigned transcripts are handled with connected components
    if use_cc is True.
    Adds a 'bound' column to indicate if the transcript is assigned to a cell (bound=1)
    or unassigned (bound=0).

    Args:
        lit_segger (torch.nn.Module): The lightning module wrapping the segmentation model.
        batch (Batch): A batch of transcript and cell data.
        score_cut (float): The threshold for assigning transcripts to cells based on similarity scores.
        receptive_field (Dict[str, float]): Dictionary defining the receptive field for transcript-cell
                                            and transcript-transcript relations.
        use_cc (bool, optional): If True, perform connected components analysis for unassigned transcripts.
                                 Defaults to True.
        knn_method (str, optional): The method to use for nearest neighbors. Defaults to 'cuda'.

    Returns:
        pd.DataFrame: A DataFrame containing the transcript IDs, similarity scores,
                      assigned cell IDs, and 'bound' column.
    """

    def _get_id():
        """Generate a random Xenium-style ID."""
        return "".join(np.random.choice(list("abcdefghijklmnopqrstuvwxyz"), 8)) + "-nx"

    # Use CuPy with GPU context
    with cp.cuda.Device(0):
        # Move batch to GPU
        batch = batch.to("cuda")

        # Extract transcript IDs and initialize assignments DataFrame
        transcript_id = batch["tx"].id.cpu()
        assignments = pd.DataFrame({"transcript_id": transcript_id})
        transcript_id = cp.asnumpy(batch["tx"].id)
        assignments = pd.DataFrame({"transcript_id": transcript_id})

        if len(batch["bd"].pos) >= 10:
            # Compute similarity scores between 'tx' and 'bd'
            scores = get_similarity_scores(lit_segger.model, batch, "tx", "bd", receptive_field, knn_method=knn_method)
            scores = get_similarity_scores(lit_segger.model, batch, "tx", "bd", receptive_field, knn_method)
            torch.cuda.empty_cache()
            # Convert sparse matrix to dense format
            dense_scores = scores.toarray()  # Convert to dense NumPy array
            del scores  # Remove from memory
            cp.get_default_memory_pool().free_all_blocks()  # Free CuPy memory

            # Get direct assignments from similarity matrix
            belongs = cp.max(dense_scores, axis=1)  # Max score per transcript
            assignments["score"] = cp.asnumpy(belongs)  # Move back to CPU

            mask = assignments["score"] > score_cut
            all_ids = np.concatenate(batch["bd"].id)  # Keep IDs as NumPy array
            assignments["segger_cell_id"] = None  # Initialize as None
            max_indices = cp.argmax(dense_scores, axis=1).get()
            assignments.loc[mask, "segger_cell_id"] = all_ids[max_indices[mask]]

            assignments.loc[mask, "segger_cell_id"] = all_ids[max_indices[mask]]  # Assign IDs

            del dense_scores  # Remove from memory
            cp.get_default_memory_pool().free_all_blocks()  # Free CuPy memory
            torch.cuda.empty_cache()

            assignments["bound"] = 0
            assignments.loc[mask, "bound"] = 1

            # Handle unassigned transcripts with connected components
            # Move back to CPU
            assignments["bound"] = 0
            assignments.loc[mask, "bound"] = 1

            if use_cc:
                # Compute similarity scores between 'tx' and 'tx'
                scores_tx = get_similarity_scores(
                    lit_segger.model, batch, "tx", "tx", receptive_field, knn_method=knn_method
                )

                # Convert to dense NumPy array
                data_cpu = scores_tx.data.get()  # Transfer data to CPU (NumPy)
                row_cpu = scores_tx.row.get()  # Transfer row indices to CPU (NumPy)
                col_cpu = scores_tx.col.get()  # Transfer column indices to CPU (NumPy)
                # Remove from memory
                scores_tx = get_similarity_scores(
                    lit_segger.model, batch, "tx", "tx", receptive_field, knn_method=knn_method
                )
                # Convert to dense NumPy array
                data_cpu = scores_tx.data.get()  # Transfer data to CPU (NumPy)
                row_cpu = scores_tx.row.get()  # Transfer row indices to CPU (NumPy)
                col_cpu = scores_tx.col.get()  # Transfer column indices to CPU (NumPy)

                # dense_scores_tx = scores_tx.toarray().astype(cp.float16)
                # Rebuild the matrix on CPU using SciPy
                dense_scores_tx = scipy_coo_matrix((data_cpu, (row_cpu, col_cpu)), shape=scores_tx.shape).toarray()
                del scores_tx
                np.fill_diagonal(dense_scores_tx, 0)  # Ignore self-similarity

                # Get the indices of unassigned transcripts
                no_id = assignments["segger_cell_id"].isna().values  # Convert to numpy for indexing

                del scores_tx  # Remove from memory
                cp.get_default_memory_pool().free_all_blocks()  # Free CuPy memory

                # Assign unassigned transcripts using connected components
                no_id = assignments["segger_cell_id"].isna()
                if np.any(no_id):  # Only compute if there are unassigned transcripts
                    # Transfer the relevant parts of the sparse matrix (unassigned transcripts)
                    no_id_scores = dense_scores_tx[no_id][:, no_id]

                    # Apply score cut-off
                    # no_id_scores = no_id_scores.toarray()  # Move to dense temporarily
                    no_id_scores[no_id_scores < score_cut] = 0  # Apply threshold
                    no_id_scores = scipy_coo_matrix(no_id_scores)  # Convert back to sparse

                    # Find the non-zero entries in the no_id_scores to construct edge_index
                    non_zero_rows, non_zero_cols = no_id_scores.nonzero()

                    # Map these indices back to the actual transcript IDs (no_id_mask gives us their original position)
                    unassigned_ids = batch["tx"].id[no_id]  # Unassigned transcript IDs

                    # Construct edge index (source, target) based on non-zero connections in the no_id_scores matrix
                    source_nodes = unassigned_ids[non_zero_rows].cpu()
                    target_nodes = unassigned_ids[non_zero_cols].cpu()

                    # Convert to Dask array for later concatenation
                    edge_index = dd.from_array(np.stack([source_nodes, target_nodes], axis=0))

                    # Append this batch's edges to a global list to be used later in segment
                    global_edge_index_list.append(edge_index)
                    del dense_scores_tx  # Remove from memory
                    no_id_scores[no_id_scores < score_cut] = 0
                    n, comps = cc(no_id_scores, connection="weak", directed=False)
                    new_ids = np.array([_get_id() for _ in range(n)])
                    assignments["segger_cell_id"][no_id] = new_ids[comps]

                    # Free memory after computation
                    del no_id_scores
                    cp.get_default_memory_pool().free_all_blocks()
                    torch.cuda.empty_cache()

        return assignments


def predict(
    lit_segger: LitSegger,
    data_loader: DataLoader,
    score_cut: float,
    receptive_field: dict,
    use_cc: bool = True,
    knn_method: str = "cuda",
) -> pd.DataFrame:  # Change return type to Dask DataFrame if applicable
    """
    Optimized prediction for multiple batches of transcript data, using parallel execution
    and displaying a progress bar.

    Args:
        lit_segger (LitSegger): The trained segmentation model.
        data_loader (DataLoader): PyTorch DataLoader to iterate over transcript data batches.
        score_cut (float): Threshold for assigning transcripts to cells based on similarity scores.
        receptive_field (dict): Defines the receptive field for transcript-cell and transcript-transcript relations.
        use_cc (bool, optional): If True, perform connected components analysis for unassigned transcripts. Defaults to True.
        knn_method (str, optional): Method to use for nearest neighbors ('cuda', 'kd_tree', etc.). Defaults to 'cuda'.
        n_jobs (int, optional): Number of parallel jobs to run. Defaults to 4.

    Returns:
        pd.DataFrame: Final DataFrame containing the transcript IDs, similarity scores, and assigned cell IDs.
    """

    all_assignments = []
    # Use Dask delayed to parallelize the predict_batch execution and create assignments
    delayed_batches = [
        delayed(predict_batch)(lit_segger, batch, score_cut, receptive_field, use_cc, knn_method)
        for batch in data_loader
    ]

    for batch in data_loader:
        assignments = predict_batch(lit_segger, batch, score_cut, receptive_field, use_cc, knn_method)
        all_assignments.append(dd.from_pandas(assignments, npartitions=1))

        cp.get_default_memory_pool().free_all_blocks()
        torch.cuda.empty_cache()

    # Concatenate all assignments into a single Dask DataFrame
    final_assignments = dd.concat(all_assignments, ignore_index=True)

    # Sort the Dask DataFrame by 'transcript_id' before setting it as an index
    final_assignments = final_assignments.sort_values(by="transcript_id")

    # Set a unique index for Dask DataFrame
    final_assignments = final_assignments.set_index("transcript_id", sorted=True)

    # Max score selection logic
    max_bound_idx = final_assignments[final_assignments["bound"] == 1].groupby("transcript_id")["score"].idxmax()
    max_unbound_idx = final_assignments[final_assignments["bound"] == 0].groupby("transcript_id")["score"].idxmax()

    # Combine indices, prioritizing bound=1 scores
    final_idx = max_bound_idx.combine_first(max_unbound_idx).compute()  # Ensure it's computed

    # Now use the computed final_idx for indexing
    result = final_assignments.loc[final_idx].compute().reset_index(names=["transcript_id"])

    # result = results.reset_index()

    # Handle cases where there's only one entry per 'segger_cell_id'
    # single_entry_mask = result.groupby('segger_cell_id').size() == 1
    # Handle cases where there's only one entry per 'segger_cell_id'
    # single_entry_counts = result['segger_cell_id'].value_counts()  # Count occurrences of each ID
    # single_entry_mask = single_entry_counts[single_entry_counts == 1].index  # Get IDs with a count of 1

    # # Update 'segger_cell_id' for single entries
    # for segger_id in single_entry_mask:
    #     result.loc[result['segger_cell_id'] == segger_id, 'segger_cell_id'] = 'floating'

    return result


def segment(
    model: LitSegger,
    dm: SeggerDataModule,
    save_dir: Union[str, Path],
    seg_tag: str,
    transcript_file: Union[str, Path],
    score_cut: float = 0.5,
    use_cc: bool = True,
    file_format: str = "anndata",
    receptive_field: dict = {"k_bd": 4, "dist_bd": 10, "k_tx": 5, "dist_tx": 3},
    knn_method: str = "kd_tree",
    verbose: bool = False,
    **anndata_kwargs,
) -> None:
    """
    Perform segmentation using the model, merge segmentation results with transcripts_df, and save in the specified format.

    Parameters:
    ----------
    model : LitSegger
        The trained segmentation model.
    dm : SeggerDataModule
        The SeggerDataModule instance for data loading.
    save_dir : Union[str, Path]
        Directory to save the final segmentation results.
    seg_tag : str
        Tag to include in the saved filename.
    transcript_file : Union[str, Path]
        Path to the transcripts parquet file.
    file_format : str, optional
        File format to save the results ('csv', 'parquet', or 'anndata'). Defaults to 'anndata'.
    score_cut : float, optional
        The threshold for assigning transcripts to cells based on similarity scores.
    use_cc : bool, optional
        If to further re-group transcripts that have not been assigned to any nucleus.
    knn_method : str, optional
        The method to use for nearest neighbors ('cuda' by default).
    **anndata_kwargs : dict, optional
        Additional keyword arguments passed to the create_anndata function.
    Perform segmentation using the model, merge segmentation results with transcripts data,
    and save the results in the specified format. The function also handles unassigned
    transcripts through connected components analysis if `use_cc` is True.

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
        file_format (str, optional): The file format to save the results ('csv', 'parquet', or 'anndata').
                                     Defaults to 'anndata'.
        receptive_field (dict, optional): Defines the receptive field for transcript-cell and
                                          transcript-transcript relations. Defaults to
                                          {'k_bd': 4, 'dist_bd': 10, 'k_tx': 5, 'dist_tx': 3}.
        knn_method (str, optional): The method to use for nearest neighbors ('cuda' or 'kd_tree').
                                    Defaults to 'kd_tree'.
        verbose (bool, optional): Whether to print verbose status updates. Defaults to False.
        **anndata_kwargs: Additional keyword arguments passed to the `create_anndata` function.

    Returns:
        None
    """

    # Initialize global_edge_index_list to store edge indices across batches
    global global_edge_index_list
    global_edge_index_list = []

    # Start the timer
    start_time = time.time()

    # Ensure the save directory exists
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Starting segmentation for {seg_tag}...")

    # Step 1: Prediction
    step_start_time = time.time()

    train_dataloader = dm.train_dataloader()
    test_dataloader = dm.test_dataloader()
    val_dataloader = dm.val_dataloader()

    segmentation_train = predict(model, train_dataloader, score_cut, receptive_field, use_cc, knn_method)
    torch.cuda.empty_cache()
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()

    segmentation_val = predict(model, val_dataloader, score_cut, receptive_field, use_cc, knn_method)
    torch.cuda.empty_cache()
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()

    segmentation_test = predict(model, test_dataloader, score_cut, receptive_field, use_cc, knn_method)
    torch.cuda.empty_cache()
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()

    if verbose:
        elapsed_time = format_time(time.time() - step_start_time)
        print(f"Predictions completed in {elapsed_time}.")

    # Step 2: Combine and group by transcript_id
    step_start_time = time.time()

    # Combine the segmentation data
    seg_combined = pd.concat([segmentation_train, segmentation_val, segmentation_test], ignore_index=True)

    # Drop any unassigned rows
    seg_final = seg_combined.dropna(subset=["segger_cell_id"]).reset_index(drop=True)

    if verbose:
        elapsed_time = format_time(time.time() - step_start_time)
        print(f"Segmentation results processed in {elapsed_time}.")

    # Step 3: Load transcripts and merge
    step_start_time = time.time()

    transcripts_df = dd.read_parquet(transcript_file)

    if verbose:
        print("Merging segmentation results with transcripts...")

    # Convert the segmentation results to a Dask DataFrame, keeping npartitions consistent
    seg_final_dd = dd.from_pandas(seg_final, npartitions=transcripts_df.npartitions)

    # Merge the segmentation results with the transcript data (still as Dask DataFrame)
    transcripts_df_filtered = transcripts_df.merge(seg_final_dd, on="transcript_id", how="inner")

    if verbose:
        elapsed_time = format_time(time.time() - step_start_time)
        print(f"Transcripts merged in {elapsed_time}.")

    # Step 4: Handle unassigned transcripts using connected components (if use_cc is True)
    if use_cc:
        if verbose:
            print(f"Computing connected components for unassigned transcripts...")

        # Concatenate all edge indices stored across batches
        concatenated_edge_index = dd.concat(global_edge_index_list)

        # Compute connected components using the concatenated edge_index
        edge_index_computed = concatenated_edge_index.compute()  # Get the full edge_index (source, target)

        # Map transcript_ids to their index positions in the DataFrame
        transcript_idx_map = pd.Series(
            transcripts_df_filtered.index, index=transcripts_df_filtered["transcript_id"]
        ).to_dict()

        # Convert the transcript_ids in edge_index_computed to positional indices
        source_indices = [transcript_idx_map[tid] for tid in edge_index_computed[0]]
        target_indices = [transcript_idx_map[tid] for tid in edge_index_computed[1]]

        # Use SciPy's connected components algorithm
        n, comps = cc(
            scipy_coo_matrix(
                (np.ones(len(source_indices)), (source_indices, target_indices)),
                shape=(transcripts_df_filtered.shape[0], transcripts_df_filtered.shape[0]),
            ),
            connection="weak",
            directed=False,
        )

        # Generate new cell IDs based on connected components
        new_ids = np.array([_get_id() for _ in range(n)])

        # Assign new cell IDs to the unassigned transcripts in the final assignments
        unassigned_mask = transcripts_df_filtered["segger_cell_id"].isna()
        transcripts_df_filtered.loc[unassigned_mask, "segger_cell_id"] = new_ids[comps]

        if verbose:
            elapsed_time = format_time(time.time() - step_start_time)
            print(f"Connected components computed in {elapsed_time}.")

    # Step 5: Save the merged result
    step_start_time = time.time()

    if verbose:
        print(f"Saving results in {file_format} format...")

    if file_format == "csv":
        save_path = save_dir / f"{seg_tag}_segmentation.csv"
        transcripts_df_filtered.compute().to_csv(save_path, index=False)  # Use pandas after computing
    elif file_format == "parquet":
        save_path = save_dir / f"{seg_tag}_segmentation.parquet"
        transcripts_df_filtered.to_parquet(save_path, index=False)  # Dask handles Parquet fine
    elif file_format == "anndata":
        save_path = save_dir / f"{seg_tag}_segmentation.h5ad"
        segger_adata = create_anndata(transcripts_df_filtered.compute(), **anndata_kwargs)  # Compute for AnnData
        segger_adata.write(save_path)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")

    if verbose:
        elapsed_time = format_time(time.time() - step_start_time)
        print(f"Results saved in {elapsed_time} at {save_path}.")

    # Total time
    if verbose:
        total_time = format_time(time.time() - start_time)
        print(f"Total segmentation process completed in {total_time}.")

    # Step 6: Garbage collection and memory cleanup
    torch.cuda.empty_cache()
    gc.collect()
