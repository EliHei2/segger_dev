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
    get_edge_index_cuda,
    get_edge_index,
    format_time,
    create_anndata
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
        checkpoints = glob.glob(str(checkpoint_path / '*.ckpt'))
        if len(checkpoints) == 0:
            raise FileNotFoundError(msg)
        # Sort checkpoints by epoch and step
        def sort_order(c):
            match = re.match(r'.*epoch=(\d+)-step=(\d+).ckpt', c)
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
    receptive_field: dict
) -> coo_matrix:
    """
    Compute similarity scores between embeddings for 'from_type' and 'to_type' nodes 
    using sparse matrix multiplication with CuPy and the 'sees' edge relation.

    Args:
        model (torch.nn.Module): The segmentation model used to generate embeddings.
        batch (Batch): A batch of data containing input features and edge indices.
        from_type (str): The type of node from which the similarity is computed.
        to_type (str): The type of node to which the similarity is computed.

    Returns:
        coo_matrix: A sparse matrix containing the similarity scores between 
                    'from_type' and 'to_type' nodes.
    """
    # Step 1: Get embeddings from the model
    batch = batch.to("cuda")
    embeddings = model(batch.x_dict, batch.edge_index_dict)
    
    # Convert PyTorch embeddings to CuPy using DLPack
    mat1 = cp.fromDlpack(to_dlpack(embeddings[from_type]))
    mat2_T = cp.fromDlpack(to_dlpack(embeddings[to_type]))

    # Step 2: Get edge indices using the 'sees' relationship from the batch
    edge_index = get_edge_index(
                batch[to_type].pos[:, :2],  # 'tx' positions
                batch[from_type].pos[:, :2],  # 'bd' positions
                k=receptive_field['k_bd'],
                dist=receptive_field['dist_bd'],
                method='cuda'
        )
    
    print(edge_index)

    # Step 3: Perform sparse matrix multiplication using CuPy
    def sparse_multiply(mat1, mat2_T, edge_index) -> coo_matrix:
        rows, cols = edge_index
        print(rows)
        print(cols)
        dense_result = cp.dot(mat1, mat2_T.T)
        
        # Create sparse result matrix
        sparse_result = coo_matrix((dense_result[rows, cols], (rows, cols)), shape=dense_result.shape)
        
        # Free GPU memory after computation
        cp.cuda.Stream.null.synchronize()
        cp.get_default_memory_pool().free_all_blocks()
        
        return sparse_result

    # Call the sparse multiply function
    sparse_similarity = sparse_multiply(mat1, mat2_T, edge_index.T)

    
    scores = torch.from_dlpack(sparse_similarity.toarray().toDlpack()).to("cuda")
    
    torch.cuda.empty_cache()
    cp.get_default_memory_pool().free_all_blocks()

    return scores




def predict_batch(
    lit_segger: torch.nn.Module,
    batch: Batch,
    score_cut: float,
    receptive_field: Dict[str, float],
    use_cc: bool = True,
    knn_method: str = 'cuda'
) -> pd.DataFrame:
    """
    Predict cell assignments for a batch of transcript data using a segmentation model.
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
        return ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), 8)) + '-nx'

    with torch.no_grad():
        # Move batch to GPU
        batch = batch.to("cuda")
        print(batch)

        # Initialize assignments DataFrame
        transcript_id = batch['tx'].id.cpu().numpy()
        assignments = pd.DataFrame({'transcript_id': transcript_id})

        if len(batch['bd'].id[0]) > 0:
            # Compute edge index using get_edge_index_cuda for 'tx' sees 'bd'
            # batch['tx', 'sees', 'bd'].edge_index = get_edge_index(
            #     batch['tx'].pos[:, :2],  # 'tx' positions
            #     batch['bd'].pos[:, :2],  # 'bd' positions
            #     k=receptive_field['k_bd'],
            #     dist=receptive_field['dist_bd'],
            #     method='cuda'
            # )

            # Compute similarity scores between 'tx' and 'bd'
            scores = get_similarity_scores(lit_segger.model, batch, "tx", "bd", receptive_field)

            # Get direct assignments from similarity matrix
            belongs = scores.max(axis=1)
            assignments['score'] = belongs.values.cpu()
            mask = assignments['score'] > score_cut

            all_ids = np.concatenate(batch['bd'].id)[np.argmax(scores.cpu().numpy(), axis=1)]
            assignments.loc[mask, 'segger_cell_id'] = all_ids[mask]

            # Add 'bound' column (1 for assigned, 0 for unassigned)
            assignments['bound'] = 0
            assignments.loc[mask, 'bound'] = 1

            if use_cc:
                # Compute edge index for 'tx' sees 'tx' using get_edge_index
                batch['tx', 'sees', 'tx'].edge_index = get_edge_index(
                    batch['tx'].pos[:, :2],  # 'tx' positions
                    batch['tx'].pos[:, :2],  # 'tx' positions (self-reference)
                    k=receptive_field['k_tx'],
                    dist=receptive_field['dist_tx'],
                    method='cuda'
                )

                # Compute similarity scores between 'tx' and 'tx'
                scores_tx = get_similarity_scores(lit_segger.model, batch, "tx", "tx")
                scores_tx.setdiag(0)  # Ignore self-similarity

                # Assign unassigned transcripts using connected components
                no_id = assignments['segger_cell_id'].isna().values
                no_id_scores = scores_tx[no_id][:, no_id]
                n, comps = cc(no_id_scores, connection="weak", directed=False)
                new_ids = np.array([_get_id() for _ in range(n)])
                assignments.loc[no_id, 'segger_cell_id'] = new_ids[comps]

        # Perform memory cleanup to avoid OOM issues
        # rmm.reinitialize(pool_allocator=True)
        torch.cuda.empty_cache()
        gc.collect()

        return assignments

def predict(
    lit_segger: LitSegger,
    data_loader: DataLoader,
    score_cut: float,
    receptive_field: dict,
    use_cc: bool = True,
    knn_method: str = 'cuda'
) -> dd.DataFrame:
    """
    Optimized prediction for multiple batches of transcript data using Dask and delayed processing with progress bar.
    
    Args:
        lit_segger (LitSegger): The lightning module wrapping the segmentation model.
        data_loader (DataLoader): A data loader providing batches of transcript and cell data.
        score_cut (float): The threshold for assigning transcripts to cells based on similarity scores.
        receptive_field (dict): Dictionary defining the receptive field for transcript-cell and transcript-transcript relations.
        use_cc (bool, optional): If True, perform connected components analysis for unassigned transcripts. Defaults to True.
        knn_method (str, optional): The method to use for nearest neighbors ('cuda' by default). Defaults to 'cuda'.

    Returns:
        dd.DataFrame: A Dask DataFrame containing the transcript IDs, similarity scores, and assigned cell IDs.
    """
    if len(data_loader) == 0:
        return None

    # Create a meta DataFrame for the Dask DataFrame
    meta = pd.DataFrame({
        'transcript_id': pd.Series(dtype='int64'),
        'score': pd.Series(dtype='float32'),
        'segger_cell_id': pd.Series(dtype='object'),
        'bound': pd.Series(dtype='int64')
    })

    # Convert the entire data loader to delayed predictions
    delayed_assignments = [
        delayed(predict_batch)(lit_segger, batch, score_cut, receptive_field, use_cc, knn_method)
        for batch in data_loader
    ]
    
    # Build the Dask DataFrame from the delayed assignments
    assignments_dd = dd.from_delayed(delayed_assignments, meta=meta)

    # Max score selection logic, with fallback to unbound scores if no bound=1
    def select_max_score_partition(df):
        max_bound_idx = df[df['bound'] == 1].groupby('transcript_id')['score'].idxmax()
        max_unbound_idx = df[df['bound'] == 0].groupby('transcript_id')['score'].idxmax()

        # Combine indices, prioritizing bound=1 scores
        final_idx = max_bound_idx.combine_first(max_unbound_idx)
        result = df.loc[final_idx].reset_index(drop=True)

        # Handle cases where there's only one entry per 'segger_cell_id'
        single_entry_mask = result.groupby('segger_cell_id').size() == 1
        result.loc[single_entry_mask, 'segger_cell_id'] = 'floating'
        
        return result

    # Map the logic over each partition using Dask
    final_assignments = assignments_dd.map_partitions(select_max_score_partition, meta=meta)

    # Trigger garbage collection and free GPU memory
    # rmm.reinitialize(pool_allocator=True)
    torch.cuda.empty_cache()
    gc.collect()

    return final_assignments


def segment(
    model: LitSegger, 
    dm: SeggerDataModule, 
    save_dir: Union[str, Path], 
    seg_tag: str, 
    transcript_file: Union[str, Path], 
    score_cut: float = .25,
    use_cc: bool = True,
    file_format: str = 'anndata', 
    receptive_field: dict = {'k_bd': 4, 'dist_bd': 10, 'k_tx': 5, 'dist_tx': 3},
    knn_method: str = 'kd_tree',
    verbose: bool = False,
    **anndata_kwargs
) -> None:
    """
    Perform segmentation using the model, merge segmentation results with transcripts_df, 
    and save in the specified format. Memory is managed efficiently using Dask and GPU 
    memory optimizations.

    Args:
        model (LitSegger): The trained segmentation model.
        dm (SeggerDataModule): The SeggerDataModule instance for data loading.
        save_dir (Union[str, Path]): Directory to save the final segmentation results.
        seg_tag (str): Tag to include in the saved filename.
        transcript_file (Union[str, Path]): Path to the transcripts parquet file.
        score_cut (float, optional): The threshold for assigning transcripts to cells based on similarity scores. Defaults to 0.25.
        use_cc (bool, optional): If True, re-group transcripts that have not been assigned to any nucleus. Defaults to True.
        file_format (str, optional): File format to save the results ('csv', 'parquet', or 'anndata'). Defaults to 'anndata'.
        receptive_field (dict, optional): Defines the receptive field for transcript-cell and transcript-transcript relations.
        knn_method (str, optional): The method to use for nearest neighbors ('kd_tree' by default).
        **anndata_kwargs: Additional keyword arguments passed to the create_anndata function.

    Returns:
        None
    """
    start_time = time.time()
    # rmm.reinitialize(pool_allocator=True, initial_pool_size=2**26, maximum_pool_size=2**30)
    # cp.cuda.set_allocator(rmm_cupy_allocator)

    # Ensure the save directory exists
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Starting segmentation for {seg_tag}...")

    # Step 1: Prediction
    step_start_time = time.time()
    
    train_dataloader = dm.train_dataloader()
    test_dataloader  = dm.test_dataloader()
    val_dataloader   = dm.val_dataloader()
    
    # delayed_train = predict(model, test_dataloader, score_cut=score_cut, receptive_field=receptive_field, use_cc=use_cc, knn_method=knn_method)
    # delayed_val   = predict(model, test_dataloader, score_cut=score_cut, receptive_field=receptive_field, use_cc=use_cc, knn_method=knn_method)
    delayed_test  = predict(model, test_dataloader, score_cut=score_cut, receptive_field=receptive_field, use_cc=use_cc, knn_method=knn_method)
    
    delayed_test = delayed_test.compute()
    # Compute all predictions at once using Dask
    # with ProgressBar():
    #     segmentation_train, segmentation_val, segmentation_test = dask.compute(delayed_train, delayed_val, delayed_test)

    if verbose:
        elapsed_time = format_time(time.time() - step_start_time)
        print(f"Predictions completed in {elapsed_time}.")

    # Step 2: Combine and group by transcript_id
    step_start_time = time.time()

    # Combine the segmentation data
    seg_combined = dd.concat([segmentation_train, segmentation_val, segmentation_test])

    # No need to handle max score logic here, as it's done inside the `predict` function
    seg_final = seg_combined.compute()

    # Drop any unassigned rows
    seg_final = seg_final.dropna(subset=['segger_cell_id']).reset_index(drop=True)

    if verbose:
        elapsed_time = format_time(time.time() - step_start_time)
        print(f"Segmentation results processed in {elapsed_time}.")

    # Step 3: Load transcripts and merge
    step_start_time = time.time()

    transcripts_df = dd.read_parquet(transcript_file)

    if verbose:
        print("Merging segmentation results with transcripts...")

    # Merge the segmentation results with the transcript data
    seg_final_dd = dd.from_pandas(seg_final, npartitions=transcripts_df.npartitions)
    transcripts_df_filtered = transcripts_df.merge(seg_final_dd, on='transcript_id', how='inner').compute()

    if verbose:
        elapsed_time = format_time(time.time() - step_start_time)
        print(f"Transcripts merged in {elapsed_time}.")

    # Step 4: Save the merged result
    step_start_time = time.time()
    
    if verbose:
        print(f"Saving results in {file_format} format...")

    if file_format == 'csv':
        save_path = save_dir / f'{seg_tag}_segmentation.csv'
        transcripts_df_filtered.to_csv(save_path, index=False)
    elif file_format == 'parquet':
        save_path = save_dir / f'{seg_tag}_segmentation.parquet'
        transcripts_df_filtered.to_parquet(save_path, index=False)
    elif file_format == 'anndata':
        save_path = save_dir / f'{seg_tag}_segmentation.h5ad'
        segger_adata = create_anndata(transcripts_df_filtered, **anndata_kwargs)
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

    # Step 5: Garbage collection and memory cleanup
    # rmm.reinitialize(pool_allocator=True)
    # torch.cuda.empty_cache()
    gc.collect()
