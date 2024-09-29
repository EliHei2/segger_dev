import torch
import cupy as cp
import dask.array as da
from dask.distributed import Client, wait
from dask_cuda import LocalCUDACluster
from torch.utils.dlpack import to_dlpack, from_dlpack  # DLPack conversion
from segger.training.segger_data_module import SeggerDataModule
from segger.prediction.predict import load_model
from cupyx.scipy.sparse import coo_matrix
import torch.distributed as dist
from pathlib import Path
from cuvs.neighbors import cagra
import os
import cuvs
import rmm

# Initialize RMM
# from rmm.allocators.cupy import rmm_cupy_allocator

# Initialize RMM with a pool allocator
rmm.reinitialize(
    pool_allocator=True,  # Enable memory pool
    initial_pool_size=2**30  # Set 1GB initial pool size, adjust as needed
)

# Set RMM as the allocator for CuPy
# cp.cuda.set_allocator(rmm_cupy_allocator)



# Function to compute edge indices using spatial locations
def get_edge_index_cuda(coords_1: torch.Tensor, coords_2: torch.Tensor, k: int = 10, dist: float = 10.0) -> torch.Tensor:
    def cupy_to_torch(cupy_array):
        return torch.from_dlpack((cupy_array.toDlpack()))
    def torch_to_cupy(tensor):
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()  # Ensure tensor is contiguous
        return cp.fromDlpack(to_dlpack(tensor))  # Convert PyTorch tensor to CuPy
    print("Converting tensors to CuPy...")  # Debug log
    cp_coords_1 = torch_to_cupy(coords_1)
    cp_coords_2 = torch_to_cupy(coords_2)
    cp_dist = cp.float32(dist)
    print("Building index...")  # Debug log
    index_params = cagra.IndexParams()
    search_params = cagra.SearchParams()
    try:
        # Build index and search for nearest neighbors
        index = cagra.build_index(index_params, cp_coords_1)
        D, I = cagra.search(search_params, index, cp_coords_2, k)
    except cuvs.common.exceptions.CuvsException as e:
        print(f"cuVS Exception: {e}")
        raise
    print("Processing search results...")  # Debug log
    valid_mask = cp.asarray(D < cp_dist ** 2)
    repeats = valid_mask.sum(axis=1).tolist()
    row_indices = cp.repeat(cp.arange(len(cp_coords_2)), repeats)
    valid_indices = cp.asarray(I)[cp.where(valid_mask)]
    edges = cp.vstack((row_indices, valid_indices)).T
    edge_index = cupy_to_torch(edges).long().contiguous()
    return edge_index

# Set up a Dask cluster with local GPUs
cluster = LocalCUDACluster(rmm_pool_size="5GB", scheduler_port=8786, dashboard_address=":8787", worker_port=(9000, 9100))
client = Client(cluster, timeout='500s')



def initialize_distributed(rank, world_size):
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'  # Any free port can be used here
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)



# Initialize the PyTorch distributed environment (NCCL backend)
# dist.init_process_group(backend='nccl')

# Load the model (only done once on the main process)
model_version = 2
models_dir = Path('./models/bc_embedding_0919')
model_path = models_dir / 'lightning_logs' / f'version_{model_version}'
model = load_model(model_path / 'checkpoints')

# Scatter the model to all workers (GPUs)
# scattered_model = client.scatter(model)

# Define the sparse_multiply function
def sparse_multiply(mat1, mat2_T, edge_index, block_info=None):
    mat1 = cp.asarray(mat1)
    mat2_T = cp.asarray(mat2_T)
    # If block_info is provided, we adjust the edge indices for the local chunk
    if block_info is not None:
        row_block_start, row_block_end = block_info[0]['array-location'][0]  
        col_block_start, col_block_end = block_info[1]['array-location'][1]      
        rows, cols = edge_index
        row_mask = (rows >= row_block_start) & (rows < row_block_end)
        col_mask = (cols >= col_block_start) & (cols < col_block_end)
        mask = row_mask & col_mask
        # Adjust to local chunk indices for rows and columns
        rows = rows[mask] - row_block_start
        cols = cols[mask] - col_block_start
    else:
        # If block_info is None, assume we use the entire matrix
        rows, cols = edge_index
    # Perform dense multiplication for the current chunk or the full matrix
    dense_result = cp.dot(mat1, mat2_T)
    # Create the sparse result using the provided edge index
    sparse_result = coo_matrix((dense_result[rows, cols], (rows, cols)), shape=dense_result.shape)
    # Free GPU memory after each chunk computation
    cp.cuda.Stream.null.synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    return sparse_result


def inference_with_sparse_multiply(batch, model, rank, world_size, k=10, dist_r=10.0):
    # Initialize the distributed process group inside the worker
    if not dist.is_initialized():
        initialize_distributed(rank, world_size) 
    batch.to(f'cuda:{rank}') 
    # Load model inside the function to avoid pickling it
    model = model.to(f'cuda:{rank}')  # Make sure the model is on the correct GPU
    # Perform inference using the loaded model
    with torch.no_grad():
        output = model.model(batch.x_dict, batch.edge_index_dict)
    # Convert output to CuPy arrays using dlpack for further operations
    mat1 = cp.fromDlpack(to_dlpack(output['tx']))
    mat2 = cp.fromDlpack(to_dlpack(output['tx']))
    # Transpose mat2 for matrix multiplication
    mat2_T = cp.transpose(mat2)
    # Compute edge_index based on the 2D positions of tx nodes
    coords_1 = batch['tx'].pos[:, :2]  # Extract 2D positions
    coords_2 = batch['tx'].pos[:, :2]  # Assuming the same set of coordinates for the example
    edge_index = get_edge_index_cuda(coords_1, coords_2, k=k, dist=dist_r)
    # Perform sparse multiplication using the function
    result = sparse_multiply(mat1, mat2_T, edge_index)
    return result

# Initialize DataLoader
segger_data_dir = Path('./data_tidy/pyg_datasets/bc_embedding_0919')
dm = SeggerDataModule(
    data_dir=segger_data_dir,
    batch_size=1,
    num_workers=1,
)
dm.setup()


world_size = 1  # Adjust based on number of GPUs

futures = []
for i, batch in enumerate(dm.train_dataloader()):
    # Scatter the batch to each GPU worker
    scattered_batch = client.scatter(batch)
    for rank in range(world_size):
        futures.append(client.submit(inference_with_sparse_multiply, scattered_batch, model, rank, world_size, k=10, dist_r=3, retries=3))
    # Gather results from all GPUs
        print(f"Batch {i} processed with dynamic edge index and sparse multiplication.")

with ProgressBar():
    results = client.gather(futures)


# Call the function and get results in memory
all_results = process_all_batches()
print("All batches processed.")

# Clean up NCCL
dist.destroy_process_group()
