# Segger Model Inference

## Overview

Inference with the trained Segger model involves using the learned Graph Neural Network to predict transcript-to-cell associations in spatial transcriptomics data. The inference process transforms spatial data into cell segmentation results through link prediction, enabling both cell identification and fragment detection.

## Inference Pipeline

### Overall Workflow

```
Trained Model → Spatial Data → Graph Construction → Node Embeddings → Similarity Scores → Cell Assignment → Post-processing
     ↓              ↓              ↓                ↓              ↓              ↓              ↓
Learned Weights   Transcripts    Heterogeneous   GNN Forward    Link Prediction  Thresholding   Final Results
                 Boundaries     Graphs          Pass            Scores           Filtering      + Fragments
```

### Key Steps

1. **Model Loading**: Load trained Segger model from checkpoint
2. **Data Preparation**: Construct heterogeneous graphs from spatial data
3. **Embedding Generation**: Generate node embeddings using the trained model
4. **Similarity Computation**: Calculate transcript-to-cell similarity scores
5. **Assignment Decision**: Assign transcripts to cells based on confidence scores
6. **Fragment Detection**: Group unassigned transcripts into fragments

## Model Loading

### Loading Trained Model

```python
from segger.models.segger_model import Segger
import torch

# Load trained model
model = Segger(
    num_tx_tokens=5000,
    init_emb=16,
    hidden_channels=64,
    num_mid_layers=3,
    out_channels=32,
    heads=4
)

# Load trained weights
checkpoint = torch.load('path/to/checkpoint.ckpt')
model.load_state_dict(checkpoint['state_dict'])
model.eval()  # Set to evaluation mode
```

### Model Configuration

Ensure inference parameters match training configuration:

```python
# Verify model configuration matches training
assert model.num_tx_tokens == 5000, "Vocabulary size mismatch"
assert model.hidden_channels == 64, "Hidden dimension mismatch"
assert model.out_channels == 32, "Output dimension mismatch"
assert model.heads == 4, "Attention heads mismatch"
```

## Data Preparation

### Graph Construction

Inference requires the same graph structure used during training:

```python
# Construct heterogeneous graph for inference
data = {
    "tx": {  # Transcript nodes
        "id": transcript_ids,
        "pos": spatial_coordinates,
        "x": feature_vectors
    },
    "bd": {  # Boundary nodes
        "id": boundary_ids,
        "pos": centroid_coordinates,
        "x": geometric_features
    },
    "tx,neighbors,tx": {  # Transcript proximity edges
        "edge_index": neighbor_connections
    },
    "tx,belongs,bd": {  # Transcript-boundary edges (for inference)
        "edge_index": containment_relationships
    }
}
```

### Feature Processing

#### Transcript Features
```python
# Use same feature processing as training
if has_scrnaseq_embeddings:
    transcript_features = gene_celltype_embeddings[gene_labels]
else:
    transcript_features = embedding_layer(gene_token_ids)
```

#### Boundary Features
```python
# Compute geometric features for boundaries
def compute_boundary_features(boundary_polygons):
    features = []
    for polygon in boundary_polygons:
        area_val = polygon.area
        convex_hull = polygon.convex_hull
        convexity = convex_hull.area / area_val
        
        # Minimum bounding rectangle
        mbr = polygon.minimum_rotated_rectangle
        mbr_area = mbr.area
        
        # Envelope (axis-aligned bounding box)
        envelope = polygon.envelope
        env_area = envelope.area
        elongation = mbr_area / env_area
        
        # Circularity
        min_radius = compute_minimum_bounding_radius(polygon)
        circularity = area_val / (min_radius ** 2)
        
        features.append([area_val, convexity, elongation, circularity])
    
    return torch.tensor(features, dtype=torch.float32)
```

## Inference Process

### Forward Pass

```python
# Generate node embeddings
with torch.no_grad():
    embeddings = model(data.x, data.edge_index)

# Separate transcript and boundary embeddings
tx_embeddings = embeddings[data.tx_mask]
bd_embeddings = embeddings[data.bd_mask]
```

### Similarity Score Computation

#### Transcript-to-Cell Similarity

```python
def compute_similarity_scores(tx_embeddings, bd_embeddings, edge_index):
    """Compute similarity scores between transcripts and boundaries."""
    
    # Extract source and target indices
    tx_indices = edge_index[0]
    bd_indices = edge_index[1]
    
    # Get embeddings for connected nodes
    tx_emb = tx_embeddings[tx_indices]
    bd_emb = bd_embeddings[bd_indices]
    
    # Compute dot product similarity
    similarity_scores = torch.sum(tx_emb * bd_emb, dim=1)
    
    # Apply sigmoid for probability
    probabilities = torch.sigmoid(similarity_scores)
    
    return probabilities, similarity_scores
```

#### Receptive Field Construction

```python
def construct_receptive_field(transcripts, boundaries, k_bd=3, dist_bd=10.0):
    """Construct receptive field for transcript-to-cell assignment."""
    
    # Find nearest boundary cells for each transcript
    from sklearn.neighbors import NearestNeighbors
    
    # Extract coordinates
    tx_coords = transcripts[['x', 'y']].values
    bd_coords = boundaries[['centroid_x', 'centroid_y']].values
    
    # Build nearest neighbor index
    nn = NearestNeighbors(n_neighbors=k_bd, radius=dist_bd)
    nn.fit(bd_coords)
    
    # Find neighbors
    distances, indices = nn.kneighbors(tx_coords)
    
    # Filter by distance threshold
    mask = distances <= dist_bd
    filtered_indices = []
    filtered_distances = []
    
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        valid_mask = mask[i]
        filtered_indices.append(idx[valid_mask])
        filtered_distances.append(dist[valid_mask])
    
    return filtered_indices, filtered_distances
```

## Cell Assignment

### Assignment Decision

```python
def assign_transcripts_to_cells(similarity_scores, score_threshold=0.7):
    """Assign transcripts to cells based on similarity scores."""
    
    # Find best matching cell for each transcript
    best_scores, best_cells = torch.max(similarity_scores, dim=1)
    
    # Apply confidence threshold
    confident_mask = best_scores >= score_threshold
    
    # Create assignment results
    assignments = {
        'transcript_id': [],
        'cell_id': [],
        'confidence_score': [],
        'assigned': []
    }
    
    for i, (score, cell_id) in enumerate(zip(best_scores, best_cells)):
        assignments['transcript_id'].append(i)
        assignments['cell_id'].append(cell_id.item())
        assignments['confidence_score'].append(score.item())
        assignments['assigned'].append(confident_mask[i].item())
    
    return assignments
```

### Confidence Score Analysis

```python
def analyze_confidence_scores(scores):
    """Analyze distribution of confidence scores."""
    
    import numpy as np
    from scipy import stats
    
    # Convert to numpy for analysis
    scores_np = scores.detach().cpu().numpy()
    
    # Basic statistics
    stats_summary = {
        'mean': np.mean(scores_np),
        'std': np.std(scores_np),
        'min': np.min(scores_np),
        'max': np.max(scores_np),
        'median': np.median(scores_np)
    }
    
    # Percentiles
    percentiles = [25, 50, 75, 90, 95, 99]
    for p in percentiles:
        stats_summary[f'p{p}'] = np.percentile(scores_np, p)
    
    # Find knee point for automatic thresholding
    # Use the method from the Segger paper
    knee_point = find_knee_point(scores_np)
    stats_summary['knee_point'] = knee_point
    
    return stats_summary

def find_knee_point(scores):
    """Find knee point in score distribution for automatic thresholding."""
    
    from kneed import KneeLocator
    
    # Sort scores
    sorted_scores = np.sort(scores)
    
    # Create cumulative distribution
    cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    
    # Find knee point
    kneedle = KneeLocator(
        sorted_scores, cumulative, 
        S=1.0, curve='concave', direction='increasing'
    )
    
    return kneedle.knee if kneedle.knee else np.median(scores)
```

## Fragment Detection

### Unassigned Transcript Handling

```python
def detect_fragments(unassigned_transcripts, k_tx=4, dist_tx=5.0, similarity_threshold=0.5):
    """Group unassigned transcripts into fragments."""
    
    # Construct transcript-transcript similarity graph
    fragment_graph = construct_fragment_graph(
        unassigned_transcripts, k_tx, dist_tx, similarity_threshold
    )
    
    # Find connected components
    from scipy.sparse.csgraph import connected_components
    
    n_components, labels = connected_components(fragment_graph, directed=False)
    
    # Assign fragment IDs
    fragment_assignments = {}
    for i, label in enumerate(labels):
        transcript_id = unassigned_transcripts[i]
        fragment_assignments[transcript_id] = f"fragment_{label}"
    
    return fragment_assignments, n_components
```

### Fragment Graph Construction

```python
def construct_fragment_graph(transcripts, k_tx, dist_tx, similarity_threshold):
    """Construct similarity graph for unassigned transcripts."""
    
    # Extract coordinates and features
    coords = transcripts[['x', 'y']].values
    features = transcripts['features'].values
    
    # Build nearest neighbor graph
    nn = NearestNeighbors(n_neighbors=k_tx, radius=dist_tx)
    nn.fit(coords)
    
    # Find neighbors
    distances, indices = nn.radius_neighbors(coords, radius=dist_tx)
    
    # Compute similarity scores
    edges = []
    for i, neighbors in enumerate(indices):
        for j in neighbors:
            if i != j:
                # Compute feature similarity
                sim_score = compute_feature_similarity(features[i], features[j])
                
                # Add edge if similarity exceeds threshold
                if sim_score >= similarity_threshold:
                    edges.append((i, j, sim_score))
    
    # Convert to sparse matrix
    from scipy.sparse import csr_matrix
    
    if edges:
        rows, cols, data = zip(*edges)
        n_transcripts = len(transcripts)
        fragment_graph = csr_matrix((data, (rows, cols)), shape=(n_transcripts, n_transcripts))
    else:
        fragment_graph = csr_matrix((len(transcripts), len(transcripts)))
    
    return fragment_graph

def compute_feature_similarity(feature1, feature2):
    """Compute similarity between transcript features."""
    
    # Cosine similarity
    dot_product = np.dot(feature1, feature2)
    norm1 = np.linalg.norm(feature1)
    norm2 = np.linalg.norm(feature2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    similarity = dot_product / (norm1 * norm2)
    return similarity
```

## Batch Processing

### Large Dataset Handling

```python
def batch_inference(model, data_loader, device='cuda'):
    """Perform inference on large datasets in batches."""
    
    model.to(device)
    model.eval()
    
    all_results = []
    
    with torch.no_grad():
        for batch in data_loader:
            # Move batch to device
            batch = batch.to(device)
            
            # Generate embeddings
            embeddings = model(batch.x, batch.edge_index)
            
            # Compute similarity scores
            scores = compute_batch_similarities(batch, embeddings)
            
            # Store results
            batch_results = {
                'transcript_ids': batch.transcript_ids,
                'cell_ids': batch.cell_ids,
                'similarity_scores': scores,
                'batch_idx': batch.batch
            }
            
            all_results.append(batch_results)
    
    # Combine results from all batches
    combined_results = combine_batch_results(all_results)
    
    return combined_results
```

### Memory Management

```python
def optimize_memory_usage(batch_size, model_size):
    """Optimize memory usage during inference."""
    
    # Estimate memory requirements
    estimated_memory = estimate_inference_memory(batch_size, model_size)
    
    # Adjust batch size if needed
    if estimated_memory > available_memory:
        optimal_batch_size = find_optimal_batch_size(model_size, available_memory)
        print(f"Reducing batch size from {batch_size} to {optimal_batch_size}")
        return optimal_batch_size
    
    return batch_size

def estimate_inference_memory(batch_size, model_size):
    """Estimate memory usage for inference."""
    
    # Model parameters
    model_memory = model_size * 4  # 4 bytes per float32
    
    # Activations (approximate)
    activation_memory = batch_size * 1000 * 64 * 4  # Assume 1000 nodes, 64 features
    
    # Total memory
    total_memory = model_memory + activation_memory
    
    return total_memory
```

## Post-processing

### Result Aggregation

```python
def aggregate_inference_results(assignments, fragments):
    """Aggregate inference results into final segmentation."""
    
    # Combine cell assignments and fragment assignments
    final_results = {
        'transcript_id': [],
        'final_cell_id': [],
        'assignment_type': [],  # 'cell' or 'fragment'
        'confidence_score': []
    }
    
    # Add cell assignments
    for tx_id, cell_id, score, assigned in zip(
        assignments['transcript_id'],
        assignments['cell_id'],
        assignments['confidence_score'],
        assignments['assigned']
    ):
        if assigned:
            final_results['transcript_id'].append(tx_id)
            final_results['final_cell_id'].append(cell_id)
            final_results['assignment_type'].append('cell')
            final_results['confidence_score'].append(score)
    
    # Add fragment assignments
    for tx_id, fragment_id in fragments.items():
        final_results['transcript_id'].append(tx_id)
        final_results['final_cell_id'].append(fragment_id)
        final_results['assignment_type'].append('fragment')
        final_results['confidence_score'].append(0.0)  # No confidence for fragments
    
    return final_results
```

### Quality Control

```python
def quality_control_checks(results, min_transcripts_per_cell=5):
    """Perform quality control checks on inference results."""
    
    # Count transcripts per cell
    from collections import Counter
    cell_counts = Counter(results['final_cell_id'])
    
    # Filter cells with too few transcripts
    valid_cells = {cell_id: count for cell_id, count in cell_counts.items() 
                   if count >= min_transcripts_per_cell}
    
    # Filter results
    filtered_results = {
        'transcript_id': [],
        'final_cell_id': [],
        'assignment_type': [],
        'confidence_score': []
    }
    
    for i, cell_id in enumerate(results['final_cell_id']):
        if cell_id in valid_cells:
            for key in filtered_results:
                filtered_results[key].append(results[key][i])
    
    return filtered_results, valid_cells
```

## Output Formats

### Standard Output

```python
def save_inference_results(results, output_path, format='parquet'):
    """Save inference results in specified format."""
    
    import pandas as pd
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    if format == 'parquet':
        df.to_parquet(output_path, index=False)
    elif format == 'csv':
        df.to_csv(output_path, index=False)
    elif format == 'h5ad':
        # Convert to AnnData format
        adata = convert_to_anndata(df)
        adata.write(output_path)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Results saved to {output_path}")

def convert_to_anndata(results_df):
    """Convert results to AnnData format for downstream analysis."""
    
    import scanpy as sc
    import anndata as ad
    
    # Group by cell ID
    cell_groups = results_df.groupby('final_cell_id')
    
    # Create cell-gene matrix
    cell_gene_matrix = []
    cell_ids = []
    
    for cell_id, group in cell_groups:
        # Count transcripts per gene
        gene_counts = group['gene_name'].value_counts()
        cell_gene_matrix.append(gene_counts)
        cell_ids.append(cell_id)
    
    # Convert to DataFrame
    cell_gene_df = pd.DataFrame(cell_gene_matrix, index=cell_ids)
    cell_gene_df = cell_gene_df.fillna(0)
    
    # Create AnnData object
    adata = ad.AnnData(X=cell_gene_df.values, 
                       obs=pd.DataFrame(index=cell_ids),
                       var=pd.DataFrame(index=cell_gene_df.columns))
    
    return adata
```

## Performance Optimization

### GPU Acceleration

```python
def optimize_gpu_inference(model, data, device='cuda'):
    """Optimize GPU inference performance."""
    
    # Move model and data to GPU
    model = model.to(device)
    data = data.to(device)
    
    # Enable CUDA optimizations
    torch.backends.cudnn.benchmark = True
    
    # Use mixed precision if available
    if hasattr(torch, 'autocast'):
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            embeddings = model(data.x, data.edge_index)
    else:
        embeddings = model(data.x, data.edge_index)
    
    return embeddings
```

### Parallel Processing

```python
def parallel_inference(model, data_list, num_workers=4):
    """Perform inference in parallel across multiple workers."""
    
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    # Split data across workers
    chunk_size = len(data_list) // num_workers
    data_chunks = [data_list[i:i+chunk_size] for i in range(0, len(data_list), chunk_size)]
    
    # Process in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_chunk, model, chunk) for chunk in data_chunks]
        results = [future.result() for future in futures]
    
    # Combine results
    combined_results = combine_chunk_results(results)
    
    return combined_results
```

## Troubleshooting

### Common Issues

#### Memory Errors
```python
# Solutions:
# 1. Reduce batch size
batch_size = 1  # Reduce from default

# 2. Use gradient checkpointing
model.gradient_checkpointing_enable()

# 3. Clear GPU cache
torch.cuda.empty_cache()
```

#### Slow Inference
```python
# Solutions:
# 1. Enable CUDA optimizations
torch.backends.cudnn.benchmark = True

# 2. Use mixed precision
with torch.autocast(device_type='cuda'):
    embeddings = model(data.x, data.edge_index)

# 3. Optimize data loading
data_loader = DataLoader(dataset, batch_size=batch_size, 
                        num_workers=4, pin_memory=True)
```

#### Poor Quality Results
```python
# Solutions:
# 1. Check model configuration matches training
# 2. Verify data preprocessing is identical
# 3. Adjust confidence threshold
# 4. Check for data distribution shifts
```

## Best Practices

### Inference Configuration

1. **Model Consistency**: Ensure inference parameters match training exactly
2. **Data Preprocessing**: Use identical preprocessing as training
3. **Confidence Thresholds**: Start with recommended thresholds and adjust based on data
4. **Memory Management**: Monitor GPU memory usage and optimize batch sizes

### Quality Assurance

1. **Validation Checks**: Verify inference results against known ground truth
2. **Confidence Analysis**: Analyze score distributions for appropriate thresholds
3. **Fragment Detection**: Enable fragment detection for comprehensive coverage
4. **Post-processing**: Apply quality control filters to remove low-quality assignments

### Performance Optimization

1. **GPU Utilization**: Maximize GPU usage with appropriate batch sizes
2. **Parallel Processing**: Use multiple workers for data loading
3. **Memory Efficiency**: Optimize memory usage with mixed precision
4. **Batch Processing**: Process large datasets in manageable chunks

## Future Enhancements

Planned inference improvements include:

- **Real-time Inference**: Streaming inference for live data
- **Model Compression**: Quantized models for faster inference
- **Distributed Inference**: Multi-node inference capabilities
- **Adaptive Thresholding**: Dynamic confidence thresholds based on data
- **Uncertainty Quantification**: Confidence intervals for predictions
