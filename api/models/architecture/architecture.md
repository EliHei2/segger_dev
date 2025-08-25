# Segger Model Architecture

## Overview

The Segger model implements a sophisticated Graph Neural Network architecture specifically designed for spatial transcriptomics data analysis. The model formulates cell segmentation as a transcript-to-cell link prediction task, leveraging heterogeneous graphs with specialized attention mechanisms to learn complex spatial relationships.

## Core Design Principles

### 1. Heterogeneous Graph Representation

Segger models spatial transcriptomics data as a heterogeneous graph G = (V, E) with two distinct node types:

- **Transcript Nodes (T)**: Represent individual gene expression measurements with spatial coordinates
- **Boundary Nodes (C)**: Represent cell or region boundaries as geometric polygons

The graph contains two types of edges:
- **ETT**: Transcript-to-transcript edges capturing spatial colocalization
- **ETC**: Transcript-to-cell edges representing initial assignments

### 2. Link Prediction Framework

Rather than traditional image-based segmentation, Segger frames cell segmentation as a link prediction problem:

```
Given: Heterogeneous graph with transcript and boundary nodes
Task: Predict transcript-to-cell associations
Output: Probability scores for each transcript-cell pair
```

This approach enables the model to:
- Learn from spatial relationships between transcripts
- Leverage biological priors through gene embeddings
- Handle incomplete initial segmentations
- Identify both cells and cell fragments

## Model Architecture Details

### Input Processing Layer

The model automatically handles different input types through specialized processing:

#### Transcript Node Processing

```python
# For transcript nodes with gene embeddings
if has_scrnaseq_embeddings:
    x_tx = gene_celltype_embeddings[gene_labels]
else:
    # One-hot encoding fallback
    x_tx = embedding_layer(gene_token_ids)
```

#### Boundary Node Processing

```python
# For boundary nodes, compute geometric features
boundary_features = [
    area(Bi),           # Surface area
    convexity(Bi),      # A(Conv(Bi))/A(Bi)
    elongation(Bi),     # A(MBR(Bi))/A(Env(Bi))
    circularity(Bi)     # A(Bi)/r_min(Bi)²
]
x_bd = linear_transform(boundary_features)
```

### Graph Attention Layers

The core of the model consists of multiple GATv2Conv layers:

#### Layer Structure

```
Input Features → Node Type Detection → Feature Processing → GATv2Conv Layers → Output Embeddings
     ↓              ↓                    ↓                ↓                ↓
Transcripts    Auto-routing         Embedding/Linear   Attention Mech.   Learned Features
Boundaries     (1D vs Multi-D)     Transformations    Spatial Learning   Biological Insights
```

#### GATv2Conv Implementation

Each GATv2Conv layer computes attention coefficients:

```python
# Attention computation for edge (i, j)
e_ij = a^T LeakyReLU([Wh_i || Wh_j])

# Normalized attention weights
α_ij = softmax(e_ij) = exp(e_ij) / Σ_k exp(e_ik)

# Node update
h_i^(l+1) = σ(Σ_j∈N(i) α_ij W^(l) h_j^(l))
```

Where:
- `W^(l)` is a learnable weight matrix for layer l
- `a` is a learnable attention vector
- `σ` is the activation function (ReLU)
- `N(i)` represents the neighborhood of node i

#### Multi-head Attention

The model uses multiple attention heads in parallel:

```python
# Multi-head attention with K heads
h_i^(l+1) = ||_k=1^K σ(Σ_j∈N(i) α_ij^(k) W^(k) h_j^(l))
```

This allows the model to capture different types of relationships simultaneously.

### Layer Configuration

The model architecture is configurable with the following parameters:

```python
class Segger(torch.nn.Module):
    def __init__(
        self,
        num_tx_tokens: int,      # Vocabulary size for transcripts
        init_emb: int = 16,      # Initial embedding dimension
        hidden_channels: int = 32, # Hidden layer size
        num_mid_layers: int = 3,  # Number of hidden GAT layers
        out_channels: int = 32,   # Output embedding dimension
        heads: int = 3,           # Number of attention heads
    ):
```

#### Recommended Architecture Sizes

Based on the Segger paper:

- **Small Datasets** (< 10k nodes): 2-3 layers, 32-64 hidden channels
- **Medium Datasets** (10k-100k nodes): 3-4 layers, 64-128 hidden channels  
- **Large Datasets** (> 100k nodes): 4-5 layers, 128-256 hidden channels

### Output Processing

The final layer produces embeddings in a common latent space:

```python
# Final embeddings for link prediction
f(t_i) ∈ R^d3  # Transcript embedding
f(c_j) ∈ R^d3  # Cell embedding

# Similarity score computation
s_ij = ⟨f(t_i), f(c_j)⟩ = f(t_i)^T f(c_j)

# Probability of association
P(link_ij) = σ(s_ij) = 1 / (1 + e^(-s_ij))
```

## Heterogeneous Graph Processing

### Edge Type Handling

The model processes different edge types with specialized attention:

#### Transcript-Transcript Edges (ETT)

- **Purpose**: Capture spatial proximity and gene co-expression patterns
- **Construction**: k-nearest neighbor graph with distance threshold
- **Parameters**: `k_tx` (number of neighbors), `dist_tx` (maximum distance)

#### Transcript-Cell Edges (ETC)

- **Purpose**: Represent initial transcript-to-cell assignments
- **Construction**: Spatial overlap between transcripts and boundaries
- **Training**: Used as positive examples for link prediction

### Message Passing Mechanism

Information flows through the heterogeneous graph:

```
Transcript Nodes ←→ Transcript Nodes (spatial proximity)
       ↓                    ↑
       ↓                    ↑
    Cell Nodes ←→ Transcript Nodes (containment)
```

This enables:
- **Spatial Context**: Transcripts learn from nearby neighbors
- **Biological Context**: Transcripts learn from cell assignments
- **Geometric Context**: Cells learn from transcript distributions

## Training Strategy

### Loss Function

The model uses binary cross-entropy loss for link prediction:

```python
L = -Σ_(t_i,c_j)∈E_TC [y_ij log(σ(s_ij)) + (1-y_ij) log(1-σ(s_ij))]
```

Where:
- `y_ij = 1` for positive edges (observed transcript-cell associations)
- `y_ij = 0` for negative edges (sampled non-associations)

### Negative Sampling

To handle class imbalance, negative edges are sampled:

```python
# Sample negative edges at ratio 1:5 (positive:negative)
E^- = random_sample(E_TC^c, size=5|E^+|)
```

### Training Process

1. **Forward Pass**: Compute embeddings for all nodes
2. **Link Prediction**: Calculate similarity scores for all transcript-cell pairs
3. **Loss Computation**: Binary cross-entropy on positive and negative edges
4. **Backpropagation**: Update model parameters
5. **Validation**: Monitor AUROC and F1 score on validation set

## Performance Characteristics

### Computational Complexity

- **Time per layer**: O(|E| × F × H)
  - |E|: Number of edges
  - F: Feature dimension
  - H: Number of attention heads

- **Memory usage**: O(|V| × F + |E| × H)
  - |V|: Number of nodes
  - |E|: Number of edges

### Scalability Features

- **Efficient Attention**: GATv2Conv optimized for sparse graphs
- **GPU Acceleration**: Full CUDA support with PyTorch
- **Batch Processing**: Mini-batch training for large datasets
- **Multi-GPU**: Distributed training with PyTorch Lightning

## Integration with Segger Pipeline

### Data Flow

```
Spatial Data → Graph Construction → Segger Model → Node Embeddings → Link Prediction → Cell Segmentation
     ↓              ↓                ↓              ↓                ↓                ↓
Transcripts    Heterogeneous    GNN Processing   Learned Features   Similarity      Final
Boundaries     Graph Creation   Attention Mech.   Spatial Context    Scores         Assignments
```

### Key Integration Points

1. **Data Preprocessing**: Tiled graph construction with balanced regions
2. **Model Training**: PyTorch Lightning integration with validation
3. **Inference**: Batch-wise prediction with GPU acceleration
4. **Post-processing**: Connected components for unassigned transcripts

## Best Practices

### Architecture Selection

- **Layer Depth**: Balance expressiveness with over-smoothing (3-5 layers recommended)
- **Hidden Dimensions**: Scale with dataset size and task complexity
- **Attention Heads**: Use 4-8 heads for robust learning
- **Embedding Size**: Match output dimension to downstream tasks

### Training Configuration

- **Learning Rate**: Start with 0.001 and adjust based on convergence
- **Batch Size**: Use largest size that fits in memory
- **Regularization**: Apply weight decay (1e-5) to prevent overfitting
- **Early Stopping**: Monitor validation AUROC to prevent overfitting

### Data Preparation

- **Graph Construction**: Ensure proper spatial edge construction
- **Feature Engineering**: Provide meaningful transcript and boundary features
- **Validation Strategy**: Use spatial-aware train/val/test splits
- **Quality Control**: Filter low-quality transcripts and boundaries

## Future Enhancements

Planned improvements include:

- **Additional Attention Types**: Support for different attention mechanisms
- **Multi-modal Integration**: Support for additional data types (images, protein markers)
- **Distributed Training**: Multi-node training capabilities
- **Model Compression**: Efficient deployment of trained models
- **Interpretability Tools**: Understanding learned spatial relationships

## References

- **Graph Attention Networks**: Veličković et al. (2018) - ICLR
- **GATv2**: Brody et al. (2022) - ICLR  
- **Heterogeneous GNNs**: Hong et al. (2020) - AAAI
- **Link Prediction**: Kipf & Welling (2016) - arXiv
- **Spatial Transcriptomics**: Nature Reviews Genetics (2016)
