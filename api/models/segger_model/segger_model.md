# segger.models.segger_model

The `segger_model` module contains the core Graph Neural Network architecture for spatial transcriptomics analysis. This module implements the `Segger` class, a sophisticated attention-based GNN designed specifically for processing heterogeneous graphs with transcript and boundary nodes.

## Core Classes

::: src.segger.models.segger_model.Segger
    options:
      show_root_heading: true
      show_root_toc_entry: true
      show_submodules: true
      merge_init_into_class: false
      members: true                 
      heading_level: 2            
      show_root_full_path: false

## Overview

The `Segger` class implements a Graph Neural Network architecture specifically designed for spatial transcriptomics data. It uses Graph Attention Networks (GAT) with GATv2Conv layers to learn complex spatial relationships between transcripts and cellular boundaries.

## Key Features

- **Heterogeneous Graph Processing**: Automatically handles different node types (transcripts vs. boundaries)
- **Attention Mechanisms**: GATv2Conv layers for learning spatial relationships
- **Configurable Architecture**: Adjustable depth, width, and attention heads
- **PyTorch Integration**: Native PyTorch module with full compatibility
- **Spatial Optimization**: Designed specifically for spatial transcriptomics data

## Architecture Details

### Node Type Processing

The model automatically differentiates between node types based on input feature dimensions:

1. **Transcript Nodes (1D features)**: Processed through embedding layers
2. **Boundary Nodes (Multi-dimensional features)**: Processed through linear transformations

### Layer Structure

```
Input Features → Node Type Detection → Feature Processing → GATv2Conv Layers → Output Embeddings
     ↓              ↓                    ↓                ↓                ↓
Transcripts    Auto-routing         Embedding/Linear   Attention Mech.   Learned Features
Boundaries     (1D vs Multi-D)     Transformations    Spatial Learning   Biological Insights
```

### Attention Mechanism

The model uses Graph Attention Networks (GAT) with the following attention computation:

```
α_ij = softmax(LeakyReLU(a^T [Wh_i || Wh_j]))
```

Where:
- `α_ij` is the attention coefficient between nodes i and j
- `a` is a learnable attention vector
- `W` is a learnable weight matrix
- `h_i, h_j` are node features

## Usage Examples

### Basic Model Initialization

```python
from segger.models.segger_model import Segger

# Initialize with default parameters
model = Segger(
    num_tx_tokens=5000,      # Number of unique transcript types
    init_emb=16,             # Initial embedding dimension
    hidden_channels=32,       # Hidden layer size
    num_mid_layers=3,        # Number of hidden layers
    out_channels=32,          # Output dimension
    heads=3                   # Number of attention heads
)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### Forward Pass

```python
import torch

# Create sample data
batch_size = 100
num_nodes = 1000
x = torch.randn(num_nodes, 64)  # Node features
edge_index = torch.randint(0, num_nodes, (2, 2000))  # Edge indices

# Forward pass
with torch.no_grad():
    output = model(x, edge_index)
    
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Edge index shape: {edge_index.shape}")
```

### Training Configuration

```python
import torch.nn as nn
import torch.optim as optim

# Model configuration for large dataset
model = Segger(
    num_tx_tokens=10000,     # Large vocabulary
    init_emb=64,             # Larger embeddings
    hidden_channels=128,      # Wider layers
    num_mid_layers=5,        # Deeper architecture
    out_channels=256,         # Rich output features
    heads=8                   # More attention heads
)

# Optimizer and loss
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()

# Training loop
model.train()
for epoch in range(100):
    optimizer.zero_grad()
    
    # Forward pass
    out = model(x, edge_index)
    
    # Compute loss (example: node classification)
    loss = criterion(out, labels)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

## Model Parameters

### Required Parameters

- **`num_tx_tokens`** (int): Number of unique transcript types in your dataset
  - This determines the size of the transcript embedding layer
  - Should match the number of unique genes/transcripts in your data

### Optional Parameters

- **`init_emb`** (int, default=16): Initial embedding dimension
  - Used for both transcript embeddings and boundary feature transformation
  - Larger values provide more expressive features but increase memory usage

- **`hidden_channels`** (int, default=32): Number of hidden channels
  - Size of intermediate layer representations
  - Affects model capacity and computational cost

- **`num_mid_layers`** (int, default=3): Number of hidden GAT layers
  - More layers enable learning of more complex patterns
  - Balance between expressiveness and overfitting

- **`out_channels`** (int, default=32): Output embedding dimension
  - Size of final node representations
  - Should match your downstream task requirements

- **`heads`** (int, default=3): Number of attention heads
  - Multiple heads learn different types of relationships
  - More heads generally improve performance but increase computation

## Architecture Components

### 1. Input Processing Layer

```python
# Automatic node type detection and processing
if x.ndim == 1:  # Transcript nodes
    x = self.tx_embedding(x.int())
else:  # Boundary nodes
    x = self.lin0(x.float())
```

### 2. Graph Attention Layers

```python
# First attention layer
x = F.relu(x)
x = self.conv_first(x, edge_index)
x = F.relu(x)

# Middle attention layers
for conv_mid in self.conv_mid_layers:
    x = conv_mid(x, edge_index)
    x = F.relu(x)

# Final attention layer
x = self.conv_last(x, edge_index)
```

### 3. Output Processing

```python
# Final embeddings can be used for various tasks
# - Node classification
# - Link prediction
# - Graph-level tasks
# - Downstream analysis
```

## Performance Characteristics

### Computational Complexity

- **Time Complexity**: O(|E| × F × H) per layer
  - |E|: Number of edges
  - F: Feature dimension
  - H: Number of attention heads

- **Memory Usage**: Scales with graph size and model parameters
  - Node features: O(|V| × F)
  - Edge attention: O(|E| × H)
  - Model parameters: O(F² × L × H)

### Optimization Features

- **Efficient Attention**: GATv2Conv optimized for sparse graphs
- **Memory Management**: Automatic handling of different node types
- **PyTorch Optimization**: Leverages PyTorch's optimized operations
- **GPU Acceleration**: Full CUDA support for training and inference

## Integration with PyTorch Geometric

The model is designed to work seamlessly with PyTorch Geometric:

```python
from torch_geometric.data import Data
from torch_geometric.transforms import ToUndirected

# Create PyG data object
data = Data(x=x, edge_index=edge_index)

# Apply transformations
data = ToUndirected()(data)

# Process with model
output = model(data.x, data.edge_index)
```

## Training Strategies

### 1. Learning Rate Scheduling

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(optimizer, T_max=100)
# Use in training loop
scheduler.step()
```

### 2. Regularization

```python
# Weight decay in optimizer
optimizer = optim.AdamW(model.parameters(), weight_decay=1e-5)

# Dropout (can be added to model if needed)
dropout = nn.Dropout(0.1)
x = dropout(x)
```

### 3. Early Stopping

```python
# Monitor validation loss
best_val_loss = float('inf')
patience = 10
patience_counter = 0

for epoch in range(max_epochs):
    # Training...
    val_loss = validate(model, val_loader)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        
    if patience_counter >= patience:
        print("Early stopping triggered")
        break
```

## Best Practices

### Model Architecture Selection

- **Small Datasets** (< 10k nodes): Use fewer layers and smaller dimensions
- **Medium Datasets** (10k-100k nodes): Balanced architecture with moderate complexity
- **Large Datasets** (> 100k nodes): Deeper models with more attention heads

### Training Configuration

- **Learning Rate**: Start with 0.001 and adjust based on convergence
- **Batch Size**: Use largest size that fits in memory
- **Regularization**: Apply weight decay and consider dropout
- **Monitoring**: Track both training and validation metrics

### Data Preparation

- **Feature Normalization**: Normalize input features for stable training
- **Graph Construction**: Ensure proper edge construction for spatial relationships
- **Validation Strategy**: Use spatial-aware validation splits
- **Data Augmentation**: Consider spatial augmentations for robustness

## Common Use Cases

### 1. Cell Type Classification

```python
# Train model for cell type prediction
model = Segger(num_tx_tokens=5000, out_channels=num_cell_types)
# ... training ...
predictions = model(x, edge_index)
cell_types = torch.argmax(predictions, dim=1)
```

### 2. Spatial Relationship Learning

```python
# Learn spatial relationships between transcripts and boundaries
embeddings = model(x, edge_index)
# Use embeddings for downstream analysis
similarity = torch.mm(embeddings, embeddings.t())
```

### 3. Tissue Architecture Analysis

```python
# Analyze tissue-level patterns
model = Segger(num_tx_tokens=5000, out_channels=128)
embeddings = model(x, edge_index)
# Apply clustering or other analysis to embeddings
```

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce model size or batch size
2. **Training Instability**: Lower learning rate or add regularization
3. **Poor Performance**: Check data quality and feature engineering
4. **Slow Convergence**: Adjust learning rate or model architecture

### Performance Tips

1. **Use appropriate model size** for your dataset
2. **Monitor training metrics** to detect issues early
3. **Validate on held-out data** to prevent overfitting
4. **Use mixed precision training** for faster training on modern GPUs

## Future Enhancements

Planned improvements include:

- **Additional Attention Types**: Support for different attention mechanisms
- **Multi-modal Integration**: Support for additional data types
- **Distributed Training**: Multi-GPU and multi-node support
- **Model Compression**: Efficient deployment of trained models
- **Interpretability Tools**: Understanding learned spatial relationships

## Dependencies

- **PyTorch**: Core neural network functionality
- **PyTorch Geometric**: Graph neural network operations
- **NumPy**: Numerical operations (optional, for data preprocessing)

## Contributing

Contributions to improve the Segger model are welcome:

- **Architecture Improvements**: Better attention mechanisms and layer designs
- **Performance Optimization**: Faster training and inference
- **Feature Extensions**: Support for additional node and edge types
- **Testing**: Comprehensive test coverage and validation
