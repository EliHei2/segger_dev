# Models Module API Reference

This page provides a comprehensive reference to all the classes, functions, and modules in the `segger.models` package.

## Module Overview

The `segger.models` package provides the machine learning core of the Segger framework, implementing Graph Neural Network architectures specifically designed for spatial transcriptomics data analysis.

## Core Modules

### [Segger Model](segger_model.md)
The main GNN architecture module containing the core Segger model.

**Key Classes:**
- `Segger`: Main Graph Neural Network model for spatial transcriptomics

**Main Functions:**
- Graph attention network processing
- Heterogeneous node type handling
- Spatial relationship learning
- PyTorch integration

### [Architecture Details](architecture.md)
Comprehensive documentation of the Segger model architecture and design principles.

**Key Topics:**
- Heterogeneous graph representation
- Link prediction framework
- GATv2Conv implementation
- Multi-head attention mechanisms
- Performance characteristics

### [Training Workflows](training.md)
Complete guide to training the Segger model with PyTorch Lightning.

**Key Topics:**
- PyTorch Lightning integration
- Multi-GPU training strategies
- Data preparation and validation
- Performance optimization
- Troubleshooting and best practices

### [Inference & Prediction](inference.md)
Comprehensive guide to using trained models for inference and cell segmentation.

**Key Topics:**
- Model loading and configuration
- Inference pipeline workflow
- Fragment detection
- Performance optimization
- Output formats and post-processing

## Core Classes

### Segger

::: src.segger.models.segger_model.Segger
    options:
      show_root_heading: false
      show_root_full_path: false
      show_if_no_docstring: false

## Model Architecture

The Segger model is built around a sophisticated Graph Neural Network architecture:

### Input Processing

The model automatically handles different input types:

- **Transcript Nodes**: 1D features processed through embedding layers
- **Boundary Nodes**: Multi-dimensional features processed through linear layers
- **Mixed Features**: Automatic routing based on feature dimensions

### Graph Attention Layers

Multiple GATv2Conv layers process the graph:

1. **Initial Layer**: Transforms input features to hidden representations
2. **Middle Layers**: Learn complex spatial relationships
3. **Final Layer**: Produces output embeddings

### Output Processing

Final node embeddings can be used for:

- **Node Classification**: Predicting cell types or states
- **Link Prediction**: Predicting spatial relationships
- **Graph Classification**: Tissue-level analysis
- **Downstream Tasks**: Integration with other ML models

## Data Flow

The typical model workflow follows this pattern:

```
Spatial Data → Graph Construction → Segger Model → Node Embeddings → Downstream Tasks
     ↓              ↓                ↓              ↓
Transcripts    Spatial Edges    GNN Processing   Learned Features
Boundaries     Node Features    Attention Mech.   Biological Insights
```

1. **Data Preparation**: Spatial data converted to graph format
2. **Model Initialization**: Segger model configured for specific task
3. **Forward Pass**: Graph processed through attention layers
4. **Feature Learning**: Spatial relationships captured in embeddings
5. **Application**: Embeddings used for biological analysis

## Usage Examples

### Basic Model Usage

```python
from segger.models.segger_model import Segger

# Initialize model
model = Segger(
    num_tx_tokens=5000,
    init_emb=32,
    hidden_channels=64,
    num_mid_layers=3,
    out_channels=128,
    heads=4
)

# Forward pass
output = model(x, edge_index)
```

### Training Configuration

```python
import torch.nn as nn
import torch.optim as optim

# Model configuration
model = Segger(
    num_tx_tokens=10000,
    hidden_channels=128,
    out_channels=256
)

# Optimizer and loss
optimizer = optim.AdamW(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    out = model(x, edge_index)
    loss = criterion(out, labels)
    loss.backward()
    optimizer.step()
```

### PyTorch Lightning Integration

```python
import pytorch_lightning as pl

class SeggerModule(pl.LightningModule):
    def __init__(self, num_tx_tokens, hidden_channels):
        super().__init__()
        self.model = Segger(
            num_tx_tokens=num_tx_tokens,
            hidden_channels=hidden_channels
        )
    
    def forward(self, x, edge_index):
        return self.model(x, edge_index)
    
    def training_step(self, batch, batch_idx):
        x, edge_index, labels = batch
        out = self(x, edge_index)
        loss = F.cross_entropy(out, labels)
        return loss

# Training
trainer = pl.Trainer(max_epochs=100)
trainer.fit(model, train_loader)
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

## Performance Characteristics

### Scalability

- **Memory Usage**: Scales with graph size and model complexity
- **Training Time**: Linear scaling with number of layers and attention heads
- **Inference**: Optimized for fast prediction on new data

### Optimization Features

- **Attention Mechanisms**: Efficient learning of spatial relationships
- **Residual Connections**: Stable training for deep architectures
- **Multi-head Processing**: Parallel attention for robust learning
- **Graph Optimization**: Optimized for PyTorch Geometric operations

## Integration with Segger Pipeline

The Segger model is designed to work seamlessly with the broader Segger framework:

1. **Data Processing**: Use `segger.data` modules to prepare spatial data
2. **Graph Construction**: Convert spatial data to PyTorch Geometric format
3. **Model Training**: Train Segger model on prepared graphs
4. **Inference**: Use trained model for predictions and analysis

## Error Handling

The module includes comprehensive error handling:

- **Input Validation**: Checks for valid input data and dimensions
- **Memory Management**: Handles out-of-memory situations gracefully
- **Graph Validation**: Ensures proper graph structure and connectivity
- **User Feedback**: Clear error messages for common issues

## Best Practices

### Model Architecture Selection

Choose appropriate architecture based on:

- **Data Size**: Larger datasets benefit from deeper models
- **Task Complexity**: Complex tasks need more attention heads
- **Computational Resources**: Balance model size with available resources

### Training Strategy

- **Learning Rate**: Start with 0.001 and adjust based on convergence
- **Batch Size**: Use largest batch size that fits in memory
- **Regularization**: Apply weight decay and consider dropout
- **Early Stopping**: Monitor validation performance to prevent overfitting

### Data Preparation

- **Graph Construction**: Ensure proper edge construction for spatial relationships
- **Feature Engineering**: Provide meaningful input features
- **Normalization**: Normalize features for stable training
- **Validation Split**: Use spatial-aware validation strategies

## Common Use Cases

### Research Applications

- **Cell Type Identification**: Process large tissue sections
- **Spatial Gene Expression**: Analyze gene expression patterns
- **Tissue Architecture**: Study spatial organization of cells

### Machine Learning

- **Graph Neural Networks**: Train GNNs on spatial transcriptomics data
- **Transfer Learning**: Adapt to new datasets and technologies
- **Scalable Training**: Process datasets too large for single machines

### Pipeline Development

- **Automated Processing**: Build reproducible analysis pipelines
- **Quality Control**: Integrate model predictions and validation
- **Multi-platform Support**: Handle data from different technologies

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce model size or batch size
2. **Training Instability**: Lower learning rate or add regularization
3. **Poor Performance**: Check data quality and feature engineering
4. **Slow Convergence**: Adjust learning rate or model architecture

### Performance Tips

1. **Use appropriate model size** for your hardware
2. **Enable parallel processing** for large datasets
3. **Cache frequently accessed data** when possible
4. **Filter data early** to reduce processing overhead

## Future Developments

The module is actively developed with plans for:

- **Additional Architectures**: Support for different GNN types
- **Enhanced Attention**: More sophisticated attention mechanisms
- **Multi-modal Integration**: Support for additional data types
- **Cloud Computing**: Support for distributed processing

## Contributing

Contributions are welcome! Areas for improvement include:

- **New Architectures**: Additional GNN architectures and attention mechanisms
- **Performance Optimization**: Better training and inference performance
- **Documentation**: Examples, tutorials, and best practices
- **Testing**: Comprehensive test coverage and validation

## Dependencies

### Core Dependencies

- **PyTorch**: Core neural network functionality
- **PyTorch Geometric**: Graph neural network support
- **NumPy**: Numerical operations

### Optional Dependencies

- **PyTorch Lightning**: Training framework integration
- **CUDA**: GPU acceleration support
- **TensorBoard**: Training visualization and monitoring
