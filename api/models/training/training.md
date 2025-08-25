# Segger Model Training

## Overview

Training the Segger model involves optimizing a Graph Neural Network for transcript-to-cell link prediction in spatial transcriptomics data. The training process leverages PyTorch Lightning for scalable multi-GPU training, with specialized data handling and validation strategies designed for heterogeneous graphs.

## Training Framework

### PyTorch Lightning Integration

Segger uses PyTorch Lightning for training orchestration, providing:

- **Multi-GPU Training**: Automatic data parallel training across multiple devices
- **Mixed Precision**: Support for 16-bit mixed precision training
- **Distributed Training**: Multi-node training capabilities
- **Automatic Logging**: Built-in metrics tracking and visualization
- **Checkpoint Management**: Automatic model saving and restoration

### Training Architecture

```
Training Tiles → Data Loaders → Segger Model → Link Prediction → Loss Computation → Optimization
     ↓              ↓              ↓              ↓              ↓              ↓
Spatial Graphs   Mini-batches   GNN Forward    Similarity      Binary CE      Adam Optimizer
Validation Set   GPU Transfer   Attention      Scores          Loss + AUROC   Weight Updates
```

## Data Preparation

### Training Data Structure

Training data consists of spatial tiles represented as PyTorch Geometric graphs:

```python
# Each tile contains:
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
    "tx,belongs,bd": {  # Transcript-boundary edges
        "edge_index": containment_relationships,
        "edge_label": positive/negative labels
    }
}
```

### Data Splitting Strategy

Tiles are randomly assigned to training, validation, and test sets:

```python
# Recommended split ratios
train_ratio = 0.7    # 70% for training
val_ratio = 0.2      # 20% for validation  
test_ratio = 0.1     # 10% for testing

# Spatial-aware splitting ensures:
# - No information leakage between splits
# - Representative spatial coverage in each split
# - Balanced distribution of cell types
```

### Negative Edge Sampling

To handle class imbalance, negative edges are sampled during training:

```python
# Sample negative edges at 1:5 ratio (positive:negative)
neg_sampling_ratio = 5

# Negative edges represent:
# - Transcripts assigned to wrong cells
# - Random transcript-cell pairs
# - Spatially distant but transcriptionally similar pairs
```

## Training Configuration

### Model Parameters

Key training parameters based on the Segger paper:

```python
# Architecture configuration
model_config = {
    'num_tx_tokens': 5000,      # Vocabulary size (adjust for dataset)
    'init_emb': 16,             # Initial embedding dimension
    'hidden_channels': 64,       # Hidden layer size
    'num_mid_layers': 3,        # Number of GAT layers
    'out_channels': 32,          # Output dimension
    'heads': 4                   # Number of attention heads
}

# Training configuration
training_config = {
    'learning_rate': 0.001,     # Initial learning rate
    'batch_size': 2,            # Batch size per GPU
    'max_epochs': 200,          # Maximum training epochs
    'weight_decay': 1e-5,       # L2 regularization
    'patience': 10,             # Early stopping patience
}
```

### Hardware Configuration

```python
# GPU configuration
gpu_config = {
    'accelerator': 'cuda',       # Use CUDA acceleration
    'devices': 4,               # Number of GPUs
    'strategy': 'ddp',          # Distributed data parallel
    'precision': '16-mixed'     # Mixed precision training
}

# Memory optimization
memory_config = {
    'gradient_clip_val': 1.0,   # Gradient clipping
    'accumulate_grad_batches': 1, # Gradient accumulation
    'num_workers': 4            # Data loading workers
}
```

## Training Process

### Training Loop

The training process follows this sequence:

```python
# Training loop (PyTorch Lightning handles this automatically)
for epoch in range(max_epochs):
    # Training phase
    for batch in train_loader:
        # Forward pass
        embeddings = model(batch.x, batch.edge_index)
        
        # Link prediction
        scores = model.decode(embeddings, batch.edge_label_index)
        
        # Loss computation
        loss = criterion(scores, batch.edge_label)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # Validation phase
    for batch in val_loader:
        with torch.no_grad():
            embeddings = model(batch.x, batch.edge_index)
            scores = model.decode(embeddings, batch.edge_label_index)
            val_loss = criterion(scores, batch.edge_label)
            
            # Compute metrics
            auroc = compute_auroc(scores, batch.edge_label)
            f1_score = compute_f1(scores, batch.edge_label)
```

### Loss Function

The model uses binary cross-entropy loss for link prediction:

```python
# Binary cross-entropy loss
criterion = nn.BCEWithLogitsLoss()

# Loss computation
loss = -Σ_(t_i,c_j) [y_ij log(σ(s_ij)) + (1-y_ij) log(1-σ(s_ij))]

# Where:
# y_ij: Ground truth label (1 for positive, 0 for negative)
# s_ij: Raw similarity score from model
# σ(s_ij): Sigmoid activation for probability
```

### Optimization

Training uses the Adam optimizer with learning rate scheduling:

```python
# Optimizer configuration
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay,
    betas=(0.9, 0.999)
)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=max_epochs,
    eta_min=1e-6
)
```

## Validation and Monitoring

### Validation Metrics

The model is evaluated using:

#### AUROC (Area Under ROC Curve)
```python
def compute_auroc(scores, labels):
    """Compute Area Under ROC Curve for link prediction."""
    fpr, tpr, _ = roc_curve(labels, scores)
    return auc(fpr, tpr)
```

#### F1 Score
```python
def compute_f1(scores, labels):
    """Compute F1 score for link prediction."""
    predictions = (scores > 0.5).float()
    return f1_score(labels, predictions)
```

### Training Monitoring

PyTorch Lightning provides automatic logging:

```python
# Metrics logged automatically
self.log('train_loss', train_loss, on_step=True, on_epoch=True)
self.log('val_loss', val_loss, on_epoch=True)
self.log('val_auroc', val_auroc, on_epoch=True)
self.log('val_f1', val_f1, on_epoch=True)

# Learning rate logging
self.log('lr', self.optimizer.param_groups[0]['lr'], on_epoch=True)
```

### Early Stopping

Training stops automatically when validation performance plateaus:

```python
# Early stopping callback
early_stopping = EarlyStopping(
    monitor='val_auroc',
    mode='max',
    patience=patience,
    verbose=True
)

# Model checkpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor='val_auroc',
    mode='max',
    save_top_k=3,
    filename='segger-{epoch:02d}-{val_auroc:.3f}'
)
```

## Multi-GPU Training

### Data Parallel Strategy

Segger supports distributed training across multiple GPUs:

```python
# Distributed training configuration
trainer = pl.Trainer(
    accelerator='cuda',
    devices=4,                    # Use 4 GPUs
    strategy='ddp',               # Distributed data parallel
    precision='16-mixed',         # Mixed precision
    max_epochs=max_epochs,
    callbacks=[early_stopping, checkpoint_callback]
)
```

### Batch Size Scaling

```python
# Effective batch size = batch_size × num_gpus
effective_batch_size = batch_size * num_gpus

# Example: batch_size=2, num_gpus=4
# Effective batch size = 8

# Adjust learning rate for larger effective batch size
scaled_lr = base_lr * (effective_batch_size / 32)  # Linear scaling rule
```

### Memory Management

```python
# Memory optimization techniques
memory_config = {
    'gradient_checkpointing': True,    # Trade compute for memory
    'find_unused_parameters': False,   # Optimize for DDP
    'sync_batchnorm': False,           # Not needed for GNNs
    'deterministic': False             # Allow non-deterministic operations
}
```

## Training Strategies

### Learning Rate Scheduling

#### Cosine Annealing
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=max_epochs,
    eta_min=1e-6
)
```

#### Warmup + Cosine
```python
# Warmup for first 10% of training
warmup_epochs = int(0.1 * max_epochs)

def get_lr_multiplier(epoch):
    if epoch < warmup_epochs:
        return epoch / warmup_epochs
    else:
        # Cosine decay
        progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))
```

### Regularization Techniques

#### Weight Decay
```python
# L2 regularization in optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    weight_decay=1e-5
)
```

#### Dropout (Optional)
```python
# Add dropout to attention layers if needed
class SeggerWithDropout(Segger):
    def __init__(self, *args, dropout=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index):
        # Apply dropout after attention layers
        x = super().forward(x, edge_index)
        x = self.dropout(x)
        return x
```

## Performance Optimization

### Mixed Precision Training

```python
# Enable mixed precision for faster training
trainer = pl.Trainer(
    precision='16-mixed',  # 16-bit mixed precision
    # Automatic mixed precision provides:
    # - Faster training (1.5-2x speedup)
    # - Lower memory usage
    # - Maintained numerical stability
)
```

### Gradient Accumulation

```python
# Accumulate gradients over multiple batches
trainer = pl.Trainer(
    accumulate_grad_batches=4,  # Effective batch size = batch_size × 4
    # Useful when:
    # - GPU memory is limited
    # - Large effective batch size is desired
    # - Training stability is important
)
```

### Data Loading Optimization

```python
# Optimize data loading
dataloader_config = {
    'num_workers': 4,           # Parallel data loading
    'pin_memory': True,         # Faster GPU transfer
    'persistent_workers': True,  # Keep workers alive between epochs
    'prefetch_factor': 2        # Prefetch batches
}
```

## Troubleshooting

### Common Training Issues

#### Training Instability
```python
# Solutions:
# 1. Reduce learning rate
learning_rate = 0.0001  # Reduce from 0.001

# 2. Add gradient clipping
trainer = pl.Trainer(gradient_clip_val=1.0)

# 3. Check data quality and normalization
```

#### Memory Errors
```python
# Solutions:
# 1. Reduce batch size
batch_size = 1  # Reduce from 2

# 2. Enable gradient checkpointing
trainer = pl.Trainer(enable_checkpointing=True)

# 3. Use mixed precision
trainer = pl.Trainer(precision='16-mixed')
```

#### Poor Convergence
```python
# Solutions:
# 1. Check learning rate schedule
# 2. Verify data preprocessing
# 3. Adjust model architecture
# 4. Check for data leakage
```

### Performance Monitoring

```python
# Monitor training progress
class TrainingMonitor(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        # Log training metrics
        train_loss = trainer.callback_metrics['train_loss']
        print(f"Epoch {trainer.current_epoch}: Train Loss = {train_loss:.4f}")
    
    def on_validation_epoch_end(self, trainer, pl_module):
        # Log validation metrics
        val_auroc = trainer.callback_metrics['val_auroc']
        val_f1 = trainer.callback_metrics['val_f1']
        print(f"Validation: AUROC = {val_auroc:.4f}, F1 = {val_f1:.4f}")
```

## Best Practices

### Training Configuration

1. **Start with Default Parameters**: Use recommended settings from the Segger paper
2. **Monitor Validation Metrics**: Focus on AUROC and F1 score, not just loss
3. **Use Early Stopping**: Prevent overfitting with patience-based stopping
4. **Enable Mixed Precision**: Use 16-bit training for speed and memory efficiency

### Data Preparation

1. **Quality Control**: Filter low-quality transcripts and boundaries
2. **Spatial Validation**: Ensure train/val/test splits are spatially representative
3. **Feature Normalization**: Normalize transcript and boundary features
4. **Negative Sampling**: Use appropriate negative sampling ratios

### Hardware Utilization

1. **Multi-GPU Training**: Scale training across multiple GPUs
2. **Memory Optimization**: Use mixed precision and gradient checkpointing
3. **Data Loading**: Optimize data loading with multiple workers
4. **Batch Size**: Use largest batch size that fits in memory

## Future Enhancements

Planned training improvements include:

- **Advanced Scheduling**: More sophisticated learning rate schedules
- **Automated Hyperparameter Tuning**: Integration with Optuna or similar tools
- **Curriculum Learning**: Progressive difficulty training strategies
- **Multi-task Training**: Joint training on multiple objectives
- **Federated Learning**: Distributed training across multiple institutions
