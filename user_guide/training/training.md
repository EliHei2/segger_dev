# Training the `segger` Model

## The Model

The `segger` model is a graph neural network designed to handle heterogeneous graphs with two primary node types: **transcripts** and **nuclei or cell boundaries**. It leverages attention-based graph convolutional layers to compute node embeddings and relationships in spatial transcriptomics data. The architecture includes an initial embedding layer for node feature transformation, multiple graph attention layers (GATv2Conv), and residual linear connections.

### Model Architecture

- **Input Node Features**:  
   For input node features $\mathbf{x}$, the model distinguishes between transcript nodes and boundary (or nucleus) nodes.

   - **Transcript Nodes**: If $\mathbf{x}$ is 1-dimensional (e.g., for tokenized transcript data), the model applies an embedding layer:

$$
\mathbf{h}_{i}^{(0)} = \text{nn.Embedding}(i)
$$

   where $i$ is the transcript token index.

   - **Nuclei or Cell Boundary Nodes**: If $\mathbf{x}$ has multiple dimensions, the model applies a linear transformation:

$$
\mathbf{h}_{i}^{(0)} = \mathbf{W} \mathbf{x}_{i}
$$

   where $\mathbf{W}$ is a learnable weight matrix.

- **Graph Attention Layers (GATv2Conv)**:  
   The node embeddings are updated through multiple attention-based layers. The update for a node $i$ at layer $l+1$ is given by:

$$
\mathbf{h}_{i}^{(l+1)} = \text{ReLU}\left( \sum_{j \in \mathcal{N}(i)} \alpha_{ij} \mathbf{W}^{(l)} \mathbf{h}_{j}^{(l)} \right)
$$

   where:
   - $\alpha_{ij}$ is the attention coefficient between node $i$ and node $j$, computed as:

$$
\alpha_{ij} = \frac{\exp\left( \text{LeakyReLU}\left( \mathbf{a}^{\top} \left[\mathbf{W}^{(l)} \mathbf{h}_{i}^{(l)} || \mathbf{W}^{(l)} \mathbf{h}_{j}^{(l)}\right] \right)\right)}{\sum_{k \in \mathcal{N}(i)} \exp\left( \text{LeakyReLU}\left( \mathbf{a}^{\top} \left[\mathbf{W}^{(l)} \mathbf{h}_{i}^{(l)} || \mathbf{W}^{(l)} \mathbf{h}_{k}^{(l)}\right] \right)\right)}
$$

   - $\mathbf{a}$ is a learnable attention vector.

- **Residual Linear Connections**:  
   After each attention layer, a residual connection is added via a linear transformation to stabilize learning:

$$
\mathbf{h}_{i}^{(l+1)} = \text{ReLU}\left( \mathbf{h}_{i}^{(l+1)} + \mathbf{W}_{res} \mathbf{h}_{i}^{(l)} \right)
$$

   where $\mathbf{W}_{res}$ is a residual weight matrix.

- **L2 Normalization**:  
   Finally, the embeddings are normalized using L2 normalization:

$$
\mathbf{h}_{i} = \frac{\mathbf{h}_{i}}{\|\mathbf{h}_{i}\|}
$$

   ensuring the final node embeddings have unit norm.

### Heterogeneous Graph Transformation

In the next step, the `segger` model is transformed into a **heterogeneous graph neural network** using PyTorch Geometric's `to_hetero` function. This transformation enables the model to handle distinct node and edge types (transcripts and nuclei or cell boundaries) with separate mechanisms for modeling their relationships.

### Usage

To instantiate and run the `segger` model:

```python
model = segger(
    num_tx_tokens=5000,  # Number of unique 'tx' tokens
    init_emb=32,  # Initial embedding dimension
    hidden_channels=64,  # Number of hidden channels
    num_mid_layers=2,  # Number of middle layers
    out_channels=128,  # Number of output channels
    heads=4,  # Number of attention heads
)

output = model(x, edge_index)
```

Once transformed to a heterogeneous model and trained using PyTorch Lightning, the model can efficiently learn relationships between transcripts and nuclei or cell boundaries.

## Training the heterogeneous GNN with `pytorch-lightning`

The training module makes use of **PyTorch Lightning** for efficient and scalable training, alongside **PyTorch Geometric** for processing the graph-based data. The module is built to handle multi-GPU setups and allows the flexibility to adjust hyperparameters, aggregation methods, and embedding sizes.


The `SpatialTranscriptomicsDataset` class is used to load and manage spatial transcriptomics data stored in the format of PyTorch Geometric `Data` objects. It inherits from `InMemoryDataset` to load preprocessed datasets, ensuring efficient in-memory data handling for training and validation phases.

### Example Training Command

```bash
python scripts/train_model.py \ 
  --train_dir path/to/train/tiles \
  --val_dir path/to/val/tiles \
  --batch_size_train 4 \
  --batch_size_val 4 \
  --num_tx_tokens 500 \
  --init_emb 8 \
  --hidden_channels 64 \
  --out_channels 16 \
  --heads 4 \
  --mid_layers 1 \
  --aggr sum \
  --accelerator cuda \
  --strategy auto \
  --precision 16-mixed \
  --devices 4 \
  --epochs 100 \
  --default_root_dir ./models/clean2
```

The `scripts/train_model.py` file can be found on the github repo. This example submits a job to train the `segger` model on 4 GPUs with a batch size of 4 for both training and validation, utilizing 16-bit mixed precision.

