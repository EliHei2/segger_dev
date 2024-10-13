# segger: Graph Neural Network Model

The `segger` model is a graph neural network designed to handle heterogeneous graphs with two primary node types: **transcripts** and **nuclei or cell boundaries**. It leverages attention-based convolutional layers to compute node embeddings and relationships in spatial transcriptomics data. The architecture includes an initial embedding layer for node feature transformation, multiple graph attention layers (GATv2Conv), and residual linear connections.

## Model Architecture

1. **Input Node Features**:  
   For input node features \( \mathbf{x} \), the model distinguishes between one-dimensional (transcript) nodes and multi-dimensional (boundary or nucleus) nodes by checking the dimension of \( \mathbf{x} \).

   - **Transcript Nodes**: If \( \mathbf{x} \) is 1-dimensional (e.g., for tokenized transcript data), the model applies an embedding layer:

   $$
   \mathbf{h}_{i}^{(0)} = \text{Embedding}(i)
   $$

   where \( i \) is the transcript token index.

   - **Nuclei or Cell Boundary Nodes**: If \( \mathbf{x} \) has multiple dimensions, the model applies a linear transformation:

   $$
   \mathbf{h}_{i}^{(0)} = \mathbf{W} \mathbf{x}_{i}
   $$

   where \( \mathbf{W} \) is a learnable weight matrix.

2. **Graph Attention Layers (GATv2Conv)**:  
   The node embeddings are updated through multiple attention-based layers. The update for a node \( i \) at layer \( l+1 \) is given by:

   $$
   \mathbf{h}_{i}^{(l+1)} = \text{ReLU}\left( \sum_{j \in \mathcal{N}(i)} \alpha_{ij} \mathbf{W}^{(l)} \mathbf{h}_{j}^{(l)} \right)
   $$

   where:

   - \( \alpha\_{ij} \) is the attention coefficient between node \( i \) and node \( j \), computed as:

   $$
   \alpha_{ij} = \frac{\exp\left( \text{LeakyReLU}\left( \mathbf{a}^{\top} [\mathbf{W}^{(l)} \mathbf{h}_{i}^{(l)} || \mathbf{W}^{(l)} \mathbf{h}_{j}^{(l)}] \right)\right)}{\sum_{k \in \mathcal{N}(i)} \exp\left( \text{LeakyReLU}\left( \mathbf{a}^{\top} [\mathbf{W}^{(l)} \mathbf{h}_{i}^{(l)} || \mathbf{W}^{(l)} \mathbf{h}_{k}^{(l)}] \right)\right)}
   $$

   - \( \mathbf{a} \) is a learnable attention vector.

3. **Residual Linear Connections**:  
   After each attention layer, a residual connection is added via a linear transformation to stabilize learning:

   $$
   \mathbf{h}_{i}^{(l+1)} = \text{ReLU}\left( \mathbf{h}_{i}^{(l+1)} + \mathbf{W}_{res} \mathbf{h}_{i}^{(l)} \right)
   $$

   where \( \mathbf{W}\_{res} \) is a residual weight matrix.

4. **L2 Normalization**:  
   Finally, the embeddings are normalized using L2 normalization:

   $$
   \mathbf{h}_{i} = \frac{\mathbf{h}_{i}}{\|\mathbf{h}_{i}\|}
   $$

   ensuring the final node embeddings have unit norm.

## Heterogeneous Graph Transformation

In the next step, the `segger` model is transformed into a **heterogeneous graph neural network** using PyTorch Geometric's `to_hetero` function. This transformation enables the model to handle distinct node and edge types (transcripts and nuclei or cell boundaries) with separate mechanisms for modeling their relationships.

## Usage

To instantiate and run the segger model:

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
