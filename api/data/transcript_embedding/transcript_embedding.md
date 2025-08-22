# segger.data.transcript_embedding

The `transcript_embedding` module provides utilities for encoding transcript features into numerical representations suitable for machine learning models. This module handles the conversion of gene names and transcript labels into embeddings that can be used in graph neural networks.



::: src.segger.data.transcript_embedding.TranscriptEmbedding
    options:
      show_root_heading: true
      show_root_toc_entry: true
      show_submodules: true
      merge_init_into_class: false
      members: true                 
      heading_level: 2            
      show_root_full_path: false

## Overview

The `TranscriptEmbedding` class is designed to handle transcript feature encoding in a flexible and extensible way. It supports both simple index-based encoding and weighted embeddings, making it suitable for various machine learning applications.

## Usage Examples

### Basic Index-based Encoding

```python
from segger.data.transcript_embedding import TranscriptEmbedding
import pandas as pd

# Create a list of gene names
gene_names = ["GENE1", "GENE2", "GENE3", "GENE4"]

# Initialize embedding without weights (index-based)
embedding = TranscriptEmbedding(classes=gene_names)

# Encode transcript labels
transcript_labels = ["GENE1", "GENE3", "GENE2"]
encoded = embedding.embed(transcript_labels)
# Returns: tensor([0, 2, 1])
```

### Weighted Embeddings

```python
import pandas as pd

# Create weights DataFrame
weights_df = pd.DataFrame({
    'weight1': [0.1, 0.2, 0.3, 0.4],
    'weight2': [0.5, 0.6, 0.7, 0.8]
}, index=["GENE1", "GENE2", "GENE3", "GENE4"])

# Initialize embedding with weights
embedding = TranscriptEmbedding(
    classes=gene_names,
    weights=weights_df
)

# Encode transcript labels
transcript_labels = ["GENE1", "GENE3"]
encoded = embedding.embed(transcript_labels)
# Returns: tensor([[0.1, 0.5], [0.3, 0.7]])
```

### Integration with PyTorch

```python
import torch
from segger.data.transcript_embedding import TranscriptEmbedding

# Create embedding module
embedding = TranscriptEmbedding(classes=gene_names)

# Use in a neural network
class TranscriptEncoder(torch.nn.Module):
    def __init__(self, gene_names):
        super().__init__()
        self.embedding = TranscriptEmbedding(gene_names)
        self.projection = torch.nn.Linear(len(gene_names), 128)
    
    def forward(self, transcript_labels):
        encoded = self.embedding.embed(transcript_labels)
        projected = self.projection(encoded)
        return projected

# Initialize and use
encoder = TranscriptEncoder(gene_names)
output = encoder(["GENE1", "GENE2"])
```
