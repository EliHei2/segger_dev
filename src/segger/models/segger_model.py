import torch
from torch_geometric.nn import GATv2Conv, Linear
from torch.nn import Embedding
from torch import Tensor
from typing import Union

# from torch_sparse import SparseTensor


class Segger(torch.nn.Module):
    def __init__(
        self,
        num_tx_tokens: int,
        init_emb: int = 16,
        hidden_channels: int = 32,
        num_mid_layers: int = 3,
        out_channels: int = 32,
        heads: int = 3,
    ):
        """
        Initializes the Segger model.

        Args:
            num_tx_tokens (int)  : Number of unique 'tx' tokens for embedding.
            init_emb (int)       : Initial embedding size for both 'tx' and boundary (non-token) nodes.
            hidden_channels (int): Number of hidden channels.
            num_mid_layers (int) : Number of hidden layers (excluding first and last layers).
            out_channels (int)   : Number of output channels.
            heads (int)          : Number of attention heads.
        """
        super().__init__()

        # Embedding for 'tx' (transcript) nodes
        self.tx_embedding = Embedding(num_tx_tokens, init_emb)

        # Linear layer for boundary (non-token) nodes
        self.lin0 = Linear(-1, init_emb, bias=False)

        # First GATv2Conv layer
        self.conv_first = GATv2Conv((-1, -1), hidden_channels, heads=heads, add_self_loops=False)
        self.lin_first = Linear(-1, hidden_channels * heads)

        # Middle GATv2Conv layers
        self.num_mid_layers = num_mid_layers
        if num_mid_layers > 0:
            self.conv_mid_layers = torch.nn.ModuleList()
            self.lin_mid_layers = torch.nn.ModuleList()
            for _ in range(num_mid_layers):
                self.conv_mid_layers.append(GATv2Conv((-1, -1), hidden_channels, heads=heads, add_self_loops=False))
                self.lin_mid_layers.append(Linear(-1, hidden_channels * heads))

        # Last GATv2Conv layer
        self.conv_last = GATv2Conv((-1, -1), out_channels, heads=heads, add_self_loops=False)
        self.lin_last = Linear(-1, out_channels * heads)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        Forward pass for the Segger model.

        Args:
            x (Tensor): Node features.
            edge_index (Tensor): Edge indices.

        Returns:
            Tensor: Output node embeddings.
        """
        x = torch.nan_to_num(x, nan=0)
        is_one_dim = (x.ndim == 1) * 1
        # x = x[:, None]
        x = self.tx_embedding(((x.sum(1) * is_one_dim).int())) * is_one_dim + self.lin0(x.float()) * (1 - is_one_dim)
        # First layer
        x = x.relu()
        x = self.conv_first(x, edge_index)  + self.lin_first(x)
        x = x.relu()

        # Middle layers
        if self.num_mid_layers > 0:
            for i in range(self.num_mid_layers):
                conv_mid = self.conv_mid_layers[i]
                lin_mid  = self.lin_mid_layers[i]
                x = conv_mid(x, edge_index)  + lin_mid(x)
                x = x.relu()

        # Last layer
        x = self.conv_last(x, edge_index)  + self.lin_last(x)

        return x

    def decode(self, z: Tensor, edge_index: Union[Tensor]) -> Tensor:
        """
        Decode the node embeddings to predict edge values.

        Args:
            z (Tensor): Node embeddings.
            edge_index (EdgeIndex): Edge label indices.

        Returns:
            Tensor: Predicted edge values.
        """
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
