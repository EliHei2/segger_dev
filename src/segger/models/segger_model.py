import torch
from torch_geometric.nn import GATv2Conv, Linear, to_hetero
from torch import Tensor
from typing import Tuple, Union
from torch_geometric.typing import SparseTensor

class Segger(torch.nn.Module):
    def __init__(self, init_emb: int = 16, hidden_channels: int = 32, num_mid_layers: int = 3, out_channels: int = 32, heads: int = 3):
        """
        Initializes the Segger model.

        Args:
            init_emb (int)       : Initial embedding size.
            hidden_channels (int): Number of hidden channels.
            num_mid_layers (int) : Number of hidden layers (excluding first and last layers).
            out_channels (int)   : Number of output channels.
            heads (int)          : Number of attention heads.
        """
        super().__init__()
        self.num_mid_layers = num_mid_layers
        self.lin0 = Linear(-1, init_emb, bias=False)
        self.conv_first = GATv2Conv((-1, -1), hidden_channels, heads=heads, add_self_loops=False)
        self.lin_first = Linear(-1, hidden_channels * heads)
        if num_mid_layers > 0:
            self.conv_mid = GATv2Conv((-1, -1), hidden_channels, heads=heads, add_self_loops=False)
            self.lin_mid = Linear(-1, hidden_channels * heads)
        self.conv_last = GATv2Conv((-1, -1), out_channels, heads=heads, add_self_loops=False)
        self.lin_last = Linear(-1, out_channels * heads)

    def forward(self, x: Tensor, edge_index: Union[Tensor, SparseTensor]) -> Tensor:
        """
        Forward pass for the Segger model.

        Args:
            x (Tensor): Node features.
            edge_index (EdgeIndex): Edge indices.

        Returns:
            Tensor: Output node embeddings.
        """
        x = self.lin0(x)
        x = x.relu()
        x = self.conv_first(x, edge_index) + self.lin_first(x)
        x = x.relu()
        if self.num_mid_layers > 0:
            for i in range(self.num_mid_layers):
                x = self.conv_mid(x, edge_index) + self.lin_mid(x)
                x = x.relu()
        x = self.conv_last(x, edge_index) + self.lin_last(x)
        x = x / x.norm(dim=-1, keepdim=True)  # Normalize to L2 norm of 1
        return x
    
    def decode(self, z: Tensor, edge_index: Union[Tensor, SparseTensor]) -> Tensor:
        """
        Decode the node embeddings to predict edge values.

        Args:
            z (Tensor): Node embeddings.
            edge_label_index (EdgeIndex): Edge label indices.

        Returns:
            Tensor: Predicted edge values.
        """
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
