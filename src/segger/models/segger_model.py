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
        self.conv_first = GATv2Conv(
            (-1, -1), hidden_channels, heads=heads, add_self_loops=False
        )
        # self.lin_first = Linear(-1, hidden_channels * heads)

        # Middle GATv2Conv layers
        self.num_mid_layers = num_mid_layers
        if num_mid_layers > 0:
            self.conv_mid_layers = torch.nn.ModuleList()
            # self.lin_mid_layers = torch.nn.ModuleList()
            for _ in range(num_mid_layers):
                self.conv_mid_layers.append(
                    GATv2Conv(
                        (-1, -1), hidden_channels, heads=heads, add_self_loops=False
                    )
                )
                # self.lin_mid_layers.append(Linear(-1, hidden_channels * heads))

        # Last GATv2Conv layer
        self.conv_last = GATv2Conv(
            (-1, -1), out_channels, heads=heads, add_self_loops=False
        )
        # self.lin_last = Linear(-1, out_channels * heads)

    def forward(self, x: Tensor, edge_index: Tensor) -> tuple[Tensor, list]:
        """
        Forward pass for the Segger model.

        Args:
            x (Tensor): Node features.
            edge_index (Tensor): Edge indices.

        Returns:
            Tensor: (Output node embeddings, list of (edge_index, attention_weights) tuples or empty list).
        """
        # record the attention weights
        attention_weights = []
        
        x = torch.nan_to_num(x, nan=0)
        is_one_dim = (x.ndim == 1) * 1
        x = x[:, None]
        x = self.tx_embedding(
            ((x.sum(-1) * is_one_dim).int())
        ) * is_one_dim + self.lin0(x.float()) * (1 - is_one_dim)
        x = x.squeeze()
        x = x.relu()
        # First layer
        x, (edge_index_first, alpha_first) = self.conv_first(x, edge_index, return_attention_weights=True)  # + self.lin_first(x)
        attention_weights.append((edge_index_first, alpha_first))
        x = x.relu()

        # Middle layers
        if self.num_mid_layers > 0:
            for conv_mid in self.conv_mid_layers:
                x, (edge_index_mid, alpha_mid) = conv_mid(x, edge_index, return_attention_weights=True)  # + lin_mid(x)
                attention_weights.append((edge_index_mid, alpha_mid))
                x = x.relu()

        x, (edge_index_last, alpha_last) = self.conv_last(x, edge_index, return_attention_weights=True)  # + self.lin_last(x)
        attention_weights.append((edge_index_last, alpha_last))

        return x, attention_weights

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
