import torch
from torch_geometric.nn import HeteroConv, GATv2Conv, HeteroDictLinear
from torch import nn
from torch import Tensor
from typing import Union

class SkipGAT(nn.Module):
    def __init__(self, in_channels, out_channels, heads, apply_activation=True):
        super().__init__()
        self.apply_activation = apply_activation
        self.conv = HeteroConv({
            ('tx', 'neighbors', 'tx'): GATv2Conv(in_channels, out_channels, heads=heads, add_self_loops=False),
            ('tx', 'belongs', 'bd'): GATv2Conv(in_channels, out_channels, heads=heads, add_self_loops=False),
        }, aggr='sum')
        self.lin = HeteroDictLinear(in_channels, out_channels * heads, types=('tx', 'bd'))

    def forward(self, x_dict, edge_index_dict):
        x_conv = self.conv(x_dict, edge_index_dict)
        x_lin = self.lin(x_dict)
        x_dict = {key: x_conv[key] + x_lin[key] for key in x_dict}
        if self.apply_activation:
            x_dict = {key: x_dict[key].relu() for key in x_dict}
        return x_dict


class Segger(nn.Module):
    def __init__(
        self,
        is_token_based: int,
        num_node_features: dict[str, int],
        init_emb: int = 16,
        hidden_channels: int = 32,
        num_mid_layers: int = 3,
        out_channels: int = 32,
        heads: int = 3,
    ):
        """
        Initializes the Segger model.

        Args:
            is_token_based (int)   : Whether the model is using token-based embeddings or scRNAseq embeddings.
            num_node_features (dict[str, int]): Number of node features for each node type.
            init_emb (int)         : Initial embedding size for both 'tx' and boundary (non-token) nodes.
            hidden_channels (int)  : Number of hidden channels.
            num_mid_layers (int)   : Number of hidden layers (excluding first and last layers).
            out_channels (int)     : Number of output channels.
            heads (int)            : Number of attention heads.
        """
        super().__init__()

        # Initialize node embeddings
        if is_token_based:
            # Using token-based embeddings for transcript ('tx') nodes
            self.node_init = nn.ModuleDict({
                'tx': nn.Embedding(num_node_features['tx'], init_emb),
                'bd': nn.Linear(num_node_features['bd'], init_emb),
            })
        else:
            # Using scRNAseq embeddings (i.e. prior biological knowledge) for transcript ('tx') nodes
            self.node_init = nn.ModuleDict({
                'tx': nn.Linear(num_node_features['tx'], init_emb),
                'bd': nn.Linear(num_node_features['bd'], init_emb),
            })

        # First GATv2Conv layer
        self.conv1 = SkipGAT(init_emb, hidden_channels, heads)

        # Middle GATv2Conv layers
        self.num_mid_layers = num_mid_layers
        if num_mid_layers > 0:
            self.conv_mid_layers = nn.ModuleList()
            for _ in range(num_mid_layers):
                self.conv_mid_layers.append(SkipGAT(heads * hidden_channels, hidden_channels, heads))
        
        # Last GATv2Conv layer
        self.conv_last = SkipGAT(heads * hidden_channels, out_channels, heads)

        # Finalize node embeddings
        self.node_final = HeteroDictLinear(heads * out_channels, out_channels, types=('tx', 'bd'))

        # # Edge probability predictor
        # self.edge_predictor = nn.Sequential(
        #     nn.Linear(2 * out_channels, out_channels),
        #     nn.ReLU(),
        #     nn.Linear(out_channels, 1),
        # )

    def forward(
        self,
        x_dict: dict[str, Tensor],
        edge_index_dict: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """
        Forward pass for the Segger model.

        Args:
            x_dict (dict[str, Tensor]): Node features for each node type.
            edge_index_dict (dict[str, Tensor]): Edge indices for each edge type.
        """

        x_dict = {key: self.node_init[key](x) for key, x in x_dict.items()}

        x_dict = self.conv1(x_dict, edge_index_dict)

        if self.num_mid_layers > 0:
            for i in range(self.num_mid_layers):
                x_dict = self.conv_mid_layers[i](x_dict, edge_index_dict)

        x_dict = self.conv_last(x_dict, edge_index_dict)

        x_dict = self.node_final(x_dict)

        return x_dict

    def decode(
            self,
            z_dict: dict[str, Tensor],
            edge_index: Union[Tensor],
        ) -> Tensor:
        """
        Decode the node embeddings to predict edge values.

        Args:
            z (dict[str, Tensor]): Node embeddings for each node type.
            edge_index (EdgeIndex): Edge label indices.

        Returns:
            Tensor: Predicted edge values.
        """
        z_left = z_dict['tx'][edge_index[0]]
        z_right = z_dict['bd'][edge_index[1]]
        return (z_left * z_right).sum(dim=-1)
        # return self.edge_predictor(torch.cat([z_left, z_right], dim=-1)).squeeze()
