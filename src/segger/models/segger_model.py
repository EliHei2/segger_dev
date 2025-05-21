import torch
from torch_geometric.nn import GATv2Conv, Linear, HeteroDictLinear, HeteroConv
from torch.nn import (
    Embedding,
    ModuleDict,
    ModuleList,
    Module,
    functional as F
)
from torch import Tensor
from typing import Dict, Tuple, List, Union, Optional


class SkipGAT(Module):
    """
    Graph Attention module that encapsulates a HeteroConv layer with two GATv2
    convolutions for different edge types. The attention weights from the last
    forward pass are stored internally and can be accessed via the
    `attention_weights` property.

    Parameters
    ----------
    in_channels : int
        Number of input features.
    out_channels : int
        Number of output features.
    heads : int
        Number of attention heads.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_heads: int,
        add_self_loops_tx: bool = True,
    ) -> None:
        super().__init__()

        # Build a HeteroConv that internally uses GATv2Conv for each edge type.
        self.conv = HeteroConv(
            convs={
                ('tx', 'neighbors', 'tx'): GATv2Conv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    heads=n_heads,
                    add_self_loops=add_self_loops_tx,
                ),
                ('tx', 'belongs', 'bd'): GATv2Conv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    heads=n_heads,
                    add_self_loops=False,
                ),
            },
            aggr='sum'
        )

        # This will store the attention weights from the last forward pass.
        self._attn_weights: Dict[Tuple[str, str, str], Tensor] = {}

        # Register a forward hook to capture attention weights internally.
        edge_type = 'tx', 'neighbors', 'tx'
        self.conv.convs[edge_type].register_forward_hook(
            self._make_hook(edge_type),
            with_kwargs=True,
        )

    def _make_hook(self, edge_type: Tuple[str, str, str]):
        """
        Internal hook function that captures attention weights from the
        forward pass of each GATv2Conv submodule.

        Parameters
        ----------
        edge_type : tuple of str
            The edge type associated with this GATv2Conv.
        """
        def _store_attn_weights(module, inputs, kwargs, outputs) -> None:
            self._attn_weights[edge_type] = outputs[1][1]
        return _store_attn_weights

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        """
        Forward pass for SkipGAT. Always calls HeteroConv with
        `return_attention_weights=True`, but never returns them from
        this method. Attention weights are stored internally via the hook.

        Parameters
        ----------
        x_dict : dict of str -> Tensor
            Node features for each node type.
        edge_index_dict : dict of str -> Tensor
            Edge indices for each edge type.

        Returns
        -------
        x_dict_out : dict of str -> Tensor
            Updated node embeddings after convolution.
        """
        # Always request attention weights, but do not return them here.
        x_dict = self.conv(
            x_dict,
            edge_index_dict,
            return_attention_weights_dict = {'tx': True, 'bd': False},
        )
        x_dict['tx'] = x_dict['tx'][0]
        return x_dict

    @property
    def attention_weights(self) -> Dict[Tuple[str, str, str], Tensor]:
        """
        The attention weights from the most recent forward pass.

        Raises
        ------
        RuntimeError
            If no forward pass has been performed yet.

        Returns
        -------
        dict of (str, str, str) -> Tensor
            Mapping each edge type to its attention weight tensor of shape
            [num_edges, num_heads].
        """
        if not self._attn_weights:
            msg = "Attention weights are empty. Please perform a forward pass."
            raise AttributeError(msg)
        return self._attn_weights


class Segger(torch.nn.Module):
    """
    TODO: Add description.
    """

    def __init__(
        self,
        n_genes: int,
        in_channels: int = 16,
        hidden_channels: int = 32,
        out_channels: int = 32,
        n_mid_layers: int = 3,
        heads: int = 3,
        embedding_weights: Optional[Tensor] = None,
    ):
        """
        Initialize the Segger model.

        Parameters
        ----------
        n_genes : int
            Number of unique genes for embedding.
        in_channels : int, optional
            Initial embedding size for both 'tx' and boundary nodes.
            Default is 16.
        hidden_channels : int, optional
            Number of hidden channels. Default is 32.
        out_channels : int, optional
            Number of output channels. Default is 32.
        n_mid_layers : int, optional
            Number of hidden layers (excluding first and last layers).
            Default is 3.
        heads : int, optional
            Number of attention heads. Default is 3.
        embedding_weights : Tensor, optional
            Pretrained embedding weights for genes. If None, weights are 
            initialized randomly. Default is None.
        """
        super().__init__()
        # Store hyperparameters for PyTorch Lightning
        self.hparams = locals()
        for k in ['self', '__class__']: 
            self.hparams.pop(k)
        # First layer: ? -> in
        self.lin_first = ModuleDict(
            {
                'tx': Embedding(
                    n_genes,
                    in_channels,
                    _weight=embedding_weights,
                ),
                'bd': Linear(-1, in_channels),
            }
        )
        self.conv_layers = ModuleList()
        # First convolution: in -> hidden x heads
        self.conv_layers.append(
            SkipGAT((-1, -1), hidden_channels, heads)
        )
        # Middle convolutions: hidden x heads -> hidden x heads
        for _ in range(n_mid_layers):
            self.conv_layers.append(
                SkipGAT((-1, -1), hidden_channels, heads)
            )
        # Last convolution: hidden x heads -> out x heads
        self.conv_layers.append(
            SkipGAT((-1, -1), out_channels, heads)
        )
        # Last layer: out x heads -> out
        self.lin_last = HeteroDictLinear(
            -1,
            out_channels,
            types=("tx", "bd")
        )

    def forward(
        self,
        x_dict: dict[str, Tensor],
        edge_index_dict: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """
        Forward pass for the Segger model.

        Parameters
        ----------
        x_dict : dict[str, Tensor]
            Node features for each node type.
        edge_index_dict : dict[str, Tensor]
            Edge indices for each edge type.

        Returns
        -------
        Tensor
            Output node features after passing through the Segger model.
        """
        # Linearly project embedding to input dim
        x_dict = {k: self.lin_first[k](x) for k, x in x_dict.items()}

        # ReLu for some reason
        x_dict = {k: F.leaky_relu(x) for k, x in x_dict.items()}

        # Graph convolutions with GATv2
        for conv_layer in self.conv_layers[:-1]:
            x_dict = conv_layer(x_dict, edge_index_dict)
            x_dict = {k: F.leaky_relu(x) for k, x in x_dict.items()}

        x_dict = self.conv_layers[-1](x_dict, edge_index_dict)
        # Linearly project to output dim
        # x_dict = self.lin_last(x_dict)

        return x_dict

    def decode(
        self,
        z: dict[str, Tensor],
        edge_index: Union[Tensor],
    ) -> Tensor:
        """
        Decode the node embeddings to predict edge values.

        Parameters
        ----------
        z : dict[str, Tensor]
            Node embeddings for each node type.
        edge_index : EdgeIndex
            Edge label indices.

        Returns
        -------
        Tensor
            Predicted edge values.
        """
        z_tx = z["tx"][edge_index[0]]
        z_bd = z["bd"][edge_index[1]]

        return (z_tx * z_bd).sum(dim=-1)

    def get_attention_weights(self, edge_type: Tuple[str]) -> Tensor:
        """
        Return a stacked tensor of attention weights for the given edge type
        from each SkipGAT layer, raising an error if the edge type is missing
        in any layer.

        Parameters
        ----------
        edge_type : str
            The edge type key, e.g. ('tx','neighbors','tx')

        Returns
        -------
        Tensor
            A 3D tensor of shape [num_layers, num_edges, num_heads].
        """
        try:
            attention_weights = []
            for i, layer in enumerate(self.conv_layers):
                attention_weights.append(layer.attention_weights[edge_type])
        except KeyError as e:
            msg = f"Edge type '{edge_type}' not found in layer {i}."
            raise KeyError(msg) from e

        return torch.stack(attention_weights)
