import pytest
import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, to_hetero
from torch_geometric.data import HeteroData

from segger.models.segger_model import SkipGAT


class GATEncoder(nn.Module):
    """
    A simple wrapper around GATv2Conv for use with to_hetero().

    This module defines a single GATv2Conv layer intended to be used with
    PyTorch Geometric's `to_hetero` utility to convert it into a heterogeneous
    graph model. It is intended to be compared to our custom SkipGAT, which
    should match its behavior.

    Parameters
    ----------
    in_channels : int
        Input feature dimensionality.
    out_channels : int
        Output feature dimensionality per attention head.
    heads : int
        Number of attention heads.
    """
    def __init__(self, in_channels, out_channels, heads):
        super().__init__()
        self.conv = GATv2Conv(
            in_channels,
            out_channels,
            heads=heads,
            add_self_loops=False
        )

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


def match_weights(model1, model2):
    with torch.no_grad():
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            p2.copy_(p1)


def generate_heterodata(
    tx_size: int,
    bd_size: int,
    num_edges_tx: int,
    num_edges_bd: int,
    feat_dim: int,
):
    """
    Creates a synthetic heterogeneous graph with 'tx' and 'bd' node types and
    two edge types: ('tx', 'neighbors', 'tx') and ('tx', 'belongs', 'bd').

    Duplicate edges are removed automatically.

    Parameters
    ----------
    tx_size : int
        Number of 'tx' nodes.
    bd_size : int
        Number of 'bd' nodes.
    num_edges_tx : int
        Number of edges for the 'neighbors' edge type.
    num_edges_bd : int
        Number of edges for the 'belongs' edge type.
    feat_dim : int
        Feature dimensionality for all node types.

    Returns
    -------
    data : HeteroData
        A heterogeneous graph ready for input to a hetero GNN model.
    """
    data = HeteroData()

    data['tx'].x = torch.randn(tx_size, feat_dim)
    data['bd'].x = torch.randn(bd_size, feat_dim)

    tx_edges = set()
    while len(tx_edges) < num_edges_tx:
        src = torch.randint(0, tx_size, (1,)).item()
        dst = torch.randint(0, tx_size, (1,)).item()
        tx_edges.add((src, dst))
    tx_edge_index = torch.tensor(list(tx_edges)).t()

    bd_edges = set()
    while len(bd_edges) < num_edges_bd:
        src = torch.randint(0, tx_size, (1,)).item()
        dst = torch.randint(0, bd_size, (1,)).item()
        bd_edges.add((src, dst))
    bd_edge_index = torch.tensor(list(bd_edges)).t()

    data['tx', 'neighbors', 'tx'].edge_index = tx_edge_index
    data['tx', 'belongs', 'bd'].edge_index = bd_edge_index

    return data


# ----------------------------------------------------------------------
# Test 1: Fixed dataset, varying architectures
# ----------------------------------------------------------------------

@pytest.fixture
def fixed_heterodata():
    return generate_heterodata(
        tx_size=5,
        bd_size=4,
        num_edges_tx=3,
        num_edges_bd=3,
        feat_dim=8
    )


@pytest.mark.parametrize("in_channels,out_channels,heads", [
    (4, 4, 1),
    (8, 4, 2),
    (8, 16, 3),
])
def test_skipgat_architectures_match(
    fixed_heterodata: HeteroData,
    in_channels: int,
    out_channels: int,
    heads: int,
):
    torch.manual_seed(42)

    fixed_heterodata['tx'].x = torch.randn(
        fixed_heterodata['tx'].num_nodes,
        in_channels
    )
    fixed_heterodata['bd'].x = torch.randn(
        fixed_heterodata['bd'].num_nodes,
        in_channels
    )

    base_model = GATEncoder(in_channels, out_channels, heads)
    hetero_model = to_hetero(
        base_model,
        fixed_heterodata.metadata(),
        aggr='sum'
    )
    skipgat_model = SkipGAT(in_channels, out_channels, heads, 
                            add_self_loops_tx=False)

    for edge_type in fixed_heterodata.edge_types:
        hetero_key = '__'.join(edge_type)
        match_weights(
            hetero_model.conv[hetero_key],
            skipgat_model.conv.convs[edge_type]
        )

    out_hetero = hetero_model(
        fixed_heterodata.x_dict,
        fixed_heterodata.edge_index_dict
    )
    out_skipgat = skipgat_model(
        fixed_heterodata.x_dict,
        fixed_heterodata.edge_index_dict
    )

    for node_type in fixed_heterodata.node_types:
        assert out_skipgat[node_type].shape == \
            out_hetero[node_type].shape
        assert torch.allclose(
            out_skipgat[node_type],
            out_hetero[node_type],
            atol=1e-6
        )


# ----------------------------------------------------------------------
# Test 2: Fixed architecture, varying datasets
# ----------------------------------------------------------------------

@pytest.mark.parametrize("tx_size,bd_size,num_edges_tx,num_edges_bd", [
    (3, 2, 2, 2),
    (10, 5, 5, 4),
    (15, 10, 20, 15),
])
def test_skipgat_datasets_match(
    tx_size: int,
    bd_size: int,
    num_edges_tx: int,
    num_edges_bd: int,
):
    torch.manual_seed(42)

    in_channels = 8
    out_channels = 4
    heads = 2

    data = generate_heterodata(
        tx_size,
        bd_size,
        num_edges_tx,
        num_edges_bd,
        feat_dim=in_channels
    )

    base_model = GATEncoder(in_channels, out_channels, heads)
    hetero_model = to_hetero(base_model, data.metadata(), aggr='sum')
    skipgat_model = SkipGAT(in_channels, out_channels, heads, 
                            add_self_loops_tx=False)

    for edge_type in data.edge_types:
        hetero_key = '__'.join(edge_type)
        match_weights(
            hetero_model.conv[hetero_key],
            skipgat_model.conv.convs[edge_type]
        )

    out_hetero = hetero_model(data.x_dict, data.edge_index_dict)
    out_skipgat = skipgat_model(data.x_dict, data.edge_index_dict)

    for node_type in data.node_types:
        assert out_skipgat[node_type].shape == \
            out_hetero[node_type].shape
        assert torch.allclose(
            out_skipgat[node_type],
            out_hetero[node_type],
            atol=1e-6
        )
