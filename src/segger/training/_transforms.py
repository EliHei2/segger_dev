import torch
from torch_geometric.data import HeteroData
from torch_geometric.edge_index import EdgeIndex
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import negative_sampling
from typing import Optional, Tuple

class MaskEdgeIndex(BaseTransform):
    #TODO: Add documentation

    def __init__(
        self,
        edge_type: Tuple[str],
        k: Optional[int] = None,
        dist: Optional[float] = None,
    ):
        #TODO: Add documentation
        super().__init__()
        self.edge_type = edge_type
        self.k = k
        self.dist = dist

    def forward(self, data: HeteroData):
        #TODO: Add documentation
        # Input checks
        max_k = data[self.edge_type].k
        if (self.k is None or self.k >= max_k) and self.dist is None:
            return data
        if self.edge_type not in data.edge_types:
            msg = (
                f"Edge type {self.edge_type} not found in HeteroData. Valid "
                f"edge types include: {', '.join(map(str, data.edge_types))}."
            )
            raise KeyError(msg)

        # Mask edge index and distances
        edge_index = data[self.edge_type].edge_index
        edge_dists = data[self.edge_type].edge_attr
        ni, _ = edge_index.sparse_size()
        nj = max_k
        mask = torch.ones(ni * nj).bool()
        
        if self.k is not None:
            col_mask = torch.concat([
                torch.ones(self.k),
                torch.zeros(nj - self.k)
            ])
            mask &= col_mask.repeat(ni).bool()
            nj = self.k
        if self.dist is not None:
            mask &= edge_dists <= self.dist

        edge_index = EdgeIndex(
            edge_index[:, mask],
            sort_order='row',
            sparse_size=(ni, ni),
        )
        edge_dists = edge_dists[mask]

        # Update data in place
        data[self.edge_type].edge_index = edge_index
        data[self.edge_type].edge_attr = edge_dists
        data[self.edge_type].k = nj

        return data


class NegativeSampling(BaseTransform):
    #TODO: Add documentation
    def __init__(
        self,
        edge_type: Tuple[str],
        sampling_ratio: float,
        pos_index: str = 'edge_index',
        neg_index: str = 'neg_edge_index',
    ):
        #TODO: Add documentation
        super().__init__()
        self.edge_type = edge_type
        self.pos_index = pos_index
        self.neg_index = neg_index
        self.sampling_ratio = sampling_ratio

    def forward(self, data):
        # Return early if no positive edges
        pos_idx = data[self.edge_type][self.pos_index]
        if pos_idx.size(1) == 0:
            data[self.edge_type][self.neg_index] = pos_idx.clone()
            return data
        # Construct negative index with mapped transcript indices
        val, key = torch.unique(pos_idx[0], return_inverse=True)
        pos_idx[0] = key
        neg_idx = negative_sampling(
            pos_idx,
            pos_idx.max(1).values + 1,
            num_neg_samples=int(pos_idx.shape[1] * self.sampling_ratio),
        )
        # Reset transcript indices
        pos_idx[0] = val
        neg_idx[0] = val[neg_idx[0]]
        data[self.edge_type][self.neg_index] = neg_idx

        return data
