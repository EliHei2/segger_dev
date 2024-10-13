import unittest
import torch
from segger.models.segger_model import Segger
from torch_geometric.nn import to_hetero
from torch_geometric.data import HeteroData


class TestSeggerModel(unittest.TestCase):

    def setUp(self):
        model = Segger(init_emb=16, hidden_channels=32, out_channels=32, heads=3)
        metadata = (["tx", "nc"], [("tx", "belongs", "nc"), ("tx", "neighbors", "tx")])
        self.model = to_hetero(model, metadata=metadata, aggr="sum")
        self.data = HeteroData()
        self.data["tx"].x = torch.randn(10, 16)
        self.data["nc"].x = torch.randn(5, 16)
        self.data["tx", "belongs", "nc"].edge_index = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
        self.data["tx", "neighbors", "tx"].edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

    def test_forward(self):
        out = self.model(self.data.x_dict, self.data.edge_index_dict)
        self.assertTrue("tx" in out)
        self.assertTrue("nc" in out)
        self.assertEqual(out["tx"].shape[1], 32 * 3)
        self.assertEqual(out["nc"].shape[1], 32 * 3)

    """
    def test_decode(self):
        z = {'tx': torch.randn(10, 16), 'nc': torch.randn(5, 16)}
        edge_label_index = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
        out = self.model.decode(z, edge_label_index)
        self.assertEqual(out.shape[0], 3)
    """


if __name__ == "__main__":
    unittest.main()
