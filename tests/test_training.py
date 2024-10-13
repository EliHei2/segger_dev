import unittest
from segger.training.train import LitSegger
from segger.models.segger_model import Segger
from torch_geometric.data import HeteroData
import torch


class TestTraining(unittest.TestCase):

    def setUp(self):

        # Setup model and data
        metadata = (["tx", "nc"], [("tx", "belongs", "nc"), ("tx", "neighbors", "tx")])
        self.lit_segger = LitSegger(
            init_emb=16,
            hidden_channels=32,
            out_channels=32,
            heads=3,
            metadata=metadata,
            aggr="sum",
        )
        self.data = HeteroData()
        self.data["tx"].x = torch.randn(10, 16)
        self.data["nc"].x = torch.randn(5, 16)
        self.data["tx", "belongs", "nc"].edge_label_index = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
        self.data["tx", "belongs", "nc"].edge_label = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float)
        self.data["tx", "neighbors", "tx"].edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

        # Move model and data to GPU
        self.lit_segger.cuda()
        self.data.to("cuda")

    def test_training_step(self):
        optimizer = self.lit_segger.configure_optimizers()
        self.lit_segger.train()
        optimizer.zero_grad()
        loss = self.lit_segger.training_step(self.data, batch_idx=0)
        loss.backward()
        optimizer.step()
        self.assertGreater(loss.item(), 0)


if __name__ == "__main__":
    unittest.main()
