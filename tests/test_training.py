import unittest
from segger.training.train import LitSegger, train_model
from segger.models.segger_model import Segger
from torch_geometric.data import HeteroData
import torch

class TestTraining(unittest.TestCase):

    def setUp(self):
        self.model = Segger(init_emb=16, hidden_channels=32, out_channels=32, heads=3)
        self.lit_model = LitSegger(self.model)
        self.data = HeteroData()
        self.data['tx'].x = torch.randn(10, 16)
        self.data['nc'].x = torch.randn(5, 16)
        self.data['tx', 'belongs', 'nc'].edge_label_index = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
        self.data['tx', 'belongs', 'nc'].edge_label = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float)
        self.data['tx', 'neighbors', 'tx'].edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

    def test_training_step(self):
        optimizer = self.lit_model.configure_optimizers()
        self.lit_model.train()
        self.data.to('cuda')
        optimizer.zero_grad()
        loss = self.lit_model.training_step(self.data, batch_idx=0)
        loss.backward()
        optimizer.step()
        self.assertGreater(loss.item(), 0)

if __name__ == '__main__':
    unittest.main()
