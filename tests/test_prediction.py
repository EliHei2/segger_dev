import unittest
import torch
from segger.prediction.predict import load_model, predict
from segger.models.segger_model import Segger
from torch_geometric.data import HeteroData


class TestPrediction(unittest.TestCase):

    def setUp(self):
        self.model = Segger(init_emb=16, hidden_channels=32, out_channels=32, heads=3)
        self.lit_model = load_model("path/to/checkpoint", 16, 32, 32, 3, "sum")
        self.data = HeteroData()
        self.data["tx"].x = torch.randn(10, 16)
        self.data["nc"].x = torch.randn(5, 16)
        self.data["tx", "belongs", "nc"].edge_label_index = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
        self.data["tx", "neighbors", "tx"].edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

    def test_predict(self):
        output_path = "path/to/output.csv.gz"
        predict(self.lit_model, "path/to/dataset", output_path, 0.5, 4, 20, 5, 10)
        self.assertTrue(os.path.exists(output_path))


if __name__ == "__main__":
    unittest.main()
