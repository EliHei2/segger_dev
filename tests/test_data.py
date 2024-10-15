import unittest
import pandas as pd
import unittest
import torch
from torch_geometric.data import Data
from segger.data.utils import *
from segger.data import XeniumSample
import unittest
import pandas as pd


class TestDataUtils(unittest.TestCase):

    def test_filter_transcripts(self):
        data = {"qv": [30, 10, 25], "feature_name": ["gene1", "NegControlProbe_gene2", "gene3"]}
        df = pd.DataFrame(data)
        filtered_df = filter_transcripts(df, min_qv=20)
        self.assertEqual(len(filtered_df), 2)
        self.assertTrue("gene1" in filtered_df["feature_name"].values)
        self.assertTrue("gene3" in filtered_df["feature_name"].values)

    def test_compute_transcript_metrics(self):
        data = {
            "qv": [40, 40, 25, 25],
            "feature_name": ["gene1", "gene2", "gene1", "gene2"],
            "cell_id": [1, 1, -1, 2],
            "overlaps_nucleus": [1, 0, 0, 1],
        }
        df = pd.DataFrame(data)
        metrics = compute_transcript_metrics(df, qv_threshold=30)
        self.assertAlmostEqual(metrics["percent_assigned"], 50.0)
        self.assertAlmostEqual(metrics["percent_cytoplasmic"], 50.0)
        self.assertAlmostEqual(metrics["percent_nucleus"], 50.0)
        self.assertAlmostEqual(metrics["percent_non_assigned_cytoplasmic"], 100.0)
        self.assertEqual(len(metrics["gene_metrics"]), 2)
        self.assertTrue("gene1" in metrics["gene_metrics"]["feature_name"].values)
        self.assertTrue("gene2" in metrics["gene_metrics"]["feature_name"].values)

    def setUp(self):
        data = {
            "x_location": [100, 200, 300],
            "y_location": [100, 200, 300],
            "z_location": [0, 0, 0],
            "qv": [40, 40, 25],
            "feature_name": ["gene1", "gene2", "gene3"],
            "transcript_id": [1, 2, 3],
            "overlaps_nucleus": [1, 0, 1],
            "cell_id": [1, -1, 2],
        }
        self.df = pd.DataFrame(data)
        self.sample = XeniumSample(self.df)

    def test_crop_transcripts(self):
        cropped_sample = self.sample.crop_transcripts(50, 50, 200, 200)
        self.assertEqual(len(cropped_sample.transcripts_df), 1)
        self.assertEqual(cropped_sample.transcripts_df.iloc[0]["feature_name"], "gene1")

    def test_filter_transcripts(self):
        filtered_df = XeniumSample.filter_transcripts(self.df, min_qv=30)
        self.assertEqual(len(filtered_df), 2)
        self.assertTrue("gene1" in filtered_df["feature_name"].values)
        self.assertTrue("gene2" in filtered_df["feature_name"].values)

    def test_unassign_all_except_nucleus(self):
        unassigned_df = XeniumSample.unassign_all_except_nucleus(self.df)
        self.assertEqual(unassigned_df.loc[unassigned_df["overlaps_nucleus"] == 0, "cell_id"].values[0], "UNASSIGNED")


if __name__ == "__main__":
    unittest.main()
