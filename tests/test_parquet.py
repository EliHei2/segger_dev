import unittest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
import geopandas as gpd
import shapely
from segger.data.parquet.sample import STSampleParquet
from segger.data.parquet._utils import get_polygons_from_xy, compute_nuclear_transcripts


class TestParquetData(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        self.test_dir = Path(self.temp_dir)

        # Create sample transcripts data
        self.transcripts_data = {
            "x_global_px": [100, 200, 300, 400],
            "y_global_px": [100, 200, 300, 400],
            "feature_name": ["gene1", "gene2", "gene3", "gene4"],
            "cell": [1, 1, 2, 2],
        }
        self.transcripts_df = pd.DataFrame(self.transcripts_data)
        self.transcripts_df.to_parquet(self.test_dir / "transcripts.parquet")

        # Create sample boundaries data
        self.boundaries_data = {
            "x_global_px": [100, 200, 300, 400, 100, 200, 300, 400],
            "y_global_px": [100, 100, 100, 100, 200, 200, 200, 200],
            "cell": [1, 1, 1, 1, 2, 2, 2, 2],
        }
        self.boundaries_df = pd.DataFrame(self.boundaries_data)
        self.boundaries_df.to_parquet(self.test_dir / "nucleus_boundaries.parquet")

    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.temp_dir)

    def test_transcript_id_generation(self):
        """Test automatic generation of transcript IDs based on coordinates"""
        sample = STSampleParquet(
            base_dir=self.test_dir, sample_type="cosmx", buffer_ratio=1.0
        )

        # Get the first tile's data
        tiles = sample._get_balanced_regions()
        xm = sample._get_in_memory_dataset(tiles[0])

        # Verify that transcript IDs are generated and unique
        tx_ids = xm.transcripts["transcript_id"].values
        self.assertEqual(len(tx_ids), len(np.unique(tx_ids)))

        # Verify IDs are consistent for same coordinates
        coords = xm.transcripts[["x_global_px", "y_global_px"]].values
        expected_ids = [hash(f"{x}_{y}") % (2**32) for x, y in coords]
        np.testing.assert_array_equal(tx_ids, expected_ids)

    def test_nuclear_overlap_computation(self):
        """Test computation of nuclear overlap when not pre-computed"""
        sample = STSampleParquet(
            base_dir=self.test_dir, sample_type="cosmx", buffer_ratio=1.0
        )

        # Get the first tile's data
        tiles = sample._get_balanced_regions()
        xm = sample._get_in_memory_dataset(tiles[0])

        # Verify nuclear overlap computation
        polygons = get_polygons_from_xy(
            xm.boundaries, "x_global_px", "y_global_px", "cell"
        )

        computed_overlap = compute_nuclear_transcripts(
            polygons, xm.transcripts, "x_global_px", "y_global_px"
        )

        # Verify that overlap computation is consistent
        self.assertEqual(len(computed_overlap), len(xm.transcripts))
        self.assertTrue(isinstance(computed_overlap, np.ndarray))
        self.assertTrue(
            np.all(np.logical_or(computed_overlap == 0, computed_overlap == 1))
        )

    def test_boundary_buffering(self):
        """Test boundary buffering functionality"""
        sample = STSampleParquet(
            base_dir=self.test_dir, sample_type="cosmx", buffer_ratio=1.2
        )

        # Get the first tile's data
        tiles = sample._get_balanced_regions()
        xm = sample._get_in_memory_dataset(tiles[0])

        # Get original and buffered polygons
        original_polygons = get_polygons_from_xy(
            xm.boundaries, "x_global_px", "y_global_px", "cell", buffer_ratio=1.0
        )

        buffered_polygons = get_polygons_from_xy(
            xm.boundaries, "x_global_px", "y_global_px", "cell", buffer_ratio=1.2
        )

        # Verify that buffered polygons are larger
        self.assertTrue(np.all(buffered_polygons.area > original_polygons.area))

        # Verify buffer distance calculation
        areas = original_polygons.area
        expected_buffer_distances = np.sqrt(areas / np.pi) * 0.2
        actual_buffer_distances = np.sqrt(buffered_polygons.area / np.pi) - np.sqrt(
            areas / np.pi
        )
        np.testing.assert_allclose(
            actual_buffer_distances, expected_buffer_distances, rtol=0.1
        )

    def test_missing_qv_handling(self):
        """Test handling of missing quality value field"""
        # Create sample without QV field
        transcripts_no_qv = self.transcripts_df.copy()
        transcripts_no_qv.to_parquet(self.test_dir / "transcripts_no_qv.parquet")

        sample = STSampleParquet(
            base_dir=self.test_dir, sample_type="cosmx", buffer_ratio=1.0
        )

        # Verify that filtering works without QV field
        tiles = sample._get_balanced_regions()
        xm = sample._get_in_memory_dataset(tiles[0])

        # Should not raise any errors and process all transcripts
        self.assertEqual(len(xm.transcripts), len(self.transcripts_df))

    def test_save_debug(self):
        """Test debug save functionality"""
        sample = STSampleParquet(
            base_dir=self.test_dir, sample_type="cosmx", buffer_ratio=1.0
        )

        # Create output directory
        output_dir = self.test_dir / "output"
        output_dir.mkdir(exist_ok=True)

        # Test debug save
        sample.save_debug(
            data_dir=output_dir,
            k_bd=3,
            dist_bd=15.0,
            k_tx=3,
            dist_tx=5.0,
            tile_width=100,
            tile_height=100,
            neg_sampling_ratio=5.0,
            frac=1.0,
            val_prob=0.1,
            test_prob=0.2,
        )

        # Verify output structure
        self.assertTrue((output_dir / "train_tiles" / "processed").exists())
        self.assertTrue((output_dir / "test_tiles" / "processed").exists())
        self.assertTrue((output_dir / "val_tiles" / "processed").exists())

        # Verify that files were created
        train_files = list((output_dir / "train_tiles" / "processed").glob("*.pt"))
        test_files = list((output_dir / "test_tiles" / "processed").glob("*.pt"))
        val_files = list((output_dir / "val_tiles" / "processed").glob("*.pt"))

        self.assertTrue(len(train_files) > 0)
        self.assertTrue(len(test_files) > 0)
        self.assertTrue(len(val_files) > 0)


if __name__ == "__main__":
    unittest.main()
