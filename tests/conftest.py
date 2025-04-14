import pytest
from pathlib import Path
from lightning.pytorch.plugins.environments import SLURMEnvironment

# Remove SLURM environment autodetect for all tests
SLURMEnvironment.detect = lambda: False


@pytest.fixture(scope="session")
def xenium_human_lung_10k() -> Path:
    """Path to the test dataset directory."""
    sample_name = "10x_preview_non-diseased_human_lung_10k_transcripts"
    return Path(__file__).parent / "data" / sample_name
