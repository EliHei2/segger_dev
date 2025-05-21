import os

from ..data.universal._enum import FEATURE_INDEX_FILE
from .data_module import SeggerDataModule
from ..config.config import SeggerConfig
from .lightning_model import LitSegger


def datamodule_from_config(config_path: os.PathLike) -> SeggerDataModule:
    """
    Utility function to create a SeggerDataModule from a configuration file.

    Parameters
    ----------
    config : SeggerConfig
        A filepath to a YAML segger configuration.

    Returns
    -------
    LitSegger
        An instance of the SeggerDataModule.
    """
    config = SeggerConfig.from_yaml(config_path)

    if config.train.ckpt_path:
        return SeggerDataModule.load_from_checkpoint(config.train.ckpt_path)

    return SeggerDataModule(
        data_dir=config.data.save_dir,
        batch_size=config.train.batch_size,
        num_workers=config.train.n_workers,
        negative_sampling_ratio=config.train.neg_edge_ratio,
        k_tx=config.train.max_transcripts_k,
        dist_tx=config.train.max_transcripts_dist,
    )


def model_from_config(config_path: os.PathLike) -> LitSegger:
    """
    Utility function to create a LitSegger model from a configuration file.

    Parameters
    ----------
    config : SeggerConfig
        A filepath to a YAML segger configuration.

    Returns
    -------
    LitSegger
        An instance of the LitSegger model.
    """
    config = SeggerConfig.from_yaml(config_path)

    if config.train.ckpt_path:
        return LitSegger.load_from_checkpoint(config.train.ckpt_path)

    return LitSegger(
        gene_embedding_indices=config.data.save_dir / FEATURE_INDEX_FILE,
        gene_embedding_weights=config.train.gene_emb_path,
        in_channels=config.train.in_channels,
        hidden_channels=config.train.hidden_channels,
        out_channels=config.train.out_channels,
        num_mid_layers=config.train.n_mid_layers,
        heads=config.train.n_heads,
        learning_rate=config.train.learning_rate,
    )
