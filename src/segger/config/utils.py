from pathlib import Path
import os

from ..data.universal._enum import FEATURE_INDEX_FILE
from ..training import LitSegger, SeggerDataModule
from ..data.universal.ist_sample import ISTSample
from ..prediction.predict import predict
from .config import SeggerConfig


def ist_sample_from_config(config: SeggerConfig) -> ISTSample:
    """
    Create an ISTSample instance from a SeggerConfig object.

    Parameters
    ----------
    config : SeggerConfig
        Parsed Pydantic configuration model.

    Returns
    -------
    ISTSample
        Initialized ISTSample object ready for saving.
    """
    data = config.data
    return ISTSample(
        save_dir=data.save_dir,
        tx_path=data.tx_path,
        bd_path=data.bd_path,
        tx_feature_name=data.tx_feature_name,
        tx_cell_id=data.tx_cell_id,
        tx_x=data.tx_x,
        tx_y=data.tx_y,
        tx_k=data.tx_k,
        tx_dist=data.tx_dist,
        bd_k=data.bd_k,
        bd_dist=data.bd_dist,
        max_cells_per_tile=data.max_cells_per_tile,
        tile_margin=data.tile_margin,
        frac_train=data.frac_train,
        frac_test=data.frac_test,
        frac_val=data.frac_val,
        n_workers=data.n_workers,
    )


def data_module_from_config(config: SeggerConfig) -> SeggerDataModule:
    """
    Create a SeggerDataModule instance from a SeggerConfig object.

    Parameters
    ----------
    config : SeggerConfig
        Parsed Pydantic configuration model.

    Returns
    -------
    SeggerDataModule
        Initialized Lightning DataModule for loading tile data.
    """
    train = config.train
    if train.checkpoint_path:
        return SeggerDataModule.load_from_checkpoint(train.checkpoint_path)
    return SeggerDataModule(
        data_dir=config.data.save_dir,
        batch_size=train.batch_size,
        n_workers=train.n_workers,
        k_tx_max=train.k_tx_max,
        dist_tx_max=train.dist_tx_max,
        negative_edge_sampling_ratio=train.negative_edge_sampling_ratio,
    )


def model_from_config(config: SeggerConfig) -> LitSegger:
    """
    Create a LitSegger model from a SeggerConfig object.

    Parameters
    ----------
    config : SeggerConfig
        Parsed Pydantic configuration model.

    Returns
    -------
    LitSegger
        Initialized Lightning module for training.
    """
    train = config.train
    if train.checkpoint_path:
        return LitSegger.load_from_checkpoint(train.checkpoint_path)
    return LitSegger(
        gene_embedding_indices=config.data.save_dir / FEATURE_INDEX_FILE,
        gene_embedding_weights=train.gene_embedding_weights,
        in_channels=train.in_channels,
        hidden_channels=train.hidden_channels,
        out_channels=train.out_channels,
        n_mid_layers=train.n_mid_layers,
        n_heads=train.n_heads,
        learning_rate=train.learning_rate,
    )


def get_last_checkpoint(save_dir: Path) -> Path:
    """
    Returns the most recent checkpoint from the training save directory or 
    raises error if no checkpoint found.
    """
    checkpoints = sorted(
        save_dir.glob("*/*/checkpoints/*.ckpt"),
        key=os.path.getmtime,
        reverse=True,
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {save_dir}")
    return checkpoints[0]


def predict_from_config(
    config: SeggerConfig,
    use_cc: bool = False,
    show_pbar: bool = True,
):
    """
    Load model and data module from checkpoint and config, then run prediction.

    Parameters
    ----------
    config : SeggerConfig
        Full config containing model, data, and predict settings.
    use_cc : bool, default=False
        Use connected components to resolve unassigned transcripts.
    show_pbar : bool, default=False
        Show progress bar during prediction.

    Returns
    -------
    pd.DataFrame
        Segmentation predictions for all transcripts.
    """
    checkpoint_path = get_last_checkpoint(config.train.save_dir)
    data_module = SeggerDataModule.load_from_checkpoint(checkpoint_path)
    model = LitSegger.load_from_checkpoint(checkpoint_path)

    return predict(
        save_dir=config.predict.save_dir,
        model=model,
        data_module=data_module,
        min_score=config.predict.min_score,
        receptive_field_k=config.predict.receptive_field_k,
        receptive_field_dist=config.predict.receptive_field_dist,
        use_cc=use_cc,
        show_pbar=show_pbar,
    )
