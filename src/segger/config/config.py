from pydantic.types import (
    PositiveInt, PositiveFloat, FilePath, DirectoryPath, AnyType
)
from pydantic import (
    BaseModel, Field, model_validator, field_validator, AfterValidator
)
from pandas.api.types import is_string_dtype, is_float_dtype
from pyarrow import parquet as pq
from typing import List, Optional
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import os

def _check_extension(path: os.PathLike, allowed_exts: set[str]):
    """
    Validate that the file at `path` has an allowed extension.

    Parameters
    ----------
    path : os.PathLike
        Path to the file being validated.
    allowed_exts : set of str
        Allowed file extensions (e.g., {".parquet"}).

    Raises
    ------
    ValueError
        If the file extension is not in the allowed set.
    """
    if Path(path).suffix.lower() not in allowed_exts:
        msg = f"File must have one of the extensions: {allowed_exts}"
        raise ValueError(msg)

def _check_field_names(path: Path, names: List[str]):
    """
    Ensure all specified field names exist in a Parquet file.

    Parameters
    ----------
    path : Path
        Path to the Parquet file.
    names : list of str
        Field names that must be present in the file schema.

    Raises
    ------
    ValueError
        If any expected field is missing from the file schema.
    """
    schema = pq.read_schema(path)
    missing = set(names) - set(schema.names)
    if len(missing) > 0:
        raise ValueError(
            f"Fields {', '.join(missing)} not in '{path.stem}.parquet' "
            f"columns: {schema.names}."
        )

class DataConfig(BaseModel):
    """
    Configuration schema for dataset creation and validation.
    See "segger/config/_configs/template.yaml" for more detail.
    """
    tx_path: FilePath = Field(validation_alias="transcripts_parquet")
    bd_path: FilePath = Field(validation_alias="boundaries_parquet")
    tx_feature_name: str = Field(validation_alias="transcripts_feature_name")
    tx_cell_id: str = Field(validation_alias="transcripts_cell_id")
    tx_x: str = Field(validation_alias="transcripts_x")
    tx_y: str = Field(validation_alias="transcripts_y")
    tx_k: PositiveInt = Field(validation_alias="transcripts_k")
    tx_dist: PositiveFloat = Field(validation_alias="transcripts_dist")
    bd_k: PositiveInt = Field(validation_alias="boundaries_k")
    bd_dist: PositiveFloat = Field(validation_alias="boundaries_dist")
    max_cells_per_tile: PositiveInt
    tile_margin: PositiveFloat
    frac_train: PositiveFloat = Field(validation_alias="fraction_train", lt=1)
    frac_test: PositiveFloat = Field(validation_alias="fraction_test", lt=1)
    frac_val: PositiveFloat = Field(validation_alias="fraction_val", lt=1)
    save_dir: DirectoryPath
    n_workers: int = Field(ge=-1)

    @model_validator(mode="after")
    def validate_parquets(self):
        """
        Validate that parquet files have the expected extension and columns.

        Raises
        ------
        ValueError
            If files are not parquet or expected columns are missing.
        """
        tx_fields = [
            self.tx_feature_name,
            self.tx_cell_id,
            self.tx_x,
            self.tx_y
        ]
        _check_extension(self.tx_path, {".parquet"})
        _check_field_names(self.tx_path, tx_fields)
        _check_extension(self.bd_path, {".parquet"})
        _check_field_names(self.bd_path, ["geometry"])
        return self
    
    @model_validator(mode="after")
    def validate_index_match(self):
        """
        Ensure the cell ID column in transcripts and the index in boundaries 
        have matching data types.

        Raises
        ------
        ValueError
            If the data types of the transcript cell ID column and the 
            boundaries index column do not match.
        """
        tx_schema = pq.read_schema(self.tx_path)
        tx_id = tx_schema.field(self.tx_cell_id)
        
        bd_schema = pq.read_schema(self.bd_path)
        bd_id = bd_schema.pandas_metadata["index_columns"][0]
        bd_id = bd_schema.field(bd_id)

        if tx_id.type != bd_id.type:
            raise ValueError(
                "Data types of transcripts cell ID and boundaries index must "
                f"match, but got {tx_id.type} and {bd_id.type}."
            )
        return self


    @model_validator(mode="after")
    def validate_splits(self):
        """
        Validate that train/test/val fractions sum to 1.0.

        Raises
        ------
        ValueError
            If the total of the fractions deviates from 1.0 beyond tolerance.
        """
        total = self.frac_train + self.frac_test + self.frac_val
        if not np.isclose(total, 1.0, atol=1e-5):
            msg = f"Invalid data split: Total must equal 1.0, but got {total}."
            raise ValueError(msg)
        return self


class TrainConfig(BaseModel):
    """
    Configuration schema for model training.
    See "segger/config/_configs/template.yaml" for more detail.
    """
    gene_emb_path: Optional[FilePath] = Field(
        validation_alias="gene_embedding_csv",
        default=None,
    )
    in_channels: PositiveInt
    hidden_channels: PositiveInt
    out_channels: PositiveInt
    n_mid_layers: PositiveInt 
    n_heads: PositiveInt
    n_workers: int = Field(ge=-1)
    batch_size: PositiveInt
    max_transcripts_k: Optional[int] = None
    max_transcripts_dist: Optional[float] = None
    neg_edge_ratio: PositiveInt = Field(
        validation_alias="negative_edge_sampling_ratio",
    )
    learning_rate: PositiveFloat
    n_epochs: PositiveInt
    root_dir: DirectoryPath

    @field_validator('gene_emb_path', mode='after')
    @classmethod
    def validate_gene_embedding(cls, val: Path):
        """
        Validates the gene embedding file.

        Raises
        ------
        TypeError
            If the file extension is not '.csv', the index is not string-type,
            or embedding columns contain non-float values.
        ValueError
            If any NaN values are found within the embedding data.
        """
        if val is None: return val

        _check_extension(val, {'.csv'})
        embedding = pd.read_csv(val, index_col=0)
        if not is_string_dtype(embedding.index.dtype):
            msg = "Gene embedding index must be string-type (gene identifiers)."
            raise TypeError(msg)

        bad_dtype = [not is_float_dtype(d) for d in embedding.dtypes]
        if any(bad_dtype):
            msg = f"Gene embedding contains {sum(bad_dtype)} non-float columns."
            raise TypeError(msg)

        if embedding.isna().any().any():
            n_genes = embedding.isna().any(axis=1).sum()
            msg = f"Gene embedding contains empty values for {n_genes} genes."
            raise ValueError(msg)
        return val

    
class SeggerConfig(BaseModel):
    #TODO: Add documentation
    data: DataConfig = Field(validation_alias="create_dataset")
    train: TrainConfig = Field(validation_alias="train_model")

    @classmethod
    def from_yaml(cls, config_path: os.PathLike):
        #TODO: Add documentation
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return SeggerConfig(**config_dict)