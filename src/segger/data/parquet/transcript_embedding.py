import torch.nn
import torch.nn.functional as F
from torch import Tensor, LongTensor
from sklearn.preprocessing import LabelEncoder
from typing import Optional, Union
from numpy.typing import ArrayLike
import pandas as pd


# TODO: Add documentation
class TranscriptEmbedding(torch.nn.Module):
    """
    Utility class to handle transcript embeddings in PyTorch so that they are
    optionally learnable in the future.

    Default behavior is to use the index of gene names.
    """

    # TODO: Add documentation
    @staticmethod
    def _check_inputs(
        classes: ArrayLike,
        weights: Union[pd.DataFrame, None],
    ):
        # Classes is a 1D array
        if len(classes.shape) > 1:
            msg = "'classes' should be a 1D array, got an array of shape " f"{classes.shape} instead."
            raise ValueError(msg)
        # Items appear exactly once
        if len(classes) != len(set(classes)):
            msg = "All embedding classes must be unique. One or more items in " "'classes' appears twice."
            raise ValueError(msg)
        # All classes have an entry in weights
        elif weights is not None:
            missing = set(classes).difference(weights.index)
            if len(missing) > 0:
                msg = f"Index of 'weights' DataFrame is missing {len(missing)} " "entries compared to classes."
                raise ValueError(msg)

    # TODO: Add documentation
    def __init__(
        self,
        classes: ArrayLike,
        weights: Optional[pd.DataFrame] = None,
    ):
        # check input arguments
        TranscriptEmbedding._check_inputs(classes, weights)
        # Setup as PyTorch module
        super(TranscriptEmbedding, self).__init__()
        self._encoder = LabelEncoder().fit(classes)
        if weights is None:
            self._weights = None
        else:
            self._weights = Tensor(weights.loc[classes].values)

    # TODO: Add documentation
    def embed(self, classes: ArrayLike):
        indices = LongTensor(self._encoder.transform(classes))
        # Default, one-hot encoding
        if self._weights is None:
            return indices  # F.one_hot(indices, len(self._encoder.classes_))
        else:
            return F.embedding(indices, self._weights)
