import pytest
import numpy as np
from sklearn.preprocessing import LabelEncoder
from segger.data.parquet.transcript_embedding import TranscriptEmbedding
from torch import Tensor
from torch.testing import assert_close
import pandas as pd

@pytest.fixture
def classes():
    return np.array(['A', 'B', 'C', 'D', 'E'])

# Classes only, no weights
@pytest.mark.dependency(name='no_weights_correct')
def test_no_weights_correct(classes):
    embedding = TranscriptEmbedding(classes)
    actual = embedding.embed(classes)
    expected = Tensor(LabelEncoder().fit_transform(classes))
    assert_close(actual, expected, check_dtype=False)

def test_no_weights_missing(classes):
    embedding = TranscriptEmbedding(classes[1:])
    with pytest.raises(ValueError):
        embedding.embed(classes[:1])

@pytest.mark.dependency(depends=['no_weights_correct'])
@pytest.mark.parametrize('n,s', [(1, 1), (4, 1), (4, 2), (4, 4)])
def test_no_weights_shape(n, s):
    classes = np.arange(n)
    embedding = TranscriptEmbedding(classes)
    expected = (s,)
    actual = tuple(embedding.embed(classes[:s]).shape)
    assert actual == expected

# Classes and weights
@pytest.fixture
def weights(classes):
    weights = np.random.random((len(classes), 3))
    return pd.DataFrame(weights, index=classes)

def test_weights_missing(classes, weights):
    weights = weights.drop(classes[0], axis=0)
    with pytest.raises(ValueError):
        embedding = TranscriptEmbedding(classes, weights)

@pytest.mark.dependency(name='weights_correct')
def test_weights_correct(classes, weights):
    embedding = TranscriptEmbedding(classes, weights)
    expected = Tensor(weights.values[:1])  # shape is (1, 3)
    actual = embedding.embed(classes[:1])
    assert_close(actual, expected, check_dtype=False)

@pytest.mark.dependency(depends=['weights_correct'])
@pytest.mark.parametrize('n,s', [(1, 1), (4, 1), (4, 2), (4, 4)])
def test_weights_shape(n, s):
    classes = np.arange(n)
    weights = pd.DataFrame(np.random.random((len(classes), 3)), index=classes)
    embedding = TranscriptEmbedding(classes, weights)
    expected = (s, 3)
    actual = tuple(embedding.embed(classes[:s]).shape)
    assert actual == expected
