import numpy as np
import pytest

from visual.interface import VisualFixture, standardize
from visual.lib.hasher import (
    FixedPattern,
    hash_text,
    vector_hash_array,
    vector_hash_equal_mean,
)

# Test FixedPattern


def test_fixed_pattern_consistency():
    # Check that the pattern is consistent
    assert (FixedPattern().pattern == FixedPattern().pattern).all()

    # Check that the pattern does not change over implementations
    fixedPattern = FixedPattern()
    patterns = fixedPattern.create_patterns(n_patterns=2, side_length=3, dims=2)

    assert (2, 3, 3) == patterns.shape
    assert pytest.approx(0.639426798) == patterns[0, 0, 0]
    assert pytest.approx(0.025010755) == patterns[0, 0, 1]
    assert pytest.approx(0.223210738) == patterns[0, 1, 0]
    assert pytest.approx(0.029797219) == patterns[1, 0, 0]


def test_fixed_pattern_visual(visual: VisualFixture):
    side_length = 8

    fixedPattern = FixedPattern()
    patterns = fixedPattern.create_patterns(n_patterns=4, side_length=side_length, dims=2)
    patterns = [standardize(pattern) for pattern in patterns]

    visual.text(f"Random, but fixed patterns of side length {side_length} and 2 dimensions")
    visual.images(list(patterns))


# Test hashing functions


def test_hash_text():
    assert "021ced8799" == hash_text("hello world")[:10]


def test_vector_hash_array_consistency():
    array = np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]])
    hashed_array1 = vector_hash_array(array)
    hashed_array2 = vector_hash_array(array)

    assert (hashed_array1 == hashed_array2).all()
    assert pytest.approx(10.269079760) == hashed_array1[0]
    assert pytest.approx(17.233428217) == hashed_array1[1]


def test_vector_hash_similarity():
    array1 = np.zeros((100, 100), dtype=np.uint8)
    array2 = np.zeros((100, 100), dtype=np.uint8)
    array3 = np.zeros((100, 100), dtype=np.uint8)

    array2[50:60, 50] = 255  # Change 10 pixels
    array3[50:60, :] = 255  # Change 1000 pixels

    hash1 = vector_hash_array(array1)
    hash2 = vector_hash_array(array2)
    hash3 = vector_hash_array(array3)

    assert vector_hash_equal_mean(hash1, hash2, 100, 255)
    assert not vector_hash_equal_mean(hash1, hash3, 100, 255)
