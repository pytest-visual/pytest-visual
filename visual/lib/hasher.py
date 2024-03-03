import hashlib
import math
import random

import numpy as np

# FixedPattern


class FixedPattern:
    def __init__(self) -> None:
        """
        Create a random pattern of floats for use in approximate hashing. Note that We use a fixed seed for reproducibility, and Python random module for cross-platform consistency.
        """
        seed = 42
        length = 10000

        rng = random.Random(seed)
        self.pattern = np.array([rng.random() for _ in range(length)], dtype=np.float64)

    def create_patterns(self, n_patterns: int, side_length: int, dims: int) -> np.ndarray:
        """
        Create a N-dimensional array of patterns.

        param: n_patterns: The number of patterns to create.
        param: side_length: The length of each side of the pattern.
        param: dims: The number of dimensions in the pattern.
        return: A multi-dimensional array of patterns with shape (n_patterns, side_length, side_length, ...).
        """

        # Create a 1D pattern of the desired length
        total_length = n_patterns * side_length**dims
        resized_pattern = self.pattern[:total_length]
        if len(resized_pattern) < total_length:  # Wrap around the pattern if it is too short
            resized_pattern = np.pad(resized_pattern, (0, total_length - len(resized_pattern)), mode="wrap")

        # Reshape the pattern into the desired dimensions
        side_length_shape = [side_length] * dims
        shape = (n_patterns, *side_length_shape)
        reshaped_pattern = resized_pattern.reshape(shape)

        return reshaped_pattern


fixedPattern = FixedPattern()

# Hashing functions

n_float_hashes = 16
repeat_size = 8


def hash_text(text: str) -> str:
    """
    Create a hash of a string.
    """
    return hashlib.blake2b(text.encode("utf-8")).hexdigest()


def vector_hash_array(array: np.ndarray) -> np.ndarray:
    """
    Create an approximate hash of an array using FixedPattern.
    """

    # Create a pattern of random floats
    fixedPattern = FixedPattern()
    patterns = fixedPattern.create_patterns(n_patterns=n_float_hashes, side_length=repeat_size, dims=array.ndim)

    # Pad the array to a multiple of the repeat size
    padding = []
    for i in range(array.ndim):
        padding.append((0, repeat_size - (array.shape[i] % repeat_size)))
    padded_array = np.pad(array, padding, mode="constant", constant_values=0)

    # Reshape pattern and array to match dimensions

    # For example, if padded_array.shape == (32, 24) and repeat_size == 8,
    # then reshaped_array.shape == (1, 4, 8, 3, 8).
    # For the pattern to match, we need to reshape it to (n_patterns, 1, 8, 1, 8).
    # Then, we can multiply the two arrays together and sum over all but the first dimension to get the hash.

    # Reshape the array
    array_shape = [1]
    pattern_shape = [n_float_hashes]
    for i in range(array.ndim):
        array_shape.append(padded_array.shape[i] // repeat_size)
        array_shape.append(repeat_size)
        pattern_shape.append(1)
        pattern_shape.append(repeat_size)

    reshaped_array = padded_array.reshape(array_shape)
    reshaped_patterns = patterns.reshape(pattern_shape)

    # Hash the array
    hashed_array = reshaped_array * reshaped_patterns
    hashed_array = np.sum(hashed_array, axis=tuple(range(1, hashed_array.ndim)))

    return hashed_array


def vector_hash_equal(hash1: np.ndarray, hash2: np.ndarray, threshold_error: float) -> bool:
    """
    Check if two hashes are approximately equal. This is done by checking if the mean of the absolute difference between the hashes is less than a certain threshold.
    """

    mean_diff = compute_vector_hash_mean_diff(hash1, hash2)
    return mean_diff < threshold_error


def compute_vector_hash_mean_diff(hash1: np.ndarray, hash2: np.ndarray) -> float:
    """
    Compute the mean difference between two hashes.
    """
    if hash1.size == hash2.size == 0:
        return 0.0
    if hash1.size != hash2.size:
        raise ValueError("Comparing two unequal sized hashes")

    return math.sqrt(np.mean((hash1 - hash2) ** 2))


def vector_hash_threshold(cells_different: float, cell_range: float) -> float:
    """
    Compute the threshold for comparing hashes. Note that such a threshold may fail even when less cells are different than expected, but it asymptotically approaches the expected value as the number of cells increases.
    """

    return cell_range * math.sqrt(cells_different)
