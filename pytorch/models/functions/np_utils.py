"""
utilities for ndarray
"""
import numpy as np
from typing import Tuple


def normalize(
    x: np.ndarray,
    input_range: Tuple[float, float] = (0.0, 255.0),
    output_range: Tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """
    Function to normalize a numpy array within a specified range
    Args:
        x (np.ndarray): Data array
        input_range (Tuple[float, float]):  List of maximum and minimum values of original data, e.g. indataRange=[0.0, 255.0].
        output_range (Tuple[float, float]): List of maximum and minimum values of output data, e.g. indataRange=[0.0, 1.0].
    Return:
        x (np.ndarray): Normalized data array
    """
    x = (x - input_range[0]) / (input_range[1] - input_range[0])
    x = x * (output_range[1] - output_range[0]) + output_range[0]
    return x


def normalize_fixed(
    x: np.ndarray,
    input_max: float = 255.0,
) -> np.ndarray:
    """
    Function to normalize a numpy array within a specified range
    input range is fixed to [0.0, input_max]
    output range is fixed to [0.0, 1.0]
    Args:
        x (np.ndarray): Data array
        input_max (float):  Maximum value of original data
    Return:
        x (np.ndarray): Normalized data array
    """
    x = x / input_max
    return x
