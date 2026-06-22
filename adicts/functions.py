"""
Implementations of adict functions.
"""

from typing import TypeVar, Union, Callable, Any
import numpy as np
import numpy.typing as npt
from collections.abc import Mapping

A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")

Numeric = Union[int, float]


def subtract(
    d1: Mapping[A, Numeric],
    d2: Mapping[A, Numeric],
) -> Mapping[A, Numeric]:
    """
    Subtracts values of two dictionaries for common keys, ignoring others.

    :param d1: First dictionary
    :type d1: Mapping[A, Numeric]
    :param d2: Second dictionary
    :type d2: Mapping[A, Numeric]
    :return: Dictionary with subtracted values for common keys
    :rtype: Mapping[A, Numeric]
    """
    common_keys = d1.keys() & d2.keys()
    return {k: d1[k] - d2[k] for k in common_keys}


def add(
    d1: Mapping[A, Numeric],
    d2: Mapping[A, Numeric],
) -> Mapping[A, Numeric]:
    """
    Adds values of two dictionaries for common keys, ignoring others.

    :param d1: First dictionary
    :type d1: Mapping[A, Numeric]
    :param d2: Second dictionary
    :type d2: Mapping[A, Numeric]
    :return: Dictionary with summed values for common keys
    :rtype: Mapping[A, Numeric]
    """
    common_keys = d1.keys() & d2.keys()
    return {k: d1[k] + d2[k] for k in common_keys}


def k_conditional_remove(d: Mapping[A, B], condition: Callable[[A], bool]) -> Mapping[A, B]:
    """
    Removes items from dictionary if condition returns true when applied to keys.

    :param d: Input dictionary
    :type d: Mapping[A, B]
    :param condition: Condition function that returns True for keys to remove
    :type condition: Callable[[A], bool]
    :return: Filtered dictionary
    :rtype: Mapping[A, B]
    """
    return {k: v for k, v in d.items() if not condition(k)}


def v_conditional_remove(d: Mapping[A, B], condition: Callable[[B], bool]) -> Mapping[A, B]:
    """
    Removes items from dictionary if condition returns true when applied to values.

    :param d: Input dictionary
    :type d: Mapping[A, B]
    :param condition: Condition function that returns True for values to remove
    :type condition: Callable[[B], bool]
    :return: Filtered dictionary
    :rtype: Mapping[A, B]
    """
    return {k: v for k, v in d.items() if not condition(v)}


def remove(d: Mapping[A, B], keys: list[A]) -> Mapping[A, B]:
    """
    Removes specified keys from the dictionary.

    :param d: Input dictionary
    :type d: Mapping[A, B]
    :param keys: List of keys to remove
    :type keys: list[A]
    :return: Filtered dictionary
    :rtype: Mapping[A, B]
    """
    return {k: v for k, v in d.items() if k not in keys}


def k_to_np(d: Mapping[A, B]) -> npt.NDArray[Any]:
    """
    Converts dictionary keys to a NumPy array.

    :param d: Input dictionary
    :type d: Mapping[A, B]
    :return: NumPy array of keys
    :rtype: npt.NDArray[Any]
    """
    return np.array(list(d.keys()))


def v_to_np(d: Mapping[A, Numeric]) -> npt.NDArray[np.number]:
    """
    Converts dictionary values to a NumPy array.

    :param d: Input dictionary
    :type d: Mapping[A, Numeric]
    :return: NumPy array of values
    :rtype: npt.NDArray[Numeric]
    """

    return np.array(list(d.values()))


def d_multiply(d: Mapping[A, Numeric], dmul: Mapping[A, Numeric]) -> Mapping[A, Numeric]:
    """
    Multiplies dictionary values by values of another dictionary for common keys.

    :param d: Input dictionary
    :type d: Mapping[A, Numeric]
    :param dmul: Dictionary of factors
    :type dmul: Mapping[A, Numeric]
    :return: Dictionary with multiplied values
    :rtype: Mapping[A, Numeric]
    """
    return {k: d[k] * dmul.get(k, 1) for k in d.keys()}


def f_multiply(d: Mapping[A, Numeric], factor: Numeric) -> Mapping[A, Numeric]:
    """
    Multiplies dictionary values by a scalar factor.

    :param d: Input dictionary
    :type d: Mapping[A, Numeric]
    :param factor: Scalar factor
    :type factor: Numeric
    :return: Dictionary with multiplied values
    :rtype: Mapping[A, Numeric]
    """
    return {k: d[k] * factor for k in d.keys()}


def d_pow(d: Mapping[A, Numeric], exponent: Mapping[A, Numeric]) -> Mapping[A, Numeric]:
    """
    Raises dictionary values to powers specified in another dictionary.

    :param d: Input dictionary
    :type d: Mapping[A, Numeric]
    :param exponent: Dictionary of exponents
    :type exponent: Mapping[A, Numeric]
    :return: Dictionary with values raised to the specified powers
    :rtype: Mapping[A, Numeric]
    """

    return {k: d[k] ** exponent.get(k, 1) for k in d.keys()}


def e_pow(d: Mapping[A, Numeric], exponent: Numeric) -> Mapping[A, Numeric]:
    """
    Raises dictionary values to a scalar exponent.

    :param d: Input dictionary
    :type d: Mapping[A, Numeric]
    :param exponent: Scalar exponent
    :type exponent: Numeric
    :return: Dictionary with values raised to the specified power
    :rtype: Mapping[A, Numeric]
    """
    return {k: d[k] ** exponent for k in d.keys()}


def k_apply(d: Mapping[A, B], func: Callable[[A], C]) -> Mapping[C, B]:
    """
    Applies a function to all keys in the dictionary.

    :param d: Input dictionary
    :type d: Mapping[A, B]
    :param func: Function to apply to keys
    :type func: Callable[[A], C]
    :return: Dictionary with transformed keys
    :rtype: Mapping[C, B]
    """
    return {func(k): v for k, v in d.items()}


def v_apply(d: Mapping[A, B], func: Callable[[B], C]) -> Mapping[A, C]:
    """
    Applies a function to all values in the dictionary.

    :param d: Input dictionary
    :type d: Mapping[A, B]
    :param func: Function to apply to values
    :type func: Callable[[B], C]
    :return: Dictionary with transformed values
    :rtype: Mapping[A, C]
    """
    return {k: func(v) for k, v in d.items()}
