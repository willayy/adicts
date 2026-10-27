"""
Implementations of adict functions.
"""

from collections.abc import Mapping
from typing import Callable, TypeVar, cast

import numpy as np
import numpy.typing as npt

A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")

Numeric = int | float


def subtract(
    d1: Mapping[A, object],
    d2: Mapping[A, object],
) -> Mapping[A, Numeric]:
    """
    Subtracts values of two dictionaries for common keys, ignoring others.

    :param d1: First dictionary
    :type d1: Mapping[A, object]
    :param d2: Second dictionary
    :type d2: Mapping[A, object]
    :return: Dictionary with subtracted values for common keys
    :rtype: Mapping[A, Numeric]
    """
    common_keys = d1.keys() & d2.keys()
    return {k: cast(Numeric, d1[k]) - cast(Numeric, d2[k]) for k in common_keys}


def add(
    d1: Mapping[A, object],
    d2: Mapping[A, object],
) -> Mapping[A, Numeric]:
    """
    Adds values of two dictionaries for common keys, ignoring others.

    :param d1: First dictionary
    :type d1: Mapping[A, object]
    :param d2: Second dictionary
    :type d2: Mapping[A, object]
    :return: Dictionary with summed values for common keys
    :rtype: Mapping[A, Numeric]
    """
    common_keys = d1.keys() & d2.keys()
    return {k: cast(Numeric, d1[k]) + cast(Numeric, d2[k]) for k in common_keys}


def multiply(d: Mapping[A, object], dmul: Mapping[A, object]) -> Mapping[A, Numeric]:
    """
    Multiplies dictionary values by values of another dictionary for common keys.

    :param d: Input dictionary
    :type d: Mapping[A, object]
    :param dmul: Dictionary of factors
    :type dmul: Mapping[A, object]
    :return: Dictionary with multiplied values
    :rtype: Mapping[A, Numeric]
    """
    return {k: cast(Numeric, d[k]) * cast(Numeric, dmul.get(k, 1)) for k in d.keys()}


def power(d: Mapping[A, object], exponent: Mapping[A, object]) -> Mapping[A, Numeric]:
    """
    Raises dictionary values to powers specified in another dictionary.

    :param d: Input dictionary
    :type d: Mapping[A, object]
    :param exponent: Dictionary of exponents
    :type exponent: Mapping[A, object]
    :return: Dictionary with values raised to the specified powers
    :rtype: Mapping[A, Numeric]
    """
    return {
        k: cast(Numeric, d[k]) ** cast(Numeric, exponent.get(k, 1)) for k in d.keys()
    }


def values_multiply(d: Mapping[A, Numeric], factor: Numeric) -> Mapping[A, Numeric]:
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


def values_power(d: Mapping[A, Numeric], exponent: Numeric) -> Mapping[A, Numeric]:
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


def values_apply(d: Mapping[A, B], func: Callable[[B], C]) -> Mapping[A, C]:
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


def values_conditional_remove(
    d: Mapping[A, B], condition: Callable[[B], bool]
) -> Mapping[A, B]:
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


def values_to_np(d: Mapping[A, Numeric]) -> npt.NDArray[np.number]:
    """
    Converts dictionary values to a NumPy array.

    :param d: Input dictionary
    :type d: Mapping[A, Numeric]
    :return: NumPy array of values
    :rtype: npt.NDArray[Numeric]
    """
    return np.array(list(d.values()))


def keys_apply(d: Mapping[A, B], func: Callable[[A], C]) -> Mapping[C, B]:
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


def keys_conditional_remove(
    d: Mapping[A, B], condition: Callable[[A], bool]
) -> Mapping[A, B]:
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


def keys_remove(d: Mapping[A, B], keys: list[A]) -> Mapping[A, B]:
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


def keys_to_np(d: Mapping[Numeric, B]) -> npt.NDArray[np.number]:
    """
    Converts dictionary keys to a NumPy array.

    :param d: Input dictionary
    :type d: Mapping[A, B]
    :return: NumPy array of keys
    :rtype: npt.NDArray[Any]
    """
    return np.array(list(d.keys()))


# Aliases for backward compatibility and alternative conventions
d_multiply = multiply
d_pow = power
f_multiply = values_multiply
e_pow = values_power
v_apply = values_apply
v_conditional_remove = values_conditional_remove
v_to_np = values_to_np
values_to_numpy = values_to_np

k_apply = keys_apply
k_conditional_remove = keys_conditional_remove
remove = keys_remove
keys_to_numpy = keys_to_np
k_to_np = keys_to_np
