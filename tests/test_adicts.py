"""
Tests for adict functions.
"""

from typing import Any, Dict, List, Union
import numpy as np
from hypothesis import given, settings, strategies as st
import adicts as a

# Type aliases for readability in tests
Key = Union[int, str, float, bool, complex, bytes, None]
Numeric = Union[int, float]

# Strategies for keys and values
keys: st.SearchStrategy[Key] = (
    st.integers()
    | st.text()
    | st.floats()
    | st.booleans()
    | st.none()
    | st.complex_numbers()
    | st.binary()
)

numeric_values: st.SearchStrategy[Numeric] = st.integers(
    min_value=-1000, max_value=1000
) | st.floats(min_value=-1000, max_value=1000)

positive_numeric_values: st.SearchStrategy[Numeric] = st.integers(
    min_value=1, max_value=100
) | st.floats(min_value=1, max_value=100)

# For exponentiation, use smaller ranges to avoid OverflowError
exponent_values: st.SearchStrategy[Numeric] = st.integers(
    min_value=-10, max_value=10
) | st.floats(min_value=-10, max_value=10)

dict_numeric: st.SearchStrategy[Dict[Key, Numeric]] = st.dictionaries(
    keys, numeric_values
)

dict_positive: st.SearchStrategy[Dict[Key, Numeric]] = st.dictionaries(
    keys, positive_numeric_values
)


@settings(max_examples=10000)
@given(dict_numeric, dict_numeric)
def test_subtract_property(d1: Dict[Key, Numeric], d2: Dict[Key, Numeric]) -> None:
    """Test subtract property."""
    res = a.subtract(d1, d2)
    common_keys = d1.keys() & d2.keys()
    assert set(res.keys()) == common_keys
    for k in common_keys:
        assert res[k] == d1[k] - d2[k]


def test_subtract_mixed_types() -> None:
    """Test subtract with non-numeric keys that are not shared."""
    d1: Dict[str, Any] = {"a": 10, "name": "item", "extra": None}
    d2: Dict[str, Any] = {"a": 3, "other": 5}
    res = a.subtract(d1, d2)
    assert res == {"a": 7}


@settings(max_examples=10000)
@given(dict_numeric, dict_numeric)
def test_add_property(d1: Dict[Key, Numeric], d2: Dict[Key, Numeric]) -> None:
    """Test add property."""
    res = a.add(d1, d2)
    common_keys = d1.keys() & d2.keys()
    assert set(res.keys()) == common_keys
    for k in common_keys:
        assert res[k] == d1[k] + d2[k]


@settings(max_examples=10000)
@given(st.dictionaries(st.integers(), st.integers()))
def test_keys_conditional_remove_property(d: Dict[int, int]) -> None:
    """Test keys_conditional_remove property."""
    res = a.keys_conditional_remove(d, lambda k: k % 2 == 0)
    for k in res:
        assert k % 2 != 0
        assert res[k] == d[k]
    for k in d:
        if k % 2 != 0:
            assert k in res
    assert res == a.k_conditional_remove(d, lambda k: k % 2 == 0)


@settings(max_examples=10000)
@given(st.dictionaries(st.integers(), st.integers()))
def test_values_conditional_remove_property(d: Dict[int, int]) -> None:
    """Test values_conditional_remove property."""
    res = a.values_conditional_remove(d, lambda v: v % 2 == 0)
    for k, v in res.items():
        assert v % 2 != 0
        assert v == d[k]
    for k, v in d.items():
        if v % 2 != 0:
            assert k in res
    assert res == a.v_conditional_remove(d, lambda v: v % 2 == 0)


@settings(max_examples=10000)
@given(dict_numeric, st.lists(keys))
def test_keys_remove_property(d: Dict[Key, Numeric], keys_to_remove: List[Key]) -> None:
    """Test keys_remove property."""
    res = a.keys_remove(d, keys_to_remove)
    for k in res:
        assert k in d
        assert k not in keys_to_remove
    for k in d:
        if k not in keys_to_remove:
            assert k in res
    assert res == a.remove(d, keys_to_remove)


@settings(max_examples=10000)
@given(
    st.dictionaries(st.integers(), numeric_values)
    | st.dictionaries(st.text(min_size=1), numeric_values)
)
def test_keys_to_np_property(d: Dict[Any, Numeric]) -> None:
    """Test keys_to_np property."""
    res = a.keys_to_np(d)
    assert isinstance(res, np.ndarray)
    assert len(res) == len(d)
    assert all(np.equal(res, list(d.keys())))
    assert all(np.equal(res, a.keys_to_numpy(d)))
    assert all(np.equal(res, a.k_to_np(d)))


@settings(max_examples=10000)
@given(
    st.dictionaries(keys, st.integers())
    | st.dictionaries(keys, st.floats(allow_nan=False, allow_infinity=False))
)
def test_values_to_np_property(d: Dict[Key, Any]) -> None:
    """Test values_to_np property."""
    res = a.values_to_np(d)
    assert isinstance(res, np.ndarray)
    assert len(res) == len(d)
    assert list(res) == list(d.values())
    assert list(res) == list(a.values_to_numpy(d))
    assert list(res) == list(a.v_to_np(d))


@settings(max_examples=10000)
@given(dict_numeric, dict_numeric)
def test_multiply_property(d: Dict[Key, Numeric], dmul: Dict[Key, Numeric]) -> None:
    """Test multiply property."""
    res = a.multiply(d, dmul)
    assert set(res.keys()) == set(d.keys())
    for k in d:
        expected = d[k] * dmul.get(k, 1)
        assert res[k] == expected
    assert res == a.d_multiply(d, dmul)


@settings(max_examples=10000)
@given(dict_numeric, numeric_values)
def test_values_multiply_property(d: Dict[Key, Numeric], factor: Numeric) -> None:
    """Test values_multiply property."""
    res = a.values_multiply(d, factor)
    assert set(res.keys()) == set(d.keys())
    for k in d:
        assert res[k] == d[k] * factor
    assert res == a.f_multiply(d, factor)


@settings(max_examples=10000)
@given(dict_positive, st.dictionaries(keys, exponent_values))
def test_power_property(d: Dict[Key, Numeric], exponent: Dict[Key, Numeric]) -> None:
    """Test power property."""
    res = a.power(d, exponent)
    assert set(res.keys()) == set(d.keys())
    for k in d:
        expected = d[k] ** exponent.get(k, 1)
        assert res[k] == expected
    assert res == a.d_pow(d, exponent)


@settings(max_examples=10000)
@given(dict_positive, exponent_values)
def test_values_power_property(d: Dict[Key, Numeric], exponent: Numeric) -> None:
    """Test values_power property."""
    res = a.values_power(d, exponent)
    assert set(res.keys()) == set(d.keys())
    for k in d:
        assert res[k] == d[k] ** exponent
    assert res == a.e_pow(d, exponent)


@settings(max_examples=10000)
@given(st.dictionaries(st.integers(), st.integers()))
def test_keys_apply_property(d: Dict[int, int]) -> None:
    """Test keys_apply property."""
    res = a.keys_apply(d, lambda k: k + 1)
    assert len(res) <= len(d)
    for k, v in d.items():
        assert res[k + 1] == v
    assert res == a.k_apply(d, lambda k: k + 1)


@settings(max_examples=10000)
@given(st.dictionaries(st.integers(), st.integers()))
def test_values_apply_property(d: Dict[int, int]) -> None:
    """Test values_apply property."""
    res = a.values_apply(d, lambda v: v * 2)
    assert set(res.keys()) == set(d.keys())
    for k in d:
        assert res[k] == d[k] * 2
    assert res == a.v_apply(d, lambda v: v * 2)
