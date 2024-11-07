import numpy as np
import pytest
from ssbgm.utils import np_seed


@pytest.mark.parametrize(
    "seed",
    [0, 1],
)
def test_np_seed(seed: int) -> None:
    with np_seed(seed):
        x = np.random.rand()
    with np_seed(seed):
        y = np.random.rand()
    with np_seed(seed):
        z = np.random.rand()
    w = np.random.rand()
    assert x == y == z
    assert x != w


def test_np_seed_none() -> None:
    with np_seed():
        x = np.random.rand()
    with np_seed():
        y = np.random.rand()
    with np_seed():
        z = np.random.rand()
    assert x != y
    assert x != z
    assert y != z
