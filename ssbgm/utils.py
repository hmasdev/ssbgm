from contextlib import contextmanager
from typing import Generator
import numpy as np


@contextmanager
def np_seed(
    seed: int | None = None
) -> Generator[None, None, None]:
    """context manager for setting numpy random seed temporarily.

    Args:
        seed (int | None, optional): random seed. Defaults to None.

    Yields:
        None
    """

    if seed is None:
        # do nothing
        yield
    else:
        rng_state = np.random.get_state()
        np.random.seed(seed)
        try:
            yield
        finally:
            np.random.set_state(rng_state)
