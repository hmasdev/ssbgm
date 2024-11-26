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


def build_experimental_warn_message(
    prefix: str = '[experimental warning]',
) -> str:
    """build warning message for experimental features.

    Args:
        prefix (str, optional): prefix of warning message. Defaults to ''.

    Returns:
        str: warning message.
    """  # noqa
    return (
        f'{prefix}. '
        'You are using an experimental feature. '
        'This feature can cause unexpected behavior. '
        'Please use it at your own risk.'
        'If you notice any issues, please report them to the developers:'
        'https://github.com/hmasdev/ssbgm/issues/new'
    )
