from contextlib import contextmanager
from logging import Logger, getLogger
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


def experimental_warn(
    prefix: str = '[experimental warning]',
    logger: Logger = getLogger(__name__),
) -> None:
    """log warning message for experimental feature.

    Args:
        prefix (str, optional): prefix of warning message. Defaults to ''.
        logger (Logger, optional): logger object. Defaults to getLogger(__name__).
    """  # noqa
    logger.warning(
        f'{prefix}. '
        'You are using an experimental feature. '
        'This feature can cause unexpected behavior. '
        'Please use it at your own risk.'
        'If you notice any issues, please report them to the developers:'
        'https://github.com/hmasdev/ssbgm/issues/new'
    )
