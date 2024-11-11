from typing import Callable
import numpy as np
from tqdm import trange
from .exceptions import MaximumIterationError


def langevin_montecarlo(
    x0: np.ndarray,
    nabla_U: Callable[[np.ndarray], np.ndarray],
    delta_t: float = 0.1,
    n_steps: int = 1000,
    pdf: Callable[[np.ndarray], float] | None = None,
    max_n_iter_until_accept: int = 100,
) -> np.ndarray:
    """Generate samples from the Langevin Monte Carlo algorithm.

    Args:
        x0 (np.ndarray): initial position of the chain.
        nabla_U (Callable[[np.ndarray], np.ndarray]): gradient of the potential energy function.
        delta_t (float, optional): time step. Defaults to 0.1.
        n_steps (int, optional): number of steps. Defaults to 1000.
        pdf (Callable[[np.ndarray], np.ndarray], optional): probability density function. Defaults to None.
            When pdf is given, `langenvin_montecarlo` behaves as Metropolis-Adjusted Langevin Algorithm.
            See https://en.wikipedia.org/wiki/Metropolis-adjusted_Langevin_algorithm .
            Note that the shape of argument of pdf must be the same as x0.
        max_n_iter_until_accept (int, optional): maximum number of iterations until accept. Defaults to 100.

    Raises:
        ValueError: raised when the initial position is not in the support of pdf (Only when pdf is given).
        MaximumIterationError: raised when the maximum iteration is reached before accepting the proposal.

    Returns:
        np.ndarray: samples from the Langevin Monte Carlo algorithm.
            (n_steps, *x0.shape) shape array.

    NOTE:
        the shape of an output of nabla_U must be the same as x0.

    Algorithm:
        Langevin Monte Carlo:
            1. Initialize x0
            2. for k = 1, 2, ..., n_steps:
                1. x_k = x_{k-1} - nabla_U(x_{k-1}) * delta_t + sqrt(2*delta_t) * N(0, I)
            3. return x0, x1, ..., x_{n_steps}
        Metropolis-Adjusted Langevin Algorithm:
            1. Initialize x0
            2. for k = 1, 2, ..., n_steps:
                1. for l = 1, 2, ..., L:
                    1. z_k^l = x_{k-1} - nabla_U(x_{k-1}) * delta_t + sqrt(2*delta_t) * N(0, I)
                    2. alpha = min(1, (pdf(z_k^l) q(x_{k-1}|z_k^l)) / (pdf(x_{k-1}) q(z_k^l|x_{k-1})))
                    3. If U(0, 1) < alpha, x_k := z_k^l and exit the loop of l; otherwise, continue the loop of l.
            3. return x0, x1, ..., x_{n_steps}
    """  # noqa

    # validation
    if pdf is not None and pdf(x0) <= 0:
        raise ValueError(f"x0 must be in the support of pdf. But pdf(x0) = {pdf(x0)}")  # noqa

    # Initalize
    xs = np.zeros((n_steps,)+x0.shape)
    xs[0] = x0

    # define the proposal function
    def suc(x: np.ndarray) -> np.ndarray:
        return x - nabla_U(x) * delta_t + np.random.randn(*x0.shape) * np.sqrt(2*delta_t)  # type: ignore # noqa

    if pdf is not None:
        _suc = suc

        def q(x: np.ndarray, y: np.ndarray) -> float:
            return np.exp(-np.sum((x-y+nabla_U(y)*delta_t)**2)/(4*delta_t))  # type: ignore # noqa

        def suc(x: np.ndarray) -> np.ndarray:
            for _ in range(max_n_iter_until_accept):
                z = _suc(x)
                nume = pdf(z) * q(x, z)
                denom = pdf(x) * q(z, x)
                if np.random.rand() < min(1, nume/denom):
                    return z
            raise MaximumIterationError()

    for k in trange(1, n_steps):
        xs[k] = suc(xs[k-1])

    return xs


def euler(
    x0: np.ndarray,
    f: Callable[[np.ndarray, float], np.ndarray],
    t0: float,
    t1: float,
    n_steps: int
) -> tuple[np.ndarray, np.ndarray]:
    """Generate samples from the Euler method.

    Args:
        x0 (np.ndarray): initial position of the chain.
        f (Callable[[np.ndarray], np.ndarray]): vector field.
        t0 (float): initial time.
        t1 (float): final time.
        n_steps (int): number of steps.

    Returns:
        np.ndarray: time points.
        np.ndarray: samples from the Euler method.
            (n_steps, *x0.shape) shape array.

    NOTE:
        the shape of an output of f must be the same as x0.
    """  # noqa
    ts = np.linspace(t0, t1, n_steps)
    dt = ts[1] - ts[0]
    xs = np.zeros((n_steps,)+x0.shape)
    xs[0] = x0
    for k in trange(1, n_steps):
        xs[k] = xs[k-1] + f(xs[k-1], ts[k-1]) * dt
    return ts, xs


def euler_maruyama(
    x0: np.ndarray,
    f: Callable[[np.ndarray, float], np.ndarray],
    g: Callable[[np.ndarray, float], np.ndarray],
    t0: float,
    t1: float,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate samples from the Euler-Maruyama method.

    Args:
        x0 (np.ndarray): initial position of the chain.
        f (Callable[[np.ndarray], np.ndarray]): vector field.
        g (Callable[[np.ndarray], np.ndarray]): noise vector field.
        t0 (float): initial time.
        t1 (float): final time.
        n_steps (int): number of steps.

    Returns:
        np.ndarray: time points.
        np.ndarray: samples from the Euler-Maruyama method.
            (n_steps, *x0.shape) shape array.

    NOTE:
        the shape of an output of f and g must be the same as x0.
    """  # noqa
    ts = np.linspace(t0, t1, n_steps)
    dt = ts[1] - ts[0]
    xs = np.zeros((n_steps,)+x0.shape)
    xs[0] = x0
    for k in trange(1, n_steps):
        xs[k] = xs[k-1]
        xs[k] += f(xs[k-1], ts[k-1]) * dt
        xs[k] += g(xs[k-1], ts[k-1]) * np.sqrt(abs(dt)) * np.random.randn(*x0.shape)  # noqa
    return ts, xs
