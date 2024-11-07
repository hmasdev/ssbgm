from typing import Callable
import numpy as np
from tqdm import trange


def langevin_montecarlo(
    x0: np.ndarray,
    nabla_U: Callable[[np.ndarray], np.ndarray],
    delta_t: float = 0.1,
    n_steps: int = 1000,
) -> np.ndarray:
    """Generate samples from the Langevin Monte Carlo algorithm.

    Args:
        x0 (np.ndarray): initial position of the chain.
        nabla_U (Callable[[np.ndarray], np.ndarray]): gradient of the potential energy function.
        delta_t (float, optional): time step. Defaults to 0.1.
        n_steps (int, optional): number of steps. Defaults to 1000.

    Returns:
        np.ndarray: samples from the Langevin Monte Carlo algorithm.
            (n_steps, *x0.shape) shape array.

    NOTE:
        the shape of an output of nabla_U must be the same as x0.
    """  # noqa
    xs = np.zeros((n_steps,)+x0.shape)
    xs[0] = x0
    for k in trange(1, n_steps):
        xs[k] = xs[k-1]
        xs[k] += - nabla_U(xs[k-1]) * delta_t
        xs[k] += np.random.randn(*x0.shape) * np.sqrt(2*delta_t)
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
