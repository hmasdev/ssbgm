from logging import getLogger, Logger
from typing import Callable
import numpy as np
from tqdm import trange
from .exceptions import MaximumIterationError
from .utils import build_experimental_warn_message


def langevin_montecarlo(
    x0: np.ndarray,
    nabla_U: Callable[[np.ndarray], np.ndarray],
    delta_t: float = 0.1,
    n_steps: int = 1000,
    pdf: Callable[[np.ndarray], float] | None = None,
    max_n_iter_until_accept: int = 1000,
    *,
    use_pdf_as_domain_indicator: bool = False,
    verbose: bool = False,
    logger: Logger = getLogger(__name__),
) -> np.ndarray:
    """Generate samples from the Langevin Monte Carlo algorithm.

    Args:
        x0 (np.ndarray): initial position of the chain.
            Shape: (N, n_outputs) or (n_outputs,).
        nabla_U (Callable[[np.ndarray], np.ndarray]): gradient of the potential energy function.
            Shape: (N, n_outputs) -> (N, n_outputs).
        delta_t (float, optional): time step. Defaults to 0.1.
        n_steps (int, optional): number of steps. Defaults to 1000.
        pdf (Callable[[np.ndarray], np.ndarray], optional): probability density function. Defaults to None.
            When pdf is given, `langenvin_montecarlo` behaves as Metropolis-Adjusted Langevin Algorithm.
            See https://en.wikipedia.org/wiki/Metropolis-adjusted_Langevin_algorithm .
            Shape: (n_outputs, ) -> float.
        max_n_iter_until_accept (int, optional): maximum number of iterations until accept. Defaults to 100.
        use_pdf_as_domain_indicator (bool, optional): whether to use pdf as a domain indicator. Defaults to False.
            If True, pdf is used as a domain indicator. That is, the proposal is rejected when pdf(x) = 0.
            Just valid when pdf is given.
        verbose (bool, optional): whether to show the progress bar. Defaults to False.
        logger (Logger, optional): logger object. Defaults to getLogger(__name__).

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

    # Initalize
    x0dim = x0.ndim
    assert x0dim in (1, 2), f"x0 must be 1D or 2D array. But x0.ndim = {x0.ndim}"  # noqa
    if x0dim == 1:
        x0 = x0.reshape(1, -1)
    # NOTE: x0.shape = (1, n_outputs) or (N, n_outputs)
    xs = np.zeros((n_steps,)+x0.shape)
    xs[0] = x0

    # validation
    if pdf is not None and pdf(x0[0]) <= 0:
        raise ValueError(f"x0 must be in the support of pdf. But pdf(x0) = {pdf(x0[0])}")  # noqa

    # define the proposal function
    def _suc(x: np.ndarray) -> np.ndarray:
        return x - nabla_U(x) * delta_t + np.random.randn(*x.shape) * np.sqrt(2*delta_t)  # type: ignore # noqa

    if pdf is None:
        suc = _suc
    elif use_pdf_as_domain_indicator:
        logger.warning(build_experimental_warn_message(
            prefix='[experimental warning](proposal/rejection by domain indicator)',  # noqa
        ))

        def suc(x: np.ndarray) -> np.ndarray:  # type: ignore
            N = x.shape[0]
            L = 1  # FIXME: magic number
            for ctr in range(max_n_iter_until_accept):
                # x: (N, n_outputs)
                # zs: (2L, N, n_outputs)
                if ctr == 0:
                    zs = np.vstack([
                        np.array([_suc(x) for _ in range(L)]),  # noqa
                        np.array([_suc(x) for _ in range(L)]),  # noqa
                    ])
                else:
                    zs = np.vstack([zs, np.array([_suc(x) for _ in range(L)])])
                # acceptance threshold
                prob_z_ln = np.apply_along_axis(pdf, -1, zs).reshape(2*L, N)  # noqa
                # accept or not
                accept = prob_z_ln > 0
                accepted_idx = np.argmax(accept, axis=0)
                # retry or end
                if np.all(~accept, axis=0).any():
                    # if all samples are rejected, retry
                    # FIXME: Inefficient. Need to be optimized.
                    zs = zs[accepted_idx, np.arange(N)][np.newaxis, :, :]
                    continue
                else:
                    z = zs[accepted_idx, np.arange(N)]
                    return z  # type: ignore
            raise MaximumIterationError()
    else:
        logger.warning(build_experimental_warn_message(
            prefix='[experimental warning](Metropolis-Adjusted Langevin Algorithm)',  # noqa
        ))

        def suc(x: np.ndarray) -> np.ndarray:  # type: ignore
            N = x.shape[0]
            L = 1  # FIXME: magic number
            for ctr in range(max_n_iter_until_accept):
                # x: (N, n_outputs)
                # zs: (L*(ctr+1), N, n_outputs)
                if ctr == 0:
                    zs = np.vstack([
                        np.array([_suc(x) for _ in range(L)]),  # noqa
                        np.array([_suc(x) for _ in range(L)]),  # noqa
                    ])
                else:
                    zs = np.vstack([zs, np.array([_suc(x) for _ in range(L)])])
                # acceptance threshold
                prob_z_ln = np.apply_along_axis(pdf, -1, zs).reshape(L*2, N)  # noqa
                prob_x_n = np.apply_along_axis(pdf, -1, x).reshape(N)
                q_xz_nl = np.array([np.exp(-np.sum((x[n]-zs[:, n]+nabla_U(zs[:, n])*delta_t)**2, axis=1)/(4*delta_t)) for n in range(N)]).reshape(N, L)  # noqa
                q_zx_ln = np.array([np.exp(-np.sum((zs[l]-x+nabla_U(x)*delta_t)**2, axis=1)/(4*delta_t)) for l in range(L)]).reshape(L, N)  # noqa
                alphas = np.einsum('ln,n,nl,ln->ln', prob_z_ln, 1/prob_x_n, q_xz_nl, 1/q_zx_ln)  # type: ignore # noqa
                # accept or not
                u = np.random.rand(*alphas.shape)
                accept = u < alphas
                accepted_idx = np.argmax(accept, axis=0)
                # retry or end
                if np.all(~accept, axis=0).any():
                    # if all samples are rejected, retry
                    # FIXME: Inefficient. Need to be optimized.
                    zs = zs[accepted_idx, np.arange(N)][np.newaxis, :, :]
                    continue
                else:
                    z = zs[accepted_idx, np.arange(N)]
                    return z  # type: ignore
            raise MaximumIterationError()

    for k in (trange if verbose else range)(1, n_steps):
        xs[k] = suc(xs[k-1])

    assert xs.shape == (n_steps, *x0.shape), "Internal Error"
    if x0dim == 1:
        return xs.reshape(n_steps, -1)
    else:
        return xs


def euler(
    x0: np.ndarray,
    f: Callable[[np.ndarray, float], np.ndarray],
    t0: float,
    t1: float,
    n_steps: int,
    *,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate samples from the Euler method.

    Args:
        x0 (np.ndarray): initial position of the chain.
            Shape: (n_outputs,) or (N, n_outputs).
        f (Callable[[np.ndarray, float], np.ndarray]): vector field.
            Shape: (N, n_outputs,), float -> (N, n_outputs,).
        t0 (float): initial time.
        t1 (float): final time.
        n_steps (int): number of steps.
        verbose (bool, optional): whether to show the progress bar. Defaults to False.

    Returns:
        np.ndarray: time points.
        np.ndarray: samples from the Euler method.
            (n_steps, *x0.shape) shape array.

    NOTE:
        the shape of an output of f must be the same as x0.
    """  # noqa
    ts = np.linspace(t0, t1, n_steps)
    dt = ts[1] - ts[0]

    x0dim = x0.ndim
    assert x0dim in (1, 2), f"x0 must be 1D or 2D array. But x0.ndim = {x0.ndim}"  # noqa
    if x0dim == 1:
        x0 = x0.reshape(1, -1)
    xs = np.zeros((n_steps,)+x0.shape)
    xs[0] = x0
    for k in (trange if verbose else range)(1, n_steps):
        xs[k] = xs[k-1] + f(xs[k-1], ts[k-1]) * dt

    assert xs.shape == (n_steps, *x0.shape), "Internal Error"
    if x0dim == 1:
        return ts, xs.reshape(n_steps, -1)
    else:
        return ts, xs


def euler_maruyama(
    x0: np.ndarray,
    f: Callable[[np.ndarray, float], np.ndarray],
    g: Callable[[np.ndarray, float], np.ndarray],
    t0: float,
    t1: float,
    n_steps: int,
    *,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate samples from the Euler-Maruyama method.

    Args:
        x0 (np.ndarray): initial position of the chain.
            Shape: (n_outputs,) or (N, n_outputs).
        f (Callable[[np.ndarray, float], np.ndarray]): vector field.
            Shape: (N, n_outputs,), float -> (N, n_outputs,).
        g (Callable[[np.ndarray, float], np.ndarray]): noise vector field.
            Shape: (N, n_outputs,), float -> (N, n_outputs,).
        t0 (float): initial time.
        t1 (float): final time.
        n_steps (int): number of steps.
        verbose (bool, optional): whether to show the progress bar. Defaults to False.

    Returns:
        np.ndarray: time points.
        np.ndarray: samples from the Euler-Maruyama method.
            (n_steps, *x0.shape) shape array.

    NOTE:
        the shape of an output of f and g must be the same as x0.
    """  # noqa
    ts = np.linspace(t0, t1, n_steps)
    dt = ts[1] - ts[0]

    x0dim = x0.ndim
    assert x0dim in (1, 2), f"x0 must be 1D or 2D array. But x0.ndim = {x0.ndim}"  # noqa
    if x0dim == 1:
        x0 = x0.reshape(1, -1)

    xs = np.zeros((n_steps,)+x0.shape)
    xs[0] = x0
    for k in (trange if verbose else range)(1, n_steps):
        xs[k] = xs[k-1]
        xs[k] += f(xs[k-1], ts[k-1]) * dt
        xs[k] += g(xs[k-1], ts[k-1]) * np.sqrt(abs(dt)) * np.random.randn(*x0.shape)  # noqa

    assert xs.shape == (n_steps, *x0.shape), "Internal Error"
    if x0dim == 1:
        return ts, xs.reshape(n_steps, -1)
    else:
        return ts, xs
