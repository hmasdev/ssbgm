from enum import Enum
from functools import partial
from typing import Callable, Iterable, Literal, overload, TypeVar
import numpy as np
import sklearn
from sklearn.base import (
    BaseEstimator,
    check_array,
    check_is_fitted,
    check_X_y,
)
from .math import (
    euler,
    euler_maruyama,
    langevin_montecarlo,
)
from .utils import np_seed

TScoreBasedGenerator = TypeVar('TScoreBasedGenerator', bound='ScoreBasedGenerator')  # noqa


def create_noised_data(
    X: np.ndarray,
    sigmas: Iterable[float],
    conditions: np.ndarray | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create noised data.

    Args:
        X (np.ndarray): features. Shape: (N,) or (N, D).
        sigmas (Iterable[float]): noise strengths.
        conditions (np.ndarray | None, optional): conditions. Defaults to None.
            If conditions is not None, shape of conditions must be (N, M).
        seed (int | None, optional): random seed. Defaults to None.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: noised features, noised targets, and weights.
            noised features:
                (N*n_sigmas, D+1) shape array if X shape is (N, D)
                (N*n_sigmas, 2) shape array if X shape is (N,)
            noised targets:
                (N*n_sigmas, 1) shape array if X shape is (N,)
                (N*n_sigmas, D) shape array if X shape is (N, D)
            weights: (N*n_sigmas,) shape array.

    NOTE:
        weights are the square of noise strengths.
    """  # noqa

    with np_seed(seed):
        X = X if X.ndim > 1 else X.reshape(-1, 1)
        sigmas = sorted(sigmas)
        X_, y_, w_ = [], [], []

        for sigma in sigmas:
            # generate noise
            noise = np.random.randn(*X.shape) * sigma
            # define the alternative target
            y_.append(-noise/sigma**2)
            # define the features
            if conditions is not None:
                X_.append(np.hstack([
                    conditions.reshape(X.shape[0], -1),
                    X + noise,
                    np.array([[sigma]]*len(X)),
                ]))
            else:
                X_.append(np.hstack([
                    X + noise,
                    np.array([[sigma]]*len(X))
                ]))
            # define the weights
            w_.append(np.array([sigma**2]*len(X)))

        return np.vstack(X_), np.vstack(y_), np.hstack(w_)


class ScoreBasedGenerator(BaseEstimator):
    '''Score-based generator

    This model learns the score function of the target distribution from training data.

    - learn p'(x)/p(x) //ssif a training dataset \\{x_n\\}_{n=1}^N is given
    - learn p'(y|x)/p(y|x) //ssif a training dataset \\{(x_n, y_n)\\}_{n=1}^N is given
    '''  # noqa

    class SamplingMethod(Enum):
        LANGEVIN_MONTECARLO: str = 'langevin_montecarlo'
        EULER: str = 'euler'
        EULER_MARUYAMA: str = 'euler_maruyama'

    def __init__(
        self,
        estimator: BaseEstimator,
        random_state: int = 0,
        verbose: bool = False,
    ):
        self.estimator = estimator
        self.random_state = random_state
        self.verbose = verbose

        self._large_value: float = 1e12

    def fit(
        self: TScoreBasedGenerator,
        X: np.ndarray,
        y: np.ndarray | None = None,
        *,
        noise_strengths: Iterable[float] | None = None,
        keep_noised_data: bool = False,
    ) -> TScoreBasedGenerator:
        r"""Train the score-function

        Args:
            X (np.ndarray): Generated data, or conditions if y is given. Shape: (N, M or n_outputs).
            y (np.ndarray | None, optional): Generated data given X. Defaults to None. Shape: (N, n_outputs) if given.
            noise_strengths (Iterable[float] | None, optional): noise strengths. Defaults to None.
                If noise_strengths is None, noise_strengths is set to np.sqrt(np.logspace(-3, {OBSERVED STD}, 11))  # noqa
            keep_noised_data (bool, optional): flag to keep noised data. Defaults to False.

        Returns:
            Self: trained model.
'
        NOTE:
            if y is not given, model learns the score function of X: $\partial log p(X) / \partial X$
            if y is given, model learns the score function of y given X: $\partial log p(y|X) / \partial y$
        """  # noqa

        if y is None:
            # if y is not given, model learns the score function of X
            X, y = None, check_array(X, ensure_2d=False)  # type: ignore
            assert y is not None
        else:
            # if y is given, model learns the score function of y given X
            X, y = check_X_y(X, y, multi_output=True)
            assert y is not None

        # preprocess noise_strengths
        if noise_strengths is None:
            noise_strengths = np.sqrt(np.logspace(-3, y.var(axis=0).max(), 11))  # noqa

        self.noise_strengths_ = noise_strengths
        self.n_outputs_ = y.shape[1] if y.ndim > 1 else 1

        X_, y_, w_ = create_noised_data(y, noise_strengths, conditions=X, seed=self.random_state)  # noqa
        self.X_ = X_ if keep_noised_data else None
        self.y_ = y_ if keep_noised_data else None
        self.w_ = w_ if keep_noised_data else None

        self.estimator_ = sklearn.clone(self.estimator)
        self.estimator_.fit(
            X_,
            y_ if self.n_outputs_ > 1 else y_.flatten(),
            sample_weight=w_.flatten(),
        )

        return self

    @overload
    def predict(
        self,
        X: np.ndarray | None = None,
        *,
        aggregate: Literal['mean', 'median'] = 'mean',
        return_std: Literal[False] = False,
        **kwargs,
    ) -> np.ndarray:
        ...

    @overload
    def predict(
        self,
        X: np.ndarray | None = None,
        *,
        aggregate: Literal['mean', 'median'] = 'mean',
        return_std: Literal[True] = True,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        ...

    def predict(
        self,
        X: np.ndarray | None = None,
        *,
        aggregate: Literal['mean', 'median'] = 'mean',
        return_std: bool = False,
        **kwargs,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Predict the mean and standard deviation of the generated data.

        Args:
            X (np.ndarray | None, optional): conditions. Defaults to None. Shape: (N, M) if given.
            aggregate (Literal['mean', 'median'], optional): aggregation method. Defaults to 'mean'.
            return_std (bool, optional): flag to return standard deviation. Defaults to False.

        Returns:
            np.ndarray | tuple[np.ndarray, np.ndarray]:
                - mean of the generated data if return_std is False.
                    - (N, n_outputs), or (N,) if n_outputs is 1.
                - mean and standard deviation of the generated data if return_std is True.
                    - a pair of 2 arrays with (N, n_outputs), or (N,) if n_outputs is 1.

        NOTE:
            the output is squeezed if the number of outputs is 1.
        """  # noqa
        agg_func = np.mean if aggregate == 'mean' else np.median

        # samples: (n_samples, N, n_outputs)
        samples = self.sample(X, **(kwargs | {'return_paths': False}))

        if return_std:
            return (
                agg_func(samples, axis=0).squeeze(),
                np.std(samples, axis=0).squeeze(),
            )  # type: ignore
        else:
            # (N, n_outputs) or (n_outputs,)
            return agg_func(samples, axis=0).squeeze()  # type: ignore

    def _sample_langenvin_montecarlo(
        self,
        X: np.ndarray | None = None,
        *,
        init_sample: np.ndarray | None = None,
        n_samples: int = 1000,
        n_steps: int = 1000,
        alpha: float = 0.1,
        sigma: float | None = None,
        return_paths: bool = False,
        is_in_valid_domain_func: Callable[[np.ndarray], bool] | None = None,
    ) -> np.ndarray:
        """Generate samples from the Langevin Monte Carlo algorithm.

        Args:
            X (np.ndarray | None, optional): conditions. Defaults to None.
                Shape: (N, M) if it is not None.
            init_sample (np.ndarray | None, optional): initial sample. Defaults to None.
                (n_outputs,) shape array if it is not None.
            n_samples (int, optional): number of samples. Defaults to 1000.
            n_steps (int, optional): number of steps. Defaults to 1000.
            alpha (float, optional): time step size of the Langevin Monte Carlo algorithm. Defaults to 0.1.
            sigma (float | None, optional): noise strength. Defaults to None.
            return_paths (bool, optional): flag to return paths. Defaults to False.
            is_in_valid_domain_func (Callable[[np.ndarray], bool] | None, optional): function to check whether the sample is in the valid domain. Defaults to None.
                When is_in_valid_domain_func is given, _sample_langenvin_montecarlo draws samples with Metropolis-adjusted Langevin algorithm in stead of Langevin Monte Carlo algorithm.

        Returns:
            np.ndarray: samples from the Langevin Monte Carlo algorithm.
                (n_steps, n_samples, N, n_outputs) shape array if return_paths is True.
                (n_samples, N, n_outputs) shape array if return_paths is False.
        """  # noqa

        # validation
        if init_sample is not None:
            assert init_sample.ndim == 1 and init_sample.size == self.n_outputs_, f'init_sample must be (n_outputs,) shape array. But init_sample.shape = {init_sample.shape}'  # noqa

        if X is None:
            # x: (n_samples, n_outputs)
            if init_sample is not None:
                x0 = np.vstack([init_sample.reshape(1, self.n_outputs_)]*n_samples)  # noqa
            else:
                x0 = np.random.randn(n_samples, self.n_outputs_) * max(self.noise_strengths_)  # noqa

            def dU(x, sigma):
                return - self.estimator_.predict(np.hstack([x, np.array([[sigma]]*len(x))])).reshape(*x.shape)  # noqa
        else:
            # x: (n_samples * N, n_outputs)
            if init_sample is not None:
                if init_sample.size == self.n_outputs_:
                    x0 = np.vstack([init_sample.reshape(1, self.n_outputs_)]*X.shape[0]*n_samples)  # noqa
                else:
                    assert init_sample.shape == (X.shape[0], self.n_outputs_)
                    x0 = np.repeat(init_sample, n_samples, axis=0)
            else:
                x0 = np.random.randn(n_samples * X.shape[0], self.n_outputs_)*max(self.noise_strengths_)  # noqa

            X = np.repeat(X, n_samples, axis=0)

            def dU(x, sigma):
                return - self.estimator_.predict(np.hstack([X, x, np.array([[sigma]]*len(x))])).reshape(*x.shape)  # noqa

        if sigma is None:
            for sigma in sorted(self.noise_strengths_)[::-1]:
                # NOTE: decrease the noise strength step by step
                x0 = langevin_montecarlo(
                    x0=x0,
                    nabla_U=partial(dU, sigma=sigma),
                    n_steps=n_steps,
                    delta_t=alpha,
                    pdf=(lambda x: self._large_value * is_in_valid_domain_func(x)) if is_in_valid_domain_func is not None else None,  # noqa
                    # NOTE:
                    # is_in_valid_domain_func is not a valid probability density function.  # noqa
                    # But is_in_valid_domain_func is used for the purpose of filtering out invalid values  # noqa
                    # FIXME:
                    # self.large_value is multiplied to pdf to avoid unexpected rejection.  # noqa
                    # MALA use the min(1, (pdf(z_k^l) q(x_{k-1}|z_k^l)) / (pdf(x_{k-1}) q(z_k^l|x_{k-1}))).  # noqa
                    # So, there is a pair of (z_k^l, x_{k-1}) that both are in the valid domain but z_k^l is rejected.  # noqa
                    # To avoid this, the large value is multiplied to pdf.
                    use_pdf_as_domain_indicator=True,
                    verbose=self.verbose,
                )[-1]

        # Output: (n_samples, N, n_outputs)
        paths = langevin_montecarlo(
            x0=x0,
            nabla_U=partial(dU, sigma=sigma),
            n_steps=n_steps,
            delta_t=alpha,
            pdf=(lambda x: self._large_value * is_in_valid_domain_func(x)) if is_in_valid_domain_func is not None else None,  # noqa
            # NOTE:
            # is_in_valid_domain_func is not a valid probability density function.  # noqa
            # But is_in_valid_domain_func is used for the purpose of filtering out invalid values  # noqa
            # FIXME:
            # self.large_value is multiplied to pdf to avoid unexpected rejection.  # noqa
            # MALA use the min(1, (pdf(z_k^l) q(x_{k-1}|z_k^l)) / (pdf(x_{k-1}) q(z_k^l|x_{k-1}))).  # noqa
            # So, there is a pair of (z_k^l, x_{k-1}) that both are in the valid domain but z_k^l is rejected.  # noqa
            # To avoid this, the large value is multiplied to pdf.
            use_pdf_as_domain_indicator=True,
            verbose=self.verbose,
        ).reshape(n_steps, n_samples, -1, self.n_outputs_)

        # Output: (n_steps, n_samples, N, n_outputs) if return_paths else (n_samples, N, n_outputs)  # noqa
        return paths if return_paths else paths[-1]

    def _sample_euler(
        self,
        X: np.ndarray | None = None,
        *,
        n_samples: int = 1000,
        n_steps: int = 1000,
        return_paths: bool = False,
    ) -> np.ndarray:
        """Generate samples from the Euler method.

        Args:
            X (np.ndarray | None, optional): conditions. Defaults to None.
                Shape: (N, M) if it is not None.
            n_samples (int, optional): number of samples. Defaults to 1000.
            n_steps (int, optional): number of steps. Defaults to 1000.
            return_paths (bool, optional): flag to return paths. Defaults to False.

        Returns:
            np.ndarray: samples from the Euler method.
                (n_steps, n_samples, N, n_outputs) shape array if return_paths is True.
                (n_samples, N, n_outputs) shape array if return_paths is False.
        """  # noqa
        if X is None:
            # x: (n_samples, n_outputs)
            x0 = np.random.randn(n_samples, self.n_outputs_) * max(self.noise_strengths_)  # noqa
            N = 1

            def f(x, t):
                return - 0.5 * self.estimator_.predict(np.hstack([x, np.array([[np.sqrt(t)]]*len(x))])).reshape(*x.shape)  # noqa
        else:
            # x: (n_samples * N, n_outputs)
            x0 = np.random.randn(n_samples * X.shape[0], self.n_outputs_)*max(self.noise_strengths_)  # noqa
            N = X.shape[0]
            X = np.repeat(X, n_samples, axis=0)

            def f(x, t):
                return - 0.5 * self.estimator_.predict(np.hstack([X, x, np.array([[np.sqrt(t)]]*len(x))])).reshape(*x.shape)  # noqa

        paths = euler(
            x0=x0,
            f=f,
            t0=max(self.noise_strengths_)**2,
            t1=0,
            n_steps=n_steps,
            verbose=self.verbose,
        )[1].reshape(n_steps, n_samples, N, self.n_outputs_)

        # Output: (n_steps, n_samples, N, n_outputs) if return_paths else (n_samples, N, n_outputs)  # noqa
        return paths if return_paths else paths[-1]

    def _sample_euler_maruyama(
        self,
        X: np.ndarray | None = None,
        *,
        n_samples: int = 1000,
        n_steps: int = 1000,
        return_paths: bool = False,
    ) -> np.ndarray:
        """Generate samples from the Euler-Maruyama method.

        Args:
            X (np.ndarray | None, optional): conditions. Defaults to None.
            n_samples (int, optional): number of samples. Defaults to 1000.
            n_steps (int, optional): number of steps. Defaults to 1000.
            return_paths (bool, optional): flag to return paths. Defaults to False.

        Returns:
            np.ndarray: samples from the Euler-Maruyama method.
                (n_step, n_samples, N, n_outputs) shape array if return_paths is True.
                (n_samples, N, n_outputs) shape array if return_paths is False.
        """  # noqa
        if X is None:
            # x: (n_samples, n_outputs)
            x0 = np.random.randn(n_samples, self.n_outputs_) * max(self.noise_strengths_)  # noqa
            N = 1

            def f(x, t):
                return - self.estimator_.predict(np.hstack([x, np.array([[np.sqrt(t)]]*len(x))])).reshape(*x.shape)  # noqa
        else:
            # x: (n_samples * N, n_outputs)
            x0 = np.random.randn(n_samples * X.shape[0], self.n_outputs_)*max(self.noise_strengths_)  # noqa
            N = X.shape[0]
            X = np.repeat(X, n_samples, axis=0)

            def f(x, t):
                return - self.estimator_.predict(np.hstack([X, x, np.array([[np.sqrt(t)]]*len(x))])).reshape(*x.shape)  # noqa

        paths = euler_maruyama(
            x0=x0,
            f=f,
            g=lambda x, t: 1.,  # type: ignore
            t0=max(self.noise_strengths_)**2,
            t1=0,
            n_steps=n_steps,
            verbose=self.verbose,
        )[1].reshape(n_steps, n_samples, N, self.n_outputs_)

        # Output: (n_steps, n_samples, N, n_outputs) if return_paths else (n_samples, N, n_outputs)  # noqa
        return paths if return_paths else paths[-1]

    def sample(
        self,
        X: np.ndarray | None = None,
        *,
        n_samples: int = 1000,
        sampling_method: SamplingMethod = SamplingMethod.LANGEVIN_MONTECARLO,
        n_steps: int = 1000,
        return_paths: bool = False,
        alpha: float = 0.1,  # only for langevin monte carlo
        sigma: float | None = None,  # only for langevin monte carlo
        init_sample: np.ndarray | None = None,  # only for langevin monte carlo
        is_in_valid_domain_func: Callable[[np.ndarray], bool] | None = None,  # only for langevin monte carlo # noqa
        seed: int | None = None,
    ) -> np.ndarray:
        """Generate samples from the score-based generator.

        Args:
            X (np.ndarray | None, optional): conditions. Defaults to None.
            n_samples (int, optional): number of samples. Defaults to 1000.
            sampling_method (SamplingMethod, optional): sampling method. Defaults to SamplingMethod.LANGEVIN_MONTECARLO.
            n_steps (int, optional): number of steps. Defaults to 1000.
            return_paths (bool, optional): flag to return paths. Defaults to False.

            alpha (float, optional): time step size of the Langevin Monte Carlo algorithm. Defaults to 0.1. (NOTE: only for langevin monte carlo)
            sigma (float | None, optional): noise strength. Defaults to None. (NOTE: only for langevin monte carlo)
            init_sample (np.ndarray | None, optional): initial sample. Defaults to None. (n_outputs,) shape array if it is not None.  (NOTE: only for langevin monte carlo)
            is_in_valid_domain_func (Callable[[np.ndarray], bool] | None, optional): function to check whether the sample is in the valid domain. Defaults to None. (NOTE: only for langevin monte carlo)
                When is_in_valid_domain_func is given, _sample_langenvin_montecarlo draws samples with Metropolis-adjusted Langevin algorithm in stead of Langevin Monte Carlo algorithm.

            seed (int, optional): random seed. Defaults to None.

        Returns:
            np.ndarray: samples from the score-based generator.
                (n_step, n_samples, N, n_outputs) shape array if return_paths is True and sampling_method is Euler or Euler-Maruyama.
                (n_samples, N, n_outputs) shape array otherwise.
        """  # noqa

        check_is_fitted(self, 'estimator_')
        check_is_fitted(self, 'noise_strengths_')

        with np_seed(seed):
            if sampling_method == ScoreBasedGenerator.SamplingMethod.LANGEVIN_MONTECARLO:  # noqa
                return self._sample_langenvin_montecarlo(
                    X,
                    n_samples=n_samples,
                    n_steps=n_steps,
                    alpha=alpha,
                    sigma=sigma,
                    init_sample=init_sample,
                    is_in_valid_domain_func=is_in_valid_domain_func,
                    return_paths=return_paths,
                )
            elif sampling_method == ScoreBasedGenerator.SamplingMethod.EULER:  # noqa
                return self._sample_euler(X, n_samples=n_samples, n_steps=n_steps, return_paths=return_paths)  # noqa
            elif sampling_method == ScoreBasedGenerator.SamplingMethod.EULER_MARUYAMA:  # noqa
                return self._sample_euler_maruyama(X, n_samples=n_samples, n_steps=n_steps, return_paths=return_paths)  # noqa
            else:
                raise ValueError(f'Invalid sampling method: {sampling_method}')
