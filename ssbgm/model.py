from enum import Enum
from functools import partial
from typing import Callable, Iterable, Literal, Mapping, overload, TypeVar
import numpy as np
import sklearn
from sklearn.base import BaseEstimator
from sklearn.utils.validation import (
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
        # Ref. https://typing.readthedocs.io/en/latest/spec/enums.html#defining-members  # noqa
        LANGEVIN_MONTECARLO = 'langevin_montecarlo'
        EULER = 'euler'
        EULER_MARUYAMA = 'euler_maruyama'

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
            X, y = None, check_array(X, ensure_2d=False, force_all_finite='allow-nan')  # type: ignore # noqa
            assert y is not None
            self.require_X_ = False
        else:
            # if y is given, model learns the score function of y given X
            X, y = check_X_y(X, y, multi_output=True, force_all_finite='allow-nan')  # type: ignore # noqa
            assert y is not None
            self.require_X_ = True

        # preprocess noise_strengths
        if noise_strengths is None:
            noise_strengths = np.sqrt(np.logspace(-3, np.log10(y.var(axis=0).max()), 11))  # noqa

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

    @overload
    def _sample_langenvin_montecarlo(
        self,
        X: np.ndarray,
        *,
        init_sample: np.ndarray | None = None,
        n_samples: int = 1000,
        n_steps: int = 1000,
        alpha: float = 0.1,
        sigma: Iterable[float] | float | None = None,
        return_paths: bool = False,
        conditioned_by: Mapping[int, bool | int | float | np.ndarray] = {},
        is_in_valid_domain_func: Callable[[np.ndarray], bool] | None = None,
    ) -> np.ndarray:
        ...

    @overload
    def _sample_langenvin_montecarlo(
        self,
        X: None = None,
        *,
        init_sample: np.ndarray | None = None,
        n_samples: int = 1000,
        n_steps: int = 1000,
        alpha: float = 0.1,
        sigma: Iterable[float] | float | None = None,
        return_paths: bool = False,
        conditioned_by: Mapping[int, bool | int | float] = {},
        is_in_valid_domain_func: Callable[[np.ndarray], bool] | None = None,
    ) -> np.ndarray:
        ...

    def _sample_langenvin_montecarlo(
        self,
        X: np.ndarray | None = None,
        *,
        init_sample: np.ndarray | None = None,
        n_samples: int = 1000,
        n_steps: int = 1000,
        alpha: float = 0.1,
        sigma: Iterable[float] | float | None = None,
        return_paths: bool = False,
        conditioned_by: Mapping[int, bool | int | float | np.ndarray] = {},
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
            conditioned_by (Mapping[int, bool | int | float | np.ndarray], optional): conditions. Defaults to {}.
                The key is the index of the output betwee 0 and n_outputs-1.
                The type of value is bool, int, float, or np.ndarray.
                When the value is bool, int or float, all output samples are conditioned by the value.
                When X is None, the value cannot be np.ndarray.
                When X is not None, the value can be np.ndarray, and then the shape of the value must be (X.shape[0],).
                If the value is np.ndarray, each output samples are conditioned by the corresponding value.
            is_in_valid_domain_func (Callable[[np.ndarray], bool] | None, optional): function to check whether the sample is in the valid domain. Defaults to None.
                When is_in_valid_domain_func is given, _sample_langenvin_montecarlo draws samples with Metropolis-adjusted Langevin algorithm in stead of Langevin Monte Carlo algorithm.

        Returns:
            np.ndarray: samples from the Langevin Monte Carlo algorithm.
                (n_steps, n_samples, N, n_outputs) shape array if return_paths is True.
                (n_samples, N, n_outputs) shape array if return_paths is False.
        """  # noqa

        # validation
        self._validate_kwargs_for_sample(
            X=X,
            n_samples=n_samples,
            n_steps=n_steps,
            init_sample=init_sample,
            conditioned_by=conditioned_by,
            return_paths=return_paths,
            alpha=alpha,
            sigma=sigma,
            is_in_valid_domain_func=is_in_valid_domain_func,
        )

        # preparation
        _col2idx = {c: i for i, c in enumerate([c_ for c_ in range(self.n_outputs_) if c_ not in conditioned_by.keys()])}  # noqa
        _vals_of_col2idx = sorted(_col2idx.values())
        x0 = self._initialize_samples(X, n_samples, init_sample, conditioned_by)  # noqa
        conditioned_by_processed = self._preprocess_conditioned_by(X, n_samples, conditioned_by)  # noqa

        # define the gradient of the potential energy function
        if X is None:
            def dU(x, sigma):
                x = self._insert_conditiond_x_to_unconditioned_x(x, conditioned_by_processed, _col2idx)  # noqa
                return - self.estimator_.predict(np.hstack([x, np.array([[sigma]]*len(x))])).reshape(*x.shape)[:, _vals_of_col2idx]  # noqa
        else:
            X = np.repeat(X, n_samples, axis=0)

            def dU(x, sigma):
                x = self._insert_conditiond_x_to_unconditioned_x(x, conditioned_by_processed, _col2idx)  # noqa
                return - self.estimator_.predict(np.hstack([X, x, np.array([[sigma]]*len(x))])).reshape(*x.shape)[:, _vals_of_col2idx]  # noqa

        # decrease the noise strength step by step
        sigmas: Iterable[float]
        if isinstance(sigma, Iterable):
            sigmas = sorted(sigma)[::-1][:-1]
            sigma = min(sigma)
        elif sigma is None:
            sigmas = sorted(self.noise_strengths_)[::-1]
            sigma = min(self.noise_strengths_)
        else:
            sigmas = []
            sigma = float(sigma)
        for sigma_ in sigmas:
            # NOTE: decrease the noise strength step by step
            x0 = langevin_montecarlo(
                x0=x0,
                nabla_U=partial(dU, sigma=sigma_),
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
        )
        paths = self._postprocess_sample_paths(paths, n_steps, n_samples, conditioned_by_processed, _col2idx)  # noqa

        # Output: (n_steps, n_samples, N, n_outputs) if return_paths else (n_samples, N, n_outputs)  # noqa
        return paths if return_paths else paths[-1]

    @overload
    def _sample_euler(
        self,
        X: np.ndarray,
        *,
        n_samples: int = 1000,
        n_steps: int = 1000,
        init_sample: np.ndarray | None = None,
        conditioned_by: Mapping[int, bool | int | float | np.ndarray] = {},
        return_paths: bool = False,
    ) -> np.ndarray:
        ...

    @overload
    def _sample_euler(
        self,
        X: None = None,
        *,
        n_samples: int = 1000,
        n_steps: int = 1000,
        init_sample: np.ndarray | None = None,
        conditioned_by: Mapping[int, bool | int | float] = {},
        return_paths: bool = False,
    ) -> np.ndarray:
        ...

    def _sample_euler(
        self,
        X: np.ndarray | None = None,
        *,
        n_samples: int = 1000,
        n_steps: int = 1000,
        init_sample: np.ndarray | None = None,
        conditioned_by: Mapping[int, bool | int | float | np.ndarray] = {},
        return_paths: bool = False,
    ) -> np.ndarray:
        """Generate samples from the Euler method.

        Args:
            X (np.ndarray | None, optional): conditions. Defaults to None.
                Shape: (N, M) if it is not None.
            n_samples (int, optional): number of samples. Defaults to 1000.
            n_steps (int, optional): number of steps. Defaults to 1000.
            init_sample (np.ndarray | None, optional): initial sample. Defaults to None.
                (n_outputs,) shape array if it is not None.
                NOTE: n_samples should be 1 if init_sample is not None because the same sample paths are generated.
            conditioned_by (Mapping[int, bool | int | float | np.ndarray], optional): conditions. Defaults to {}.
                The key is the index of the output betwee 0 and n_outputs-1.
                The type of value is bool, int, float, or np.ndarray.
                When the value is bool, int or float, all output samples are conditioned by the value.
                When X is None, the value cannot be np.ndarray.
                When X is not None, the value can be np.ndarray, and then the shape of the value must be (X.shape[0],).
            return_paths (bool, optional): flag to return paths. Defaults to False.

        Returns:
            np.ndarray: samples from the Euler method.
                (n_steps, n_samples, N, n_outputs) shape array if return_paths is True.
                (n_samples, N, n_outputs) shape array if return_paths is False.
        """  # noqa
        # validation
        self._validate_kwargs_for_sample(
            X=X,
            n_samples=n_samples,
            n_steps=n_steps,
            init_sample=init_sample,
            conditioned_by=conditioned_by,
            return_paths=return_paths,
        )
        # preparation
        _col2idx = {c: i for i, c in enumerate([c_ for c_ in range(self.n_outputs_) if c_ not in conditioned_by.keys()])}  # noqa
        _vals_of_col2idx = sorted(_col2idx.values())
        x0 = self._initialize_samples(X, n_samples, init_sample, conditioned_by)  # noqa
        conditioned_by_processed = self._preprocess_conditioned_by(X, n_samples, conditioned_by)  # noqa
        # define ODE
        if X is None:
            def f(x, t):
                x = self._insert_conditiond_x_to_unconditioned_x(x, conditioned_by_processed, _col2idx)  # noqa
                return - 0.5 * self.estimator_.predict(np.hstack([x, np.array([[np.sqrt(t)]]*len(x))])).reshape(*x.shape)[:, _vals_of_col2idx]  # noqa
        else:
            X = np.repeat(X, n_samples, axis=0)

            def f(x, t):
                x = self._insert_conditiond_x_to_unconditioned_x(x, conditioned_by_processed, _col2idx)  # noqa
                return - 0.5 * self.estimator_.predict(np.hstack([X, x, np.array([[np.sqrt(t)]]*len(x))])).reshape(*x.shape)[:, _vals_of_col2idx]  # noqa
        # generate samples
        paths = euler(
            x0=x0,
            f=f,
            t0=max(self.noise_strengths_)**2,
            t1=0,
            n_steps=n_steps,
            verbose=self.verbose,
        )[1]
        paths = self._postprocess_sample_paths(paths, n_steps, n_samples, conditioned_by_processed, _col2idx)  # noqa

        # Output: (n_steps, n_samples, N, n_outputs) if return_paths else (n_samples, N, n_outputs)  # noqa
        return paths if return_paths else paths[-1]

    @overload
    def _sample_euler_maruyama(
        self,
        X: np.ndarray,
        *,
        n_samples: int = 1000,
        n_steps: int = 1000,
        init_sample: np.ndarray | None = None,
        conditioned_by: Mapping[int, bool | int | float | np.ndarray] = {},
        return_paths: bool = False,
    ) -> np.ndarray:
        ...

    @overload
    def _sample_euler_maruyama(
        self,
        X: None = None,
        *,
        n_samples: int = 1000,
        n_steps: int = 1000,
        init_sample: np.ndarray | None = None,
        conditioned_by: Mapping[int, bool | int | float | np.ndarray] = {},
        return_paths: bool = False,
    ) -> np.ndarray:
        ...

    def _sample_euler_maruyama(
        self,
        X: np.ndarray | None = None,
        *,
        n_samples: int = 1000,
        n_steps: int = 1000,
        init_sample: np.ndarray | None = None,
        conditioned_by: Mapping[int, bool | int | float | np.ndarray] = {},
        return_paths: bool = False,
    ) -> np.ndarray:
        """Generate samples from the Euler-Maruyama method.

        Args:
            X (np.ndarray | None, optional): conditions. Defaults to None.
            n_samples (int, optional): number of samples. Defaults to 1000.
            n_steps (int, optional): number of steps. Defaults to 1000.
            init_sample (np.ndarray | None, optional): initial sample. Defaults to None.
                (n_outputs,) shape array if it is not None.
                NOTE: n_samples should be 1 if init_sample is not None because the same sample paths are generated.
            conditioned_by (Mapping[int, bool | int | float | np.ndarray], optional): conditions. Defaults to {}.
                The key is the index of the output betwee 0 and n_outputs-1.
                The type of value is bool, int, float, or np.ndarray.
                When the value is bool, int or float, all output samples are conditioned by the value.
                When X is None, the value cannot be np.ndarray.
                When X is not None, the value can be np.ndarray, and then the shape of the value must be (X.shape
            return_paths (bool, optional): flag to return paths. Defaults to False.

        Returns:
            np.ndarray: samples from the Euler-Maruyama method.
                (n_step, n_samples, N, n_outputs) shape array if return_paths is True.
                (n_samples, N, n_outputs) shape array if return_paths is False.
        """  # noqa
        # validation
        self._validate_kwargs_for_sample(
            X=X,
            n_samples=n_samples,
            n_steps=n_steps,
            init_sample=init_sample,
            conditioned_by=conditioned_by,
            return_paths=return_paths,
        )
        # preparation
        _col2idx = {c: i for i, c in enumerate([c_ for c_ in range(self.n_outputs_) if c_ not in conditioned_by.keys()])}  # noqa
        _vals_of_col2idx = sorted(_col2idx.values())
        x0 = self._initialize_samples(X, n_samples, init_sample, conditioned_by)  # noqa
        conditioned_by_processed = self._preprocess_conditioned_by(X, n_samples, conditioned_by)  # noqa
        # define SDE
        if X is None:
            def f(x, t):
                x = self._insert_conditiond_x_to_unconditioned_x(x, conditioned_by_processed, _col2idx)  # noqa
                return - self.estimator_.predict(np.hstack([x, np.array([[np.sqrt(t)]]*len(x))])).reshape(*x.shape)[:, _vals_of_col2idx]  # noqa
        else:
            X = np.repeat(X, n_samples, axis=0)

            def f(x, t):
                x = self._insert_conditiond_x_to_unconditioned_x(x, conditioned_by_processed, _col2idx)  # noqa
                return - self.estimator_.predict(np.hstack([X, x, np.array([[np.sqrt(t)]]*len(x))])).reshape(*x.shape)[:, _vals_of_col2idx]  # noqa
        # generate samples
        paths = euler_maruyama(
            x0=x0,
            f=f,
            g=lambda x, t: 1.,  # type: ignore
            t0=max(self.noise_strengths_)**2,
            t1=0,
            n_steps=n_steps,
            verbose=self.verbose,
        )[1]
        paths = self._postprocess_sample_paths(paths, n_steps, n_samples, conditioned_by_processed, _col2idx)  # noqa

        # Output: (n_steps, n_samples, N, n_outputs) if return_paths else (n_samples, N, n_outputs)  # noqa
        return paths if return_paths else paths[-1]

    @overload
    def sample(
        self,
        X: np.ndarray,
        *,
        n_samples: int = 1000,
        sampling_method: SamplingMethod = SamplingMethod.LANGEVIN_MONTECARLO,
        n_steps: int = 1000,
        return_paths: bool = False,
        init_sample: np.ndarray | None = None,
        conditioned_by: Mapping[int, bool | int | float | np.ndarray] = {},
        alpha: float = 0.1,  # only for langevin monte carlo
        sigma: Iterable[float] | float | None = None,  # only for langevin monte carlo  # noqa
        is_in_valid_domain_func: Callable[[np.ndarray], bool] | None = None,  # only for langevin monte carlo # noqa
        seed: int | None = None,
    ) -> np.ndarray:
        ...

    @overload
    def sample(
        self,
        X: None = None,
        *,
        n_samples: int = 1000,
        sampling_method: SamplingMethod = SamplingMethod.LANGEVIN_MONTECARLO,
        n_steps: int = 1000,
        return_paths: bool = False,
        init_sample: np.ndarray | None = None,
        conditioned_by: Mapping[int, bool | int | float] = {},
        alpha: float = 0.1,  # only for langevin monte carlo
        sigma: Iterable[float] | float | None = None,  # only for langevin monte carlo  # noqa
        is_in_valid_domain_func: Callable[[np.ndarray], bool] | None = None,  # only for langevin monte carlo # noqa
        seed: int | None = None,
    ) -> np.ndarray:
        ...

    def sample(
        self,
        X: np.ndarray | None = None,
        *,
        n_samples: int = 1000,
        sampling_method: SamplingMethod = SamplingMethod.LANGEVIN_MONTECARLO,
        n_steps: int = 1000,
        return_paths: bool = False,
        init_sample: np.ndarray | None = None,
        conditioned_by: Mapping[int, bool | int | float | np.ndarray] = {},
        alpha: float = 0.1,  # only for langevin monte carlo
        sigma: Iterable[float] | float | None = None,  # only for langevin monte carlo  # noqa
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
            init_sample (np.ndarray | None, optional): initial sample. Defaults to None. (n_outputs,) shape array if it is not None.  (NOTE: only for langevin monte carlo)
            conditioned_by (Mapping[int, bool | int | float | np.ndarray], optional): conditions. Defaults to {}.  (NOTE: only for euler and euler-maruyama)
                The key is the index of the output betwee 0 and n_outputs-1.
                The type of value is bool, int, float, or np.ndarray.
                When the value is bool, int or float, all output samples are conditioned by the value.
                When X is None, the value cannot be np.ndarray.
                When X is not None, the value can be np.ndarray, and then the shape of the value must be (X.shape[0],).

            alpha (float, optional): time step size of the Langevin Monte Carlo algorithm. Defaults to 0.1. (NOTE: only for langevin monte carlo)
            sigma (float | None, optional): noise strength. Defaults to None. (NOTE: only for langevin monte carlo)
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
                    X,  # type: ignore
                    n_samples=n_samples,
                    n_steps=n_steps,
                    alpha=alpha,
                    sigma=sigma,
                    init_sample=init_sample,
                    is_in_valid_domain_func=is_in_valid_domain_func,
                    conditioned_by=conditioned_by,
                    return_paths=return_paths,
                )
            elif sampling_method == ScoreBasedGenerator.SamplingMethod.EULER:  # noqa
                return self._sample_euler(
                    X,  # type: ignore
                    n_samples=n_samples,
                    n_steps=n_steps,
                    init_sample=init_sample,
                    conditioned_by=conditioned_by,
                    return_paths=return_paths,
                )
            elif sampling_method == ScoreBasedGenerator.SamplingMethod.EULER_MARUYAMA:  # noqa
                return self._sample_euler_maruyama(
                    X,  # type: ignore
                    n_samples=n_samples,
                    n_steps=n_steps,
                    init_sample=init_sample,
                    conditioned_by=conditioned_by,
                    return_paths=return_paths,
                )
            else:
                raise ValueError(f'Invalid sampling method: {sampling_method}')

    def predict_score(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        sigma: int | float | np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict the score function of the target distribution.

        Args:
            X (np.ndarray): Generated data, or conditions if y is given. Shape: (N, M or n_outputs).
            y (np.ndarray | None, optional): Generated data given X. Defaults to None. Shape: (N, n_outputs) if given.
            sigma (int | float | np.ndarray | None, optional): noise strength. Defaults to None.
                If sigma is None, sigma is automatically set to min(self.noise_strengths_).

        Returns:
            np.ndarray: score function of the target distribution.
                shape is the same as the input X if y is not given. Otherwise, shape is the same as the input y.
        """  # noqa

        # validation
        check_is_fitted(self, 'estimator_')

        # preprocess X and y
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if y is None:
            # if y is not given, model learns the score function of X
            X = check_array(X, ensure_2d=False, force_all_finite='allow-nan')  # type: ignore # noqa
        else:
            # if y is given, model learns the score function of y given X
            X, y = check_X_y(X, y, multi_output=True, force_all_finite='allow-nan')  # type: ignore # noqa
            assert y is not None
            if y.ndim == 1:
                y = y.reshape(-1, 1)

        if sigma is None:
            sigma = min(self.noise_strengths_)
        if not isinstance(sigma, np.ndarray):
            sigma = np.array([[sigma]]*len(X))

        # predict the score function
        X_ = np.hstack([X,]+([] if y is None else [y])+[sigma.reshape(-1, 1)])
        return self.estimator_.predict(X_)  # type: ignore

    # Utilities

    def _validate_kwargs_for_sample(
        self,
        X: np.ndarray | None = None,
        *,
        n_samples: int = 1000,
        sampling_method: SamplingMethod = SamplingMethod.LANGEVIN_MONTECARLO,
        n_steps: int = 1000,
        return_paths: bool = False,
        init_sample: np.ndarray | None = None,
        conditioned_by: Mapping[int, bool | int | float | np.ndarray] = {},
        alpha: float = 0.1,  # only for langevin monte carlo
        sigma: Iterable[float] | float | None = None,  # only for langevin monte carlo  # noqa
        is_in_valid_domain_func: Callable[[np.ndarray], bool] | None = None,  # only for langevin monte carlo # noqa
    ) -> None:
        # check X is valid
        if self.require_X_ and X is None:
            raise TypeError('X must be given because requires_X_ is True.')

        # check n_samples
        if n_samples <= 0:
            raise ValueError(f'n_samples must be positive but n_samples = {n_samples}')  # noqa

        # check sampling_method
        # Nothing

        # check n_steps
        if n_steps <= 0:
            raise ValueError(f'n_steps must be positive but n_steps = {n_steps}')  # noqa

        # check return_paths
        # Nothing

        # check conditioned_by
        if len(conditioned_by) >= self.n_outputs_:
            raise ValueError(f'len(conditioned_by) must be less than n_outputs but len(conditioned_by) = {len(conditioned_by)} for n_outputs = {self.n_outputs_}.')  # noqa
        if not all([0 <= k < self.n_outputs_ for k in conditioned_by.keys()]):
            raise KeyError(f'the key of conditioned_by must be between 0 and n_outputs-1. But conditioned_by.keys() = {conditioned_by.keys()}')  # noqa
        if X is None and not all([not isinstance(v, np.ndarray) for v in conditioned_by.values()]):  # noqa
            raise TypeError(f'the value of conditioned_by must be bool, int, or float when X is None but conditioned_by = {conditioned_by}')  # noqa
        elif X is not None and not all([v.shape == (X.shape[0],) for v in conditioned_by.values() if isinstance(v, np.ndarray)]):    # noqa
            raise ValueError(f'the shape of a np.ndarray value of conditioned_by must be (X.shape[0],). But conditioned_by = {conditioned_by}')  # noqa

        # check init_sample
        if init_sample is not None:
            if init_sample.ndim != 1 or init_sample.size != self.n_outputs_:  # noqa
                raise ValueError(f'init_sample must be (n_outputs,) shape array. But init_sample.shape = {init_sample.shape}')  # noqa

        # check alpha
        if alpha <= 0:
            raise ValueError(f'alpha must be positive but alpha = {alpha}')

        # sigma
        if sigma is not None:
            if isinstance(sigma, Iterable) and not all([s > 0 for s in sigma]):
                raise ValueError(f'all values of sigma must be positive but sigma = {sigma}')  # noqa
            if (not isinstance(sigma, Iterable)) and sigma <= 0:
                raise ValueError(f'sigma must be positive but sigma = {sigma}')

    def _initialize_samples(
        self,
        X: np.ndarray | None,
        n_samples: int,
        init_sample: np.ndarray | None,
        conditioned_by: Mapping[int, bool | int | float | np.ndarray],
    ) -> np.ndarray:
        '''Initialize the sample paths, i.e. create the initial samples

        Args:
            X (np.ndarray | None): conditions.
            n_samples (int): number of samples.
            init_sample (np.ndarray | None): initial sample.
            conditioned_by (Mapping[int, bool | int | float | np.ndarray]): conditions of x0.

        Returns:
            np.ndarray: initial samples
                Shape: (n_samples, n_outputs - len(conditioned_by)) if X is None.
                Shape: (n_samples * N, n_outputs - len(conditioned_by)) if X is not None, where N = X.shape[0].

        NOTE:
            the elements of x0 are independently generated from the normal distribution N(0, max(noise_strengths_)^2).  # noqa
        '''  # noqa

        N: int = 1 if X is None else X.shape[0]

        if init_sample is not None:
            x0 = np.vstack([init_sample.reshape(1, self.n_outputs_)]*N*n_samples)  # noqa
        else:
            x0 = np.random.randn(n_samples * N, self.n_outputs_)*max(self.noise_strengths_)  # noqa

        return x0[:, [i for i in range(self.n_outputs_) if i not in conditioned_by.keys()]]  # noqa

    @staticmethod
    def _preprocess_conditioned_by(
        X: np.ndarray | None,
        n_samples: int,
        conditioned_by: Mapping[int, bool | int | float | np.ndarray],
    ) -> Mapping[int, np.ndarray]:
        '''Preprocess the conditioned_by

        Args:
            X (np.ndarray | None): conditions.
            n_samples (int): number of samples.
            conditioned_by (Mapping[int, bool | int | float | np.ndarray]): conditions of x0.

        Returns:
            Mapping[int, np.ndarray]: preprocessed conditioned_by
                The shape of the value is (n_samples, 1) if X is None.
                The shape of the value is (n_samples * N, 1) if X is not None, where N = X.shape[0].
        '''  # noqa
        N = 1 if X is None else X.shape[0]
        preprocessed_conditioned_by = {
            k: (
                np.repeat(v[:, np.newaxis], n_samples, axis=0)
                if isinstance(v, np.ndarray) else
                np.array([[v]]*N*n_samples)
            )
            for k, v in conditioned_by.items()
        }
        return preprocessed_conditioned_by

    def _insert_conditiond_x_to_unconditioned_x(
        self,
        x: np.ndarray,
        conditioned_by_processed: Mapping[int, np.ndarray],
        map_dim_in_output_2_dim_in_x: Mapping[int, int] | None = None,
    ):
        '''Insert conditioned x to unconditioned x

        Args:
            x (np.ndarray): unconditioned x.
                Shape: (*, n_outputs - len(conditioned_by_processed)).
            conditioned_by_processed (Mapping[int, np.ndarray]): conditioned x.
                The shape of the value is (*, 1).
            map_dim_in_output_2_dim_in_x (Mapping[int, int] | None): mapping from the dimension of the output to the dimension of x. Defaults to None.

        Returns:
            np.ndarray: x with conditioned x.
                Shape: (*, n_outputs).
        '''  # noqa
        if conditioned_by_processed:
            if map_dim_in_output_2_dim_in_x is None:
                map_dim_in_output_2_dim_in_x = {
                    dio: i
                    for i, dio in enumerate([
                        dio_ for dio_ in range(self.n_outputs_)
                        if dio_ not in conditioned_by_processed
                    ])
                }
            x = np.hstack([
                conditioned_by_processed[c]
                if c in conditioned_by_processed else
                x[:, [map_dim_in_output_2_dim_in_x[c]]]
                for c in range(self.n_outputs_)
            ])
        return x

    def _postprocess_sample_paths(
        self,
        paths: np.ndarray,
        n_steps: int,
        n_samples: int,
        conditioned_by_processed: Mapping[int, np.ndarray],
        map_dim_in_output_2_dim_in_x: Mapping[int, int] | None = None,
    ) -> np.ndarray:
        '''Postprocess the sample paths

        Args:
            paths (np.ndarray): sample paths.
                Shape should be (n_steps, n_samples*N, (n_outputs - len(conditioned_by_processed))).
                At leat, paths.size must be n_steps * n_samples * N * (n_outputs - len(conditioned_by_processed)).
            n_steps (int): number of steps.
            n_samples (int): number of samples.
            conditioned_by_processed (Mapping[int, np.ndarray]): conditioned x.
                The shape of the value is (n_samples * N, 1).
            map_dim_in_output_2_dim_in_x (Mapping[int, int] | None): mapping from the dimension of the output to the dimension of x. Defaults to None.

        Returns:
            np.ndarray: postprocessed sample paths.
                Shape: (n_steps, n_samples, N, n_outputs).
        '''  # noqa
        paths = paths.reshape(n_steps, -1, n_samples, self.n_outputs_ - len(conditioned_by_processed))  # noqa
        # NOTE: The order of (..., -1, n_samples, ...) is based on np.repeat(X, n_samples, axis=0)  # noqa

        if conditioned_by_processed:
            if map_dim_in_output_2_dim_in_x is None:
                map_dim_in_output_2_dim_in_x = {
                    dio: i
                    for i, dio in enumerate([
                        dio_ for dio_ in range(self.n_outputs_)
                        if dio_ not in conditioned_by_processed
                    ])
                }
            # Attach conditioned x to the paths
            paths = np.array([
                np.hstack([
                    conditioned_by_processed[c]
                    if c in conditioned_by_processed else
                    paths[step, :, :, map_dim_in_output_2_dim_in_x[c]].reshape(-1, 1)  # noqa
                    for c in range(self.n_outputs_)
                ])
                for step in range(n_steps)
            ]).reshape(n_steps, -1, n_samples, self.n_outputs_)

        paths = paths.transpose(0, 2, 1, 3)
        # NOTE: transponse it because the structure of the output is easier to use when the shape is (n_steps, n_samples, N, n_outputs)  # noqa

        # validation
        assert paths.shape == (n_steps, n_samples, paths.shape[2], self.n_outputs_), "Internal Error"  # noqa

        return paths
