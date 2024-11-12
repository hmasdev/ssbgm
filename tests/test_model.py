from itertools import product
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from ssbgm.model import (
    create_noised_data,
    ScoreBasedGenerator,
)

# TODO: test whether calling ScoreBasedGenerator.sample raises NotFittedError when ScoreBasedGenerator is not fitted  # noqa
# TODO: test whether calling ScoreBasedGenerator.predict


@pytest.mark.parametrize(
    'X, sigmas, expected_shapes',
    [
        (
            np.array([[1, 2], [3, 4], [5, 6]]),
            [0.1, 0.2],
            (
                (3*2, 2+1),
                (3*2, 2),
                (3*2,),
            ),
        ),
        (
            np.array([1, 2, 3]),
            np.array([0.1, 0.2, 0.3, 0.4]),
            (
                (3*4, 1+1),
                (3*4, 1),
                (3*4,),
            ),
        ),
    ]
)
def test_create_noised_data_wo_conditions(
    X: np.ndarray,
    sigmas: list[float] | np.ndarray,
    expected_shapes: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]  # noqa
) -> None:
    X_, y_, w_ = create_noised_data(X, sigmas)
    assert X_.shape == expected_shapes[0]
    assert y_.shape == expected_shapes[1]
    assert w_.shape == expected_shapes[2]
    np.testing.assert_array_equal(X_[:, -1], np.repeat(sigmas, len(X)))


@pytest.mark.parametrize(
    'X, sigmas, conditions, expected_shapes',
    [
        (
            np.array([[1, 2], [3, 4], [5, 6]]),
            [0.1, 0.2],
            np.array([0, 1, 0]),
            (
                (3*2, 2+1+1),
                (3*2, 2),
                (3*2,),
            ),
        ),
        (
            np.array([1, 2, 3]),
            np.array([0.1, 0.2, 0.3, 0.4]),
            np.array([[1, 2], [3, 4], [5, 6]]),
            (
                (3*4, 1+1+2),
                (3*4, 1),
                (3*4,),
            ),
        ),
    ]
)
def test_create_noised_data_w_conditions(
    X: np.ndarray,
    sigmas: list[float] | np.ndarray,
    conditions: np.ndarray,
    expected_shapes: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]  # noqa
) -> None:
    X_, y_, w_ = create_noised_data(X, sigmas, conditions)
    assert X_.shape == expected_shapes[0]
    assert y_.shape == expected_shapes[1]
    assert w_.shape == expected_shapes[2]
    np.testing.assert_array_equal(X_[:, -1], np.repeat(sigmas, len(X)))


@pytest.mark.parametrize(
    'X,y',
    [
        (
            np.array([[1, 2], [3, 4], [5, 6]]),
            None,
        ),
        (
            np.array([[1, 2], [3, 4], [5, 6]]),
            np.array([1, 2, 3]),
        ),
        (
            np.array([1, 2, 3]),
            None,
        ),
    ]
)
def test_ScoreBasedGenerator_fit(
    X: np.ndarray,
    y: np.ndarray | None
) -> None:
    # TODO: test with sigmas and keep_noised_data kwargs
    ssg = ScoreBasedGenerator(estimator=LinearRegression())
    ssg.fit(X, y)


def test_ScoreBasedGenerator__sample_langevin_montecarlo_wo_conditions() -> None:  # noqa
    n_samples = 128
    alpha = 0.1
    X = np.array([[1, 2], [3, 4], [5, 6]])
    sbm = ScoreBasedGenerator(estimator=LinearRegression())
    sbm.fit(X)
    samples = sbm._sample_langenvin_montecarlo(
        n_samples=n_samples,
        alpha=alpha,
    )
    assert samples.shape == (n_samples, 1, X.shape[1])

    # Without conditions, giving X to sample method raises an error
    with pytest.raises(Exception):
        # FIXME: make it more specific
        sbm._sample_langenvin_montecarlo(
            X,
            n_samples=n_samples,
            alpha=alpha,
        )


def test_ScoreBasedGenerator__sample_langevin_montecarlo_wo_conditions_with_domain_specified() -> None:  # noqa

    minx0 = 1.5
    maxx0 = 4.5
    minx1 = 2
    maxx1 = 5.5

    n_steps = 101
    n_samples = 64
    alpha = 0.1
    X = np.array([[1, 2], [3, 4], [5, 6]])
    sbm = ScoreBasedGenerator(estimator=LinearRegression(), verbose=True)
    sbm.fit(X)
    samples = sbm._sample_langenvin_montecarlo(
        n_samples=n_samples,
        alpha=alpha,
        init_sample=np.array([2, 3]),
        n_steps=n_steps,
        is_in_valid_domain_func=lambda x: ((minx0 <= x[0])*(x[0] <= maxx0)*(minx1 <= x[1])*(x[1] <= maxx1)),  # noqa
    )
    assert samples.shape == (n_samples, 1, X.shape[1])
    samples = samples.reshape(n_samples, X.shape[1])
    assert np.all((minx0 <= samples[:, 0])*(samples[:, 0] <= maxx0)*(minx1 <= samples[:, 1])*(samples[:, 1] <= maxx1))  # noqa

    # Without conditions, giving X to sample method raises an error
    with pytest.raises(Exception):
        # FIXME: make it more specific
        sbm._sample_langenvin_montecarlo(
            X,
            n_samples=n_samples,
            alpha=alpha,
        )

    # with init_sample is out-of-domain
    with pytest.raises(ValueError):
        sbm._sample_langenvin_montecarlo(
            n_samples=n_samples,
            alpha=alpha,
            init_sample=np.array([minx0-1000, minx1-1000]),
            is_in_valid_domain_func=lambda x: ((minx0 <= x[0])*(x[0] <= maxx0)*(minx1 <= x[1])*(x[1] <= maxx1)),  # noqa
        )


@pytest.mark.parametrize(
    'X,y',
    [
        (
            np.array([[1, 2], [3, 4], [5, 6]]),
            np.array([0, 1, 0]),
        ),
        (
            np.array([[1, 2], [3, 4], [5, 6]]),
            np.array([[1, 2], [3, 4], [5, 6]]),
        )
    ]
)
def test_ScoreBasedGenerator__sample_langevin_montecarlo_w_conditions(
    X: np.ndarray,
    y: np.ndarray,
) -> None:
    n_samples = 128
    alpha = 0.1
    sbm = ScoreBasedGenerator(estimator=LinearRegression())
    sbm.fit(X, y)
    samples = sbm._sample_langenvin_montecarlo(
        X,
        n_samples=n_samples,
        alpha=alpha,
    )
    assert samples.shape == (n_samples, X.shape[0], 1 if y.ndim == 1 else y.shape[1])  # noqa

    # With conditions, not giving X to sample method raises an error
    with pytest.raises(Exception):
        # FIXME: make it more specific
        sbm._sample_langenvin_montecarlo(
            n_samples=n_samples,
            alpha=alpha,
        )


def test_ScoreBasedGenerator__sample_euler_wo_conditions() -> None:
    n_samples = 128
    n_steps = 101
    X = np.array([[1, 2], [3, 4], [5, 6]])
    sbm = ScoreBasedGenerator(estimator=LinearRegression())
    sbm.fit(X)

    # return_paths=False
    samples = sbm._sample_euler(
        n_samples=n_samples,
        n_steps=n_steps,
    )
    assert samples.shape == (n_samples, 1, X.shape[1])

    # return_paths=True
    samples = sbm._sample_euler(
        n_samples=n_samples,
        n_steps=n_steps,
        return_paths=True,
    )
    assert samples.shape == (n_steps, n_samples, 1, X.shape[1])

    # Without conditions, giving X to sample method raises an error
    with pytest.raises(Exception):
        # FIXME: make it more specific
        sbm._sample_euler(
            X,
            n_samples=n_samples,
            n_steps=n_steps,
        )


@pytest.mark.parametrize(
    'X,y',
    [
        (
            np.array([[1, 2], [3, 4], [5, 6]]),
            np.array([0, 1, 0]),
        ),
        (
            np.array([[1, 2], [3, 4], [5, 6]]),
            np.array([[1, 2], [3, 4], [5, 6]]),
        )
    ]
)
def test_ScoreBasedGenerator__sample_euler_w_conditions(
    X: np.ndarray,
    y: np.ndarray,
) -> None:
    n_samples = 128
    n_steps = 101
    sbm = ScoreBasedGenerator(estimator=LinearRegression())
    sbm.fit(X, y)

    # return_paths=False
    samples = sbm._sample_euler(
        X,
        n_samples=n_samples,
        n_steps=n_steps,
    )
    assert samples.shape == (n_samples, X.shape[0], 1 if y.ndim == 1 else y.shape[1])  # noqa

    # return_paths=True
    samples = sbm._sample_euler(
        X,
        n_samples=n_samples,
        n_steps=n_steps,
        return_paths=True,
    )
    assert samples.shape == (n_steps, n_samples, X.shape[0], 1 if y.ndim == 1 else y.shape[1])  # noqa

    # With conditions, not giving X to sample method raises an error
    with pytest.raises(Exception):
        # FIXME: make it more specific
        sbm._sample_euler(
            n_samples=n_samples,
            n_steps=n_steps,
            return_paths=True,
        )


def test_ScoreBasedGenerator__sample_euler_maruyama_wo_conditions() -> None:
    n_samples = 128
    n_steps = 101
    X = np.array([[1, 2], [3, 4], [5, 6]])
    sbm = ScoreBasedGenerator(estimator=LinearRegression())
    sbm.fit(X)

    # return_paths=False
    samples = sbm._sample_euler_maruyama(
        n_samples=n_samples,
        n_steps=n_steps,
    )
    assert samples.shape == (n_samples, 1, X.shape[1])

    # return_paths=True
    samples = sbm._sample_euler_maruyama(
        n_samples=n_samples,
        n_steps=n_steps,
        return_paths=True,
    )
    assert samples.shape == (n_steps, n_samples, 1, X.shape[1])

    # Without conditions, giving X to sample method raises an error
    with pytest.raises(Exception):
        sbm._sample_euler_maruyama(
            X,
            n_samples=n_samples,
            n_steps=n_steps,
        )


@pytest.mark.parametrize(
    'X,y',
    [
        (
            np.array([[1, 2], [3, 4], [5, 6]]),
            np.array([0, 1, 0]),
        ),
        (
            np.array([[1, 2], [3, 4], [5, 6]]),
            np.array([[1, 2], [3, 4], [5, 6]]),
        )
    ]
)
def test_ScoreBasedGenerator__sample_euler_maruyama_w_conditions(
    X: np.ndarray,
    y: np.ndarray,
) -> None:
    n_samples = 128
    n_steps = 101
    sbm = ScoreBasedGenerator(estimator=LinearRegression())
    sbm.fit(X, y)

    # return_paths=False
    samples = sbm._sample_euler_maruyama(
        X,
        n_samples=n_samples,
        n_steps=n_steps,
    )
    assert samples.shape == (n_samples, X.shape[0], 1 if y.ndim == 1 else y.shape[1])  # noqa

    # return_paths=True
    samples = sbm._sample_euler_maruyama(
        X,
        n_samples=n_samples,
        n_steps=n_steps,
        return_paths=True,
    )
    assert samples.shape == (n_steps, n_samples, X.shape[0], 1 if y.ndim == 1 else y.shape[1])  # noqa

    # With conditions, not giving X to sample method raises an error
    with pytest.raises(Exception):
        sbm._sample_euler_maruyama(
            n_samples=n_samples,
            n_steps=n_steps,
        )


@pytest.mark.parametrize(
    'X,sample_method,kwargs',
    [
        (X, sample_method, kwargs)
        for X, (sample_method, kwargs) in product(
            [
                np.array([[1, 2], [3, 4], [5, 6]]),
                np.array([1, 2, 3]),
            ],
            [
                (
                    ScoreBasedGenerator.SamplingMethod.LANGEVIN_MONTECARLO,
                    dict(alpha=0.1),
                ),
                (
                    ScoreBasedGenerator.SamplingMethod.LANGEVIN_MONTECARLO,
                    dict(alpha=0.1, init_sample=True),
                    # NOTE: init_sample will be replaced with an array
                ),
                (
                    ScoreBasedGenerator.SamplingMethod.EULER,
                    dict(n_steps=101),
                ),
                (
                    ScoreBasedGenerator.SamplingMethod.EULER_MARUYAMA,
                    dict(n_steps=101),
                )
            ],
        )
    ],
)
def test_sample_wo_conditions(
    X: np.ndarray,
    sample_method: ScoreBasedGenerator.SamplingMethod,
    kwargs: dict[str, int | float]
) -> None:

    if kwargs.get('init_sample'):
        kwargs = kwargs.copy()  # NOTE: kwargs is shared among parameters
        kwargs['init_sample'] = np.arange(1 if X.ndim == 1 else X.shape[1])  # type: ignore # noqa

    n_samples = 128
    sbm = ScoreBasedGenerator(estimator=LinearRegression())
    sbm.fit(X)
    samples = sbm.sample(sampling_method=sample_method, n_samples=n_samples, **kwargs)  # type: ignore # noqa
    assert samples.shape == (n_samples, 1, 1 if X.ndim == 1 else X.shape[1])


@pytest.mark.parametrize(
    'X,y,sample_method,kwargs',
    [
        (X, y, sample_method, kwargs)
        for (X, y), (sample_method, kwargs) in product(
            # X, y
            [
                (
                    np.array([[1, 2], [3, 4], [5, 6]]),
                    np.array([0, 1, 0]),
                ),
                (
                    np.array([[1, 2], [3, 4], [5, 6]]),
                    np.array([[1, 2], [3, 4], [5, 6]]),
                )
            ],
            # sample_method and kwargs
            [
                (
                    ScoreBasedGenerator.SamplingMethod.LANGEVIN_MONTECARLO,
                    dict(alpha=0.1),
                ),
                (
                    ScoreBasedGenerator.SamplingMethod.LANGEVIN_MONTECARLO,
                    dict(alpha=0.1, init_sample=True),
                    # NOTE: init_sample will be replaced with an array
                ),
                (
                    ScoreBasedGenerator.SamplingMethod.EULER,
                    dict(n_steps=101),
                ),
                (
                    ScoreBasedGenerator.SamplingMethod.EULER_MARUYAMA,
                    dict(n_steps=101),
                )
            ],
        )
    ],
)
def test_sample_w_conditions(
    X: np.ndarray,
    y: np.ndarray,
    sample_method: ScoreBasedGenerator.SamplingMethod,
    kwargs: dict[str, int | float]
) -> None:

    if kwargs.get('init_sample'):
        kwargs = kwargs.copy()
        kwargs['init_sample'] = np.arange(1 if y.ndim == 1 else y.shape[1])  # type: ignore # noqa

    n_samples = 128
    sbm = ScoreBasedGenerator(estimator=LinearRegression())
    sbm.fit(X, y)
    samples = sbm.sample(X, sampling_method=sample_method, n_samples=n_samples, **kwargs)  # type: ignore # noqa
    assert samples.shape == (n_samples, X.shape[0], 1 if y.ndim == 1 else y.shape[1])  # noqa


def test_sample_with_invalid_sampling_method() -> None:
    X = np.array([[1, 2], [3, 4], [5, 6]])
    sbm = ScoreBasedGenerator(estimator=LinearRegression())
    sbm.fit(X)

    with pytest.raises(ValueError):
        sbm.sample(sampling_method='invalid')  # type: ignore
