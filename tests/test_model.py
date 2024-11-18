from itertools import product
from typing import Mapping
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from ssbgm.model import (
    create_noised_data,
    ScoreBasedGenerator,
)
from ssbgm.utils import np_seed

# TODO: test whether calling ScoreBasedGenerator.sample raises NotFittedError when ScoreBasedGenerator is not fitted  # noqa
# TODO: test _postprocess_sample_paths

DEFAULT_N_SAMPLES = 64


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
    # check noise_strengths
    if y is None:
        assert max(ssg.noise_strengths_) == X.std(axis=0).max()
    elif y.ndim == 1:
        assert max(ssg.noise_strengths_) == y.std()
    else:
        assert max(ssg.noise_strengths_) == y.std(axis=0).max()


@pytest.mark.parametrize(
    'X,init_sample,sigma',
    [
        (X, init_sample, sigma)
        for X, init_sample, sigma in product(
            [
                np.array([1, 2, 3]),
                np.array([[1], [2], [3]]),
                np.array([[1, 2], [3, 4], [5, 6]]),
            ],
            [None, np.array([1]), np.array([1, 2])],
            [None, 0.1, [0.1, 0.2]],
        )
        if (
            (init_sample is None)
            or (X.ndim == 1 and len(init_sample) == 1)
            or (X.ndim > 1 and X.shape[1] == len(init_sample))
        )
    ]
)
def test_ScoreBasedGenerator__sample_langevin_montecarlo_wo_conditions(
    X: np.ndarray,
    init_sample: np.ndarray | None,
    sigma: float | None
) -> None:  # noqa
    n_samples = DEFAULT_N_SAMPLES
    alpha = 0.1
    sbm = ScoreBasedGenerator(estimator=LinearRegression())
    sbm.fit(X)
    samples = sbm._sample_langenvin_montecarlo(
        n_samples=n_samples,
        alpha=alpha,
        init_sample=init_sample,
        sigma=sigma,
    )
    assert samples.shape == (n_samples, 1, X.shape[1] if X.ndim > 1 else 1)

    # Without conditions, giving X to sample method raises an error
    with pytest.raises(Exception):
        # FIXME: make it more specific
        sbm._sample_langenvin_montecarlo(
            X,
            n_samples=n_samples,
            alpha=alpha,
        )


def test_ScoreBasedGenerator__sample_langevin_montecarlo_wo_conditions_with_conditioned_by() -> None:  # noqa

    conditioned_by = {0: 3}

    X = np.array([[1, 2], [3, 4], [5, 6]])
    n_samples = DEFAULT_N_SAMPLES
    alpha = 0.1
    sigma = 1e-4
    sbm = ScoreBasedGenerator(estimator=LinearRegression())
    sbm.fit(X)
    samples = sbm._sample_langenvin_montecarlo(
        n_samples=n_samples,
        alpha=alpha,
        sigma=sigma,
        conditioned_by=conditioned_by,
    )
    assert samples.shape == (n_samples, 1, X.shape[1] if X.ndim > 1 else 1)
    assert np.all(samples[:, 0, 0] == conditioned_by[0])  # Check if the first column is conditioned by the value of conditioned_by  # noqa


@pytest.mark.parametrize(
    'conditioned_by,ExpectedException',
    [
        ({0: 1, 1: 2}, ValueError),
        ({3: 2}, KeyError),
    ]
)
def test_ScoreBasedGenerator__sample_langevin_montecarlo_wo_conditions_with_invalid_conditioned_by(  # noqa
    conditioned_by: dict[int, int],
    ExpectedException: type[Exception],
) -> None:
    X = np.array([[1, 2], [3, 4], [5, 6]])
    n_samples = DEFAULT_N_SAMPLES
    alpha = 0.1
    sigma = 1e-4
    sbm = ScoreBasedGenerator(estimator=LinearRegression())
    sbm.fit(X)
    with pytest.raises(ExpectedException):
        sbm._sample_langenvin_montecarlo(
            n_samples=n_samples,
            alpha=alpha,
            sigma=sigma,
            conditioned_by=conditioned_by,
        )


def test_ScoreBasedGenerator__sample_langevin_montecarlo_wo_conditions_with_domain_specified() -> None:  # noqa

    minx0 = 1.5
    maxx0 = 4.5
    minx1 = 2
    maxx1 = 5.5

    n_steps = 101
    n_samples = DEFAULT_N_SAMPLES
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
    'X,y,init_sample,sigma',
    [
        (X, y, init_sample, sigma)
        for (X, y), init_sample, sigma in product(
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
            [None, np.array([1]), np.array([1, 2])],
            [None, 0.1, [0.1, 0.2]],
        )
        if (
            (init_sample is None)
            or (y.ndim == 1 and len(init_sample) == 1)
            or (y.ndim > 1 and y.shape[1] == len(init_sample))
        )
    ] + [

    ]
)
def test_ScoreBasedGenerator__sample_langevin_montecarlo_w_conditions(
    X: np.ndarray,
    y: np.ndarray,
    init_sample: np.ndarray | None,
    sigma: float | None
) -> None:
    n_samples = DEFAULT_N_SAMPLES
    alpha = 0.1
    sbm = ScoreBasedGenerator(estimator=LinearRegression())
    sbm.fit(X, y)
    samples = sbm._sample_langenvin_montecarlo(
        X,
        n_samples=n_samples,
        alpha=alpha,
        init_sample=init_sample,
        sigma=sigma,
    )
    assert samples.shape == (n_samples, X.shape[0], 1 if y.ndim == 1 else y.shape[1])  # noqa

    # With conditions, not giving X to sample method raises an error
    with pytest.raises(Exception):
        # FIXME: make it more specific
        sbm._sample_langenvin_montecarlo(
            n_samples=n_samples,
            alpha=alpha,
        )


@pytest.mark.parametrize(
    'conditioned_by',
    [
        {0: 1},
        {0: np.array([1, 2, 3])},
    ]
)
def test_ScoreBasedGenerator__sample_langevin_montecarlo_w_conditions_with_conditioned_by(  # noqa
    conditioned_by: dict[int, int | np.ndarray]
) -> None:

    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([[1, 2], [3, 4], [5, 6]])
    n_samples = DEFAULT_N_SAMPLES
    alpha = 0.1
    sigma = 1e-4
    sbm = ScoreBasedGenerator(estimator=LinearRegression())
    sbm.fit(X, y)
    samples = sbm._sample_langenvin_montecarlo(
        X,
        n_samples=n_samples,
        alpha=alpha,
        sigma=sigma,
        conditioned_by=conditioned_by,
    )
    assert samples.shape == (n_samples, X.shape[0], X.shape[1] if X.ndim > 1 else 1)  # noqa
    for k, v in conditioned_by.items():
        # Check if the first column is conditioned by the value of conditioned_by  # noqa
        if isinstance(v, np.ndarray):
            for n in range(n_samples):
                assert np.all(samples[n, :, k] == v)
        else:
            assert np.all(samples[:, :, k] == v)


@pytest.mark.parametrize(
    'X,init_sample',
    [
        (X, init_sample)
        for X, init_sample in product(
            [
                np.array([1, 2, 3]),
                np.array([[1], [2], [3]]),
                np.array([[1, 2], [3, 4], [5, 6]]),
            ],
            [None, np.array([1]), np.array([1, 2])],
        )
        if (
            (init_sample is None)
            or (X.ndim == 1 and len(init_sample) == 1)
            or (X.ndim > 1 and X.shape[1] == len(init_sample))
        )
    ]
)
def test_ScoreBasedGenerator__sample_euler_wo_conditions(
    X: np.ndarray,
    init_sample: np.ndarray | None,
) -> None:
    n_samples = DEFAULT_N_SAMPLES
    n_steps = 101
    sbm = ScoreBasedGenerator(estimator=LinearRegression())
    sbm.fit(X)

    # return_paths=False
    samples = sbm._sample_euler(
        n_samples=n_samples,
        n_steps=n_steps,
        init_sample=init_sample,
    )
    assert samples.shape == (n_samples, 1, X.shape[1] if X.ndim > 1 else 1)

    # return_paths=True
    samples = sbm._sample_euler(
        n_samples=n_samples,
        n_steps=n_steps,
        return_paths=True,
    )
    assert samples.shape == (n_steps, n_samples, 1, X.shape[1] if X.ndim > 1 else 1)  # noqa

    # Without conditions, giving X to sample method raises an error
    with pytest.raises(Exception):
        # FIXME: make it more specific
        sbm._sample_euler(
            X,
            n_samples=n_samples,
            n_steps=n_steps,
        )


def test_ScoreBasedGenerator__sample_euler_wo_conditions_with_conditioned_by() -> None:  # noqa

    conditioned_by = {0: 3}

    X = np.array([[1, 2], [3, 4], [5, 6]])
    n_samples = DEFAULT_N_SAMPLES
    n_steps = 101
    sbm = ScoreBasedGenerator(estimator=LinearRegression())
    sbm.fit(X)

    samples = sbm._sample_euler(
        n_samples=n_samples,
        n_steps=n_steps,
        conditioned_by=conditioned_by,
    )
    assert samples.shape == (n_samples, 1, X.shape[1] if X.ndim > 1 else 1)
    assert np.all(samples[:, 0, 0] == conditioned_by[0])  # Check if the first column is conditioned by the value of conditioned_by  # noqa


@pytest.mark.parametrize(
    'conditioned_by,ExpectedException',
    [
        ({0: 1, 1: 2}, ValueError),
        ({3: 2}, KeyError),
    ]
)
def test_ScoreBasedGenerator__sample_euler_wo_conditions_with_invalid_conditioned_by(  # noqa
    conditioned_by: dict[int, int],
    ExpectedException: type[Exception],
) -> None:

    X = np.array([[1, 2], [3, 4], [5, 6]])
    n_samples = DEFAULT_N_SAMPLES
    n_steps = 101
    sbm = ScoreBasedGenerator(estimator=LinearRegression())
    sbm.fit(X)

    with pytest.raises(ExpectedException):
        sbm._sample_euler(
            n_samples=n_samples,
            n_steps=n_steps,
            conditioned_by=conditioned_by,
        )


@pytest.mark.parametrize(
    'X,y,init_sample',
    [
        (
            np.array([[1, 2], [3, 4], [5, 6]]),
            np.array([0, 1, 0]),
            np.array([1]),
        ),
        (
            np.array([[1, 2], [3, 4], [5, 6]]),
            np.array([0, 1, 0]),
            None,
        ),
        (
            np.array([[1, 2], [3, 4], [5, 6]]),
            np.array([[1, 2], [3, 4], [5, 6]]),
            np.array([1, 2]),
        ),
        (
            np.array([[1, 2], [3, 4], [5, 6]]),
            np.array([[1, 2], [3, 4], [5, 6]]),
            None,
        ),
    ]
)
def test_ScoreBasedGenerator__sample_euler_w_conditions(
    X: np.ndarray,
    y: np.ndarray,
    init_sample: np.ndarray | None,
) -> None:
    n_samples = DEFAULT_N_SAMPLES
    n_steps = 101
    sbm = ScoreBasedGenerator(estimator=LinearRegression())
    sbm.fit(X, y)

    # return_paths=False
    samples = sbm._sample_euler(
        X,
        n_samples=n_samples,
        n_steps=n_steps,
        init_sample=init_sample,
    )
    assert samples.shape == (n_samples, X.shape[0], 1 if y.ndim == 1 else y.shape[1])  # noqa

    # return_paths=True
    samples = sbm._sample_euler(
        X,
        n_samples=n_samples,
        n_steps=n_steps,
        init_sample=init_sample,
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


@pytest.mark.parametrize(
    'conditioned_by',
    [
        {0: 1},
        {0: np.array([1, 2, 3])},
    ]
)
def test_ScoreBasedGenerator__sample_euler_w_conditions_with_conditioned_by(
    conditioned_by: dict[int, int | np.ndarray]
) -> None:

    conditioned_by = {0: 3}

    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([[1, 2], [3, 4], [5, 6]])

    n_samples = DEFAULT_N_SAMPLES
    n_steps = 101
    sbm = ScoreBasedGenerator(estimator=LinearRegression())
    sbm.fit(X, y)

    samples = sbm._sample_euler(
        X,
        n_samples=n_samples,
        n_steps=n_steps,
        conditioned_by=conditioned_by,
    )
    assert samples.shape == (n_samples, X.shape[0], 1 if y.ndim == 1 else y.shape[1])  # noqa
    for k, v in conditioned_by.items():
        if isinstance(v, np.ndarray):
            for n in range(n_samples):
                assert np.all(samples[n, :, k] == v)
        else:
            assert np.all(samples[:, :, k] == v)


@pytest.mark.parametrize(
    'X,init_sample',
    [
        (X, init_sample)
        for X, init_sample in product(
            [
                np.array([1, 2, 3]),
                np.array([[1], [2], [3]]),
                np.array([[1, 2], [3, 4], [5, 6]]),
            ],
            [None, np.array([1]), np.array([1, 2])],
        )
        if (
            (init_sample is None)
            or (X.ndim == 1 and len(init_sample) == 1)
            or (X.ndim > 1 and X.shape[1] == len(init_sample))
        )
    ]
)
def test_ScoreBasedGenerator__sample_euler_maruyama_wo_conditions(
    X: np.ndarray,
    init_sample: np.ndarray | None,
) -> None:
    n_samples = DEFAULT_N_SAMPLES
    n_steps = 101
    sbm = ScoreBasedGenerator(estimator=LinearRegression())
    sbm.fit(X)

    # return_paths=False
    samples = sbm._sample_euler_maruyama(
        n_samples=n_samples,
        n_steps=n_steps,
        init_sample=init_sample,
    )
    assert samples.shape == (n_samples, 1, X.shape[1] if X.ndim > 1 else 1)

    # return_paths=True
    samples = sbm._sample_euler_maruyama(
        n_samples=n_samples,
        n_steps=n_steps,
        init_sample=init_sample,
        return_paths=True,
    )
    assert samples.shape == (n_steps, n_samples, 1, X.shape[1] if X.ndim > 1 else 1)  # noqa

    # Without conditions, giving X to sample method raises an error
    with pytest.raises(Exception):
        sbm._sample_euler_maruyama(
            X,
            n_samples=n_samples,
            n_steps=n_steps,
        )


def test_ScoreBasedGenerator__sample_euler_maruyama_wo_conditions_with_conditioned_by() -> None:  # noqa

    conditioned_by = {0: 3}

    X = np.array([[1, 2], [3, 4], [5, 6]])
    n_samples = DEFAULT_N_SAMPLES
    n_steps = 101
    sbm = ScoreBasedGenerator(estimator=LinearRegression())
    sbm.fit(X)

    samples = sbm._sample_euler_maruyama(
        n_samples=n_samples,
        n_steps=n_steps,
        conditioned_by=conditioned_by,
    )
    assert samples.shape == (n_samples, 1, X.shape[1] if X.ndim > 1 else 1)
    assert np.all(samples[:, 0, 0] == conditioned_by[0])  # Check if the first column is conditioned by the value of conditioned_by  # noqa


@pytest.mark.parametrize(
    'conditioned_by,ExpectedException',
    [
        ({0: 1, 1: 2}, ValueError),
        ({3: 2}, KeyError),
    ]
)
def test_ScoreBasedGenerator__sample_euler_maruyama_wo_conditions_with_invalid_conditioned_by(  # noqa
    conditioned_by: dict[int, int],
    ExpectedException: type[Exception],
) -> None:

    X = np.array([[1, 2], [3, 4], [5, 6]])
    n_samples = DEFAULT_N_SAMPLES
    n_steps = 101
    sbm = ScoreBasedGenerator(estimator=LinearRegression())
    sbm.fit(X)

    with pytest.raises(ExpectedException):
        sbm._sample_euler_maruyama(
            n_samples=n_samples,
            n_steps=n_steps,
            conditioned_by=conditioned_by,
        )


@pytest.mark.parametrize(
    'X,y,init_sample',
    [
        (
            np.array([[1, 2], [3, 4], [5, 6]]),
            np.array([0, 1, 0]),
            np.array([1]),
        ),
        (
            np.array([[1, 2], [3, 4], [5, 6]]),
            np.array([0, 1, 0]),
            None,
        ),
        (
            np.array([[1, 2], [3, 4], [5, 6]]),
            np.array([[1, 2], [3, 4], [5, 6]]),
            np.array([1, 2]),
        ),
        (
            np.array([[1, 2], [3, 4], [5, 6]]),
            np.array([[1, 2], [3, 4], [5, 6]]),
            None,
        ),
    ]
)
def test_ScoreBasedGenerator__sample_euler_maruyama_w_conditions(
    X: np.ndarray,
    y: np.ndarray,
    init_sample: np.ndarray | None
) -> None:
    n_samples = DEFAULT_N_SAMPLES
    n_steps = 101
    sbm = ScoreBasedGenerator(estimator=LinearRegression())
    sbm.fit(X, y)

    # return_paths=False
    samples = sbm._sample_euler_maruyama(
        X,
        n_samples=n_samples,
        n_steps=n_steps,
        init_sample=init_sample,
    )
    assert samples.shape == (n_samples, X.shape[0], 1 if y.ndim == 1 else y.shape[1])  # noqa

    # return_paths=True
    samples = sbm._sample_euler_maruyama(
        X,
        n_samples=n_samples,
        n_steps=n_steps,
        init_sample=init_sample,
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
    'conditioned_by',
    [
        {0: 1},
        {0: np.array([1, 2, 3])},
    ]
)
def test_ScoreBasedGenerator__sample_euler_maruyama_w_conditions_with_conditioned_by(  # noqa
    conditioned_by: dict[int, int | np.ndarray]
) -> None:

    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([[1, 2], [3, 4], [5, 6]])

    n_samples = DEFAULT_N_SAMPLES
    n_steps = 101
    sbm = ScoreBasedGenerator(estimator=LinearRegression())
    sbm.fit(X, y)

    samples = sbm._sample_euler_maruyama(
        X,
        n_samples=n_samples,
        n_steps=n_steps,
        conditioned_by=conditioned_by,
    )
    assert samples.shape == (n_samples, X.shape[0], 1 if y.ndim == 1 else y.shape[1])  # noqa
    for k, v in conditioned_by.items():
        if isinstance(v, np.ndarray):
            for n in range(n_samples):
                assert np.all(samples[n, :, k] == v)
        else:
            assert np.all(samples[:, :, k] == v)


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

    n_samples = DEFAULT_N_SAMPLES
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

    n_samples = DEFAULT_N_SAMPLES
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


@pytest.mark.parametrize(
    'X',
    [
        np.array([[1, 2], [3, 4], [5, 6], [10, 9]]),
        np.array([[1], [2], [3]]),
        np.array([1, 2, 3]),
    ]
)
def test_ScoreBasedGenerator_predict(
    X: np.ndarray,
) -> None:
    sbm = ScoreBasedGenerator(estimator=LinearRegression())
    sbm.fit(X, noise_strengths=[1e-3, 1, 10])

    mean_pred = sbm.predict()
    if X.ndim == 1 or X.shape[1] == 1:
        assert mean_pred.ndim == 0
    else:
        assert mean_pred.shape == (X.shape[1],)

    mean_pred, std_pred = sbm.predict(return_std=True)
    if X.ndim == 1 or X.shape[1] == 1:
        assert mean_pred.ndim == 0
        assert std_pred.ndim == 0
    else:
        assert mean_pred.shape == (X.shape[1],)
        assert std_pred.shape == (X.shape[1],)

# TODO: test ScoreBasedGenerator.predict with conditions


@pytest.mark.parametrize(
    'X',
    [
        np.array([[1, 2], [3, 4], [5, 6], [10, 9]]),
        np.array([[1], [2], [3]]),
        np.array([1, 2, 3]),
    ]
)
def test_ScoreBasedGenerator_predict_score(
    X: np.ndarray,
) -> None:
    sbm = ScoreBasedGenerator(estimator=LinearRegression())
    sbm.fit(X, noise_strengths=[1e-3, 1, 10])
    score = sbm.predict_score(X)
    if X.ndim == 1 or X.shape[1] == 1:
        assert score.shape == X.shape[:1]
    else:
        assert score.shape == X.shape


@pytest.mark.parametrize(
    'X,y',
    [
        (
            np.array([[1, 2], [3, 4], [5, 6]]),
            np.array([0, 1, 0]),
        ),
        (
            np.array([[1, 2], [3, 4], [5, 6]]),
            np.array([[1], [2], [3]]),
        ),
        (
            np.array([[1, 2], [3, 4], [5, 6]]),
            np.array([[1, 2], [3, 4], [5, 6]]),
        )
    ]
)
def test_ScoreBasedGenerator_predict_score_w_conditions(
    X: np.ndarray,
    y: np.ndarray,
) -> None:
    sbm = ScoreBasedGenerator(estimator=LinearRegression())
    sbm.fit(X, y, noise_strengths=[1e-3, 1, 10])
    score = sbm.predict_score(X, y)
    if y.ndim == 1 or y.shape[1] == 1:
        assert score.shape == y.shape[:1]
    else:
        assert score.shape == y.shape


@pytest.mark.parametrize(
    'fit_kwargs,kwargs,expected_exception_cls',
    [
        (
            # Default case is passed
            {
                'X': np.array([[1, 2], [3, 4], [5, 6]]),
                'y': None,
            },
            {},
            None,  # None means that no exception is expected
        ),
        (
            # Case: X is not given to _validate_kwargs_for_sample when y is given  # noqa
            {
                'X': np.array([[1, 2], [3, 4], [5, 6]]),
                'y': np.array([0, 1, 0]),
            },
            {
                'X': None,
            },
            TypeError,
        ),
        (
            # Case: n_samples is 0
            {
                'X': np.array([[1, 2], [3, 4], [5, 6]]),
                'y': None,
            },
            {
                'n_samples': 0,
            },
            ValueError,
        ),
        (
            # Case: n_samples is negative
            {
                'X': np.array([[1, 2], [3, 4], [5, 6]]),
                'y': None,
            },
            {
                'n_samples': -1,
            },
            ValueError,
        ),
        (
            # Case: n_steps is 0
            {
                'X': np.array([[1, 2], [3, 4], [5, 6]]),
                'y': None,
            },
            {
                'n_steps': 0,
            },
            ValueError,
        ),
        (
            # Case: n_steps is negative
            {
                'X': np.array([[1, 2], [3, 4], [5, 6]]),
                'y': None,
            },
            {
                'n_steps': -1,
            },
            ValueError,
        ),
        (
            # Case: the size of conditioned_by is equal to the expected output
            {
                'X': np.array([[1, 2], [3, 4], [5, 6]]),
                'y': None,
            },
            {
                'conditioned_by': {0: 1, 1: 2},
            },
            ValueError,
        ),
        (
            # Case: conditioned_by is larger than the expected output
            {
                'X': np.array([[1, 2], [3, 4], [5, 6]]),
                'y': None,
            },
            {
                'conditioned_by': {0: 1, 1: 2, 2: 3},
            },
            ValueError,
        ),
        (
            # Case: conditioned_by has invalid keys
            {
                'X': np.array([[1, 2], [3, 4], [5, 6]]),
                'y': None,
            },
            {
                'conditioned_by': {-1: 1},
            },
            KeyError,
        ),
        (
            # Case: the value of conditioned_by is np.ndarray when X is None
            {
                'X': np.array([[1, 2], [3, 4], [5, 6]]),
                'y': None,
            },
            {
                'conditioned_by': {0: np.array([1, 2, 3])},
            },
            TypeError,
        ),
        (
            # Case: the shape of a np.ndarray value of conditioned_by is not (X.shape[0], )  # noqa
            {
                'X': np.array([[1, 2], [3, 4], [5, 6]]),
                'y': np.array([[1, 2], [3, 4], [5, 6]]),
            },
            {
                'X': np.array([[1, 2], [3, 4], [5, 6]]),  # NOTE: X.shape[0] is 3  # noqa
                'conditioned_by': {0: np.array([1, 2,])},  # NOTE: conditioned_by[0].shape[0] is 2  # noqa
            },
            ValueError,
        ),
        (
            # Case: the shape of init_sample is not (n_outputs_, )
            {
                'X': np.array([[1, 2], [3, 4], [5, 6]]),  # NOTE: n_outputs_ is 2  # noqa
                'y': None,
            },
            {
                'init_sample': np.array([1, 2, 3]),
            },
            ValueError,
        ),
        (
            # Case: the shape of init_sample is not (n_outputs_, )
            {
                'X': np.array([[1, 2], [3, 4], [5, 6]]),  # NOTE: n_outputs_ is 2  # noqa
                'y': None,
            },
            {
                'init_sample': np.array([[1, 2], [3, 4], [5, 6]]),
            },
            ValueError,
        ),
        (
            # Case: alpha is 0
            {
                'X': np.array([[1, 2], [3, 4], [5, 6]]),
                'y': None,
            },
            {
                'alpha': 0,
            },
            ValueError,
        ),
        (
            # Case: alpha is negative
            {
                'X': np.array([[1, 2], [3, 4], [5, 6]]),
                'y': None,
            },
            {
                'alpha': -1,
            },
            ValueError,
        ),
        (
            # Case: When sigma is an iterable of float, but it contains 0
            {
                'X': np.array([[1, 2], [3, 4], [5, 6]]),
                'y': None,
            },
            {
                'sigma': [0, 0.1],
            },
            ValueError,
        ),
        (
            # Case: When sigma is an iterable of float, but it contains a negative value  # noqa
            {
                'X': np.array([[1, 2], [3, 4], [5, 6]]),
                'y': None,
            },
            {
                'sigma': [-1, 0.1],
            },
            ValueError,
        ),
        (
            # Case: sigma is 0
            {
                'X': np.array([[1, 2], [3, 4], [5, 6]]),
                'y': None,
            },
            {
                'sigma': 0,
            },
            ValueError,
        ),
        (
            # Case: sigma is negative
            {
                'X': np.array([[1, 2], [3, 4], [5, 6]]),
                'y': None,
            },
            {
                'sigma': -1,
            },
            ValueError,
        )
    ]
)
def test_ScoreBasedGenerator__validate_kwargs_for_sample(
    fit_kwargs: dict[str, np.ndarray | None],
    kwargs: dict[str, object],
    expected_exception_cls: type[Exception] | None
) -> None:
    sbm = ScoreBasedGenerator(estimator=LinearRegression())
    sbm.fit(**fit_kwargs)  # type: ignore
    if expected_exception_cls:
        with pytest.raises(expected_exception_cls):
            sbm._validate_kwargs_for_sample(**kwargs)  # type: ignore
    else:
        sbm._validate_kwargs_for_sample(**kwargs)  # type: ignore


@pytest.mark.parametrize(
    'Xtr,ytr,X,n_samples,init_sample,conditioned_by',
    [
        # Requires ytr and X
        (
            np.array([[1, 2], [3, 4], [5, 6]]),
            np.array([0, 1, 0]),
            np.array([[1, 2], [3, 4], [5, 6]]),
            10,
            None,  # init_sample is None
            {},  # conditioned_by is empty
        ),
        (
            np.array([[1, 2], [3, 4], [5, 6]]),
            np.array([[1, 2], [3, 4], [5, 6]]),
            np.array([[1, 2], [3, 4], [5, 6]]),
            10,
            np.array([1, 2]),
            {},  # conditioned_by is empty
        ),
        (
            np.array([[1, 2], [3, 4], [5, 6]]),
            np.array([[1, 2], [3, 4], [5, 6]]),
            np.array([[1, 2], [3, 4], [5, 6]]),
            10,
            None,  # init_sample is None
            {0: 1},  # conditioned_by is not empty
        ),
        (
            np.array([[1, 2], [3, 4], [5, 6]]),
            np.array([[1, 2], [3, 4], [5, 6]]),
            np.array([[1, 2], [3, 4], [5, 6]]),
            10,
            np.array([1, 2]),
            {0: 1},  # conditioned_by is not empty
        ),
        # Does not require ytr and X
        (
            np.array([[1, 2], [3, 4], [5, 6]]),
            None,
            None,
            10,
            None,  # init_sample is None
            {},  # conditioned_by is empty
        ),
        (
            np.array([[1, 2], [3, 4], [5, 6]]),
            None,
            None,
            10,
            np.array([1, 2]),
            {},  # conditioned_by is empty
        ),
        (
            np.array([[1, 2], [3, 4], [5, 6]]),
            None,
            None,
            10,
            None,  # init_sample is None
            {0: 1},  # conditioned_by is not empty
        ),
        (
            np.array([[1, 2], [3, 4], [5, 6]]),
            None,
            None,
            10,
            np.array([1, 2]),
            {0: 1},  # conditioned_by is not empty
        ),
    ]
)
def test_ScoreBasedGenerator__initialize_samples(
    Xtr: np.ndarray,
    ytr: np.ndarray | None,
    X: np.ndarray | None,
    n_samples: int,
    init_sample: np.ndarray | None,
    conditioned_by: Mapping[int, bool | int | float | np.ndarray],
) -> None:
    # TODO: Test with invalid inputs

    sbm = ScoreBasedGenerator(estimator=LinearRegression())
    sbm.fit(Xtr, ytr)
    with np_seed(0):
        x0 = sbm._initialize_samples(
            X=X,
            n_samples=n_samples,
            init_sample=init_sample,
            conditioned_by=conditioned_by,
        )

    # check shape
    N = 1 if X is None else X.shape[0]
    n_outputs = sbm.n_outputs_
    assert n_samples > 0
    assert N > 0
    assert n_outputs - len(conditioned_by) > 0
    assert x0.shape == (n_samples * N, n_outputs - len(conditioned_by))

    # check random
    if init_sample is None:
        # randomly generated
        assert np.any(x0 != x0.mean())
    else:
        # initialized by init_sample
        init_sample_ = init_sample[[i for i in range(n_outputs) if i not in conditioned_by]]  # noqa
        assert np.all(x0 == np.repeat(init_sample_[np.newaxis, :], n_samples * N, axis=0))  # noqa


@pytest.mark.parametrize(
    'X,n_samples,conditioned_by,expected',
    [
        (
            None,
            3,
            {},
            {},
        ),
        (
            None,
            3,  # =: n_samples
            {0: 1},
            {0: np.array([[1], [1], [1]])},  # Shape: (n_samples, 1)
        ),
        (
            np.array([[1, 2], [3, 4]]),
            3,
            {},
            {},
        ),
        (
            np.array([[1, 2], [3, 4]]),  # N = 2
            3,  # =: n_samples
            {0: 1},
            {0: np.array([[1], [1], [1], [1], [1], [1]])},  # Shape: (N * n_samples, 1)  # noqa
        ),
        (
            np.array([[1, 2], [3, 4], [5, 6]]),  # N = 3
            2,  # =: n_samples
            {0: np.array([1, 2, 3])},
            {0: np.array([[1], [1], [2], [2], [3], [3]])},  # Shape: (N * n_samples, 1)  # noqa
        ),
        (
            np.array([[1, 2], [3, 4], [5, 6]]),  # N = 3
            2,  # =: n_samples
            {0: np.array([1, 2, 3]), 1: 1},
            {
                0: np.array([[1], [1], [2], [2], [3], [3]]),
                1: np.array([[1], [1], [1], [1], [1], [1]]),
            },  # noqa
        )

    ]
)
def test_ScoreBasedGenerator__preprocess_conditioned_by(
    X: np.ndarray | None,
    n_samples: int,
    conditioned_by: dict[int, int | np.ndarray],
    expected: dict[int, np.ndarray],
) -> None:
    actual = ScoreBasedGenerator._preprocess_conditioned_by(X, n_samples, conditioned_by)  # noqa
    for k, v in actual.items():
        np.testing.assert_array_equal(v, expected[k])


@pytest.mark.parametrize(
    'x,conditioned_by,expected',
    [
        (
            np.array([[1, 2], [3, 4], [5, 6]]),
            {},
            np.array([[1, 2], [3, 4], [5, 6]]),
        ),
        (
            np.array([[1, 2], [3, 4], [5, 6]]),
            {1: np.array([[10], [20], [30]])},
            np.array([[1, 10, 2], [3, 20, 4], [5, 30, 6]]),
        ),
        (
            np.array([[1, 2], [3, 4], [5, 6]]),
            {2: np.array([[10], [20], [30]])},
            np.array([[1, 2, 10], [3, 4, 20], [5, 6, 30]]),
        ),
    ]
)
def test_ScoreBasedGenerator__insert_conditiond_x_to_unconditioned_x(
    x: np.ndarray,
    conditioned_by: Mapping[int, np.ndarray],
    expected: np.ndarray,
) -> None:
    # TODO: test invalid inputs
    sbm = ScoreBasedGenerator(estimator=LinearRegression())
    sbm.n_outputs_ = x.shape[1] + len(conditioned_by)

    actual = sbm._insert_conditiond_x_to_unconditioned_x(x, conditioned_by)

    np.testing.assert_array_equal(actual, expected)
