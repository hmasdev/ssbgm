# `ssbgm`: Score Based Generative Model with scikit-learn

`ssbgm` is a python library which enables you to generate synthetic data using a score based generative model.

You can use `ssbgm` to predict a target value with some features by generating synthetic data given the features.

## Installation

### Requirements

- Python (>= 3.10)
- libraries:
  - catboost>=1.2.7
  - lightgbm>=4.5.0
  - numpy>=1.26.4
  - scikit-learn>=1.5.2
  - tqdm>=4.67.0
  - types-tqdm>=4.66.0.20240417

See [./pyproject.toml](./pyproject.toml) for more details.

### How to Install

You can install `ssbgm` via pip:

```bash
pip install git+https://github.com/hmasdev/ssbgm.git
```

or

```bash
git clone https://github.com/hmasdev/ssbgm.git
pip install .
```

## Usage

### Generate Synthetic Data

Here is an example of generating synthetic data using `ssbgm`:

```python
from sklearn.linear_model import LinearRegression
from ssbgm import ScoreBasedGenerator

# Prepare the dataset which you want to generate synthetic data
# row: sample, column: output dimension
X: np.ndarray = ...

# initialize the generator with LinearRegression
generator = ScoreBasedGenerator(estimator=LinearRegression())

# fit the generator
generator.fit(X)

# generate synthetic data
# Langevin Monte Carlo is used to generate synthetic data
X_syn_lmc = sbmgenerator2.sample(n_samples=128, sampling_method=sbmgenerator2.SamplingMethod.LANGEVIN_MONTECARLO, alpha=0.2, n_warmup=1000).squeeze()
X_syn_euler = sbmgenerator2.sample(n_samples=128, sampling_method=sbmgenerator2.SamplingMethod.EULER).squeeze()
X_syn_em = sbmgenerator2.sample(n_samples=128, sampling_method=sbmgenerator2.SamplingMethod.EULER_MARUYAMA).squeeze()
# The shape of each X_syn_* is (128, X.shape[1])
```

### Conditional Generation

You can use `ssbgm` to predict a target value with some features by generating synthetic data given the features.

```python
from sklearn.linear_model import LinearRegression
from ssbgm import ScoreBasedGenerator

# Prepare the dataset which you want to generate synthetic data
# row: sample, column: features
X: np.ndarray = ...
# row: sample, column: target value
y: np.ndarray = ...

# initialize the generator with LinearRegression
generator = ScoreBasedGenerator(estimator=LinearRegression())

# fit the generator
generator.fit(X, y)

# predict the target value with on X
y_pred_by_mean, y_pred_std = generator.predict(X, aggregate='mean', return_std=True)  # Shape: (X.shape[0], y.shape[1]), (X.shape[0], y.shape[1])
y_pred_by_median = generator.predict(X, aggregate='median')  # Shape: (X.shape[0], y.shape[1])

# generate synthetic data conditioned by X
# Langevin Monte Carlo is used to generate synthetic data
X_syn_lmc = sbmgenerator2.sample(X, n_samples=128, sampling_method=sbmgenerator2.SamplingMethod.LANGEVIN_MONTECARLO, alpha=0.2, n_warmup=1000).squeeze()
X_syn_euler = sbmgenerator2.sample(X, n_samples=128, sampling_method=sbmgenerator2.SamplingMethod.EULER).squeeze()
X_syn_em = sbmgenerator2.sample(X, n_samples=128, sampling_method=sbmgenerator2.SamplingMethod.EULER_MARUYAMA).squeeze()
# The shape of each X_syn_* is (128, X.shape[0], X.shape[1])
```

### Examples

TBD

## How to Develop

1. Fork the repository: [https://github.com/hmasdev/ssbgm](https://github.com/hmasdev/ssbgm)
2. Clone the repository

   ```bash
   git clone https://github.com/{YOURE_NAME}/ssbgm
   cd ssbgm
   ```

3. Create a virtual environment

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

4. Install the required packages

   ```bash
   pip install -e .[dev]
   ```

5. Checkout your working branch

   ```bash
   git checkout -b your-working-branch
   ```

6. Make your changes

7. Test your changes

   ```bash
   pytest
   flake8 ssbgm tests
   mypy ssbgm tests
   ```

8. Commit your changes

   ```bash
   git add .
   git commit -m "Your commit message"
   ```

9. Push your changes

   ```bash
   git push origin your-working-branch
   ```

10. Create a pull request: [https://github.com/hmasdev/ssbgm/compare](https://github.com/hmasdev/ssbgm/compare)

## License

[MIT](./LICENSE)

## Author

[hmasdev](https://github.com/hmasdev)
