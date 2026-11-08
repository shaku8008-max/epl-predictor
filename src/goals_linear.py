"""Linear goal-difference model: numpy only (plus json and math).

This one file can be copied to epl-predictor/src and used for serving without
sklearn, scipy or pandas.

Idea: predict the expected goal difference (home goals minus away goals) with a
straight-line model, assume the real difference scatters around it in a bell curve,
and read H/D/A off the curve. A draw is a difference between -0.5 and +0.5.

Works on plain numpy arrays: the columns of X must be in the order of `columns`.
"""
import json
import math
from pathlib import Path

import numpy as np

DRAW_HALF_WIDTH = 0.5      # goal difference between -0.5 and +0.5 counts as a draw


def normal_cdf(z):
    """Standard normal CDF (the bell curve's running total), using only math.erf."""
    z = np.asarray(z, dtype=float)
    erf = np.vectorize(math.erf, otypes=[float])
    return 0.5 * (1.0 + erf(z / math.sqrt(2.0)))


class LinearGoalsModel:
    def __init__(self, columns):
        self.columns = list(columns)
        self.medians = None
        self.means = None
        self.scales = None
        self.coef = None
        self.intercept = None
        self.sigma = None

    def _standardise(self, X):
        X = np.asarray(X, dtype=float)
        assert X.ndim == 2 and X.shape[1] == len(self.columns), "X needs one column per feature"
        X = np.where(np.isnan(X), self.medians, X)          # fill gaps with training medians
        return (X - self.means) / self.scales

    def fit(self, X, goal_diff):
        """X: (n, n_features) array. goal_diff: home goals minus away goals, length n."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(goal_diff, dtype=float)
        assert X.ndim == 2 and X.shape[1] == len(self.columns) and len(y) == len(X)
        assert np.isfinite(y).all(), "goal_diff has gaps"

        self.medians = np.nanmedian(X, axis=0)
        assert np.isfinite(self.medians).all(), "a feature column is entirely empty"
        filled = np.where(np.isnan(X), self.medians, X)
        self.means = filled.mean(axis=0)
        self.scales = filled.std(axis=0)
        self.scales[self.scales == 0] = 1.0                 # a constant column stays harmless

        Z = (filled - self.means) / self.scales
        design = np.column_stack([np.ones(len(Z)), Z])      # first column = the intercept
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        self.intercept = float(beta[0])
        self.coef = beta[1:]
        self.sigma = float((y - design @ beta).std())       # spread of the TRAINING residuals
        return self

    def predict_mu(self, X):
        """Expected goal difference (home minus away) per match."""
        return self._standardise(X) @ self.coef + self.intercept

    def predict_probs(self, X):
        """[P(H), P(D), P(A)] per match."""
        mu = self.predict_mu(X)
        upper = normal_cdf((DRAW_HALF_WIDTH - mu) / self.sigma)
        lower = normal_cdf((-DRAW_HALF_WIDTH - mu) / self.sigma)
        return np.column_stack([1.0 - upper, upper - lower, lower])

    def to_dict(self):
        return {
            "model": "linear_goal_difference",
            "columns": self.columns,
            "medians": self.medians.tolist(),
            "means": self.means.tolist(),
            "scales": self.scales.tolist(),
            "coef": self.coef.tolist(),
            "intercept": self.intercept,
            "sigma": self.sigma,
            "draw_half_width": DRAW_HALF_WIDTH,
        }

    @classmethod
    def from_dict(cls, d):
        assert d["draw_half_width"] == DRAW_HALF_WIDTH, "saved file used a different draw width"
        model = cls(d["columns"])
        model.medians = np.array(d["medians"])
        model.means = np.array(d["means"])
        model.scales = np.array(d["scales"])
        model.coef = np.array(d["coef"])
        model.intercept = float(d["intercept"])
        model.sigma = float(d["sigma"])
        return model

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))
