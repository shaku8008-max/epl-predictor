"""Linear goal-difference model: numpy only (plus json and math).

This one file can be copied to epl-predictor/src and used for serving without
sklearn, scipy or pandas.

Idea: predict the expected goal difference (home goals minus away goals) with a
straight-line model, assume the real difference scatters around it in a bell curve,
and read H/D/A off the curve. A draw is a difference between -0.5 and +0.5.

For display it also fits two more straight-line models on the same inputs: one for
home goals and one for away goals. A least-squares fit is linear in its target, so
expected home goals minus expected away goals equals the expected goal difference
exactly. The probabilities (which use only the goal difference) are therefore not
affected by these extra "goal heads".

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


def _least_squares(design, y):
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return beta


class LinearGoalsModel:
    def __init__(self, columns):
        self.columns = list(columns)
        self.medians = None
        self.means = None
        self.scales = None
        self.coef = None
        self.intercept = None
        self.sigma = None
        # Optional display heads: expected home goals and expected away goals.
        self.home_coef = None
        self.home_intercept = None
        self.away_coef = None
        self.away_intercept = None

    def _standardise(self, X):
        X = np.asarray(X, dtype=float)
        assert X.ndim == 2 and X.shape[1] == len(self.columns), "X needs one column per feature"
        X = np.where(np.isnan(X), self.medians, X)          # fill gaps with training medians
        return (X - self.means) / self.scales

    def fit(self, X, goal_diff, home_goals=None, away_goals=None):
        """X: (n, n_features) array. goal_diff: home goals minus away goals, length n.

        Give home_goals and away_goals as well to also fit the display heads.
        """
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
        beta = _least_squares(design, y)
        self.intercept = float(beta[0])
        self.coef = beta[1:]
        self.sigma = float((y - design @ beta).std())       # spread of the TRAINING residuals

        if home_goals is not None and away_goals is not None:
            home = np.asarray(home_goals, dtype=float)
            away = np.asarray(away_goals, dtype=float)
            assert len(home) == len(X) and len(away) == len(X)
            assert np.isfinite(home).all() and np.isfinite(away).all(), "goals have gaps"
            beta_home = _least_squares(design, home)
            beta_away = _least_squares(design, away)
            self.home_intercept, self.home_coef = float(beta_home[0]), beta_home[1:]
            self.away_intercept, self.away_coef = float(beta_away[0]), beta_away[1:]
        return self

    @property
    def has_goal_heads(self):
        return self.home_coef is not None

    def predict_mu(self, X):
        """Expected goal difference (home minus away) per match."""
        return self._standardise(X) @ self.coef + self.intercept

    def predict_goals(self, X):
        """Expected goals [home, away] per match, shape (n, 2). Display only."""
        assert self.has_goal_heads, "this model file has no goal heads; retrain it"
        Z = self._standardise(X)
        return np.column_stack([Z @ self.home_coef + self.home_intercept,
                                Z @ self.away_coef + self.away_intercept])

    def predict_probs(self, X):
        """[P(H), P(D), P(A)] per match."""
        mu = self.predict_mu(X)
        upper = normal_cdf((DRAW_HALF_WIDTH - mu) / self.sigma)
        lower = normal_cdf((-DRAW_HALF_WIDTH - mu) / self.sigma)
        return np.column_stack([1.0 - upper, upper - lower, lower])

    def to_dict(self):
        out = {
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
        if self.has_goal_heads:
            out.update({
                "home_coef": self.home_coef.tolist(), "home_intercept": self.home_intercept,
                "away_coef": self.away_coef.tolist(), "away_intercept": self.away_intercept,
            })
        return out

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
        if "home_coef" in d:                                # older files have no goal heads
            model.home_coef = np.array(d["home_coef"])
            model.home_intercept = float(d["home_intercept"])
            model.away_coef = np.array(d["away_coef"])
            model.away_intercept = float(d["away_intercept"])
        return model

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))
