import numpy as np
import pandas as pd
from scipy.optimize import minimize

from models import outcome_probs


class DixonColes:
    """One attack and one defence strength per team, plus a home edge, fitted on goals."""
    name = "dixon_coles"

    def __init__(self, reg: float = 5.0, xi: float = 0.0, history: pd.DataFrame | None = None):
        self.reg = reg
        self.xi = xi                      
        self.history = history

    def fit(self, df: pd.DataFrame, cols=None):
        # Use the long history if given, but never anything after the last date in df.
        if self.history is not None:
            data = self.history[self.history["date"] <= df["date"].max()]
        else:
            data = df
        data = data.dropna(subset=["home_goals", "away_goals"])

        teams = sorted(set(data["home"]) | set(data["away"]))
        self.n = len(teams)
        self.index = {t: i for i, t in enumerate(teams)}
        n, reg = self.n, self.reg

        h = data["home"].map(self.index).to_numpy()
        a = data["away"].map(self.index).to_numpy()
        yh = data["home_goals"].to_numpy(dtype=float)
        ya = data["away_goals"].to_numpy(dtype=float)
        years_ago = (df["date"].max() - data["date"]).dt.days.to_numpy() / 365.25
        w = np.exp(-self.xi * years_ago)          # every match counts equally (decay comes next chunk)

        def loss(theta):
            mu, edge = theta[0], theta[1]
            att, dfn = theta[2:2 + n], theta[2 + n:]
            eta_h = mu + edge + att[h] - dfn[a]
            eta_a = mu + att[a] - dfn[h]
            lam_h, lam_a = np.exp(eta_h), np.exp(eta_a)

            # Poisson negative log-likelihood (dropping constants) + L2 pull toward 0
            nll = np.sum(w * (lam_h - yh * eta_h + lam_a - ya * eta_a))
            value = nll + reg * (np.sum(att ** 2) + np.sum(dfn ** 2))

            rh, ra = w * (lam_h - yh), w * (lam_a - ya)
            g_att = np.bincount(h, rh, n) + np.bincount(a, ra, n) + 2 * reg * att
            g_dfn = -np.bincount(a, rh, n) - np.bincount(h, ra, n) + 2 * reg * dfn
            grad = np.concatenate([[rh.sum() + ra.sum(), rh.sum()], g_att, g_dfn])
            return value, grad

        x0 = np.zeros(2 + 2 * n)
        x0[0] = np.log(np.concatenate([yh, ya]).mean())
        result = minimize(loss, x0, jac=True, method="L-BFGS-B")

        self.mu, self.edge = result.x[0], result.x[1]
        self.att, self.dfn = result.x[2:2 + n], result.x[2 + n:]
        self.teams = teams
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        # Unknown team -> index n -> strength 0 (league average).
        att, dfn = np.append(self.att, 0.0), np.append(self.dfn, 0.0)
        h = df["home"].map(self.index).fillna(self.n).astype(int).to_numpy()
        a = df["away"].map(self.index).fillna(self.n).astype(int).to_numpy()
        lam_h = np.exp(self.mu + self.edge + att[h] - dfn[a])
        lam_a = np.exp(self.mu + att[a] - dfn[h])
        return outcome_probs(lam_h, lam_a)

    def ratings(self) -> pd.DataFrame:
        table = pd.DataFrame({"attack": self.att, "defence": self.dfn}, index=self.teams)
        table["overall"] = table["attack"] + table["defence"]
        return table.sort_values("overall", ascending=False).round(3)