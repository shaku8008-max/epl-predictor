import sys
from pathlib import Path

import numpy as np
import pandas as pd

from models import outcome_probs

MODEL_FILE = Path("models/goals_nn_v1.npz")


class GoalsPredictor:
    """The trained ensemble, running on numpy alone. Loads the file nn_train.py exports."""

    def __init__(self, path: Path = MODEL_FILE):
        f = np.load(path, allow_pickle=False)
        self.columns = [str(c) for c in f["columns"]]
        self.medians, self.mean, self.scale = f["medians"], f["mean"], f["scale"]
        self.w1, self.b1 = f["w1"], f["b1"]           # shape [network, out, in] and [network, out]
        self.w2, self.b2 = f["w2"], f["b2"]
        self.w3, self.b3 = f["w3"], f["b3"]
        self.n_networks = self.w1.shape[0]
        self.trained_rows = int(f["trained_rows"])
        self.created = str(f["created"])

    def _inputs(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.columns].to_numpy(dtype=float)
        x = np.where(np.isnan(x), self.medians, x)         # same median fill as training
        return (x - self.mean) / self.scale                 # same standardising as training

    def _lambdas(self, df: pd.DataFrame) -> np.ndarray:
        """Expected goals [home, away] per network: shape [network, match, 2]."""
        x = self._inputs(df)
        out = []
        for k in range(self.n_networks):
            h1 = np.maximum(x @ self.w1[k].T + self.b1[k], 0)      # Linear, ReLU (dropout is off at predict time)
            h2 = np.maximum(h1 @ self.w2[k].T + self.b2[k], 0)      # Linear, ReLU
            out.append(np.exp(h2 @ self.w3[k].T + self.b3[k]))      # Linear, then exp: log goals -> goals
        return np.array(out)

    def expected_goals(self, df: pd.DataFrame) -> np.ndarray:
        """Average expected goals [home, away] per match, over the ensemble."""
        return self._lambdas(df).mean(axis=0)

    def predict_probs(self, df: pd.DataFrame) -> np.ndarray:
        """[P(home win), P(draw), P(away win)] per match, averaged over the ensemble."""
        per_network = [outcome_probs(lam[:, 0], lam[:, 1]) for lam in self._lambdas(df)]
        return np.mean(per_network, axis=0)


def self_test() -> None:
    """Needs torch (training side only): numpy must agree with the torch model it was exported from."""
    from nn_train import GoalsEnsemble, load_features   # imported here so the webapp never needs torch
    from evaluate import score
    from train_final import LIVE_SEASON

    df = load_features()
    train = df[df["season"] < LIVE_SEASON]
    sample = df[df["season"] == LIVE_SEASON - 1]

    tiny = GoalsEnsemble(seeds=(1, 2), epochs=2).fit(train)     # small and quick: only the arithmetic matters here
    scratch = Path("models/_parity_test.npz")
    tiny.export(scratch)
    try:
        torch_probs = tiny.predict_probs(sample)
        numpy_probs = GoalsPredictor(scratch).predict_probs(sample)
    finally:
        scratch.unlink(missing_ok=True)
    worst = np.abs(torch_probs - numpy_probs).max()
    print(f"Torch vs numpy, {len(sample)} matches: largest probability difference {worst:.2e}")
    assert worst < 1e-4, "numpy and torch disagree"
    print("Parity test passed.")

    # The real exported model on the live season so far (never used for fitting).
    model = GoalsPredictor()
    live = df[df["season"] == LIVE_SEASON]
    print(f"\n{MODEL_FILE}: {model.n_networks} networks, trained on {model.trained_rows} matches ({model.created})")
    if len(live):
        s = score(live["result"], model.predict_probs(live))
        print(f"Live {LIVE_SEASON} so far: {s['matches']} matches, log loss {s['log_loss']:.4f}, accuracy {s['accuracy']:.2%}")
        print("\nFirst three live matches:")
        show = live[["home", "away"]].head(3).copy()
        show[["p_home", "p_draw", "p_away"]] = model.predict_probs(live.head(3)).round(3)
        show[["xg_home", "xg_away"]] = model.expected_goals(live.head(3)).round(2)
        print(show.to_string(index=False))


if __name__ == "__main__":
    self_test()