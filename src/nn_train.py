import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from evaluate import score
from models import outcome_probs
from train_final import LIVE_SEASON

FEATURES_FILE = Path("data/processed/features_2005.csv")
MODEL_FILE = Path("models/goals_nn_v1.npz")

FIRST_SEASON = 2005
TEST_SEASONS = [2024, 2025]          # the final exam from the notebook
HIDDEN = 32
DROPOUT = 0.3
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
EPOCHS = 10
SEEDS = tuple(range(1, 11))          # ten networks, probabilities averaged
LEAGUE_AVG_GOALS = (1.53, 1.17)      # home, away: where the output bias starts

# The 29 inputs, in the fixed order the network was developed with.
_FORM = [f"{s}_last5" for s in
         ("points", "goals_for", "goals_against", "shots_for", "shots_against", "sot_for", "sot_against")]
_EXTRA = (["rest_days"] + [f"{s}_venue5" for s in ("points", "goals_for", "goals_against")]
          + ["goal_diff_ewm20", "goal_diff_ewm40"])
FEATURE_COLUMNS = (
    ["elo_home", "elo_away", "elo_diff"]
    + [f"{side}_{f}" for side in ("home", "away") for f in _FORM]
    + [f"{side}_{f}" for side in ("home", "away") for f in _EXTRA]
)


class GoalsNet(nn.Module):
    """Features in, log expected goals out: [home, away]."""

    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, HIDDEN // 2), nn.ReLU(),
            nn.Linear(HIDDEN // 2, 2),
        )
        with torch.no_grad():
            self.net[-1].bias.copy_(torch.log(torch.tensor(LEAGUE_AVG_GOALS)))

    def forward(self, x):
        return self.net(x)


class GoalsEnsemble:
    """Imputer + scaler + one GoalsNet per seed, used as a single model."""

    def __init__(self, seeds=SEEDS, epochs: int = EPOCHS):
        self.columns = list(FEATURE_COLUMNS)
        self.seeds = tuple(seeds)
        self.epochs = epochs

    def _tensor(self, df: pd.DataFrame) -> torch.Tensor:
        x = self.scaler.transform(self.imputer.transform(df[self.columns]))
        return torch.tensor(x, dtype=torch.float32)

    def fit(self, train: pd.DataFrame) -> "GoalsEnsemble":
        self.imputer = SimpleImputer(strategy="median").fit(train[self.columns])
        self.scaler = StandardScaler().fit(self.imputer.transform(train[self.columns]))
        X = self._tensor(train)
        G = torch.tensor(train[["home_goals", "away_goals"]].to_numpy(), dtype=torch.float32)
        loss_fn = nn.PoissonNLLLoss(log_input=True)

        self.nets = []
        for seed in self.seeds:
            torch.manual_seed(seed)
            net = GoalsNet(X.shape[1])
            optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
            loader = DataLoader(TensorDataset(X, G), batch_size=BATCH_SIZE, shuffle=True)
            for _ in range(self.epochs):
                net.train()
                for xb, gb in loader:
                    optimizer.zero_grad()
                    loss_fn(net(xb), gb).backward()
                    optimizer.step()
            net.eval()
            self.nets.append(net)
        self.trained_rows = len(train)
        return self

    def predict_probs(self, df: pd.DataFrame) -> np.ndarray:
        X = self._tensor(df)
        per_seed = []
        with torch.no_grad():
            for net in self.nets:
                lam = torch.exp(net(X)).numpy()
                per_seed.append(outcome_probs(lam[:, 0], lam[:, 1]))
        return np.mean(per_seed, axis=0)

    def export(self, path: Path) -> None:
        """Save everything the webapp needs to predict with numpy alone (no torch)."""
        arrays = {}
        for n, index in ((1, 0), (2, 3), (3, 5)):       # the three Linear layers inside net.net
            layers = [net.net[index] for net in self.nets]
            arrays[f"w{n}"] = np.stack([l.weight.detach().numpy() for l in layers]).astype(np.float64)
            arrays[f"b{n}"] = np.stack([l.bias.detach().numpy() for l in layers]).astype(np.float64)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path, **arrays,
            columns=np.array(self.columns),
            medians=self.imputer.statistics_,
            mean=self.scaler.mean_,
            scale=self.scaler.scale_,
            seeds=np.array(self.seeds),
            trained_rows=np.array(self.trained_rows),
            created=np.array(date.today().isoformat()),
        )


def load_features() -> pd.DataFrame:
    return pd.read_csv(FEATURES_FILE, parse_dates=["date"])


def check(df: pd.DataFrame) -> None:
    """Re-run the notebook's final exam on this pipeline: walk-forward over the test seasons."""
    print(f"Walk-forward over {TEST_SEASONS}, {len(SEEDS)} seeds each (notebook: 1.0069 / 51.05%):")
    parts = []
    for season in TEST_SEASONS:
        train = df[(df["season"] >= FIRST_SEASON) & (df["season"] < season)]
        test = df[df["season"] == season]
        probs = GoalsEnsemble().fit(train).predict_probs(test)
        print(f"  season {season}: log loss {score(test['result'], probs)['log_loss']:.4f}")
        parts.append((test["result"], probs))
    results = pd.concat([r for r, _ in parts])
    probs = np.vstack([p for _, p in parts])
    s = score(results, probs)
    print(f"  both seasons ({s['matches']} matches): log loss {s['log_loss']:.4f} | accuracy {s['accuracy']:.2%}")


def export_final(df: pd.DataFrame) -> None:
    """Train on every finished season and write the weights file."""
    train = df[(df["season"] >= FIRST_SEASON) & (df["season"] < LIVE_SEASON)]
    print(f"Training on {len(train)} matches, seasons {train['season'].min()}-{train['season'].max()}.")
    model = GoalsEnsemble().fit(train)
    model.export(MODEL_FILE)
    print(f"Saved {MODEL_FILE} ({MODEL_FILE.stat().st_size / 1e3:.0f} KB)")


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    df = load_features()
    if mode in ("check", "both"):
        check(df)
    if mode in ("export", "both"):
        export_final(df)


if __name__ == "__main__":
    main()