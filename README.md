# EPL predictor (backend)

Python project that produces the Premier League predictions the website reads.
It runs by itself every morning on GitHub Actions and writes to Supabase. The
website (`~/epl_webapp/web`, repo `epl-web`) never runs a model, it only reads
the rows uploaded here.

Always run commands from this folder (`~/epl_webapp/epl-predictor`) with the
venv active, for example `python src/predict.py`. Paths in the code are
relative to this folder.

## What it makes

| Model | What it is | Files |
|---|---|---|
| Lean (`lean_v1`) | Stack of Poisson, logistic and boosting models on form, Elo, rest days and xG, combined by a logistic stacker | `src/predict.py`, `models/lean_v1.joblib` |
| Fallback (`fallback_v1`) | Same idea without xG. Used for comparison and as a safety net | `models/fallback_v1.joblib` |
| Neural network (`goals_nn_v1`) | Ensemble of 10 small networks that predict expected goals for each side, turned into H/D/A by a Poisson grid. Served with numpy only (no torch in production) | `src/predict_nn.py`, `src/nn_infer.py`, `models/goals_nn_v1.npz` |
| Season outlook | 10,000 simulated seasons using the NN's expected goals | `src/season_sim.py` |
| Jev | FPL injury/suspension news fed to an LLM that nudges the Lean probabilities. An experiment, for fun only | `src/jev_job.py`, `src/jev_shift.py`, `src/fpl_news.py` |

## Daily flow (GitHub Actions, times are UTC)

- **Predict**, 05:17 (`.github/workflows/predict.yml`):
  `predict.py` -> `upload.py` -> `jev_job.py`
- **Predict NN**, 05:47 (`.github/workflows/predict_nn.yml`):
  `predict_nn.py` -> `season_sim.py` (continue-on-error) -> `upload_nn.py`

Both can be started by hand from the Actions tab (Run workflow). The website's
Refresh page dispatches whichever workflow file the web env var
`GITHUB_WORKFLOW` names (check it, I believe it is `predict.yml`).

GitHub secrets needed: `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`,
`TYPESAFE_API_KEY` (Jev only).

## Data sources

- Results and bookmaker odds: football-data.co.uk (`data/raw/E0_*.csv`).
- xG: Understat copy (`data/raw/understat_EPL_*.json`, `data/processed/xg.csv`).
  **The live feed has been dead since about 24-26 Oct 2026.** `download_xg.py`
  keeps the saved data when the download is unusable. The `runs` table records
  `xg_ok` and `xg_through` and the site shows a note on the Lean page.
- Fixtures: the free FPL API (`src/fpl_fixtures.py`, shared by `predict.py`
  and `predict_nn.py`).
- Team news for Jev: FPL bootstrap-static.

## Supabase tables written here

`predictions`, `team_stats`, `prediction_explain`, `prediction_history`,
`results`, `runs`, `nn_predictions`, `nn_team_stats`, `nn_prediction_explain`,
`nn_prediction_history`, `nn_season_outlook`, `jev_predictions`.

Rules to remember:
- Uploads are upsert + delete-older-than-this-run, so each table always holds
  exactly the latest run (history tables are the exception: they accumulate).
- `nn_season_outlook` is a pre-existing table with the column names
  `goal_diff`, `expected_points`, `expected_position`, `simulations`. The
  script was changed to match the table, not the other way round.
- `prediction_history` and `nn_prediction_history` store the next round's
  predictions. Old Lean rows (Oct 12 to Nov 2) may have been overwritten after
  matches, so the website's track record only counts rows with
  `predicted_at < (fixture_date + 1)`. This is a proxy, not a guarantee.
- `prediction_history`'s own result columns are unused. Always join to
  `results` on `(home, away, fixture_date)`.

## Setup

```bash
cd ~/epl_webapp/epl-predictor
source .venv/bin/activate
pip install -r requirements.txt          # production (numpy, pandas, sklearn...)
pip install -r requirements-train.txt    # only for retraining the NN (adds torch)
```

Local `.env` (git-ignored) holds `SUPABASE_URL` and `SUPABASE_SERVICE_KEY`.
`upload.py` reads it. Careful: running `python src/upload_nn.py` locally
overwrites the live NN tables with your local files. To test only the outlook:

```bash
PYTHONPATH=src python -c "from upload_nn import upload_outlook; from upload import Database, load_settings; print(upload_outlook(Database(*load_settings())))"
```

## Common commands

```bash
python src/predict.py          # Lean + fallback predictions into data/predictions/
python src/predict_nn.py       # NN predictions, team stats, explain
python src/season_sim.py       # season outlook, prints table and sanity checks
python src/serve_check.py      # Lean vs fallback vs NN vs bookmaker on the 760 test matches
python src/jev_job.py --dry-run
```

## Yearly / retraining notes

- `LIVE_SEASON = 2026` in `src/train_final.py` is the season in progress and
  is never used for fitting. **Bump it every August**, and add the new season
  to `SEASONS` in `download_data.py` and `download_xg.py`.
- Tuning seasons 2018-2023, stacker learns from walk-forward predictions
  2018-2025, final test seasons 2024-2025 (760 matches). Never tune on 2024-25.
- Retraining touches (order is my best reading, check each file before
  running): `download_data.py`, `download_xg.py`, `features.py`,
  `train_final.py` for Lean/fallback, and `nn_train.py` for the NN (needs
  torch). `nn_infer.py` has a self-test that checks numpy against torch.

## Results so far (760 test matches, log loss, lower is better)

Lean 1.0037, Neural network 1.0069, fallback 1.0096, bookmaker 0.9947.
Random guessing is 1.0986. Lean and the NN are statistically tied.

Experiments that did not help, so no model change was made: Dixon-Coles
(`dixon_coles.py`, `dc_*.py`), blending the NN with the stacker, feeding
bookmaker odds into the models (worse than the raw odds). No free EPL odds are
currently in `fixtures.csv`.

## Gotchas

- Wrong-folder mistakes are the usual cause of weird errors. Check
  `pwd` first.
- zsh needs quotes around paths with parentheses, for example
  `"src/app/(predict)/page.tsx"` in the web repo.
- Use `git --no-pager` so output doesn't open a pager.
- `data/predictions/` is git-ignored, so in CI a CSV only exists if that run
  created it.
- Left uncommitted on purpose: `src/check_history.py`,
  `data/processed/matches_2005.csv`, `data/raw/E0_2627.csv`.
- Scheduled runs occasionally skip a day (no `runs` rows Nov 2-4 once). Parked.

## Known limits (the website says the same)

- Lean's xG is stale and degrades gradually.
- The live track record is thin (a few dozen matches).
- The season outlook keeps each team's strength fixed at today's values and
  ignores injuries, transfers and model error, so extreme percentages are
  probably too extreme.