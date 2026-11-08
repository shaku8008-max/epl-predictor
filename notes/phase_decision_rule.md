# Season-phase check: decision rule (written BEFORE running anything)

Written Sun 8 Nov 2026.

## Question
Does the gap between the bookmaker and our results-only model change across the season? Early season is the most plausible place for it to be larger: form windows reach back into last season, Elo has no summer reset, promoted teams start at 1400, and the bookmaker knows transfers and injuries from day one.

## Data (dev only, hold-out untouched)
- Seasons 2012..2023, 4,560 matches. 2024 and 2025 are NOT used.
- No new training. Saved predictions only:
  - dev_logistic_preds.csv (logistic p_*, bookmaker mkt_*)
  - goals_check_preds.csv (logistic, linear, poisson)
- Merge on date, home, away (one_to_one). Must be exactly 4,560 rows.

## Phase definition (fixed before looking)
Within each season, sort matches by date (stable) and use the rank:
- early = ranks 1..127
- mid   = ranks 128..253
- late  = ranks 254..380
Check that the 2020 (COVID) season still gives 380 matches and sensible ordering.

## Measure
Per-match log loss. gap = model loss minus bookmaker loss.
Positive gap = bookmaker is ahead. (Opposite sign to yardstick's gain.)
Model for the primary test: the standard logistic.

## PRIMARY TEST (the only decision)
difference = (early gap) - (late gap)
Bootstrap by match within each phase, 10,000 resamples, seed 0.
"Phase matters" only if the 95% interval of the difference excludes 0 (entirely above 0 means the bookmaker's lead is larger early).
Otherwise: no detectable phase effect.

## Context only (NOT decisions)
- Gap per phase (early, mid, late) with intervals.
- Goals-model gain over the logistic per phase (is it concentrated in one phase?).

## Prior (stated in advance)
A modest early-season gap, nothing dramatic. A null is the most likely outcome. If it is a null I will say "no detectable effect", not "proven zero".

## What happens next
- Null: stop. No phase models. Hold-out stays unspent.
- Real difference: Step 2 (a separate rule file) comparing the single model with ONE model plus a phase indicator, dev walk-forward. At most one hold-out run after that, settings fixed beforehand.

## Not being done
- No cut of the training window to 2015..2023 (separate hypothesis, separate test).
- No separate models per phase (about 1,100 matches each, data-starved).
- No looking at the hold-out.

## RESULT (Sun 8 Nov 2026, dev 2012..2023, 4,560 matches, one run)
Phases: early 1524, mid 1512, late 1524. 2020 ordering sane.
Gap (logistic minus bookmaker, positive = bookmaker ahead):
- early +0.0214 [+0.0112, +0.0320]
- mid   +0.0205 [+0.0097, +0.0315]
- late  +0.0137 [+0.0036, +0.0239]

PRIMARY TEST (early gap minus late gap): +0.0077, 95% interval [-0.0069, +0.0222].
Interval includes 0 -> NO DETECTABLE PHASE EFFECT.

Reading: direction matches the prior (bookmaker's lead largest early), but
the interval is about +/-0.015 wide, so the test could not detect an effect
smaller than that. Not evidence of zero. Context only: the Poisson gain looks
smaller late, the linear gain is flat across phases; differences between
phases were not tested, so no conclusion is drawn.

DECISION: stop. No phase-indicator model, no hold-out run, no re-run with
other phase boundaries. Hold-out (2024 + 2025) remains unspent for this idea.
