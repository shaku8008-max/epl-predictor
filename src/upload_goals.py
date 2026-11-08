"""Upload goals_predictions.csv (written by predict_goals.py) to Supabase.

Tables: goals_predictions (one row per ordered pairing) and goals_prediction_history
(next-round fixtures, kept so the track record can score them later).
Reuses the Database helper from upload.py, same settings (.env or environment).

Run from ~/epl_webapp/epl-predictor:   python src/upload_goals.py
"""
from upload import Database, load_settings, next_round, read_rows


def main() -> None:
    db = Database(*load_settings())

    predictions = read_rows("goals_predictions.csv")
    assert len(predictions) == 380, f"expected 380 predictions, found {len(predictions)}"
    for row in predictions:
        if row["fixture_date"]:
            row["fixture_date"] = row["fixture_date"][:10]
    stamp = predictions[0]["predicted_at"]

    db.upsert("goals_predictions", predictions, "home,away")
    db.delete_older("goals_predictions", "predicted_at", stamp)

    fixtures = next_round(predictions)
    if fixtures:
        db.upsert("goals_prediction_history", fixtures, "home,away,fixture_date")

    print(f"Uploaded {len(predictions)} predictions and {len(fixtures)} next-round fixtures "
          f"({predictions[0]['model']} {predictions[0]['model_version']}, {stamp})")


if __name__ == "__main__":
    main()
