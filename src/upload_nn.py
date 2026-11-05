import json
from pathlib import Path
from upload import Database, load_settings, next_round, read_rows


def main() -> None:
    db = Database(*load_settings())

    predictions = read_rows("nn_predictions.csv")
    for row in predictions:
        if row["fixture_date"]:
            row["fixture_date"] = row["fixture_date"][:10]
    stamp = predictions[0]["predicted_at"]

    db.upsert("nn_predictions", predictions, "home,away")
    db.delete_older("nn_predictions", "predicted_at", stamp)

    team_stats = read_rows("nn_team_stats.csv")
    db.upsert("nn_team_stats", team_stats, "team")
    db.delete_older("nn_team_stats", "updated_at", stamp)     
    
    explain_path = Path("data/predictions/nn_explain.json")
    explain_rows = json.loads(explain_path.read_text()) if explain_path.exists() else []
    if explain_rows and explain_rows[0]["updated_at"] == stamp:
        db.upsert("nn_prediction_explain", explain_rows, "home,away")
        db.delete_older("nn_prediction_explain", "updated_at", stamp)
    else:
        explain_rows = []
        print("nn_explain.json is missing or not from this run, skipping explain upload")

    fixtures = next_round(predictions)
    if fixtures:
        db.upsert("nn_prediction_history", fixtures, "home,away,fixture_date")

    print(f"Uploaded {len(predictions)} predictions, {len(explain_rows)} explain rows and {len(fixtures)} next-round fixtures "
          f"({predictions[0]['model']} {predictions[0]['model_version']}, {stamp})")


if __name__ == "__main__":
    main()