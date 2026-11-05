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

    fixtures = next_round(predictions)
    if fixtures:
        db.upsert("nn_prediction_history", fixtures, "home,away,fixture_date")

    print(f"Uploaded {len(predictions)} predictions and {len(fixtures)} next-round fixtures "
          f"({predictions[0]['model']} {predictions[0]['model_version']}, {stamp})")


if __name__ == "__main__":
    main()