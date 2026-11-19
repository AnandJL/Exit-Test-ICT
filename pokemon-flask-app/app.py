from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "pokemon_type_model.pkl"

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static")
)

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
label_encoder = bundle["label_encoder"]
features = bundle["features"]
feature_medians = bundle.get("feature_medians", {})


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html")

    try:
        values = {
            "height_dm": float(request.form["height_dm"]),
            "weight_hg": float(request.form["weight_hg"]),
            "base_experience": float(request.form["base_experience"]),
            "hp": float(request.form["hp"]),
            "attack": float(request.form["attack"]),
            "defense": float(request.form["defense"])
        }

        input_data = pd.DataFrame([values])
        input_data = input_data.reindex(columns=features)

        input_data = input_data.fillna(
            pd.Series(feature_medians)
        )

        prediction = model.predict(input_data)[0]

        if hasattr(label_encoder, "inverse_transform"):
            try:
                pokemon_type = label_encoder.inverse_transform(
                    [int(prediction)]
                )[0]
            except (TypeError, ValueError):
                pokemon_type = prediction
        else:
            pokemon_type = prediction

        return render_template(
            "result.html",
            pokemon_type=str(pokemon_type).capitalize()
        )

    except (ValueError, KeyError, TypeError):
        return render_template(
            "predict.html",
            error="Please enter a valid number in every field."
        )


if __name__ == "__main__":
    app.run(debug=False)