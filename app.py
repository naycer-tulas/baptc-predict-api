from flask import Flask, request, jsonify
from flask_cors import CORS
from prophet.serialize import model_from_json
import pandas as pd
import os
import json

app = Flask(__name__)
CORS(app)

# === Load Prophet Models ===
model_dir = "models"
models = {}

model_files = {
    "potato_granula_low": "potato_granula_low.json",
    "potato_granula_high": "potato_granula_high.json",
    "potato_lbr_low": "potato_lbr_low.json",
    "potato_lbr_high": "potato_lbr_high.json",
}

for name, filename in model_files.items():
    with open(os.path.join(model_dir, filename), "r") as fin:
        models[name] = model_from_json(fin.read())

@app.route('/')
def home():
    return "Great success. BAPTC Prophet API is working."

# === Predict for multiple days (e.g., 14-day forecast) ===
@app.route('/baptc-models/predict', methods=['POST'])
def predict_all():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No data posted. Please provide JSON payload."}), 400

        # Expect a list of daily entries (14 days)
        if not isinstance(data, list):
            return jsonify({"error": "Input must be a list of daily weather data."}), 400

        required = ['ds', 'rainfall', 'tmax', 'tmin', 'tmean', 'rh']
        for day in data:
            for field in required:
                if field not in day:
                    return jsonify({"error": f"Missing field '{field}' in one of the entries."}), 400

        # Convert to DataFrame
        future_df = pd.DataFrame(data)
        future_df["ds"] = pd.to_datetime(future_df["ds"])

        # Prepare a base result DataFrame with date column
        combined_df = pd.DataFrame({"ds": future_df["ds"]})

        # Predict for each model and merge results
        for name, model in models.items():
            forecast = model.predict(future_df)
            forecast = forecast[["ds", "yhat"]].rename(columns={"yhat": name})
            forecast[name] = forecast[name].round(2)
            combined_df = combined_df.merge(forecast, on="ds", how="left")

        # Convert datetime to string for JSON output
        combined_df["ds"] = combined_df["ds"].dt.strftime("%Y-%m-%d")

        return jsonify(combined_df.to_dict(orient="records"))

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
