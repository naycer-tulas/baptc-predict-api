from flask import Flask, request, jsonify
from flask_cors import CORS
from prophet.serialize import model_from_json
import pandas as pd
import os
import json

app = Flask(__name__)
CORS(app)

# Load models from JSON
model_dir = "models"
models = {}

model_files = {
    "cabbage_scorpio_low": "cabbage_scorpio_low.json",
    "cabbage_scorpio_high": "cabbage_scorpio_high.json",
    "potato_granula_low": "potato_granula_low.json",
    "potato_granula_high": "potato_granula_high.json",
}

for name, filename in model_files.items():
    with open(os.path.join(model_dir, filename), "r") as fin:
        models[name] = model_from_json(fin.read())  # No need to parse with json.load

@app.route('/')
def home():
    return "✅ BAPTC Prophet API is running!"

@app.route('/baptc-models/predict', methods=['POST'])
def predict_all():
    try:
        data = request.get_json()
        required = ['ds', 'rainfall', 'tmax', 'tmin', 'tmean', 'rh']
        for field in required:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        future_df = pd.DataFrame([{
            "ds": pd.to_datetime(data["ds"]),
            "rainfall": data["rainfall"],
            "tmax": data["tmax"],
            "tmin": data["tmin"],
            "tmean": data["tmean"],
            "rh": data["rh"]
        }])

        results = {}
        for name, model in models.items():
            forecast = model.predict(future_df)
            results[name] = round(forecast['yhat'].iloc[0], 2)

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
