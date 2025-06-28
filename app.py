from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# Load models
model_dir = "models"
models = {
    "cabbage_scorpio_low": joblib.load(os.path.join(model_dir, "cabbage_scorpio_low.pkl")),
    "cabbage_scorpio_high": joblib.load(os.path.join(model_dir, "cabbage_scorpio_high.pkl")),
    "potato_granula_low": joblib.load(os.path.join(model_dir, "potato_granula_low.pkl")),
    "potato_granula_high": joblib.load(os.path.join(model_dir, "potato_granula_high.pkl")),
}

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
