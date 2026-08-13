# ============================================================
# app.py (for Hugging Face Space)
# ============================================================

from flask import Flask, request, jsonify
from joblib import load
import pandas as pd

# Create Flask app
app = Flask(__name__)

# Load trained model
model = load("model.joblib")


@app.route("/")
def home():
    """Health check endpoint."""
    return jsonify(
        {
            "message": "Iris Prediction API is running.",
            "endpoint": "/predict"
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict iris species from feature values.

    Expected JSON:
    {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    """

    try:
        data = request.get_json()

        required_features = [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width"
        ]

        missing = [
            feature for feature in required_features
            if feature not in data
        ]

        if missing:
            return jsonify(
                {
                    "error": f"Missing required features: {missing}"
                }
            ), 400

        df = pd.DataFrame([{
            "sepal_length": float(data["sepal_length"]),
            "sepal_width": float(data["sepal_width"]),
            "petal_length": float(data["petal_length"]),
            "petal_width": float(data["petal_width"]),
        }])

        prediction = model.predict(df)[0]

        return jsonify(
            {
                "prediction": str(prediction)
            }
        )

    except Exception as e:
        return jsonify(
            {
                "error": str(e)
            }
        ), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
