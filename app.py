from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app)

model_path = os.path.join(os.path.dirname(__file__), "house_model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "House Price Prediction API is running",
        "usage": "POST /predict with { area_sqft: number, bedrooms: number }",
        "total_samples":501
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data:
        return jsonify({"error": "No data provided"}), 400
    if "area_sqft" not in data or "bedrooms" not in data:
        return jsonify({"error": "Provide both 'area_sqft' and 'bedrooms'"}), 400

    try:
        area = float(data["area_sqft"])
        bedrooms = int(data["bedrooms"])

        if area <= 0 or area > 10000:
            return jsonify({"error": "area_sqft must be between 1 and 10000"}), 400
        if bedrooms < 1 or bedrooms > 10:
            return jsonify({"error": "bedrooms must be between 1 and 10"}), 400

        prediction = model.predict([[area, bedrooms]])
        price = round(float(prediction[0]), 2)

        return jsonify({
            "area_sqft": area,
            "bedrooms": bedrooms,
            "predicted_price": price,
            "predicted_price_formatted": f"${price:,.0f}"
        })

    except (ValueError, TypeError) as e:
        return jsonify({"error": "Invalid input values"}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
