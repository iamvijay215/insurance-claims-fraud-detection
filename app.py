import os
import traceback
from datetime import datetime, timezone

import joblib
import pandas as pd
from flask import Flask, request, jsonify

from config import config

app = Flask(__name__)
env = os.getenv("FLASK_ENV", "development")
app.config.from_object(config.get(env, config["default"]))

try:
    model = joblib.load(app.config["MODEL_PATH"])
    feature_names = joblib.load(app.config["FEATURES_PATH"])

    # Scaler is optional. Your final XGBoost model does not require scaling,
    # but this is loaded for completeness if you later deploy a scaled model.
    try:
        scaler = joblib.load(app.config["SCALER_PATH"])
    except FileNotFoundError:
        scaler = None

    print("✅ Model artifacts loaded successfully")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    model = None
    scaler = None
    feature_names = []


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same feature engineering used in the notebook."""
    df = df.copy()

    df["injury_to_total_ratio"] = df["injury_claim"] / (df["total_claim_amount"] + 1)
    df["property_to_total_ratio"] = df["property_claim"] / (df["total_claim_amount"] + 1)
    df["vehicle_to_total_ratio"] = df["vehicle_claim"] / (df["total_claim_amount"] + 1)
    df["premium_to_claim_ratio"] = df["policy_annual_premium"] / (df["total_claim_amount"] + 1)

    # IMPORTANT: Keep this threshold aligned with training.
    # In production, save the training median as an artifact.
    total_claim_median = float(os.getenv("TOTAL_CLAIM_MEDIAN", "58055"))
    df["no_police_high_claim"] = (
        (df["police_report_available"] == 0)
        & (df["total_claim_amount"] > total_claim_median)
    ).astype(int)

    df["multi_vehicle_injury"] = df["number_of_vehicles_involved"] * df["bodily_injuries"]
    df["net_capital"] = df["capital-gains"] + df["capital-loss"]
    df["claim_per_vehicle"] = df["total_claim_amount"] / (df["number_of_vehicles_involved"] + 1)
    df["policy_age_years"] = df["months_as_customer"] / 12
    df["quick_policy_flag"] = (df["days_between_policy_and_incident"] < 180).astype(int)

    return df


def assign_priority(risk_score: float) -> str:
    if risk_score >= 0.65:
        return "HIGH"
    if risk_score >= 0.35:
        return "MEDIUM"
    return "LOW"


def recommendation_for(priority: str) -> str:
    return {
        "HIGH": "Escalate to SIU immediately — manual investigation required.",
        "MEDIUM": "Detailed review required — request supporting documents.",
        "LOW": "Auto-approve or fast-track, subject to standard business checks.",
    }[priority]


def validate_and_score(claims):
    if model is None:
        raise RuntimeError("Model not loaded. Check that .pkl files exist in models/.")

    df = pd.DataFrame(claims)
    df = engineer_features(df)

    missing = [col for col in feature_names if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    X = df[feature_names]
    risk_scores = model.predict_proba(X)[:, 1]

    results = []
    for i, score in enumerate(risk_scores):
        score = float(score)
        priority = assign_priority(score)
        results.append({
            "claim_id": i,
            "risk_score": round(score, 4),
            "risk_percentage": f"{score * 100:.1f}%",
            "priority": priority,
            "ml_verdict": "FRAUD" if score >= 0.5 else "LEGITIMATE",
            "recommendation": recommendation_for(priority),
        })

    return results


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Insurance Claims Fraud Detection API",
        "endpoints": ["/health", "/info", "/predict", "/predict-batch"],
    })


@app.route("/health", methods=["GET"])
def health():
    if model is None:
        return jsonify({"status": "unhealthy", "model_loaded": False}), 500

    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "features_count": len(feature_names),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }), 200


@app.route("/info", methods=["GET"])
def info():
    return jsonify({
        "model_name": "Insurance Claims Fraud Detection",
        "model_type": "XGBoost Classifier",
        "features_count": len(feature_names),
        "feature_names": feature_names,
        "known_notebook_metrics": {
            "roc_auc": 0.8278,
            "fraud_f1": 0.5918,
            "fraud_precision": 0.5918,
            "fraud_recall": 0.5918,
            "high_priority_fraud_concentration": "approximately 93%",
        },
        "endpoints": {
            "health": "GET /health",
            "info": "GET /info",
            "single_prediction": "POST /predict",
            "batch_prediction": "POST /predict-batch",
        },
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400

        result = validate_and_score([data])[0]
        return jsonify({"success": True, **result}), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }), 400


@app.route("/predict-batch", methods=["POST"])
def predict_batch():
    try:
        data = request.get_json()
        claims = data.get("claims", []) if data else []

        if not claims:
            return jsonify({"success": False, "error": "No claims provided"}), 400

        results = validate_and_score(claims)
        summary = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}

        for item in results:
            summary[item["priority"]] += 1

        return jsonify({
            "success": True,
            "total_claims": len(results),
            "results": results,
            "summary": summary,
        }), 200

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=app.config["DEBUG"])
