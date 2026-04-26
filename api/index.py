import json
import os
import tempfile
from pathlib import Path

from flask import Flask, jsonify, request, send_file
import joblib
import pandas as pd

# Path configuration
ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
BUNDLE_PATH = ARTIFACTS / "model_bundle.joblib"

app = Flask(__name__)

# Global variable for the model bundle
_bundle = None

def get_bundle():
    global _bundle
    if _bundle is None:
        if BUNDLE_PATH.exists():
            _bundle = joblib.load(BUNDLE_PATH)
        else:
            print(f"Warning: Bundle not found at {BUNDLE_PATH}")
    return _bundle

@app.route("/api/health")
def health():
    bundle = get_bundle()
    if not bundle:
        return jsonify({"error": "Model bundle not loaded"}), 500
    return jsonify({
        "status": "ready",
        "best_model": bundle["best_model_name"],
        "features": bundle["feature_names"],
        "metrics": bundle["metrics"][bundle["best_model_name"]],
    })

@app.route("/api/sample")
def get_sample():
    bundle = get_bundle()
    if not bundle:
        return jsonify({"error": "Model bundle not loaded"}), 500
    return jsonify({
        "feature_names": bundle["feature_names"],
        "feature_means": bundle["feature_means"],
        "benign_example": bundle["benign_example"],
        "malignant_example": bundle["malignant_example"],
        "target_names": bundle["target_names"],
        "metrics": bundle["metrics"],
        "best_model": bundle["best_model_name"],
        "cluster_summary": bundle["cluster_summary"],
        "class_distribution": bundle.get("class_distribution", {}),
        "feature_importances": bundle["feature_importances"],
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    bundle = get_bundle()
    if not bundle:
        return jsonify({"error": "Model bundle not loaded"}), 500
    
    try:
        payload = request.json
        features = bundle["feature_names"]
        values = {name: float(payload["features"][name]) for name in features}
        frame = pd.DataFrame([values], columns=features)
        
        prediction = int(bundle["model"].predict(frame)[0])
        probabilities = bundle["model"].predict_proba(frame)[0]
        labels = bundle["target_names"]
        
        return jsonify({
            "prediction_index": prediction,
            "prediction_label": labels[prediction],
            "probabilities": {
                labels[i]: float(probabilities[i]) for i in range(len(labels))
            },
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/train", methods=["POST"])
def train():
    # Training is restricted on Vercel due to read-only filesystem and timeouts
    if os.environ.get("VERCEL"):
        return jsonify({
            "error": "Training is disabled in the production environment. Please run training locally and push the updated model."
        }), 403
    
    try:
        import sys
        sys.path.append(str(ROOT / "src"))
        from train_model import main as train_main
        
        train_main()
        global _bundle
        _bundle = joblib.load(BUNDLE_PATH) # Reload
        
        return jsonify({
            "status": "trained",
            "best_model": _bundle["best_model_name"],
            "metrics": _bundle["metrics"][_bundle["best_model_name"]],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/report/generate", methods=["POST"])
def generate_report_api():
    try:
        import sys
        sys.path.append(str(ROOT / "src"))
        import export_report_docx
        from export_report_docx import generate_report
        
        payload = request.json
        input_values = payload.get("input_values")
        prediction = payload.get("prediction")
        average_deviation = payload.get("average_deviation")
        
        # Use /tmp for report generation in production
        output_dir = Path("/tmp") if os.environ.get("VERCEL") else (ROOT / "report")
        output_dir.mkdir(exist_ok=True)
        
        # We need to monkey-patch or pass the output path to generate_report
        # For simplicity, we'll just let it generate and then we find the latest file or return it
        # Actually, let's modify generate_report to be more flexible if needed, 
        # but the current one returns the path.
        
        report_path = generate_report(
            sample_snapshot=input_values,
            prediction=prediction,
            average_deviation=average_deviation
        )
        
        return jsonify({
            "status": "ready",
            "file_name": report_path.name,
            "download_url": f"/api/report/download?name={report_path.name}"
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/report/download")
def download_report():
    name = request.args.get("name")
    if not name:
        return "Missing filename", 400
    
    # Check both report dir and /tmp
    paths = [
        ROOT / "report" / name,
        Path("/tmp") / name
    ]
    
    for p in paths:
        if p.exists():
            return send_file(str(p), as_attachment=True)
            
    return "File not found", 404

# For Vercel, the app object must be available at the module level
# and usually named 'app'
if __name__ == "__main__":
    app.run(port=8000)
