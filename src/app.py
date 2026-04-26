import json
import mimetypes
import sys
import traceback
import importlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "static"
ARTIFACTS = ROOT / "artifacts"
BUNDLE_PATH = ARTIFACTS / "model_bundle.joblib"


class PredictionServer(SimpleHTTPRequestHandler):
    bundle = None

    @classmethod
    def reload_bundle(cls) -> None:
        cls.bundle = joblib.load(BUNDLE_PATH)

    def translate_path(self, path: str) -> str:
        parsed_path = urlparse(path).path
        if parsed_path == "/":
            return str(STATIC / "index.html")
        if parsed_path.startswith("/artifacts/"):
            return str(ROOT / parsed_path.lstrip("/"))
        return str(STATIC / parsed_path.lstrip("/"))

    def send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path, download_name: str | None = None) -> None:
        if not path.exists():
            self.send_error(404)
            return
        data = path.read_bytes()
        content_type, _ = mimetypes.guess_type(str(path))
        self.send_response(200)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        if download_name:
            self.send_header("Content-Disposition", f'attachment; filename="{download_name}"')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        parsed_path = parsed.path
        if parsed_path == "/api/health":
            self.send_json(
                {
                    "status": "ready",
                    "best_model": self.bundle["best_model_name"],
                    "features": self.bundle["feature_names"],
                    "metrics": self.bundle["metrics"][self.bundle["best_model_name"]],
                }
            )
            return
        if parsed_path == "/api/sample":
            self.send_json(
                {
                    "feature_names": self.bundle["feature_names"],
                    "feature_means": self.bundle["feature_means"],
                    "benign_example": self.bundle["benign_example"],
                    "malignant_example": self.bundle["malignant_example"],
                    "target_names": self.bundle["target_names"],
                    "metrics": self.bundle["metrics"],
                    "best_model": self.bundle["best_model_name"],
                    "cluster_summary": self.bundle["cluster_summary"],
                    "class_distribution": self.bundle.get("class_distribution", {}),
                    "feature_importances": self.bundle["feature_importances"],
                }
            )
            return
        if parsed_path == "/api/report/download":
            query = parse_qs(parsed.query)
            file_name = query.get("name", ["Breast_Cancer_Diagnosis_Prediction_Report.docx"])[0]
            output_path = ROOT / "report" / file_name
            self.send_file(output_path, download_name=file_name)
            return
        return super().do_GET()

    def do_POST(self) -> None:
        parsed_path = urlparse(self.path).path
        if parsed_path == "/api/train":
            try:
                from train_model import main as train_main

                train_main()
                self.reload_bundle()
                best_model = self.bundle["best_model_name"]
                self.send_json(
                    {
                        "status": "trained",
                        "best_model": best_model,
                        "metrics": self.bundle["metrics"][best_model],
                    }
                )
            except Exception as exc:
                traceback.print_exc()
                self.send_json({"error": str(exc)}, status=500)
            return

        if parsed_path == "/api/report/generate":
            try:
                # 1. Ensure the model is fresh
                self.reload_bundle()
                
                # 2. FORCE RELOAD the report generator code
                # This ensures any updates to the logic or styling are applied instantly
                import export_report_docx
                importlib.reload(export_report_docx)
                from export_report_docx import generate_report

                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}
                input_values = payload.get("input_values") if isinstance(payload, dict) else None
                prediction = payload.get("prediction") if isinstance(payload, dict) else None
                average_deviation = payload.get("average_deviation") if isinstance(payload, dict) else None

                print(f"[REPORTER] Generating new case report for input sample...")
                
                output_path = generate_report(
                    sample_snapshot=input_values,
                    prediction=prediction,
                    average_deviation=average_deviation,
                )
                self.send_json(
                    {
                        "status": "ready",
                        "file_name": output_path.name,
                        "download_url": f"/api/report/download?name={output_path.name}",
                    }
                )
            except Exception as exc:
                traceback.print_exc()
                self.send_json({"error": str(exc)}, status=500)
            return

        if parsed_path == "/api/predict":
            try:
                self.reload_bundle()
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                features = self.bundle["feature_names"]
                values = {name: float(payload["features"][name]) for name in features}
                frame = pd.DataFrame([values], columns=features)
                prediction = int(self.bundle["model"].predict(frame)[0])
                probabilities = self.bundle["model"].predict_proba(frame)[0]
                labels = self.bundle["target_names"]
                self.send_json(
                    {
                        "prediction_index": prediction,
                        "prediction_label": labels[prediction],
                        "probabilities": {
                            labels[i]: float(probabilities[i]) for i in range(len(labels))
                        },
                    }
                )
            except Exception as exc:
                traceback.print_exc()
                self.send_json({"error": str(exc)}, status=500)
            return

        self.send_error(404)


def main() -> None:
    if not BUNDLE_PATH.exists():
        raise SystemExit("Model artifacts not found. Run: python src/train_model.py")
    PredictionServer.reload_bundle()
    preferred_port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    server = None
    for port in range(preferred_port, preferred_port + 20):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), PredictionServer)
            break
        except OSError:
            continue
    if server is None:
        raise SystemExit("Could not find a free local port.")
    print(f"AIML project app running at http://127.0.0.1:{server.server_port}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
