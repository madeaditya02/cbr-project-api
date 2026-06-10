from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request

try:
    from .cbr_system import (
        DEFAULT_NEED_REVISE_PATH,
        LABEL_COLUMN,
        TARGET_COLUMN,
        recommend,
    )
except ImportError:
    from cbr_system import (
        DEFAULT_NEED_REVISE_PATH,
        LABEL_COLUMN,
        TARGET_COLUMN,
        recommend,
    )


app = Flask(__name__)


def _json_error(message: str, status_code: int):
    response = jsonify({"error": message})
    response.status_code = status_code
    return response


def _as_float_query(name: str, default: float) -> float:
    value = request.args.get(name)
    if value is None or value == "":
        return default
    return float(value)


def _as_int_query(name: str, default: int) -> int:
    value = request.args.get(name)
    if value is None or value == "":
        return default
    return int(value)


def _case_summary(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_index": case.get("case_index"),
        "solution": case.get(LABEL_COLUMN),
        "diet_recommendation_label": case.get(LABEL_COLUMN),
        "diet_recommendation": case.get(TARGET_COLUMN),
        "global_similarity": case.get("global_similarity"),
        "weighted_euclidean_distance": case.get("weighted_euclidean_distance"),
    }


def _read_need_revise_cases(path: Path = DEFAULT_NEED_REVISE_PATH) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []

    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


@app.get("/")
def home():
    return jsonify(
        {
            "name": "CBR Diet Recommendation API",
            "endpoints": {
                "predict": "POST /predict",
                "revise": "GET /revise",
            },
        }
    )


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return _json_error("Request body harus berupa JSON object.", 400)

    try:
        k = _as_int_query("k", 5)
        threshold = _as_float_query("threshold", 0.75)
        result = recommend(payload, k=k, similarity_threshold=threshold)
    except ValueError as exc:
        return _json_error(str(exc), 400)
    except FileNotFoundError as exc:
        return _json_error(f"File data tidak ditemukan: {exc.filename}", 500)

    solution = result.get("diet_recommendation_label")
    response_body = {
        "status": result.get("status"),
        "solution": solution,
        "diet_recommendation_label": solution,
        "threshold": result.get("threshold"),
        "global_similarity": result.get("global_similarity"),
        "weighted_euclidean_distance": result.get("weighted_euclidean_distance"),
        "revision_method": result.get("revision_method"),
        "matched_rule": result.get("matched_rule"),
        "message": result.get("message"),
        "top_cases": [_case_summary(case) for case in result.get("top_cases", [])],
    }

    if result.get("need_revise") is not None:
        response_body["need_revise"] = result["need_revise"]

    return jsonify(response_body)


@app.get("/revise")
def revise_cases():
    try:
        cases = _read_need_revise_cases()
    except OSError as exc:
        return _json_error(f"Gagal membaca need_revise_case.csv: {exc}", 500)

    return jsonify({"count": len(cases), "cases": cases})


if __name__ == "__main__":
    app.run(debug=True)
