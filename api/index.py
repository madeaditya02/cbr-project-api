from __future__ import annotations

import csv
import hmac
import secrets
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable

from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from werkzeug.security import check_password_hash

try:
    from .cbr_system import (
        DEFAULT_NEED_REVISE_PATH,
        DEFAULT_CASE_BASE_PATH,
        LABEL_COLUMN,
        TARGET_COLUMN,
        recommend,
        retain_case,
        solution_label,
    )
except ImportError:
    from cbr_system import (
        DEFAULT_NEED_REVISE_PATH,
        DEFAULT_CASE_BASE_PATH,
        LABEL_COLUMN,
        TARGET_COLUMN,
        recommend,
        retain_case,
        solution_label,
    )


app = Flask(__name__)

cors = CORS(app) # allow CORS for all domains on all routes.
app.config['CORS_HEADERS'] = 'Content-Type'
app.config["CORS_ALLOW_HEADERS"] = ["Content-Type", "Authorization"]

API_DIR = Path(__file__).resolve().parent
EXPERT_USERS_PATH = API_DIR / "expert_users.csv"
EXPERT_SESSION_TTL_SECONDS = 8 * 60 * 60
_expert_sessions: dict[str, dict[str, Any]] = {}


def _json_error(message: str, status_code: int):
    response = jsonify({"error": message})
    response.status_code = status_code
    return response


def _safe_expert_user(user: dict[str, str]) -> dict[str, str]:
    return {
        "id": user.get("id", ""),
        "name": user.get("name", ""),
        "email": user.get("email", ""),
    }


def _read_expert_users(path: Path = EXPERT_USERS_PATH) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(newline="", encoding="utf-8") as file:
        users = list(csv.DictReader(file))

    return [
        {
            "id": (user.get("id") or "").strip(),
            "email": (user.get("email") or "").strip(),
            "name": (user.get("name") or "").strip(),
            "password": user.get("password") or "",
        }
        for user in users
    ]


def _find_expert_by_email(email: str) -> dict[str, str] | None:
    normalized_email = email.strip().lower()
    for user in _read_expert_users():
        if user["email"].lower() == normalized_email:
            return user
    return None


def _check_bcrypt_password(stored_password: str, plain_password: str) -> bool:
    try:
        import bcrypt
    except ImportError:
        return False

    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            stored_password.encode("utf-8"),
        )
    except ValueError:
        return False


def _check_expert_password(stored_password: str, plain_password: str) -> bool:
    if not stored_password:
        return False

    if stored_password.startswith(("pbkdf2:", "scrypt:")):
        return check_password_hash(stored_password, plain_password)

    if stored_password.startswith(("$2a$", "$2b$", "$2y$")):
        return _check_bcrypt_password(stored_password, plain_password)

    return hmac.compare_digest(stored_password, plain_password)


def _create_expert_session(user: dict[str, str]) -> tuple[str, dict[str, Any]]:
    token = secrets.token_urlsafe(32)
    expires_at = int(time.time()) + EXPERT_SESSION_TTL_SECONDS
    session = {
        "expert": _safe_expert_user(user),
        "expires_at": expires_at,
    }
    _expert_sessions[token] = session
    return token, session


def _get_bearer_token() -> str | None:
    auth_header = request.headers.get("Authorization", "")
    auth_type, _, token = auth_header.partition(" ")
    if auth_type.lower() != "bearer" or not token:
        return None
    return token.strip()


def _get_current_expert_session() -> dict[str, Any] | None:
    token = _get_bearer_token()
    if token is None:
        return None

    session = _expert_sessions.get(token)
    if session is None:
        return None

    if int(session["expires_at"]) <= int(time.time()):
        _expert_sessions.pop(token, None)
        return None

    return session


def _expert_login_required(view: Callable[..., Any]):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if _get_current_expert_session() is None:
            return _json_error("Akses ditolak. Silakan login sebagai pakar.", 401)
        return view(*args, **kwargs)

    return wrapped_view


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
        return [_normalize_need_revise_case(row) for row in csv.DictReader(file)]


def _normalize_need_revise_case(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    status = str(normalized.get("status") or "").strip().lower()
    solution = str(normalized.get(TARGET_COLUMN) or "").strip()

    if not status and solution.lower() in {"pending", "revised"}:
        status = solution.lower()
        solution = ""

    normalized[TARGET_COLUMN] = solution
    normalized["status"] = status or "pending"
    return normalized


def _write_need_revise_cases(
    cases: list[dict[str, Any]],
    path: Path = DEFAULT_NEED_REVISE_PATH,
) -> None:
    columns = [
        "patient_id",
        "age",
        "bmi",
        "disease_type",
        "cholesterol",
        "glucose",
        "blood_pressure",
        TARGET_COLUMN,
        "status",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for case in cases:
            writer.writerow({column: case.get(column, "") for column in columns})


def _filter_need_revise_cases(status: str) -> list[dict[str, Any]]:
    return [
        case
        for case in _read_need_revise_cases()
        if str(case.get("status") or "").strip().lower() == status
    ]


@app.get("/")
@cross_origin()
def home():
    return jsonify(
        {
            "name": "CBR Diet Recommendation API",
            "endpoints": {
                "predict": "POST /predict",
                "expert_login": "POST /expert/login",
                "expert_logout": "POST /expert/logout",
                "expert_me": "GET /expert/me",
                "expert_pending": "GET /expert/pending",
                "expert_history": "GET /expert/history",
                "expert_validate": "POST /expert/validate",
                "revise": "GET /revise",
            },
        }
    )


@app.post("/expert/login")
@cross_origin()
def expert_login():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return _json_error("Request body harus berupa JSON object.", 400)

    email = str(payload.get("email") or "").strip()
    password = str(payload.get("password") or "")
    if not email or not password:
        return _json_error("Email dan password wajib diisi.", 400)

    try:
        user = _find_expert_by_email(email)
    except FileNotFoundError:
        return _json_error("File expert_users.csv tidak ditemukan.", 500)
    except OSError as exc:
        return _json_error(f"Gagal membaca expert_users.csv: {exc}", 500)

    if user is None or not _check_expert_password(user["password"], password):
        return _json_error("Email atau password salah.", 401)

    token, session = _create_expert_session(user)
    return jsonify(
        {
            "message": "Login pakar berhasil.",
            "token_type": "Bearer",
            "access_token": token,
            "expires_at": session["expires_at"],
            "expert": session["expert"],
        }
    )


@app.post("/expert/logout")
@cross_origin()
def expert_logout():
    token = _get_bearer_token()
    if token is not None:
        _expert_sessions.pop(token, None)

    return jsonify({"message": "Logout pakar berhasil."})


@app.get("/expert/me")
@cross_origin()
@_expert_login_required
def expert_me():
    session = _get_current_expert_session()
    return jsonify(
        {
            "expert": session["expert"],
            "expires_at": session["expires_at"],
        }
    )


@app.get("/expert/dashboard")
@cross_origin()
@_expert_login_required
def expert_dashboard():
    path = DEFAULT_CASE_BASE_PATH
    cases = []
    distribution = {}
    try:
        if not path.exists() or path.stat().st_size == 0:
            cases = []
        with path.open(newline="", encoding="utf-8") as file:
            cases = csv.DictReader(file)
            cases = list(cases)
            total_cases = len(cases)
            if total_cases > 0:
                counts = {}
                for case in cases:
                    label = case.get(TARGET_COLUMN, "unknown")
                    counts[label] = counts.get(label, 0) + 1
                distribution = [
                    {
                        "label": label,
                        "count": count,
                        "percentage": round((count / total_cases) * 100, 2)
                    }
                    for label, count in counts.items()
                ]
                distribution.sort(key=lambda item: item["count"], reverse=True)
    except OSError as exc:
        return _json_error(f"Gagal membaca case base: {exc}", 500)
    return jsonify({"count": len(cases), "distribution": distribution})


@app.get("/expert/pending")
@cross_origin()
@_expert_login_required
def expert_pending_cases():
    try:
        cases = _filter_need_revise_cases("pending")
    except OSError as exc:
        return _json_error(f"Gagal membaca need_revise_case.csv: {exc}", 500)

    return jsonify({"count": len(cases), "cases": cases})


@app.get("/expert/history")
@cross_origin()
@_expert_login_required
def expert_history_cases():
    try:
        cases = _filter_need_revise_cases("revised")
    except OSError as exc:
        return _json_error(f"Gagal membaca need_revise_case.csv: {exc}", 500)

    return jsonify({"count": len(cases), "cases": cases})


@app.post("/expert/validate")
@cross_origin()
@_expert_login_required
def expert_validate_case():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return _json_error("Request body harus berupa JSON object.", 400)

    case_id = str(payload.get("case_id") or "").strip()
    solution = solution_label(payload.get("solution"))
    if not case_id or not solution:
        return _json_error("case_id dan solution wajib diisi.", 400)

    try:
        cases = _read_need_revise_cases()
        matched_index = next(
            (
                index
                for index, case in enumerate(cases)
                if str(case.get("patient_id") or "").strip() == case_id
            ),
            None,
        )
        if matched_index is None:
            return _json_error("Case tidak ditemukan.", 404)

        case = cases[matched_index]
        if str(case.get("status") or "").strip().lower() == "revised":
            return _json_error("Case sudah divalidasi.", 409)

        retain_result = retain_case(case, solution)
        revised_case = {**case, TARGET_COLUMN: solution, "status": "revised"}
        cases[matched_index] = revised_case
        _write_need_revise_cases(cases)
    except ValueError as exc:
        return _json_error(str(exc), 400)
    except FileNotFoundError as exc:
        return _json_error(f"File data tidak ditemukan: {exc.filename}", 500)
    except OSError as exc:
        return _json_error(f"Gagal memproses validasi case: {exc}", 500)

    return jsonify(
        {
            "message": "Case berhasil direvisi dan disimpan ke case base.",
            "case": revised_case,
            "retain": retain_result,
        }
    )


@app.post("/predict")
@cross_origin()
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return _json_error("Request body harus berupa JSON object.", 400)

    try:
        k = _as_int_query("k", 5)
        threshold = _as_float_query("threshold", 0.90)
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
        "top_cases": result.get("top_cases", []),
        # "top_cases": [_case_summary(case) for case in result.get("top_cases", [])],
    }

    if result.get("need_revise") is not None:
        response_body["need_revise"] = result["need_revise"]

    return jsonify(response_body)


@app.get("/revise")
@cross_origin()
@_expert_login_required
def revise_cases():
    try:
        cases = _filter_need_revise_cases("pending")
    except OSError as exc:
        return _json_error(f"Gagal membaca need_revise_case.csv: {exc}", 500)

    return jsonify({"count": len(cases), "cases": cases})


if __name__ == "__main__":
    app.run(debug=True)
