from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CASE_BASE_PATH = BASE_DIR / "diet_case_base.csv"
DEFAULT_NEED_REVISE_PATH = BASE_DIR / "need_revise_case.csv"
TARGET_COLUMN = "diet_recommendation"
LABEL_COLUMN = "diet_recommendation_label"

RAW_NUMERIC_FEATURES = ["bmi", "cholesterol", "blood_pressure", "age", "glucose"]
DISEASE_PREFIX = "disease_type_"
DISEASE_ALIASES = {
    "": "none",
    "normal": "none",
    "no": "none",
    "none": "none",
    "tidak ada": "none",
    "diabetes": "diabetes",
    "hypertension": "hypertension",
    "hipertensi": "hypertension",
    "obesity": "obesity",
    "obesitas": "obesity",
}

# Rentang ini dipakai untuk mengubah input mentah menjadi skala 0-1 yang
# sama dengan case base saat ini. Jika input sudah 0-1, nilainya tetap dipakai.
RAW_FEATURE_RANGES = {
    "bmi": (10.0, 50.0),
    "cholesterol": (100.0, 300.0),
    "blood_pressure": (70.0, 180.0),
    "age": (0.0, 100.0),
    "glucose": (70.0, 250.0),
}

weights = {
    "disease_type": 0.30,
    "bmi": 0.20,
    "cholesterol": 0.15,
    "blood_pressure": 0.15,
    "age": 0.10,
    "glucose": 0.10,
}

NEED_REVISE_COLUMNS = [
    "bmi",
    "cholesterol",
    "blood_pressure",
    "age",
    "glucose",
    "disease_type",
    "reason",
]

SOLUTION_LABELS = {
    "0": "balanced",
    "1": "low_carb",
    "2": "low_sodium",
}


def normalize_weights(raw_weights: dict[str, float]) -> dict[str, float]:
    total = sum(raw_weights.values())
    if total <= 0:
        raise ValueError("Total weight harus lebih dari 0.")
    return {feature: value / total for feature, value in raw_weights.items()}


WEIGHTS = normalize_weights(weights)


def _to_float(value: Any, default: float | None = 0.0) -> float:
    if value is None or value == "":
        if default is None:
            raise ValueError("Nilai numeric wajib diisi.")
        return default
    return float(value)


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _normalize_feature(feature: str, value: Any) -> float:
    numeric_value = _to_float(value, default=None)
    if 0.0 <= numeric_value <= 1.0:
        return numeric_value

    minimum, maximum = RAW_FEATURE_RANGES[feature]
    if maximum <= minimum:
        raise ValueError(f"Range fitur {feature} tidak valid.")

    return _clip((numeric_value - minimum) / (maximum - minimum))


def _normalize_disease(value: Any) -> str:
    disease = str(value or "none").strip().lower().replace("-", " ").replace("_", " ")
    return DISEASE_ALIASES.get(disease, disease)


def solution_label(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return SOLUTION_LABELS.get(str(value), str(value))


def _raw_numeric_case(raw_case: dict[str, Any]) -> dict[str, float]:
    missing_features = [feature for feature in RAW_NUMERIC_FEATURES if feature not in raw_case]
    if missing_features:
        raise ValueError(f"Input kurang fitur: {', '.join(missing_features)}")

    return {
        feature: _to_float(raw_case[feature], default=None)
        for feature in RAW_NUMERIC_FEATURES
    }


def _is_normalized_case(numeric_case: dict[str, float]) -> bool:
    return all(0.0 <= value <= 1.0 for value in numeric_case.values())


def _denormalize_feature(feature: str, value: float) -> float:
    minimum, maximum = RAW_FEATURE_RANGES[feature]
    return minimum + (value * (maximum - minimum))


def raw_case_for_rules(raw_case: dict[str, Any]) -> dict[str, Any]:
    numeric_case = _raw_numeric_case(raw_case)
    if _is_normalized_case(numeric_case):
        numeric_case = {
            feature: _denormalize_feature(feature, value)
            for feature, value in numeric_case.items()
        }

    return {
        **numeric_case,
        "disease_type": _normalize_disease(raw_case.get("disease_type")),
    }


def revise_by_rules(raw_case: dict[str, Any]) -> dict[str, Any] | None:
    case = raw_case_for_rules(raw_case)
    disease = case["disease_type"]
    bmi = case["bmi"]
    cholesterol = case["cholesterol"]
    blood_pressure = case["blood_pressure"]
    glucose = case["glucose"]

    if disease == "diabetes" or glucose >= 126:
        return {
            "diet_recommendation": "1",
            "diet_recommendation_label": solution_label("1"),
            "rule": "diabetes_or_high_glucose",
            "reason": "disease_type diabetes atau glucose >= 126.",
        }

    if disease == "hypertension" or blood_pressure >= 140:
        return {
            "diet_recommendation": "2",
            "diet_recommendation_label": solution_label("2"),
            "rule": "hypertension_or_high_blood_pressure",
            "reason": "disease_type hypertension atau blood_pressure >= 140.",
        }

    if disease == "obesity" or bmi >= 30:
        return {
            "diet_recommendation": "0",
            "diet_recommendation_label": solution_label("0"),
            "rule": "obesity_or_high_bmi",
            "reason": "disease_type obesity atau bmi >= 30.",
        }

    if cholesterol >= 240:
        return {
            "diet_recommendation": "0",
            "diet_recommendation_label": solution_label("0"),
            "rule": "high_cholesterol",
            "reason": "cholesterol >= 240.",
        }

    return None


def load_case_base(path: str | Path = DEFAULT_CASE_BASE_PATH) -> list[dict[str, Any]]:
    case_base_path = Path(path)
    with case_base_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        cases = []

        for row in reader:
            case = {}
            for column, value in row.items():
                if column in {TARGET_COLUMN, LABEL_COLUMN}:
                    case[column] = value
                else:
                    case[column] = _to_float(value)
            cases.append(case)

    if not cases:
        raise ValueError(f"Case base kosong: {case_base_path}")

    return cases


def case_base_columns(path: str | Path = DEFAULT_CASE_BASE_PATH) -> list[str]:
    case_base_path = Path(path)
    with case_base_path.open(newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        try:
            return next(reader)
        except StopIteration as exc:
            raise ValueError(f"Case base kosong: {case_base_path}") from exc


def save_need_revise_case(
    raw_case: dict[str, Any],
    reason: str,
    path: str | Path = DEFAULT_NEED_REVISE_PATH,
) -> dict[str, Any]:
    need_revise_path = Path(path)
    need_revise_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not need_revise_path.exists() or need_revise_path.stat().st_size == 0

    row = {
        "bmi": raw_case.get("bmi", ""),
        "cholesterol": raw_case.get("cholesterol", ""),
        "blood_pressure": raw_case.get("blood_pressure", ""),
        "age": raw_case.get("age", ""),
        "glucose": raw_case.get("glucose", ""),
        "disease_type": _normalize_disease(raw_case.get("disease_type")),
        "reason": reason,
    }

    try:
        with need_revise_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=NEED_REVISE_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    except OSError as exc:
        return {
            "status": "QUEUE_WRITE_FAILED",
            "message": (
                "Case perlu revisi manual, tetapi file need_revise_case.csv "
                "tidak dapat ditulis pada runtime ini."
            ),
            "queued_case": row,
            "error": str(exc),
        }

    return {
        "status": "QUEUED_FOR_EXPERT_REVISION",
        "message": "Case disimpan ke need_revise_case.csv untuk direvisi manual.",
        "queued_case": row,
    }


def disease_features(case: dict[str, Any]) -> list[str]:
    return [column for column in case if column.startswith(DISEASE_PREFIX)]


def disease_type(case: dict[str, Any]) -> str:
    if "disease_type" in case:
        return _normalize_disease(case["disease_type"])

    columns = disease_features(case)
    if not columns:
        return "none"

    selected_column = max(columns, key=lambda column: _to_float(case.get(column)))
    return selected_column.replace(DISEASE_PREFIX, "")


def preprocess_case(
    raw_case: dict[str, Any],
    template_case: dict[str, Any] | None = None,
) -> dict[str, Any]:
    processed = {}
    missing_features = [feature for feature in RAW_NUMERIC_FEATURES if feature not in raw_case]
    if missing_features:
        raise ValueError(f"Input kurang fitur: {', '.join(missing_features)}")

    for feature in RAW_NUMERIC_FEATURES:
        processed[feature] = _normalize_feature(feature, raw_case[feature])

    disease = _normalize_disease(raw_case.get("disease_type"))
    disease_columns = disease_features(template_case or {})
    if not disease_columns:
        disease_columns = [
            f"{DISEASE_PREFIX}diabetes",
            f"{DISEASE_PREFIX}hypertension",
            f"{DISEASE_PREFIX}none",
            f"{DISEASE_PREFIX}obesity",
        ]

    known_diseases = {column.replace(DISEASE_PREFIX, "") for column in disease_columns}
    if disease not in known_diseases:
        raise ValueError(
            "disease_type tidak dikenali. Gunakan salah satu: "
            f"{', '.join(sorted(known_diseases))}."
        )

    for column in disease_columns:
        processed[column] = 1.0 if column == f"{DISEASE_PREFIX}{disease}" else 0.0

    if TARGET_COLUMN in raw_case:
        processed[TARGET_COLUMN] = raw_case[TARGET_COLUMN]
    if LABEL_COLUMN in raw_case:
        processed[LABEL_COLUMN] = raw_case[LABEL_COLUMN]

    return processed


def local_similarity(query_case: dict[str, Any], db_case: dict[str, Any]) -> dict[str, float]:
    similarities = {}

    for feature in RAW_NUMERIC_FEATURES:
        distance = abs(_to_float(query_case.get(feature)) - _to_float(db_case.get(feature)))
        similarities[feature] = _clip(1.0 - distance)

    similarities["disease_type"] = (
        1.0 if disease_type(query_case) == disease_type(db_case) else 0.0
    )

    return similarities


def weighted_euclidean_distance(query_case: dict[str, Any], db_case: dict[str, Any]) -> float:
    total = 0.0

    for feature in RAW_NUMERIC_FEATURES:
        distance = _to_float(query_case.get(feature)) - _to_float(db_case.get(feature))
        total += WEIGHTS[feature] * (distance**2)

    disease_distance = 0.0 if disease_type(query_case) == disease_type(db_case) else 1.0
    total += WEIGHTS["disease_type"] * (disease_distance**2)

    return math.sqrt(total)


def global_similarity(query_case: dict[str, Any], db_case: dict[str, Any]) -> float:
    max_distance = math.sqrt(sum(WEIGHTS.values()))
    distance = weighted_euclidean_distance(query_case, db_case)
    similarity = 1.0 - (distance / max_distance)
    return _clip(similarity)


def _rank_cases(
    processed_query: dict[str, Any],
    cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ranked_cases = []

    for index, db_case in enumerate(cases):
        similarities = local_similarity(processed_query, db_case)
        distance = weighted_euclidean_distance(processed_query, db_case)
        similarity = global_similarity(processed_query, db_case)

        result = dict(db_case)
        result["case_index"] = index
        result["weighted_euclidean_distance"] = distance
        result["global_similarity"] = similarity

        for feature, score in similarities.items():
            result[f"local_similarity_{feature}"] = score

        ranked_cases.append(result)

    ranked_cases.sort(
        key=lambda case: (
            -case["global_similarity"],
            case["weighted_euclidean_distance"],
        )
    )
    return ranked_cases


def retrieve(
    query_case: dict[str, Any],
    case_base: list[dict[str, Any]] | None = None,
    k: int = 5,
    similarity_threshold: float = 0.75,
) -> list[dict[str, Any]]:
    if k <= 0:
        raise ValueError("k harus lebih dari 0.")
    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold harus berada pada rentang 0 sampai 1.")

    cases = case_base if case_base is not None else load_case_base()
    processed_query = preprocess_case(query_case, cases[0])
    ranked_cases = _rank_cases(processed_query, cases)
    return [
        case
        for case in ranked_cases
        if case["global_similarity"] >= similarity_threshold
    ][:k]


def revise(
    query_case: dict[str, Any],
    case_base: list[dict[str, Any]] | None = None,
    k: int = 5,
    need_revise_path: str | Path = DEFAULT_NEED_REVISE_PATH,
) -> dict[str, Any]:
    if k <= 0:
        raise ValueError("k harus lebih dari 0.")

    cases = case_base if case_base is not None else load_case_base()
    processed_query = None
    nearest_cases = []

    try:
        processed_query = preprocess_case(query_case, cases[0])
        nearest_cases = _rank_cases(processed_query, cases)[:k]
    except ValueError as exc:
        if "disease_type tidak dikenali" not in str(exc):
            raise

    rule_result = revise_by_rules(query_case)

    if rule_result is not None:
        return {
            "diet_recommendation": rule_result["diet_recommendation"],
            "diet_recommendation_label": rule_result["diet_recommendation_label"],
            "status": "REVISE",
            "revision_method": "RULE_BASED",
            "matched_rule": rule_result["rule"],
            "message": (
                "Tidak ada case yang memenuhi threshold. Rekomendasi dibuat "
                "dengan proses revise berbasis rule."
            ),
            "rule_reason": rule_result["reason"],
            "global_similarity": (
                nearest_cases[0]["global_similarity"] if nearest_cases else None
            ),
            "weighted_euclidean_distance": (
                nearest_cases[0]["weighted_euclidean_distance"] if nearest_cases else None
            ),
            "top_cases": nearest_cases,
            "processed_query": processed_query,
        }

    queue_result = save_need_revise_case(
        query_case,
        reason="Tidak ada case yang memenuhi threshold dan tidak ada rule revise yang cocok.",
        path=need_revise_path,
    )
    return {
        "diet_recommendation": None,
        "diet_recommendation_label": None,
        "status": "NEEDS_EXPERT_REVISION",
        "revision_method": "MANUAL_QUEUE",
        "message": (
            "Tidak ada case yang memenuhi threshold dan tidak ada rule revise "
            "yang cocok. Case sudah dimasukkan ke need_revise_case.csv."
        ),
        "global_similarity": (
            nearest_cases[0]["global_similarity"] if nearest_cases else None
        ),
        "weighted_euclidean_distance": (
            nearest_cases[0]["weighted_euclidean_distance"] if nearest_cases else None
        ),
        "top_cases": nearest_cases,
        "processed_query": processed_query,
        "need_revise": queue_result,
    }


def retain_case(
    raw_case: dict[str, Any],
    diet_recommendation: Any,
    path: str | Path = DEFAULT_CASE_BASE_PATH,
) -> dict[str, Any]:
    case_base_path = Path(path)
    columns = case_base_columns(case_base_path)
    cases = load_case_base(case_base_path)
    processed_case = preprocess_case(raw_case, cases[0])
    processed_case[TARGET_COLUMN] = diet_recommendation
    processed_case[LABEL_COLUMN] = solution_label(diet_recommendation)

    row = {column: processed_case.get(column, "") for column in columns}
    with case_base_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writerow(row)

    return {
        "status": "RETAIN",
        "message": "Case baru berhasil ditambahkan ke case base.",
        "retained_case": row,
    }


def recommend(
    query_case: dict[str, Any],
    k: int = 5,
    similarity_threshold: float = 0.75,
    retain_on_revise: bool = False,
    retain_path: str | Path = DEFAULT_CASE_BASE_PATH,
    need_revise_path: str | Path = DEFAULT_NEED_REVISE_PATH,
) -> dict[str, Any]:
    cases = load_case_base(retain_path)
    try:
        top_cases = retrieve(
            query_case,
            case_base=cases,
            k=k,
            similarity_threshold=similarity_threshold,
        )
    except ValueError as exc:
        if "disease_type tidak dikenali" not in str(exc):
            raise
        top_cases = []

    if not top_cases:
        revised_result = revise(
            query_case,
            case_base=cases,
            k=k,
            need_revise_path=need_revise_path,
        )
        revised_result["threshold"] = similarity_threshold

        if (
            retain_on_revise
            and revised_result["diet_recommendation"] is not None
            and revised_result.get("processed_query") is not None
        ):
            retain_result = retain_case(
                query_case,
                revised_result["diet_recommendation"],
                path=retain_path,
            )
            revised_result["retain"] = retain_result

        return revised_result

    best_case = top_cases[0]
    processed_query = preprocess_case(query_case, cases[0])
    return {
        "diet_recommendation": best_case[TARGET_COLUMN],
        "diet_recommendation_label": best_case.get(
            LABEL_COLUMN,
            solution_label(best_case[TARGET_COLUMN]),
        ),
        "status": "REUSE",
        "threshold": similarity_threshold,
        "global_similarity": best_case["global_similarity"],
        "weighted_euclidean_distance": best_case["weighted_euclidean_distance"],
        "top_cases": top_cases,
        "processed_query": processed_query,
    }


if __name__ == "__main__":
    sample_query = {
        "bmi": 27.8,
        "cholesterol": 210,
        "blood_pressure": 135,
        "age": 42,
        "glucose": 145,
        "disease_type": "diabetes",
    }

    threshold = 0.75
    result = recommend(sample_query, k=5, similarity_threshold=threshold)

    print("Status:", result["status"])
    print("Threshold:", result["threshold"])
    print("Recommendation:", result["diet_recommendation"])
    print("Global similarity:", round(result["global_similarity"], 4))
    print("Weighted euclidean distance:", round(result["weighted_euclidean_distance"], 4))

    print("\nTop cases:")
    for case in result["top_cases"]:
        print(
            f"case={case['case_index']}, "
            f"target={case[TARGET_COLUMN]}, "
            f"similarity={case['global_similarity']:.4f}, "
            f"distance={case['weighted_euclidean_distance']:.4f}"
        )
