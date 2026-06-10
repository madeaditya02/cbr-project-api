from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CASE_BASE_PATH = BASE_DIR / "diet_case_base.csv"
TARGET_COLUMN = "diet_recommendation"

weights = {
    "disease_type": 0.30,
    "bmi": 0.20,
    "cholesterol": 0.15,
    "blood_pressure": 0.15,
    "age": 0.10,
    "glucose": 0.10,
}

NUMERIC_FEATURES = ["bmi", "cholesterol", "blood_pressure", "age", "glucose"]
DISEASE_PREFIX = "disease_type_"


def normalize_weights(raw_weights: dict[str, float]) -> dict[str, float]:
    total = sum(raw_weights.values())
    if total <= 0:
        raise ValueError("Total weight harus lebih dari 0.")
    return {feature: value / total for feature, value in raw_weights.items()}


WEIGHTS = normalize_weights(weights)


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return float(value)


def load_case_base(path: str | Path = DEFAULT_CASE_BASE_PATH) -> list[dict[str, Any]]:
    case_base_path = Path(path)
    with case_base_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        cases = []

        for row in reader:
            case = {}
            for column, value in row.items():
                if column == TARGET_COLUMN:
                    case[column] = value
                else:
                    case[column] = _to_float(value)
            cases.append(case)

    if not cases:
        raise ValueError(f"Case base kosong: {case_base_path}")

    return cases


def disease_features(case: dict[str, Any]) -> list[str]:
    return [column for column in case if column.startswith(DISEASE_PREFIX)]


def disease_type(case: dict[str, Any]) -> str:
    if "disease_type" in case:
        return str(case["disease_type"]).strip().lower()

    columns = disease_features(case)
    if not columns:
        return "none"

    selected_column = max(columns, key=lambda column: _to_float(case.get(column)))
    return selected_column.replace(DISEASE_PREFIX, "")


def prepare_query(query_case: dict[str, Any], template_case: dict[str, Any]) -> dict[str, Any]:
    query = dict(query_case)

    if "disease_type" in query:
        selected_disease = str(query["disease_type"]).strip().lower()
        for column in disease_features(template_case):
            query[column] = 1.0 if column == f"{DISEASE_PREFIX}{selected_disease}" else 0.0

    for feature in NUMERIC_FEATURES:
        query[feature] = _to_float(query.get(feature))

    for column in disease_features(template_case):
        query[column] = _to_float(query.get(column))

    return query


def local_similarity(query_case: dict[str, Any], db_case: dict[str, Any]) -> dict[str, float]:
    similarities = {}

    for feature in NUMERIC_FEATURES:
        distance = abs(_to_float(query_case.get(feature)) - _to_float(db_case.get(feature)))
        similarities[feature] = max(0.0, min(1.0, 1.0 - distance))

    similarities["disease_type"] = (
        1.0 if disease_type(query_case) == disease_type(db_case) else 0.0
    )

    return similarities


def weighted_euclidean_distance(query_case: dict[str, Any], db_case: dict[str, Any]) -> float:
    total = 0.0

    for feature in NUMERIC_FEATURES:
        distance = _to_float(query_case.get(feature)) - _to_float(db_case.get(feature))
        total += WEIGHTS[feature] * (distance**2)

    disease_distance = 0.0 if disease_type(query_case) == disease_type(db_case) else 1.0
    total += WEIGHTS["disease_type"] * (disease_distance**2)

    return math.sqrt(total)


def global_similarity(query_case: dict[str, Any], db_case: dict[str, Any]) -> float:
    max_distance = math.sqrt(sum(WEIGHTS.values()))
    distance = weighted_euclidean_distance(query_case, db_case)
    similarity = 1.0 - (distance / max_distance)
    return max(0.0, min(1.0, similarity))


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
    query = prepare_query(query_case, cases[0])
    retrieved_cases = []

    for index, db_case in enumerate(cases):
        similarities = local_similarity(query, db_case)
        distance = weighted_euclidean_distance(query, db_case)
        similarity = global_similarity(query, db_case)

        result = dict(db_case)
        result["case_index"] = index
        result["weighted_euclidean_distance"] = distance
        result["global_similarity"] = similarity

        for feature, score in similarities.items():
            result[f"local_similarity_{feature}"] = score

        if similarity >= similarity_threshold:
            retrieved_cases.append(result)

    retrieved_cases.sort(
        key=lambda case: (
            -case["global_similarity"],
            case["weighted_euclidean_distance"],
        )
    )
    return retrieved_cases[:k]


def recommend(
    query_case: dict[str, Any],
    k: int = 5,
    similarity_threshold: float = 0.75,
) -> dict[str, Any]:
    top_cases = retrieve(
        query_case,
        k=k,
        similarity_threshold=similarity_threshold,
    )

    if not top_cases:
        return {
            "diet_recommendation": None,
            "status": "NEEDS_REVISION",
            "threshold": similarity_threshold,
            "message": (
                "Tidak ada case yang memenuhi threshold "
                f"{similarity_threshold:.2f}."
            ),
            "global_similarity": None,
            "weighted_euclidean_distance": None,
            "top_cases": [],
        }

    best_case = top_cases[0]

    return {
        "diet_recommendation": best_case[TARGET_COLUMN],
        "status": "REUSE",
        "threshold": similarity_threshold,
        "global_similarity": best_case["global_similarity"],
        "weighted_euclidean_distance": best_case["weighted_euclidean_distance"],
        "top_cases": top_cases,
    }


if __name__ == "__main__":
    sample_query = {
        "bmi": 0.32,
        "cholesterol": 0.20,
        "blood_pressure": 0.35,
        "age": 0.40,
        "glucose": 0.30,
        "disease_type": "diabetes",
    }

    threshold = 0.75
    result = recommend(sample_query, k=5, similarity_threshold=threshold)

    print("Status:", result["status"])
    print("Threshold:", result["threshold"])
    print("Recommendation:", result["diet_recommendation"])

    if result["global_similarity"] is not None:
        print("Global similarity:", round(result["global_similarity"], 4))
        print("Weighted euclidean distance:", round(result["weighted_euclidean_distance"], 4))
    else:
        print(result["message"])

    print("\nTop cases:")
    for case in result["top_cases"]:
        print(
            f"case={case['case_index']}, "
            f"target={case[TARGET_COLUMN]}, "
            f"similarity={case['global_similarity']:.4f}, "
            f"distance={case['weighted_euclidean_distance']:.4f}"
        )
