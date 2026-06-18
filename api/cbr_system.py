from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CASE_BASE_PATH = BASE_DIR / "diet_case_base.csv"
DEFAULT_NEED_REVISE_PATH = BASE_DIR / "need_revise_case.csv"

TARGET_COLUMN = "diet_recommendation"
LABEL_COLUMN = TARGET_COLUMN
PATIENT_ID_COLUMN = "patient_id"
RAW_NUMERIC_FEATURES = ["bmi", "cholesterol", "blood_pressure", "age", "glucose"]
RAW_CASE_COLUMNS = [
    PATIENT_ID_COLUMN,
    "age",
    "bmi",
    "disease_type",
    "cholesterol",
    "glucose",
    "blood_pressure",
    TARGET_COLUMN,
]
NEED_REVISE_COLUMNS = RAW_CASE_COLUMNS + ["status"]

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

WEIGHT_CONFIG = {
    "disease_type": 0.30,
    "bmi": 0.20,
    "cholesterol": 0.15,
    "blood_pressure": 0.15,
    "age": 0.10,
    "glucose": 0.10,
}

_CASE_BASE_CACHE: dict[str, Any] | None = None


def _to_float(value: Any, default: float | None = 0.0) -> float:
    if value is None or value == "":
        if default is None:
            raise ValueError("Nilai numeric wajib diisi.")
        return default
    return float(value)


def _normalize_disease(value: Any) -> str:
    disease = str(value or "none").strip().lower().replace("-", " ").replace("_", " ")
    return DISEASE_ALIASES.get(disease, disease)


def _normalize_target(value: Any) -> str:
    label = str(value or "").strip().lower()
    aliases = {
        "0": "balanced",
        "1": "low_carb",
        "2": "low_sodium",
        "balanced": "balanced",
        "low carb": "low_carb",
        "low_carb": "low_carb",
        "low sodium": "low_sodium",
        "low_sodium": "low_sodium",
    }
    return aliases.get(label, label)


def solution_label(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return _normalize_target(value)


def _clean_raw_case(row: dict[str, Any], require_target: bool = True) -> dict[str, Any]:
    missing_features = [feature for feature in RAW_NUMERIC_FEATURES if feature not in row]
    if missing_features:
        raise ValueError(f"Input kurang fitur: {', '.join(missing_features)}")

    cleaned = {
        PATIENT_ID_COLUMN: str(row.get(PATIENT_ID_COLUMN, "")).strip(),
        "age": _to_float(row.get("age"), default=None),
        "bmi": _to_float(row.get("bmi"), default=None),
        "disease_type": _normalize_disease(row.get("disease_type")),
        "cholesterol": _to_float(row.get("cholesterol"), default=None),
        "glucose": _to_float(row.get("glucose"), default=None),
        "blood_pressure": _to_float(row.get("blood_pressure"), default=None),
    }

    if require_target:
        if TARGET_COLUMN not in row or row.get(TARGET_COLUMN) == "":
            raise ValueError(f"Kolom {TARGET_COLUMN} wajib ada pada case base.")
        cleaned[TARGET_COLUMN] = _normalize_target(row[TARGET_COLUMN])
    elif TARGET_COLUMN in row:
        cleaned[TARGET_COLUMN] = _normalize_target(row[TARGET_COLUMN])
    
    # print(cleaned)

    return cleaned


def _fill_missing_numeric_with_median(cases: list[dict[str, Any]]) -> None:
    for feature in RAW_NUMERIC_FEATURES:
        values = sorted(
            _to_float(case.get(feature), default=0.0)
            for case in cases
            if case.get(feature) not in {None, ""}
        )
        if not values:
            median = 0.0
        elif len(values) % 2:
            median = values[len(values) // 2]
        else:
            mid = len(values) // 2
            median = (values[mid - 1] + values[mid]) / 2

        for case in cases:
            if case.get(feature) in {None, ""}:
                case[feature] = median


def load_case_base(path: str | Path = DEFAULT_CASE_BASE_PATH) -> list[dict[str, Any]]:
    case_base_path = Path(path)
    with case_base_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"Case base kosong: {case_base_path}")

        cases = [_clean_raw_case(row, require_target=True) for row in reader]

    if not cases:
        raise ValueError(f"Case base kosong: {case_base_path}")

    _fill_missing_numeric_with_median(cases)
    return cases


def case_base_columns(path: str | Path = DEFAULT_CASE_BASE_PATH) -> list[str]:
    with Path(path).open(newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        try:
            return next(reader)
        except StopIteration as exc:
            raise ValueError(f"Case base kosong: {path}") from exc


def bmi_category(value: float) -> str:
    if value < 18.5:
        return "UW"
    if value < 25.0:
        return "NR"
    if value < 30.0:
        return "OW"
    return "OB"


def bp_category(value: float) -> str:
    if value < 120:
        return "Low"
    if value < 130:
        return "Normal"
    if value < 140:
        return "High1"
    return "High2"


def build_index_key(disease: Any, bmi_value: Any, blood_pressure_value: Any) -> str:
    disease_key = _normalize_disease(disease).replace(" ", "_")
    return (
        f"{disease_key}_"
        f"{bmi_category(_to_float(bmi_value, default=None))}_"
        f"{bp_category(_to_float(blood_pressure_value, default=None))}"
    )


def _fit_minmax(cases: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    ranges = {}
    for feature in RAW_NUMERIC_FEATURES:
        values = [_to_float(case[feature], default=0.0) for case in cases]
        ranges[feature] = (min(values), max(values))
    return ranges


def _scale_feature(feature: str, value: Any, ranges: dict[str, tuple[float, float]]) -> float:
    minimum, maximum = ranges[feature]
    numeric_value = _to_float(value, default=None)
    if maximum == minimum:
        return 0.0
    return max(0.0, min(1.0, (numeric_value - minimum) / (maximum - minimum)))


def _disease_columns(cases: list[dict[str, Any]]) -> list[str]:
    diseases = sorted({_normalize_disease(case["disease_type"]) for case in cases})
    return [f"disease_type_{disease}" for disease in diseases]


def _feature_order(disease_cols: list[str]) -> list[str]:
    return RAW_NUMERIC_FEATURES + disease_cols


def _column_weights(disease_cols: list[str]) -> dict[str, float]:
    weights = {feature: WEIGHT_CONFIG[feature] for feature in RAW_NUMERIC_FEATURES}
    if not disease_cols:
        return weights

    disease_weight = WEIGHT_CONFIG["disease_type"] / len(disease_cols)
    for column in disease_cols:
        weights[column] = disease_weight

    total = sum(weights.values())
    return {feature: value / total for feature, value in weights.items()}


def _preprocess_case(
    raw_case: dict[str, Any],
    ranges: dict[str, tuple[float, float]],
    disease_cols: list[str],
) -> dict[str, float]:
    cleaned = _clean_raw_case(raw_case, require_target=False)
    disease = cleaned["disease_type"]
    known_diseases = {column.replace("disease_type_", "") for column in disease_cols}
    # if disease not in known_diseases:
    #     raise ValueError(
    #         "disease_type tidak dikenali. Gunakan salah satu: "
    #         f"{', '.join(sorted(known_diseases))}."
    #     )

    processed = {
        feature: _scale_feature(feature, cleaned[feature], ranges)
        for feature in RAW_NUMERIC_FEATURES
    }
    for column in disease_cols:
        processed[column] = 1.0 if column == f"disease_type_{disease}" else 0.0
    return processed


def _vectorize(processed_case: dict[str, float], feature_order: list[str]) -> list[float]:
    return [processed_case[feature] for feature in feature_order]


def _build_case_base_cache(path: str | Path = DEFAULT_CASE_BASE_PATH) -> dict[str, Any]:
    raw_cases = load_case_base(path)
    ranges = _fit_minmax(raw_cases)
    disease_cols = _disease_columns(raw_cases)
    feature_order = _feature_order(disease_cols)
    weights = _column_weights(disease_cols)
    weight_vector = [weights[feature] for feature in feature_order]

    processed_cases = [
        _preprocess_case(case, ranges, disease_cols)
        for case in raw_cases
    ]
    vectors = [_vectorize(case, feature_order) for case in processed_cases]
    targets = [case[TARGET_COLUMN] for case in raw_cases]

    hash_table: dict[str, list[int]] = defaultdict(list)
    for index, case in enumerate(raw_cases):
        key = build_index_key(
            case["disease_type"],
            case["bmi"],
            case["blood_pressure"],
        )
        hash_table[key].append(index)

    return {
        "raw_cases": raw_cases,
        "processed_cases": processed_cases,
        "vectors": vectors,
        "targets": targets,
        "ranges": ranges,
        "disease_cols": disease_cols,
        "feature_order": feature_order,
        "weights": weights,
        "weight_vector": weight_vector,
        "hash_table": dict(hash_table),
    }


def get_case_base_cache(
    path: str | Path = DEFAULT_CASE_BASE_PATH,
    force_reload: bool = False,
) -> dict[str, Any]:
    global _CASE_BASE_CACHE
    if _CASE_BASE_CACHE is None or force_reload:
        _CASE_BASE_CACHE = _build_case_base_cache(path)
    return _CASE_BASE_CACHE


def _weighted_distance(
    query_vector: list[float],
    case_vector: list[float],
    weight_vector: list[float],
) -> float:
    total = 0.0
    for query_value, case_value, weight in zip(query_vector, case_vector, weight_vector):
        total += weight * ((case_value - query_value) ** 2)
    return math.sqrt(total)


def _rank_indexed_candidates(
    query_case: dict[str, Any],
    cache: dict[str, Any],
    k: int,
) -> tuple[list[dict[str, Any]], str, int, bool, dict[str, float]]:
    processed_query = _preprocess_case(
        query_case,
        cache["ranges"],
        cache["disease_cols"],
    )
    query_vector = _vectorize(processed_query, cache["feature_order"])
    cleaned_query = _clean_raw_case(query_case, require_target=False)
    index_key = build_index_key(
        cleaned_query["disease_type"],
        cleaned_query["bmi"],
        cleaned_query["blood_pressure"],
    )

    candidates = cache["hash_table"].get(index_key, [])
    used_fallback = len(candidates) < k
    if used_fallback:
        candidates = list(range(len(cache["raw_cases"])))

    ranked = []
    for index in candidates:
        distance = _weighted_distance(
            query_vector,
            cache["vectors"][index],
            cache["weight_vector"],
        )
        similarity = max(0.0, min(1.0, 1.0 - distance))
        result = dict(cache["raw_cases"][index])
        result["case_index"] = index
        result["index_key"] = index_key
        result["weighted_euclidean_distance"] = distance
        result["global_similarity"] = similarity
        # print(result)
        ranked.append(result)

    ranked.sort(key=lambda case: (case["weighted_euclidean_distance"], -case["global_similarity"]))
    return ranked, index_key, len(candidates), used_fallback, processed_query


def _majority_vote(top_cases: list[dict[str, Any]]) -> str:
    counts = Counter(case[TARGET_COLUMN] for case in top_cases)
    max_count = max(counts.values())
    tied_labels = {label for label, count in counts.items() if count == max_count}
    if len(tied_labels) == 1:
        return next(iter(tied_labels))

    avg_distances = {}
    for label in tied_labels:
        distances = [
            case["weighted_euclidean_distance"]
            for case in top_cases
            if case[TARGET_COLUMN] == label
        ]
        avg_distances[label] = sum(distances) / len(distances)
    return min(avg_distances, key=avg_distances.get)


def retrieve(
    query_case: dict[str, Any],
    k: int = 5,
    similarity_threshold: float = 0.75,
    cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if k <= 0:
        raise ValueError("k harus lebih dari 0.")
    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold harus berada pada rentang 0 sampai 1.")

    case_cache = cache if cache is not None else get_case_base_cache()
    ranked, index_key, candidate_count, used_fallback, processed_query = _rank_indexed_candidates(
        query_case,
        case_cache,
        k,
    )
    qualified = [
        case
        for case in ranked
        if case["global_similarity"] >= similarity_threshold
    ][:k]
    # print(qualified)

    return {
        "cases": qualified,
        "ranked_cases": ranked,
        "index_key": index_key,
        "candidate_count": candidate_count,
        "used_fallback": used_fallback,
        "processed_query": processed_query,
    }


def revise_by_rules(raw_case: dict[str, Any]) -> dict[str, Any] | None:
    case = _clean_raw_case(raw_case, require_target=False)
    disease = case["disease_type"]

    if disease == "diabetes" or case["glucose"] >= 126:
        return {
            "diet_recommendation": "low_carb_restrict",
            "rule": "diabetes_or_high_glucose",
            "reason": "disease_type diabetes atau glucose >= 126.",
        }

    if disease == "hypertension" or case["blood_pressure"] >= 140:
        return {
            "diet_recommendation": "low_sodium",
            "rule": "hypertension_or_high_blood_pressure",
            "reason": "disease_type hypertension atau blood_pressure >= 140.",
        }

    if disease == "obesity" or case["bmi"] >= 30:
        return {
            "diet_recommendation": "balanced",
            "rule": "obesity_or_high_bmi",
            "reason": "disease_type obesity atau bmi >= 30.",
        }

    if case["cholesterol"] >= 240:
        return {
            "diet_recommendation": "low_fat",
            "rule": "high_cholesterol",
            "reason": "cholesterol >= 240.",
        }

    return None


def _next_need_revise_patient_id(path: str | Path = DEFAULT_NEED_REVISE_PATH) -> str:
    revise_path = Path(path)
    if not revise_path.exists() or revise_path.stat().st_size == 0:
        return "REV0001"

    with revise_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return f"REV{len(rows) + 1:04d}"


def save_need_revise_case(
    raw_case: dict[str, Any],
    path: str | Path = DEFAULT_NEED_REVISE_PATH,
) -> dict[str, Any]:
    cleaned = _clean_raw_case(raw_case, require_target=False)
    row = {
        PATIENT_ID_COLUMN: cleaned.get(PATIENT_ID_COLUMN) or _next_need_revise_patient_id(path),
        "age": cleaned["age"],
        "bmi": cleaned["bmi"],
        "disease_type": cleaned["disease_type"],
        "cholesterol": cleaned["cholesterol"],
        "glucose": cleaned["glucose"],
        "blood_pressure": cleaned["blood_pressure"],
        TARGET_COLUMN: cleaned.get(TARGET_COLUMN, ""),
        "status": "pending",
    }

    revise_path = Path(path)
    revise_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not revise_path.exists() or revise_path.stat().st_size == 0

    try:
        with revise_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=NEED_REVISE_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    except OSError as exc:
        return {
            "status": "QUEUE_WRITE_FAILED",
            "message": "Case perlu revisi manual, tetapi file tidak dapat ditulis.",
            "queued_case": row,
            "error": str(exc),
        }

    return {
        "status": "QUEUED_FOR_EXPERT_REVISION",
        "message": "Case disimpan ke need_revise_case.csv untuk direvisi pakar.",
        "queued_case": row,
    }


def revise(
    query_case: dict[str, Any],
    retrieval_result: dict[str, Any] | None = None,
    need_revise_path: str | Path = DEFAULT_NEED_REVISE_PATH,
) -> dict[str, Any]:
    rule_result = revise_by_rules(query_case)
    nearest_cases = (
        retrieval_result.get("ranked_cases", [])[:5]
        if retrieval_result is not None
        else []
    )

    if rule_result is not None:
        return {
            "diet_recommendation": rule_result["diet_recommendation"],
            "diet_recommendation_label": rule_result["diet_recommendation"],
            "status": "REVISE",
            "revision_method": "RULE_BASED",
            "matched_rule": rule_result["rule"],
            "rule_reason": rule_result["reason"],
            "message": "Rekomendasi dibuat dengan proses revise berbasis rule.",
            "top_cases": nearest_cases,
        }

    queue_result = save_need_revise_case(query_case, path=need_revise_path)
    return {
        "diet_recommendation": None,
        "diet_recommendation_label": None,
        "status": "NEEDS_EXPERT_REVISION",
        "revision_method": "MANUAL_QUEUE",
        "message": (
            "Tidak ada case yang memenuhi threshold dan tidak ada rule revise "
            "yang cocok. Case disimpan untuk direvisi pakar."
        ),
        "top_cases": nearest_cases,
        "need_revise": queue_result,
    }


def recommend(
    query_case: dict[str, Any],
    k: int = 5,
    similarity_threshold: float = 0.75,
    need_revise_path: str | Path = DEFAULT_NEED_REVISE_PATH,
) -> dict[str, Any]:
    cache = get_case_base_cache()
    retrieval_result = retrieve(
        query_case,
        k=k,
        similarity_threshold=similarity_threshold,
        cache=cache,
    )
    top_cases = retrieval_result["ranked_cases"]
    # print(retrieval_result['cases'])
    if not retrieval_result['cases']:
        print("Revised")
        revised_result = revise(
            query_case,
            retrieval_result=retrieval_result,
            need_revise_path=need_revise_path,
        )
        revised_result.update(
            {
                "threshold": similarity_threshold,
                "index_key": retrieval_result["index_key"],
                "candidate_count": retrieval_result["candidate_count"],
                "used_fallback": retrieval_result["used_fallback"],
                "processed_query": retrieval_result["processed_query"],
            }
        )
        if revised_result.get("top_cases"):
            best_case = revised_result["top_cases"][0]
            revised_result["global_similarity"] = best_case["global_similarity"]
            revised_result["weighted_euclidean_distance"] = best_case[
                "weighted_euclidean_distance"
            ]
            if revised_result["status"] == "REVISE":
                # Mencegah duplikasi: Hanya retain jika similarity tertinggi (global_similarity)
                # masih di bawah 99%. Jika 99% atau 100%, berarti kasus ini sudah ada di database,
                similarity = revised_result.get('global_similarity')
                
                if similarity is None or similarity < 0.99:
                    print("[SYSTEM] Kasus Rule-Based baru terdeteksi. Menyimpan ke Case Base...")
                    # Simpan diam-diam di background
                    retain_case(query_case, revised_result['diet_recommendation'])
                    revised_result['retain_status'] = "AUTO_RETAINED"
                else:
                    revised_result['retain_status'] = "IGNORED_DUPLICATE"
        else:
            revised_result["global_similarity"] = None
            revised_result["weighted_euclidean_distance"] = None
        return revised_result

    # solution = _majority_vote(top_cases)
    best_case = retrieval_result['cases'][0]
    solution = best_case['diet_recommendation']
    return {
        "diet_recommendation": solution,
        "diet_recommendation_label": solution.replace("_", " ").title(),
        "status": "REUSE",
        "threshold": similarity_threshold,
        "index_key": retrieval_result["index_key"],
        "candidate_count": retrieval_result["candidate_count"],
        "used_fallback": retrieval_result["used_fallback"],
        "global_similarity": best_case["global_similarity"],
        "weighted_euclidean_distance": best_case["weighted_euclidean_distance"],
        "top_cases": top_cases,
        "processed_query": retrieval_result["processed_query"],
    }


def retain_case(
    raw_case: dict[str, Any],
    diet_recommendation: Any,
    path: str | Path = DEFAULT_CASE_BASE_PATH,
) -> dict[str, Any]:
    columns = case_base_columns(path)
    cleaned = _clean_raw_case(
        {**raw_case, TARGET_COLUMN: diet_recommendation},
        require_target=True,
    )
    if not cleaned.get(PATIENT_ID_COLUMN):
        cleaned[PATIENT_ID_COLUMN] = f"PNEW{len(get_case_base_cache()['raw_cases']) + 1:04d}"

    row = {column: cleaned.get(column, "") for column in columns}
    with Path(path).open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writerow(row)

    get_case_base_cache(force_reload=True)
    return {
        "status": "RETAIN",
        "message": "Case baru berhasil ditambahkan ke case base.",
        "retained_case": row,
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
    result = recommend(sample_query, k=5, similarity_threshold=0.75)
    print("Status:", result["status"])
    print("Index key:", result["index_key"])
    print("Recommendation:", result["diet_recommendation_label"])
