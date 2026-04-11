"""
correction_aggregator.py — Analyze misclassification patterns from feedback (E3).

Reads accumulated feedback, computes:
  - Confusion matrix (predicted vs actual)
  - Per-disease accuracy / error rate
  - Most common misclassification pairs
  - Per-model accuracy (classifier, rule engine, LLM)
  - Trending errors (recent vs overall)
  - Actionable recommendations for KB tuning
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta
from loguru import logger

from .feedback_store import get_all_feedback, get_misclassified_pairs, get_feedback_count


# ════════════════════════════════════════════════════════════════
# Core aggregation
# ════════════════════════════════════════════════════════════════

def compute_confusion_matrix(records: list[dict] | None = None) -> dict:
    """Build a confusion matrix from feedback records.

    Returns {predicted: {actual: count}}.
    """
    if records is None:
        records = get_all_feedback(limit=10000)

    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in records:
        pred = r["predicted_disease"]
        actual = r["correct_disease"]
        matrix[pred][actual] += 1

    # Convert nested defaultdicts to plain dicts for JSON serialization
    return {pred: dict(actuals) for pred, actuals in matrix.items()}


def compute_disease_accuracy(records: list[dict] | None = None) -> list[dict]:
    """Per-disease accuracy stats.

    Returns list of {disease, total_predictions, correct, incorrect, accuracy, top_confused_with}.
    """
    if records is None:
        records = get_all_feedback(limit=10000)

    stats: dict[str, dict] = defaultdict(lambda: {
        "total_predictions": 0,
        "correct": 0,
        "incorrect": 0,
        "confused_with": Counter(),
    })

    for r in records:
        pred = r["predicted_disease"]
        actual = r["correct_disease"]
        entry = stats[pred]
        entry["total_predictions"] += 1
        if pred == actual:
            entry["correct"] += 1
        else:
            entry["incorrect"] += 1
            entry["confused_with"][actual] += 1

    result = []
    for disease, s in sorted(stats.items(), key=lambda x: x[1]["incorrect"], reverse=True):
        accuracy = s["correct"] / s["total_predictions"] if s["total_predictions"] > 0 else 0
        top_confused = s["confused_with"].most_common(3)
        result.append({
            "disease": disease,
            "total_predictions": s["total_predictions"],
            "correct": s["correct"],
            "incorrect": s["incorrect"],
            "accuracy": round(accuracy, 3),
            "top_confused_with": [
                {"disease": d, "count": c} for d, c in top_confused
            ],
        })

    return result


def compute_model_accuracy(records: list[dict] | None = None) -> dict:
    """Per-model accuracy: which model gets it right most often?

    Returns {classifier: {correct, total, accuracy}, rule_engine: ..., llm: ...}.
    """
    if records is None:
        records = get_all_feedback(limit=10000)

    models = {
        "classifier": {"correct": 0, "total": 0},
        "rule_engine": {"correct": 0, "total": 0},
        "llm": {"correct": 0, "total": 0},
    }

    for r in records:
        actual = r["correct_disease"]

        cls_pred = r.get("classifier_prediction", "")
        if cls_pred:
            models["classifier"]["total"] += 1
            if cls_pred == actual:
                models["classifier"]["correct"] += 1

        re_pred = r.get("rule_engine_prediction", "")
        if re_pred:
            models["rule_engine"]["total"] += 1
            if re_pred == actual:
                models["rule_engine"]["correct"] += 1

        llm_pred = r.get("llm_prediction", "")
        if llm_pred:
            models["llm"]["total"] += 1
            if llm_pred == actual:
                models["llm"]["correct"] += 1

    for m in models.values():
        m["accuracy"] = round(m["correct"] / m["total"], 3) if m["total"] > 0 else 0.0

    return models


def compute_trending_errors(
    records: list[dict] | None = None,
    recent_days: int = 7,
) -> list[dict]:
    """Find misclassification pairs that are getting worse recently.

    Returns list of {predicted, correct, recent_count, overall_count, trend}.
    Trend: "worsening" if recent rate > overall rate, else "stable" or "improving".
    """
    if records is None:
        records = get_all_feedback(limit=10000)

    if not records:
        return []

    cutoff = (datetime.now() - timedelta(days=recent_days)).isoformat()

    overall: Counter = Counter()
    recent: Counter = Counter()

    for r in records:
        pred = r["predicted_disease"]
        actual = r["correct_disease"]
        if pred == actual:
            continue
        pair = (pred, actual)
        overall[pair] += 1
        if r.get("created_at", "") >= cutoff:
            recent[pair] += 1

    total_overall = sum(overall.values()) or 1
    total_recent = sum(recent.values()) or 1

    results = []
    for pair, count in overall.most_common(20):
        overall_rate = count / total_overall
        recent_count = recent.get(pair, 0)
        recent_rate = recent_count / total_recent if total_recent > 0 else 0

        if recent_rate > overall_rate * 1.3:
            trend = "worsening"
        elif recent_rate < overall_rate * 0.7:
            trend = "improving"
        else:
            trend = "stable"

        results.append({
            "predicted": pair[0],
            "correct": pair[1],
            "recent_count": recent_count,
            "overall_count": count,
            "overall_rate": round(overall_rate, 3),
            "recent_rate": round(recent_rate, 3),
            "trend": trend,
        })

    return results


# ════════════════════════════════════════════════════════════════
# Actionable recommendations
# ════════════════════════════════════════════════════════════════

def generate_recommendations(
    min_errors: int = 3,
    records: list[dict] | None = None,
) -> list[dict]:
    """Generate actionable tuning recommendations from feedback patterns.

    Returns list of {type, priority, description, affected_diseases, suggestion}.
    """
    if records is None:
        records = get_all_feedback(limit=10000)

    if not records:
        return []

    recs = []

    # 1. Frequent misclassification pairs → suggest differential rule tuning
    disease_acc = compute_disease_accuracy(records)
    for entry in disease_acc:
        if entry["incorrect"] >= min_errors:
            for confused in entry["top_confused_with"]:
                if confused["count"] >= min_errors:
                    recs.append({
                        "type": "differential_tuning",
                        "priority": "high" if confused["count"] >= min_errors * 2 else "medium",
                        "description": (
                            f"System predicts '{entry['disease']}' but agronomists "
                            f"correct to '{confused['disease']}' ({confused['count']} times)"
                        ),
                        "affected_diseases": [entry["disease"], confused["disease"]],
                        "suggestion": (
                            f"Increase rule weight for distinguishing features between "
                            f"'{entry['disease']}' and '{confused['disease']}'. "
                            f"Consider adding KB differential rule if missing."
                        ),
                    })

    # 2. Per-model failures → suggest model-specific fixes
    model_acc = compute_model_accuracy(records)
    for name, stats in model_acc.items():
        if stats["total"] >= 10 and stats["accuracy"] < 0.6:
            recs.append({
                "type": "model_accuracy",
                "priority": "high",
                "description": f"{name} accuracy is {stats['accuracy']:.0%} ({stats['correct']}/{stats['total']})",
                "affected_diseases": [],
                "suggestion": (
                    f"{'Retrain classifier with corrected data' if name == 'classifier' else ''}"
                    f"{'Review feature extraction thresholds' if name == 'rule_engine' else ''}"
                    f"{'Review LLM prompt templates' if name == 'llm' else ''}"
                ) or f"Investigate {name} failure patterns",
            })

    # 3. Trending errors → urgent attention
    trending = compute_trending_errors(records)
    for t in trending:
        if t["trend"] == "worsening" and t["recent_count"] >= 2:
            recs.append({
                "type": "trending_error",
                "priority": "urgent",
                "description": (
                    f"Worsening: '{t['predicted']}' → '{t['correct']}' "
                    f"({t['recent_count']} recent, {t['overall_count']} total)"
                ),
                "affected_diseases": [t["predicted"], t["correct"]],
                "suggestion": "Immediate KB weight adjustment recommended",
            })

    # Sort by priority
    priority_order = {"urgent": 0, "high": 1, "medium": 2, "low": 3}
    recs.sort(key=lambda r: priority_order.get(r["priority"], 9))

    return recs


# ════════════════════════════════════════════════════════════════
# Full report
# ════════════════════════════════════════════════════════════════

def generate_full_report() -> dict:
    """Generate a complete feedback analysis report."""
    records = get_all_feedback(limit=10000)
    total = len(records)

    if total == 0:
        return {
            "total_feedback": 0,
            "message": "No feedback data yet. Submit agronomist corrections to enable analysis.",
        }

    correct = sum(1 for r in records if r["predicted_disease"] == r["correct_disease"])

    return {
        "total_feedback": total,
        "overall_accuracy": round(correct / total, 3) if total > 0 else 0,
        "correct_predictions": correct,
        "incorrect_predictions": total - correct,
        "confusion_matrix": compute_confusion_matrix(records),
        "disease_accuracy": compute_disease_accuracy(records),
        "model_accuracy": compute_model_accuracy(records),
        "trending_errors": compute_trending_errors(records),
        "recommendations": generate_recommendations(records=records),
    }
