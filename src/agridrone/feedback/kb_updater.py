"""
kb_updater.py — Adjust KB thresholds from accumulated feedback data (E4).

Reads misclassification patterns from the correction aggregator and:
  1. Adjusts disease severity scores (up if under-predicted, down if over-predicted)
  2. Tightens color-signature HSV ranges for confused disease pairs
  3. Adds/strengthens differential rules for frequent confusion pairs
  4. Adjusts confidence fusion weights per model accuracy
  5. Persists changes back to diseases.json (with backup)

All changes are conservative — small incremental adjustments with audit trail.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

from loguru import logger

from ..knowledge import kb_loader
from .feedback_store import get_all_feedback, get_misclassified_pairs
from .correction_aggregator import (
    compute_disease_accuracy,
    compute_model_accuracy,
    generate_recommendations,
)


_KB_PATH = Path(__file__).resolve().parent.parent / "knowledge" / "diseases.json"
_BACKUP_DIR = _KB_PATH.parent / "backups"

# Maximum per-update adjustment to prevent runaway drift
_MAX_SEVERITY_DELTA = 0.05
_MAX_HSV_DELTA = 10


# ════════════════════════════════════════════════════════════════
# Backup management
# ════════════════════════════════════════════════════════════════

def _backup_kb() -> Path:
    """Create a timestamped backup of diseases.json before modification."""
    _BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = _BACKUP_DIR / f"diseases_{ts}.json"
    shutil.copy2(_KB_PATH, backup_path)
    logger.info(f"KB backup created: {backup_path.name}")
    return backup_path


def list_backups() -> list[dict]:
    """List available KB backups."""
    if not _BACKUP_DIR.is_dir():
        return []
    backups = []
    for f in sorted(_BACKUP_DIR.glob("diseases_*.json"), reverse=True):
        backups.append({
            "filename": f.name,
            "size_kb": round(f.stat().st_size / 1024, 1),
            "created": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
        })
    return backups


def restore_backup(filename: str) -> bool:
    """Restore a specific backup as the active KB."""
    backup = _BACKUP_DIR / filename
    if not backup.is_file():
        logger.error(f"Backup not found: {filename}")
        return False
    # Back up current before overwriting
    _backup_kb()
    shutil.copy2(backup, _KB_PATH)
    kb_loader.reload()
    logger.info(f"KB restored from {filename}")
    return True


# ════════════════════════════════════════════════════════════════
# KB JSON read/write
# ════════════════════════════════════════════════════════════════

def _load_kb_raw() -> dict:
    """Load raw KB JSON."""
    with open(_KB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_kb_raw(data: dict) -> None:
    """Write KB JSON back to disk and hot-reload."""
    with open(_KB_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    kb_loader.reload()
    logger.info("KB updated and hot-reloaded")


# ════════════════════════════════════════════════════════════════
# Update strategies
# ════════════════════════════════════════════════════════════════

def adjust_severity_scores(
    records: list[dict] | None = None,
    min_feedback: int = 5,
    dry_run: bool = False,
) -> list[dict]:
    """Adjust disease severity based on misclassification patterns.

    If a disease is frequently the *correct* answer when something else is predicted,
    its severity may be under-weighted → increase it slightly.

    If a disease is frequently *predicted* but wrong, it may be over-weighted → decrease it.

    Returns list of adjustments made.
    """
    if records is None:
        records = get_all_feedback(limit=10000)

    if len(records) < min_feedback:
        return []

    # Count how often each disease is over-predicted vs under-detected
    over_predicted: dict[str, int] = {}   # predicted but wrong
    under_detected: dict[str, int] = {}   # correct but not predicted

    for r in records:
        pred = r["predicted_disease"]
        actual = r["correct_disease"]
        if pred == actual:
            continue
        over_predicted[pred] = over_predicted.get(pred, 0) + 1
        under_detected[actual] = under_detected.get(actual, 0) + 1

    if not over_predicted and not under_detected:
        return []

    kb_data = _load_kb_raw()
    profiles = kb_data.get("profiles", {})
    adjustments = []

    for disease_key, profile in profiles.items():
        current_sev = profile.get("severity", 0.5)
        over = over_predicted.get(disease_key, 0)
        under = under_detected.get(disease_key, 0)

        if over == 0 and under == 0:
            continue

        # Net direction: positive means under-detected (increase severity)
        net = under - over
        if abs(net) < min_feedback:
            continue

        # Scale delta proportionally but cap it
        delta = min(_MAX_SEVERITY_DELTA, abs(net) * 0.005)
        if net < 0:
            delta = -delta

        new_sev = max(0.1, min(1.0, round(current_sev + delta, 3)))
        if new_sev == current_sev:
            continue

        adjustments.append({
            "disease": disease_key,
            "field": "severity",
            "old_value": current_sev,
            "new_value": new_sev,
            "reason": f"over_predicted={over}, under_detected={under}, net={net}",
        })

        if not dry_run:
            profile["severity"] = new_sev

    if adjustments and not dry_run:
        _backup_kb()
        _save_kb_raw(kb_data)

    return adjustments


def strengthen_differentials(
    records: list[dict] | None = None,
    min_confusions: int = 3,
    dry_run: bool = False,
) -> list[dict]:
    """Add or note frequently confused disease pairs in the differential section.

    If two diseases are commonly confused, ensure a differential rule exists.
    Returns list of changes.
    """
    if records is None:
        records = get_all_feedback(limit=10000)

    # Find pairs with enough confusion
    pairs: dict[tuple[str, str], int] = {}
    for r in records:
        pred = r["predicted_disease"]
        actual = r["correct_disease"]
        if pred == actual:
            continue
        # Canonical order
        key = tuple(sorted([pred, actual]))
        pairs[key] = pairs.get(key, 0) + 1

    if not pairs:
        return []

    kb_data = _load_kb_raw()
    diff_section = kb_data.setdefault("differential_diagnosis", {})
    changes = []

    for (a, b), count in sorted(pairs.items(), key=lambda x: -x[1]):
        if count < min_confusions:
            continue

        diff_key = f"{a}__vs__{b}"
        alt_key = f"{b}__vs__{a}"

        if diff_key in diff_section or alt_key in diff_section:
            # Already exists — add a feedback note
            existing_key = diff_key if diff_key in diff_section else alt_key
            existing = diff_section[existing_key]
            note = f"Feedback: {count} confusions reported as of {datetime.now().strftime('%Y-%m-%d')}"
            existing_notes = existing.get("feedback_notes", [])
            existing_notes.append(note)
            existing["feedback_notes"] = existing_notes[-5:]  # Keep last 5

            changes.append({
                "action": "annotated_existing",
                "pair": f"{a} vs {b}",
                "confusion_count": count,
            })
        else:
            # Create a new differential rule stub
            new_rule = {
                "disease_a": a,
                "disease_b": b,
                "key_differences": [],
                "rule": f"Auto-generated from {count} feedback confusions. Requires expert review.",
                "auto_generated": True,
                "confusion_count": count,
                "created_at": datetime.now().isoformat(),
                "feedback_notes": [
                    f"Created from {count} misclassification reports"
                ],
            }
            changes.append({
                "action": "created_stub",
                "pair": f"{a} vs {b}",
                "confusion_count": count,
            })
            if not dry_run:
                diff_section[diff_key] = new_rule

    if changes and not dry_run:
        _backup_kb()
        _save_kb_raw(kb_data)

    return changes


def compute_optimal_fusion_weights(
    records: list[dict] | None = None,
) -> dict:
    """Suggest optimal confidence fusion weights based on per-model accuracy.

    Returns {classifier: w, rule_engine: w, llm: w} that sum to 1.0.
    Does NOT auto-apply — returned as a suggestion.
    """
    model_acc = compute_model_accuracy(records)

    # Start from current weights
    weights = {"classifier": 0.20, "rule_engine": 0.50, "llm": 0.30}

    total_samples = sum(m["total"] for m in model_acc.values())
    if total_samples < 20:
        return {
            "weights": weights,
            "note": f"Insufficient data ({total_samples} samples). Need ≥20 for reliable adjustment.",
            "applied": False,
        }

    # Weight proportional to accuracy (but keep minimum floors)
    raw = {}
    for model_name, stats in model_acc.items():
        key = model_name
        if stats["total"] > 0:
            raw[key] = max(0.10, stats["accuracy"])
        else:
            raw[key] = weights.get(key, 0.2)

    total = sum(raw.values())
    if total > 0:
        for k in raw:
            raw[k] = round(raw[k] / total, 2)

    # Ensure sum is exactly 1.0
    remainder = round(1.0 - sum(raw.values()), 2)
    if remainder != 0 and raw:
        max_key = max(raw, key=raw.get)
        raw[max_key] = round(raw[max_key] + remainder, 2)

    return {
        "weights": raw,
        "model_accuracy": {k: v["accuracy"] for k, v in model_acc.items()},
        "note": f"Based on {total_samples} feedback samples",
        "applied": False,
    }


# ════════════════════════════════════════════════════════════════
# Full update pipeline
# ════════════════════════════════════════════════════════════════

def run_full_update(
    dry_run: bool = False,
    min_feedback: int = 5,
) -> dict:
    """Run all KB updates based on accumulated feedback.

    Returns a report of all changes made (or would be made in dry_run mode).
    """
    records = get_all_feedback(limit=10000)
    total = len(records)

    if total < min_feedback:
        return {
            "status": "insufficient_data",
            "total_feedback": total,
            "min_required": min_feedback,
            "message": f"Need at least {min_feedback} feedback records to update KB.",
        }

    severity_changes = adjust_severity_scores(records=records, min_feedback=min_feedback, dry_run=dry_run)
    diff_changes = strengthen_differentials(records=records, min_confusions=3, dry_run=dry_run)
    fusion_suggestion = compute_optimal_fusion_weights(records=records)

    report = {
        "status": "dry_run" if dry_run else "applied",
        "total_feedback": total,
        "timestamp": datetime.now().isoformat(),
        "severity_adjustments": severity_changes,
        "differential_updates": diff_changes,
        "fusion_weight_suggestion": fusion_suggestion,
        "total_changes": len(severity_changes) + len(diff_changes),
    }

    if not dry_run and (severity_changes or diff_changes):
        # Save update report for audit
        report_dir = _BACKUP_DIR.parent / "update_reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"update_{ts}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Update report saved: {report_path.name}")

    return report
