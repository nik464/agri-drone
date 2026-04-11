"""
temporal_tracker.py — Track disease progression across multiple scans (F4).

Uses detection_history.json to identify repeated scans of the same field/crop
and computes disease progression trends:
  - Improving (health increasing / disease resolving)
  - Worsening (health decreasing / disease spreading)
  - Stable (no significant change)
  - New outbreak (first detection of a disease)

Links by filename pattern, crop_type, and time windows.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from loguru import logger


# ════════════════════════════════════════════════════════════════
# History file path
# ════════════════════════════════════════════════════════════════

_HISTORY_FILE = Path(__file__).resolve().parent.parent.parent.parent / "outputs" / "history" / "detection_history.json"


def _load_history() -> list[dict]:
    """Load detection history from JSON file."""
    if _HISTORY_FILE.is_file():
        try:
            return json.loads(_HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


# ════════════════════════════════════════════════════════════════
# Filename normalization for matching repeat scans
# ════════════════════════════════════════════════════════════════

def _normalize_filename(filename: str) -> str:
    """Extract a base pattern from filename to match repeat scans.

    Handles patterns like:
      field_A_20240101.jpg, field_A_20240115.jpg → "field_a"
      IMG_001.jpg, IMG_002.jpg → "img"
      wheat_plot3_scan1.jpg → "wheat_plot3"
    """
    name = Path(filename).stem.lower()
    # Remove common date patterns
    name = re.sub(r'\d{4}[-_]?\d{2}[-_]?\d{2}', '', name)
    # Remove scan/image numbering
    name = re.sub(r'[-_]?(scan|img|image|photo|pic)[-_]?\d+', '', name)
    # Remove trailing numbers and separators
    name = re.sub(r'[-_]?\d+$', '', name)
    # Remove trailing separators
    name = name.strip('-_ ')
    return name or "unknown"


# ════════════════════════════════════════════════════════════════
# Temporal analysis
# ════════════════════════════════════════════════════════════════

def get_temporal_context(
    current_filename: str,
    current_disease: str,
    current_health: int,
    current_confidence: float,
    crop_type: str = "wheat",
    lookback_days: int = 90,
) -> dict:
    """Analyze disease progression for a given scan.

    Returns dict with:
      - trend: "improving" | "worsening" | "stable" | "new_outbreak" | "first_scan"
      - trend_confidence: 0-1 how confident we are in the trend
      - previous_scans: list of relevant previous detections
      - health_trajectory: list of {date, health_score} for charting
      - disease_timeline: list of {date, disease, confidence}
      - days_since_first: how long this disease has been tracked
      - recommendations: trend-specific advice
    """
    history = _load_history()
    if not history:
        return _first_scan_result(current_disease, current_health)

    # Find related scans by filename pattern and crop type
    base_name = _normalize_filename(current_filename)
    cutoff = datetime.now() - timedelta(days=lookback_days)

    related: list[dict] = []
    for entry in history:
        # Match by filename pattern
        entry_base = _normalize_filename(entry.get("filename", ""))
        if entry_base != base_name:
            continue
        # Match by crop type
        if entry.get("crop_type", "wheat") != crop_type:
            continue
        # Time window
        try:
            ts = datetime.fromisoformat(entry["timestamp"])
            if ts < cutoff:
                continue
        except (KeyError, ValueError):
            continue

        related.append(entry)

    if not related:
        # No history for this field — also check for same disease regardless of filename
        related = _find_same_disease_history(history, current_disease, crop_type, cutoff)
        if not related:
            return _first_scan_result(current_disease, current_health)

    # Sort by timestamp ascending
    related.sort(key=lambda e: e.get("timestamp", ""))

    # ── Build trajectories ──
    health_trajectory = []
    disease_timeline = []
    for entry in related:
        ts = entry.get("timestamp", "")
        date_str = ts[:10] if len(ts) >= 10 else ts
        health_trajectory.append({
            "date": date_str,
            "health_score": entry.get("health_score", 50),
            "confidence": entry.get("confidence", 0),
        })
        disease_timeline.append({
            "date": date_str,
            "disease": entry.get("disease", "Unknown"),
            "confidence": entry.get("confidence", 0),
        })

    # Add current scan to trajectory
    now_str = datetime.now().strftime("%Y-%m-%d")
    health_trajectory.append({
        "date": now_str,
        "health_score": current_health,
        "confidence": round(current_confidence * 100),
    })
    disease_timeline.append({
        "date": now_str,
        "disease": current_disease,
        "confidence": round(current_confidence * 100),
    })

    # ── Compute trend ──
    trend, trend_confidence = _compute_trend(health_trajectory, current_disease, disease_timeline)

    # ── Time span ──
    try:
        first_ts = datetime.fromisoformat(related[0]["timestamp"])
        days_since_first = (datetime.now() - first_ts).days
    except (ValueError, KeyError):
        days_since_first = 0

    # ── Recommendations ──
    recommendations = _trend_recommendations(trend, current_disease, current_health, days_since_first)

    # Build previous scans summary (last 10)
    prev_scans = []
    for entry in related[-10:]:
        prev_scans.append({
            "date": entry.get("timestamp", "")[:10],
            "disease": entry.get("disease", "Unknown"),
            "health_score": entry.get("health_score", 50),
            "confidence": entry.get("confidence", 0),
            "filename": entry.get("filename", ""),
        })

    return {
        "trend": trend,
        "trend_confidence": round(trend_confidence, 2),
        "num_previous_scans": len(related),
        "days_since_first": days_since_first,
        "previous_scans": prev_scans,
        "health_trajectory": health_trajectory,
        "disease_timeline": disease_timeline,
        "recommendations": recommendations,
    }


def _compute_trend(
    health_traj: list[dict],
    current_disease: str,
    disease_timeline: list[dict],
) -> tuple[str, float]:
    """Compute trend from health trajectory.

    Returns (trend_label, confidence).
    """
    if len(health_traj) < 2:
        return ("first_scan", 0.5)

    scores = [h["health_score"] for h in health_traj]

    # Check if disease is new (wasn't in previous scans)
    prev_diseases = set(d["disease"] for d in disease_timeline[:-1])
    if current_disease not in prev_diseases and current_disease != "Healthy":
        return ("new_outbreak", 0.8)

    # Compute linear trend using simple regression
    n = len(scores)
    x_mean = (n - 1) / 2
    y_mean = sum(scores) / n
    num = sum((i - x_mean) * (s - y_mean) for i, s in enumerate(scores))
    den = sum((i - x_mean) ** 2 for i in range(n))
    slope = num / den if den > 0 else 0

    # Recent change (last vs second-to-last)
    recent_delta = scores[-1] - scores[-2]

    # Determine trend
    if slope > 3 and recent_delta >= 0:
        trend = "improving"
        confidence = min(0.95, 0.5 + abs(slope) / 20)
    elif slope < -3 and recent_delta <= 0:
        trend = "worsening"
        confidence = min(0.95, 0.5 + abs(slope) / 20)
    else:
        trend = "stable"
        confidence = min(0.9, 0.6 + (1 - abs(slope) / 10) * 0.3) if abs(slope) < 10 else 0.4

    return (trend, confidence)


def _find_same_disease_history(
    history: list[dict],
    disease: str,
    crop_type: str,
    cutoff: datetime,
) -> list[dict]:
    """Fallback: find entries with same disease + crop type."""
    results = []
    for entry in history:
        if entry.get("disease") != disease:
            continue
        if entry.get("crop_type", "wheat") != crop_type:
            continue
        try:
            ts = datetime.fromisoformat(entry["timestamp"])
            if ts >= cutoff:
                results.append(entry)
        except (KeyError, ValueError):
            continue
    return results[-20:]  # Keep latest 20


def _first_scan_result(disease: str, health: int) -> dict:
    """Return result when no history exists."""
    return {
        "trend": "first_scan",
        "trend_confidence": 0.5,
        "num_previous_scans": 0,
        "days_since_first": 0,
        "previous_scans": [],
        "health_trajectory": [{
            "date": datetime.now().strftime("%Y-%m-%d"),
            "health_score": health,
            "confidence": 0,
        }],
        "disease_timeline": [{
            "date": datetime.now().strftime("%Y-%m-%d"),
            "disease": disease,
            "confidence": 0,
        }],
        "recommendations": [
            "First scan recorded — future scans will enable trend analysis",
            "Continue monitoring with regular scans every 3-7 days",
        ],
    }


def _trend_recommendations(
    trend: str,
    disease: str,
    health: int,
    days_tracked: int,
) -> list[str]:
    """Generate trend-specific management recommendations."""
    recs: list[str] = []

    if trend == "worsening":
        recs.append("Disease is progressing — escalate treatment urgency")
        if health < 40:
            recs.append("Health critically low — consider emergency fungicide application")
        if days_tracked > 14:
            recs.append(f"Disease has been worsening for {days_tracked} days — current treatment may be ineffective")
        recs.append("Increase scan frequency to every 2-3 days to monitor response")

    elif trend == "improving":
        recs.append("Disease is responding to treatment — continue current management")
        if health > 70:
            recs.append("Health recovering well — maintain monitoring at weekly intervals")
        if days_tracked > 21:
            recs.append(f"Recovery underway over {days_tracked} days — treatment appears effective")

    elif trend == "stable":
        recs.append("Disease level is stable — monitor for changes")
        if health < 50:
            recs.append("Stable but unhealthy — consider adjusting treatment strategy")
        recs.append("Scan again in 5-7 days to confirm stability")

    elif trend == "new_outbreak":
        recs.append(f"New disease detected: {disease} — immediate assessment recommended")
        recs.append("Compare with previous healthy status to gauge progression speed")
        recs.append("Begin scouting adjacent areas for spread")

    elif trend == "first_scan":
        recs.append("Baseline scan recorded — track progression from here")
        recs.append("Perform follow-up scan in 3-7 days for trend data")

    return recs
