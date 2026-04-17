"""LLM-generated rule baseline.

Strictly **optional**. Activated only when ``ENABLE_LLM_RULES=1`` is set in
the environment. Otherwise, the evaluate function loads a **cached fixture**
so that tests and dry runs never hit the network.

Fixture: ``src/agridrone/vision/rules_llm_fixtures/cached_rules.json``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_FIXTURE_DIR = Path(__file__).parent / "rules_llm_fixtures"
_FIXTURE_PATH = _FIXTURE_DIR / "cached_rules.json"


def _load_fixture() -> dict:
    if not _FIXTURE_PATH.exists():
        return {}
    try:
        return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _llm_online_disabled_reason() -> str:
    if os.environ.get("ENABLE_LLM_RULES") != "1":
        return "ENABLE_LLM_RULES != 1 (using cached fixture)"
    return "ENABLE_LLM_RULES is 1 but online call path not wired on this host"


def evaluate_llm(features, classifier_result, crop_type: str = "wheat", **_):
    """`RuleEngine`-compatible. Pure-offline by default."""
    fixture = _load_fixture()
    disease_rules = fixture.get("rules", {})
    if not disease_rules:
        return None

    # Naive matching: pick the disease whose rule predicates are most satisfied
    # by the feature vector.
    scores: dict[str, float] = {}
    for disease, preds in disease_rules.items():
        hits = 0
        total = 0
        for pred in preds:
            total += 1
            feat = pred.get("feature")
            op = pred.get("op")
            thr = float(pred.get("threshold", 0.0))
            if feat is None:
                continue
            v = float(getattr(features, feat, 0.0)) if features else 0.0
            if op == ">" and v > thr:
                hits += 1
            elif op == "<" and v < thr:
                hits += 1
            elif op == ">=" and v >= thr:
                hits += 1
            elif op == "<=" and v <= thr:
                hits += 1
        if total:
            scores[disease] = hits / total
    if not scores:
        return None
    top_disease = max(scores, key=lambda k: scores[k])
    return {
        "top_disease": top_disease,
        "top_confidence": float(scores[top_disease]),
        "candidates": [
            {"disease_key": d, "rule_score": float(s)}
            for d, s in sorted(scores.items(), key=lambda x: -x[1])[:5]
        ],
        "source": "llm_generated",
        "mode": _llm_online_disabled_reason(),
    }
