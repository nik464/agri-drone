"""Learned-rule baseline (decision tree / rule list).

Trains a `sklearn.tree.DecisionTreeClassifier` on the existing 20+ visual
feature vector produced by ``feature_extractor``. Serialisable as a set of
``if X > t then disease`` rules (interpretable by construction, so it plays
the same role as the handcrafted engine but is learned from data).

This module is new and non-breaking: ``rule_engine.py`` is left untouched
and remains the default engine. The matrix runner dispatches here only when
``rule_engine: learned_tree`` is specified.

Training is GPU-free; use ``train_learned.py`` helper (TODO) or the matrix
runner. The evaluate function below gracefully degrades to a ``None``-like
result when no serialised tree is found on disk, so that ``--dry-run`` and
CI-without-weights still work.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

_ARTIFACT_DIR = PROJECT_ROOT / "models" / "new" / "learned_rules"
_DEFAULT_TREE = _ARTIFACT_DIR / "decision_tree_v1.json"


def _features_to_vec(features) -> list[float]:
    """Flatten the `ImageFeatures` dataclass to a numeric vector.

    We deliberately inspect ``__dict__`` rather than hard-coding field names
    so this works even if `feature_extractor.py` adds fields later.
    """
    vec: list[float] = []
    if features is None:
        return vec
    d = features.__dict__ if hasattr(features, "__dict__") else dict(features)
    for _, v in sorted(d.items()):
        if isinstance(v, (int, float)):
            vec.append(float(v))
    return vec


def _load_tree(path: Path = _DEFAULT_TREE) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def evaluate_learned(features, classifier_result, crop_type: str = "wheat", **_):
    """Compatible with `RuleEngine` protocol.

    Returns a minimal dict-like object. The matrix runner treats None as
    "no rule engine override".
    """
    tree = _load_tree()
    if tree is None or not features:
        return None

    # Trivial interpretation: each node is {"feat": name, "thr": x,
    # "left": node_or_label, "right": node_or_label}.
    cursor = tree
    for _ in range(64):  # depth safety
        if isinstance(cursor, str):
            # leaf label
            return {
                "top_disease": cursor,
                "top_confidence": 0.5,
                "candidates": [{"disease_key": cursor, "rule_score": 0.5}],
                "source": "learned_tree",
            }
        feat_name = cursor.get("feat")
        thr = cursor.get("thr", 0.0)
        v = float(getattr(features, feat_name, 0.0)) if feat_name else 0.0
        cursor = cursor["left"] if v <= thr else cursor["right"]
    return None


def train_from_predictions_csv(
    preds_csv: Path, feature_fn, *, max_depth: int = 6, out_path: Path = _DEFAULT_TREE,
) -> dict:
    """Train a decision tree from a per-image predictions CSV + feature extractor.

    Kept as a stub signature so the matrix runner can call into it once a
    GPU-computed feature cache exists. See ``docs/training_recipe.md`` for
    the expected CSV layout.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stub = {
        "status": "stub",
        "notes": (
            "train_from_predictions_csv is a placeholder. Run on a host with "
            "scikit-learn available and wire up feature extraction."
        ),
        "max_depth": max_depth,
    }
    out_path.write_text(json.dumps(stub, indent=2), encoding="utf-8")
    return stub
