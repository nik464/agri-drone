"""
kb_loader.py — Load and manage the structured disease knowledge base.

Reads diseases.json at startup, provides query functions for the reasoning engine.
Supports hot-reload: call reload() to pick up KB edits without restarting the server.
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
from loguru import logger
from datetime import datetime

_KB_PATH = Path(__file__).parent / "diseases.json"

# ── Data structures ──

@dataclass
class DiseaseProfile:
    """Complete disease profile loaded from JSON KB."""
    name: str
    display_name: str
    type: str                                # "disease_fungal" | "disease_bacterial" | "pest_insect" | "healthy"
    severity: float
    crop: str
    symptoms: list[str]
    color_signatures: list[dict]
    texture_keywords: list[str]
    affected_parts: list[str]
    confusion_diseases: list[str]
    treatment: list[str]
    urgency: str
    yield_loss_pct: str
    pathogen: str = ""
    favorable_conditions: str = ""


@dataclass
class DifferentialRule:
    """Pairwise comparison between two diseases."""
    disease_a: str
    disease_b: str
    key_differences: list[dict]              # [{feature, a_value, b_value}, ...]
    rule: str                                # Human-readable decision rule


@dataclass
class SeasonalRisk:
    """Disease risk levels for a crop growth stage."""
    stage: str
    months: list[str]
    temperature_range_c: list[float]
    high_risk: list[str]
    moderate_risk: list[str]
    low_risk: list[str]
    unlikely: list[str]


# ── Module-level state ──

_PROFILES: dict[str, DiseaseProfile] = {}
_DIFFERENTIALS: dict[str, DifferentialRule] = {}
_SEASONAL: dict[str, dict[str, SeasonalRisk]] = {}   # crop → stage → risk
_META: dict = {}
_LOADED: bool = False


def _parse_profiles(data: dict) -> dict[str, DiseaseProfile]:
    """Parse profile dicts from JSON into DiseaseProfile dataclasses."""
    profiles = {}
    for key, p in data.get("profiles", {}).items():
        try:
            profiles[key] = DiseaseProfile(
                name=key,
                display_name=p.get("display_name", key.replace("_", " ").title()),
                type=p.get("type", "disease_fungal"),
                severity=p.get("severity", 0.5),
                crop=p.get("crop", "wheat"),
                symptoms=p.get("symptoms", []),
                color_signatures=p.get("color_signatures", []),
                texture_keywords=p.get("texture_keywords", []),
                affected_parts=p.get("affected_parts", []),
                confusion_diseases=p.get("confusion_diseases", []),
                treatment=p.get("treatment", []),
                urgency=p.get("urgency", "within_7_days"),
                yield_loss_pct=p.get("yield_loss_pct", "unknown"),
                pathogen=p.get("pathogen", ""),
                favorable_conditions=p.get("favorable_conditions", ""),
            )
        except Exception as exc:
            logger.warning(f"Skipping malformed KB profile '{key}': {exc}")
    return profiles


def _parse_differentials(data: dict) -> dict[str, DifferentialRule]:
    """Parse differential diagnosis tables from JSON."""
    diffs = {}
    for key, d in data.get("differential_diagnosis", {}).items():
        if key.startswith("_"):
            continue
        try:
            diffs[key] = DifferentialRule(
                disease_a=d["disease_a"],
                disease_b=d["disease_b"],
                key_differences=d.get("key_differences", []),
                rule=d.get("rule", ""),
            )
        except Exception as exc:
            logger.warning(f"Skipping malformed differential '{key}': {exc}")
    return diffs


def _parse_seasonal(data: dict) -> dict[str, dict[str, SeasonalRisk]]:
    """Parse seasonal context from JSON."""
    seasonal = {}
    for crop, stages in data.get("seasonal_context", {}).items():
        if crop.startswith("_"):
            continue
        seasonal[crop] = {}
        for stage_name, s in stages.items():
            if stage_name.startswith("_"):
                continue
            try:
                seasonal[crop][stage_name] = SeasonalRisk(
                    stage=stage_name,
                    months=s.get("months", []),
                    temperature_range_c=s.get("temperature_range_c", []),
                    high_risk=s.get("high_risk", []),
                    moderate_risk=s.get("moderate_risk", []),
                    low_risk=s.get("low_risk", []),
                    unlikely=s.get("unlikely", []),
                )
            except Exception as exc:
                logger.warning(f"Skipping malformed seasonal stage '{crop}/{stage_name}': {exc}")
    return seasonal


# ── Public API ──

def load(path: Path | str | None = None) -> None:
    """Load (or reload) the knowledge base from JSON file."""
    global _PROFILES, _DIFFERENTIALS, _SEASONAL, _META, _LOADED

    kb_path = Path(path) if path else _KB_PATH
    if not kb_path.is_file():
        logger.error(f"Knowledge base not found: {kb_path}")
        _LOADED = False
        return

    try:
        with open(kb_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        _META = data.get("_meta", {})
        _PROFILES = _parse_profiles(data)
        _DIFFERENTIALS = _parse_differentials(data)
        _SEASONAL = _parse_seasonal(data)
        _LOADED = True

        logger.info(
            f"Knowledge base loaded: {len(_PROFILES)} profiles, "
            f"{len(_DIFFERENTIALS)} differentials, "
            f"{sum(len(s) for s in _SEASONAL.values())} seasonal stages "
            f"(v{_META.get('version', '?')})"
        )
    except json.JSONDecodeError as exc:
        logger.error(f"Invalid JSON in KB file {kb_path}: {exc}")
        _LOADED = False
    except Exception as exc:
        logger.error(f"Failed to load KB: {exc}")
        _LOADED = False


def reload() -> None:
    """Hot-reload the KB from disk. Call after editing diseases.json."""
    load()


def is_loaded() -> bool:
    return _LOADED


def get_all_profiles() -> dict[str, DiseaseProfile]:
    """Return all disease profiles."""
    if not _LOADED:
        load()
    return _PROFILES


def get_profile(key: str) -> DiseaseProfile | None:
    """Get a single disease profile by key."""
    if not _LOADED:
        load()
    return _PROFILES.get(key)


def get_profiles_for_crop(crop: str) -> dict[str, DiseaseProfile]:
    """Get all profiles for a specific crop (wheat/rice)."""
    if not _LOADED:
        load()
    return {k: p for k, p in _PROFILES.items() if p.crop == crop or p.crop == "both"}


def get_disease_profiles(crop: str | None = None) -> dict[str, DiseaseProfile]:
    """Get disease profiles (excluding healthy), optionally filtered by crop."""
    if not _LOADED:
        load()
    profiles = _PROFILES
    if crop:
        profiles = {k: p for k, p in profiles.items() if p.crop == crop or p.crop == "both"}
    return {k: p for k, p in profiles.items() if p.type != "healthy"}


def get_differential(disease_a: str, disease_b: str) -> DifferentialRule | None:
    """Get the differential diagnosis rule for two diseases (order-independent)."""
    if not _LOADED:
        load()
    key1 = f"{disease_a}__vs__{disease_b}"
    key2 = f"{disease_b}__vs__{disease_a}"
    return _DIFFERENTIALS.get(key1) or _DIFFERENTIALS.get(key2)


def get_all_differentials() -> dict[str, DifferentialRule]:
    """Return all differential diagnosis rules."""
    if not _LOADED:
        load()
    return _DIFFERENTIALS


def get_seasonal_risk(crop: str, month: str | None = None) -> list[SeasonalRisk]:
    """Get seasonal risk stages for a crop. If month given, return only matching stages."""
    if not _LOADED:
        load()
    stages = _SEASONAL.get(crop, {})
    if month is None:
        return list(stages.values())

    matching = []
    for stage in stages.values():
        if month in stage.months:
            matching.append(stage)
    return matching


def get_current_seasonal_risk(crop: str) -> SeasonalRisk | None:
    """Get the seasonal risk for the current month."""
    current_month = datetime.now().strftime("%B")  # e.g., "April"
    risks = get_seasonal_risk(crop, current_month)
    return risks[0] if risks else None


def get_seasonal_adjustment(disease_key: str, crop: str, month: str | None = None) -> float:
    """Return a confidence adjustment multiplier based on seasonal likelihood.

    Returns:
      1.2  if disease is high_risk this season
      1.0  if moderate_risk
      0.8  if low_risk
      0.5  if unlikely
      1.0  if no seasonal data available
    """
    if month is None:
        month = datetime.now().strftime("%B")

    risks = get_seasonal_risk(crop, month)
    if not risks:
        return 1.0

    for stage in risks:
        if disease_key in stage.high_risk:
            return 1.2
        if disease_key in stage.moderate_risk:
            return 1.0
        if disease_key in stage.low_risk:
            return 0.8
        if disease_key in stage.unlikely:
            return 0.5

    return 1.0


def get_kb_info() -> dict:
    """Return KB metadata for API info endpoints."""
    if not _LOADED:
        load()
    return {
        "version": _META.get("version", "unknown"),
        "last_updated": _META.get("last_updated", "unknown"),
        "total_profiles": len(_PROFILES),
        "total_differentials": len(_DIFFERENTIALS),
        "crops": list(set(p.crop for p in _PROFILES.values())),
        "types": list(set(p.type for p in _PROFILES.values())),
        "disease_count": len([p for p in _PROFILES.values() if p.type != "healthy"]),
        "pest_count": len([p for p in _PROFILES.values() if p.type == "pest_insect"]),
    }
