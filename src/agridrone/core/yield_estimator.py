"""
yield_estimator.py — Yield loss and treatment cost estimator for crop diseases.

Estimates economic impact of a detected disease given:
  - Disease identity and severity
  - Crop type and growth stage
  - Field area in acres
  - Market prices and regional input costs

All base data is hardcoded here and mirrors the 21 diseases in diseases.json.
A future version can load from a JSON config for easy field calibration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# ════════════════════════════════════════════════════════════════
# Commodity prices (INR)
# ════════════════════════════════════════════════════════════════

# MSP / prevailing prices per quintal (100 kg)
CROP_PRICE_INR_PER_QUINTAL: dict[str, float] = {
    "wheat": 2275.0,   # 2024-25 MSP
    "rice":  2300.0,   # 2024-25 MSP (common grade)
    "maize": 2090.0,
}

# Average yield in kg per acre (rainfed + irrigated blended)
CROP_YIELD_KG_PER_ACRE: dict[str, float] = {
    "wheat": 1350.0,
    "rice":  1500.0,
    "maize": 1100.0,
}

# ════════════════════════════════════════════════════════════════
# Disease economics table
# Keys must match diseases.json profiles exactly.
# ════════════════════════════════════════════════════════════════

@dataclass
class DiseaseEconomics:
    """Per-disease yield loss percentages and treatment cost per acre."""

    # Yield loss (fraction) by severity level
    yield_loss_pct: dict[str, float]    # keys: mild, moderate, severe

    # Fungicide / pesticide cost in INR per acre (material only)
    fungicide_cost_inr: float

    # Labor cost for one spray application in INR per acre
    labor_cost_inr: float = 150.0

    # Number of spray applications typically needed
    spray_applications: int = 1

    # Efficacy: fraction of yield loss prevented by treatment (0–1)
    treatment_efficacy: float = 0.70

    # Notes
    notes: str = ""


# Complete table — all 21 diseases + 2 healthy classes
_DISEASE_TABLE: dict[str, DiseaseEconomics] = {

    # ── Healthy baseline ──────────────────────────────────────────
    "healthy_wheat": DiseaseEconomics(
        yield_loss_pct={"mild": 0.0, "moderate": 0.0, "severe": 0.0},
        fungicide_cost_inr=0.0, labor_cost_inr=0.0, spray_applications=0,
        treatment_efficacy=0.0, notes="No disease — no action required",
    ),
    "healthy_rice": DiseaseEconomics(
        yield_loss_pct={"mild": 0.0, "moderate": 0.0, "severe": 0.0},
        fungicide_cost_inr=0.0, labor_cost_inr=0.0, spray_applications=0,
        treatment_efficacy=0.0, notes="No disease — no action required",
    ),

    # ── Wheat diseases ───────────────────────────────────────────
    "wheat_fusarium_head_blight": DiseaseEconomics(
        yield_loss_pct={"mild": 0.08, "moderate": 0.28, "severe": 0.50},
        fungicide_cost_inr=320.0, labor_cost_inr=150.0, spray_applications=2,
        treatment_efficacy=0.60,
        notes="Apply metconazole at early anthesis. DON mycotoxin also degrades grain quality.",
    ),
    "wheat_yellow_rust": DiseaseEconomics(
        yield_loss_pct={"mild": 0.10, "moderate": 0.40, "severe": 0.80},
        fungicide_cost_inr=280.0, labor_cost_inr=150.0, spray_applications=2,
        treatment_efficacy=0.80,
        notes="Fast-spreading; treat at first pustule appearance. Triazoles effective.",
    ),
    "wheat_black_rust": DiseaseEconomics(
        yield_loss_pct={"mild": 0.10, "moderate": 0.40, "severe": 0.65},
        fungicide_cost_inr=300.0, labor_cost_inr=150.0, spray_applications=2,
        treatment_efficacy=0.75,
        notes="Propiconazole or tebuconazole. Monitor temperature — rust spreads fast above 25°C.",
    ),
    "wheat_brown_rust": DiseaseEconomics(
        yield_loss_pct={"mild": 0.05, "moderate": 0.20, "severe": 0.40},
        fungicide_cost_inr=250.0, labor_cost_inr=150.0, spray_applications=2,
        treatment_efficacy=0.78,
        notes="Tebuconazole 250EC @ 1 ml/L. Common across North India.",
    ),
    "wheat_powdery_mildew": DiseaseEconomics(
        yield_loss_pct={"mild": 0.04, "moderate": 0.15, "severe": 0.28},
        fungicide_cost_inr=200.0, labor_cost_inr=150.0, spray_applications=2,
        treatment_efficacy=0.75,
        notes="Sulfur 80WP or propiconazole. Worse in cool humid conditions.",
    ),
    "wheat_blast": DiseaseEconomics(
        yield_loss_pct={"mild": 0.15, "moderate": 0.45, "severe": 0.95},
        fungicide_cost_inr=400.0, labor_cost_inr=200.0, spray_applications=3,
        treatment_efficacy=0.55,
        notes="Highly aggressive. Tricyclazole + pyraclostrobin. Report outbreaks to authorities.",
    ),
    "wheat_septoria": DiseaseEconomics(
        yield_loss_pct={"mild": 0.05, "moderate": 0.18, "severe": 0.32},
        fungicide_cost_inr=280.0, labor_cost_inr=150.0, spray_applications=2,
        treatment_efficacy=0.72,
        notes="Azoxystrobin at flag-leaf stage. Wet weather promotes spread.",
    ),
    "wheat_leaf_blight": DiseaseEconomics(
        yield_loss_pct={"mild": 0.04, "moderate": 0.15, "severe": 0.25},
        fungicide_cost_inr=250.0, labor_cost_inr=150.0, spray_applications=2,
        treatment_efficacy=0.70,
        notes="Mancozeb + carbendazim combination spray.",
    ),
    "wheat_tan_spot": DiseaseEconomics(
        yield_loss_pct={"mild": 0.03, "moderate": 0.10, "severe": 0.20},
        fungicide_cost_inr=220.0, labor_cost_inr=150.0, spray_applications=1,
        treatment_efficacy=0.68,
        notes="Pyraclostrobin effective. Crop rotation and residue burial important.",
    ),
    "wheat_smut": DiseaseEconomics(
        yield_loss_pct={"mild": 0.03, "moderate": 0.15, "severe": 0.30},
        fungicide_cost_inr=180.0, labor_cost_inr=100.0, spray_applications=1,
        treatment_efficacy=0.90,
        notes="Seed treatment with carboxin + thiram provides excellent prevention.",
    ),
    "wheat_root_rot": DiseaseEconomics(
        yield_loss_pct={"mild": 0.05, "moderate": 0.15, "severe": 0.30},
        fungicide_cost_inr=200.0, labor_cost_inr=120.0, spray_applications=1,
        treatment_efficacy=0.55,
        notes="Soil-borne; fludioxonil seed treatment. Improve drainage.",
    ),
    "wheat_aphid": DiseaseEconomics(
        yield_loss_pct={"mild": 0.03, "moderate": 0.12, "severe": 0.25},
        fungicide_cost_inr=180.0, labor_cost_inr=150.0, spray_applications=1,
        treatment_efficacy=0.85,
        notes="Imidacloprid 17.8SL @ 0.5 ml/L. Single spray usually sufficient.",
    ),
    "wheat_mite": DiseaseEconomics(
        yield_loss_pct={"mild": 0.02, "moderate": 0.08, "severe": 0.15},
        fungicide_cost_inr=160.0, labor_cost_inr=150.0, spray_applications=1,
        treatment_efficacy=0.80,
        notes="Dicofol 18.5EC. Damage worse under dry, hot conditions.",
    ),
    "wheat_stem_fly": DiseaseEconomics(
        yield_loss_pct={"mild": 0.04, "moderate": 0.12, "severe": 0.28},
        fungicide_cost_inr=200.0, labor_cost_inr=150.0, spray_applications=1,
        treatment_efficacy=0.75,
        notes="Thiamethoxam 30FS seed treatment most effective preventive measure.",
    ),

    # ── Rice diseases ────────────────────────────────────────────
    "rice_blast": DiseaseEconomics(
        yield_loss_pct={"mild": 0.08, "moderate": 0.35, "severe": 0.80},
        fungicide_cost_inr=350.0, labor_cost_inr=150.0, spray_applications=2,
        treatment_efficacy=0.72,
        notes="Tricyclazole 75WP @ 0.6 g/L. Apply at neck emergence stage.",
    ),
    "rice_bacterial_blight": DiseaseEconomics(
        yield_loss_pct={"mild": 0.05, "moderate": 0.22, "severe": 0.50},
        fungicide_cost_inr=200.0, labor_cost_inr=150.0, spray_applications=2,
        treatment_efficacy=0.65,
        notes="Streptocycline 500 ppm + copper oxychloride. Bacterial — antibiotics only partially effective.",
    ),
    "rice_brown_spot": DiseaseEconomics(
        yield_loss_pct={"mild": 0.03, "moderate": 0.12, "severe": 0.30},
        fungicide_cost_inr=220.0, labor_cost_inr=150.0, spray_applications=2,
        treatment_efficacy=0.70,
        notes="Mancozeb 75WP @ 2.5 g/L. Often linked to potassium deficiency.",
    ),
    "rice_sheath_blight": DiseaseEconomics(
        yield_loss_pct={"mild": 0.05, "moderate": 0.15, "severe": 0.30},
        fungicide_cost_inr=280.0, labor_cost_inr=150.0, spray_applications=2,
        treatment_efficacy=0.73,
        notes="Validamycin 3L @ 2.5 ml/L. High plant density promotes spread.",
    ),
    "rice_leaf_scald": DiseaseEconomics(
        yield_loss_pct={"mild": 0.02, "moderate": 0.08, "severe": 0.18},
        fungicide_cost_inr=200.0, labor_cost_inr=150.0, spray_applications=1,
        treatment_efficacy=0.68,
        notes="Carbendazim 50WP @ 1 g/L. Less economic impact than blast.",
    ),
}

# Aliases — map common short keys to full keys
_ALIASES: dict[str, str] = {
    "healthy": "healthy_wheat",
    "rice_leaf_blast": "rice_blast",
    "leaf_blast": "rice_blast",
    "leaf_rust": "wheat_brown_rust",
    "brown_rust": "wheat_brown_rust",
    "yellow_rust": "wheat_yellow_rust",
    "stripe_rust": "wheat_yellow_rust",
    "black_rust": "wheat_black_rust",
    "stem_rust": "wheat_black_rust",
    "fusarium": "wheat_fusarium_head_blight",
    "fhb": "wheat_fusarium_head_blight",
    "blast": "wheat_blast",
    "septoria": "wheat_septoria",
    "powdery_mildew": "wheat_powdery_mildew",
    "tan_spot": "wheat_tan_spot",
    "smut": "wheat_smut",
    "root_rot": "wheat_root_rot",
    "aphid": "wheat_aphid",
    "mite": "wheat_mite",
    "stem_fly": "wheat_stem_fly",
    "brown_spot": "rice_brown_spot",
    "bacterial_blight": "rice_bacterial_blight",
    "sheath_blight": "rice_sheath_blight",
    "leaf_scald": "rice_leaf_scald",
    "leaf_blight": "wheat_leaf_blight",
}

# ════════════════════════════════════════════════════════════════
# Stage multipliers — disease impact varies by growth stage
# ════════════════════════════════════════════════════════════════

STAGE_MULTIPLIERS: dict[str, float] = {
    "seedling":     0.50,   # Early infection — plant can partially recover
    "tillering":    0.70,
    "jointing":     0.90,
    "booting":      1.10,
    "heading":      1.30,   # Critical window — grain fill at risk
    "flowering":    1.40,
    "grain_fill":   1.20,
    "ripening":     0.60,   # Late infection — limited impact
    "harvest":      0.20,
    "unknown":      1.00,
}

# ════════════════════════════════════════════════════════════════
# Result type
# ════════════════════════════════════════════════════════════════

@dataclass
class YieldEstimate:
    disease_key: str
    severity: str                   # mild | moderate | severe
    crop: str
    area_acres: float
    stage: str

    yield_loss_percent: float       # e.g. 20.0
    yield_loss_kg_per_acre: float
    total_yield_loss_kg: float
    revenue_loss_inr: float

    treatment_cost_inr: float       # total for the area
    yield_saved_inr: float          # revenue recovered by treating
    net_benefit_inr: float          # yield_saved - treatment_cost
    roi_ratio: float                # yield_saved / treatment_cost (>1 = profitable)

    roi_label: str                  # "₹3.20 saved per ₹1 spent"
    recommendation: str             # "TREAT" or "MONITOR"
    recommendation_detail: str
    notes: str


# ════════════════════════════════════════════════════════════════
# Estimator class
# ════════════════════════════════════════════════════════════════

class YieldEstimator:
    """Estimate yield loss and treatment ROI for a detected crop disease."""

    def estimate(
        self,
        disease: str,
        severity: Literal["mild", "moderate", "severe"],
        crop: str = "wheat",
        area_acres: float = 1.0,
        stage: str = "unknown",
    ) -> YieldEstimate:
        """Compute complete economic estimate.

        Args:
            disease:    Disease key from diseases.json (or an alias).
            severity:   One of "mild", "moderate", "severe".
            crop:       Crop type ("wheat", "rice", "maize").
            area_acres: Field area in acres (default 1).
            stage:      Growth stage key (default "unknown").

        Returns:
            YieldEstimate with full cost-benefit breakdown.
        """
        area_acres = max(0.01, float(area_acres))
        severity = severity.lower() if severity else "moderate"
        if severity not in ("mild", "moderate", "severe"):
            severity = "moderate"
        crop = crop.lower()
        stage = stage.lower() if stage else "unknown"

        # Resolve disease key
        disease_key = self._resolve_key(disease, crop)
        econ = _DISEASE_TABLE.get(disease_key)

        # Fallback if unknown disease
        if econ is None:
            econ = DiseaseEconomics(
                yield_loss_pct={"mild": 0.05, "moderate": 0.15, "severe": 0.30},
                fungicide_cost_inr=250.0, labor_cost_inr=150.0, spray_applications=1,
                treatment_efficacy=0.70, notes="Unknown disease — using conservative estimates",
            )

        # Commodity and yield data
        price_per_quintal = CROP_PRICE_INR_PER_QUINTAL.get(crop, 2300.0)
        base_yield_kg = CROP_YIELD_KG_PER_ACRE.get(crop, 1350.0)
        price_per_kg = price_per_quintal / 100.0

        # Stage adjustment
        stage_mult = STAGE_MULTIPLIERS.get(stage, 1.0)

        # Yield loss calculation
        raw_loss_frac = econ.yield_loss_pct.get(severity, econ.yield_loss_pct.get("moderate", 0.15))
        adj_loss_frac = min(raw_loss_frac * stage_mult, 0.99)
        yield_loss_pct = round(adj_loss_frac * 100, 2)
        yield_loss_kg_per_acre = round(base_yield_kg * adj_loss_frac, 1)
        total_yield_loss_kg = round(yield_loss_kg_per_acre * area_acres, 1)
        revenue_loss_inr = round(total_yield_loss_kg * price_per_kg, 0)

        # Treatment cost calculation
        material_cost = econ.fungicide_cost_inr * econ.spray_applications * area_acres
        labor_cost = econ.labor_cost_inr * econ.spray_applications * area_acres
        treatment_cost_inr = round(material_cost + labor_cost, 0)

        # ROI calculation
        yield_saved_inr = round(revenue_loss_inr * econ.treatment_efficacy, 0)
        net_benefit_inr = round(yield_saved_inr - treatment_cost_inr, 0)

        if treatment_cost_inr > 0:
            roi_ratio = round(yield_saved_inr / treatment_cost_inr, 2)
        else:
            roi_ratio = 0.0

        # Labels
        if treatment_cost_inr == 0:
            roi_label = "No treatment needed"
        elif roi_ratio >= 1.0:
            roi_label = f"₹{roi_ratio:.2f} saved per ₹1 spent"
        else:
            roi_label = f"₹{roi_ratio:.2f} recovered per ₹1 spent (marginal)"

        # Recommendation logic
        is_healthy = econ.fungicide_cost_inr == 0
        if is_healthy:
            recommendation = "HEALTHY"
            recommendation_detail = "No disease detected — continue regular monitoring"
        elif net_benefit_inr >= 500 and roi_ratio >= 1.5:
            recommendation = "TREAT"
            recommendation_detail = (
                f"Treatment saves ₹{int(net_benefit_inr):,} net over {area_acres:.1f} acre(s). "
                f"{econ.notes}"
            )
        elif net_benefit_inr > 0 and roi_ratio >= 1.0:
            recommendation = "TREAT"
            recommendation_detail = (
                f"Marginal benefit of ₹{int(net_benefit_inr):,}. "
                f"Treat if disease is spreading. {econ.notes}"
            )
        else:
            recommendation = "MONITOR"
            recommendation_detail = (
                f"Treatment cost (₹{int(treatment_cost_inr):,}) may exceed recoverable yield. "
                f"Scout field again in 3–5 days. {econ.notes}"
            )

        return YieldEstimate(
            disease_key=disease_key,
            severity=severity,
            crop=crop,
            area_acres=area_acres,
            stage=stage,
            yield_loss_percent=yield_loss_pct,
            yield_loss_kg_per_acre=yield_loss_kg_per_acre,
            total_yield_loss_kg=total_yield_loss_kg,
            revenue_loss_inr=revenue_loss_inr,
            treatment_cost_inr=treatment_cost_inr,
            yield_saved_inr=yield_saved_inr,
            net_benefit_inr=net_benefit_inr,
            roi_ratio=roi_ratio,
            roi_label=roi_label,
            recommendation=recommendation,
            recommendation_detail=recommendation_detail,
            notes=econ.notes,
        )

    def _resolve_key(self, disease: str, crop: str) -> str:
        """Map raw disease name/key to diseases.json key."""
        if not disease:
            return f"healthy_{crop}" if crop in ("wheat", "rice") else "healthy_wheat"

        # Normalize
        key = disease.lower().strip().replace(" ", "_").replace("-", "_")

        # Direct match
        if key in _DISEASE_TABLE:
            return key

        # Alias match
        if key in _ALIASES:
            return _ALIASES[key]

        # Prefix with crop if not already
        prefixed = f"{crop}_{key}"
        if prefixed in _DISEASE_TABLE:
            return prefixed

        # Fuzzy: find any key containing the disease name
        for dk in _DISEASE_TABLE:
            if key in dk:
                return dk

        return f"healthy_{crop}" if crop in ("wheat", "rice") else "healthy_wheat"


def estimate_to_dict(est: YieldEstimate) -> dict:
    """Serialize YieldEstimate for JSON API response."""
    return {
        "disease_key": est.disease_key,
        "severity": est.severity,
        "crop": est.crop,
        "area_acres": est.area_acres,
        "stage": est.stage,
        "yield_loss_percent": est.yield_loss_percent,
        "yield_loss_kg_per_acre": est.yield_loss_kg_per_acre,
        "total_yield_loss_kg": est.total_yield_loss_kg,
        "revenue_loss_inr": est.revenue_loss_inr,
        "treatment_cost_inr": est.treatment_cost_inr,
        "yield_saved_inr": est.yield_saved_inr,
        "net_benefit_inr": est.net_benefit_inr,
        "roi_ratio": est.roi_ratio,
        "roi_label": est.roi_label,
        "recommendation": est.recommendation,
        "recommendation_detail": est.recommendation_detail,
        "notes": est.notes,
    }


# Module-level singleton
_estimator: YieldEstimator | None = None


def get_estimator() -> YieldEstimator:
    global _estimator
    if _estimator is None:
        _estimator = YieldEstimator()
    return _estimator
