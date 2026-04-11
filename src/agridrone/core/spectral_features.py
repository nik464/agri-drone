"""
spectral_features.py — Pseudo-hyperspectral vegetation indices from RGB imagery.

Extracts vegetation indices typically computed from multispectral sensors,
approximated here from standard RGB channels. These indices serve as early
stress indicators — they can flag chlorosis and nutrient deficiency before
visible symptoms appear to the human eye or the classifier.

Indices computed:
  - VARI   (Visible Atmospherically Resistant Index) – overall plant health
  - GLI    (Green Leaf Index) – greenness / chlorophyll proxy
  - NGRDI  (Normalized Green-Red Difference Index) – canopy greenness
  - ExG    (Excess Green Index) – vegetation segmentation
  - RGRI   (Red-Green Ratio Index) – senescence / necrosis
  - TGI    (Triangular Greenness Index) – chlorophyll concentration proxy
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class SpectralIndex:
    """Statistics for a single vegetation index computed over the image."""
    name: str
    mean: float
    std: float
    p10: float      # 10th percentile
    p90: float      # 90th percentile
    coverage: float  # fraction of valid (non-masked) pixels


@dataclass
class SpectralResult:
    """Complete spectral analysis of one image."""
    indices: dict[str, SpectralIndex] = field(default_factory=dict)
    stress_detected: bool = False
    stress_level: str = "none"          # none | mild | moderate | severe
    stress_type: str = ""               # chlorosis | necrosis | nutrient_deficiency | ""
    stress_signals: list[str] = field(default_factory=list)


def _safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Element-wise division with zero-denominator protection."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(den != 0, num / den, 0.0)
    return result


def _index_stats(arr: np.ndarray, name: str, mask: np.ndarray | None = None) -> SpectralIndex:
    """Compute summary stats for a vegetation index array."""
    if mask is not None:
        vals = arr[mask]
    else:
        vals = arr.ravel()
    if vals.size == 0:
        return SpectralIndex(name=name, mean=0.0, std=0.0, p10=0.0, p90=0.0, coverage=0.0)
    total = arr.size if mask is None else mask.size
    return SpectralIndex(
        name=name,
        mean=float(np.nanmean(vals)),
        std=float(np.nanstd(vals)),
        p10=float(np.nanpercentile(vals, 10)),
        p90=float(np.nanpercentile(vals, 90)),
        coverage=float(vals.size / total) if total > 0 else 0.0,
    )


def extract_spectral_indices(image_bgr: np.ndarray) -> SpectralResult:
    """Compute pseudo-hyperspectral vegetation indices from a BGR image.

    Args:
        image_bgr: OpenCV BGR uint8 image (H×W×3).

    Returns:
        SpectralResult with per-index statistics and stress assessment.
    """
    if image_bgr is None or image_bgr.size == 0:
        return SpectralResult()

    # Convert to float [0, 1] to avoid overflow
    img = image_bgr.astype(np.float32) / 255.0
    B, G, R = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    # Mask out near-black pixels (background / shadow)
    brightness = (R + G + B) / 3.0
    valid = brightness > 0.05

    indices: dict[str, SpectralIndex] = {}

    # ── VARI: Visible Atmospherically Resistant Index ──
    # (G - R) / (G + R - B);  healthy vegetation → positive, stress → near zero/negative
    vari_den = G + R - B
    vari = _safe_divide(G - R, vari_den)
    vari = np.clip(vari, -1.0, 1.0)
    indices["VARI"] = _index_stats(vari, "VARI", valid)

    # ── GLI: Green Leaf Index ──
    # (2*G - R - B) / (2*G + R + B);  range [-1, 1], higher = more green
    gli_num = 2.0 * G - R - B
    gli_den = 2.0 * G + R + B
    gli = _safe_divide(gli_num, gli_den)
    gli = np.clip(gli, -1.0, 1.0)
    indices["GLI"] = _index_stats(gli, "GLI", valid)

    # ── NGRDI: Normalized Green-Red Difference Index ──
    # (G - R) / (G + R);  > 0 = green canopy, < 0 = senescent/stressed
    ngrdi = _safe_divide(G - R, G + R)
    ngrdi = np.clip(ngrdi, -1.0, 1.0)
    indices["NGRDI"] = _index_stats(ngrdi, "NGRDI", valid)

    # ── ExG: Excess Green Index ──
    # 2*G - R - B;  positive = vegetation, negative = non-veg
    exg = 2.0 * G - R - B
    exg = np.clip(exg, -2.0, 2.0)
    indices["ExG"] = _index_stats(exg, "ExG", valid)

    # ── RGRI: Red-Green Ratio Index ──
    # R / G;  > 1.0 = reddish (necrosis / senescence), < 1.0 = green
    rgri = _safe_divide(R, G)
    rgri = np.clip(rgri, 0.0, 5.0)
    indices["RGRI"] = _index_stats(rgri, "RGRI", valid)

    # ── TGI: Triangular Greenness Index (simplified from RGB) ──
    # -0.5 * [(R-G)*670 + (R-B)*(-480)] approximated for RGB-only sensors
    # Simplified: -0.5 * (190*(R-G) - 120*(R-B))
    tgi = -0.5 * (190.0 * (R - G) - 120.0 * (R - B))
    indices["TGI"] = _index_stats(tgi, "TGI", valid)

    # ── Stress assessment logic ──
    result = SpectralResult(indices=indices)
    _assess_stress(result)

    logger.debug(
        f"Spectral indices: VARI={indices['VARI'].mean:.3f}, "
        f"RGRI={indices['RGRI'].mean:.3f}, "
        f"GLI={indices['GLI'].mean:.3f}, "
        f"stress={result.stress_level}"
    )

    return result


def _assess_stress(result: SpectralResult) -> None:
    """Evaluate stress thresholds and populate stress fields."""
    vari = result.indices.get("VARI")
    gli = result.indices.get("GLI")
    ngrdi = result.indices.get("NGRDI")
    rgri = result.indices.get("RGRI")
    exg = result.indices.get("ExG")

    signals: list[str] = []
    severity = 0  # accumulator: 0 = none, 1 = mild, 2+ = moderate, 4+ = severe

    # Low VARI → reduced photosynthetic capacity
    if vari and vari.mean < 0.05:
        signals.append(f"Low VARI ({vari.mean:.3f}) — reduced photosynthetic capacity")
        severity += 2 if vari.mean < 0.0 else 1

    # Low GLI → chlorophyll loss
    if gli and gli.mean < 0.0:
        signals.append(f"Negative GLI ({gli.mean:.3f}) — possible chlorosis")
        severity += 2 if gli.mean < -0.1 else 1

    # Negative NGRDI → red dominance (stress / senescence)
    if ngrdi and ngrdi.mean < 0.0:
        signals.append(f"Negative NGRDI ({ngrdi.mean:.3f}) — red-dominant canopy")
        severity += 1

    # High RGRI → necrosis / blast / blight indicators
    if rgri and rgri.mean > 1.3:
        signals.append(f"Elevated RGRI ({rgri.mean:.3f}) — possible necrosis or blight")
        severity += 2 if rgri.mean > 1.8 else 1

    # Low ExG → poor green vegetation signal
    if exg and exg.mean < 0.0:
        signals.append(f"Negative ExG ({exg.mean:.3f}) — weak vegetation signal")
        severity += 1

    result.stress_signals = signals

    if severity == 0:
        result.stress_detected = False
        result.stress_level = "none"
        result.stress_type = ""
    elif severity <= 2:
        result.stress_detected = True
        result.stress_level = "mild"
    elif severity <= 4:
        result.stress_detected = True
        result.stress_level = "moderate"
    else:
        result.stress_detected = True
        result.stress_level = "severe"

    # Classify stress type
    if result.stress_detected:
        if rgri and rgri.mean > 1.5:
            result.stress_type = "necrosis"
        elif vari and vari.mean < 0.05 and gli and gli.mean < 0.0:
            result.stress_type = "chlorosis"
        elif exg and exg.mean < -0.1:
            result.stress_type = "nutrient_deficiency"
        else:
            result.stress_type = "general_stress"


def spectral_to_dict(result: SpectralResult) -> dict:
    """Serialize SpectralResult for JSON API response."""
    return {
        "indices": {
            name: {
                "mean": round(idx.mean, 4),
                "std": round(idx.std, 4),
                "p10": round(idx.p10, 4),
                "p90": round(idx.p90, 4),
                "coverage": round(idx.coverage, 4),
            }
            for name, idx in result.indices.items()
        },
        "stress_detected": result.stress_detected,
        "stress_level": result.stress_level,
        "stress_type": result.stress_type,
        "stress_signals": result.stress_signals,
    }
