"""
feature_extractor.py — Extract visual features from crop disease images.

Pulls raw image features into a structured dataclass that can be consumed by
the rule engine, reasoning engine, or any downstream analysis.

Features extracted:
  - Color ratios per HSV range (per-disease color signatures)
  - Texture metrics (bleaching, spots, edge density)
  - Spatial patterns (stripe vs spot, linearity, circularity)
  - Saturation analysis (vivid vs dull)
  - Global image statistics (brightness, greenness)
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from loguru import logger


# ════════════════════════════════════════════════════════════════
# Feature dataclass — the "observation" that downstream modules consume
# ════════════════════════════════════════════════════════════════

@dataclass
class ImageFeatures:
    """Structured visual features extracted from a crop disease image.

    This is the single source of truth for what the camera "sees."
    The rule engine and reasoning engine consume this — they never touch raw pixels.
    """

    # ── Color features ──
    # disease_key:sig_name → ratio of image pixels matching that HSV range (0.0–1.0)
    color_ratios: dict[str, float] = field(default_factory=dict)
    # disease_key:sig_name → scaled confidence (ratio * 20, capped at 1.0)
    color_confidences: dict[str, float] = field(default_factory=dict)

    # ── Texture features ──
    bleaching_ratio: float = 0.0           # Fraction of bleached (high V, low S) pixels
    spot_count: int = 0                     # Number of blob keypoints detected
    edge_density: float = 0.0               # Fraction of Canny edge pixels

    # ── Spatial pattern features (stripe vs spot) ──
    yellow_orange_ratio: float = 0.0        # Fraction of saturated yellow-orange pixels
    linear_pixels: int = 0                  # Pixels retained by horizontal/vertical morpho open
    circular_pixels: int = 0                # Pixels retained by elliptical morpho open
    hough_line_count: int = 0               # Number of HoughLinesP detections
    stripe_confidence: float = 0.0          # Composite stripe pattern score (0.0–1.0)
    spot_confidence: float = 0.0            # Composite discrete-spot pattern score (0.0–1.0)

    # ── Directional energy (Sobel) ──
    h_energy: float = 0.0                   # Mean |Sobel_x| (horizontal edges)
    v_energy: float = 0.0                   # Mean |Sobel_y| (vertical edges)
    directionality: float = 1.0             # max/min energy ratio

    # ── Saturation analysis ──
    vivid_yellow_orange_ratio: float = 0.0  # High-saturation yellow-orange fraction
    mean_saturation: float = 0.0            # Global mean saturation of non-black pixels
    mean_brightness: float = 0.0            # Global mean V-channel

    # ── Greenness (healthy tissue indicator) ──
    green_ratio: float = 0.0                # Fraction of green pixels (H 35–85, S>50, V>50)

    # ── Convenience derived booleans ──
    has_stripe_pattern: bool = False
    has_spot_pattern: bool = False
    has_vivid_yellow: bool = False
    has_bleaching: bool = False
    has_significant_spots: bool = False

    # ── Image metadata ──
    width: int = 0
    height: int = 0
    total_pixels: int = 0


# ════════════════════════════════════════════════════════════════
# Extraction functions
# ════════════════════════════════════════════════════════════════

def _extract_color_features(
    hsv: np.ndarray,
    total_pixels: int,
    profiles: dict,
) -> tuple[dict[str, float], dict[str, float]]:
    """Extract per-disease color signature ratios.

    Returns (color_ratios, color_confidences).
    """
    ratios: dict[str, float] = {}
    confidences: dict[str, float] = {}

    for disease_key, profile in profiles.items():
        for sig in profile.color_signatures:
            lower = np.array([sig["h_range"][0], sig["s_range"][0], sig["v_range"][0]])
            upper = np.array([sig["h_range"][1], sig["s_range"][1], sig["v_range"][1]])
            mask = cv2.inRange(hsv, lower, upper)
            count = cv2.countNonZero(mask)
            ratio = count / total_pixels

            # For "healthy_green" signatures, require >70% coverage to avoid
            # firing on background grass/stems behind diseased crop tissue.
            if sig["name"] == "healthy_green":
                min_ratio = 0.70
            else:
                min_ratio = 0.005  # >0.5% of image for disease signatures

            if ratio > min_ratio:
                key = f"{disease_key}:{sig['name']}"
                ratios[key] = ratio
                confidences[key] = min(1.0, ratio * 20)

    return ratios, confidences


def _extract_texture_features(
    gray: np.ndarray,
    hsv: np.ndarray,
    total_pixels: int,
) -> dict:
    """Extract texture metrics: bleaching, spots, edge density.

    Returns dict of raw texture values.
    """
    result = {}

    # Bleaching detection
    bleach_mask = cv2.inRange(hsv, np.array([10, 10, 170]), np.array([40, 80, 255]))
    result["bleaching_ratio"] = cv2.countNonZero(bleach_mask) / total_pixels

    # Spot/pustule blob detection
    detector = cv2.SimpleBlobDetector_create(cv2.SimpleBlobDetector_Params())
    keypoints = detector.detect(gray)
    result["spot_count"] = len(keypoints)

    # Edge density
    edges = cv2.Canny(gray, 50, 150)
    result["edge_density"] = cv2.countNonZero(edges) / total_pixels

    return result


def _extract_spatial_pattern(
    hsv: np.ndarray,
    total_pixels: int,
) -> dict:
    """Analyze stripe-vs-spot spatial patterns in yellow-orange regions.

    Key differentiator for Yellow/Stripe Rust vs Tan Spot.
    """
    result = {
        "yellow_orange_ratio": 0.0,
        "linear_pixels": 0,
        "circular_pixels": 0,
        "hough_line_count": 0,
        "stripe_confidence": 0.0,
        "spot_confidence": 0.0,
    }

    # Saturated yellow-orange mask
    yo_mask = cv2.inRange(hsv, np.array([10, 100, 120]), np.array([40, 255, 255]))
    yo_count = cv2.countNonZero(yo_mask)
    yo_ratio = yo_count / total_pixels
    result["yellow_orange_ratio"] = yo_ratio

    if yo_ratio < 0.02:
        return result

    # Morphological linearity analysis
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    kernel_circ = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    closed = cv2.morphologyEx(
        yo_mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )

    linear_h = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_h)
    linear_v = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_v)
    circular = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_circ)

    linear_px = cv2.countNonZero(linear_h) + cv2.countNonZero(linear_v)
    circular_px = cv2.countNonZero(circular)
    result["linear_pixels"] = linear_px
    result["circular_pixels"] = circular_px

    # Hough line detection
    line_count = 0
    try:
        lines = cv2.HoughLinesP(closed, 1, np.pi / 180, 30, minLineLength=30, maxLineGap=10)
        if lines is not None:
            line_count = len(lines)
    except Exception:
        pass
    result["hough_line_count"] = line_count

    # Stripe confidence
    if linear_px > 0 and (linear_px > circular_px * 0.3 or line_count > 5):
        stripe_conf = min(
            1.0,
            (linear_px / max(1, yo_count)) * 2.0 + line_count * 0.03,
        )
        result["stripe_confidence"] = stripe_conf
        logger.debug(
            f"Stripe pattern: linear_px={linear_px}, circular_px={circular_px}, "
            f"lines={line_count}, conf={stripe_conf:.2f}"
        )

    # Spot confidence
    if circular_px > linear_px * 2 and line_count < 3:
        result["spot_confidence"] = min(1.0, circular_px / max(1, yo_count))

    return result


def _extract_directional_energy(gray: np.ndarray) -> dict:
    """Sobel-based directional energy analysis."""
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    h_energy = float(np.mean(np.abs(sobelx)))
    v_energy = float(np.mean(np.abs(sobely)))
    directionality = max(h_energy, v_energy) / max(min(h_energy, v_energy), 1.0)
    return {
        "h_energy": h_energy,
        "v_energy": v_energy,
        "directionality": directionality,
    }


def _extract_saturation_features(hsv: np.ndarray, total_pixels: int) -> dict:
    """Analyze saturation: vivid yellow-orange ratio, mean saturation, mean brightness."""
    # Vivid yellow-orange (high saturation)
    vivid_mask = cv2.inRange(hsv, np.array([10, 150, 140]), np.array([40, 255, 255]))
    vivid_ratio = cv2.countNonZero(vivid_mask) / total_pixels

    # Global mean saturation and brightness (excluding near-black pixels)
    v_channel = hsv[:, :, 2]
    non_black = v_channel > 20
    mean_sat = float(np.mean(hsv[:, :, 1][non_black])) if np.any(non_black) else 0.0
    mean_bright = float(np.mean(v_channel[non_black])) if np.any(non_black) else 0.0

    return {
        "vivid_yellow_orange_ratio": vivid_ratio,
        "mean_saturation": mean_sat,
        "mean_brightness": mean_bright,
    }


def _extract_greenness(hsv: np.ndarray, total_pixels: int) -> float:
    """Fraction of green (healthy tissue) pixels."""
    green_mask = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
    return cv2.countNonZero(green_mask) / total_pixels


# ════════════════════════════════════════════════════════════════
# Main entry point
# ════════════════════════════════════════════════════════════════

def extract_features(image_bgr: np.ndarray, profiles: dict) -> ImageFeatures:
    """Extract all visual features from an image.

    Args:
        image_bgr: BGR image (OpenCV format).
        profiles: Disease profile dict from kb_loader.get_all_profiles().

    Returns:
        Populated ImageFeatures dataclass.
    """
    h, w = image_bgr.shape[:2]
    total_pixels = h * w

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Extract all feature groups
    color_ratios, color_confidences = _extract_color_features(hsv, total_pixels, profiles)
    tex = _extract_texture_features(gray, hsv, total_pixels)
    spatial = _extract_spatial_pattern(hsv, total_pixels)
    direction = _extract_directional_energy(gray)
    sat = _extract_saturation_features(hsv, total_pixels)
    green_ratio = _extract_greenness(hsv, total_pixels)

    # Combine directional stripe signal
    has_stripe = (
        spatial["stripe_confidence"] > 0.1
        or (direction["directionality"] > 1.5)
    )
    has_spot = spatial["spot_confidence"] > 0.1
    has_vivid = sat["vivid_yellow_orange_ratio"] > 0.03

    features = ImageFeatures(
        # Color
        color_ratios=color_ratios,
        color_confidences=color_confidences,
        # Texture
        bleaching_ratio=tex["bleaching_ratio"],
        spot_count=tex["spot_count"],
        edge_density=tex["edge_density"],
        # Spatial
        yellow_orange_ratio=spatial["yellow_orange_ratio"],
        linear_pixels=spatial["linear_pixels"],
        circular_pixels=spatial["circular_pixels"],
        hough_line_count=spatial["hough_line_count"],
        stripe_confidence=spatial["stripe_confidence"],
        spot_confidence=spatial["spot_confidence"],
        # Directional
        h_energy=direction["h_energy"],
        v_energy=direction["v_energy"],
        directionality=direction["directionality"],
        # Saturation
        vivid_yellow_orange_ratio=sat["vivid_yellow_orange_ratio"],
        mean_saturation=sat["mean_saturation"],
        mean_brightness=sat["mean_brightness"],
        # Greenness
        green_ratio=green_ratio,
        # Derived booleans
        has_stripe_pattern=has_stripe,
        has_spot_pattern=has_spot,
        has_vivid_yellow=has_vivid,
        has_bleaching=tex["bleaching_ratio"] > 0.03,
        has_significant_spots=tex["spot_count"] > 10,
        # Metadata
        width=w,
        height=h,
        total_pixels=total_pixels,
    )

    logger.debug(
        f"Features extracted: {len(color_confidences)} color sigs, "
        f"stripe={features.has_stripe_pattern}, spots={features.has_spot_pattern}, "
        f"vivid={features.has_vivid_yellow}, green={green_ratio:.2%}"
    )

    return features
