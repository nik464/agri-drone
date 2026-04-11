#!/usr/bin/env python3
"""
LLaVA Ensemble Evaluation — Config B vs Config C

Evaluates whether LLaVA (via Ollama) adds value as a third voter in the
AgriDrone ensemble pipeline.

Setup:
  - 200 stratified samples (equal per class, seed=42)
  - Config B: YOLO + Rules + Ensemble (2-model)
  - Config C: YOLO + Rules + LLaVA + Ensemble (3-model)

Outputs:
  results/llava_analysis.csv    — per-image B vs C classification
  results/hh_ratio.json         — Help/Harm ratio + FN reduction on critical diseases
  results/mcnemar.json          — McNemar's χ² test for statistical significance

Usage:
    python evaluate/llava_eval.py
    python evaluate/llava_eval.py --model-path models/india_agri_cls_21class_backup.pt
    python evaluate/llava_eval.py --subset-size 200 --seed 42
"""

import argparse
import base64
import csv
import json
import re
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Suppress verbose loguru output from agridrone modules
try:
    from loguru import logger
    logger.disable("agridrone")
except ImportError:
    pass

# ════════════════════════════════════════════════════════════════
# Constants
# ════════════════════════════════════════════════════════════════

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

OLLAMA_URL = "http://localhost:11434"
LLAVA_MODEL = "llava"
LLAVA_TIMEOUT = 300  # seconds per image (CPU-only needs more time)

SEVERITY_TIERS = {
    "wheat_fusarium_head_blight": ("critical", 10),
    "wheat_yellow_rust":         ("critical", 10),
    "wheat_black_rust":          ("critical", 10),
    "wheat_blast":               ("critical", 10),
    "rice_blast":                ("critical", 10),
    "rice_bacterial_blight":     ("critical", 10),
    "wheat_brown_rust":          ("high", 5),
    "wheat_septoria":            ("high", 5),
    "wheat_leaf_blight":         ("high", 5),
    "rice_sheath_blight":        ("high", 5),
    "wheat_root_rot":            ("high", 5),
    "rice_leaf_scald":           ("high", 5),
    "wheat_powdery_mildew":      ("moderate", 2),
    "wheat_tan_spot":            ("moderate", 2),
    "wheat_aphid":               ("moderate", 2),
    "wheat_mite":                ("moderate", 2),
    "wheat_smut":                ("moderate", 2),
    "wheat_stem_fly":            ("moderate", 2),
    "rice_brown_spot":           ("moderate", 2),
    "healthy_wheat":             ("healthy", 1),
    "healthy_rice":              ("healthy", 1),
}

CRITICAL_DISEASES = {k for k, (tier, _) in SEVERITY_TIERS.items() if tier == "critical"}


# ════════════════════════════════════════════════════════════════
# Data loading
# ════════════════════════════════════════════════════════════════

def load_test_set(test_dir: Path) -> list[dict]:
    """Load all test images with ground-truth labels."""
    images = []
    classes = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])
    for cls in classes:
        cls_dir = test_dir / cls
        for img_path in sorted(cls_dir.glob("*")):
            if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                images.append({
                    "path": str(img_path),
                    "ground_truth": cls,
                    "crop_type": "rice" if cls.startswith("rice") or cls == "healthy_rice" else "wheat",
                })
    return images


def stratified_sample(images: list[dict], n: int, seed: int) -> list[dict]:
    """Stratified sampling: equal images per class, up to n total."""
    rng = np.random.RandomState(seed)
    by_class = defaultdict(list)
    for img in images:
        by_class[img["ground_truth"]].append(img)

    n_classes = len(by_class)
    per_class = n // n_classes
    remainder = n % n_classes

    sampled = []
    for i, (cls, cls_images) in enumerate(sorted(by_class.items())):
        k = per_class + (1 if i < remainder else 0)
        k = min(k, len(cls_images))
        chosen = rng.choice(len(cls_images), size=k, replace=False)
        for idx in chosen:
            sampled.append(cls_images[idx])

    rng.shuffle(sampled)
    return sampled


# ════════════════════════════════════════════════════════════════
# Config B: YOLO + Rules (no LLaVA)
# ════════════════════════════════════════════════════════════════

def run_config_b(model, image_bgr: np.ndarray, crop_type: str) -> dict:
    """YOLO + Rule Engine + 2-model Ensemble (same as ablation Config B)."""
    from agridrone.vision.disease_reasoning import run_full_pipeline, diagnosis_to_dict
    from agridrone.vision.ensemble_voter import ensemble_vote

    results = model(image_bgr, verbose=False)
    if not results or results[0].probs is None:
        return {"predicted": "unknown", "confidence": 0.0}

    probs = results[0].probs
    names = model.names
    top5_indices = probs.top5
    top5_confs = probs.top5conf.tolist()
    top_key = names[probs.top1]
    top_conf = round(probs.top1conf.item(), 4)
    top_is_healthy = "healthy" in top_key.lower()
    severity = SEVERITY_TIERS.get(top_key, ("moderate", 2))
    health_score_yolo = 95 if top_is_healthy else max(5, round(100 - severity[1] * 10 * top_conf))

    classifier_result = {
        "top_prediction": top_key,
        "top_confidence": top_conf,
        "confidence": top_conf,
        "health_score": health_score_yolo,
        "is_healthy": top_is_healthy,
        "disease_probability": round(1 - top_conf if top_is_healthy else top_conf, 4),
        "top5": [
            {"index": idx, "class_key": names[idx],
             "class_name": names[idx].replace("_", " ").title(),
             "confidence": round(conf, 4)}
            for idx, conf in zip(top5_indices, top5_confs)
        ],
    }

    try:
        output = run_full_pipeline(image_bgr, classifier_result, crop_type)
        reasoning_result = diagnosis_to_dict(output.diagnosis)

        ensemble_result = ensemble_vote(
            classifier_result=classifier_result,
            reasoning_result=reasoning_result,
            llm_validation=None,
            crop_type=crop_type,
        )

        return {
            "predicted": ensemble_result.final_disease,
            "confidence": round(ensemble_result.final_confidence, 4),
            "agreement_level": ensemble_result.agreement_level,
            "classifier_result": classifier_result,
            "reasoning_result": reasoning_result,
            "rule_result": output.rule_result,
            "features": output.features,
            "pipeline_output": output,
        }
    except Exception as e:
        warnings.warn(f"Config B error: {e}")
        return {"predicted": top_key, "confidence": top_conf, "classifier_result": classifier_result}


# ════════════════════════════════════════════════════════════════
# LLaVA call
# ════════════════════════════════════════════════════════════════

def call_llava(image_bgr: np.ndarray, prompt: str) -> str | None:
    """Send image+prompt to Ollama LLaVA and return raw text response."""
    try:
        # Resize large images to 512x512 max for faster LLaVA processing
        # (LLaVA's CLIP encoder resizes to 336x336 internally anyway)
        h, w = image_bgr.shape[:2]
        if max(h, w) > 512:
            scale = 512 / max(h, w)
            image_bgr = cv2.resize(image_bgr, (int(w * scale), int(h * scale)),
                                   interpolation=cv2.INTER_AREA)
        success, encoded = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            return None
        image_b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")

        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": LLAVA_MODEL,
                "messages": [{
                    "role": "user",
                    "content": prompt,
                    "images": [image_b64],
                }],
                "stream": False,
            },
            timeout=LLAVA_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except Exception as e:
        warnings.warn(f"LLaVA call failed: {e}")
        return None


# ════════════════════════════════════════════════════════════════
# Config C: YOLO + Rules + LLaVA
# ════════════════════════════════════════════════════════════════

def run_config_c(model, image_bgr: np.ndarray, crop_type: str,
                 config_b_result: dict, cached_raw: str | None = None) -> dict:
    """Config B + LLaVA validation as third voter.

    Reuses Config B's pipeline output to avoid duplicate YOLO+rules inference.
    Only adds the LLaVA call and re-runs ensemble voting with 3 models.
    """
    from agridrone.vision.llm_validator import build_validation_prompt, parse_validation_response
    from agridrone.vision.ensemble_voter import ensemble_vote

    classifier_result = config_b_result.get("classifier_result")
    reasoning_result = config_b_result.get("reasoning_result")
    rule_result = config_b_result.get("rule_result")
    features = config_b_result.get("features")
    pipeline_output = config_b_result.get("pipeline_output")

    if not classifier_result or not rule_result or not features:
        return {
            "predicted": config_b_result.get("predicted", "unknown"),
            "confidence": config_b_result.get("confidence", 0.0),
            "llava_raw": None,
            "llava_validation": None,
            "llava_error": "missing_pipeline_data",
        }

    # Build scenario-appropriate prompt
    try:
        prompt, scenario = build_validation_prompt(
            rule_result=rule_result,
            features=features,
            classifier_result=classifier_result,
            crop_type=crop_type,
        )
    except Exception as e:
        warnings.warn(f"Prompt build error: {e}")
        prompt = f"Analyze this {crop_type} image for disease. Respond in JSON."
        scenario = "validate"

    our_diagnosis = config_b_result.get("predicted", "unknown")

    # Call LLaVA (or use cached response)
    if cached_raw is not None:
        raw_response = cached_raw
        llava_latency = 0.0
    else:
        t0 = time.perf_counter()
        raw_response = call_llava(image_bgr, prompt)
        llava_latency = (time.perf_counter() - t0) * 1000

    if raw_response is None:
        return {
            "predicted": config_b_result.get("predicted", "unknown"),
            "confidence": config_b_result.get("confidence", 0.0),
            "llava_raw": None,
            "llava_validation": None,
            "llava_error": "call_failed",
            "llava_latency_ms": round(llava_latency, 1),
        }

    # Parse LLaVA response → LLMValidation
    llm_validation = parse_validation_response(
        raw=raw_response,
        scenario=scenario,
        our_diagnosis=our_diagnosis,
    )

    # Coerce numeric fields that LLaVA may return as strings
    try:
        llm_validation.health_score = int(float(llm_validation.health_score))
    except (TypeError, ValueError):
        llm_validation.health_score = 50
    try:
        llm_validation.agreement_score = float(llm_validation.agreement_score)
    except (TypeError, ValueError):
        llm_validation.agreement_score = 0.5
    try:
        val = llm_validation.llm_confidence
        if isinstance(val, str):
            # Could be "high", "medium", "low" or a number string
            conf_map = {"high": 0.85, "medium": 0.6, "low": 0.3}
            llm_validation.llm_confidence = conf_map.get(val.lower(), float(val) if val.replace('.','',1).isdigit() else 0.5)
    except (TypeError, ValueError):
        llm_validation.llm_confidence = 0.5

    # Re-run ensemble with 3 voters
    try:
        ensemble_result = ensemble_vote(
            classifier_result=classifier_result,
            reasoning_result=reasoning_result,
            llm_validation=llm_validation,
            crop_type=crop_type,
        )

        return {
            "predicted": ensemble_result.final_disease,
            "confidence": round(ensemble_result.final_confidence, 4),
            "agreement_level": ensemble_result.agreement_level,
            "llava_raw": raw_response,
            "llava_validation": {
                "agrees": llm_validation.agrees,
                "agreement_score": llm_validation.agreement_score,
                "llm_diagnosis": llm_validation.llm_diagnosis,
                "llm_confidence": llm_validation.llm_confidence,
                "scenario": llm_validation.scenario,
                "health_score": llm_validation.health_score,
                "risk_level": llm_validation.risk_level,
                "visible_symptoms": llm_validation.visible_symptoms,
                "reasons": llm_validation.reasons,
            },
            "llava_latency_ms": round(llava_latency, 1),
        }
    except Exception as e:
        warnings.warn(f"Config C ensemble error: {e}")
        return {
            "predicted": config_b_result.get("predicted", "unknown"),
            "confidence": config_b_result.get("confidence", 0.0),
            "llava_raw": raw_response,
            "llava_validation": None,
            "llava_error": f"ensemble_error: {e}",
            "llava_latency_ms": round(llava_latency, 1),
        }


# ════════════════════════════════════════════════════════════════
# LLaVA cache
# ════════════════════════════════════════════════════════════════

def load_cache(cache_path: Path) -> dict:
    """Load cached LLaVA responses from JSON."""
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict, cache_path: Path):
    """Save LLaVA response cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2, default=str)


# ════════════════════════════════════════════════════════════════
# McNemar's test
# ════════════════════════════════════════════════════════════════

def mcnemar_test(b_correct: list[bool], c_correct: list[bool]) -> dict:
    """McNemar's test for paired nominal data.

    Compares two classifiers on the same samples.
    Uses exact binomial test if n_discordant < 25, else χ² approximation.
    """
    n = len(b_correct)
    # Count discordant pairs
    b_right_c_wrong = 0  # "broken" by C
    b_wrong_c_right = 0  # "corrected" by C
    both_right = 0
    both_wrong = 0

    for b, c in zip(b_correct, c_correct):
        if b and c:
            both_right += 1
        elif not b and not c:
            both_wrong += 1
        elif b and not c:
            b_right_c_wrong += 1
        else:
            b_wrong_c_right += 1

    n_discordant = b_right_c_wrong + b_wrong_c_right

    if n_discordant == 0:
        return {
            "b_right_c_wrong": b_right_c_wrong,
            "b_wrong_c_right": b_wrong_c_right,
            "both_right": both_right,
            "both_wrong": both_wrong,
            "n_discordant": 0,
            "chi2": 0.0,
            "p_value": 1.0,
            "significant_at_005": False,
            "method": "no_discordant_pairs",
        }

    # χ² approximation with continuity correction
    chi2 = (abs(b_right_c_wrong - b_wrong_c_right) - 1) ** 2 / n_discordant

    # p-value from chi-squared distribution with df=1
    try:
        from scipy.stats import chi2 as chi2_dist
        p_value = 1 - chi2_dist.cdf(chi2, df=1)
    except ImportError:
        # Manual chi-squared CDF approximation for df=1
        # Using complementary error function
        import math
        p_value = math.erfc(math.sqrt(chi2 / 2))

    return {
        "b_right_c_wrong": b_right_c_wrong,
        "b_wrong_c_right": b_wrong_c_right,
        "both_right": both_right,
        "both_wrong": both_wrong,
        "n_discordant": n_discordant,
        "chi2": float(round(chi2, 4)),
        "p_value": float(round(p_value, 6)),
        "significant_at_005": bool(p_value < 0.05),
        "method": "chi2_continuity_corrected",
    }


# ════════════════════════════════════════════════════════════════
# FN reduction on critical diseases
# ════════════════════════════════════════════════════════════════

def compute_fn_reduction(results: list[dict]) -> dict:
    """Compute false-negative reduction for critical diseases."""
    fn_b = defaultdict(int)
    fn_c = defaultdict(int)
    tp_b = defaultdict(int)
    tp_c = defaultdict(int)

    for r in results:
        gt = r["ground_truth"]
        if gt not in CRITICAL_DISEASES:
            continue
        pred_b = r["config_b_predicted"]
        pred_c = r["config_c_predicted"]

        if pred_b == gt:
            tp_b[gt] += 1
        else:
            fn_b[gt] += 1

        if pred_c == gt:
            tp_c[gt] += 1
        else:
            fn_c[gt] += 1

    per_disease = {}
    total_fn_b = 0
    total_fn_c = 0
    for disease in sorted(CRITICAL_DISEASES):
        fb = fn_b[disease]
        fc = fn_c[disease]
        total_fn_b += fb
        total_fn_c += fc
        support = fb + tp_b[disease]
        reduction = fb - fc
        per_disease[disease] = {
            "fn_config_b": fb,
            "fn_config_c": fc,
            "fn_reduction": reduction,
            "support": support,
            "recall_b": round(tp_b[disease] / support, 4) if support else 0,
            "recall_c": round(tp_c[disease] / support, 4) if support else 0,
        }

    return {
        "total_fn_config_b": total_fn_b,
        "total_fn_config_c": total_fn_c,
        "total_fn_reduction": total_fn_b - total_fn_c,
        "reduction_pct": round((total_fn_b - total_fn_c) / total_fn_b * 100, 2) if total_fn_b else 0,
        "per_disease": per_disease,
    }


# ════════════════════════════════════════════════════════════════
# Main evaluation
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="LLaVA ensemble evaluation: Config B vs C")
    parser.add_argument("--model-path", type=str,
                        default="models/india_agri_cls_21class_backup.pt",
                        help="Path to YOLOv8 classification model")
    parser.add_argument("--test-dir", type=str, default="data/training/test",
                        help="Test dataset directory")
    parser.add_argument("--output-dir", type=str, default="evaluate/results",
                        help="Output directory")
    parser.add_argument("--subset-size", type=int, default=200,
                        help="Number of stratified samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=str, default="evaluate/llava_cache",
                        help="Directory for LLaVA response cache")
    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    model_path = Path(args.model_path)

    # Validate
    if not test_dir.exists():
        print(f"ERROR: Test dir not found: {test_dir}")
        sys.exit(1)
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        sys.exit(1)

    # Check Ollama
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        if not any("llava" in m for m in models):
            print(f"ERROR: LLaVA not found in Ollama. Available: {models}")
            sys.exit(1)
        print(f"Ollama OK — LLaVA available")
    except Exception as e:
        print(f"ERROR: Cannot connect to Ollama at {OLLAMA_URL}: {e}")
        sys.exit(1)

    # Load model
    print(f"\nLoading model: {model_path}")
    from ultralytics import YOLO
    model = YOLO(str(model_path))
    n_classes = len(model.names)
    print(f"  Classes: {n_classes}")

    # Load and sample test set
    all_images = load_test_set(test_dir)
    subset = stratified_sample(all_images, args.subset_size, args.seed)
    print(f"\nStratified sample: {len(subset)} images from {len(all_images)} total")

    class_counts = defaultdict(int)
    for img in subset:
        class_counts[img["ground_truth"]] += 1
    print(f"  Classes: {len(class_counts)}, per-class: {sorted(class_counts.values())}")

    # Load cache
    cache_path = cache_dir / "llava_responses.json"
    cache = load_cache(cache_path)
    print(f"  Cache: {len(cache)} existing entries")

    # Progress log file for monitoring
    progress_log = output_dir / "llava_progress.log"
    def log_progress(msg):
        with open(progress_log, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        print(msg, flush=True)

    # ── Run evaluation ──
    log_progress(f"\n{'='*70}")
    log_progress(f"  EVALUATING: Config B vs Config C ({len(subset)} images)")
    log_progress(f"{'='*70}\n")

    results = []
    b_correct_list = []
    c_correct_list = []
    llava_latencies = []
    cache_hits = 0
    cache_misses = 0
    eval_start = time.perf_counter()

    for i, img_info in enumerate(subset):
        img_path = img_info["path"]
        gt = img_info["ground_truth"]
        crop_type = img_info["crop_type"]

        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            warnings.warn(f"Cannot read: {img_path}")
            continue

        # Resize large images to max 640px for faster pipeline processing
        # (YOLO resizes to 224×224 internally, features don't need 4K)
        h, w = image_bgr.shape[:2]
        if max(h, w) > 640:
            scale = 640 / max(h, w)
            image_bgr = cv2.resize(image_bgr, (int(w * scale), int(h * scale)),
                                   interpolation=cv2.INTER_AREA)

        # Config B
        log_progress(f"    [{i+1}] Config B starting...")
        t0 = time.perf_counter()
        b_result = run_config_b(model, image_bgr, crop_type)
        b_latency = (time.perf_counter() - t0) * 1000
        log_progress(f"    [{i+1}] Config B done ({b_latency:.0f}ms)")

        # Config C (reuses B's pipeline output, adds LLaVA)
        cache_key = Path(img_path).name
        cached_raw = cache.get(cache_key, {}).get("raw") if cache_key in cache else None
        log_progress(f"    [{i+1}] Config C starting... (cached={cached_raw is not None})")
        t0 = time.perf_counter()
        c_result = run_config_c(model, image_bgr, crop_type, b_result, cached_raw=cached_raw)
        c_latency = (time.perf_counter() - t0) * 1000
        log_progress(f"    [{i+1}] Config C done ({c_latency:.0f}ms)")

        if c_result.get("llava_latency_ms"):
            llava_latencies.append(c_result["llava_latency_ms"])

        # Cache LLaVA response
        if c_result.get("llava_raw"):
            cache[cache_key] = {
                "raw": c_result["llava_raw"],
                "validation": c_result.get("llava_validation"),
                "ground_truth": gt,
                "config_b_predicted": b_result.get("predicted"),
                "config_c_predicted": c_result.get("predicted"),
            }
            cache_misses += 1
        else:
            cache_hits += 1 if cache_key in cache else 0

        # Classify outcome
        b_pred = b_result.get("predicted", "unknown")
        c_pred = c_result.get("predicted", "unknown")
        b_ok = b_pred == gt
        c_ok = c_pred == gt
        b_correct_list.append(b_ok)
        c_correct_list.append(c_ok)

        if not b_ok and c_ok:
            category = "corrected"
        elif b_ok and not c_ok:
            category = "broken"
        elif b_ok and c_ok:
            category = "both_correct"
        else:
            category = "both_wrong"

        llava_val = c_result.get("llava_validation", {}) or {}

        results.append({
            "image": Path(img_path).name,
            "ground_truth": gt,
            "crop_type": crop_type,
            "config_b_predicted": b_pred,
            "config_b_confidence": b_result.get("confidence", 0),
            "config_b_correct": b_ok,
            "config_c_predicted": c_pred,
            "config_c_confidence": c_result.get("confidence", 0),
            "config_c_correct": c_ok,
            "category": category,
            "llava_diagnosis": llava_val.get("llm_diagnosis", ""),
            "llava_agrees": llava_val.get("agrees", ""),
            "llava_agreement_score": llava_val.get("agreement_score", ""),
            "llava_scenario": llava_val.get("scenario", ""),
            "llava_latency_ms": c_result.get("llava_latency_ms", 0),
            "is_critical": gt in CRITICAL_DISEASES,
        })

        # Progress with ETA
        status = f"{'✓' if c_ok else '✗'}"
        elapsed = time.perf_counter() - eval_start
        avg_per_img = elapsed / (i + 1)
        eta_s = avg_per_img * (len(subset) - i - 1)
        eta_min = eta_s / 60
        src_tag = "cache" if cached_raw else "live"
        log_progress(f"  [{i+1:3d}/{len(subset)}] {gt:<35} B={b_pred:<35} C={c_pred:<35} [{category}] {status} ({src_tag}, ETA {eta_min:.0f}m)")

        # Save cache periodically
        if (i + 1) % 5 == 0:
            save_cache(cache, cache_path)
            print(f"         ... cache saved ({len(cache)} entries)")

    # Final cache save
    save_cache(cache, cache_path)

    # ════════════════════════════════════════════════════════════
    # Compute metrics
    # ════════════════════════════════════════════════════════════

    n = len(results)
    corrected = sum(1 for r in results if r["category"] == "corrected")
    broken = sum(1 for r in results if r["category"] == "broken")
    both_correct = sum(1 for r in results if r["category"] == "both_correct")
    both_wrong = sum(1 for r in results if r["category"] == "both_wrong")

    b_accuracy = sum(1 for r in results if r["config_b_correct"]) / n
    c_accuracy = sum(1 for r in results if r["config_c_correct"]) / n

    # Help/Harm ratio
    hh_ratio = corrected / broken if broken > 0 else float("inf")

    # Macro-F1 for B and C
    classes = sorted(set(r["ground_truth"] for r in results))
    b_f1 = _macro_f1(results, classes, "config_b_predicted")
    c_f1 = _macro_f1(results, classes, "config_c_predicted")

    # McNemar's test
    mcnemar = mcnemar_test(b_correct_list, c_correct_list)

    # FN reduction on critical diseases
    fn_reduction = compute_fn_reduction(results)

    # ════════════════════════════════════════════════════════════
    # Output
    # ════════════════════════════════════════════════════════════

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. llava_analysis.csv
    csv_path = output_dir / "llava_analysis.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image", "ground_truth", "crop_type",
            "config_b_predicted", "config_b_confidence", "config_b_correct",
            "config_c_predicted", "config_c_confidence", "config_c_correct",
            "category", "llava_diagnosis", "llava_agrees",
            "llava_agreement_score", "llava_scenario", "llava_latency_ms",
            "is_critical",
        ])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  CSV saved: {csv_path}")

    # 2. hh_ratio.json
    hh_data = {
        "n_samples": n,
        "config_b_accuracy": round(b_accuracy, 4),
        "config_c_accuracy": round(c_accuracy, 4),
        "accuracy_delta": round(c_accuracy - b_accuracy, 4),
        "config_b_macro_f1": round(b_f1, 4),
        "config_c_macro_f1": round(c_f1, 4),
        "macro_f1_delta": round(c_f1 - b_f1, 4),
        "corrected": corrected,
        "broken": broken,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "help_to_harm_ratio": round(hh_ratio, 4) if hh_ratio != float("inf") else "inf",
        "net_improvement": corrected - broken,
        "fn_reduction_critical": fn_reduction,
        "mean_llava_latency_ms": float(round(np.mean(llava_latencies), 1)) if llava_latencies else 0,
        "conclusion": (
            "LLaVA adds value (H:H > 1, net positive)"
            if corrected > broken
            else "LLaVA does NOT add value (H:H ≤ 1, net negative or zero)"
        ),
    }
    hh_path = output_dir / "hh_ratio.json"
    with open(hh_path, "w") as f:
        json.dump(hh_data, f, indent=2)
    print(f"  JSON saved: {hh_path}")

    # 3. mcnemar.json
    mcnemar_path = output_dir / "mcnemar.json"
    with open(mcnemar_path, "w") as f:
        json.dump(mcnemar, f, indent=2)
    print(f"  JSON saved: {mcnemar_path}")

    # ── Summary report ──
    print(f"\n{'='*70}")
    print(f"  LLAVA ENSEMBLE EVALUATION REPORT")
    print(f"{'='*70}")
    print(f"\n  Samples: {n} (stratified, {len(classes)} classes, seed={args.seed})")
    print(f"\n  AGGREGATE METRICS")
    print(f"  {'Metric':<25} {'Config B':>12} {'Config C':>12} {'Delta':>12}")
    print(f"  {'-'*60}")
    print(f"  {'Accuracy':<25} {b_accuracy*100:>11.2f}% {c_accuracy*100:>11.2f}% {(c_accuracy-b_accuracy)*100:>+11.2f}pp")
    print(f"  {'Macro-F1':<25} {b_f1:>12.4f} {c_f1:>12.4f} {c_f1-b_f1:>+12.4f}")

    print(f"\n  HELP / HARM ANALYSIS")
    print(f"  {'-'*45}")
    print(f"  Corrected (B wrong → C right):  {corrected:>4}")
    print(f"  Broken    (B right → C wrong):  {broken:>4}")
    print(f"  Both correct:                   {both_correct:>4}")
    print(f"  Both wrong:                     {both_wrong:>4}")
    print(f"  Help:Harm ratio:                {hh_ratio:>7.2f}" if hh_ratio != float("inf") else f"  Help:Harm ratio:                     inf")
    print(f"  Net improvement:                {corrected - broken:>+4}")

    print(f"\n  MCNEMAR'S TEST")
    print(f"  {'-'*45}")
    print(f"  χ²:                             {mcnemar['chi2']:>7.4f}")
    print(f"  p-value:                        {mcnemar['p_value']:>7.6f}")
    print(f"  Significant (α=0.05):           {'YES' if mcnemar['significant_at_005'] else 'NO'}")

    print(f"\n  CRITICAL DISEASE FN REDUCTION")
    print(f"  {'-'*60}")
    print(f"  {'Disease':<35} {'FN(B)':>6} {'FN(C)':>6} {'ΔFN':>6}")
    for disease, data in sorted(fn_reduction["per_disease"].items()):
        if data["support"] > 0:
            print(f"  {disease:<35} {data['fn_config_b']:>6} {data['fn_config_c']:>6} {data['fn_reduction']:>+6}")
    print(f"  {'TOTAL':<35} {fn_reduction['total_fn_config_b']:>6} {fn_reduction['total_fn_config_c']:>6} {fn_reduction['total_fn_reduction']:>+6}")
    if fn_reduction["total_fn_config_b"] > 0:
        print(f"  FN reduction: {fn_reduction['reduction_pct']:+.1f}%")

    print(f"\n  LATENCY")
    print(f"  Mean LLaVA latency: {np.mean(llava_latencies):.0f} ms/image" if llava_latencies else "  No LLaVA latency data")

    print(f"\n  CONCLUSION: {hh_data['conclusion']}")
    print(f"\n{'='*70}")


def _macro_f1(results: list[dict], classes: list[str], pred_key: str) -> float:
    """Compute macro-F1 for a given prediction column."""
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    for r in results:
        gt = r["ground_truth"]
        pr = r[pred_key]
        if gt == pr:
            tp[gt] += 1
        else:
            fn[gt] += 1
            fp[pr] += 1

    f1_sum = 0.0
    for cls in classes:
        p = tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) > 0 else 0
        r = tp[cls] / (tp[cls] + fn[cls]) if (tp[cls] + fn[cls]) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        f1_sum += f1
    return f1_sum / len(classes) if classes else 0


if __name__ == "__main__":
    main()
