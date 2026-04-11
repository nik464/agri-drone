"""
gradcam.py — Grad-CAM heatmap generation for YOLOv8 classifier (F1).

Shows WHERE on the image the model is focusing when making its disease prediction.
Uses the last convolutional layer of the classifier backbone to produce a class-
discriminative heatmap, then overlays it on the original image.

Works with YOLO classify models (YOLOv8n-cls architecture):
  model.model.model[-2]  → last Conv layer before classification head
"""

from __future__ import annotations

import base64
import io
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger


# ── Hook storage ──
_activations: list[torch.Tensor] = []


def _forward_hook_retain(module, input, output):
    """Forward hook that captures activations WITH gradient tracking."""
    _activations.clear()
    _activations.append(output)  # Keep grad_fn for autograd.grad


def _find_last_conv(model) -> Optional[torch.nn.Module]:
    """Walk the YOLO model to find the last Conv2d layer before the classify head."""
    try:
        sequential = model.model.model
        for i in range(len(sequential) - 1, -1, -1):
            layer = sequential[i]
            if hasattr(layer, 'conv') and isinstance(layer.conv, torch.nn.Conv2d):
                return layer.conv
            if isinstance(layer, torch.nn.Conv2d):
                return layer
            for child in reversed(list(layer.modules())):
                if isinstance(child, torch.nn.Conv2d):
                    return child
    except Exception:
        pass

    last_conv = None
    for module in model.model.modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    return last_conv


def generate_gradcam(
    model,
    image_bgr: np.ndarray,
    target_class_idx: int | None = None,
    img_size: int = 224,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Generate a Grad-CAM heatmap for the given image and class.

    Uses torch.autograd.grad on captured activations (avoids backward hook
    issues with YOLOv8's inplace view operations).
    """
    _activations.clear()

    target_layer = _find_last_conv(model)
    if target_layer is None:
        raise RuntimeError("Could not find a Conv2d layer in the model")

    # Only a forward hook — capture activations WITH grad tracking
    fwd_handle = target_layer.register_forward_hook(_forward_hook_retain)

    torch_model = model.model
    torch_model.eval()

    # Save and enable gradients
    orig_requires_grad = {}
    for name, p in torch_model.named_parameters():
        orig_requires_grad[name] = p.requires_grad
        p.requires_grad_(True)

    try:
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (img_size, img_size))
        tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0

        with torch.enable_grad():
            output = torch_model(tensor)

            if isinstance(output, (list, tuple)):
                logits = output[0]
            else:
                logits = output
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)

            probs = F.softmax(logits, dim=1)
            if target_class_idx is None:
                target_class_idx = probs.argmax(dim=1).item()
            target_confidence = probs[0, target_class_idx].item()

            if not _activations:
                raise RuntimeError("Forward hook did not capture activations")

            activations = _activations[0]  # (1, C, H, W) — has grad_fn

            # Compute gradients via autograd.grad (avoids backward hook issues)
            score = logits[0, target_class_idx]
            grads = torch.autograd.grad(
                score, activations, retain_graph=False, create_graph=False
            )[0]  # (1, C, H, W)

        # Compute Grad-CAM
        weights = grads.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
        cam = (weights * activations.detach()).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        h, w = image_bgr.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        cam_coverage = float((cam_resized > 0.3).sum()) / (h * w)

        raw_heatmap = (cam_resized * 255).astype(np.uint8)
        colored_heatmap = cv2.applyColorMap(raw_heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image_bgr, 0.55, colored_heatmap, 0.45, 0)

        class_names = model.names or {}
        target_class_name = class_names.get(target_class_idx, f"class_{target_class_idx}")

        info = {
            "target_class": target_class_name,
            "target_idx": target_class_idx,
            "confidence": round(target_confidence, 4),
            "cam_coverage": round(cam_coverage, 3),
            "heatmap_size": [w, h],
        }

        return overlay, raw_heatmap, info

    finally:
        fwd_handle.remove()
        for name, p in torch_model.named_parameters():
            if name in orig_requires_grad:
                p.requires_grad_(orig_requires_grad[name])
        _activations.clear()


def gradcam_to_base64(overlay: np.ndarray, quality: int = 85) -> str:
    """Encode a BGR overlay image as a base64 data URI."""
    success, buffer = cv2.imencode('.jpg', overlay, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise RuntimeError("Failed to encode heatmap image")
    b64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{b64}"


def generate_gradcam_response(
    model,
    image_bgr: np.ndarray,
    target_class_idx: int | None = None,
) -> dict:
    """Full pipeline: generate Grad-CAM and return JSON-ready response.

    Returns:
        {
            "heatmap_image": "data:image/jpeg;base64,...",
            "raw_heatmap": "data:image/jpeg;base64,...",
            "target_class": "Fusarium Head Blight",
            "target_idx": 2,
            "confidence": 0.87,
            "cam_coverage": 0.34,
            "regions": [...]  # high-activation regions
        }
    """
    try:
        overlay, raw_heatmap, info = generate_gradcam(model, image_bgr, target_class_idx)

        # Find high-activation regions (connected components above threshold)
        regions = _find_activation_regions(raw_heatmap)

        return {
            "heatmap_image": gradcam_to_base64(overlay),
            "raw_heatmap": gradcam_to_base64(
                cv2.applyColorMap(raw_heatmap, cv2.COLORMAP_JET)
            ),
            "target_class": info["target_class"],
            "target_idx": info["target_idx"],
            "confidence": info["confidence"],
            "cam_coverage": info["cam_coverage"],
            "regions": regions,
        }
    except Exception as exc:
        logger.warning(f"Grad-CAM generation failed: {exc}")
        return {"error": str(exc)}


def _find_activation_regions(
    heatmap: np.ndarray,
    threshold: int = 128,
    min_area: int = 100,
) -> list[dict]:
    """Find connected regions of high activation in the heatmap."""
    _, binary = cv2.threshold(heatmap, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    h, w = heatmap.shape[:2]
    total_area = h * w

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        # Mean intensity in region
        mask = np.zeros_like(heatmap)
        cv2.drawContours(mask, [c], -1, 255, -1)
        mean_intensity = cv2.mean(heatmap, mask=mask)[0]

        regions.append({
            "bbox": {"x": x, "y": y, "w": bw, "h": bh},
            "area_pct": round(area / total_area * 100, 1),
            "intensity": round(mean_intensity / 255, 2),
            "centroid": {
                "x": round(x + bw / 2),
                "y": round(y + bh / 2),
            },
        })

    # Sort by intensity descending
    regions.sort(key=lambda r: r["intensity"], reverse=True)
    return regions[:10]  # Top 10 regions
