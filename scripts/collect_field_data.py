#!/usr/bin/env python3
"""
collect_field_data.py — Interactive tool for building a labelled Indian
wheat / rice training dataset from drone or smartphone images.

Workflow
--------
1. Point at a folder of field images.
2. The current YOLOv8 model runs on each image and draws detections.
3. Review each image and press:
      SPACE  — accept current detections
      W      — set crop type to wheat, then accept
      R      — set crop type to rice, then accept
      D      — skip / discard this image
      E      — enter manual bbox-drawing mode
      Q      — quit early (progress is saved)
4. Labels are saved in YOLO format under  data/labels/
5. A collection_log.json is written with session metadata.

Usage
-----
    python scripts/collect_field_data.py  path/to/images/
    python scripts/collect_field_data.py  path/to/images/ --model yolov8n-seg.pt --device cpu
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

# ── Project root (repo top-level) ────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── 14 India-specific classes (same order as configs/model.yaml) ─────
CLASSES: dict[int, str] = {
    0: "healthy_wheat",
    1: "wheat_lodging",
    2: "wheat_leaf_rust",
    3: "wheat_yellow_rust",
    4: "wheat_powdery_mildew",
    5: "wheat_nitrogen_def",
    6: "wheat_weed",
    7: "healthy_rice",
    8: "rice_blast",
    9: "rice_brown_planthopper",
    10: "rice_water_stress",
    11: "rice_weed",
    12: "poor_row_spacing",
    13: "good_row_spacing",
}
NAME_TO_ID: dict[str, int] = {v: k for k, v in CLASSES.items()}
NUM_CLASSES = len(CLASSES)

# Crop-type lookup
WHEAT_IDS = set(range(0, 7))
RICE_IDS = set(range(7, 14))

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Colours: one per class (BGR)
_RNG = np.random.RandomState(42)
CLASS_COLOURS = {
    cid: tuple(int(c) for c in _RNG.randint(60, 255, 3))
    for cid in CLASSES
}

# ── Helpers ───────────────────────────────────────────────────────────

def _collect_images(folder: Path) -> list[Path]:
    """Return sorted list of image paths in *folder*."""
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def _run_yolo(model, image: np.ndarray, conf: float) -> list[dict]:
    """Run YOLOv8 on a BGR image.  Returns list of dicts with keys
    class_id, class_name, conf, bbox_xyxy (pixels), bbox_yolo (normalised).
    """
    h, w = image.shape[:2]
    results = model(image, conf=conf, verbose=False)
    detections: list[dict] = []
    if not results:
        return detections
    result = results[0]
    if result.boxes is None:
        return detections
    for box in result.boxes:
        cid = int(box.cls.item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        name = CLASSES.get(cid, result.names.get(cid, f"class_{cid}"))
        detections.append({
            "class_id": cid,
            "class_name": name,
            "conf": float(box.conf.item()),
            "bbox_xyxy": (int(x1), int(y1), int(x2), int(y2)),
            "bbox_yolo": (cx, cy, bw, bh),
        })
    return detections


def _draw_detections(
    image: np.ndarray, detections: list[dict], crop_label: str
) -> np.ndarray:
    """Return a copy of *image* with bounding boxes and class labels drawn."""
    vis = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox_xyxy"]
        cid = det["class_id"]
        col = CLASS_COLOURS.get(cid, (0, 255, 0))
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
        label = f"{det['class_name']} {det['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), col, -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # HUD bar at top
    bar_h = 36
    cv2.rectangle(vis, (0, 0), (vis.shape[1], bar_h), (40, 40, 40), -1)
    hud = (
        f"Crop: {crop_label}  |  Detections: {len(detections)}  |  "
        "SPACE=accept  W=wheat  R=rice  D=skip  E=edit  Q=quit"
    )
    cv2.putText(vis, hud, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (200, 255, 200), 1, cv2.LINE_AA)
    return vis


def _guess_crop(detections: list[dict]) -> str:
    """Guess crop type from majority class ids."""
    wheat = sum(1 for d in detections if d["class_id"] in WHEAT_IDS)
    rice = sum(1 for d in detections if d["class_id"] in RICE_IDS)
    if wheat > rice:
        return "wheat"
    if rice > wheat:
        return "rice"
    return "unknown"


# ── Manual bbox editor ────────────────────────────────────────────────

class _BBoxEditor:
    """Simple OpenCV-based bounding-box drawing tool."""

    def __init__(self, image: np.ndarray):
        self.original = image.copy()
        self.canvas = image.copy()
        self.boxes: list[tuple[int, int, int, int]] = []  # (x1,y1,x2,y2)
        self.box_classes: list[int] = []
        self.drawing = False
        self.ix = 0
        self.iy = 0

    # ── mouse callback ────────────────────────────────────────────────
    def _mouse_cb(self, event: int, x: int, y: int, flags: int, _: object):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            tmp = self.canvas.copy()
            cv2.rectangle(tmp, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
            cv2.imshow("Edit", tmp)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            x1, y1 = min(self.ix, x), min(self.iy, y)
            x2, y2 = max(self.ix, x), max(self.iy, y)
            if (x2 - x1) > 5 and (y2 - y1) > 5:
                self.boxes.append((x1, y1, x2, y2))
                self.box_classes.append(-1)  # assigned later
                cv2.rectangle(self.canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imshow("Edit", self.canvas)

    # ── class picker overlay ──────────────────────────────────────────
    @staticmethod
    def _show_class_menu() -> None:
        """Print class menu to the terminal (more readable than overlay)."""
        print("\n--- Select class ID ---")
        for cid, name in CLASSES.items():
            print(f"  [{cid:>2}] {name}")
        print("  Type class ID and press Enter:")

    def _pick_class(self) -> int:
        """Ask user to type a class id in the terminal."""
        self._show_class_menu()
        while True:
            try:
                raw = input("  > ").strip()
                cid = int(raw)
                if cid in CLASSES:
                    return cid
                print(f"  Invalid — enter 0..{NUM_CLASSES - 1}")
            except (ValueError, EOFError):
                print(f"  Invalid — enter 0..{NUM_CLASSES - 1}")

    # ── run ───────────────────────────────────────────────────────────
    def run(self) -> list[dict]:
        """Open an edit window.  Returns list of detection dicts."""
        win = "Edit"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win, self._mouse_cb)

        bar_h = 30
        overlay = self.canvas.copy()
        cv2.rectangle(overlay, (0, 0), (overlay.shape[1], bar_h), (40, 40, 40), -1)
        help_text = "Draw boxes (click-drag)  |  ENTER=done  ESC=cancel"
        cv2.putText(overlay, help_text, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
        self.canvas = overlay
        cv2.imshow(win, self.canvas)

        while True:
            key = cv2.waitKey(50) & 0xFF
            if key == 13:  # Enter
                break
            if key == 27:  # Escape — discard edits
                cv2.destroyWindow(win)
                return []

        cv2.destroyWindow(win)

        # Assign classes to each drawn box
        h, w = self.original.shape[:2]
        detections: list[dict] = []
        for idx, (x1, y1, x2, y2) in enumerate(self.boxes):
            cid = self._pick_class()
            self.box_classes[idx] = cid
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            detections.append({
                "class_id": cid,
                "class_name": CLASSES[cid],
                "conf": 1.0,
                "bbox_xyxy": (x1, y1, x2, y2),
                "bbox_yolo": (cx, cy, bw, bh),
            })
        return detections


# ── Save helpers ──────────────────────────────────────────────────────

def _save_label(
    label_dir: Path, image_dir: Path,
    image_path: Path, detections: list[dict],
) -> None:
    """Copy the image and write a YOLO-format .txt label file."""
    label_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    stem = image_path.stem
    suffix = image_path.suffix

    # Copy image
    dst_img = image_dir / f"{stem}{suffix}"
    if not dst_img.exists():
        shutil.copy2(image_path, dst_img)

    # Write label  — YOLO format: class_id cx cy w h
    label_file = label_dir / f"{stem}.txt"
    lines: list[str] = []
    for det in detections:
        cid = det["class_id"]
        cx, cy, bw, bh = det["bbox_yolo"]
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    label_file.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_log(
    log_path: Path,
    *,
    input_folder: str,
    crop_type: str,
    total_images: int,
    accepted: int,
    skipped: int,
    edited: int,
    labels_saved: int,
) -> None:
    """Append a session record to collection_log.json."""
    record = {
        "date": datetime.now(timezone.utc).isoformat(),
        "field_location": input_folder,
        "crop_type": crop_type,
        "total_images": total_images,
        "accepted_images": accepted,
        "skipped_images": skipped,
        "manually_edited": edited,
        "labels_saved": labels_saved,
    }
    # Append to existing log (or start fresh)
    entries: list[dict] = []
    if log_path.exists():
        try:
            entries = json.loads(log_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError):
            entries = []
    entries.append(record)
    log_path.write_text(json.dumps(entries, indent=2, default=str) + "\n",
                        encoding="utf-8")
    print(f"\n  Log saved → {log_path}")


# ── Main loop ─────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect & label Indian wheat/rice field images for YOLO training."
    )
    parser.add_argument("input_folder", type=Path,
                        help="Folder of field images (drone or phone camera).")
    parser.add_argument("--model", type=str, default="yolov8n-seg.pt",
                        help="YOLOv8 model checkpoint (default: yolov8n-seg.pt)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Inference device: cpu or cuda (default: cpu)")
    parser.add_argument("--confidence", type=float, default=0.4,
                        help="Detection confidence threshold (default: 0.4)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output root (default: <project>/data)")
    args = parser.parse_args()

    # Resolve paths
    input_folder = args.input_folder.resolve()
    if not input_folder.is_dir():
        sys.exit(f"Error: {input_folder} is not a directory.")

    output_root = (args.output or PROJECT_ROOT / "data").resolve()
    label_dir = output_root / "labels"
    image_dir = output_root / "images"
    log_path = output_root / "collection_log.json"

    images = _collect_images(input_folder)
    if not images:
        sys.exit(f"No images found in {input_folder}")

    # Load model
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics not installed — run: pip install ultralytics")

    model_path = Path(args.model)
    if not model_path.exists():
        model_path = PROJECT_ROOT / args.model
    print(f"Loading model: {model_path}  (device={args.device})")
    model = YOLO(str(model_path))
    if args.device == "cuda":
        model.to("cuda")

    # Session counters
    accepted = 0
    skipped = 0
    edited = 0
    labels_saved = 0
    session_crop = "unknown"

    win = "AgriDrone — Field Data Collector"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    print(f"\n  {len(images)} images in {input_folder}")
    print("  Controls:  SPACE=accept  W=wheat  R=rice  D=skip  E=edit  Q=quit\n")

    for idx, img_path in enumerate(images):
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  [{idx+1}/{len(images)}] Skipping unreadable: {img_path.name}")
            skipped += 1
            continue

        # Run model
        detections = _run_yolo(model, image, args.confidence)
        crop_guess = _guess_crop(detections) if detections else session_crop

        # Show image with detections
        title = f"[{idx+1}/{len(images)}] {img_path.name}"
        vis = _draw_detections(image, detections, crop_guess)
        cv2.setWindowTitle(win, title)
        cv2.imshow(win, vis)

        # Wait for user action
        while True:
            key = cv2.waitKey(0) & 0xFF

            if key == ord(" "):  # SPACE — accept
                _save_label(label_dir, image_dir, img_path, detections)
                accepted += 1
                labels_saved += len(detections)
                session_crop = crop_guess
                print(f"  [{idx+1}] ACCEPTED  {img_path.name}  "
                      f"({len(detections)} dets, crop={crop_guess})")
                break

            elif key == ord("w"):  # W — mark wheat + accept
                session_crop = "wheat"
                _save_label(label_dir, image_dir, img_path, detections)
                accepted += 1
                labels_saved += len(detections)
                print(f"  [{idx+1}] WHEAT     {img_path.name}  "
                      f"({len(detections)} dets)")
                break

            elif key == ord("r"):  # R — mark rice + accept
                session_crop = "rice"
                _save_label(label_dir, image_dir, img_path, detections)
                accepted += 1
                labels_saved += len(detections)
                print(f"  [{idx+1}] RICE      {img_path.name}  "
                      f"({len(detections)} dets)")
                break

            elif key == ord("d"):  # D — skip
                skipped += 1
                print(f"  [{idx+1}] SKIPPED   {img_path.name}")
                break

            elif key == ord("e"):  # E — manual edit
                editor = _BBoxEditor(image)
                manual_dets = editor.run()
                if manual_dets:
                    detections = manual_dets
                    crop_guess = _guess_crop(detections)
                    edited += 1
                    # Re-draw so user can review before accepting
                    vis = _draw_detections(image, detections, crop_guess)
                    cv2.imshow(win, vis)
                    print(f"  [{idx+1}] EDITED    {img_path.name}  "
                          f"({len(detections)} boxes drawn)")
                    # Fall through — user still needs to press SPACE/W/R to accept
                else:
                    print(f"  [{idx+1}] Edit cancelled — press another key")

            elif key == ord("q"):  # Q — quit early
                print("\n  Quitting early — saving progress …")
                cv2.destroyAllWindows()
                _write_log(
                    log_path,
                    input_folder=str(input_folder),
                    crop_type=session_crop,
                    total_images=len(images),
                    accepted=accepted,
                    skipped=skipped,
                    edited=edited,
                    labels_saved=labels_saved,
                )
                print(f"\n  Session: {accepted} accepted, {skipped} skipped, "
                      f"{edited} edited, {labels_saved} labels")
                return

    cv2.destroyAllWindows()

    # Write session log
    _write_log(
        log_path,
        input_folder=str(input_folder),
        crop_type=session_crop,
        total_images=len(images),
        accepted=accepted,
        skipped=skipped,
        edited=edited,
        labels_saved=labels_saved,
    )
    print(f"\n  Done! {accepted} accepted, {skipped} skipped, "
          f"{edited} edited, {labels_saved} labels")
    print(f"  Images → {image_dir}")
    print(f"  Labels → {label_dir}")


if __name__ == "__main__":
    main()
