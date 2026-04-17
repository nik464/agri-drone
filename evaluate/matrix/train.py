"""Shared training entrypoint used by the matrix runner.

Implements real training for the six backbones documented in
``docs/training_recipe.md`` on the PlantVillage classification subset
(Kaggle slug ``abdallahalidev/plantvillage-dataset``, ``color/`` variant).

Design goals (unchanged from the scaffolding commit):
* **One recipe, all backbones.** Optimizer, LR schedule, augmentation,
  label smoothing come from ``cfg["training_recipe"]``.
* **Non-breaking.** Does not import from ``src/agridrone`` and cannot
  mutate v1 frozen results.
* **CPU-safe imports.** torch/torchvision/ultralytics load inside the
  call path so ``--dry-run`` never pulls them in.
* **Graceful degradation.** If torch or the dataset is missing, return
  ``status: "skipped"`` with a clear ``notes`` field (keeps CI green on
  non-GPU hosts).

The returned record always conforms to ``docs/results_schema.md``.
"""

from __future__ import annotations

import datetime as _dt
import os
import random
import shutil
import time
from pathlib import Path

# Public API
__all__ = ["train_and_eval", "BACKBONE_REGISTRY"]


BACKBONE_REGISTRY = {
    "yolov8n-cls":       {"family": "yolo",        "params": 1_440_000},
    "yolov8s-cls":       {"family": "yolo",        "params": 6_360_000},
    "efficientnet_b0":   {"family": "torchvision", "params": 4_034_449},
    "convnext_tiny":     {"family": "torchvision", "params": 28_589_128},
    "mobilenetv3_small": {"family": "torchvision", "params": 2_542_856},
    "vit_b_16":          {"family": "torchvision", "params": 86_567_656},
}


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SPLIT_CACHE: dict[tuple, Path] = {}


# ---------------------------------------------------------------------------
# Dataset resolution + on-disk split materialization
# ---------------------------------------------------------------------------

def _resolve_plantvillage_root() -> Path | None:
    """Locate the PlantVillage ``color/`` ImageFolder root on disk.

    Matches the two unpacking variants produced by the Kaggle archive
    ``abdallahalidev/plantvillage-dataset`` and a few common rewrites.
    """
    candidates = [
        PROJECT_ROOT / "datasets" / "externals" / "plantvillage dataset" / "color",
        PROJECT_ROOT / "datasets" / "externals" / "PlantVillage" / "color",
        PROJECT_ROOT / "datasets" / "externals" / "plantvillage" / "color",
        PROJECT_ROOT / "datasets" / "externals" / "color",
    ]
    for c in candidates:
        if c.is_dir() and any(c.iterdir()):
            return c
    return None


def _build_splits(src: Path, seed: int, train_fraction: float,
                  pool_cap_per_class: int = 300) -> Path:
    """Deterministic 80/10/10 split cached under ``datasets/_matrix_cache/``.

    To keep the Colab T4 quick-matrix run under ~1 h for 8 cells, the per-class
    pool is capped at ``pool_cap_per_class`` images before splitting. The test
    and val splits are sized from the full pool, then ``train_fraction`` scales
    only the training split (so the test set stays comparable across cells).
    """
    key = (str(src), seed, float(train_fraction), pool_cap_per_class)
    cached = _SPLIT_CACHE.get(key)
    if cached is not None and cached.is_dir():
        return cached

    rng = random.Random(seed)
    split_root = (
        PROJECT_ROOT / "datasets" / "_matrix_cache"
        / f"pv_seed{seed}_cap{pool_cap_per_class}_frac{train_fraction}"
    )
    if split_root.exists():
        shutil.rmtree(split_root, ignore_errors=True)

    for cls_dir in sorted(src.iterdir()):
        if not cls_dir.is_dir():
            continue
        imgs = [p for p in cls_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        imgs.sort()
        rng.shuffle(imgs)
        imgs = imgs[:pool_cap_per_class]
        n = len(imgs)
        if n < 10:
            continue
        n_test = max(1, n // 10)
        n_val = max(1, n // 10)
        n_train_full = n - n_val - n_test
        n_train = max(1, int(round(n_train_full * float(train_fraction))))

        splits = {
            "train": imgs[:n_train],
            "val":   imgs[n_train_full:n_train_full + n_val],
            "test":  imgs[n_train_full + n_val:],
        }
        for split_name, items in splits.items():
            dst = split_root / split_name / cls_dir.name
            dst.mkdir(parents=True, exist_ok=True)
            for src_img in items:
                link = dst / src_img.name
                try:
                    os.symlink(src_img, link)
                except (OSError, NotImplementedError):
                    shutil.copy2(src_img, link)

    _SPLIT_CACHE[key] = split_root
    return split_root


# ---------------------------------------------------------------------------
# Metric helpers (no sklearn dependency — paper already locks numpy usage)
# ---------------------------------------------------------------------------

def _metrics_from_confusion(cm, class_names: list[str]) -> dict:
    import numpy as np

    cm = np.asarray(cm, dtype=np.float64)
    n_classes = cm.shape[0]
    total = cm.sum()
    acc = float(np.trace(cm) / max(total, 1))

    per_class_f1 = {}
    f1s = []
    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class_f1[class_names[i]] = round(float(f1), 4)
        f1s.append(f1)
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    # Multiclass MCC (Gorodkin, 2004).
    t = cm.sum(axis=1)
    p = cm.sum(axis=0)
    s = total
    c = float(np.trace(cm))
    num = c * s - float((p * t).sum())
    den = ((s * s - float((p * p).sum())) * (s * s - float((t * t).sum()))) ** 0.5
    mcc = float(num / den) if den > 0 else 0.0

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "mcc": round(mcc, 4),
        "per_class_f1": per_class_f1,
    }


def _measure_latency(forward_fn, n_warmup: int = 5, n_iters: int = 30) -> dict:
    import numpy as np

    for _ in range(n_warmup):
        forward_fn()
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        forward_fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    t = np.asarray(times)
    return {
        "mean": round(float(t.mean()), 2),
        "p50": round(float(np.percentile(t, 50)), 2),
        "p95": round(float(np.percentile(t, 95)), 2),
    }


# ---------------------------------------------------------------------------
# Training — torchvision family
# ---------------------------------------------------------------------------

def _build_torchvision_model(backbone: str, n_classes: int):
    import torch.nn as nn
    from torchvision import models

    if backbone == "efficientnet_b0":
        m = models.efficientnet_b0(weights="IMAGENET1K_V1")
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)
    elif backbone == "convnext_tiny":
        m = models.convnext_tiny(weights="IMAGENET1K_V1")
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, n_classes)
    elif backbone == "mobilenetv3_small":
        m = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, n_classes)
    elif backbone == "vit_b_16":
        m = models.vit_b_16(weights="IMAGENET1K_V1")
        m.heads.head = nn.Linear(m.heads.head.in_features, n_classes)
    else:
        raise ValueError(f"unsupported torchvision backbone: {backbone}")
    return m


def _train_torchvision(backbone: str, data_dir: Path, cfg: dict, device) -> dict:
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    recipe = cfg.get("training_recipe", {})
    epochs = int(recipe.get("epochs", 5))
    batch = int(recipe.get("batch_size", 32))
    lr = float(recipe.get("lr", 1e-3))
    wd = float(recipe.get("weight_decay", 1e-4))
    img_size = int(recipe.get("input_size", 224))
    label_smoothing = float(recipe.get("label_smoothing", 0.0))

    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        normalize,
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])

    train_ds = datasets.ImageFolder(str(data_dir / "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(str(data_dir / "val"),   transform=eval_tf)
    test_ds  = datasets.ImageFolder(str(data_dir / "test"),  transform=eval_tf)
    class_names = train_ds.classes
    n_classes = len(class_names)

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                              num_workers=2, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch, shuffle=False,
                              num_workers=2, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=batch, shuffle=False,
                              num_workers=2, pin_memory=pin)

    model = _build_torchvision_model(backbone, n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    best_val = -1.0
    best_state = None
    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=pin)
            y = y.to(device, non_blocking=pin)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=pin)
                y = y.to(device, non_blocking=pin)
                pred = model(x).argmax(1)
                correct += int((pred == y).sum().item())
                total += int(y.size(0))
        val_acc = correct / max(total, 1)
        print(f"    ep{ep + 1}/{epochs}  val_acc={val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=pin)
            pred = model(x).argmax(1).cpu().numpy()
            for t_i, p_i in zip(y.numpy(), pred):
                cm[int(t_i), int(p_i)] += 1
    metrics = _metrics_from_confusion(cm, class_names)

    x_one = torch.randn(1, 3, img_size, img_size, device=device)

    def _fwd():
        with torch.no_grad():
            _ = model(x_one)
            if device.type == "cuda":
                import torch as _t
                _t.cuda.synchronize()

    latency = _measure_latency(_fwd)

    return {
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "metrics": metrics,
        "latency_ms": latency,
        "best_val_acc": round(best_val, 4),
    }


# ---------------------------------------------------------------------------
# Training — YOLO classification family
# ---------------------------------------------------------------------------

def _train_yolo(backbone: str, data_dir: Path, cfg: dict, device) -> dict:
    import numpy as np
    from torchvision import datasets as _ds
    from ultralytics import YOLO

    recipe = cfg.get("training_recipe", {})
    epochs = int(recipe.get("epochs", 5))
    batch = int(recipe.get("batch_size", 32))
    lr = float(recipe.get("lr", 1e-3))
    img_size = int(recipe.get("input_size", 224))

    weights_name = backbone + ".pt"  # e.g. "yolov8n-cls.pt"
    yolo_device = 0 if device.type == "cuda" else "cpu"
    model = YOLO(weights_name)
    model.train(
        data=str(data_dir),
        epochs=epochs,
        imgsz=img_size,
        batch=batch,
        device=yolo_device,
        workers=2,
        verbose=False,
        exist_ok=True,
        project=str(data_dir / "_yolo_runs"),
        name=backbone,
        optimizer="AdamW",
        lr0=lr,
    )

    # Inference over the held-out test split.
    test_ds = _ds.ImageFolder(str(data_dir / "test"))
    class_names = test_ds.classes
    n_classes = len(class_names)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for path, label in test_ds.samples:
        r = model.predict(path, verbose=False, imgsz=img_size, device=yolo_device)[0]
        pred_idx_yolo = int(r.probs.top1)
        pred_name = r.names.get(pred_idx_yolo, "")
        pred_idx = test_ds.class_to_idx.get(pred_name, pred_idx_yolo)
        cm[int(label), int(pred_idx)] += 1
    metrics = _metrics_from_confusion(cm, class_names)

    probe_path = test_ds.samples[0][0]

    def _fwd():
        _ = model.predict(probe_path, verbose=False, imgsz=img_size, device=yolo_device)

    latency = _measure_latency(_fwd, n_warmup=3, n_iters=30)

    n_train = sum(1 for _ in (data_dir / "train").rglob("*")
                  if _.suffix.lower() in {".jpg", ".jpeg", ".png"})
    n_val = sum(1 for _ in (data_dir / "val").rglob("*")
                if _.suffix.lower() in {".jpg", ".jpeg", ".png"})

    return {
        "n_train": n_train,
        "n_val": n_val,
        "n_test": len(test_ds.samples),
        "metrics": metrics,
        "latency_ms": latency,
    }


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def train_and_eval(cell, cfg: dict) -> dict:
    backbone = cell.backbone
    rule = cell.rule_engine

    if backbone not in BACKBONE_REGISTRY:
        return _record(cell, cfg, status="failed",
                       notes=f"unknown backbone: {backbone}")

    try:
        import torch
    except Exception as e:  # noqa: BLE001
        return _record(cell, cfg, status="skipped",
                       notes=f"torch not installed on this host: {e}")

    family = BACKBONE_REGISTRY[backbone]["family"]
    src = _resolve_plantvillage_root()
    if src is None:
        return _record(cell, cfg, status="skipped",
                       notes=("PlantVillage color/ dir not found under "
                              "datasets/externals/. Run notebook 1 data-source "
                              "cell first (DATA_SOURCE=kaggle)."))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pool_cap = int(cfg.get("training_recipe", {}).get("pool_cap_per_class", 300))
    try:
        data_dir = _build_splits(src, cell.seed, cell.train_fraction,
                                 pool_cap_per_class=pool_cap)
        print(f"  [{cell.slug()}] training {backbone} ({family}) on {data_dir}")
        if family == "yolo":
            try:
                import ultralytics  # noqa: F401
            except Exception as e:  # noqa: BLE001
                return _record(cell, cfg, status="skipped",
                               notes=f"ultralytics not installed: {e}")
            out = _train_yolo(backbone, data_dir, cfg, device)
        else:
            out = _train_torchvision(backbone, data_dir, cfg, device)
    except Exception as e:  # noqa: BLE001
        return _record(cell, cfg, status="failed",
                       notes=f"training raised: {type(e).__name__}: {e}")

    rule_note = (
        "rule=none (backbone-only prediction)"
        if rule == "none"
        else f"rule={rule} (pass-through for pv_subset classification task)"
    )
    return _record(
        cell, cfg,
        status="ok",
        metrics=out["metrics"],
        latency=out["latency_ms"],
        notes=f"device={device.type}; pv_subset; {rule_note}",
        n_train=out["n_train"],
        n_val=out["n_val"],
        n_test=out["n_test"],
    )


def _record(cell, cfg: dict, *, status: str, notes: str,
            metrics: dict | None = None, latency: dict | None = None,
            n_train: int = 0, n_val: int = 0, n_test: int = 0) -> dict:
    return {
        "run_id": cfg["run_id"],
        "backbone": cell.backbone,
        "rule_engine": cell.rule_engine,
        "train_fraction": cell.train_fraction,
        "dataset": cell.dataset,
        "seed": cell.seed,
        "fold": cell.fold,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "status": status,
        "metrics": metrics,
        "latency_ms": latency,
        "notes": notes,
        "trained_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "training_recipe": "docs/training_recipe.md@v1",
    }
