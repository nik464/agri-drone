"""Quick smoke test for sensitivity_analysis.py (optimized version)"""
import sys, time
sys.path.insert(0, 'src')
sys.path.insert(0, 'evaluate')

from ultralytics import YOLO
import cv2
from pathlib import Path
from agridrone.knowledge import kb_loader

# Suppress debug logging
try:
    from loguru import logger
    logger.disable("agridrone.vision.feature_extractor")
    logger.disable("agridrone.vision.rule_engine")
except ImportError:
    pass

model = YOLO('models/india_agri_cls.pt', task='classify')
kb_loader.load()

val_dir = Path('data/training/val')
img_path = next((val_dir / 'rice_blast').glob('*'))
bgr = cv2.imread(str(img_path))
print(f'Image: {img_path.name}')

from sensitivity_analysis import precompute_yolo_and_features, run_rule_engine_patched, compute_metrics_fast

# Precompute once
loaded = [(bgr, 'rice_blast', 'rice')]
t0 = time.perf_counter()
cached = precompute_yolo_and_features(model, loaded)
print(f'Precompute: {time.perf_counter()-t0:.2f}s')

# Test default params
t0 = time.perf_counter()
p1 = run_rule_engine_patched(cached[0], 20, 0.5, 0.85)
print(f'Default (cs=20, sw=0.5, yt=0.85): {p1} ({time.perf_counter()-t0:.3f}s)')

# Test alt params
t0 = time.perf_counter()
p2 = run_rule_engine_patched(cached[0], 15, 0.3, 0.95)
print(f'Alt    (cs=15, sw=0.3, yt=0.95): {p2} ({time.perf_counter()-t0:.3f}s)')

# Test extreme
t0 = time.perf_counter()
p3 = run_rule_engine_patched(cached[0], 25, 0.7, 0.75)
print(f'Extreme(cs=25, sw=0.7, yt=0.75): {p3} ({time.perf_counter()-t0:.3f}s)')

# Quick metrics test
gts = ['rice_blast', 'rice_blast', 'healthy_rice']
prs = ['rice_blast', 'rice_brown_spot', 'healthy_rice']
m = compute_metrics_fast(gts, prs, sorted(set(gts) | set(prs)))
print(f"Metrics test: acc={m['accuracy']:.2f} rwa={m['rwa']:.2f} f1={m['macro_f1']:.2f}")

print('ALL TESTS PASSED')
