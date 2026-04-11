#!/usr/bin/env python3
"""Quick test: 4 specific disease images through full pipeline."""
import sys, time, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

import cv2
from ultralytics import YOLO
from agridrone.vision.disease_reasoning import run_full_pipeline, diagnosis_to_dict

model = YOLO('models/india_agri_cls_21class_backup.pt', task='classify')

test_images = [
    ('Crown Root Rot',    'data/training/test/wheat_root_rot/common_root_rot_114.png'),
    ('Leaf Rust',         'data/training/test/wheat_brown_rust/brown_rust_1005.png'),
    ('Wheat Loose Smut',  'data/training/test/wheat_smut/smut_10.png'),
    ('Black Wheat Rust',  'data/training/test/wheat_black_rust/black_rust_1.png'),
]

print('=' * 80)
print('  4-IMAGE VERIFICATION: Rule Engine No Longer Overrides YOLO')
print('=' * 80)

all_pass = True
for label, path in test_images:
    img = cv2.imread(path)
    if img is None:
        print(f'\nERROR: Cannot read {path}')
        all_pass = False
        continue
    
    t0 = time.perf_counter()
    # Run YOLO classifier
    results = model(img, verbose=False)
    probs = results[0].probs
    names = model.names
    top_key = names[probs.top1]
    top_conf = round(probs.top1conf.item(), 4)
    top5_indices = probs.top5
    top5_confs = probs.top5conf.tolist()
    
    classifier_result = {
        "top_prediction": top_key,
        "top_confidence": top_conf,
        "confidence": top_conf,
        "health_score": 95 if "healthy" in top_key else 50,
        "is_healthy": "healthy" in top_key,
        "disease_probability": round(1 - top_conf if "healthy" in top_key else top_conf, 4),
        "top5": [
            {"index": idx, "class_key": names[idx],
             "class_name": names[idx].replace("_", " ").title(),
             "confidence": round(conf, 4)}
            for idx, conf in zip(top5_indices, top5_confs)
        ],
    }
    
    # Run full pipeline
    output = run_full_pipeline(img, classifier_result, "wheat")
    diag = output.diagnosis
    latency = (time.perf_counter() - t0) * 1000
    
    final_key = diag.disease_key
    final_conf = diag.confidence
    
    override = 'healthy' in final_key.lower() and 'healthy' not in label.lower()
    if override:
        all_pass = False
    
    fname = os.path.basename(path)
    print(f'\n--- {label} ({fname}) ---')
    print(f'  YOLO:     {top_key} ({top_conf:.1%})')
    print(f'  Final:    {final_key} ({final_conf:.1%})')
    print(f'  Latency:  {latency:.0f} ms')
    print(f'  Override: {"YES - BUG!" if override else "NO - CORRECT"}')

print(f'\n{"=" * 80}')
print(f'  {"ALL 4 TESTS PASSED" if all_pass else "SOME TESTS FAILED!"}')
print(f'{"=" * 80}')
