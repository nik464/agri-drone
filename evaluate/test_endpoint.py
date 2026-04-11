"""Quick end-to-end test: send 3 wheat images to /detect and print results."""
import requests
import json
import os

IMAGES = [
    ("crown_root_rot",    r"D:\Projects\wheat-split\test\crown_root_rot\crown_root_rot_0008.jpg"),
    ("leaf_rust",         r"D:\Projects\wheat-split\test\leaf_rust\leaf_rust_0004.jpg"),
    ("wheat_loose_smut",  r"D:\Projects\wheat-split\test\wheat_loose_smut\wheat_loose_smut_0008.jpg"),
]

SEP = "=" * 70

for label, path in IMAGES:
    fname = os.path.basename(path)
    print(f"\n{SEP}")
    print(f"  TEST: {label}  ->  {fname}")
    print(SEP)

    with open(path, "rb") as f:
        resp = requests.post(
            "http://localhost:9000/detect",
            files={"file": (fname, f, "image/jpeg")},
            data={"crop_type": "wheat", "include_image": "false", "use_llava": "false"},
        )

    if resp.status_code != 200:
        print(f"  ERROR: {resp.status_code} {resp.text[:300]}")
        continue

    data = resp.json()
    s = data.get("structured", {})

    # --- YOLO Classifier ---
    cb = s.get("confidence_breakdown", {})
    sources = cb.get("sources", [])
    yolo_src = next((x for x in sources if x.get("source") == "classifier"), {})
    rule_src = next((x for x in sources if x.get("source") == "rule_engine"), {})

    print(f"  YOLO Prediction  : {yolo_src.get('disease', 'N/A')}")
    print(f"  YOLO Confidence  : {yolo_src.get('score', 0):.1%}")

    # --- Structured Diagnosis (after rule engine) ---
    diag = s.get("diagnosis", {})
    print(f"  Final Diagnosis  : {diag.get('disease_name', 'N/A')} ({diag.get('disease_key', '?')})")
    print(f"  Final Confidence : {diag.get('confidence', 0):.1%}")
    print(f"  Confidence Grade : {diag.get('confidence_grade', 'N/A')}")

    if rule_src:
        print(f"  Rule Engine      : {rule_src.get('disease', 'N/A')} ({rule_src.get('score', 0):.1%})")

    # --- Health ---
    health = s.get("health", {})
    print(f"  Health Score     : {health.get('score', 'N/A')}")
    print(f"  Risk Level       : {health.get('risk_level', 'N/A')}")
    print(f"  Yield Loss Range : {health.get('yield_loss', 'N/A')}")

    # --- Reasoning Chain ---
    chain = s.get("reasoning_chain", [])
    if chain:
        print(f"  Reasoning Chain  :")
        for step in chain:
            print(f"    {step}")

    # --- Treatment ---
    treat = s.get("treatment", {})
    recs = treat.get("recommendations", [])
    if recs:
        print(f"  Treatment        :")
        for r in recs:
            print(f"    - {r}")
        print(f"  Urgency          : {treat.get('urgency_display', treat.get('urgency', 'N/A'))}")

    # --- Differential ---
    diff = s.get("differential_diagnosis", [])
    if diff:
        print(f"  Differentials    :")
        for d in diff[:3]:
            print(f"    {d.get('disease','?'):25s} {d.get('confidence',0):.1%}  ({d.get('key_difference','')})")

    # --- Meta ---
    meta = s.get("metadata", {})
    ptime = meta.get("processing_time_ms")
    if ptime:
        print(f"  Processing Time  : {ptime:.0f}ms")

    # --- Verdict ---
    yolo_key = yolo_src.get("disease", "").lower().replace(" ", "_")
    match = "CORRECT" if yolo_key == label else "WRONG"
    print(f"  Ground Truth     : {label}")
    print(f"  YOLO Verdict     : {match}")

print(f"\n{SEP}")
print("  END-TO-END TEST COMPLETE")
print(SEP)
