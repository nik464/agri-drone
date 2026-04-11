import csv
from collections import Counter

with open("evaluate/results/predictions_B_yolo_rules.csv") as f:
    rows = list(csv.DictReader(f))

pred_counts = Counter(r["predicted"] for r in rows)
print("=== Config B predicted classes (top 15) ===")
for cls, n in pred_counts.most_common(15):
    print(f"  {cls:45s} {n:4d}")

mismatches = [(r["ground_truth"], r["predicted"]) for r in rows if r["correct"] == "0"]
mismatch_counts = Counter(mismatches)
print(f"\n=== Top 15 confusion pairs (gt -> pred) ===")
for (gt, pr), n in mismatch_counts.most_common(15):
    print(f"  {gt:35s} -> {pr:35s} {n:3d}")

print(f"\nTotal: {len(rows)}, Correct: {sum(1 for r in rows if r['correct']=='1')}")
print(f"Unique predicted: {len(pred_counts)}")
