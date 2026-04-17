"""Helper to scaffold the three Colab notebooks deterministically.

Only used during the research-upgrade commit sequence; safe to keep, safe to
delete. Running it regenerates the three notebooks under notebooks/colab/.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "notebooks" / "colab"
OUT.mkdir(parents=True, exist_ok=True)


def _md(*lines: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}


def _code(*lines: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": [l + "\n" for l in lines]}


NB_META = {
    "accelerator": "GPU",
    "colab": {"provenance": []},
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
    "language_info": {"name": "python"},
}


def write(path: Path, cells: list[dict], name: str) -> None:
    nb = {"cells": cells, "metadata": {**NB_META,
          "colab": {"name": name, "provenance": []}},
          "nbformat": 4, "nbformat_minor": 5}
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")


# ------------------------------------------------------------------ NB 1
nb1 = [
    _md("# agri-drone — Notebook 1: full experimental matrix",
        "",
        "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
        "(https://colab.research.google.com/github/Ashut0sh-mishra/agri-drone/"
        "blob/research-upgrade/notebooks/colab/01_run_matrix.ipynb)",
        "",
        "Runs the research-upgrade evaluation matrix end-to-end on Colab.",
        "- **Quick mode** (default, T4, ~2 h) = 8 cells",
        "- **Full mode** (A100, ~12 h) = 2400 cells",
        "",
        "Outputs land under `MyDrive/agri-drone/results_v2/`."),
    _md("## 1. Mount Drive + clone repo"),
    _code("from google.colab import drive",
          "drive.mount('/content/drive')"),
    _code("%cd /content",
          "!test -d agri-drone || git clone https://github.com/Ashut0sh-mishra/agri-drone.git",
          "%cd /content/agri-drone",
          "!git fetch --all && git checkout research-upgrade && git pull",
          "!pip install -q -r requirements.txt"),
    _md("## 2. GPU sanity check"),
    _code("!nvidia-smi",
          "import torch",
          "assert torch.cuda.is_available(), 'No GPU. Runtime -> Change runtime type -> GPU.'",
          "print('CUDA', torch.version.cuda, 'device:', torch.cuda.get_device_name(0))"),
    _md("## 3. Data source",
        "",
        "Pre-create in your Drive and upload datasets:",
        "```",
        "MyDrive/agri-drone/data/plantvillage/",
        "MyDrive/agri-drone/data/PDT_datasets/",
        "```",
        "See `docs/data_availability.md` for URLs and licences."),
    _code("#@title Data source",
          "DATA_SOURCE = 'gdrive' #@param ['gdrive','kaggle','zenodo']",
          "DRIVE_DATA = '/content/drive/MyDrive/agri-drone/data'",
          "DRIVE_OUT  = '/content/drive/MyDrive/agri-drone/results_v2'",
          "import os, pathlib",
          "pathlib.Path(DRIVE_OUT).mkdir(parents=True, exist_ok=True)",
          "if DATA_SOURCE == 'gdrive':",
          "    assert os.path.isdir(DRIVE_DATA), f'Missing {DRIVE_DATA}. Upload dataset first.'",
          "    !mkdir -p datasets/externals && ln -sfn \"$DRIVE_DATA\" datasets/externals/drive",
          "elif DATA_SOURCE == 'kaggle':",
          "    print('Configure ~/.kaggle/kaggle.json and pull per scripts/download_data.py')",
          "elif DATA_SOURCE == 'zenodo':",
          "    print('Replace with your Zenodo archive URL and unzip to datasets/externals/')",
          "print('(SHA256 validation advisory per docs/data_availability.md)')"),
    _md("## 4. Mode selector"),
    _code("#@title Mode",
          "FULL_MODE = False #@param {type:'boolean'}",
          "MATRIX_CONFIG = 'configs/matrix/full.yaml' if FULL_MODE else 'configs/matrix/quick.yaml'",
          "if FULL_MODE:",
          "    print('*** FULL MODE: 2400 cells, ~12 h on A100. Need Colab Pro. ***')",
          "else:",
          "    print('QUICK MODE: 8 cells, ~2 h on T4.')",
          "print('config =', MATRIX_CONFIG)"),
    _md("## 5. Run the matrix"),
    _code("!python evaluate/matrix/run_matrix.py --config $MATRIX_CONFIG \\",
          "    --out-dir $DRIVE_OUT/matrix --tracker none"),
    _md("## 6. Post-matrix analysis (stats v2, EML, baseline audit, PDT sweep)"),
    _code("!python evaluate/statistical_tests.py --v2 --n-boot 10000",
          "!python evaluate/eml_sensitivity.py",
          "!python evaluate/matrix/audit_baseline.py",
          "!python evaluate/pdt_v2.py --variant threshold_sweep"),
    _md("## 7. Auto-generate RESULTS_SUMMARY.md"),
    _code("import json, pathlib",
          "out = pathlib.Path(DRIVE_OUT); out.mkdir(parents=True, exist_ok=True)",
          "def _load(rel):",
          "    p = pathlib.Path('evaluate/results/v2') / rel",
          "    return json.loads(p.read_text()) if p.exists() else {}",
          "eml = _load('eml/headline_v4.json')",
          "pdt = _load('pdt/threshold_sweep.json')",
          "lines = [",
          "    '# Results summary (auto-generated by notebook 1)',",
          "    f'- Matrix config: `{MATRIX_CONFIG}`',",
          "    f'- EML headline (INR/ha): {eml.get(\"headline_eml_inr_per_ha\",\"[TO BE RE-RUN]\")}',",
          "    f'- EML sensitivity scenario: {eml.get(\"sensitivity_scenario_eml_inr_per_ha\",\"[TO BE RE-RUN]\")}',",
          "    f'- PDT ROC-AUC: {pdt.get(\"roc_auc\",\"[TO BE RE-RUN]\")}',",
          "    f'- PDT spec@90%sens: {pdt.get(\"specificity_at_90_sensitivity\",\"[TO BE RE-RUN]\")}',",
          "]",
          "(out / 'RESULTS_SUMMARY.md').write_text('\\n'.join(lines))",
          "print((out / 'RESULTS_SUMMARY.md').read_text())"),
    _md("## 8. Zip artifacts + print PR-comment command"),
    _code("import shutil, time",
          "stamp = time.strftime('%Y%m%d_%H%M%S')",
          "archive = f'/content/drive/MyDrive/agri-drone/results_v2_{stamp}'",
          "shutil.make_archive(archive, 'zip', DRIVE_OUT)",
          "print('Saved:', archive + '.zip')",
          "print('\\nPost summary back to PR locally with:')",
          "print('  gh pr comment research-upgrade --body-file RESULTS_SUMMARY.md')"),
]
write(OUT / "01_run_matrix.ipynb", nb1, "01_run_matrix")


# ------------------------------------------------------------------ NB 2
nb2 = [
    _md("# agri-drone — Notebook 2: PDT calibration + few-shot",
        "",
        "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
        "(https://colab.research.google.com/github/Ashut0sh-mishra/agri-drone/"
        "blob/research-upgrade/notebooks/colab/02_pdt_calibration.ipynb)",
        "",
        "Rescues the degenerate v3 PDT result. Writes `results_v2/pdt/PDT_SECTION.md` for v4 §5.4."),
    _code("from google.colab import drive",
          "drive.mount('/content/drive')",
          "%cd /content",
          "!test -d agri-drone || git clone https://github.com/Ashut0sh-mishra/agri-drone.git",
          "%cd /content/agri-drone",
          "!git fetch --all && git checkout research-upgrade && git pull",
          "!pip install -q -r requirements.txt scikit-learn matplotlib"),
    _code("import torch",
          "assert torch.cuda.is_available(), 'Need GPU runtime'",
          "DRIVE = '/content/drive/MyDrive/agri-drone'",
          "WEIGHTS = f'{DRIVE}/models/wheat_4cls.pt'",
          "PDT_DIR = f'{DRIVE}/data/PDT_datasets'",
          "OUT = f'{DRIVE}/results_v2/pdt'",
          "import os; os.makedirs(OUT, exist_ok=True)",
          "print('weights:', WEIGHTS); print('pdt:', PDT_DIR); print('out:', OUT)"),
    _md("## 1. Threshold sweep (ROC / PR)"),
    _code("import pathlib",
          "csv_path = pathlib.Path('evaluate/results/cross_dataset_PDT_predictions.csv')",
          "if not csv_path.exists():",
          "    print('Run evaluate/pdt_cross_eval.py first')",
          "!python evaluate/pdt_v2.py --variant threshold_sweep --predictions-csv $csv_path --out-dir $OUT"),
    _code("import json, matplotlib.pyplot as plt",
          "r = json.loads(open(f'{OUT}/threshold_sweep.json').read()); print(r)",
          "plt.figure(); plt.title('ROC / PR summary')",
          "plt.bar(['ROC-AUC','PR-AUC'], [r.get('roc_auc',0), r.get('pr_auc',0)])",
          "plt.ylim(0,1); plt.savefig(f'{OUT}/roc_pr_summary.png'); plt.show()"),
    _md("## 2. Few-shot fine-tune (5 / 10 / 25 / 50 per class)"),
    _code("for k in [5, 10, 25, 50]:",
          "    !python evaluate/pdt_v2.py --variant few_shot --shots $k --out-dir $OUT"),
    _md("## 3. Temperature + Platt scaling"),
    _code("!python evaluate/pdt_v2.py --variant calibration --out-dir $OUT"),
    _md("## 4. Reliability diagram (placeholder)"),
    _code("import numpy as np, matplotlib.pyplot as plt",
          "plt.figure(); plt.plot([0,1],[0,1],'--',label='perfect')",
          "plt.xlabel('confidence'); plt.ylabel('accuracy'); plt.legend()",
          "plt.title('Reliability (placeholder)'); plt.savefig(f'{OUT}/reliability.png'); plt.show()"),
    _md("## 5. Write PDT_SECTION.md"),
    _code("import json, pathlib",
          "ts = json.loads(open(f'{OUT}/threshold_sweep.json').read())",
          "md = f'''# §5.4 PDT cross-dataset evaluation (rescued)",
          "",
          "- ROC-AUC: {ts.get(\"roc_auc\",\"[TBD]\")}",
          "- PR-AUC: {ts.get(\"pr_auc\",\"[TBD]\")}",
          "- Specificity @ 90% sensitivity: {ts.get(\"specificity_at_90_sensitivity\",\"[TBD]\")}",
          "- Sensitivity @ 90% specificity: {ts.get(\"sensitivity_at_90_specificity\",\"[TBD]\")}",
          "",
          "Few-shot and calibration results: see `results_v2/pdt/*.json`.",
          "'''",
          "pathlib.Path(f'{OUT}/PDT_SECTION.md').write_text(md); print(md)"),
]
write(OUT / "02_pdt_calibration.ipynb", nb2, "02_pdt_calibration")


# ------------------------------------------------------------------ NB 3
nb3 = [
    _md("# agri-drone — Notebook 3: fair multi-backbone baseline re-audit",
        "",
        "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
        "(https://colab.research.google.com/github/Ashut0sh-mishra/agri-drone/"
        "blob/research-upgrade/notebooks/colab/03_baseline_reaudit.ipynb)",
        "",
        "Trains EfficientNet-B0, ConvNeXt-Tiny, MobileNetV3-Small under the shared recipe in `docs/training_recipe.md@v1`.",
        "Weights save to `MyDrive/agri-drone/models_v2/` (NEVER committed)."),
    _code("from google.colab import drive",
          "drive.mount('/content/drive')",
          "%cd /content",
          "!test -d agri-drone || git clone https://github.com/Ashut0sh-mishra/agri-drone.git",
          "%cd /content/agri-drone",
          "!git fetch --all && git checkout research-upgrade && git pull",
          "!pip install -q -r requirements.txt timm"),
    _code("import torch",
          "assert torch.cuda.is_available(), 'Need GPU runtime'",
          "DRIVE = '/content/drive/MyDrive/agri-drone'",
          "DATA  = f'{DRIVE}/data/plantvillage'",
          "OUT_MODELS = f'{DRIVE}/models_v2'",
          "OUT_RESULTS = f'{DRIVE}/results_v2/baselines'",
          "import os; os.makedirs(OUT_MODELS, exist_ok=True); os.makedirs(OUT_RESULTS, exist_ok=True)",
          "print('data:', DATA)"),
    _md("## Shared training recipe (`docs/training_recipe.md@v1`)"),
    _code("RECIPE = dict(",
          "    optimizer='AdamW', lr=1.25e-3, weight_decay=0.05,",
          "    scheduler='cosine', warmup_epochs=3, epochs=50,",
          "    batch_size=32, label_smoothing=0.1,",
          "    augmentation='standard_v1', precision='fp16', seed=42,",
          ")",
          "print(RECIPE)"),
    _md("## Train EfficientNet-B0"),
    _code("!python evaluate/matrix/audit_baseline.py --backbone efficientnet_b0 --out-dir $OUT_RESULTS"),
    _md("## Train ConvNeXt-Tiny"),
    _code("!python evaluate/matrix/audit_baseline.py --backbone convnext_tiny --out-dir $OUT_RESULTS"),
    _md("## Train MobileNetV3-Small"),
    _code("!python evaluate/matrix/audit_baseline.py --backbone mobilenetv3_small --out-dir $OUT_RESULTS"),
    _md("## Assemble BASELINES_TABLE.md"),
    _code("import json, pathlib, glob",
          "rows = []",
          "for f in sorted(glob.glob(f'{OUT_RESULTS}/**/audit.json', recursive=True)):",
          "    d = json.loads(open(f).read())",
          "    rows.append((d.get('backbone','?'), d.get('accuracy','[TBD]'), d.get('macro_f1','[TBD]'), d.get('params_m','[TBD]'), d.get('latency_ms','[TBD]')))",
          "md = ['# \\u00a76.7 Fair multi-backbone baselines', '',",
          "      '| Backbone | Accuracy | Macro-F1 | Params (M) | Latency (ms) |',",
          "      '|---|---:|---:|---:|---:|']",
          "for r in rows:",
          "    md.append('| ' + ' | '.join(str(x) for x in r) + ' |')",
          "md.append('')",
          "md.append('All trained under docs/training_recipe.md@v1.')",
          "pathlib.Path(f'{OUT_RESULTS}/BASELINES_TABLE.md').write_text('\\n'.join(md))",
          "print('\\n'.join(md))"),
]
write(OUT / "03_baseline_reaudit.ipynb", nb3, "03_baseline_reaudit")

print("wrote:")
for p in sorted(OUT.glob("*.ipynb")):
    print(" ", p.relative_to(HERE), p.stat().st_size, "bytes")
