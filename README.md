# Physics-Informed Deep Learning for Wave-Source Localization

This project develops a **physics-informed deep learning framework** to localize wave sources from simulated 2D wave fields.
Models are trained on HDF5 datasets with two regimes (T250 and T500), and we provide tools for inference, visualization, and activation maximization.

---

## ⚡ Quickstart

```powershell
# 1) Create and activate a virtual environment (Windows)
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2) Install dependencies
pip install -r requirements.txt
```

### Inference Demo (dual: T250 + T500)
Requirements:
- Models (place at project root):
  - `models\best_T250.pth`
  - `models\best_T500.pth`
- Datasets (under `data\`):
  - Preferred: `data\wave_dataset_T250_validation.h5`, `data\wave_dataset_T500_validation.h5`
  - Fallbacks: `data\T250\analysis.h5`, `data\T500\analysis.h5`

Run:
```powershell
.\venv\Scripts\python.exe scripts\demo\run_dual_inference_demo.py
```
What it does:
- Loads the T250 and T500 models and the corresponding datasets from `data/`
- Uses the correct TRAINING normalization stats per tag (T250, T500)
- Picks 3 random samples per tag and produces a single figure (3×2 grid)
- Logs the exact model and dataset used and per-sample errors

Where things are resolved from:
- Models: `models\best_T250.pth`, `models\best_T500.pth`
- Datasets (priority): `data/wave_dataset_<TAG>_validation.h5` → `data/<TAG>/analysis.h5` → any `data/**/*<tag>*.h5`

Note:
- The demo assumes raw wave fields. Normalization happens inside the inference pipeline using training stats.
- If ground-truth coordinates are stored normalized ([0,1]) in the dataset, the demo rescales them to pixel units before computing errors/plotting.

### Activation Maximization (AM) Demo
Requirements:
- Model checkpoint: `models\best_<TAG>.pth`
- Dataset: `data\wave_dataset_<TAG>_validation.h5` (preferred) or `data\<TAG>\analysis.h5`

Run (example for T250):
```powershell
.\venv\Scripts\python.exe scripts\demo\run_am_demo.py `
  --model_path models\best_T250.pth `
  --dataset_path data\wave_dataset_T250_validation.h5 `
  --layer_mode last_n --last_n 3 --top_k 3 `
  --activation_mode post_relu_mean --iterations 1000 --lr 0.005 `
  --num_samples 2
```
What it does:
- Infers dataset tag from `--dataset_path` and forces correct TRAINING normalization
- Randomly selects `--num_samples` samples (no seed)
- For each sample, ranks filters using the SAME normalized sample and starts optimization from this SAME sample for all selected layers/filters
- Saves outputs under: `outputs\am_demo\<model_stem>\sample_<idx>\`

---

## Repo Structure (high-level)
- `src/` core code (models, inference, utils)
- `scripts/demo/` runnable demos (inference, activation maximization)
- `data/` datasets (HDF5)
- `models/` model checkpoints (not tracked by git)
- `experiments/` experiment outputs and analysis

---

## Links
- Full technical report: see `docs/`
- AM and visualization examples: see `scripts/activation_maximization/` and `scripts/visualization/`
