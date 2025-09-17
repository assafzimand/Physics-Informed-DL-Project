# Physics-Informed Deep Learning for Wave-Source Localization

This project develops a **physics-informed deep learning framework** to localize wave sources from simulated 2D wave fields.  
Models are trained on HDF5 datasets with two regimes (T250 and T500), and tools are provided for inference, visualization, and activation maximization.

---

## ⚡ Quickstart

```powershell
# 0) Clone the repository and enter the folder
git clone https://github.com/assafzimand/Physics-Informed-DL-Project.git
cd Physics-Informed-DL-Project

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


Run:
```powershell
.\venv\Scripts\python.exe scripts\demo\run_dual_inference_demo.py
```
What it does:
- Loads the T250 and T500 models and the corresponding datasets from `data/`
- Uses the correct TRAINING normalization stats per tag (T250, T500)
- Picks 3 random samples per tag and produces a single figure (3×2 grid)
- Logs the exact model and dataset used and per-sample errors


### Activation Maximization (AM) Demo
Requirements:
- Model checkpoint: `models\best_<TAG>.pth`
- Dataset: `data\wave_dataset_<TAG>_validation.h5`

Run (example for T500):
```powershell
.\venv\Scripts\python.exe scripts\demo\run_am_demo.py `
  --model_path models\best_T500.pth `
  --dataset_path data\wave_dataset_T500_validation.h5 `
  --layer_mode last_n --last_n 3 --top_k 3 `
  --activation_mode post_relu_mean --iterations 1000 --lr 0.005 `
  --num_samples 2
```
What it does:
- Infers dataset tag from `--dataset_path` and forces correct TRAINING normalization
- Randomly selects `--num_samples` samples (no seed)
- For each sample, ranks filters using the same normalized sample and starts optimization from this same sample for all selected layers/filters
- Saves outputs under: `outputs\am_demo\<model_stem>\sample_<idx>\`

---

## Repo Structure (high-level)
- `src/`: Core Python package (imported by demos/runners)
  - `models/`: Model definitions, e.g., `WaveSourceMiniResNet` and `create_wave_source_model`
  - `inference/`: Inference pipeline (`WaveSourceInference`) with training-based normalization
  - `activation_maximization/`: AM engine (`SimpleActivationMaximizer`) and helpers
  - `common/`: Shared utilities (paths, normalization policy)
  - `data/`: Data loaders/generators (HDF5 reading, dataset utilities)
  - `training/`: Trainers and CV utilities (historical, retained for reference)
- `scripts/`: Runnable utilities and analyses (CLI-style)
  - `demo/`: Demos you can run out-of-the-box (dual inference, AM demo)
  - `improving_am/`: AM grid search runner (`run_grid.py`) reading from `configs/`
  - `activation_maximization/`: Comprehensive/simple AM exploration scripts
  - `data_management/`: Dataset generation, exploration, and validation helpers
  - `feature_analysis/`: Feature extraction/ranking and visualization utilities
  - `filter_visualization/`: Visualize learned conv filters/weights
  - `failure_analysis/`, `validation/`, `visualization/`, `model_testing/`, `training/`, `utils/`: Misc. analyses, testing, exports
- `data/`: Datasets (HDF5)
  - Validation sets used by demos: `wave_dataset_T250_validation.h5`, `wave_dataset_T500_validation.h5`
  - Analysis subsets: `wave_dataset_*_analysis_20samples.h5`
- `models/`: Model checkpoints
  - Committed allowlisted demo models: `best_T250.pth`, `best_T500.pth`
  - All other large checkpoints remain git-ignored
- `experiments/`: Results and logs of runs
  - Training/CV (`cv_full/`, `cv_test/`), validations, feature analysis, activation maximization (`improving_am/`, `activation_maximization/`)
  - Contains per-run CSV/JSON summaries and generated plots/images
- `configs/`: Configuration files for runners
  - `improving_am/baseline.yaml` (grid), training and simulation configs
- `docs/`: Documentation and assets
  - Report images, ONNX/torchviz exports, organization notes
- `colab/`: Colab notebooks and setup helpers
  - Reproducible runners for CV/grid, environment setup
- `archive/`: Archived legacy code/experiments (kept for reference)
  - `src_archive/`, `scripts_archive/`, `experiments_archive/`

---

## References
- Full technical report: [Project Report (PDF)](docs/Project%20Report.pdf)
- For full experiment results, contact: [assafzimand@gmail.com](mailto:assafzimand@gmail.com)