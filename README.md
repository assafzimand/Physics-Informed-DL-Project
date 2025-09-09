# Physics-Informed Deep Learning for Wave-Source Localization

This project develops a **physics-informed deep learning framework** to localize wave sources from simulated 2D wave fields.  
We train **WaveSourceMiniResNet** models on normalized HDF5 datasets, with two regimes (T250 and T500), and provide analysis tools for evaluation, feature extraction, and activation maximization.

---

![Placeholder](docs/figures/sample_wave_prediction.png)  
*Sample input wave field (128×128) with predicted vs. ground-truth source overlay.*

---

## 🔗 Links
- [Project organization](docs/organization.md)  
- [Full technical report (PDF)](docs/report.pdf)  
- [GitHub repository](https://github.com/assafzimand/Physics-Informed-DL-Project)

---

## 🚀 What We Built

- **Data generation and format**  
  - HDF5 files:  
    - `wave_fields`: (N, 128, 128) arrays  
    - `source_coords`: (N, 2) arrays  
    - Training HDF5 stores `wave_mean`, `wave_std` for normalization.
- **Model**  
  - [`WaveSourceMiniResNet`](src/models/wave_source_miniresnet.py): stacked Conv2D blocks with ReLU activations.
- **Training path**  
  - Local PoC → Colab hyperparameter sweeps → 5-fold CV (`cv_full`, best T500 model) → T250 pipeline and validation → final failure analysis.
- **Understanding the model**  
  - Feature extraction analysis: insight into intermediate representations (selected examples in the report).  
  - Activation Maximization: one unified pipeline. We explored objective variants (pre‑ReLU mean_abs, post_relu_mean), optional suppression of other filters in the layer, and TV/L2 input regularizers to select a final setting. Final choice and rationale are documented in the report.
- **Results**  
  - Best-model metrics (T500 cv_full).  
  - Failure analysis: grid of points colored by error percentile, with per-group mean error.  
  - [Placeholder figures available under `experiments/plots/*.png`].

---

## ⚡ Quickstart

```bash
# 1. Setup environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Run single inference (placeholder command)
# NOTE: Dataset–model tag must match (T250↔T250, T500↔T500). The code will error on mismatch.
python scripts/run_inference.py --model checkpoints/t500_cv_full.pth --data data/sample_t500.h5

# 3. Run one AM example (post-ReLU mean, no regularization)
python scripts/activation_maximization/simple_test.py \
  --iterations 1000 --learning_rate 0.001 \
  --activation_mode post_relu_mean --tv_reg 0.0 --l2_reg 0.0
```

---

## 📈 Final Results (high level)

- Best T500 cv_full model: metrics and example predictions (see report for full tables).
- Failure analysis: worst/best percentile groups with mean error per group.

Figures (placeholders; replace with your generated plots):

![Failure Analysis](docs/figures/failure_analysis_grid.png)
*Validation predictions colored by error percentile; legend shows mean error per group.*

![Multi-sample Predictions](docs/figures/multi_sample_predictions.png)
*Selected validation samples with predicted vs ground-truth source overlays.*
