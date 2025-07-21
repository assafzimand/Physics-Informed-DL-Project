# Google Colab Infrastructure for Wave Source Localization

## 🚀 Quick Start

1. **Setup Environment** (run once per session):
   ```python
   # In Colab, run each cell:
   !git clone https://github.com/your-username/Physics-Informed-DL-Project.git
   %cd Physics-Informed-DL-Project
   !python colab/setup/colab_setup.ipynb
   ```

2. **Run Grid Search Phase 1** (2×2×2 = 8 experiments):
   ```python
   !python colab/notebooks/run_optimization.py
   ```

3. **Download Results** (after experiments):
   ```python
   !python colab/mlflow/download_results.py
   ```

---

## 📁 Directory Structure

```
colab/
├── setup/
│   └── colab_setup.ipynb          # Initial environment setup
├── data/
│   └── upload_dataset.py          # Upload local dataset to Drive
├── experiments/
│   └── experiment_configs/
│       └── resnet_optimization.yaml  # Grid search configuration
├── notebooks/
│   ├── test_optimization.py       # Quick test (5 epochs)
│   └── run_optimization.py        # Full grid search (8 experiments)
├── mlflow/
│   └── download_results.py        # Download results to local
└── README.md                      # This file
```

---

## 🔬 Grid Search Phase 1

**Configuration:** 2×2×2 Grid Search
- **Learning Rate:** [0.001, 0.0001]
- **Batch Size:** [16, 32] 
- **Optimizer:** [Adam, AdamW]
- **Fixed:** 50 epochs, weight_decay=0.01, cosine scheduler

**Expected Results:**
- 8 experiments × 50 epochs = ~6 hours on L4 GPU
- Best model will be identified for Phase 2 (5-fold CV)

**Success Criteria:**
- Target: < 3.0px distance error
- Baseline to beat: 2.57px (from single model)

---

## 📊 Experiment Flow

### Phase 1: Grid Search (Current)
```
8 experiments → Best hyperparameters identified
```

### Phase 2: Rigorous Evaluation (Next)
```
Best config → 5-fold cross validation → Mean ± Std results
```

### Phase 3: Failure Analysis (Future)
```
Best model → Analyze hardest cases → Understand limitations
```

---

## 💾 Data Management

- **Dataset:** Auto-uploaded to Google Drive (wave_dataset_T500.h5)
- **Results:** MLflow tracking + auto-sync to Drive after every 2 experiments
- **Models:** Best checkpoints saved automatically
- **Plots:** Learning curves and analysis plots generated

---

## 🔧 GPU Recommendations

- **L4:** Best balance of performance/cost for this project
- **A100:** Fastest but overkill for 50-epoch experiments  
- **T4:** Adequate but slower

---

## 📈 Next Steps After Grid Search

1. **Identify best hyperparameters** from 8 experiments
2. **Run 5-fold cross validation** on winning combination
3. **Report scientific results:** Distance Error: X.X ± Y.Y px
4. **Failure analysis:** Which source positions are hardest?

---

## 🎯 Academic Requirements Addressed

- ✅ **B.** Grid search + planned 5-fold CV for statistical rigor
- ✅ **C.** Will report mean ± std (not single numbers)  
- ✅ **F.** Failure analysis planned for best model
- 🔄 **K.** Mathematical foundation analysis (future work)

Ready to run in Colab! 🚀 