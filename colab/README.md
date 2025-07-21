# Google Colab Infrastructure for Wave Source Localization

This directory contains all scripts and notebooks needed to run experiments on Google Colab efficiently.

## 📁 Directory Structure

```
colab/
├── README.md                 # This file
├── setup/
│   ├── colab_setup.ipynb    # Main setup notebook (START HERE)
│   ├── install_deps.py      # Dependency installation script
│   └── sync_codebase.py     # GitHub sync utilities
├── data/
│   ├── upload_dataset.py    # Upload local dataset to Drive
│   ├── download_dataset.py  # Download dataset in Colab
│   └── verify_data.py       # Data integrity verification
├── experiments/
│   ├── experiment_configs/  # YAML configs for experiments
│   ├── run_experiments.py   # Automated experiment runner
│   └── queue_manager.py     # Experiment queue management
├── mlflow/
│   ├── setup_mlflow.py      # MLflow configuration for Colab
│   ├── download_results.py  # Auto-download after each session
│   └── sync_models.py       # Model synchronization utilities
└── notebooks/
    ├── quick_training.ipynb # Single experiment notebook
    └── batch_training.ipynb# Multi-experiment notebook
```

## 🚀 Quick Start Guide

### 1. **First Time Setup (5 minutes)**
1. Open `setup/colab_setup.ipynb` in Google Colab
2. Run all cells to set up environment
3. Upload dataset using `data/upload_dataset.py` (one-time)

### 2. **Running Experiments**
1. Open `notebooks/batch_training.ipynb`
2. Configure your experiments in the notebook
3. Hit "Run All" - experiments will queue automatically
4. Results auto-download at end of session

### 3. **Getting Results**
- MLflow results: Auto-downloaded to `colab_results/mlruns/`
- Best models: Auto-downloaded to `colab_results/models/`
- Visualizations: Saved in `colab_results/plots/`

## 🎯 Experiment Strategy

**Target: 5-10 focused experiments to optimize ResNet**

**Phase 1**: Learning rate & optimizer optimization (2-3 experiments)
**Phase 2**: Architecture improvements (2-3 experiments)  
**Phase 3**: Training strategy refinement (2-3 experiments)

## 🔧 Key Features

- ✅ One-click environment setup
- ✅ Automatic GitHub code sync
- ✅ Persistent dataset storage on Drive
- ✅ Auto-download results after each session
- ✅ Experiment queuing with checkpoints
- ✅ Progress monitoring and visualization

## 📞 Support

If something breaks:
1. Check the error logs in the notebook
2. Restart runtime and re-run setup
3. Verify Drive mount and file permissions 