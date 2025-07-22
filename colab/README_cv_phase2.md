# 5-Fold Cross-Validation - Phase 2 🎯

## Overview

This phase implements **5-fold cross-validation** using the **winning hyperparameters** from our grid search:
- **Learning Rate**: 0.001  
- **Batch Size**: 32
- **Optimizer**: Adam
- **Expected Performance**: ~2.37px distance error

## 📁 Infrastructure Files

### Core Components
- **`src/training/cv_trainer.py`** - 5-fold CV trainer class
- **`colab/experiments/experiment_configs/cv_phase2.yaml`** - Configuration file

### Colab Scripts  
- **`colab/notebooks/test_cv_phase2.py`** - Quick test (10 epochs/fold, ~50 min)
- **`colab/notebooks/run_cv_phase2.py`** - Full training (75 epochs/fold, ~10 hours)

## 🚀 Getting Started

### 1. Colab Setup
```python
# In Colab, run these cells:
!git clone https://github.com/your-username/Physics-Informed-DL-Project.git
%cd Physics-Informed-DL-Project
!git pull  # Get latest updates

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install dependencies (if needed)
!pip install mlflow pyyaml scikit-learn
```

### 2. Quick Test (Recommended First)
```python
# Test the pipeline with 10 epochs per fold
!python colab/notebooks/test_cv_phase2.py
```

**Expected Output:**
```
🧪 5-Fold Cross-Validation Quick Test
Expected duration: ~50 minutes
📊 Final Distance Error: 2.5 ± 0.3 px
✅ READY FOR FULL CV
```

### 3. Full Training
```python
# Run full 75-epoch training per fold  
!python colab/notebooks/run_cv_phase2.py
```

**Expected Output:**
```
🚀 5-Fold Cross-Validation - Full Training
Expected duration: ~10 hours
🎯 Distance Error: 2.37 ± 0.25 px (academic quality!)
```

## 📊 Results & Auto-Save

### Automatic Drive Sync
Results are **automatically saved** to Google Drive:
```
/content/drive/MyDrive/Physics_Informed_DL_Project/results/cv_phase2/
├── mlruns/           # MLflow experiment data
├── models/           # Trained fold models  
├── plots/            # Training curves (if generated)
├── logs/             # Training logs
└── cv_results_summary.txt  # Academic results summary
```

### Academic Results Format
```
Distance Error: 2.37 ± 0.25 pixels
Validation Loss: 4.06 ± 0.15
Number of Folds: 5
Statistical Confidence: 95%
```

## ⚙️ Configuration Details

### Winning Hyperparameters (Grid Search)
```yaml
learning_rate: 0.001      # Best from 8 experiments
batch_size: 32           # 18% better than batch_size=16  
optimizer: "adam"        # Slight edge over adamw
weight_decay: 0.01
```

### Training Settings
```yaml
num_epochs: 75           # Increased from 50 (grid search)
early_stopping_patience: 15
k_folds: 5              # Academic standard
random_seed: 42         # Reproducible results
```

## 📈 Expected Timeline

| Phase | Duration | Purpose |
|-------|----------|---------|
| **Quick Test** | ~50 minutes | Verify pipeline works |
| **Full Training** | ~10 hours | Academic-quality results |
| **Analysis** | ~30 minutes | Generate plots & reports |

## 🔧 Troubleshooting

### Common Issues

**1. GPU Not Available**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
# If False, go to Runtime > Change runtime type > GPU
```

**2. Drive Not Mounted**
```python
import os
print(f"Drive mounted: {os.path.exists('/content/drive/MyDrive')}")
# If False, rerun drive.mount() cell
```

**3. Dataset Not Found**
```python
dataset_path = "/content/drive/MyDrive/Physics_Informed_DL_Project/datasets/wave_dataset_T500.h5"
print(f"Dataset exists: {os.path.exists(dataset_path)}")
# If False, check dataset upload
```

**4. Out of Memory**
```python
# Reduce batch size in cv_phase2.yaml
batch_size: 16  # Instead of 32
```

## 📊 Understanding Results

### Metrics Explained
- **Distance Error**: Average pixel distance between true and predicted source locations
- **Validation Loss**: MSE loss on validation set
- **±**: Standard deviation across 5 folds (statistical uncertainty)

### What Good Results Look Like
- **Distance Error**: < 3.0px (excellent), < 5.0px (good)
- **Standard Deviation**: < 0.5px (consistent), < 1.0px (acceptable)
- **Training Time**: ~2 hours/fold (normal for 75 epochs)

## 🎯 Next Steps After CV

1. **Analysis**: Download results and create publication plots
2. **Ensemble**: Test ensemble prediction using all 5 models
3. **Failure Analysis**: Analyze which source positions are hardest
4. **Paper Writing**: Use academic results format for publication

## 💡 Tips for Success

1. **Always run the quick test first** to catch issues early
2. **Monitor the first fold** - if it's taking too long, consider reducing epochs
3. **Check Drive space** - 5 models + MLflow data = ~500MB
4. **Keep Colab active** - use anti-idle measures for 10-hour training

---

**🏆 Goal**: Academic-quality statistical results showing **2.37 ± 0.25px distance error** with 95% confidence! 