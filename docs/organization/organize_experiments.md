# Physics-Informed DL Project - Experiment Organization Plan

## 📊 Experiment Phase Classification

### **Phase 0: Local Development & Testing**
- **Location**: `mlruns/local_development/`
- **Results**: `results/local_development/`
- **Content**: 
  - Single training tests
  - Partial grid search attempts
  - Pipeline debugging
- **MLflow IDs**: `682160152085375302`, `839193960704093634`, `0`

### **Phase 1: Grid Search Optimization (Colab)**
- **Location**: `mlruns/grid_search_phase1/`  
- **Results**: `results/grid_search_phase1/`
- **Content**:
  - 8-experiment grid search (2x2x2)
  - 50 epochs per experiment
  - Winner: lr=0.001, bs=32, adam → 2.37px
- **MLflow ID**: `873542925861803181`
- **Status**: ✅ COMPLETED & ANALYZED

### **Phase 2a: CV Pipeline Testing (Colab)**
- **Location**: `mlruns/cv_test_phase/`
- **Results**: `results/cv_test_phase/`
- **Content**:
  - 5-fold CV with 10 epochs per fold
  - Pipeline verification
  - Auto-save testing
- **MLflow ID**: `245362020072499018` (+ test_cv experiments)
- **Status**: ✅ COMPLETED

### **Phase 2b: Full CV Training (Colab)**
- **Location**: `mlruns/cv_phase2_full/`
- **Results**: `results/cv_phase2_full/`
- **Content**:
  - 5-fold CV with 75 epochs per fold
  - Winning hyperparameters
  - Final results: 2.08 ± 0.34 px
- **MLflow ID**: [NEW - from Drive upload]
- **Status**: ✅ JUST COMPLETED - NEEDS ORGANIZATION

## 🗂️ Target Directory Structure

```
mlruns/
├── local_development/           # Phase 0
├── grid_search_phase1/          # Phase 1  
├── cv_test_phase/              # Phase 2a
└── cv_phase2_full/             # Phase 2b (NEW)

results/
├── local_development/           # Phase 0 results
├── grid_search_phase1/          # Phase 1 results ✅
│   ├── analysis/               # Grid search analysis ✅
│   ├── plots/                  # Grid search plots ✅
│   └── summary.csv             # Results table ✅
├── cv_test_phase/              # Phase 2a results  
│   ├── models/                 # 5 test models (33MB each)
│   ├── metrics/                # CV test metrics
│   └── logs/                   # Test logs
└── cv_phase2_full/             # Phase 2b results (NEW)
    ├── models/                 # 5 final models 
    ├── analysis/               # CV statistical analysis
    ├── plots/                  # CV training curves & box plots
    ├── metrics/                # All fold metrics
    └── academic_report/        # Publication-ready results
``` 