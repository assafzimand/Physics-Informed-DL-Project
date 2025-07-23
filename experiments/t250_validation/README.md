# T=250 Dataset Hyperparameter Validation

**Experiment Date**: July 23, 2025  
**Experiment Type**: Single-fold validation training  
**Duration**: ~2 hours  
**Objective**: Validate winning hyperparameters on T=250 dataset before committing to full 5-fold CV training

## 📊 Quick Results

- **Final Distance Error**: Check `validation_summary_*.json` for exact value
- **Training Time**: ~2 hours (50 epochs)
- **Recommendation**: Check `VALIDATION_REPORT_*.md` for go/no-go decision

## 🎯 Experiment Design

### Dataset
- **Source**: `wave_dataset_T250.h5`
- **Splits**: 80% train, 20% validation, 0% test
- **Size**: ~0.1 GB

### Hyperparameters (from T=500 grid search winners)
- **Learning Rate**: 0.001
- **Batch Size**: 32
- **Optimizer**: Adam
- **Weight Decay**: 0.01
- **Epochs**: 50 (validation run)
- **Early Stopping**: 15 epochs patience

### Model Configuration
- **Architecture**: WaveSourceMiniResNet
- **Grid Size**: 128x128
- **Device**: CUDA

## 📁 Files in this Experiment

### Main Results
- `validation_summary_*.json` - Complete experiment data and metrics
- `training_curves_*.png` - Training and validation loss curves
- `VALIDATION_REPORT_*.md` - Human-readable results and recommendation

### Analysis Directory
- `analysis/` - Future analysis scripts and additional plots

## 🔄 Comparison Baseline

This validation compares against T=500 dataset results:
- **T=500 Grid Search Best**: 2.37 px
- **T=500 CV Average**: 2.078 px
- **T=250 Validation**: See results files

## 🚀 Next Steps

Based on the validation results:
- ✅ **If RECOMMENDED**: Proceed with full 5-fold CV training on T=250
- ⚠️ **If CAUTION**: Consider hyperparameter adjustments
- ❌ **If NOT RECOMMENDED**: Investigate T=250 dataset or tune hyperparameters

## 🔗 Related Experiments

- `../cv_full/` - T=500 5-fold CV results (baseline)
- `../grid_search_phase1/` - Original hyperparameter search on T=500
- Future: `../t250_cv_full/` - If validation passes, full T=250 CV training

---

**Purpose**: Smart validation to avoid wasting 10 hours on full CV if hyperparameters don't work well on T=250 dataset. 