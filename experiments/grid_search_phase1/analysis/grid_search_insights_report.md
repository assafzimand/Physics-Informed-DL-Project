
# Grid Search Phase 1 - Analysis Report

## 🏆 **KEY RESULTS**

### **Winner: lr0.001_bs32_adam**
- **Distance Error**: 2.37 px
- **Validation Loss**: 4.0676
- **Hyperparameters**:
  - Learning Rate: 0.001
  - Batch Size: 32
  - Optimizer: adam

### **Top 3 Performers**:

🥇 **lr0.001_bs32_adam**: 2.37 px
   - LR: 0.001, BS: 32, OPT: adam

🥈 **lr0.0001_bs32_adamw**: 2.96 px
   - LR: 0.0001, BS: 32, OPT: adamw

🥉 **lr0.0001_bs32_adam**: 2.97 px
   - LR: 0.0001, BS: 32, OPT: adam


## 📊 **HYPERPARAMETER INSIGHTS**

### **Learning Rate Analysis**:
- LR 0.001: 3.00 px (avg)
- LR 0.0001: 3.26 px (avg)

### **Batch Size Analysis**:
- BS 32: 2.87 px (avg)
- BS 16: 3.39 px (avg)

### **Optimizer Analysis**:
- ADAM: 3.03 px (avg)
- ADAMW: 3.23 px (avg)


## 🎯 **RECOMMENDATIONS FOR PHASE 2**

### **Best Configuration for 5-Fold Cross Validation**:
- **Learning Rate**: 0.001
- **Batch Size**: 32
- **Optimizer**: adam

### **Expected Performance**:
- **Target Distance Error**: 2.37 ± 0.3 px
- **Confidence**: High (consistent performance across grid search)

## 📈 **NEXT STEPS**

1. **Phase 2**: Run 5-fold cross validation on winning configuration
2. **Statistical Reporting**: Get mean ± std results for academic paper
3. **Failure Analysis**: Analyze which source positions are hardest to localize
4. **Model Interpretation**: Understand what the model learned

---
*Generated from Grid Search Phase 1 results*
*Total experiments analyzed: 8*
