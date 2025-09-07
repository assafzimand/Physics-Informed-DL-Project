# Chan-Vese Algorithm for Multiple Sclerosis Lesion Refinement

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A Python implementation of the Chan-Vese active contour algorithm for refining Multiple Sclerosis (MS) lesion segmentation in brain MRI images.

## 📖 About This Project

This project implements the **Chan-Vese curve evolution algorithm** from:

> **"Multiple Sclerosis Lesion Detection Using Constrained GMM and Curve Evolution"**  
> *Freifeld et al., International Journal of Biomedical Imaging, 2009*

The Chan-Vese algorithm is one component of the multi-step lesion detection pipeline described in the paper. This implementation focuses specifically on the **boundary refinement phase** that uses level set methods to improve initial lesion segmentations.

### 🔬 Algorithm Overview

The Chan-Vese algorithm uses level set methods to evolve an initial contour toward optimal lesion boundaries. The algorithm minimizes an energy functional that balances data fidelity with boundary smoothness.

For detailed mathematical background of the Chan-Vese algorithm, level set methods, and energy functionals, see:
📄 `documents/Mathematical Background.pdf`

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- 4GB+ RAM (for processing 3D MRI volumes)
- Git

### Installation & Setup
```bash
# Clone the repository
git clone https://github.com/assafzimand/Final-Project-Chan-Vese.git
cd Final-Project-Chan-Vese

# Run automated setup
python setup.py
```

### 🎯 Quick Demo

```bash
python demo.py
```

This will demonstrate the Chan-Vese algorithm with clear visualizations and easy-to-access results showing boundary refinement on MS lesion data.

**⚠️ Data Requirements**: The project uses BrainWeb simulated brain database. Registration may be required for data access.

## 📊 Validation Process

### Overview
The validation methodology creates **degraded lesion segmentations** that simulate the overestimated initial segmentations described in Freifeld et al. (2009). In their pipeline, the CGMM algorithm produces lesion segmentations with "**overestimation in the size of the lesions**" which then serve as input to the Chan-Vese boundary refinement step.

### Degraded Data Generation

**Simulating CGMM-style Overestimated Input:**
1. **Start with Ground Truth**: Perfect lesion segmentation from BrainWeb
2. **Create Oversized, Unsmooth Lesions** (matching CGMM characteristics):
   - Apply morphological dilation to enlarge lesion boundaries 
   - Add random boundary perturbations to create jagged, unsmooth edges
   - Target different degradation levels: `mild`, `moderate`, `severe`
3. **Brain-Constrained Segmentation**: Ensure all lesions remain within brain tissue

This approach replicates the type of **overestimated lesion segmentations** that the CGMM algorithm produces in the original paper, where "the segmentation step of the CGMM model is done voxel wise and **does not take into account the smoothness of the lesion boundaries**" (Freifeld et al., 2009). This allows us to validate the Chan-Vese boundary refinement component in isolation.

**Degradation Levels:**
- **Mild**: Dice ~0.89, subtle boundary oversegmentation
- **Moderate**: Dice ~0.86, noticeable boundary roughness  
- **Severe**: Dice ~0.83, significant oversegmentation with rough boundaries

### Validation Experiments

**Multi-Parameter Evaluation:**
```bash
# Single validation run
python scripts/validate_chan_vese.py

# Grid search across parameter space
python scripts/grid_search_chan_vese.py
```

**Parameters Tested:**
- **λ (Lambda)**: Curvature weight  
- **dt**: Time step
- **ε (Epsilon)**: Delta function width
- **Max Iterations**: Evolution steps

**Final Chosen Parameters:**
```python
dt = 0.0001        # Time step for stability
lambda = 35.0      # Curvature weight for optimal smoothness balance
epsilon = 0.001    # Delta function width for sharp boundaries
max_iterations = 10
```

**Parameter Selection Notes:**
- **λ (Lambda)** is the main parameter controlling the trade-off between data fidelity and smoothness
- Higher λ values prioritize smoothness over data fidelity
- Lower λ values prioritize data fidelity over smoothness  
- Many λ values produced good results with different balances between these goals

### Performance Metrics

**Segmentation Accuracy:**
- **Dice Coefficient**: Overlap measure (higher = better)
- **Jaccard Index**: Union over intersection (higher = better)
- **Volume Change**: Relative volume difference

**Boundary Quality:**
- **Smoothness Metric**: Mean absolute curvature (lower = smoother)
- **Boundary Energy**: Curvature regularization term

**Multi-Case Validation:**
Good performance was shown across all degradation levels (`mild`, `moderate`, `severe`) using the same parameters.

## 🔬 Technical Details

### Algorithm Features
- **Multi-Channel Data**: Uses T1, T2, and PD MRI sequences for robust tissue characterization
- **Signed Distance Function**: Proper SDF initialization for stable evolution
- **Brain-Constrained Evolution**: Restricts segmentation to brain tissue only
- **Normalized Delta Function**: Prevents numerical instabilities with ε-regularization
- **Full Covariance Energy**: Uses complete covariance matrices for tissue modeling

### Data Structure
```
data/
├── brainweb/                 # Ground truth MRI data
│   ├── t1_clean.npy         # T1 MRI sequence
│   ├── t2_clean.npy         # T2 MRI sequence  
│   ├── pd_clean.npy         # PD MRI sequence
│   └── tissue_map.npy       # Ground truth segmentation
└── validation/
    └── degraded_segmentations/   # Simulated imperfect segmentations
        ├── tissue_map_degraded_mild.npy
        ├── tissue_map_degraded_moderate.npy
        └── tissue_map_degraded_severe.npy
```

### Visualization Scripts
```bash
# Explore MRI data and tissue maps
python scripts/data_scripts/explore_data.py
python scripts/data_scripts/explore_degraded_data.py

```

## 📊 Results & Performance

### Validation Results (Optimized Parameters: dt=0.0001, λ=35.0, ε=0.001)

**All Degradation Levels:**

| Degradation Level | Input Dice | Output Dice | Improvement | Input Jaccard | Output Jaccard | Improvement | Input Smoothness | Output Smoothness | Smoothness Improvement |
|-------------------|------------|-------------|-------------|---------------|----------------|-------------|------------------|-------------------|------------------------|
| **Mild**          | 0.8892     | 0.8963      | +0.0070     | 0.8005        | 0.8120         | +0.0115     | 0.7317           | 0.7220            | -0.0097 (smoother)     |
| **Moderate**      | 0.8561     | 0.8853      | +0.0293     | 0.7483        | 0.7943         | +0.0459     | 0.7209           | 0.7204            | -0.0004 (smoother)     |
| **Severe**        | 0.8349     | 0.8922      | +0.0573     | 0.7166        | 0.8053         | +0.0887     | 0.7252           | 0.7241            | -0.0011 (smoother)     |

**Key Findings:**
- Consistent improvement across all degradation levels using the same parameters
- Larger improvements on more severely degraded inputs
- Excellent final performance (Dice >0.89) achieved for all cases
- Strong Jaccard indices (>0.79) demonstrating robust overlap

**Mild Degradation Results:**
![Mild Results](results/validation/final_results_all_data_degredation/dt0.0001_lambda35_eps0.001_iter10_mild_20250804_164030/results_comparison.png)

**Moderate Degradation Results:**
![Moderate Results](results/validation/final_results_all_data_degredation/dt0.0001_lambda35_eps0.001_iter10_moderate_20250804_164101/results_comparison.png)

**Severe Degradation Results:**
![Severe Results](results/validation/final_results_all_data_degredation/dt0.0001_lambda35_eps0.001_iter10_severe_20250804_164141/results_comparison.png)

## 🔗 References

1. Freifeld, O., et al. "Multiple Sclerosis Lesion Detection Using Constrained GMM and Curve Evolution." *International Journal of Biomedical Imaging* 2009.
2. Chan, T.F., Vese, L.A. "Active contours without edges." *IEEE Transactions on Image Processing* 10.2 (2001): 266-277.
3. BrainWeb: Simulated Brain Database. McGill Centre for Integrative Neuroscience. https://brainweb.bic.mni.mcgill.ca/

---

*🧠 Advancing Multiple Sclerosis research through computational image analysis*

