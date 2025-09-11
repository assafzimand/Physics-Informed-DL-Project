"""
Activation Maximization for WaveSourceMiniResNet
================================================

Tools for visualizing what input patterns maximally activate specific filters
in our wave source localization model using gradient-based optimization.

Main components:
- Model wrappers for Lucent compatibility
- Filter visualization utilities  
- Layer hook management
- Experiment runners

Based on Erhan et al. (2009) "Visualizing Higher-Layer Features of a Deep Network"
and the Lucent library (PyTorch port of Google's Lucid).
"""

__version__ = "1.0.0" 