"""
Custom Activation Maximization for Single-Channel Wave Models

This package provides a clean, simple implementation of activation maximization
specifically designed for single-channel wave field inputs, bypassing the
RGB assumptions of libraries like Lucent.
"""

from .simple_activation_max import SimpleActivationMaximizer, run_simple_activation_maximization

__all__ = ['SimpleActivationMaximizer', 'run_simple_activation_maximization'] 