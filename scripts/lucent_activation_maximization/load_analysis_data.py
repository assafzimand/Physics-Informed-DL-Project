#!/usr/bin/env python3
"""
Analysis Dataset Loader for Wave Pattern Comparison.

Loads the wave_dataset_analysis_20samples.h5 dataset for comparing
generated activation maximization patterns with real wave fields.
"""

import sys
import os
import torch
import h5py
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


def load_analysis_dataset(dataset_path: Optional[Path] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load the 20-sample analysis dataset.
    
    Args:
        dataset_path: Path to dataset file (default: data/wave_dataset_analysis_20samples.h5)
        
    Returns:
        Tuple of (wave_fields, source_positions)
    """
    # Default path
    if dataset_path is None:
        dataset_path = Path("data/wave_dataset_analysis_20samples.h5")
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Analysis dataset not found: {dataset_path}")
    
    print(f"📁 Loading analysis dataset: {dataset_path}")
    
    # Load HDF5 dataset
    with h5py.File(dataset_path, 'r') as f:
        # Check available keys
        print(f"📋 Dataset keys: {list(f.keys())}")
        
        # Load wave fields and source positions
        if 'wave_fields' in f:
            wave_fields = torch.tensor(f['wave_fields'][:], dtype=torch.float32)
        else:
            raise KeyError("'wave_fields' not found in dataset")
            
        if 'source_positions' in f:
            source_positions = torch.tensor(f['source_positions'][:], dtype=torch.float32)
        else:
            raise KeyError("'source_positions' not found in dataset")
    
    print(f"✅ Loaded {wave_fields.shape[0]} samples")
    print(f"   Wave fields shape: {wave_fields.shape}")
    print(f"   Source positions shape: {source_positions.shape}")
    
    return wave_fields, source_positions


def get_diverse_samples(wave_fields: torch.Tensor, 
                       source_positions: torch.Tensor,
                       num_samples: int = 5) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """
    Select diverse wave field samples for comparison.
    
    Args:
        wave_fields: All wave field samples
        source_positions: Corresponding source positions
        num_samples: Number of diverse samples to select
        
    Returns:
        Tuple of (selected_fields, selected_positions, indices)
    """
    total_samples = wave_fields.shape[0]
    
    if num_samples >= total_samples:
        return wave_fields, source_positions, list(range(total_samples))
    
    # Select diverse samples by spacing them evenly
    indices = np.linspace(0, total_samples - 1, num_samples, dtype=int)
    
    selected_fields = wave_fields[indices]
    selected_positions = source_positions[indices]
    
    print(f"📊 Selected {num_samples} diverse samples from {total_samples} total")
    for i, idx in enumerate(indices):
        pos = selected_positions[i]
        print(f"   Sample {i+1}: Source at ({pos[0]:.1f}, {pos[1]:.1f})")
    
    return selected_fields, selected_positions, indices.tolist()


def analyze_dataset_properties(wave_fields: torch.Tensor) -> dict:
    """
    Analyze statistical properties of the wave field dataset.
    
    Args:
        wave_fields: Wave field samples tensor
        
    Returns:
        properties: Statistical properties dictionary
    """
    # Convert to numpy for analysis
    fields_np = wave_fields.numpy()
    
    properties = {
        "num_samples": fields_np.shape[0],
        "spatial_resolution": fields_np.shape[-2:],  # Height, Width
        "amplitude_range": {
            "min": float(fields_np.min()),
            "max": float(fields_np.max()),
            "mean": float(fields_np.mean()),
            "std": float(fields_np.std())
        },
        "energy_distribution": {
            "mean_energy": float(np.mean(np.sum(fields_np ** 2, axis=(1, 2)))),
            "std_energy": float(np.std(np.sum(fields_np ** 2, axis=(1, 2))))
        }
    }
    
    # Compute gradient-based spatial frequency analysis
    gradients = []
    for i in range(min(5, fields_np.shape[0])):  # Sample a few for efficiency
        field = fields_np[i, 0] if fields_np.ndim == 4 else fields_np[i]
        grad_x = np.gradient(field, axis=1)
        grad_y = np.gradient(field, axis=0)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        gradients.append(grad_mag.mean())
    
    properties["spatial_frequency"] = {
        "mean": float(np.mean(gradients)),
        "std": float(np.std(gradients))
    }
    
    return properties


def print_dataset_summary(wave_fields: torch.Tensor, source_positions: torch.Tensor):
    """Print a comprehensive dataset summary."""
    
    print("\n" + "="*60)
    print("📊 WAVE ANALYSIS DATASET SUMMARY")
    print("="*60)
    
    # Basic info
    print(f"📋 Dataset Info:")
    print(f"   • Total samples: {wave_fields.shape[0]}")
    print(f"   • Wave field shape: {wave_fields.shape}")
    print(f"   • Source positions shape: {source_positions.shape}")
    
    # Wave field analysis
    properties = analyze_dataset_properties(wave_fields)
    
    print(f"\n🌊 Wave Field Properties:")
    amp = properties["amplitude_range"]
    print(f"   • Amplitude range: [{amp['min']:.3f}, {amp['max']:.3f}]")
    print(f"   • Mean amplitude: {amp['mean']:.3f} ± {amp['std']:.3f}")
    
    energy = properties["energy_distribution"]
    print(f"   • Mean energy: {energy['mean_energy']:.3f} ± {energy['std_energy']:.3f}")
    
    freq = properties["spatial_frequency"]
    print(f"   • Spatial frequency: {freq['mean']:.3f} ± {freq['std']:.3f}")
    
    # Source position analysis
    pos_np = source_positions.numpy()
    print(f"\n📍 Source Position Distribution:")
    print(f"   • X range: [{pos_np[:, 0].min():.1f}, {pos_np[:, 0].max():.1f}]")
    print(f"   • Y range: [{pos_np[:, 1].min():.1f}, {pos_np[:, 1].max():.1f}]")
    print(f"   • Mean position: ({pos_np[:, 0].mean():.1f}, {pos_np[:, 1].mean():.1f})")
    
    print("="*60 + "\n")


def main():
    """Test the dataset loading functionality."""
    
    print("🧪 Testing Analysis Dataset Loading...")
    
    try:
        # Load dataset
        wave_fields, source_positions = load_analysis_dataset()
        
        # Print summary
        print_dataset_summary(wave_fields, source_positions)
        
        # Test diverse sample selection
        diverse_fields, diverse_pos, indices = get_diverse_samples(
            wave_fields, source_positions, num_samples=5
        )
        
        print(f"✅ Dataset loading test completed successfully!")
        print(f"📋 Ready for activation maximization comparison")
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")


if __name__ == "__main__":
    main() 