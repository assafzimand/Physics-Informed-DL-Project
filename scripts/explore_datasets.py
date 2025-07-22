"""
Dataset Exploration Script

Quick visualization of random samples from both datasets.
Shows 5 random samples from T=250 and T=500 datasets.
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import random

def load_dataset(filepath):
    """Load dataset from HDF5 file."""
    with h5py.File(filepath, 'r') as f:
        wave_fields = f['wave_fields'][:]
        source_coords = f['source_coords'][:]
        timesteps = f.attrs['timesteps']
    return wave_fields, source_coords, timesteps

def plot_random_samples():
    """Plot 5 random samples from each dataset."""
    
    # Load both datasets
    print("Loading datasets...")
    wave_250, coords_250, t_250 = load_dataset('wave_dataset_T250.h5')
    wave_500, coords_500, t_500 = load_dataset('wave_dataset_T500.h5')
    
    print(f"T=250 dataset: {wave_250.shape}")
    print(f"T=500 dataset: {wave_500.shape}")
    
    # Create figure with 2 rows, 5 columns
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    
    # Get global min/max for consistent color scaling
    global_min = min(wave_250.min(), wave_500.min())
    global_max = max(wave_250.max(), wave_500.max())
    
    # Plot 5 random samples from T=250 (top row)
    print("\nT=250 samples:")
    indices_250 = random.sample(range(len(wave_250)), 5)
    for i, idx in enumerate(indices_250):
        ax = axes[0, i]
        wave = wave_250[idx]
        source_x, source_y = coords_250[idx]
        
        im = ax.imshow(wave, cmap='RdBu_r', origin='lower', 
                      vmin=global_min, vmax=global_max)
        ax.plot(source_x, source_y, 'ko', markersize=8, 
               markerfacecolor='yellow', markeredgecolor='black', markeredgewidth=2)
        
        ax.set_title(f'T={t_250}, Source: ({source_x}, {source_y})')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        print(f"  Sample {idx}: source=({source_x:3d}, {source_y:3d}), range=[{wave.min():.3f}, {wave.max():.3f}]")
    
    # Plot 5 random samples from T=500 (bottom row)
    print("\nT=500 samples:")
    indices_500 = random.sample(range(len(wave_500)), 5)
    for i, idx in enumerate(indices_500):
        ax = axes[1, i]
        wave = wave_500[idx]
        source_x, source_y = coords_500[idx]
        
        im = ax.imshow(wave, cmap='RdBu_r', origin='lower', 
                      vmin=global_min, vmax=global_max)
        ax.plot(source_x, source_y, 'ko', markersize=8, 
               markerfacecolor='yellow', markeredgecolor='black', markeredgewidth=2)
        
        ax.set_title(f'T={t_500}, Source: ({source_x}, {source_y})')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        print(f"  Sample {idx}: source=({source_x:3d}, {source_y:3d}), range=[{wave.min():.3f}, {wave.max():.3f}]")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label('Wave Amplitude', fontsize=12)
    
    # Set main title
    plt.suptitle('Random Dataset Samples: Wave Patterns and Source Locations', fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    
    print(f"\nShowing 10 random samples (5 from each dataset)")
    print(f"Global wave range: [{global_min:.3f}, {global_max:.3f}]")
    plt.show()

if __name__ == "__main__":
    plot_random_samples() 