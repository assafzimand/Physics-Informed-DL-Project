#!/usr/bin/env python3
"""
Plot Analysis Source Locations
Visualizes the spatial distribution of our 20 selected samples.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path

# Add project root to path
sys.path.append('.')
sys.path.append('src')


def plot_source_locations(dataset_file, output_dir):
    """Plot the spatial distribution of source locations."""
    print(f"📍 Loading source locations from: {dataset_file}")
    
    # Load dataset
    with h5py.File(dataset_file, 'r') as f:
        coordinates = f['coordinates'][:]
        categories = [cat.decode('utf-8') for cat in f['categories'][:]]
        original_indices = f['original_indices'][:]
        grid_size = f.attrs['grid_size']
        total_samples = f.attrs['total_samples']
    
    print(f"✅ Loaded {total_samples} samples (grid size: {grid_size}x{grid_size})")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up the plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Define colors for categories
    colors = {
        'corner': '#FF6B6B',    # Red
        'edge': '#4ECDC4',      # Teal  
        'center': '#45B7D1'     # Blue
    }
    
    # Count samples by category
    category_counts = {}
    for cat in categories:
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    # Plot each category
    for category, color in colors.items():
        # Get coordinates for this category
        cat_indices = [i for i, cat in enumerate(categories) if cat == category]
        if cat_indices:
            cat_coords = coordinates[cat_indices]
            count = len(cat_indices)
            
            ax.scatter(cat_coords[:, 0], cat_coords[:, 1], 
                      c=color, s=100, alpha=0.8, 
                      label=f'{category.capitalize()} ({count} samples)',
                      edgecolors='black', linewidth=1)
    
    # Add sample numbers
    for i, (coord, cat, orig_idx) in enumerate(zip(coordinates, categories, original_indices)):
        ax.annotate(f'{i}', (coord[0], coord[1]), 
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # Customize plot
    ax.set_xlim(-5, grid_size + 5)
    ax.set_ylim(-5, grid_size + 5)
    ax.set_xlabel('X Coordinate', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y Coordinate', fontsize=12, fontweight='bold')
    ax.set_title('Analysis Dataset: Source Location Distribution\n' + 
                f'20 Samples for Feature Visualization', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')
    
    # Add legend
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    # Add boundary indicators
    margin = 20
    # Corner boundaries
    ax.axvline(x=margin, color='red', linestyle=':', alpha=0.5, linewidth=1)
    ax.axvline(x=grid_size-margin, color='red', linestyle=':', alpha=0.5, linewidth=1)
    ax.axhline(y=margin, color='red', linestyle=':', alpha=0.5, linewidth=1)
    ax.axhline(y=grid_size-margin, color='red', linestyle=':', alpha=0.5, linewidth=1)
    
    # Add text annotations for regions
    ax.text(10, grid_size-10, 'Corner\nRegion', ha='center', va='center', 
           fontsize=10, alpha=0.7, weight='bold')
    ax.text(grid_size/2, 10, 'Edge Region', ha='center', va='center',
           fontsize=10, alpha=0.7, weight='bold')
    ax.text(grid_size/2, grid_size/2, 'Center\nRegion', ha='center', va='center',
           fontsize=10, alpha=0.7, weight='bold', 
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))
    
    # Save plot
    output_file = output_dir / "analysis_source_locations.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved: {output_file}")
    
    # Print summary
    print(f"\n📊 Source Location Summary:")
    print(f"   Total samples: {total_samples}")
    for cat, count in category_counts.items():
        print(f"   {cat.capitalize()}: {count} samples")
    
    # Show plot
    plt.show()
    
    return output_file


def main():
    """Main function to plot source locations."""
    print("🗺️  Plotting analysis dataset source locations...")
    
    # Paths
    dataset_file = "data/wave_dataset_analysis_20samples.h5"
    output_dir = "experiments/feature_analysis/plots"
    
    # Check if dataset exists
    if not Path(dataset_file).exists():
        print(f"❌ Dataset not found: {dataset_file}")
        print("💡 Run create_analysis_dataset.py first!")
        return
    
    # Create plot
    plot_file = plot_source_locations(dataset_file, output_dir)
    
    print(f"\n🎉 Source location plot complete!")
    print(f"📁 Plot saved to: {plot_file}")


if __name__ == "__main__":
    main() 