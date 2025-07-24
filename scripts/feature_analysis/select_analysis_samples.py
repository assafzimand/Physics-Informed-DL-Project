#!/usr/bin/env python3
"""
Sample Selection for Feature Analysis
Selects 20 diverse samples from T=500 validation dataset for feature visualization.
"""

import sys
import os
import numpy as np
import json
import h5py
from pathlib import Path

# Add project root to path
sys.path.append('.')
sys.path.append('src')

from data.wave_dataset import WaveDataset


def categorize_source_location(coords, grid_size=128):
    """Categorize source location as corner, edge, or center."""
    x, y = coords
    margin = 20  # Define boundary for corners/edges
    
    # Corner check
    if (x < margin and y < margin) or \
       (x < margin and y > grid_size - margin) or \
       (x > grid_size - margin and y < margin) or \
       (x > grid_size - margin and y > grid_size - margin):
        return "corner"
    
    # Edge check
    elif x < margin or x > grid_size - margin or \
         y < margin or y > grid_size - margin:
        return "edge"
    
    # Center
    else:
        return "center"


def select_diverse_samples(dataset_path, n_samples=20):
    """
    Select diverse samples from validation dataset.
    
    Target distribution:
    - 4 corner sources
    - 8 edge sources
    - 4 center sources  
    - 4 random well-distributed
    """
    print(f"🔍 Loading T=500 validation dataset: {dataset_path}")
    
    # Load dataset
    dataset = WaveDataset(dataset_path)
    total_samples = len(dataset)
    print(f"✅ Loaded {total_samples} samples")
    
    # Categorize all samples
    samples_by_category = {"corner": [], "edge": [], "center": []}
    
    print("📊 Categorizing samples by source location...")
    for i in range(total_samples):
        _, coords = dataset[i]
        coords_np = coords.numpy()
        category = categorize_source_location(coords_np)
        samples_by_category[category].append({
            'index': i,
            'coords': coords_np.tolist(),
            'category': category
        })
    
    # Print distribution
    for cat, samples in samples_by_category.items():
        print(f"   {cat.capitalize()}: {len(samples)} samples")
    
    # Select samples for each category
    selected_samples = []
    
    # Select 4 corner samples
    corner_samples = samples_by_category["corner"]
    if len(corner_samples) >= 4:
        # Try to get good distribution across all 4 corners
        selected_corners = np.random.choice(len(corner_samples), 4, replace=False)
        for idx in selected_corners:
            selected_samples.append(corner_samples[idx])
    else:
        selected_samples.extend(corner_samples)
    
    # Select 8 edge samples
    edge_samples = samples_by_category["edge"]
    if len(edge_samples) >= 8:
        selected_edges = np.random.choice(len(edge_samples), 8, replace=False)
        for idx in selected_edges:
            selected_samples.append(edge_samples[idx])
    else:
        # Take all available edge samples
        selected_samples.extend(edge_samples)
        # Fill remaining from center
        remaining = 8 - len(edge_samples)
        center_samples = samples_by_category["center"]
        if len(center_samples) >= remaining:
            extra_centers = np.random.choice(len(center_samples), remaining, replace=False)
            for idx in extra_centers:
                selected_samples.append(center_samples[idx])
    
    # Select 4 center samples
    center_samples = samples_by_category["center"]
    current_center_count = sum(1 for s in selected_samples if s['category'] == 'center')
    needed_centers = max(0, 4 - current_center_count)
    
    if len(center_samples) >= needed_centers:
        # Avoid duplicates
        used_center_indices = [s['index'] for s in selected_samples if s['category'] == 'center']
        available_centers = [s for s in center_samples if s['index'] not in used_center_indices]
        
        if len(available_centers) >= needed_centers:
            selected_center_indices = np.random.choice(len(available_centers), needed_centers, replace=False)
            for idx in selected_center_indices:
                selected_samples.append(available_centers[idx])
    
    # Select 4 random well-distributed samples
    # Get remaining samples not yet selected
    used_indices = set(s['index'] for s in selected_samples)
    remaining_samples = []
    for category_samples in samples_by_category.values():
        for sample in category_samples:
            if sample['index'] not in used_indices:
                remaining_samples.append(sample)
    
    needed_random = max(0, n_samples - len(selected_samples))
    if len(remaining_samples) >= needed_random:
        random_indices = np.random.choice(len(remaining_samples), needed_random, replace=False)
        for idx in random_indices:
            selected_samples.append(remaining_samples[idx])
    
    # Sort by index for consistent ordering
    selected_samples.sort(key=lambda x: x['index'])
    
    print(f"\n✅ Selected {len(selected_samples)} samples:")
    final_distribution = {}
    for sample in selected_samples:
        cat = sample['category']
        final_distribution[cat] = final_distribution.get(cat, 0) + 1
    
    for cat, count in final_distribution.items():
        print(f"   {cat.capitalize()}: {count} samples")
    
    return selected_samples


def save_sample_data(selected_samples, dataset_path, output_dir):
    """Save selected samples and metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset for extracting actual data
    dataset = WaveDataset(dataset_path)
    
    # Save metadata
    metadata = {
        'total_samples': len(selected_samples),
        'source_dataset': str(dataset_path),
        'samples': selected_samples,
        'selection_criteria': {
            'corner_sources': '4 samples from grid corners',
            'edge_sources': '8 samples from grid edges',
            'center_sources': '4 samples from grid center',
            'random_sources': '4 additional well-distributed samples'
        }
    }
    
    metadata_file = output_dir / "sample_info.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Metadata saved: {metadata_file}")
    
    # Save raw data for selected samples
    raw_data_dir = output_dir / "raw_data"
    raw_data_dir.mkdir(exist_ok=True)
    
    print("💾 Saving raw sample data...")
    for i, sample_info in enumerate(selected_samples):
        sample_idx = sample_info['index']
        wave_data, coords = dataset[sample_idx]
        
        # Save as npz for easy loading
        sample_file = raw_data_dir / f"sample_{i:02d}_idx_{sample_idx}.npz"
        np.savez(sample_file, 
                wave_field=wave_data.numpy(),
                coordinates=coords.numpy(),
                original_index=sample_idx,
                category=sample_info['category'])
    
    print(f"✅ Raw data saved to: {raw_data_dir}")
    return metadata_file, raw_data_dir


def main():
    """Main function to select and save analysis samples."""
    print("🚀 Starting sample selection for feature analysis...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Paths
    dataset_path = "data/wave_dataset_T500_validation.h5"
    output_dir = "experiments/feature_analysis/samples"
    
    # Check if dataset exists
    if not Path(dataset_path).exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return
    
    # Select samples
    selected_samples = select_diverse_samples(dataset_path, n_samples=20)
    
    # Save results
    metadata_file, raw_data_dir = save_sample_data(selected_samples, dataset_path, output_dir)
    
    print(f"\n🎉 Sample selection complete!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"📊 Next step: Create feature extraction script")
    

if __name__ == "__main__":
    main() 