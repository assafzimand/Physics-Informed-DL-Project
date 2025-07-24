#!/usr/bin/env python3
"""
Create Analysis Dataset
Combines the 20 selected samples into a single HDF5 dataset file.
"""

import sys
import os
import numpy as np
import h5py
import json
from pathlib import Path

# Add project root to path
sys.path.append('.')
sys.path.append('src')


def create_analysis_dataset(samples_dir, output_file):
    """Create a single HDF5 dataset from our 20 selected samples."""
    print("📦 Creating consolidated analysis dataset...")
    
    samples_dir = Path(samples_dir)
    raw_data_dir = samples_dir / "raw_data"
    
    # Load sample metadata
    metadata_file = samples_dir / "sample_info.json"
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    print(f"🔍 Found {metadata['total_samples']} samples to combine")
    
    # Get all sample files
    sample_files = sorted(raw_data_dir.glob("*.npz"))
    n_samples = len(sample_files)
    
    if n_samples == 0:
        print("❌ No sample files found!")
        return
    
    # Load first sample to get dimensions
    first_sample = np.load(sample_files[0])
    wave_shape = first_sample['wave_field'].shape  # (timesteps, height, width)
    
    print(f"📊 Sample dimensions: {wave_shape}")
    print(f"💾 Creating dataset: {output_file}")
    
    # Create HDF5 dataset
    with h5py.File(output_file, 'w') as f:
        # Create datasets
        wave_data = f.create_dataset('wave_fields', 
                                   shape=(n_samples, *wave_shape),
                                   dtype=np.float32,
                                   compression='gzip')
        
        coordinates = f.create_dataset('coordinates',
                                     shape=(n_samples, 2),
                                     dtype=np.float32)
        
        # Create metadata datasets
        original_indices = f.create_dataset('original_indices',
                                          shape=(n_samples,),
                                          dtype=np.int32)
        
        categories = f.create_dataset('categories',
                                    shape=(n_samples,),
                                    dtype=h5py.string_dtype())
        
        # Fill datasets
        print("💫 Loading and saving samples...")
        for i, sample_file in enumerate(sample_files):
            sample_data = np.load(sample_file)
            
            wave_data[i] = sample_data['wave_field']
            coordinates[i] = sample_data['coordinates']
            original_indices[i] = sample_data['original_index']
            categories[i] = str(sample_data['category'])
            
            print(f"   ✅ Sample {i:2d}: {sample_file.name} "
                  f"({sample_data['category']} source)")
        
        # Add metadata as attributes
        f.attrs['total_samples'] = n_samples
        f.attrs['source_dataset'] = metadata['source_dataset']
        f.attrs['timesteps'] = wave_shape[0]
        f.attrs['grid_size'] = wave_shape[1]
        f.attrs['wave_speed'] = 16.7  # T=500 wave speed
        f.attrs['description'] = 'Analysis dataset: 20 diverse samples for feature visualization'
        
        # Add selection criteria
        selection_info = json.dumps(metadata['selection_criteria'])
        f.attrs['selection_criteria'] = selection_info
    
    print(f"✅ Analysis dataset created: {output_file}")
    print(f"📈 Contains {n_samples} samples from diverse source locations")
    
    return output_file


def main():
    """Main function to create analysis dataset."""
    print("🚀 Creating analysis dataset from selected samples...")
    
    # Paths
    samples_dir = "experiments/feature_analysis/samples"
    output_file = "data/wave_dataset_analysis_20samples.h5"
    
    # Check if samples exist
    if not Path(samples_dir).exists():
        print(f"❌ Samples directory not found: {samples_dir}")
        return
    
    # Create dataset
    dataset_file = create_analysis_dataset(samples_dir, output_file)
    
    print(f"\n🎉 Analysis dataset ready!")
    print(f"📁 File: {dataset_file}")
    print(f"📊 Next step: Plot source locations")


if __name__ == "__main__":
    main() 