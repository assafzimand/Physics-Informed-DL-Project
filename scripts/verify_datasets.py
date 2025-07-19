"""
Dataset Verification Script

Quick verification that the generated HDF5 datasets can be loaded correctly.
"""

import sys
import os
import numpy as np
import h5py

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
data_dir = os.path.join(project_root, 'data')

def verify_dataset(filepath):
    """Verify a single dataset file."""
    print(f"\n📂 Verifying: {os.path.basename(filepath)}")
    
    try:
        with h5py.File(filepath, 'r') as f:
            # Load data
            wave_fields = f['wave_fields'][:]
            source_coords = f['source_coords'][:]
            
            # Show metadata
            print(f"   Metadata:")
            print(f"     Grid size: {f.attrs['grid_size']}")
            print(f"     Timesteps: {f.attrs['timesteps']}")
            print(f"     Wave speed: {f.attrs['wave_speed']}")
            print(f"     CFL condition: {f.attrs['cfl_condition']:.3f}")
            print(f"     Samples: {f.attrs['num_samples']}")
            print(f"     Generation time: {f.attrs['generation_time']:.1f}s")
            
            # Show data shapes and ranges
            print(f"   Data:")
            print(f"     Wave fields shape: {wave_fields.shape}")
            print(f"     Source coords shape: {source_coords.shape}")
            print(f"     Wave range: [{wave_fields.min():.3f}, {wave_fields.max():.3f}]")
            print(f"     Source X range: [{source_coords[:, 0].min()}, {source_coords[:, 0].max()}]")
            print(f"     Source Y range: [{source_coords[:, 1].min()}, {source_coords[:, 1].max()}]")
            
            # Test a few random samples
            print(f"   Sample check:")
            for i in [0, 100, 500, 1000, 1999]:
                wave = wave_fields[i]
                source = source_coords[i]
                print(f"     Sample {i}: source=({source[0]:3d}, {source[1]:3d}), wave_range=[{wave.min():.3f}, {wave.max():.3f}]")
        
        print(f"   ✅ Dataset verified successfully!")
        return True
        
    except Exception as e:
        print(f"   ❌ Error loading dataset: {e}")
        return False

def main():
    """Main verification function."""
    print("🔍 Dataset Verification")
    print("=" * 40)
    
    # Check files exist
    files_to_check = [
        os.path.join(data_dir, 'wave_dataset_T250.h5'),
        os.path.join(data_dir, 'wave_dataset_T500.h5')
    ]
    
    all_good = True
    for filepath in files_to_check:
        if os.path.exists(filepath):
            success = verify_dataset(filepath)
            all_good = all_good and success
        else:
            print(f"❌ File not found: {filepath}")
            all_good = False
    
    print(f"\n" + "=" * 40)
    if all_good:
        print("🎉 All datasets verified successfully!")
        print("✅ Ready for machine learning training!")
    else:
        print("❌ Some datasets failed verification!")

if __name__ == "__main__":
    main() 