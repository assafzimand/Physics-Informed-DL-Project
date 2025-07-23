#!/usr/bin/env python3
"""
Generate Extra Validation Datasets
Creates 500-sample validation datasets for T=250 and T=500 for model testing.
Uses the same pattern as generate_full_datasets.py but with different random seeds.
"""

import sys
import os
import numpy as np
import h5py
import random
import time
from tqdm import tqdm

# Add src and configs directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(os.path.join(project_root, 'src'))
sys.path.append(os.path.join(project_root, 'configs'))

from wave_simulation import Wave2DSimulator
import wave_simulation_config as config


def generate_random_source():
    """Generate random source location anywhere in grid."""
    x = random.randint(0, config.GRID_SIZE - 1)
    y = random.randint(0, config.GRID_SIZE - 1)
    return x, y


def generate_validation_dataset_samples(timesteps, num_samples, dataset_name, validation_seed):
    """Generate validation dataset samples with progress tracking."""
    print(f"\n🔄 Generating {dataset_name}...")
    print(f"   Samples: {num_samples}")
    print(f"   Timesteps: {timesteps}")
    print(f"   Grid: {config.GRID_SIZE}×{config.GRID_SIZE}")
    print(f"   Validation seed: {validation_seed}")
    
    # Set different random seed for validation data
    random.seed(validation_seed)
    np.random.seed(validation_seed)
    
    # Initialize simulator using config (same as generate_full_datasets.py)
    sim_params = config.get_simulator_params()
    simulator = Wave2DSimulator(**sim_params)
    
    # Initialize storage
    wave_fields = np.zeros((num_samples, config.GRID_SIZE, config.GRID_SIZE), dtype=np.float32)
    source_coords = np.zeros((num_samples, 2), dtype=np.int32)
    
    # Time tracking
    start_time = time.time()
    
    # Generate samples with progress bar (exact same pattern)
    for i in tqdm(range(num_samples), desc=f"Generating {dataset_name}"):
        # Random source location
        source_x, source_y = generate_random_source()
        
        # Run simulation (exact same as generate_full_datasets.py)
        final_wave, _ = simulator.simulate(source_x, source_y, timesteps)
        
        # Store results
        wave_fields[i] = final_wave.astype(np.float32)
        source_coords[i] = [source_x, source_y]
    
    generation_time = time.time() - start_time
    
    return wave_fields, source_coords, generation_time


def save_validation_dataset_hdf5(wave_fields, source_coords, timesteps, generation_time, filename, validation_seed):
    """Save validation dataset to HDF5 format (same format as training datasets)."""
    data_dir = "data"
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, filename)
    
    print(f"\n💾 Saving dataset to {filepath}...")
    
    with h5py.File(filepath, 'w') as f:
        # Save metadata (same format as training datasets)
        f.attrs['grid_size'] = config.GRID_SIZE
        f.attrs['wave_speed'] = config.WAVE_SPEED
        f.attrs['dt'] = config.DT
        f.attrs['dx'] = config.DX
        f.attrs['timesteps'] = timesteps
        f.attrs['num_samples'] = len(wave_fields)
        f.attrs['generation_time'] = generation_time
        f.attrs['created_by'] = 'Physics-Informed DL Project'
        f.attrs['cfl_condition'] = config.WAVE_SPEED * config.DT / config.DX
        f.attrs['purpose'] = 'Extra validation dataset'
        f.attrs['validation_seed'] = validation_seed
        
        # Save datasets (same compression as training datasets)
        f.create_dataset('wave_fields', data=wave_fields, compression='gzip', compression_opts=6)
        f.create_dataset('source_coords', data=source_coords, compression='gzip', compression_opts=6)
        
        # Calculate storage statistics
        total_size_mb = (wave_fields.nbytes + source_coords.nbytes) / (1024 * 1024)
        wave_min, wave_max = wave_fields.min(), wave_fields.max()
        wave_mean, wave_std = wave_fields.mean(), wave_fields.std()
        
        # Save statistics as attributes
        f.attrs['total_size_mb'] = total_size_mb
        f.attrs['wave_min'] = wave_min
        f.attrs['wave_max'] = wave_max
        f.attrs['wave_mean'] = wave_mean
        f.attrs['wave_std'] = wave_std
        f.attrs['source_x_min'] = source_coords[:, 0].min()
        f.attrs['source_x_max'] = source_coords[:, 0].max()
        f.attrs['source_y_min'] = source_coords[:, 1].min()
        f.attrs['source_y_max'] = source_coords[:, 1].max()
    
    print(f"✅ Dataset saved successfully!")
    print(f"   File: {filepath}")
    print(f"   Size: {total_size_mb:.1f} MB")
    print(f"   Wave range: [{wave_min:.3f}, {wave_max:.3f}]")
    
    return filepath


def main():
    """Main validation dataset generation function."""
    print("🎯 Extra Validation Dataset Generation")
    print("=" * 60)
    
    # Validate configuration
    if not config.validate_parameters():
        print("❌ Configuration validation failed!")
        return
    
    print()
    
    # Validation dataset parameters
    num_samples = 500  # 500 samples each
    timesteps_list = [250, 500]  # T=250 and T=500
    
    # Different seeds for validation data (different from training data)
    validation_seeds = {
        250: 99999,  # Different seed for T=250 validation
        500: 88888   # Different seed for T=500 validation
    }
    
    print(f"📋 Validation Generation Plan:")
    print(f"   Dataset 1: {num_samples} samples, T={timesteps_list[0]} (seed={validation_seeds[250]})")
    print(f"   Dataset 2: {num_samples} samples, T={timesteps_list[1]} (seed={validation_seeds[500]})")
    print(f"   Total samples: {2 * num_samples}")
    print(f"   Estimated time: ~{2 * num_samples * 0.092 / 60:.1f} minutes")
    
    total_start_time = time.time()
    generated_files = []
    
    # Generate Validation Dataset 1: T=250
    print(f"\n" + "="*50)
    print(f"VALIDATION DATASET 1: T={timesteps_list[0]}")
    print(f"="*50)
    
    wave_fields_250, source_coords_250, gen_time_250 = generate_validation_dataset_samples(
        timesteps_list[0], num_samples, f"Validation T={timesteps_list[0]}", validation_seeds[250]
    )
    
    filename_250 = f"wave_dataset_T{timesteps_list[0]}_validation.h5"
    filepath_250 = save_validation_dataset_hdf5(
        wave_fields_250, source_coords_250, timesteps_list[0], gen_time_250, filename_250, validation_seeds[250]
    )
    generated_files.append(filepath_250)
    
    # Generate Validation Dataset 2: T=500
    print(f"\n" + "="*50)
    print(f"VALIDATION DATASET 2: T={timesteps_list[1]}")
    print(f"="*50)
    
    wave_fields_500, source_coords_500, gen_time_500 = generate_validation_dataset_samples(
        timesteps_list[1], num_samples, f"Validation T={timesteps_list[1]}", validation_seeds[500]
    )
    
    filename_500 = f"wave_dataset_T{timesteps_list[1]}_validation.h5"
    filepath_500 = save_validation_dataset_hdf5(
        wave_fields_500, source_coords_500, timesteps_list[1], gen_time_500, filename_500, validation_seeds[500]
    )
    generated_files.append(filepath_500)
    
    # Final summary
    total_time = time.time() - total_start_time
    
    print(f"\n" + "="*60)
    print(f"🎉 VALIDATION DATASET GENERATION COMPLETE!")
    print(f"="*60)
    print(f"⏱️ Total time: {total_time/60:.1f} minutes")
    print(f"📊 Generated files:")
    
    for filepath in generated_files:
        file_size = os.path.getsize(filepath) / (1024**3)
        print(f"   {os.path.basename(filepath)}: {file_size:.2f} GB")
    
    print(f"\n📁 Files saved in: data/")
    print(f"🎯 Next step: Use scripts/extra_validation.py to test models")
    print(f"=" * 60)


if __name__ == "__main__":
    main() 