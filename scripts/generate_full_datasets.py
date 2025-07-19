"""
Full Dataset Generation Script

Generates the complete training datasets for wave source localization:
- Dataset 1: 2000 samples with T=250 timesteps
- Dataset 2: 2000 samples with T=500 timesteps

Saves both datasets in HDF5 format in the data/ directory.
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


def generate_dataset_samples(timesteps, num_samples, dataset_name):
    """Generate dataset samples with progress tracking."""
    print(f"\n🔄 Generating {dataset_name}...")
    print(f"   Samples: {num_samples}")
    print(f"   Timesteps: {timesteps}")
    print(f"   Grid: {config.GRID_SIZE}×{config.GRID_SIZE}")
    
    # Initialize simulator using config
    sim_params = config.get_simulator_params()
    simulator = Wave2DSimulator(**sim_params)
    
    # Initialize storage
    wave_fields = np.zeros((num_samples, config.GRID_SIZE, config.GRID_SIZE), dtype=np.float32)
    source_coords = np.zeros((num_samples, 2), dtype=np.int32)
    
    # Time tracking
    start_time = time.time()
    
    # Generate samples with progress bar
    for i in tqdm(range(num_samples), desc=f"Generating {dataset_name}"):
        # Random source location
        source_x, source_y = generate_random_source()
        
        # Run simulation
        final_wave, _ = simulator.simulate(source_x, source_y, timesteps)
        
        # Store results
        wave_fields[i] = final_wave.astype(np.float32)
        source_coords[i] = [source_x, source_y]
    
    generation_time = time.time() - start_time
    
    return wave_fields, source_coords, generation_time


def save_dataset_hdf5(wave_fields, source_coords, timesteps, generation_time, filename):
    """Save dataset to HDF5 format."""
    storage_params = config.get_storage_params()
    data_dir = storage_params['data_dir']
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, filename)
    
    print(f"\n💾 Saving dataset to {filepath}...")
    
    with h5py.File(filepath, 'w') as f:
        # Save metadata
        f.attrs['grid_size'] = config.GRID_SIZE
        f.attrs['wave_speed'] = config.WAVE_SPEED
        f.attrs['dt'] = config.DT
        f.attrs['dx'] = config.DX
        f.attrs['timesteps'] = timesteps
        f.attrs['num_samples'] = len(wave_fields)
        f.attrs['generation_time'] = generation_time
        f.attrs['created_by'] = 'Physics-Informed DL Project'
        f.attrs['cfl_condition'] = config.WAVE_SPEED * config.DT / config.DX
        
        # Save datasets
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


def print_dataset_summary(wave_fields, source_coords, timesteps, generation_time):
    """Print comprehensive dataset statistics."""
    total_size_mb = (wave_fields.nbytes + source_coords.nbytes) / (1024 * 1024)
    wave_min, wave_max = wave_fields.min(), wave_fields.max()
    wave_mean, wave_std = wave_fields.mean(), wave_fields.std()
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Timesteps: {timesteps}")
    print(f"   Shape: {wave_fields.shape}")
    print(f"   Data type: {wave_fields.dtype}")
    print(f"   Memory: {total_size_mb:.1f} MB")
    print(f"   Wave amplitude: [{wave_min:.3f}, {wave_max:.3f}]")
    print(f"   Wave mean ± std: {wave_mean:.3f} ± {wave_std:.3f}")
    print(f"   Source X range: [{source_coords[:, 0].min()}, {source_coords[:, 0].max()}]")
    print(f"   Source Y range: [{source_coords[:, 1].min()}, {source_coords[:, 1].max()}]")
    print(f"   Generation time: {generation_time:.1f} seconds")
    print(f"   Samples per second: {len(wave_fields)/generation_time:.2f}")


def main():
    """Main dataset generation function."""
    print("🌊 Physics-Informed DL Project - Full Dataset Generation")
    print("=" * 60)
    
    # Validate configuration
    if not config.validate_parameters():
        print("❌ Configuration validation failed!")
        return
    
    print()
    
    # Dataset parameters from config
    dataset_params = config.get_dataset_params()
    timesteps_list = dataset_params['timesteps']  # [250, 500]
    num_samples = dataset_params['num_samples']   # 2000
    
    print(f"📋 Generation Plan:")
    print(f"   Dataset 1: {num_samples} samples, T={timesteps_list[0]}")
    print(f"   Dataset 2: {num_samples} samples, T={timesteps_list[1]}")
    print(f"   Total samples: {2 * num_samples}")
    print(f"   Estimated time: ~{2 * num_samples * 0.092 / 60:.1f} minutes")
    
    confirmation = input(f"\nProceed with dataset generation? (y/n): ").strip().lower()
    if confirmation not in ['y', 'yes']:
        print("Dataset generation cancelled.")
        return
    
    total_start_time = time.time()
    generated_files = []
    
    # Generate Dataset 1: T=250
    print(f"\n" + "="*50)
    print(f"DATASET 1: T={timesteps_list[0]}")
    print(f"="*50)
    
    wave_fields_250, source_coords_250, gen_time_250 = generate_dataset_samples(
        timesteps_list[0], num_samples, f"Dataset T={timesteps_list[0]}"
    )
    
    print_dataset_summary(wave_fields_250, source_coords_250, timesteps_list[0], gen_time_250)
    
    file_250 = save_dataset_hdf5(
        wave_fields_250, source_coords_250, timesteps_list[0], 
        gen_time_250, f"wave_dataset_T{timesteps_list[0]}.h5"
    )
    generated_files.append(file_250)
    
    # Generate Dataset 2: T=500
    print(f"\n" + "="*50)
    print(f"DATASET 2: T={timesteps_list[1]}")
    print(f"="*50)
    
    wave_fields_500, source_coords_500, gen_time_500 = generate_dataset_samples(
        timesteps_list[1], num_samples, f"Dataset T={timesteps_list[1]}"
    )
    
    print_dataset_summary(wave_fields_500, source_coords_500, timesteps_list[1], gen_time_500)
    
    file_500 = save_dataset_hdf5(
        wave_fields_500, source_coords_500, timesteps_list[1], 
        gen_time_500, f"wave_dataset_T{timesteps_list[1]}.h5"
    )
    generated_files.append(file_500)
    
    # Final summary
    total_time = time.time() - total_start_time
    total_samples = 2 * num_samples
    
    print(f"\n" + "="*60)
    print(f"🎉 DATASET GENERATION COMPLETED!")
    print(f"="*60)
    print(f"📊 Final Statistics:")
    print(f"   Total samples generated: {total_samples}")
    print(f"   Total generation time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"   Average samples per second: {total_samples/total_time:.2f}")
    print(f"   Files generated:")
    for file_path in generated_files:
        file_size = os.path.getsize(file_path) / (1024 * 1024)
        print(f"     - {os.path.basename(file_path)} ({file_size:.1f} MB)")
    
    print(f"\n✅ Datasets ready for machine learning training!")
    print(f"📁 Location: data/ directory")


if __name__ == "__main__":
    main() 