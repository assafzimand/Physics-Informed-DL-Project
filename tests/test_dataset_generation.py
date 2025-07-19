"""
Dataset Generation Test Script

Tests the dataset generation pipeline with small batches and visualization.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import time

# Add src and configs directories to path (relative to tests/)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(os.path.join(project_root, 'src'))
sys.path.append(os.path.join(project_root, 'configs'))

from wave_simulation import Wave2DSimulator
import wave_simulation_config as config


def get_user_parameters():
    """Get timesteps and sample count from user."""
    print("🌊 Dataset Generation Test")
    print("=" * 40)
    
    # Get timesteps
    print("\nChoose simulation timesteps:")
    print("1. T=250 (shorter evolution)")
    print("2. T=500 (longer evolution)")
    print("3. Custom timesteps")
    
    timestep_choice = input("Enter choice (1, 2, or 3): ").strip()
    
    if timestep_choice == "1":
        timesteps = config.TIMESTEPS_T1
    elif timestep_choice == "2":
        timesteps = config.TIMESTEPS_T2
    elif timestep_choice == "3":
        while True:
            try:
                timesteps = int(input("Enter custom timesteps: "))
                if timesteps > 0:
                    break
                else:
                    print("Please enter a positive number")
            except ValueError:
                print("Please enter a valid number")
    else:
        print("Invalid choice, using default T=250")
        timesteps = config.TIMESTEPS_T1
    
    # Get number of samples
    print("\nNumber of samples to generate:")
    print("1. 50 samples (very fast)")
    print("2. 100 samples (fast)")
    print("3. 200 samples (moderate)")
    print("4. Custom number")
    
    sample_choice = input("Enter choice (1, 2, 3, or 4): ").strip()
    
    if sample_choice == "1":
        num_samples = 50
    elif sample_choice == "2":
        num_samples = 100
    elif sample_choice == "3":
        num_samples = 200
    elif sample_choice == "4":
        while True:
            try:
                num_samples = int(input("Enter number of samples: "))
                if num_samples > 0:
                    break
                else:
                    print("Please enter a positive number")
            except ValueError:
                print("Please enter a valid number")
    else:
        print("Invalid choice, using default 100 samples")
        num_samples = 100
    
    return timesteps, num_samples


def generate_random_source():
    """Generate random source location anywhere in grid."""
    x = random.randint(0, config.GRID_SIZE - 1)
    y = random.randint(0, config.GRID_SIZE - 1)
    return x, y


def generate_dataset_samples(timesteps, num_samples):
    """Generate dataset samples with progress tracking."""
    print(f"\n🔄 Generating {num_samples} samples with T={timesteps}...")
    
    # Initialize simulator using config
    sim_params = config.get_simulator_params()
    simulator = Wave2DSimulator(**sim_params)
    
    # Initialize storage
    wave_fields = np.zeros((num_samples, config.GRID_SIZE, config.GRID_SIZE), dtype=np.float32)
    source_coords = np.zeros((num_samples, 2), dtype=np.int32)
    
    # Time tracking
    start_time = time.time()
    
    # Generate samples with progress bar
    for i in tqdm(range(num_samples), desc="Generating samples"):
        # Random source location
        source_x, source_y = generate_random_source()
        
        # Run simulation
        final_wave, _ = simulator.simulate(source_x, source_y, timesteps)
        
        # Store results
        wave_fields[i] = final_wave.astype(np.float32)
        source_coords[i] = [source_x, source_y]
    
    generation_time = time.time() - start_time
    
    return wave_fields, source_coords, generation_time


def visualize_dataset_samples(wave_fields, source_coords, timesteps, num_samples, generation_time):
    """Visualize 5 random samples and show dataset statistics."""
    
    # Calculate dataset statistics
    total_size_mb = (wave_fields.nbytes + source_coords.nbytes) / (1024 * 1024)
    wave_min, wave_max = wave_fields.min(), wave_fields.max()
    wave_mean, wave_std = wave_fields.mean(), wave_fields.std()
    
    # Create figure with samples
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Select 5 random samples
    sample_indices = random.sample(range(num_samples), min(5, num_samples))
    
    # Plot 5 samples
    for i, idx in enumerate(sample_indices):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        wave_field = wave_fields[idx]
        source_x, source_y = source_coords[idx]
        
        # Plot wave field
        im = ax.imshow(wave_field, cmap='RdBu_r', origin='lower', vmin=wave_min, vmax=wave_max)
        ax.plot(source_x, source_y, 'ko', markersize=6, 
               markerfacecolor='yellow', markeredgecolor='black', markeredgewidth=1)
        
        ax.set_title(f'Sample {idx}: ({source_x}, {source_y})')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # Add colorbar to first plot
        if i == 0:
            plt.colorbar(im, ax=ax, shrink=0.8)
    
    # Hide the 6th subplot (we only use 5)
    axes[1, 2].axis('off')
    
    # Create comprehensive title with statistics
    stats_title = (
        f'Dataset Test: {num_samples} Samples, T={timesteps} | '
        f'Shape: {wave_fields.shape} | '
        f'Size: {total_size_mb:.1f} MB | '
        f'Range: [{wave_min:.3f}, {wave_max:.3f}] | '
        f'Mean±Std: {wave_mean:.3f}±{wave_std:.3f} | '
        f'Generation: {generation_time:.1f}s'
    )
    
    fig.suptitle(stats_title, fontsize=12, wrap=True)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # Make room for title
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Shape: {wave_fields.shape}")
    print(f"   Data type: {wave_fields.dtype}")
    print(f"   Memory size: {total_size_mb:.1f} MB")
    print(f"   Wave amplitude range: [{wave_min:.3f}, {wave_max:.3f}]")
    print(f"   Wave mean ± std: {wave_mean:.3f} ± {wave_std:.3f}")
    print(f"   Source coordinates range: X[{source_coords[:, 0].min()}-{source_coords[:, 0].max()}], Y[{source_coords[:, 1].min()}-{source_coords[:, 1].max()}]")
    print(f"   Generation time: {generation_time:.1f} seconds")
    print(f"   Time per sample: {generation_time/num_samples:.3f} seconds")
    
    plt.show()


def main():
    """Main dataset generation test function."""
    # Validate config
    if not config.validate_parameters():
        print("❌ Configuration validation failed!")
        return
    
    print()  # Add spacing after config validation
    
    # Get user parameters
    timesteps, num_samples = get_user_parameters()
    
    # Generate dataset
    wave_fields, source_coords, generation_time = generate_dataset_samples(timesteps, num_samples)
    
    # Visualize results
    visualize_dataset_samples(wave_fields, source_coords, timesteps, num_samples, generation_time)
    
    print(f"\n✅ Dataset generation test completed successfully!")
    print(f"Generated {num_samples} samples in {generation_time:.1f} seconds")


if __name__ == "__main__":
    main() 