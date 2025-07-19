"""
Dataset Generator for Wave Source Localization

This module generates training datasets by running wave simulations with random source locations
and saving the results in HDF5 format for machine learning training.
"""

import numpy as np
import h5py
import os
from typing import List, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
from wave_simulation import Wave2DSimulator
import random


class WaveDatasetGenerator:
    """
    Generates datasets for wave source localization using physics simulations.
    """
    
    def __init__(self, grid_size: int = 128, wave_speed: float = 1.0, 
                 dt: float = 0.05, dx: float = 1.0):
        """
        Initialize the dataset generator.
        
        Args:
            grid_size: Size of simulation grid
            wave_speed: Wave propagation speed
            dt: Time step (smaller for stability with longer simulations)
            dx: Spatial step size
        """
        self.grid_size = grid_size
        self.simulator = Wave2DSimulator(grid_size, wave_speed, dt, dx)
        
    def generate_random_source(self) -> Tuple[int, int]:
        """
        Generate a random source location within the grid boundaries.
        Keeps sources away from edges to avoid boundary effects.
        
        Returns:
            Tuple of (x, y) coordinates
        """
        # Keep sources at least 10 pixels from edges
        margin = 10
        x = random.randint(margin, self.grid_size - margin - 1)
        y = random.randint(margin, self.grid_size - margin - 1)
        return x, y
    
    def generate_single_sample(self, timesteps: int) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Generate a single training sample.
        
        Args:
            timesteps: Number of time steps to simulate
            
        Returns:
            Tuple of (wave_field, source_coordinates)
        """
        # Generate random source location
        source_x, source_y = self.generate_random_source()
        
        # Run simulation
        final_wave, _ = self.simulator.simulate(source_x, source_y, timesteps)
        
        return final_wave, (source_x, source_y)
    
    def generate_dataset(self, num_samples: int, timesteps: int, 
                        dataset_name: str = "wave_dataset") -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a complete dataset with multiple samples.
        
        Args:
            num_samples: Number of samples to generate
            timesteps: Number of time steps for each simulation
            dataset_name: Name for progress bar
            
        Returns:
            Tuple of (wave_fields, source_coordinates) arrays
        """
        # Initialize arrays to store results
        wave_fields = np.zeros((num_samples, self.grid_size, self.grid_size), dtype=np.float32)
        source_coords = np.zeros((num_samples, 2), dtype=np.int32)
        
        # Generate samples with progress bar
        for i in tqdm(range(num_samples), desc=f"Generating {dataset_name} (T={timesteps})"):
            wave_field, (source_x, source_y) = self.generate_single_sample(timesteps)
            
            wave_fields[i] = wave_field.astype(np.float32)
            source_coords[i] = [source_x, source_y]
        
        return wave_fields, source_coords
    
    def save_datasets_hdf5(self, datasets: dict, filename: str, data_dir: str = "data"):
        """
        Save multiple datasets to HDF5 format.
        
        Args:
            datasets: Dictionary with structure:
                     {timestep: (wave_fields, source_coords), ...}
            filename: Name of the HDF5 file
            data_dir: Directory to save the file
        """
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        filepath = os.path.join(data_dir, filename)
        
        print(f"\nSaving datasets to {filepath}...")
        
        with h5py.File(filepath, 'w') as f:
            # Save metadata
            f.attrs['grid_size'] = self.grid_size
            f.attrs['wave_speed'] = self.simulator.wave_speed
            f.attrs['dt'] = self.simulator.dt
            f.attrs['dx'] = self.simulator.dx
            f.attrs['created_by'] = 'Physics-Informed DL Project'
            
            # Save each dataset
            for timestep, (wave_fields, source_coords) in datasets.items():
                group_name = f'T_{timestep}'
                group = f.create_group(group_name)
                
                # Save wave fields and coordinates
                group.create_dataset('images', data=wave_fields, compression='gzip')
                group.create_dataset('coordinates', data=source_coords, compression='gzip')
                
                # Save metadata for this timestep
                group.attrs['timesteps'] = timestep
                group.attrs['num_samples'] = len(wave_fields)
                group.attrs['description'] = f'Wave fields after {timestep} time steps'
                
                print(f"  - {group_name}: {len(wave_fields)} samples, "
                      f"shape {wave_fields.shape}")
        
        print(f"Datasets saved successfully!")
        return filepath
    
    def load_datasets_hdf5(self, filepath: str) -> dict:
        """
        Load datasets from HDF5 file.
        
        Args:
            filepath: Path to HDF5 file
            
        Returns:
            Dictionary with loaded datasets
        """
        datasets = {}
        
        with h5py.File(filepath, 'r') as f:
            print(f"Loading datasets from {filepath}...")
            print(f"Metadata: grid_size={f.attrs['grid_size']}, "
                  f"wave_speed={f.attrs['wave_speed']}")
            
            for group_name in f.keys():
                if group_name.startswith('T_'):
                    timestep = int(group_name.split('_')[1])
                    group = f[group_name]
                    
                    wave_fields = group['images'][:]
                    source_coords = group['coordinates'][:]
                    
                    datasets[timestep] = (wave_fields, source_coords)
                    print(f"  - {group_name}: {len(wave_fields)} samples loaded")
        
        return datasets
    
    def visualize_samples(self, wave_fields: np.ndarray, source_coords: np.ndarray, 
                         num_samples: int = 4, timestep: int = None, save_dir: str = None):
        """
        Visualize a few random samples from the dataset.
        
        Args:
            wave_fields: Array of wave field images
            source_coords: Array of source coordinates
            num_samples: Number of samples to visualize
            timestep: Timestep value for title
            save_dir: Directory to save plots
        """
        # Select random samples
        indices = random.sample(range(len(wave_fields)), num_samples)
        
        # Create subplot grid
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, idx in enumerate(indices):
            ax = axes[i]
            wave_field = wave_fields[idx]
            source_x, source_y = source_coords[idx]
            
            # Plot wave field
            im = ax.imshow(wave_field, cmap='RdBu_r', origin='lower')
            ax.plot(source_x, source_y, 'ko', markersize=8, 
                   markerfacecolor='yellow', markeredgecolor='black', markeredgewidth=2)
            
            ax.set_title(f'Sample {idx}: Source at ({source_x}, {source_y})')
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            plt.colorbar(im, ax=ax, shrink=0.8)
        
        title = f'Wave Dataset Samples'
        if timestep:
            title += f' (T={timestep})'
        fig.suptitle(title, fontsize=16)
        
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'samples_T_{timestep}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Sample visualization saved to {save_path}")
        
        plt.show()


def main():
    """
    Main function to generate the wave datasets.
    """
    print("🌊 Wave Dataset Generation Starting...")
    print("=" * 50)
    
    # Configuration
    GRID_SIZE = 128
    NUM_SAMPLES = 2000  # Samples per dataset
    TIMESTEPS = [250, 500]  # Two different timestep values
    WAVE_SPEED = 0.8  # Adjusted for stability with longer simulations
    DT = 0.03  # Smaller timestep for stability
    
    # Initialize generator
    generator = WaveDatasetGenerator(
        grid_size=GRID_SIZE, 
        wave_speed=WAVE_SPEED, 
        dt=DT
    )
    
    print(f"Configuration:")
    print(f"  - Grid size: {GRID_SIZE}×{GRID_SIZE}")
    print(f"  - Samples per dataset: {NUM_SAMPLES}")
    print(f"  - Timesteps: {TIMESTEPS}")
    print(f"  - Wave speed: {WAVE_SPEED}")
    print(f"  - Time step: {DT}")
    print(f"  - CFL condition: {generator.simulator.cfl_condition:.3f}")
    print()
    
    # Generate datasets
    datasets = {}
    
    for timestep in TIMESTEPS:
        print(f"\n🔄 Generating dataset for T={timestep}...")
        wave_fields, source_coords = generator.generate_dataset(
            num_samples=NUM_SAMPLES,
            timesteps=timestep,
            dataset_name=f"T_{timestep}"
        )
        
        datasets[timestep] = (wave_fields, source_coords)
        
        # Show statistics
        print(f"Dataset T={timestep} completed:")
        print(f"  - Wave fields shape: {wave_fields.shape}")
        print(f"  - Source coordinates shape: {source_coords.shape}")
        print(f"  - Wave amplitude range: [{wave_fields.min():.3f}, {wave_fields.max():.3f}]")
        
        # Visualize samples
        print(f"Visualizing samples for T={timestep}...")
        generator.visualize_samples(wave_fields, source_coords, 
                                  num_samples=4, timestep=timestep, 
                                  save_dir="results/sample_visualizations")
    
    # Save all datasets to HDF5
    print(f"\n💾 Saving datasets...")
    filepath = generator.save_datasets_hdf5(datasets, "wave_datasets.h5")
    
    # Verify by loading
    print(f"\n✅ Verifying saved data...")
    loaded_datasets = generator.load_datasets_hdf5(filepath)
    
    print(f"\n🎉 Dataset generation completed successfully!")
    print(f"Generated {len(TIMESTEPS)} datasets with {NUM_SAMPLES} samples each")
    print(f"Total simulations run: {len(TIMESTEPS) * NUM_SAMPLES}")
    print(f"Data saved to: {filepath}")


if __name__ == "__main__":
    main() 