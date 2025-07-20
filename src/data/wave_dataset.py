"""
PyTorch Dataset for Wave Source Localization

Custom dataset class for loading wave simulation data from HDF5 files.
Supports train/validation splits, data normalization, and efficient loading.
"""

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Tuple, Optional, Dict, Any


class WaveDataset(Dataset):
    """
    PyTorch Dataset for wave source localization data.
    
    Loads wave fields and source coordinates from HDF5 files.
    Supports data normalization and validation splits.
    """
    
    def __init__(self, 
                 hdf5_path: str, 
                 normalize_wave_fields: bool = True,
                 normalize_coordinates: bool = False,
                 grid_size: int = 128):
        """
        Initialize the WaveDataset.
        
        Args:
            hdf5_path: Path to HDF5 file containing wave data
            normalize_wave_fields: Whether to normalize wave amplitudes
            normalize_coordinates: Whether to normalize coordinates to [0,1]
            grid_size: Size of the wave field grid (default 128)
        """
        self.hdf5_path = hdf5_path
        self.normalize_wave_fields = normalize_wave_fields
        self.normalize_coordinates = normalize_coordinates
        self.grid_size = grid_size
        
        # Verify file exists
        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
        
        # Load dataset info and compute statistics
        self._load_dataset_info()
        self._compute_statistics()
        
    def _load_dataset_info(self):
        """Load basic dataset information."""
        with h5py.File(self.hdf5_path, 'r') as f:
            self.num_samples = f['wave_fields'].shape[0]
            self.timesteps = f.attrs.get('timesteps', 'unknown')
            self.wave_speed = f.attrs.get('wave_speed', 'unknown')
            
            print(f"Loaded dataset: {self.hdf5_path}")
            print(f"  - Samples: {self.num_samples}")
            print(f"  - Timesteps: {self.timesteps}")
            print(f"  - Wave speed: {self.wave_speed}")
    
    def _compute_statistics(self):
        """Compute normalization statistics for wave fields."""
        if not self.normalize_wave_fields:
            self.wave_mean = 0.0
            self.wave_std = 1.0
            return
            
        print("Computing normalization statistics...")
        
        # Compute statistics in chunks to handle large datasets
        chunk_size = min(100, self.num_samples)
        wave_values = []
        
        with h5py.File(self.hdf5_path, 'r') as f:
            for i in range(0, self.num_samples, chunk_size):
                end_idx = min(i + chunk_size, self.num_samples)
                chunk = f['wave_fields'][i:end_idx]
                wave_values.append(chunk.flatten())
        
        all_values = np.concatenate(wave_values)
        self.wave_mean = float(np.mean(all_values))
        self.wave_std = float(np.std(all_values))
        
        print(f"  - Wave field mean: {self.wave_mean:.6f}")
        print(f"  - Wave field std: {self.wave_std:.6f}")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            wave_field: Normalized wave field tensor [1, H, W]
            coordinates: Source coordinates tensor [2]
        """
        with h5py.File(self.hdf5_path, 'r') as f:
            # Load wave field
            wave_field = f['wave_fields'][idx].astype(np.float32)
            
            # Load source coordinates
            coordinates = f['source_coords'][idx].astype(np.float32)
        
        # Normalize wave field if requested
        if self.normalize_wave_fields:
            wave_field = (wave_field - self.wave_mean) / self.wave_std
        
        # Normalize coordinates to [0, 1] if requested
        if self.normalize_coordinates:
            coordinates = coordinates / (self.grid_size - 1)
        
        # Convert to PyTorch tensors
        wave_field = torch.from_numpy(wave_field).unsqueeze(0)  # Add channel dim
        coordinates = torch.from_numpy(coordinates)
        
        return wave_field, coordinates
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """
        Get detailed information about a specific sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with sample information
        """
        with h5py.File(self.hdf5_path, 'r') as f:
            wave_field = f['wave_fields'][idx]
            coordinates = f['source_coords'][idx]
        
        return {
            'index': idx,
            'wave_field_shape': wave_field.shape,
            'wave_field_range': (wave_field.min(), wave_field.max()),
            'source_coordinates': tuple(coordinates),
            'timesteps': self.timesteps,
            'wave_speed': self.wave_speed
        }


def create_dataloaders(dataset_path: str,
                      batch_size: int = 32,
                      validation_split: float = 0.2,
                      num_workers: int = 0,
                      random_seed: int = 42,
                      **dataset_kwargs) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders from a wave dataset.
    
    Args:
        dataset_path: Path to HDF5 dataset file
        batch_size: Batch size for DataLoaders
        validation_split: Fraction of data to use for validation
        num_workers: Number of worker processes for data loading
        random_seed: Random seed for reproducible splits
        **dataset_kwargs: Additional arguments for WaveDataset
        
    Returns:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
    """
    # Create full dataset
    full_dataset = WaveDataset(dataset_path, **dataset_kwargs)
    
    # Calculate split sizes
    total_size = len(full_dataset)
    val_size = int(total_size * validation_split)
    train_size = total_size - val_size
    
    print(f"\nDataset split:")
    print(f"  - Training samples: {train_size}")
    print(f"  - Validation samples: {val_size}")
    print(f"  - Batch size: {batch_size}")
    
    # Create train/validation split
    generator = torch.Generator().manual_seed(random_seed)
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader


def create_combined_dataloaders(dataset_paths: list,
                               batch_size: int = 32,
                               validation_split: float = 0.2,
                               num_workers: int = 0,
                               random_seed: int = 42,
                               **dataset_kwargs) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders from multiple dataset files (e.g., T=250 and T=500).
    
    Args:
        dataset_paths: List of paths to HDF5 dataset files
        batch_size: Batch size for DataLoaders
        validation_split: Fraction of data to use for validation
        num_workers: Number of worker processes for data loading
        random_seed: Random seed for reproducible splits
        **dataset_kwargs: Additional arguments for WaveDataset
        
    Returns:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
    """
    from torch.utils.data import ConcatDataset
    
    # Create individual datasets
    datasets = []
    for path in dataset_paths:
        dataset = WaveDataset(path, **dataset_kwargs)
        datasets.append(dataset)
        print()  # Add spacing between dataset info
    
    # Combine datasets
    combined_dataset = ConcatDataset(datasets)
    total_size = len(combined_dataset)
    
    print(f"Combined dataset:")
    print(f"  - Total samples: {total_size}")
    
    # Calculate split sizes
    val_size = int(total_size * validation_split)
    train_size = total_size - val_size
    
    print(f"  - Training samples: {train_size}")
    print(f"  - Validation samples: {val_size}")
    print(f"  - Batch size: {batch_size}")
    
    # Create train/validation split
    generator = torch.Generator().manual_seed(random_seed)
    train_dataset, val_dataset = random_split(
        combined_dataset, [train_size, val_size], generator=generator
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader


def inspect_dataset(dataset_path: str, num_samples: int = 5) -> None:
    """
    Inspect a dataset by printing information about random samples.
    
    Args:
        dataset_path: Path to HDF5 dataset file
        num_samples: Number of samples to inspect
    """
    dataset = WaveDataset(dataset_path)
    
    print(f"\nInspecting dataset: {dataset_path}")
    print(f"Total samples: {len(dataset)}")
    
    # Get random sample indices
    indices = np.random.choice(len(dataset), size=num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        info = dataset.get_sample_info(idx)
        wave_field, coordinates = dataset[idx]
        
        print(f"\nSample {i+1} (index {idx}):")
        print(f"  - Wave field shape: {wave_field.shape}")
        print(f"  - Wave field range: [{wave_field.min():.4f}, "
              f"{wave_field.max():.4f}]")
        print(f"  - Source coordinates: ({coordinates[0]:.1f}, "
              f"{coordinates[1]:.1f})") 