"""
PyTorch Dataset for Wave Source Localization

Custom dataset class for loading wave simulation data from HDF5 files.
Supports optional normalization and convenient DataLoader creation.
"""

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Tuple, Dict, Any, List


class WaveDataset(Dataset):
    """
    PyTorch Dataset for wave source localization data.

    Loads wave fields and source coordinates from HDF5 files.
    Optionally normalizes wave amplitudes and/or coordinates.
    """

    def __init__(
        self,
        hdf5_path: str,
        normalize_wave_fields: bool = True,
        normalize_coordinates: bool = False,
        grid_size: int = 128,
    ) -> None:
        """
        Initializes the WaveDataset.

        Args:
            hdf5_path: Path to HDF5 file containing wave data.
            normalize_wave_fields: Whether to normalize wave amplitudes.
            normalize_coordinates: Whether to normalize coordinates to [0,1].
            grid_size: Size of the wave field grid (default 128).
        """
        self.hdf5_path: str = hdf5_path
        self.normalize_wave_fields: bool = normalize_wave_fields
        self.normalize_coordinates: bool = normalize_coordinates
        self.grid_size: int = grid_size

        # Attributes populated by _load_dataset_info/_compute_statistics
        self.num_samples: int = 0
        self.timesteps: Any = "unknown"
        self.wave_speed: Any = "unknown"
        self.wave_mean: float = 0.0
        self.wave_std: float = 1.0

        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

        self._load_dataset_info()
        self._compute_statistics()

    def _load_dataset_info(self) -> None:
        """Loads basic dataset information from HDF5 attributes and shapes."""
        with h5py.File(self.hdf5_path, "r") as f:
            if "wave_fields" not in f:
                raise KeyError("HDF5 missing 'wave_fields' dataset")
            self.num_samples = int(f["wave_fields"].shape[0])
            self.timesteps = f.attrs.get("timesteps", "unknown")
            self.wave_speed = f.attrs.get("wave_speed", "unknown")

            print(f"Loaded dataset: {self.hdf5_path}")
            print(f"  - Samples: {self.num_samples}")
            print(f"  - Timesteps: {self.timesteps}")
            print(f"  - Wave speed: {self.wave_speed}")

    def _compute_statistics(self) -> None:
        """Computes normalization statistics for wave fields if enabled."""
        if not self.normalize_wave_fields:
            self.wave_mean = 0.0
            self.wave_std = 1.0
            return

        print("Computing normalization statistics...")
        chunk_size = min(100, self.num_samples)
        wave_values: List[np.ndarray] = []

        with h5py.File(self.hdf5_path, "r") as f:
            for i in range(0, self.num_samples, chunk_size):
                end_idx = min(i + chunk_size, self.num_samples)
                chunk = f["wave_fields"][i:end_idx]
                wave_values.append(chunk.reshape(-1))

        all_values = np.concatenate(wave_values)
        self.wave_mean = float(np.mean(all_values))
        self.wave_std = float(np.std(all_values) if np.std(all_values) > 0 else 1.0)

        print(f"  - Wave field mean: {self.wave_mean:.6f}")
        print(f"  - Wave field std: {self.wave_std:.6f}")

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a single sample from the dataset.

        Args:
            idx: Sample index.

        Returns:
            wave_field: Normalized wave field tensor [1, H, W].
            coordinates: Source coordinates tensor [2].
        """
        with h5py.File(self.hdf5_path, "r") as f:
            wave_field = f["wave_fields"][idx].astype(np.float32)

            # Robust coordinate key handling ('source_coords' expected)
            coords_ds = "source_coords" if "source_coords" in f else "coordinates"
            coordinates = f[coords_ds][idx].astype(np.float32)

        if self.normalize_wave_fields:
            wave_field = (wave_field - self.wave_mean) / self.wave_std

        if self.normalize_coordinates:
            coordinates = coordinates / (self.grid_size - 1)

        wave_field_t = torch.from_numpy(wave_field).unsqueeze(0)  # [1, H, W]
        coordinates_t = torch.from_numpy(coordinates)

        return wave_field_t, coordinates_t

    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """
        Gets detailed information about a specific sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with sample information such as shapes, ranges, and metadata.
        """
        with h5py.File(self.hdf5_path, "r") as f:
            wave_field = f["wave_fields"][idx]
            coords_ds = "source_coords" if "source_coords" in f else "coordinates"
            coordinates = f[coords_ds][idx]

        return {
            "index": idx,
            "wave_field_shape": wave_field.shape,
            "wave_field_range": (float(np.min(wave_field)), float(np.max(wave_field))),
            "source_coordinates": tuple(float(v) for v in coordinates),
            "timesteps": self.timesteps,
            "wave_speed": self.wave_speed,
        }


def create_dataloaders(
    dataset_path: str,
    batch_size: int = 32,
    validation_split: float = 0.2,
    num_workers: int = 0,
    random_seed: int = 42,
    **dataset_kwargs,
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates train and validation DataLoaders from a wave dataset.

    Args:
        dataset_path: Path to HDF5 dataset file.
        batch_size: Batch size for DataLoaders.
        validation_split: Fraction of data to use for validation.
        num_workers: Number of worker processes for data loading.
        random_seed: Random seed for reproducible splits.
        **dataset_kwargs: Additional arguments forwarded to WaveDataset.

    Returns:
        A tuple of (train_loader, val_loader).
    """
    full_dataset = WaveDataset(dataset_path, **dataset_kwargs)

    total_size = len(full_dataset)
    val_size = int(total_size * validation_split)
    train_size = total_size - val_size

    print("\nDataset split:")
    print(f"  - Training samples: {train_size}")
    print(f"  - Validation samples: {val_size}")
    print(f"  - Batch size: {batch_size}")

    generator = torch.Generator().manual_seed(random_seed)
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


def inspect_dataset(dataset_path: str, num_samples: int = 5) -> None:
    """
    Inspects a dataset by printing information about randomly selected samples.
    """
    dataset = WaveDataset(dataset_path)

    print(f"\nInspecting dataset: {dataset_path}")
    print(f"Total samples: {len(dataset)}")

    indices = np.random.choice(len(dataset), size=num_samples, replace=False)

    for i, idx in enumerate(indices):
        info = dataset.get_sample_info(idx)
        wave_field, coordinates = dataset[idx]

        print(f"\nSample {i+1} (index {idx}):")
        print(f"  - Wave field shape: {wave_field.shape}")
        print(f"  - Wave field range: [{wave_field.min():.4f}, {wave_field.max():.4f}]")
        print(f"  - Source coordinates: ({coordinates[0]:.1f}, {coordinates[1]:.1f})")
