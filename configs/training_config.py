"""
Training Configuration for Wave Source Localization

Centralized configuration for training experiments with hyperparameter options.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class TrainingConfig:
    """Configuration for training experiments."""
    
    # Dataset Configuration
    dataset_name: str = "T500"  # T250 or T500
    dataset_path: str = "data/wave_dataset_T500.h5"
    
    # Data Split Configuration  
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    random_seed: int = 42
    
    # Model Configuration
    model_name: str = "WaveSourceMiniResNet"
    grid_size: int = 128
    
    # Training Hyperparameters
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 100
    optimizer: str = "adam"  # adam, sgd, adamw
    weight_decay: float = 1e-4
    
    # Learning Rate Scheduling
    use_scheduler: bool = True
    scheduler_type: str = "plateau"  # plateau, step, cosine
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    
    # Early Stopping
    use_early_stopping: bool = True
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 0.001
    
    # Loss Function
    loss_function: str = "mse"  # mse, huber, smooth_l1
    
    # Data Loading
    num_workers: int = 0
    pin_memory: bool = True
    
    # Experiment Tracking
    experiment_name: str = "wave_source_localization"
    run_name: Optional[str] = None
    mlflow_tracking_uri: str = "mlruns"
    save_model_every_n_epochs: int = 10
    
    # Output Directories
    output_dir: str = "experiments"
    model_save_dir: str = "models"
    logs_dir: str = "logs"
    plots_dir: str = "plots"
    
    # Hardware
    device: str = "auto"  # auto, cpu, cuda
    
    # Validation and Testing
    validate_every_n_epochs: int = 1
    save_predictions: bool = True
    visualize_predictions: bool = True
    
    def __post_init__(self):
        """Post-initialization setup."""
        # Auto-detect device
        if self.device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create output directories
        for dir_path in [self.output_dir, self.model_save_dir, 
                        self.logs_dir, self.plots_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Generate run name if not provided
        if self.run_name is None:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_name = f"{self.model_name}_{self.dataset_name}_{timestamp}"
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        # Check splits sum to 1.0
        total_split = self.train_split + self.val_split + self.test_split
        if abs(total_split - 1.0) > 1e-6:
            raise ValueError(f"Train/val/test splits must sum to 1.0, got {total_split}")
        
        # Check dataset exists
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        
        # Validate optimizer
        valid_optimizers = ["adam", "sgd", "adamw"]
        if self.optimizer.lower() not in valid_optimizers:
            raise ValueError(f"Optimizer must be one of {valid_optimizers}")
        
        # Validate scheduler
        valid_schedulers = ["plateau", "step", "cosine"]
        if self.scheduler_type.lower() not in valid_schedulers:
            raise ValueError(f"Scheduler must be one of {valid_schedulers}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for logging."""
        return {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_')
        }
    
    def get_model_save_path(self, epoch: Optional[int] = None) -> str:
        """Get path for saving model."""
        if epoch is not None:
            filename = f"{self.run_name}_epoch_{epoch:03d}.pth"
        else:
            filename = f"{self.run_name}_best.pth"
        return os.path.join(self.model_save_dir, filename)
    
    def get_log_path(self) -> str:
        """Get path for saving training logs."""
        filename = f"{self.run_name}_training.log"
        return os.path.join(self.logs_dir, filename)


# Hyperparameter Grid Search Configurations
HYPERPARAMETER_GRIDS = {
    "quick_search": {
        "learning_rate": [0.001, 0.0001],
        "batch_size": [16, 32],
        "weight_decay": [1e-4, 1e-5]
    },
    
    "comprehensive_search": {
        "learning_rate": [0.01, 0.001, 0.0001, 0.00001],
        "batch_size": [8, 16, 32, 64],
        "weight_decay": [1e-3, 1e-4, 1e-5, 0],
        "optimizer": ["adam", "adamw"],
        "scheduler_factor": [0.5, 0.3, 0.1]
    },
    
    "learning_rate_search": {
        "learning_rate": [0.1, 0.01, 0.001, 0.0001, 0.00001]
    },
    
    "batch_size_search": {
        "batch_size": [4, 8, 16, 32, 64, 128]
    }
}


def create_config_from_dict(base_config: TrainingConfig, 
                           override_dict: Dict[str, Any]) -> TrainingConfig:
    """Create a new config by overriding base config parameters."""
    import copy
    config_dict = copy.deepcopy(base_config.__dict__)
    config_dict.update(override_dict)
    
    # Remove private attributes
    config_dict = {k: v for k, v in config_dict.items() if not k.startswith('_')}
    
    return TrainingConfig(**config_dict)


def get_default_config(dataset_name: str = "T500") -> TrainingConfig:
    """Get default training configuration for specified dataset."""
    dataset_paths = {
        "T250": "data/wave_dataset_T250.h5",
        "T500": "data/wave_dataset_T500.h5"
    }
    
    if dataset_name not in dataset_paths:
        raise ValueError(f"Dataset must be one of {list(dataset_paths.keys())}")
    
    return TrainingConfig(
        dataset_name=dataset_name,
        dataset_path=dataset_paths[dataset_name]
    ) 