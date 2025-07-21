"""
Neural Network Trainer for Wave Source Localization

Comprehensive training pipeline with MLflow experiment tracking.
"""

import os
import sys
import logging
import time
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import mlflow
import mlflow.pytorch

# Add parent directory for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.models.wave_source_resnet import create_wave_source_model
from src.data.wave_dataset import WaveDataset
from configs.training_config import TrainingConfig


class WaveTrainer:
    """
    Comprehensive trainer for wave source localization models.
    
    Features:
    - MLflow experiment tracking
    - Automatic model saving
    - Learning rate scheduling
    - Early stopping
    - Comprehensive logging
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize MLflow
        self._setup_mlflow()
        
        # Create model and data
        self.model = self._create_model()
        self.train_loader, self.val_loader, self.test_loader = self._create_dataloaders()
        
        # Setup training components
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.criterion = self._create_loss_function()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_distance_error': [],
            'val_distance_error': [],
            'learning_rate': []
        }
        
        # Early stopping
        self.early_stopping_counter = 0
        
        self.logger.info("🚀 Trainer initialized successfully!")
        self.logger.info(f"🖥️ Device: {self.device}")
        if self.device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_memory_gb = gpu_memory / 1024**3
            self.logger.info(f"🎮 GPU: {gpu_name} ({gpu_memory_gb:.1f} GB)")
        else:
            self.logger.info("💻 Using CPU training")
        self.logger.info(f"🧠 Model parameters: {self.model.get_num_parameters():,}")
        
    def _setup_logging(self):
        """Setup comprehensive logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.get_log_path()),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _setup_mlflow(self):
        """Setup MLflow experiment tracking."""
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        mlflow.set_experiment(self.config.experiment_name)
        
        # Start MLflow run
        mlflow.start_run(run_name=self.config.run_name)
        
        # Log configuration
        mlflow.log_params(self.config.to_dict())
        
        self.logger.info(f"📈 MLflow experiment: {self.config.experiment_name}")
        self.logger.info(f"📊 MLflow run: {self.config.run_name}")
        
    def _create_model(self):
        """Create and initialize model."""
        model = create_wave_source_model(grid_size=self.config.grid_size)
        model = model.to(self.device)
        
        # Log model architecture
        mlflow.log_param("model_architecture", "WaveSourceMiniResNet")
        mlflow.log_param("model_parameters", model.get_num_parameters())
        
        return model
        
    def _create_dataloaders(self):
        """Create train/validation/test data loaders."""
        self.logger.info("📂 Creating data loaders...")
        
        # Load full dataset
        full_dataset = WaveDataset(
            self.config.dataset_path,
            normalize_wave_fields=True,
            normalize_coordinates=False,
            grid_size=self.config.grid_size
        )
        
        # Calculate split sizes
        total_size = len(full_dataset)
        train_size = int(total_size * self.config.train_split)
        val_size = int(total_size * self.config.val_split)
        test_size = total_size - train_size - val_size
        
        self.logger.info(f"📊 Dataset splits:")
        self.logger.info(f"  - Training: {train_size} samples")
        self.logger.info(f"  - Validation: {val_size} samples")
        self.logger.info(f"  - Test: {test_size} samples")
        
        # Create splits
        generator = torch.Generator().manual_seed(self.config.random_seed)
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size], generator=generator
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        # Log data information
        mlflow.log_param("train_samples", train_size)
        mlflow.log_param("val_samples", val_size)
        mlflow.log_param("test_samples", test_size)
        
        return train_loader, val_loader, test_loader
        
    def _create_optimizer(self):
        """Create optimizer based on configuration."""
        if self.config.optimizer.lower() == "adam":
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "adamw":
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "sgd":
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
            
        self.logger.info(f"⚙️ Optimizer: {self.config.optimizer}")
        return optimizer
        
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        if not self.config.use_scheduler:
            return None
            
        if self.config.scheduler_type.lower() == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.scheduler_factor,
                patience=self.config.scheduler_patience
            )
        elif self.config.scheduler_type.lower() == "step":
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=20,
                gamma=self.config.scheduler_factor
            )
        elif self.config.scheduler_type.lower() == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.config.scheduler_type}")
            
        self.logger.info(f"📈 Scheduler: {self.config.scheduler_type}")
        return scheduler
        
    def _create_loss_function(self):
        """Create loss function."""
        if self.config.loss_function.lower() == "mse":
            criterion = nn.MSELoss()
        elif self.config.loss_function.lower() == "huber":
            criterion = nn.HuberLoss()
        elif self.config.loss_function.lower() == "smooth_l1":
            criterion = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss function: {self.config.loss_function}")
            
        self.logger.info(f"🎯 Loss function: {self.config.loss_function}")
        return criterion
        
    def calculate_distance_error(self, predictions: torch.Tensor, 
                                targets: torch.Tensor) -> float:
        """Calculate mean Euclidean distance error."""
        distances = torch.sqrt(
            (predictions[:, 0] - targets[:, 0]) ** 2 + 
            (predictions[:, 1] - targets[:, 1]) ** 2
        )
        return distances.mean().item()
        
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_distance_error = 0.0
        
        # Progress bar for training batches
        train_pbar = tqdm(self.train_loader, 
                         desc=f"Epoch {self.current_epoch} [Train]",
                         leave=False)
        
        for wave_fields, targets in train_pbar:
            wave_fields = wave_fields.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(wave_fields)
            loss = self.criterion(predictions, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Calculate metrics
            total_loss += loss.item()
            distance_error = self.calculate_distance_error(predictions, targets)
            total_distance_error += distance_error
            
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Dist_Err': f'{distance_error:.1f}px'
            })
        
        num_batches = len(self.train_loader)
        avg_loss = total_loss / num_batches
        avg_distance_error = total_distance_error / num_batches
        
        return avg_loss, avg_distance_error
        
    def validate_epoch(self) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        total_distance_error = 0.0
        
        # Progress bar for validation batches
        val_pbar = tqdm(self.val_loader, 
                       desc=f"Epoch {self.current_epoch} [Valid]",
                       leave=False)
        
        with torch.no_grad():
            for wave_fields, targets in val_pbar:
                wave_fields = wave_fields.to(self.device)
                targets = targets.to(self.device)
                
                predictions = self.model(wave_fields)
                loss = self.criterion(predictions, targets)
                
                total_loss += loss.item()
                distance_error = self.calculate_distance_error(predictions, targets)
                total_distance_error += distance_error
                
                # Update progress bar
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Dist_Err': f'{distance_error:.1f}px'
                })
        
        num_batches = len(self.val_loader)
        avg_loss = total_loss / num_batches
        avg_distance_error = total_distance_error / num_batches
        
        return avg_loss, avg_distance_error
        
    def save_model(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config.to_dict(),
            'training_history': self.training_history
        }
        
        # Save regular checkpoint
        if epoch % self.config.save_model_every_n_epochs == 0:
            path = self.config.get_model_save_path(epoch)
            torch.save(checkpoint, path)
            self.logger.info(f"💾 Model saved: {path}")
        
        # Save best model
        if is_best:
            path = self.config.get_model_save_path()
            torch.save(checkpoint, path)
            self.logger.info(f"⭐ Best model saved: {path}")
            
            # Log model to MLflow
            mlflow.pytorch.log_model(self.model, "best_model")
            
    def train(self):
        """Main training loop."""
        self.logger.info("🏋️ Starting training...")
        self.logger.info(f"📊 Training for {self.config.num_epochs} epochs")
        
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch + 1
            
            # Training
            train_loss, train_distance_error = self.train_epoch()
            
            # Validation
            if self.current_epoch % self.config.validate_every_n_epochs == 0:
                val_loss, val_distance_error = self.validate_epoch()
            else:
                val_loss = val_distance_error = 0.0
            
            # Update learning rate
            if self.scheduler:
                if self.config.scheduler_type.lower() == "plateau":
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Record history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_distance_error'].append(train_distance_error)
            self.training_history['val_distance_error'].append(val_distance_error)
            current_lr = self.optimizer.param_groups[0]['lr']
            self.training_history['learning_rate'].append(current_lr)
            
            # Log to MLflow
            mlflow.log_metrics({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_distance_error': train_distance_error,
                'val_distance_error': val_distance_error,
                'learning_rate': current_lr
            }, step=self.current_epoch)
            
            # Check for best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
            
            # Save model
            self.save_model(self.current_epoch, is_best)
            
            # Log epoch results
            self.logger.info(f"📊 Epoch {self.current_epoch}/{self.config.num_epochs}:")
            self.logger.info(f"  Train Loss: {train_loss:.6f}, Distance Error: {train_distance_error:.2f} pixels")
            if val_loss > 0:
                self.logger.info(f"  Val Loss: {val_loss:.6f}, Distance Error: {val_distance_error:.2f} pixels")
            self.logger.info(f"  Learning Rate: {current_lr:.8f}")
            if is_best:
                self.logger.info("  ⭐ New best model!")
            
            # Early stopping
            if (self.config.use_early_stopping and 
                self.early_stopping_counter >= self.config.early_stopping_patience):
                self.logger.info(f"🛑 Early stopping triggered after {self.current_epoch} epochs")
                break
        
        training_time = time.time() - start_time
        self.logger.info(f"✅ Training completed in {training_time:.2f} seconds")
        
        # Log final metrics
        mlflow.log_metric("final_train_loss", self.training_history['train_loss'][-1])
        mlflow.log_metric("final_val_loss", self.training_history['val_loss'][-1])
        mlflow.log_metric("best_val_loss", self.best_val_loss)
        mlflow.log_metric("training_time", training_time)
        mlflow.log_metric("epochs_trained", self.current_epoch)
        
        # End MLflow run
        mlflow.end_run()
        
        return self.training_history


def train_model(config: TrainingConfig) -> Dict:
    """
    Train a model with the given configuration.
    
    Args:
        config: Training configuration
        
    Returns:
        Training history dictionary
    """
    trainer = WaveTrainer(config)
    history = trainer.train()
    return history 