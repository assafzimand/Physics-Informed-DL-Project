import os
import time
import torch
import numpy as np
import mlflow
import mlflow.pytorch
from pathlib import Path
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
import logging

from .trainer import WaveTrainer
from ..data.wave_dataset import WaveDataset
from configs.training_config import TrainingConfig


class CrossValidationTrainer:
    """5-Fold Cross-Validation trainer for wave source localization."""
    
    def __init__(self, config: TrainingConfig, k_folds: int = 5):
        self.config = config
        self.k_folds = k_folds
        self.random_seed = config.random_seed
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Results storage
        self.fold_results = []
        self.fold_models = []
        
        # MLflow experiment setup
        self.experiment_name = getattr(config, 'experiment_name', 'cv_wave_source_localization')
        self.run_name = getattr(config, 'run_name', f'5fold_cv_{config.learning_rate}_{config.batch_size}_{config.optimizer}')
        
        print(f"🔬 5-Fold Cross-Validation Trainer Initialized")
        print(f"📊 Folds: {k_folds}")
        print(f"🎯 Config: LR={config.learning_rate}, BS={config.batch_size}, OPT={config.optimizer}")
        
    def run_cross_validation(self) -> Dict[str, Any]:
        """Run complete 5-fold cross-validation."""
        print(f"\n🚀 Starting 5-Fold Cross-Validation")
        print(f"⏱️ Expected time: ~{self.config.num_epochs * self.k_folds * 2} minutes")
        print("=" * 60)
        
        start_time = time.time()
        
        # Setup MLflow experiment
        self._setup_mlflow()
        
        # Load full dataset for splitting
        full_dataset = self._load_full_dataset()
        
        # Create K-fold splits
        kfold = KFold(n_splits=self.k_folds, shuffle=True, random_state=self.random_seed)
        
        with mlflow.start_run(run_name=self.run_name):
            # Log CV configuration
            mlflow.log_params({
                'cv_folds': self.k_folds,
                'cv_method': 'k_fold',
                'random_seed': self.random_seed,
                'learning_rate': self.config.learning_rate,
                'batch_size': self.config.batch_size,
                'optimizer': self.config.optimizer,
                'num_epochs': self.config.num_epochs,
                'early_stopping_patience': self.config.early_stopping_patience
            })
            
            # Train each fold
            for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(full_dataset)):
                print(f"\n📁 Training Fold {fold_idx + 1}/{self.k_folds}")
                print(f"📊 Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")
                
                fold_result = self._train_single_fold(
                    fold_idx, full_dataset, train_indices, val_indices
                )
                self.fold_results.append(fold_result)
                
                # Log fold results to MLflow
                mlflow.log_metrics({
                    f'fold_{fold_idx + 1}_val_loss': fold_result['val_loss'],
                    f'fold_{fold_idx + 1}_distance_error': fold_result['distance_error'],
                    f'fold_{fold_idx + 1}_training_time': fold_result['training_time']
                })
                
                print(f"✅ Fold {fold_idx + 1} completed: {fold_result['distance_error']:.2f}px, {fold_result['val_loss']:.4f} loss")
            
            # Calculate and log statistics
            cv_stats = self._calculate_cv_statistics()
            self._log_cv_results(cv_stats)
            
            total_time = time.time() - start_time
            mlflow.log_metric('total_cv_time_minutes', total_time / 60)
            
            print(f"\n🎉 5-Fold CV Complete! Total time: {total_time/60:.1f} minutes")
            return cv_stats
            
    def _setup_mlflow(self):
        """Setup MLflow experiment."""
        mlflow.set_experiment(self.experiment_name)
        
    def _load_full_dataset(self) -> WaveDataset:
        """Load the complete dataset for CV splitting."""
        print("📊 Loading full dataset for cross-validation...")
        
        dataset = WaveDataset(
            dataset_path=self.config.dataset_path,
            train_split=1.0,  # Use full dataset
            val_split=0.0,
            test_split=0.0,
            random_seed=self.random_seed,
            normalize_wave_fields=True,
            device=self.config.device
        )
        
        print(f"✅ Loaded {len(dataset)} samples for CV")
        return dataset
        
    def _train_single_fold(self, fold_idx: int, full_dataset: WaveDataset, 
                          train_indices: np.ndarray, val_indices: np.ndarray) -> Dict[str, float]:
        """Train a single fold."""
        fold_start_time = time.time()
        
        # Create data subsets
        train_subset = Subset(full_dataset, train_indices)
        val_subset = Subset(full_dataset, val_indices)
        
        # Create data loaders
        train_loader = DataLoader(
            train_subset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        # Create fold-specific config
        fold_config = self._create_fold_config(fold_idx)
        
        # Create trainer for this fold
        fold_trainer = WaveTrainer(fold_config)
        fold_trainer.train_loader = train_loader
        fold_trainer.val_loader = val_loader
        
        # Train the fold
        metrics = fold_trainer.train()
        
        # Store model path
        model_path = fold_trainer.best_model_path
        self.fold_models.append(model_path)
        
        fold_time = time.time() - fold_start_time
        
        return {
            'fold_idx': fold_idx,
            'val_loss': metrics['val_loss'][-1],
            'distance_error': metrics['val_distance_error'][-1],
            'training_time': fold_time / 60,
            'model_path': model_path,
            'metrics_history': metrics
        }
        
    def _create_fold_config(self, fold_idx: int) -> TrainingConfig:
        """Create configuration for a specific fold."""
        # Convert config to dict for modification
        config_dict = {
            'dataset_path': self.config.dataset_path,
            'train_split': 0.8,  # Will be overridden by custom loaders
            'val_split': 0.2,
            'test_split': 0.0,
            'random_seed': self.config.random_seed,
            'num_epochs': self.config.num_epochs,
            'device': self.config.device,
            'learning_rate': self.config.learning_rate,
            'batch_size': self.config.batch_size,
            'optimizer': self.config.optimizer,
            'weight_decay': self.config.weight_decay,
            'scheduler_type': self.config.scheduler_type,
            'scheduler_patience': self.config.scheduler_patience,
            'early_stopping_patience': self.config.early_stopping_patience,
            'save_model_every_n_epochs': self.config.save_model_every_n_epochs,
            'model_name': self.config.model_name,
            'grid_size': self.config.grid_size,
            'num_workers': self.config.num_workers,
            'pin_memory': self.config.pin_memory,
            'experiment_name': f"{self.experiment_name}_fold_{fold_idx + 1}",
            'run_name': f"{self.run_name}_fold_{fold_idx + 1}"
        }
        
        return TrainingConfig(**config_dict)
        
    def _calculate_cv_statistics(self) -> Dict[str, Any]:
        """Calculate cross-validation statistics."""
        val_losses = [result['val_loss'] for result in self.fold_results]
        distance_errors = [result['distance_error'] for result in self.fold_results]
        training_times = [result['training_time'] for result in self.fold_results]
        
        stats = {
            # Distance Error Statistics
            'distance_error_mean': np.mean(distance_errors),
            'distance_error_std': np.std(distance_errors),
            'distance_error_min': np.min(distance_errors),
            'distance_error_max': np.max(distance_errors),
            'distance_error_values': distance_errors,
            
            # Validation Loss Statistics  
            'val_loss_mean': np.mean(val_losses),
            'val_loss_std': np.std(val_losses),
            'val_loss_min': np.min(val_losses),
            'val_loss_max': np.max(val_losses),
            'val_loss_values': val_losses,
            
            # Training Time Statistics
            'training_time_mean': np.mean(training_times),
            'training_time_std': np.std(training_times),
            'training_time_total': np.sum(training_times),
            
            # Model Paths
            'fold_models': self.fold_models,
            
            # Individual Fold Results
            'fold_results': self.fold_results
        }
        
        return stats
        
    def _log_cv_results(self, stats: Dict[str, Any]):
        """Log cross-validation results to MLflow."""
        # Log summary statistics
        mlflow.log_metrics({
            'cv_distance_error_mean': stats['distance_error_mean'],
            'cv_distance_error_std': stats['distance_error_std'],
            'cv_distance_error_min': stats['distance_error_min'],
            'cv_distance_error_max': stats['distance_error_max'],
            'cv_val_loss_mean': stats['val_loss_mean'],
            'cv_val_loss_std': stats['val_loss_std'],
            'cv_training_time_total': stats['training_time_total']
        })
        
        # Log model artifacts
        for i, model_path in enumerate(stats['fold_models']):
            if os.path.exists(model_path):
                mlflow.log_artifact(model_path, f"fold_{i+1}_model")
                
    def get_ensemble_prediction(self, wave_field: np.ndarray) -> Tuple[float, float]:
        """Make ensemble prediction using all fold models."""
        predictions = []
        
        for model_path in self.fold_models:
            if os.path.exists(model_path):
                # Load model and make prediction
                # This would need to be implemented based on your inference pipeline
                pass
                
        # Average predictions
        if predictions:
            mean_x = np.mean([pred[0] for pred in predictions])
            mean_y = np.mean([pred[1] for pred in predictions])
            return mean_x, mean_y
        else:
            raise ValueError("No models available for ensemble prediction")
            
    def print_cv_summary(self, stats: Dict[str, Any]):
        """Print comprehensive CV summary."""
        print("\n" + "="*80)
        print("📊 5-FOLD CROSS-VALIDATION RESULTS SUMMARY")
        print("="*80)
        
        print(f"\n🎯 DISTANCE ERROR PERFORMANCE:")
        print(f"   Mean: {stats['distance_error_mean']:.3f} ± {stats['distance_error_std']:.3f} px")
        print(f"   Best: {stats['distance_error_min']:.3f} px")
        print(f"   Worst: {stats['distance_error_max']:.3f} px")
        print(f"   Individual: {[f'{x:.2f}' for x in stats['distance_error_values']]}")
        
        print(f"\n📉 VALIDATION LOSS:")
        print(f"   Mean: {stats['val_loss_mean']:.4f} ± {stats['val_loss_std']:.4f}")
        print(f"   Best: {stats['val_loss_min']:.4f}")
        print(f"   Worst: {stats['val_loss_max']:.4f}")
        
        print(f"\n⏱️ TRAINING TIME:")
        print(f"   Total: {stats['training_time_total']:.1f} minutes")
        print(f"   Per fold: {stats['training_time_mean']:.1f} ± {stats['training_time_std']:.1f} minutes")
        
        print(f"\n🏆 ACADEMIC REPORTING:")
        print(f"   Distance Error: {stats['distance_error_mean']:.2f} ± {stats['distance_error_std']:.2f} px")
        print(f"   Validation Loss: {stats['val_loss_mean']:.4f} ± {stats['val_loss_std']:.4f}")
        
        print("\n" + "="*80) 