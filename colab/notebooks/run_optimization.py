#!/usr/bin/env python3
"""
ResNet Optimization Batch Training for Google Colab

This script runs the complete 8-experiment optimization pipeline on GPU.
Run this after completing the Colab setup notebook.

Usage in Colab:
    !python colab/notebooks/run_optimization.py

Expected time: 6-8 hours on L4/A100 GPU
"""

import os
import sys
import yaml
import time
import torch
import mlflow
import mlflow.pytorch
from datetime import datetime
from pathlib import Path

# Add project path to imports
sys.path.append('/content/Physics-Informed-DL-Project')

from src.training.trainer import WaveTrainer
from configs.training_config import TrainingConfig, create_config_from_dict


def load_experiment_config():
    """Load experiment configuration from YAML."""
    config_path = "/content/Physics-Informed-DL-Project/colab/experiments/experiment_configs/resnet_optimization.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"✅ Loaded experiment config: {config['experiment_name']}")
    print(f"📊 Total experiments: {len(config['experiments'])}")
    
    return config


def setup_colab_environment():
    """Setup Colab environment for training."""
    print("🔧 Setting up Colab environment...")
    
    # Verify GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("❌ No GPU available! Check runtime settings.")
        return False
    
    # Setup MLflow for Colab
    mlflow_path = "/content/mlruns"
    os.makedirs(mlflow_path, exist_ok=True)
    mlflow.set_tracking_uri(f"file://{mlflow_path}")
    mlflow.set_experiment("colab_resnet_optimization")
    
    print(f"✅ MLflow tracking: {mlflow_path}")
    
    # Create results directories
    results_base = "/content/drive/MyDrive/Physics_Informed_DL_Project/results"
    os.makedirs(f"{results_base}/mlruns", exist_ok=True)
    os.makedirs(f"{results_base}/models", exist_ok=True)
    os.makedirs(f"{results_base}/plots", exist_ok=True)
    
    print(f"✅ Results will be saved to Drive: {results_base}")
    return True


def create_experiment_config(base_config, experiment_config):
    """Create training config for a specific experiment."""
    # Merge base config with experiment-specific config
    merged_config = {**base_config, **experiment_config}
    
    # Convert to TrainingConfig object
    training_config = TrainingConfig(
        dataset_name=merged_config.get('dataset_name', 'T500'),
        dataset_path=merged_config.get('dataset_path', '/content/drive/MyDrive/Physics_Informed_DL_Project/datasets/wave_dataset_T500.h5'),
        batch_size=merged_config.get('batch_size', 32),
        learning_rate=merged_config.get('learning_rate', 0.001),
        num_epochs=merged_config.get('num_epochs', 75),
        optimizer=merged_config.get('optimizer', 'adam'),
        weight_decay=merged_config.get('weight_decay', 0.0001),
        scheduler_type=merged_config.get('scheduler_type', 'plateau'),
        scheduler_patience=merged_config.get('scheduler_patience', 10),
        early_stopping_patience=merged_config.get('early_stopping_patience', 15),
        num_workers=merged_config.get('num_workers', 2),
        pin_memory=merged_config.get('pin_memory', True),
        device='cuda'
    )
    
    return training_config


def run_single_experiment(experiment_name, experiment_config, base_config, results_summary):
    """Run a single optimization experiment."""
    print(f"\n🚀 Starting Experiment: {experiment_name}")
    print(f"📝 Description: {experiment_config.get('description', 'No description')}")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Create training configuration
        training_config = create_experiment_config(base_config, experiment_config['config'])
        training_config.run_name = f"resnet_opt_{experiment_name}"
        training_config.experiment_name = "colab_resnet_optimization"
        
        # Log experiment parameters
        print(f"⚙️  Config: LR={training_config.learning_rate}, "
              f"Batch={training_config.batch_size}, "
              f"Optimizer={training_config.optimizer}")
        
        # Create and run trainer
        trainer = WaveTrainer(training_config)
        
        # Train model
        best_metrics = trainer.train()
        
        # Extract key results
        val_loss = best_metrics.get('best_val_loss', float('inf'))
        val_distance_error = best_metrics.get('val_distance_error', float('inf'))
        training_time = time.time() - start_time
        
        # Save results
        result = {
            'experiment': experiment_name,
            'val_loss': val_loss,
            'val_distance_error': val_distance_error,
            'training_time_minutes': training_time / 60,
            'config': experiment_config['config'],
            'status': 'completed'
        }
        
        results_summary.append(result)
        
        print(f"\n✅ Experiment {experiment_name} completed!")
        print(f"📊 Validation Loss: {val_loss:.3f}")
        print(f"📏 Distance Error: {val_distance_error:.2f} px")
        print(f"⏱️  Time: {training_time/60:.1f} minutes")
        
        # Save model to Drive
        model_save_path = f"/content/drive/MyDrive/Physics_Informed_DL_Project/results/models/resnet_opt_{experiment_name}_best.pth"
        local_model_path = training_config.get_model_save_path()
        if os.path.exists(local_model_path):
            import shutil
            shutil.copy2(local_model_path, model_save_path)
            print(f"💾 Model saved to Drive: {model_save_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Experiment {experiment_name} failed: {e}")
        result = {
            'experiment': experiment_name,
            'val_loss': float('inf'),
            'val_distance_error': float('inf'),
            'training_time_minutes': (time.time() - start_time) / 60,
            'config': experiment_config['config'],
            'status': 'failed',
            'error': str(e)
        }
        results_summary.append(result)
        return False


def sync_results_to_drive():
    """Sync MLflow results to Google Drive."""
    print("\n💾 Syncing results to Google Drive...")
    
    try:
        import shutil
        
        # Copy MLflow results
        local_mlruns = "/content/mlruns"
        drive_mlruns = "/content/drive/MyDrive/Physics_Informed_DL_Project/results/mlruns"
        
        if os.path.exists(local_mlruns):
            if os.path.exists(drive_mlruns):
                shutil.rmtree(drive_mlruns)
            shutil.copytree(local_mlruns, drive_mlruns)
            print(f"✅ MLflow data synced to Drive")
        
        return True
        
    except Exception as e:
        print(f"❌ Sync error: {e}")
        return False


def print_final_summary(results_summary):
    """Print final results summary."""
    print("\n" + "="*80)
    print("🎯 OPTIMIZATION COMPLETE - FINAL RESULTS")
    print("="*80)
    
    # Sort by validation distance error
    completed_results = [r for r in results_summary if r['status'] == 'completed']
    completed_results.sort(key=lambda x: x['val_distance_error'])
    
    print(f"\n📊 Completed: {len(completed_results)}/{len(results_summary)} experiments")
    print(f"⏱️  Total time: {sum(r['training_time_minutes'] for r in results_summary):.1f} minutes")
    
    if completed_results:
        print(f"\n🏆 TOP 3 RESULTS:")
        for i, result in enumerate(completed_results[:3]):
            print(f"  {i+1}. {result['experiment']}: {result['val_distance_error']:.2f}px "
                  f"(loss: {result['val_loss']:.3f})")
        
        best_result = completed_results[0]
        print(f"\n🥇 BEST MODEL: {best_result['experiment']}")
        print(f"   📏 Distance Error: {best_result['val_distance_error']:.2f} pixels")
        print(f"   📉 Validation Loss: {best_result['val_loss']:.3f}")
        print(f"   ⚙️  Config: {best_result['config']}")
        
        improvement = 2.57 - best_result['val_distance_error']  # vs baseline
        if improvement > 0:
            print(f"   🎯 Improvement: {improvement:.2f}px better than baseline!")
        
    else:
        print("\n❌ No experiments completed successfully")
    
    print(f"\n💾 All results saved to Google Drive")
    print(f"📱 Download locally: python colab/mlflow/download_results.py")


def main():
    """Main optimization pipeline."""
    print("🌊 ResNet Optimization Pipeline - Starting!")
    print(f"🕐 Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Setup environment
    if not setup_colab_environment():
        return
    
    # Load experiment configuration
    config = load_experiment_config()
    base_config = config['base_config']
    experiments = config['experiments']
    
    # Results tracking
    results_summary = []
    
    # Run experiments sequentially
    print(f"\n🎯 Running {len(experiments)} optimization experiments...")
    
    for i, experiment in enumerate(experiments, 1):
        experiment_name = experiment['name']
        
        print(f"\n📍 Progress: {i}/{len(experiments)} experiments")
        
        success = run_single_experiment(
            experiment_name, 
            experiment, 
            base_config, 
            results_summary
        )
        
        # Early termination check
        if not success:
            print(f"⚠️  Experiment {experiment_name} failed, continuing...")
        
        # Sync after each experiment
        sync_results_to_drive()
        
        print(f"\n💾 Progress auto-saved to Drive")
    
    # Final sync and summary
    sync_results_to_drive()
    print_final_summary(results_summary)
    
    print(f"\n🎉 Optimization pipeline completed!")
    print(f"🕐 End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main() 