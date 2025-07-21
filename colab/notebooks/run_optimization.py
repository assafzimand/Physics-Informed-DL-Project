#!/usr/bin/env python3
"""
ResNet Grid Search Optimization - Phase 1
2x2x2 = 8 experiments to find best hyperparameter combination

Usage in Colab:
    !python colab/notebooks/run_optimization.py
"""

import os
import sys
import yaml
import time
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add project root to path
project_root = Path("/content/Physics-Informed-DL-Project")
sys.path.append(str(project_root))

# Import our modules
from configs.training_config import TrainingConfig
from src.training.trainer import WaveTrainer
import mlflow


def print_banner():
    print("🔬 ResNet Grid Search - Phase 1")
    print("=" * 50)
    print(f"🕐 Start: {datetime.now().strftime('%H:%M:%S')}")
    print("⚡ 2×2×2 Grid Search: 8 Experiments")
    print("🎯 Goal: Find optimal hyperparameters")
    print("=" * 50)


def load_experiment_config(config_path: str) -> Dict[str, Any]:
    """Load YAML experiment configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_experiment_config(exp_data: Dict[str, Any], 
                             base_config: Dict[str, Any], 
                             dataset_path: str) -> TrainingConfig:
    """Create TrainingConfig for a specific experiment"""
    
    # Merge base config with experiment-specific config
    config_dict = {
        # Dataset
        "dataset_path": dataset_path,
        "train_split": 0.7,  # Use 85% for train+val, then split 70/15
        "val_split": 0.15,
        "test_split": 0.15,
        "random_seed": 42,
        
        # From base config
        "num_epochs": base_config["num_epochs"],
        "device": base_config["device"],
        "save_model_every_n_epochs": 10,
        "early_stopping_patience": base_config["early_stopping_patience"],
        "weight_decay": base_config["weight_decay"],
        "scheduler_type": base_config["scheduler_type"],
        
        # From experiment
        "learning_rate": exp_data["learning_rate"],
        "batch_size": exp_data["batch_size"], 
        "optimizer": exp_data["optimizer"],
        
        # Model architecture
        "model_name": "WaveSourceMiniResNet",
        "input_channels": base_config["model_config"]["input_channels"],
        "hidden_channels": base_config["model_config"]["hidden_channels"],
        "kernel_sizes": base_config["model_config"]["kernel_sizes"],
        "num_residual_blocks": base_config["model_config"]["num_residual_blocks"],
        "dropout_rate": base_config["model_config"]["dropout_rate"],
        
        # Data loading
        "num_workers": 2,
        "pin_memory": True,
        
        # Scheduler params
        "scheduler_patience": 8
    }
    
    return TrainingConfig(**config_dict)

def run_single_experiment(exp_name: str, exp_data: Dict[str, Any], base_config: Dict[str, Any], dataset_path: str) -> Dict[str, float]:
    """Run a single experiment and return results"""
    
    print(f"\n🚀 Starting Experiment: {exp_name}")
    print(f"   📋 LR: {exp_data['learning_rate']}, BS: {exp_data['batch_size']}, OPT: {exp_data['optimizer']}")
    
    start_time = time.time()
    
    try:
        # Create config
        config = create_experiment_config(exp_data, base_config, dataset_path)
        
        # Start MLflow run
        with mlflow.start_run(run_name=exp_name):
            # Log experiment parameters
            mlflow.log_params({
                "learning_rate": exp_data["learning_rate"],
                "batch_size": exp_data["batch_size"],
                "optimizer": exp_data["optimizer"],
                "num_epochs": base_config["num_epochs"],
                "weight_decay": base_config["weight_decay"],
                "scheduler_type": base_config["scheduler_type"]
            })
            
            # Train model
            trainer = WaveTrainer(config)
            metrics = trainer.train()
            
            # Extract final metrics (get last values from lists)
            final_val_loss = metrics['val_loss'][-1]
            final_distance_error = metrics['val_distance_error'][-1]
            final_train_loss = metrics['train_loss'][-1]
            
            # Log final metrics
            mlflow.log_metrics({
                "final_val_loss": final_val_loss,
                "final_distance_error": final_distance_error,
                "final_train_loss": final_train_loss
            })
            
            elapsed = time.time() - start_time
            print(f"   ✅ Completed in {elapsed/60:.1f} minutes")
            print(f"   📊 Val Loss: {final_val_loss:.4f}, Distance Error: {final_distance_error:.2f} px")
            
            return {
                "val_loss": final_val_loss,
                "distance_error": final_distance_error,
                "train_loss": final_train_loss,
                "duration_minutes": elapsed / 60
            }
            
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"   ❌ FAILED after {elapsed/60:.1f} minutes: {str(e)}")
        return {
            "val_loss": float('inf'),
            "distance_error": float('inf'),
            "train_loss": float('inf'),
            "duration_minutes": elapsed / 60,
            "error": str(e)
        }

def sync_to_drive():
    """Sync MLflow results to Google Drive"""
    print("\n💾 Syncing results to Google Drive...")
    try:
        # Create drive folder if needed
        drive_path = "/content/drive/MyDrive/Physics_Informed_DL_Project/mlruns"
        os.makedirs(drive_path, exist_ok=True)
        
        # Copy MLflow data
        os.system(f"cp -r mlruns/* {drive_path}/")
        print("✅ Results synced to Drive")
    except Exception as e:
        print(f"❌ Sync failed: {e}")

def print_final_summary(results: List[Dict[str, Any]]):
    """Print experiment summary and find best model"""
    print("\n" + "=" * 60)
    print("🏆 GRID SEARCH RESULTS SUMMARY")
    print("=" * 60)
    
    # Sort by distance error
    valid_results = [r for r in results if 'error' not in r]
    if not valid_results:
        print("❌ No successful experiments!")
        return
        
    sorted_results = sorted(valid_results, key=lambda x: x['distance_error'])
    
    print(f"{'Rank':<4} {'Experiment':<25} {'Dist Error':<12} {'Val Loss':<10} {'Duration':<10}")
    print("-" * 70)
    
    for i, result in enumerate(sorted_results[:8]):  # Show all 8
        exp_name = result['experiment_name']
        dist_err = result['distance_error']
        val_loss = result['val_loss']
        duration = result['duration_minutes']
        
        print(f"{i+1:<4} {exp_name:<25} {dist_err:>8.2f} px {val_loss:>8.4f} {duration:>7.1f} min")
    
    # Highlight best result
    best = sorted_results[0]
    print(f"\n🥇 BEST MODEL: {best['experiment_name']}")
    print(f"   🎯 Distance Error: {best['distance_error']:.2f} px")
    print(f"   📉 Validation Loss: {best['val_loss']:.4f}")
    print(f"   ⏱️ Training Time: {best['duration_minutes']:.1f} minutes")
    
    # Calculate averages
    avg_dist = sum(r['distance_error'] for r in valid_results) / len(valid_results)
    avg_loss = sum(r['val_loss'] for r in valid_results) / len(valid_results)
    total_time = sum(r['duration_minutes'] for r in results)
    
    print(f"\n📊 GRID STATISTICS:")
    print(f"   Average Distance Error: {avg_dist:.2f} px")
    print(f"   Average Validation Loss: {avg_loss:.4f}")
    print(f"   Total Time: {total_time/60:.1f} hours")
    print(f"   Successful Experiments: {len(valid_results)}/8")

def main():
    """Main optimization pipeline"""
    print_banner()
    
    # Setup
    config_path = "colab/experiments/experiment_configs/resnet_optimization.yaml"
    
    # Load experiment configuration
    print("📋 Loading experiment configuration...")
    exp_config = load_experiment_config(config_path)
    
    # Setup MLflow
    experiment_name = exp_config["experiment_name"]
    mlflow.set_experiment(experiment_name)
    
    print(f"✅ Loaded config: {len(exp_config['experiments'])} experiments")
    print(f"📊 MLflow experiment: {experiment_name}")
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"⚡ GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("❌ No GPU available!")
        return
    
    # Run all experiments
    results = []
    base_config = exp_config["base_config"]
    dataset_path = exp_config["dataset_config"]["dataset_path"]
    
    total_start = time.time()
    
    for i, experiment in enumerate(exp_config["experiments"]):
        exp_name = experiment["name"]
        print(f"\n📍 Progress: {i+1}/8 experiments")
        
        result = run_single_experiment(exp_name, experiment, base_config, dataset_path)
        result["experiment_name"] = exp_name
        results.append(result)
        
        # Sync every 2 experiments
        if (i + 1) % 2 == 0:
            sync_to_drive()
    
    # Final sync
    sync_to_drive()
    
    # Print summary
    total_elapsed = time.time() - total_start
    print(f"\n🕐 Total Time: {total_elapsed/3600:.1f} hours")
    print_final_summary(results)
    
    print("\n🎉 Grid Search Phase 1 Complete!")
    print("Next: Choose best model for 5-fold CV evaluation")

if __name__ == "__main__":
    main() 