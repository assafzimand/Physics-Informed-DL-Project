#!/usr/bin/env python3
"""
Grid Search Pipeline Test - 2 Epochs Per Combination
Quick verification that all 8 combinations work before full run

Usage in Colab:
    !python colab/notebooks/test_grid_search.py
"""

import sys
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
    print("🧪 Grid Search Pipeline Test")
    print("=" * 40)
    print(f"🕐 Start: {datetime.now().strftime('%H:%M:%S')}")
    print("⚡ 2×2×2 Grid: 8 combinations × 2 epochs")
    print("🎯 Goal: Verify pipeline before full run")
    print("⏱️ Expected: ~20 minutes")
    print("=" * 40)


def create_test_config(exp_data: Dict[str, Any]) -> TrainingConfig:
    """Create test config with minimal epochs"""
    
    config_dict = {
        # Dataset
        "dataset_path": "/content/drive/MyDrive/Physics_Informed_DL_Project/datasets/wave_dataset_T500.h5",
        "train_split": 0.7,
        "val_split": 0.15,
        "test_split": 0.15,
        "random_seed": 42,
        
        # Minimal training for testing
        "num_epochs": 2,  # Just 2 epochs for testing!
        "device": "cuda",
        "save_model_every_n_epochs": 10,  # Won't save with 2 epochs
        "early_stopping_patience": 5,  # Won't trigger with 2 epochs
        
        # From experiment
        "learning_rate": exp_data["learning_rate"],
        "batch_size": exp_data["batch_size"], 
        "optimizer": exp_data["optimizer"],
        "weight_decay": 0.01,
        "scheduler_type": "cosine",
        
        # Model architecture
        "model_name": "WaveSourceMiniResNet",
        "grid_size": 128,
        
        # Data loading
        "num_workers": 2,
        "pin_memory": True,
        
        # Scheduler params
        "scheduler_patience": 5
    }
    
    return TrainingConfig(**config_dict)


def run_test_experiment(exp_name: str, exp_data: Dict[str, Any]) -> Dict[str, float]:
    """Run a single test experiment"""
    
    print(f"\n🚀 Testing: {exp_name}")
    print(f"   📋 LR: {exp_data['learning_rate']}, BS: {exp_data['batch_size']}, OPT: {exp_data['optimizer']}")
    
    start_time = time.time()
    
    try:
        # Create test config
        config = create_test_config(exp_data)
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"test_{exp_name}"):
            # Log experiment parameters
            mlflow.log_params({
                "learning_rate": exp_data["learning_rate"],
                "batch_size": exp_data["batch_size"],
                "optimizer": exp_data["optimizer"],
                "num_epochs": 2,
                "test_mode": True
            })
            
            # Train model
            trainer = WaveTrainer(config)
            metrics = trainer.train()
            
            # Extract final metrics
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
            print(f"   ✅ Completed in {elapsed/60:.1f} min")
            print(f"   📊 Distance Error: {final_distance_error:.2f} px (2 epochs only)")
            
            return {
                "val_loss": final_val_loss,
                "distance_error": final_distance_error,
                "train_loss": final_train_loss,
                "duration_minutes": elapsed / 60,
                "status": "success"
            }
            
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"   ❌ FAILED after {elapsed/60:.1f} min: {str(e)}")
        return {
            "val_loss": float('inf'),
            "distance_error": float('inf'),
            "train_loss": float('inf'),
            "duration_minutes": elapsed / 60,
            "status": "failed",
            "error": str(e)
        }


def print_test_summary(results: List[Dict[str, Any]]):
    """Print test results summary"""
    print("\n" + "=" * 50)
    print("🧪 GRID SEARCH PIPELINE TEST RESULTS")
    print("=" * 50)
    
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    
    print(f"✅ Successful: {len(successful)}/8")
    print(f"❌ Failed: {len(failed)}/8")
    
    if successful:
        avg_time = sum(r['duration_minutes'] for r in successful) / len(successful)
        print(f"⏱️ Avg time per experiment: {avg_time:.1f} minutes")
        print(f"📊 Estimated full run: {avg_time * 25:.1f} minutes ({avg_time * 25 / 60:.1f} hours)")
        
        print("\n📋 Quick Results (2 epochs only):")
        for i, result in enumerate(successful):
            exp_name = result['experiment_name']
            dist_err = result['distance_error']
            print(f"   {i+1}. {exp_name}: {dist_err:.1f}px")
    
    if failed:
        print("\n❌ Failed Experiments:")
        for result in failed:
            exp_name = result['experiment_name']
            error = result.get('error', 'Unknown error')
            print(f"   ❌ {exp_name}: {error}")
    
    if len(successful) == 8:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Ready to run full grid search:")
        print("   !python colab/notebooks/run_optimization.py")
    else:
        print(f"\n⚠️ Fix issues above before running full grid search")


def main():
    """Main test pipeline"""
    print_banner()
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"⚡ GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("❌ No GPU available!")
        return
    
    # Setup MLflow
    mlflow.set_experiment("grid_search_pipeline_test")
    print("📊 MLflow experiment: grid_search_pipeline_test")
    
    # Define the same 8 experiments from the config
    experiments = [
        {"name": "exp_001_lr001_bs16_adam", "learning_rate": 0.001, "batch_size": 16, "optimizer": "adam"},
        {"name": "exp_002_lr001_bs16_adamw", "learning_rate": 0.001, "batch_size": 16, "optimizer": "adamw"},
        {"name": "exp_003_lr001_bs32_adam", "learning_rate": 0.001, "batch_size": 32, "optimizer": "adam"},
        {"name": "exp_004_lr001_bs32_adamw", "learning_rate": 0.001, "batch_size": 32, "optimizer": "adamw"},
        {"name": "exp_005_lr0001_bs16_adam", "learning_rate": 0.0001, "batch_size": 16, "optimizer": "adam"},
        {"name": "exp_006_lr0001_bs16_adamw", "learning_rate": 0.0001, "batch_size": 16, "optimizer": "adamw"},
        {"name": "exp_007_lr0001_bs32_adam", "learning_rate": 0.0001, "batch_size": 32, "optimizer": "adam"},
        {"name": "exp_008_lr0001_bs32_adamw", "learning_rate": 0.0001, "batch_size": 32, "optimizer": "adamw"}
    ]
    
    # Run all test experiments
    results = []
    total_start = time.time()
    
    for i, experiment in enumerate(experiments):
        exp_name = experiment["name"]
        print(f"\n📍 Progress: {i+1}/8 experiments")
        
        result = run_test_experiment(exp_name, experiment)
        result["experiment_name"] = exp_name
        results.append(result)
    
    # Print summary
    total_elapsed = time.time() - total_start
    print(f"\n🕐 Total Test Time: {total_elapsed/60:.1f} minutes")
    print_test_summary(results)


if __name__ == "__main__":
    main() 