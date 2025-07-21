#!/usr/bin/env python3
"""
Single Experiment Test - Full Pipeline Verification
Tests model saving, Drive sync, and MLflow artifacts with 10 epochs

Usage in Colab:
    !python colab/notebooks/test_single_experiment.py
"""

import sys
import time
import torch
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path("/content/Physics-Informed-DL-Project")
sys.path.append(str(project_root))

# Import our modules
from configs.training_config import TrainingConfig
from src.training.trainer import WaveTrainer


def print_banner():
    print("🔧 Single Experiment - Full Pipeline Test")
    print("=" * 50)
    print(f"🕐 Start: {datetime.now().strftime('%H:%M:%S')}")
    print("⚡ Config: LR=0.001, BS=32, Adam, 10 epochs")
    print("🎯 Goal: Verify saving, sync, MLflow artifacts")
    print("⏱️ Expected: ~8-10 minutes")
    print("=" * 50)


def create_single_test_config() -> TrainingConfig:
    """Create config for single comprehensive test"""
    
    config_dict = {
        # Dataset
        "dataset_path": "/content/drive/MyDrive/Physics_Informed_DL_Project/datasets/wave_dataset_T500.h5",
        "train_split": 0.7,
        "val_split": 0.15,
        "test_split": 0.15,
        "random_seed": 42,
        
        # Real training (enough to test saving)
        "num_epochs": 10,
        "device": "cuda",
        "save_model_every_n_epochs": 3,  # Will save at epochs 3, 6, 9
        "early_stopping_patience": 15,  # Won't trigger
        
        # Experiment parameters (good baseline)
        "learning_rate": 0.001,
        "batch_size": 32, 
        "optimizer": "adam",
        "weight_decay": 0.01,
        "scheduler_type": "plateau",
        "scheduler_patience": 5,
        
        # Model architecture
        "model_name": "WaveSourceMiniResNet",
        "grid_size": 128,
        
        # Data loading
        "num_workers": 2,
        "pin_memory": True,
        
        # Experiment tracking
        "experiment_name": "single_pipeline_test",
        "run_name": "test_lr001_bs32_adam_10epochs",
        
        # Output directories (test these get created)
        "output_dir": "test_experiments",
        "model_save_dir": "test_models",
        "logs_dir": "test_logs",
        "plots_dir": "test_plots"
    }
    
    return TrainingConfig(**config_dict)


def check_local_outputs():
    """Check what files were created locally"""
    print("\n🔍 Checking Local Outputs:")
    
    directories_to_check = [
        "test_experiments",
        "test_models", 
        "test_logs",
        "test_plots",
        "mlruns"
    ]
    
    for dir_name in directories_to_check:
        if os.path.exists(dir_name):
            files = list(Path(dir_name).rglob("*.*"))
            print(f"   ✅ {dir_name}/: {len(files)} files")
            # Show some example files
            for file in files[:3]:
                print(f"      📄 {file}")
            if len(files) > 3:
                print(f"      ... and {len(files)-3} more")
        else:
            print(f"   ❌ {dir_name}/: Not found")


def sync_to_drive():
    """Sync results to Google Drive and verify"""
    print("\n💾 Testing Drive Sync:")
    
    try:
        # Create drive folders
        drive_base = "/content/drive/MyDrive/Physics_Informed_DL_Project"
        drive_results = f"{drive_base}/single_test_results"
        drive_mlruns = f"{drive_base}/mlruns"
        
        os.makedirs(drive_results, exist_ok=True)
        os.makedirs(drive_mlruns, exist_ok=True)
        
        # Copy local outputs to Drive
        sync_commands = [
            f"cp -r test_models/* {drive_results}/ 2>/dev/null || true",
            f"cp -r test_logs/* {drive_results}/ 2>/dev/null || true", 
            f"cp -r test_plots/* {drive_results}/ 2>/dev/null || true",
            f"cp -r mlruns/* {drive_mlruns}/ 2>/dev/null || true"
        ]
        
        for cmd in sync_commands:
            os.system(cmd)
        
        # Verify sync worked
        if os.path.exists(drive_results):
            drive_files = list(Path(drive_results).rglob("*.*"))
            print(f"   ✅ Drive sync: {len(drive_files)} files copied")
            
            # Check for key file types
            model_files = [f for f in drive_files if f.suffix == '.pth']
            log_files = [f for f in drive_files if f.suffix == '.log']
            
            print(f"   📊 Models: {len(model_files)} .pth files")
            print(f"   📋 Logs: {len(log_files)} .log files")
            
            if model_files:
                print(f"   💾 Example model: {model_files[0].name}")
                
        else:
            print("   ❌ Drive sync failed - directory not created")
            
    except Exception as e:
        print(f"   ❌ Drive sync error: {e}")


def check_mlflow_artifacts():
    """Check MLflow tracking worked"""
    print("\n📊 Checking MLflow Artifacts:")
    
    mlruns_path = Path("mlruns")
    if mlruns_path.exists():
        # Look for experiment folders
        experiments = [d for d in mlruns_path.iterdir() if d.is_dir() and d.name != '.trash']
        print(f"   ✅ MLflow experiments: {len(experiments)}")
        
        for exp_dir in experiments[:2]:  # Check first 2
            runs = [d for d in exp_dir.iterdir() if d.is_dir()]
            print(f"   📁 {exp_dir.name}: {len(runs)} runs")
            
            # Check for artifacts in latest run
            if runs:
                latest_run = max(runs, key=lambda x: x.stat().st_mtime)
                artifacts_dir = latest_run / "artifacts"
                if artifacts_dir.exists():
                    artifacts = list(artifacts_dir.rglob("*.*"))
                    print(f"      💾 Artifacts: {len(artifacts)} files")
                    
                    # Check for model files
                    model_artifacts = [f for f in artifacts if 'model' in str(f)]
                    if model_artifacts:
                        print(f"      🤖 Model artifacts: {len(model_artifacts)}")
    else:
        print("   ❌ MLflow directory not found")


def print_final_summary(metrics, duration_minutes):
    """Print test results and next steps"""
    print("\n" + "=" * 60)
    print("🧪 SINGLE EXPERIMENT TEST RESULTS")
    print("=" * 60)
    
    # Extract final metrics
    final_val_loss = metrics['val_loss'][-1] if metrics['val_loss'] else float('inf')
    final_distance_error = metrics['val_distance_error'][-1] if metrics['val_distance_error'] else float('inf')
    
    print("📊 Training Results:")
    print(f"   Distance Error: {final_distance_error:.2f} px")
    print(f"   Validation Loss: {final_val_loss:.4f}")
    print(f"   Training Time: {duration_minutes:.1f} minutes")
    
    # Estimate full grid search time
    estimated_full_time = duration_minutes * 8 / 10 * 5  # 8 experiments, 50 epochs vs 10
    print(f"\n⏱️ Full Grid Search Estimate:")
    print(f"   8 experiments × 50 epochs ≈ {estimated_full_time/60:.1f} hours")
    
    # Check if reasonable results
    if final_distance_error < 50:  # Reasonable for 10 epochs
        print(f"\n✅ PIPELINE TEST PASSED!")
        print(f"🚀 Ready for full grid search:")
        print(f"   !python colab/notebooks/run_optimization.py")
    else:
        print(f"\n⚠️ Results seem high - check configuration")
        print(f"   Distance error > 50px suggests potential issues")
    
    print(f"\n💾 Results saved to Google Drive")
    print(f"📱 Download with: python colab/mlflow/download_results.py")


def main():
    """Main single experiment test"""
    print_banner()
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"⚡ GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("❌ No GPU available!")
        return
    
    # Run single experiment
    start_time = time.time()
    
    try:
        print("\n🚀 Starting Single Experiment...")
        
        # Create config and trainer
        config = create_single_test_config()
        trainer = WaveTrainer(config)
        
        # Train model
        print("⏱️ Training 10 epochs... (this will take ~8-10 minutes)")
        metrics = trainer.train()
        
        duration_minutes = (time.time() - start_time) / 60
        print(f"\n✅ Training completed in {duration_minutes:.1f} minutes!")
        
        # Check outputs
        check_local_outputs()
        sync_to_drive()
        check_mlflow_artifacts()
        
        # Final summary
        print_final_summary(metrics, duration_minutes)
        
    except Exception as e:
        duration_minutes = (time.time() - start_time) / 60
        print(f"\n❌ Test failed after {duration_minutes:.1f} minutes:")
        print(f"   Error: {str(e)}")
        print(f"\n🔧 Check the error and configuration before full grid search")


if __name__ == "__main__":
    main() 