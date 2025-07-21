#!/usr/bin/env python3
"""
Quick Test - ResNet Optimization Pipeline

This is a test version that runs 1 experiment for 5 epochs to verify:
- GPU training works
- MLflow tracking works  
- Auto-save to Drive works
- Progress bars work

Usage in Colab:
    !python colab/notebooks/test_optimization.py

Expected time: ~5 minutes
"""

import os
import sys
import time
import torch
import mlflow
import mlflow.pytorch
from datetime import datetime

# Add project path to imports
sys.path.append('/content/Physics-Informed-DL-Project')

from src.training.trainer import WaveTrainer
from configs.training_config import TrainingConfig


def setup_test_environment():
    """Setup test environment."""
    print("🧪 TEST MODE - Quick Pipeline Verification")
    print("=" * 50)
    
    # Verify GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("❌ No GPU available!")
        return False
    
    # Setup MLflow
    mlflow_path = "/content/mlruns"
    os.makedirs(mlflow_path, exist_ok=True)
    mlflow.set_tracking_uri(f"file://{mlflow_path}")
    mlflow.set_experiment("test_optimization")
    
    # Create Drive directories
    results_base = "/content/drive/MyDrive/Physics_Informed_DL_Project/results"
    os.makedirs(f"{results_base}/test_models", exist_ok=True)
    os.makedirs(f"{results_base}/test_mlruns", exist_ok=True)
    
    print(f"✅ Environment ready for testing")
    return True


def create_test_config():
    """Create a quick test configuration."""
    config = TrainingConfig(
        dataset_name="T500",
        dataset_path="/content/data/wave_dataset_T500.h5",
        batch_size=32,
        learning_rate=0.001,
        num_epochs=5,  # Quick test - just 5 epochs
        optimizer="adam",
        weight_decay=0.0001,
        scheduler_type="plateau",
        scheduler_patience=3,
        early_stopping_patience=10,
        num_workers=2,
        pin_memory=True,
        device='cuda',
        run_name="test_optimization_run",
        experiment_name="test_optimization"
    )
    
    print(f"🔧 Test Config Created:")
    print(f"   Epochs: {config.num_epochs} (quick test)")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Device: {config.device}")
    
    return config


def run_test_experiment():
    """Run the test experiment."""
    print(f"\n🚀 Starting Test Experiment...")
    print(f"⏱️  Expected time: ~5 minutes")
    print("=" * 40)
    
    start_time = time.time()
    
    try:
        # Create config and trainer
        config = create_test_config()
        trainer = WaveTrainer(config)
        
        print(f"\n📊 Dataset loaded - starting training...")
        
        # Train model (5 epochs)
        best_metrics = trainer.train()
        
        # Extract results
        val_loss = best_metrics.get('best_val_loss', float('inf'))
        val_distance_error = best_metrics.get('val_distance_error', float('inf'))
        training_time = time.time() - start_time
        
        print(f"\n✅ Test Training Completed!")
        print(f"📊 Final Validation Loss: {val_loss:.3f}")
        print(f"📏 Final Distance Error: {val_distance_error:.2f} px")
        print(f"⏱️  Training Time: {training_time:.1f} seconds")
        
        return True, {
            'val_loss': val_loss,
            'val_distance_error': val_distance_error,
            'training_time': training_time
        }
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {}


def test_mlflow_tracking():
    """Test MLflow tracking."""
    print(f"\n🔍 Testing MLflow...")
    
    try:
        # Check if MLflow data exists
        mlflow_path = "/content/mlruns"
        if os.path.exists(mlflow_path):
            experiments = [d for d in os.listdir(mlflow_path) if d.isdigit()]
            if experiments:
                print(f"✅ MLflow tracking working - found {len(experiments)} experiments")
                return True
        
        print(f"⚠️  No MLflow data found")
        return False
        
    except Exception as e:
        print(f"❌ MLflow test failed: {e}")
        return False


def test_drive_sync():
    """Test Drive synchronization."""
    print(f"\n🔍 Testing Drive sync...")
    
    try:
        import shutil
        
        # Copy MLflow to Drive
        local_mlruns = "/content/mlruns"
        drive_mlruns = "/content/drive/MyDrive/Physics_Informed_DL_Project/results/test_mlruns"
        
        if os.path.exists(local_mlruns):
            if os.path.exists(drive_mlruns):
                shutil.rmtree(drive_mlruns)
            shutil.copytree(local_mlruns, drive_mlruns)
            print(f"✅ Drive sync working - MLflow data copied")
            return True
        
        print(f"⚠️  No data to sync")
        return False
        
    except Exception as e:
        print(f"❌ Drive sync failed: {e}")
        return False


def print_test_summary(success, results, mlflow_ok, drive_ok):
    """Print test summary."""
    print(f"\n" + "="*60)
    print(f"🧪 TEST RESULTS SUMMARY")
    print(f"="*60)
    
    # Training test
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"🚀 Training Pipeline: {status}")
    if success:
        print(f"   📏 Distance Error: {results['val_distance_error']:.2f} px")
        print(f"   ⏱️  Time: {results['training_time']:.1f}s")
    
    # MLflow test
    status = "✅ PASS" if mlflow_ok else "❌ FAIL"
    print(f"📊 MLflow Tracking: {status}")
    
    # Drive test  
    status = "✅ PASS" if drive_ok else "❌ FAIL"
    print(f"💾 Drive Sync: {status}")
    
    # Overall status
    all_good = success and mlflow_ok and drive_ok
    print(f"\n🎯 Overall Status: {'✅ READY FOR FULL OPTIMIZATION' if all_good else '❌ NEEDS FIXING'}")
    
    if all_good:
        print(f"\n🚀 Next Steps:")
        print(f"   1. Run: !python colab/notebooks/run_optimization.py")
        print(f"   2. Wait 6-8 hours for 8 experiments")
        print(f"   3. Get best model automatically!")
    else:
        print(f"\n🔧 Fix issues above before running full optimization")


def main():
    """Main test function."""
    print(f"🧪 ResNet Optimization - Quick Test")
    print(f"🕐 Start: {datetime.now().strftime('%H:%M:%S')}")
    print(f"⏱️  Expected duration: ~5 minutes")
    
    # Setup
    if not setup_test_environment():
        return
    
    # Run test training
    success, results = run_test_experiment()
    
    # Test MLflow
    mlflow_ok = test_mlflow_tracking()
    
    # Test Drive sync
    drive_ok = test_drive_sync()
    
    # Summary
    print_test_summary(success, results, mlflow_ok, drive_ok)
    
    print(f"\n🕐 End: {datetime.now().strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main() 