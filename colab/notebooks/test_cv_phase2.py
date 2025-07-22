#!/usr/bin/env python3
"""
5-Fold Cross-Validation Test Script for Colab
Quick test to verify the CV pipeline before full training run.
"""

import os
import sys
import time
import yaml
from pathlib import Path

# Add project root to path
sys.path.append('/content/Physics-Informed-DL-Project')

from src.training.cv_trainer import CrossValidationTrainer
from configs.training_config import TrainingConfig


def print_banner():
    """Print test banner."""
    print("🧪 5-Fold Cross-Validation Quick Test")
    print("=" * 50)
    print("🕐 Start:", time.strftime("%H:%M:%S"))
    print("⏱️ Expected duration: ~50 minutes")
    print("🧪 TEST MODE - Quick CV Pipeline Verification")
    print("=" * 50)


def check_gpu():
    """Check GPU availability."""
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        return True
    else:
        print("❌ No GPU available - switching to CPU")
        return False


def load_config():
    """Load CV test configuration."""
    config_path = "/content/Physics-Informed-DL-Project/colab/experiments/experiment_configs/cv_phase2.yaml"
    
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Merge base_config with test_config
    base_config = config_data['base_config']
    test_config = config_data['test_config']
    dataset_config = config_data['dataset_config']
    
    # Create final config
    final_config = {**base_config, **test_config, **dataset_config}
    
    # Extract CV-specific parameters that TrainingConfig doesn't accept
    k_folds = final_config.pop('k_folds', 5)
    final_config.pop('expected_time_minutes', None)  # Remove but don't store
    
    print(f"🔧 Test Config Created:")
    print(f"   Epochs: {final_config['num_epochs']} per fold (total: {final_config['num_epochs'] * k_folds})")
    print(f"   K-Folds: {k_folds}")
    print(f"   Batch size: {final_config['batch_size']}")
    print(f"   Learning rate: {final_config['learning_rate']}")
    print(f"   Optimizer: {final_config['optimizer']}")
    print(f"   Device: {final_config['device']}")
    
    return TrainingConfig(**final_config), k_folds


def run_cv_test():
    """Run the 5-fold CV test."""
    print("\n🚀 Starting CV Test...")
    print("⏱️ Expected time: ~50 minutes")
    print("=" * 40)
    
    try:
        config, k_folds = load_config()
        
        # Create CV trainer
        cv_trainer = CrossValidationTrainer(config, k_folds=k_folds)
        
        print("📊 Dataset loaded - starting cross-validation...")
        
        # Run cross-validation
        cv_results = cv_trainer.run_cross_validation()
        
        # Print results summary
        cv_trainer.print_cv_summary(cv_results)
        
        print("✅ CV Test Training Completed!")
        print(f"📊 Final Distance Error: {cv_results['distance_error_mean']:.2f} ± {cv_results['distance_error_std']:.2f} px")
        print(f"📉 Final Validation Loss: {cv_results['val_loss_mean']:.4f} ± {cv_results['val_loss_std']:.4f}")
        
        return True, cv_results
        
    except Exception as e:
        print(f"❌ CV test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def sync_to_drive():
    """Sync results to Google Drive."""
    print("\n🔍 Testing Drive sync...")
    
    try:
        # Check if MLflow directory exists
        mlruns_path = "/content/Physics-Informed-DL-Project/mlruns"
        if os.path.exists(mlruns_path):
            print("✅ MLflow data found")
            
            # Create Drive directory
            drive_cv_path = "/content/drive/MyDrive/Physics_Informed_DL_Project/results/cv_phase2_test"
            os.makedirs(drive_cv_path, exist_ok=True)
            
            # Copy MLflow data
            import shutil
            shutil.copytree(mlruns_path, f"{drive_cv_path}/mlruns", dirs_exist_ok=True)
            print(f"✅ MLflow data synced to Drive: {drive_cv_path}")
            
            return True
        else:
            print("⚠️ No MLflow data found")
            return False
            
    except Exception as e:
        print(f"⚠️ Drive sync failed: {e}")
        return False


def test_mlflow():
    """Test MLflow functionality."""
    print("\n🔍 Testing MLflow...")
    
    try:
        import mlflow
        
        # List experiments
        experiments = mlflow.search_experiments()
        print(f"✅ MLflow tracking working - found {len(experiments)} experiments")
        
        return True
        
    except Exception as e:
        print(f"⚠️ MLflow test failed: {e}")
        return False


def print_test_summary(cv_success, mlflow_ok, drive_ok, cv_results):
    """Print comprehensive test summary."""
    print("\n" + "=" * 60)
    print("🧪 CV PHASE 2 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    print(f"🚀 5-Fold CV Pipeline: {'✅ PASS' if cv_success else '❌ FAIL'}")
    print(f"📊 MLflow Tracking: {'✅ PASS' if mlflow_ok else '❌ FAIL'}")
    print(f"💾 Drive Sync: {'✅ PASS' if drive_ok else '❌ FAIL'}")
    
    if cv_results:
        print(f"\n📈 CV PERFORMANCE RESULTS:")
        print(f"   Distance Error: {cv_results['distance_error_mean']:.2f} ± {cv_results['distance_error_std']:.2f} px")
        print(f"   Validation Loss: {cv_results['val_loss_mean']:.4f} ± {cv_results['val_loss_std']:.4f}")
        print(f"   Training Time: {cv_results['training_time_total']:.1f} minutes")
        print(f"   Individual Fold Errors: {[f'{x:.2f}' for x in cv_results['distance_error_values']]}")
    
    overall_status = cv_success and mlflow_ok and drive_ok
    print(f"\n🎯 Overall Status: {'✅ READY FOR FULL CV' if overall_status else '❌ NEEDS FIXING'}")
    
    if not overall_status:
        print("🔧 Fix issues above before running full 5-fold cross-validation")
    else:
        print("🚀 Pipeline verified! Ready for full 75-epoch CV training")
    
    print(f"🕐 End: {time.strftime('%H:%M:%S')}")


def main():
    """Main test execution."""
    print_banner()
    
    # Environment checks
    gpu_available = check_gpu()
    if gpu_available:
        print("✅ Environment ready for testing")
    
    start_time = time.time()
    
    try:
        # Run tests
        cv_success, cv_results = run_cv_test()
        mlflow_ok = test_mlflow()
        drive_ok = sync_to_drive()
        
        duration_minutes = (time.time() - start_time) / 60
        
        # Print final summary
        print_test_summary(cv_success, mlflow_ok, drive_ok, cv_results)
        print(f"\n⏱️ Total Test Time: {duration_minutes:.1f} minutes")
        
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 