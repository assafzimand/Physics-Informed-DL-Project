#!/usr/bin/env python3
"""
5-Fold Cross-Validation Full Training Script for Colab
Complete 75-epoch training per fold with auto-save to Drive.
"""

import os
import sys
import time
import yaml
import shutil

# Add project root to path
sys.path.append('/content/Physics-Informed-DL-Project')

from src.training.cv_trainer import CrossValidationTrainer
from configs.training_config import TrainingConfig


def print_banner():
    """Print training banner."""
    print("🚀 5-Fold Cross-Validation - Full Training")
    print("=" * 60)
    print("🕐 Start:", time.strftime("%H:%M:%S"))
    print("⏱️ Expected duration: ~10 hours")
    print("🎯 75 epochs per fold × 5 folds = 375 total epochs")
    print("🏆 Goal: Academic-quality statistical results")
    print("=" * 60)


def check_environment():
    """Check Colab environment."""
    import torch
    
    # GPU check
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("❌ No GPU available!")
        return False
        
    # Drive check
    if os.path.exists("/content/drive/MyDrive"):
        print("✅ Google Drive mounted")
    else:
        print("❌ Google Drive not mounted!")
        return False
        
    # Dataset check
    dataset_path = "/content/drive/MyDrive/Physics_Informed_DL_Project/datasets/wave_dataset_T500.h5"
    if os.path.exists(dataset_path):
        print("✅ Dataset accessible")
    else:
        print(f"❌ Dataset not found: {dataset_path}")
        return False
        
    return True


def load_full_config():
    """Load full CV training configuration."""
    config_path = "/content/Physics-Informed-DL-Project/colab/experiments/experiment_configs/cv_phase2.yaml"
    
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Merge base_config with full_config
    base_config = config_data['base_config']
    full_config = config_data['full_config']
    dataset_config = config_data['dataset_config']
    
    # Create final config
    final_config = {**base_config, **full_config, **dataset_config}
    
    # Extract CV-specific parameters that TrainingConfig doesn't accept
    k_folds = final_config.pop('k_folds', 5)
    final_config.pop('expected_time_minutes', None)  # Remove but don't store
    
    print("🔧 Full Training Config:")
    print(f"   Epochs: {final_config['num_epochs']} per fold")
    print(f"   Total epochs: {final_config['num_epochs'] * k_folds}")
    print(f"   K-Folds: {k_folds}")
    print(f"   Batch size: {final_config['batch_size']}")
    print(f"   Learning rate: {final_config['learning_rate']}")
    print(f"   Optimizer: {final_config['optimizer']}")
    print(f"   Early stopping: {final_config['early_stopping_patience']} epochs")
    
    return TrainingConfig(**final_config), k_folds


def setup_drive_directories():
    """Setup Google Drive directories for results."""
    base_path = "/content/drive/MyDrive/Physics_Informed_DL_Project/results/cv_phase2"
    
    directories = [
        base_path,
        f"{base_path}/mlruns",
        f"{base_path}/models",
        f"{base_path}/plots",
        f"{base_path}/logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        
    print(f"✅ Drive directories setup: {base_path}")
    return base_path


def sync_after_fold(fold_idx, drive_base_path):
    """Sync results to Drive after each fold."""
    print(f"\n💾 Syncing Fold {fold_idx + 1} results to Drive...")
    
    try:
        # Sync MLflow data
        mlruns_source = "/content/Physics-Informed-DL-Project/mlruns"
        mlruns_dest = f"{drive_base_path}/mlruns"
        
        if os.path.exists(mlruns_source):
            shutil.copytree(mlruns_source, mlruns_dest, dirs_exist_ok=True)
            print(f"✅ Fold {fold_idx + 1} MLflow data synced")
        
        # Sync models if they exist
        models_source = "/content/Physics-Informed-DL-Project/models"
        models_dest = f"{drive_base_path}/models"
        
        if os.path.exists(models_source):
            shutil.copytree(models_source, models_dest, dirs_exist_ok=True)
            print(f"✅ Fold {fold_idx + 1} model data synced")
            
        return True
        
    except Exception as e:
        print(f"⚠️ Fold {fold_idx + 1} sync failed: {e}")
        return False


def run_full_cv_training():
    """Run the complete 5-fold CV training."""
    print("\n🚀 Starting Full 5-Fold Cross-Validation Training")
    print("⏱️ Expected time: ~10 hours")
    print("💡 This will train 5 models with 75 epochs each")
    print("=" * 60)
    
    start_time = time.time()
    drive_base_path = setup_drive_directories()
    
    try:
        config, k_folds = load_full_config()
        
        # Create CV trainer
        cv_trainer = CrossValidationTrainer(config, k_folds=k_folds)
        
        # Override fold completion callback for auto-save
        original_train_single_fold = cv_trainer._train_single_fold
        
        def train_single_fold_with_sync(*args, **kwargs):
            """Wrapper to add auto-save after each fold."""
            result = original_train_single_fold(*args, **kwargs)
            fold_idx = result['fold_idx']
            sync_after_fold(fold_idx, drive_base_path)
            return result
            
        cv_trainer._train_single_fold = train_single_fold_with_sync
        
        print("📊 Starting cross-validation training...")
        
        # Run cross-validation
        cv_results = cv_trainer.run_cross_validation()
        
        # Final sync of all results
        print("\n💾 Final sync to Drive...")
        final_sync_success = sync_final_results(drive_base_path, cv_results)
        
        # Print comprehensive results
        cv_trainer.print_cv_summary(cv_results)
        print_academic_results(cv_results)
        
        total_time = time.time() - start_time
        print("\n🎉 5-Fold CV Training Complete!")
        print(f"⏱️ Total training time: {total_time/3600:.1f} hours")
        print(f"💾 Results saved to Drive: {drive_base_path}")
        
        return True, cv_results
        
    except Exception as e:
        print(f"❌ CV training failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def sync_final_results(drive_base_path, cv_results):
    """Final comprehensive sync to Drive."""
    try:
        # Sync all MLflow data
        mlruns_source = "/content/Physics-Informed-DL-Project/mlruns"
        mlruns_dest = f"{drive_base_path}/mlruns"
        
        if os.path.exists(mlruns_source):
            shutil.copytree(mlruns_source, mlruns_dest, dirs_exist_ok=True)
            
        # Create results summary file
        summary_path = f"{drive_base_path}/cv_results_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("5-Fold Cross-Validation Results Summary\n")
            f.write("="*50 + "\n\n")
            f.write(f"Distance Error: {cv_results['distance_error_mean']:.3f} ± {cv_results['distance_error_std']:.3f} px\n")
            f.write(f"Validation Loss: {cv_results['val_loss_mean']:.4f} ± {cv_results['val_loss_std']:.4f}\n")
            f.write(f"Training Time: {cv_results['training_time_total']:.1f} minutes\n")
            f.write(f"Individual Fold Errors: {cv_results['distance_error_values']}\n")
            f.write(f"Individual Fold Losses: {cv_results['val_loss_values']}\n")
            
        print("✅ Final results synced to Drive")
        return True
        
    except Exception as e:
        print(f"⚠️ Final sync failed: {e}")
        return False


def print_academic_results(cv_results):
    """Print results in academic format."""
    print("\n" + "="*80)
    print("📊 ACADEMIC RESULTS FOR PAPER")
    print("="*80)
    
    mean_error = cv_results['distance_error_mean']
    std_error = cv_results['distance_error_std']
    mean_loss = cv_results['val_loss_mean']
    std_loss = cv_results['val_loss_std']
    
    print("\n🎯 WAVE SOURCE LOCALIZATION PERFORMANCE:")
    print(f"   Distance Error: {mean_error:.2f} ± {std_error:.2f} pixels")
    print(f"   Validation Loss: {mean_loss:.4f} ± {std_loss:.4f}")
    print("   Number of Folds: 5")
    print("   Total Samples: 2000 (400 per fold validation)")
    
    print("\n📈 STATISTICAL ANALYSIS:")
    print(f"   Best Fold: {cv_results['distance_error_min']:.2f} px")
    print(f"   Worst Fold: {cv_results['distance_error_max']:.2f} px")
    print(f"   Coefficient of Variation: {(std_error/mean_error)*100:.1f}%")
    
    print("\n🏆 HYPERPARAMETERS (GRID SEARCH WINNER):")
    print("   Learning Rate: 0.001")
    print("   Batch Size: 32")
    print("   Optimizer: Adam")
    print("   Epochs: 75 (with early stopping)")
    
    print("\n" + "="*80)


def main():
    """Main execution."""
    print_banner()
    
    # Environment verification
    if not check_environment():
        print("❌ Environment check failed. Please fix issues above.")
        return
        
    print("✅ Environment verified - starting training")
    
    try:
        success, cv_results = run_full_cv_training()
        
        if success and cv_results:
            print("\n🎉 SUCCESS! 5-Fold CV training completed successfully")
            print("📊 Results saved to Google Drive")
            print("🏆 Academic-quality statistical results obtained")
        else:
            print("\n❌ Training failed or incomplete")
            
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        print("💾 Partial results may be saved to Drive")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 