#!/usr/bin/env python3
"""
T=250 Dataset Hyperparameter Validation Script
Run from colab_setup.ipynb using: !python colab/notebooks/validate_t250_hyperparams.py

Validates winning hyperparameters on T=250 dataset with single-fold training
before committing to full 5-fold CV.
"""

import sys
import os
import time
import json
import shutil
from datetime import datetime
from pathlib import Path

# Add project paths (compatible with existing Colab setup)
if '/content/drive/MyDrive/Physics_Informed_DL_Project' not in sys.path:
    sys.path.append('/content/drive/MyDrive/Physics_Informed_DL_Project')

try:
    # Import project modules
    from src.training.trainer import WaveTrainer
    from configs.training_config import TrainingConfig
    import torch
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("🔧 Please ensure you've run the colab_setup.ipynb first!")
    sys.exit(1)

def print_header():
    """Print validation header."""
    print("🧪 T=250 Dataset Hyperparameter Validation")
    print("=" * 60)
    print(f"🕐 Start time: {datetime.now().strftime('%H:%M:%S')}")
    print("⏱️ Expected duration: ~2 hours")
    print("🎯 Goal: Validate hyperparameters before full 5-fold CV")
    print("=" * 60)

def check_environment():
    """Check if environment is properly set up."""
    print("🔍 Checking environment...")
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU: {gpu_name}")
    else:
        print("⚠️ No GPU detected - training will be slow!")
    
    # Check Drive paths (compatible with existing setup)
    drive_project_path = '/content/drive/MyDrive/Physics_Informed_DL_Project'
    if not os.path.exists(drive_project_path):
        print(f"❌ Drive project path not found: {drive_project_path}")
        return False
    
    print(f"✅ Drive project path: {drive_project_path}")
    return True

def check_dataset():
    """Check if T=250 dataset exists."""
    print("\n📊 Checking T=250 dataset...")
    
    # Try multiple possible locations
    possible_paths = [
        '/content/drive/MyDrive/Physics_Informed_DL_Project/datasets/wave_dataset_T250.h5',
        '/content/drive/MyDrive/Physics_Informed_DL_Project/data/wave_dataset_T250.h5',
        'data/wave_dataset_T250.h5',
        'datasets/wave_dataset_T250.h5'
    ]
    
    for dataset_path in possible_paths:
        if os.path.exists(dataset_path):
            file_size = os.path.getsize(dataset_path) / (1024**3)  # GB
            print(f"✅ Dataset found: {dataset_path}")
            print(f"📦 File size: {file_size:.1f} GB")
            return dataset_path
    
    print("❌ T=250 dataset not found!")
    print("🔍 Searched locations:")
    for path in possible_paths:
        print(f"   {path}")
    
    print("\n💡 Please ensure wave_dataset_T250.h5 is uploaded to one of these locations")
    return None

def create_validation_config(dataset_path):
    """Create validation configuration using winning hyperparameters."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    
    config = TrainingConfig(
        # Winning hyperparameters from T=500 grid search
        learning_rate=0.001,
        batch_size=32,
        optimizer="adam",
        weight_decay=0.01,
        
        # Dataset configuration
        dataset_path=dataset_path,
        train_split=0.8,
        val_split=0.2,
        
        # Training configuration - validation run
        num_epochs=50,  # Quick validation
        early_stopping_patience=15,
        
        # Model configuration
        model_name="WaveSourceMiniResNet",
        grid_size=128,
        
        # Training settings
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_workers=2,
        pin_memory=True,
        
        # Scheduler
        scheduler_type="plateau",
        scheduler_patience=5,
        
        # Logging and saving
        experiment_name="t250_hyperparams_validation",
        run_name=f"validation_lr001_bs32_adam_50epochs_{timestamp}",
        save_model_every_n_epochs=25,
        
        # Random seed for reproducibility
        random_seed=42
    )
    
    print("\n🔧 Validation Configuration:")
    print(f"   Dataset: T=250 ({dataset_path})")
    print(f"   Hyperparameters: lr={config.learning_rate}, bs={config.batch_size}, opt={config.optimizer}")
    print(f"   Epochs: {config.num_epochs}")
    print(f"   Expected time: ~2 hours")
    print(f"   Device: {config.device}")
    print(f"   Run name: {config.run_name}")
    
    return config

def run_training(config):
    """Run the validation training."""
    print("\n🚀 Starting validation training...")
    
    try:
        # Create trainer
        trainer = WaveTrainer(config)
        
        # Start training
        start_time = time.time()
        training_history = trainer.train()
        end_time = time.time()
        
        training_duration = (end_time - start_time) / 60  # Convert to minutes
        print(f"\n⏱️ Training completed in {training_duration:.1f} minutes")
        
        return training_history, training_duration
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def analyze_results(training_history, training_duration):
    """Analyze and display results."""
    if training_history is None:
        print("❌ No training history available - training failed")
        return None, None
    
    # Extract final results
    final_train_loss = training_history['train_loss'][-1]
    final_val_loss = training_history['val_loss'][-1]
    final_val_distance_error = training_history['val_distance_error'][-1]
    best_epoch = training_history.get('best_epoch', len(training_history['val_loss']))
    
    print("\n" + "=" * 60)
    print("🎉 VALIDATION TRAINING COMPLETE!")
    print("=" * 60)
    
    print(f"⏱️ Training Time: {training_duration:.1f} minutes")
    print(f"🏆 Best Epoch: {best_epoch}")
    print(f"📊 Final Results:")
    print(f"   Training Loss: {final_train_loss:.4f}")
    print(f"   Validation Loss: {final_val_loss:.4f}")
    print(f"   Distance Error: {final_val_distance_error:.3f} px")
    
    # Performance assessment
    print(f"\n📈 Performance Assessment:")
    if final_val_distance_error <= 2.0:
        print(f"   ✅ EXCELLENT: {final_val_distance_error:.3f} px (≤ 2.0 px)")
        recommendation = "✅ RECOMMENDED: Proceed with full 5-fold CV training"
    elif final_val_distance_error <= 3.0:
        print(f"   ✅ GOOD: {final_val_distance_error:.3f} px (≤ 3.0 px)")
        recommendation = "✅ RECOMMENDED: Proceed with full 5-fold CV training"
    elif final_val_distance_error <= 4.0:
        print(f"   ⚠️ ACCEPTABLE: {final_val_distance_error:.3f} px (≤ 4.0 px)")
        recommendation = "⚠️ CONSIDER: Maybe adjust hyperparameters or proceed with caution"
    else:
        print(f"   ❌ CONCERNING: {final_val_distance_error:.3f} px (> 4.0 px)")
        recommendation = "❌ RECOMMEND: Consider hyperparameter tuning before full CV"
    
    # Comparison with T=500 results
    print(f"\n🔄 Comparison with T=500 Results:")
    print(f"   T=500 Grid Search Best: 2.37 px")
    print(f"   T=500 CV Average: 2.078 px")
    print(f"   T=250 Validation: {final_val_distance_error:.3f} px")
    
    if final_val_distance_error < 2.5:
        print("   ✅ T=250 performance is competitive with T=500!")
    elif final_val_distance_error < 3.5:
        print("   ✅ T=250 performance is reasonable compared to T=500")
    else:
        print("   ⚠️ T=250 performance is worse than T=500 - consider investigation")
    
    print(f"\n🎯 RECOMMENDATION:")
    print(f"   {recommendation}")
    
    if "RECOMMENDED" in recommendation:
        print(f"\n🚀 Next Steps:")
        print(f"   1. Run full 5-fold CV training on T=250 dataset")
        print(f"   2. Use same hyperparameters: lr=0.001, bs=32, adam")
        print(f"   3. Expected full CV time: ~10 hours")
        print(f"   4. Expected CV performance: ~{final_val_distance_error:.1f} ± 0.3 px")
    
    return {
        'final_train_loss': final_train_loss,
        'final_val_loss': final_val_loss,
        'final_val_distance_error': final_val_distance_error,
        'best_epoch': best_epoch,
        'training_duration': training_duration,
        'recommendation': recommendation
    }, training_history

def create_plots(training_history, results):
    """Create and save training plots."""
    if training_history is None:
        return None
    
    print("\n📈 Creating training curves...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Loss curves
    epochs = range(1, len(training_history['train_loss']) + 1)
    axes[0].plot(epochs, training_history['train_loss'], 'b-', label='Training Loss', alpha=0.8)
    axes[0].plot(epochs, training_history['val_loss'], 'r-', label='Validation Loss', alpha=0.8)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('T=250 Validation: Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Distance error
    axes[1].plot(epochs, training_history['val_distance_error'], 'g-', label='Distance Error', alpha=0.8)
    axes[1].axhline(y=2.0, color='orange', linestyle='--', alpha=0.7, label='Target (2.0 px)')
    axes[1].axhline(y=2.078, color='purple', linestyle='--', alpha=0.7, label='T=500 CV Avg')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Distance Error (px)')
    axes[1].set_title('T=250 Validation: Distance Error')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Summary
    axes[2].text(0.5, 0.7, 'Final Results', ha='center', va='center', transform=axes[2].transAxes, 
                fontsize=14, fontweight='bold')
    axes[2].text(0.5, 0.5, f'Distance Error: {results["final_val_distance_error"]:.3f} px', 
                ha='center', va='center', transform=axes[2].transAxes, fontsize=12)
    axes[2].text(0.5, 0.3, f'Training Time: {results["training_duration"]:.1f} min', 
                ha='center', va='center', transform=axes[2].transAxes, fontsize=12)
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    axes[2].set_title('Summary')
    
    plt.tight_layout()
    plt.show()
    
    print("📊 Training curves displayed!")
    return fig

def save_results_to_drive(config, results, training_history, fig=None):
    """Save all results to Google Drive."""
    print("\n💾 Saving validation results to Drive...")
    
    # Create timestamp for this validation run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # Create validation results directory in Drive
    drive_project_path = '/content/drive/MyDrive/Physics_Informed_DL_Project'
    validation_dir = Path(f"{drive_project_path}/results/t250_validation_{timestamp}")
    validation_dir.mkdir(parents=True, exist_ok=True)
    
    # Save validation summary
    validation_results = {
        'timestamp': datetime.now().isoformat(),
        'experiment_type': 'T250_hyperparameter_validation',
        'dataset': 'wave_dataset_T250.h5',
        'hyperparameters': {
            'learning_rate': config.learning_rate,
            'batch_size': config.batch_size,
            'optimizer': config.optimizer,
            'weight_decay': config.weight_decay
        },
        'training_config': {
            'epochs': config.num_epochs,
            'early_stopping_patience': config.early_stopping_patience,
            'model_name': config.model_name
        },
        'results': results,
        'comparison': {
            't500_grid_search_best': 2.37,
            't500_cv_average': 2.078,
            't250_validation': results['final_val_distance_error']
        },
        'mlflow_experiment_name': config.experiment_name,
        'mlflow_run_name': config.run_name
    }
    
    # Save summary JSON
    summary_file = validation_dir / f"validation_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(validation_results, f, indent=2)
    print(f"✅ Summary saved: {summary_file}")
    
    # Save training curves plot
    if fig is not None:
        plot_file = validation_dir / f"training_curves_{timestamp}.png"
        fig.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"✅ Plots saved: {plot_file}")
    
    # Create final report
    report_file = validation_dir / f"VALIDATION_REPORT_{timestamp}.md"
    with open(report_file, 'w') as f:
        f.write(f"# T=250 Hyperparameter Validation Report\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## 🎯 Objective\n")
        f.write(f"Validate winning hyperparameters on T=250 dataset before full 5-fold CV.\n\n")
        f.write(f"## 📊 Results\n")
        f.write(f"- **Final Distance Error**: {results['final_val_distance_error']:.3f} px\n")
        f.write(f"- **Training Time**: {results['training_duration']:.1f} minutes\n")
        f.write(f"- **Best Epoch**: {results['best_epoch']}\n")
        f.write(f"- **Final Train Loss**: {results['final_train_loss']:.4f}\n")
        f.write(f"- **Final Val Loss**: {results['final_val_loss']:.4f}\n\n")
        f.write(f"## 🔄 Comparison with T=500\n")
        f.write(f"- **T=500 Grid Search Best**: 2.37 px\n")
        f.write(f"- **T=500 CV Average**: 2.078 px\n")
        f.write(f"- **T=250 Validation**: {results['final_val_distance_error']:.3f} px\n\n")
        f.write(f"## 🎯 Recommendation\n")
        f.write(f"{results['recommendation']}\n\n")
        f.write(f"## 📁 Files in this validation\n")
        f.write(f"- `validation_summary_{timestamp}.json`: Complete results data\n")
        f.write(f"- `training_curves_{timestamp}.png`: Training visualizations\n")
        f.write(f"- `VALIDATION_REPORT_{timestamp}.md`: This report\n")
    
    print(f"✅ Report saved: {report_file}")
    
    print(f"\n🎉 ALL RESULTS SAVED TO DRIVE!")
    print(f"📁 Location: {validation_dir}")
    print(f"🔄 Results are automatically synced to Google Drive!")

def main():
    """Main validation function."""
    print_header()
    
    # Check environment
    if not check_environment():
        return
    
    # Check dataset
    dataset_path = check_dataset()
    if dataset_path is None:
        return
    
    # Create configuration
    config = create_validation_config(dataset_path)
    
    # Run training
    training_history, training_duration = run_training(config)
    
    # Analyze results
    results, training_history = analyze_results(training_history, training_duration)
    if results is None:
        return
    
    # Create plots
    fig = create_plots(training_history, results)
    
    # Save results to Drive
    save_results_to_drive(config, results, training_history, fig)
    
    print("\n" + "=" * 60)
    print("🎉 T=250 Validation Complete!")
    print("Check the recommendation above for next steps.")
    print("=" * 60)

if __name__ == "__main__":
    main() 