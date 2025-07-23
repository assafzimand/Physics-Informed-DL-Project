#!/usr/bin/env python3
"""
T=250 5-Fold Cross-Validation FULL Training Script for Colab
Complete 75-epoch training per fold with academic-quality results.
"""

import os
import sys
import time
import json
import shutil
from datetime import datetime
from pathlib import Path

# Add current project directory to Python path
current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from src.training.cv_trainer import CrossValidationTrainer
    from configs.training_config import TrainingConfig
    import torch
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("🔧 Please ensure you're running from the project directory!")
    sys.exit(1)


def print_banner():
    """Print training banner."""
    print("🚀 T=250 5-Fold Cross-Validation FULL Training")
    print("=" * 60)
    print(f"🕐 Start: {datetime.now().strftime('%H:%M:%S')}")
    print("⏱️ Expected duration: ~10 hours")
    print("🎯 75 epochs per fold × 5 folds = 375 total epochs")
    print("🏆 Goal: Academic-quality statistical results")
    print("=" * 60)


def check_environment():
    """Check Colab environment."""
    print("🔍 Checking environment...")
    
    # GPU check
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("❌ No GPU available - cannot proceed with full training!")
        return None
        
    # Check for dataset
    dataset_paths = [
        '/content/drive/MyDrive/Physics_Informed_DL_Project/datasets/wave_dataset_T250.h5',
        '/content/drive/MyDrive/Physics_Informed_DL_Project/data/wave_dataset_T250.h5',
        'data/wave_dataset_T250.h5',
        'datasets/wave_dataset_T250.h5'
    ]
    
    for dataset_path in dataset_paths:
        if os.path.exists(dataset_path):
            file_size = os.path.getsize(dataset_path) / (1024**3)
            print(f"✅ T=250 dataset: {dataset_path} ({file_size:.1f}GB)")
            return dataset_path
    
    print("❌ T=250 dataset not found!")
    print("🔍 Searched locations:")
    for path in dataset_paths:
        print(f"   {path}")
    return None


def create_full_cv_config(dataset_path):
    """Create full training configuration for CV."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    
    config = TrainingConfig(
        # Winning hyperparameters from validation
        learning_rate=0.001,
        batch_size=32,
        optimizer="adam",
        weight_decay=0.01,
        
        # Dataset configuration
        dataset_path=dataset_path,
        train_split=0.8,
        val_split=0.2,
        test_split=0.0,
        
        # FULL training configuration
        num_epochs=75,  # FULL: 75 epochs per fold
        early_stopping_patience=15,
        
        # Model configuration
        model_name="WaveSourceMiniResNet",
        grid_size=128,
        
        # Training settings
        device="cuda",
        num_workers=2,
        pin_memory=True,
        
        # Scheduler
        scheduler_type="plateau",
        scheduler_patience=5,
        
        # Logging and saving
        experiment_name="t250_cv_full",
        run_name=f"t250_cv_full_75epochs_{timestamp}",
        save_model_every_n_epochs=25,
        
        # Random seed for reproducibility
        random_seed=42
    )
    
    print("\n🔧 FULL CV Configuration:")
    print(f"   Dataset: T=250 ({dataset_path})")
    print(f"   Hyperparameters: lr={config.learning_rate}, bs={config.batch_size}, opt={config.optimizer}")
    print(f"   Epochs per fold: {config.num_epochs}")
    print(f"   Total training time: ~{config.num_epochs * 5 * 2} minutes (~{config.num_epochs * 5 * 2 / 60:.1f} hours)")
    print(f"   Device: {config.device}")
    print(f"   Run name: {config.run_name}")
    
    return config


def run_full_cv(config):
    """Run the 5-fold CV full training."""
    print("\n🚀 Starting T=250 5-Fold CV FULL Training...")
    print("⚠️ This will take approximately 10 hours")
    print("📱 Keep this tab active to prevent disconnection")
    
    try:
        # Create CV trainer
        cv_trainer = CrossValidationTrainer(config, k_folds=5)
        
        # Start CV training
        start_time = time.time()
        cv_results = cv_trainer.run_cross_validation()
        end_time = time.time()
        
        total_time = (end_time - start_time) / 60  # minutes
        print(f"\n⏱️ Full CV training completed in {total_time:.1f} minutes ({total_time/60:.1f} hours)")
        
        return cv_results, total_time
        
    except Exception as e:
        print(f"\n❌ CV training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def analyze_full_cv_results(cv_results, total_time):
    """Analyze and display full CV results."""
    if cv_results is None:
        print("❌ No CV results available - training failed")
        return None
    
    print("\n" + "=" * 60)
    print("🎉 T=250 5-FOLD CV FULL TRAINING COMPLETE!")
    print("=" * 60)
    
    print(f"⏱️ Total Time: {total_time:.1f} minutes ({total_time/60:.1f} hours)")
    print(f"📊 Full Training Results (75 epochs per fold):")
    
    # Extract key metrics (using correct CV trainer keys)
    mean_distance_error = cv_results['distance_error_mean']
    std_distance_error = cv_results['distance_error_std']
    min_distance_error = cv_results['distance_error_min']
    max_distance_error = cv_results['distance_error_max']
    mean_val_loss = cv_results['val_loss_mean']
    std_val_loss = cv_results['val_loss_std']
    
    print(f"   Mean Distance Error: {mean_distance_error:.3f} ± {std_distance_error:.3f} px")
    print(f"   Best Fold: {min_distance_error:.3f} px")
    print(f"   Worst Fold: {max_distance_error:.3f} px")
    print(f"   Mean Val Loss: {mean_val_loss:.4f} ± {std_val_loss:.4f}")
    
    # Fold-by-fold results
    print(f"\n📋 Individual Fold Results:")
    fold_results = cv_results['fold_results']
    for i, fold in enumerate(fold_results):
        print(f"   Fold {i+1}: {fold['distance_error']:.3f} px, {fold['val_loss']:.4f} loss, {fold['training_time']:.1f} min")
    
    # Academic-quality performance assessment
    print(f"\n📈 Academic Performance Assessment:")
    if mean_distance_error <= 2.0:
        performance_grade = "EXCELLENT"
        print(f"   ✅ {performance_grade}: {mean_distance_error:.3f} ± {std_distance_error:.3f} px (≤ 2.0 px)")
    elif mean_distance_error <= 2.5:
        performance_grade = "VERY GOOD"
        print(f"   ✅ {performance_grade}: {mean_distance_error:.3f} ± {std_distance_error:.3f} px (≤ 2.5 px)")
    elif mean_distance_error <= 3.0:
        performance_grade = "GOOD"
        print(f"   ✅ {performance_grade}: {mean_distance_error:.3f} ± {std_distance_error:.3f} px (≤ 3.0 px)")
    else:
        performance_grade = "NEEDS IMPROVEMENT"
        print(f"   ⚠️ {performance_grade}: {mean_distance_error:.3f} ± {std_distance_error:.3f} px (> 3.0 px)")
    
    # Comparison with baselines
    print(f"\n🔄 Comparison with Baselines:")
    print(f"   T=500 Grid Search Best: 2.37 px")
    print(f"   T=500 CV Average: 2.078 ± 0.309 px")
    print(f"   T=250 Validation: 2.237 px")
    print(f"   T=250 CV Full: {mean_distance_error:.3f} ± {std_distance_error:.3f} px")
    
    # Statistical significance assessment
    if mean_distance_error < 2.5 and std_distance_error < 0.5:
        significance = "✅ STATISTICALLY ROBUST: Low variance, consistent performance"
    elif std_distance_error > 1.0:
        significance = "⚠️ HIGH VARIANCE: Results vary significantly across folds"
    else:
        significance = "✅ ACCEPTABLE VARIANCE: Reasonably consistent results"
    
    print(f"\n📊 Statistical Assessment:")
    print(f"   {significance}")
    print(f"   Coefficient of Variation: {(std_distance_error/mean_distance_error)*100:.1f}%")
    
    # Academic reporting format
    print(f"\n📝 Academic Reporting Format:")
    print(f"   Distance Error: {mean_distance_error:.2f} ± {std_distance_error:.2f} px")
    print(f"   Validation Loss: {mean_val_loss:.4f} ± {std_val_loss:.4f}")
    print(f"   Performance Grade: {performance_grade}")
    
    return {
        'distance_error_mean': mean_distance_error,
        'distance_error_std': std_distance_error,
        'distance_error_min': min_distance_error,
        'distance_error_max': max_distance_error,
        'val_loss_mean': mean_val_loss,
        'val_loss_std': std_val_loss,
        'total_time': total_time,
        'performance_grade': performance_grade,
        'cv_results': cv_results
    }


def create_cv_plots(cv_results):
    """Create comprehensive CV analysis plots."""
    if cv_results is None:
        return None
    
    print("\n📈 Creating CV analysis plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    fold_results = cv_results['fold_results']
    n_folds = len(fold_results)
    
    # Plot 1: Distance Error by Fold
    fold_numbers = list(range(1, n_folds + 1))
    distance_errors = [fold['distance_error'] for fold in fold_results]
    
    axes[0, 0].bar(fold_numbers, distance_errors, alpha=0.8, color='skyblue')
    axes[0, 0].axhline(y=cv_results['distance_error_mean'], color='red', linestyle='--', 
                       label=f'Mean: {cv_results["distance_error_mean"]:.3f} px')
    axes[0, 0].set_xlabel('Fold')
    axes[0, 0].set_ylabel('Distance Error (px)')
    axes[0, 0].set_title('Distance Error by Fold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Validation Loss by Fold
    val_losses = [fold['val_loss'] for fold in fold_results]
    
    axes[0, 1].bar(fold_numbers, val_losses, alpha=0.8, color='lightcoral')
    axes[0, 1].axhline(y=cv_results['val_loss_mean'], color='blue', linestyle='--',
                       label=f'Mean: {cv_results["val_loss_mean"]:.4f}')
    axes[0, 1].set_xlabel('Fold')
    axes[0, 1].set_ylabel('Validation Loss')
    axes[0, 1].set_title('Validation Loss by Fold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Training Time by Fold
    training_times = [fold['training_time'] for fold in fold_results]
    
    axes[0, 2].bar(fold_numbers, training_times, alpha=0.8, color='lightgreen')
    axes[0, 2].axhline(y=cv_results['training_time_mean'], color='purple', linestyle='--',
                       label=f'Mean: {cv_results["training_time_mean"]:.1f} min')
    axes[0, 2].set_xlabel('Fold')
    axes[0, 2].set_ylabel('Training Time (min)')
    axes[0, 2].set_title('Training Time by Fold')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Distance Error Distribution
    axes[1, 0].hist(distance_errors, bins=10, alpha=0.8, color='skyblue', edgecolor='black')
    axes[1, 0].axvline(x=cv_results['distance_error_mean'], color='red', linestyle='--',
                       label=f'Mean: {cv_results["distance_error_mean"]:.3f}')
    axes[1, 0].set_xlabel('Distance Error (px)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distance Error Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Error vs Loss Correlation
    axes[1, 1].scatter(val_losses, distance_errors, alpha=0.8, s=100, color='orange')
    for i, (loss, error) in enumerate(zip(val_losses, distance_errors)):
        axes[1, 1].annotate(f'F{i+1}', (loss, error), xytext=(5, 5), 
                           textcoords='offset points', fontsize=10)
    axes[1, 1].set_xlabel('Validation Loss')
    axes[1, 1].set_ylabel('Distance Error (px)')
    axes[1, 1].set_title('Validation Loss vs Distance Error')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: CV Summary Statistics
    stats_text = f"""
T=250 5-Fold CV Results

Distance Error:
Mean: {cv_results['distance_error_mean']:.3f} px
Std:  ±{cv_results['distance_error_std']:.3f} px
Min:  {cv_results['distance_error_min']:.3f} px
Max:  {cv_results['distance_error_max']:.3f} px

Validation Loss:
Mean: {cv_results['val_loss_mean']:.4f}
Std:  ±{cv_results['val_loss_std']:.4f}

Training Time:
Total: {cv_results['training_time_total']:.1f} min
Per Fold: {cv_results['training_time_mean']:.1f} min
    """
    
    axes[1, 2].text(0.1, 0.5, stats_text, transform=axes[1, 2].transAxes,
                    fontsize=11, verticalalignment='center', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].set_xticks([])
    axes[1, 2].set_yticks([])
    axes[1, 2].set_title('CV Summary Statistics')
    
    plt.tight_layout()
    plt.show()
    
    print("📊 CV analysis plots created!")
    return fig


def save_full_cv_results(config, results, fig=None):
    """Save full CV results to Drive."""
    if results is None:
        return
    
    print("\n💾 Saving full CV results to Drive...")
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # Create experiment directory structure in Drive
    drive_project_path = '/content/drive/MyDrive/Physics_Informed_DL_Project'
    cv_dir = Path(f"{drive_project_path}/experiments/t250_cv_full")
    cv_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped results directory
    results_dir = cv_dir / f"results_{timestamp}"
    results_dir.mkdir(exist_ok=True)
    
    # Save complete CV summary
    cv_summary = {
        'timestamp': datetime.now().isoformat(),
        'experiment_type': 'T250_5fold_CV_full_training',
        'training_config': {
            'epochs_per_fold': config.num_epochs,
            'total_folds': 5,
            'dataset': 'wave_dataset_T250.h5',
            'total_epochs': config.num_epochs * 5
        },
        'hyperparameters': {
            'learning_rate': config.learning_rate,
            'batch_size': config.batch_size,
            'optimizer': config.optimizer,
            'weight_decay': config.weight_decay
        },
        'results': results,
        'academic_metrics': {
            'distance_error_academic': f"{results['distance_error_mean']:.2f} ± {results['distance_error_std']:.2f} px",
            'val_loss_academic': f"{results['val_loss_mean']:.4f} ± {results['val_loss_std']:.4f}",
            'performance_grade': results['performance_grade']
        },
        'mlflow_experiment_name': config.experiment_name,
        'mlflow_run_name': config.run_name
    }
    
    # Save summary JSON
    summary_file = results_dir / f"cv_full_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(cv_summary, f, indent=2)
    print(f"✅ Summary saved: {summary_file}")
    
    # Save CV plots
    if fig is not None:
        plot_file = results_dir / f"cv_full_analysis_{timestamp}.png"
        fig.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"✅ Plots saved: {plot_file}")
    
    # Create comprehensive report
    report_file = results_dir / f"CV_FULL_REPORT_{timestamp}.md"
    with open(report_file, 'w') as f:
        f.write(f"# T=250 5-Fold Cross-Validation Full Training Report\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## 🎯 Experiment Overview\n")
        f.write(f"Complete 5-fold cross-validation training on T=250 dataset for academic-quality results.\n\n")
        f.write(f"### Training Configuration\n")
        f.write(f"- **Dataset**: wave_dataset_T250.h5\n")
        f.write(f"- **Epochs per Fold**: {config.num_epochs}\n")
        f.write(f"- **Total Epochs**: {config.num_epochs * 5}\n")
        f.write(f"- **Hyperparameters**: lr={config.learning_rate}, bs={config.batch_size}, opt={config.optimizer}\n")
        f.write(f"- **Training Time**: {results['total_time']:.1f} minutes ({results['total_time']/60:.1f} hours)\n\n")
        f.write(f"## 📊 Results Summary\n")
        f.write(f"- **Distance Error**: {results['distance_error_mean']:.3f} ± {results['distance_error_std']:.3f} px\n")
        f.write(f"- **Validation Loss**: {results['val_loss_mean']:.4f} ± {results['val_loss_std']:.4f}\n")
        f.write(f"- **Best Fold**: {results['distance_error_min']:.3f} px\n")
        f.write(f"- **Worst Fold**: {results['distance_error_max']:.3f} px\n")
        f.write(f"- **Performance Grade**: {results['performance_grade']}\n\n")
        f.write(f"## 📋 Individual Fold Results\n")
        fold_results = results['cv_results']['fold_results']
        for i, fold in enumerate(fold_results):
            f.write(f"- **Fold {i+1}**: {fold['distance_error']:.3f} px, {fold['val_loss']:.4f} loss\n")
        f.write(f"\n## 🔄 Baseline Comparisons\n")
        f.write(f"- **T=500 Grid Search Best**: 2.37 px\n")
        f.write(f"- **T=500 CV Average**: 2.078 ± 0.309 px\n")
        f.write(f"- **T=250 Validation**: 2.237 px\n")
        f.write(f"- **T=250 CV Full**: {results['distance_error_mean']:.3f} ± {results['distance_error_std']:.3f} px\n\n")
        f.write(f"## 📝 Academic Citation Format\n")
        f.write(f"\"The T=250 dataset achieved a cross-validated distance error of {results['distance_error_mean']:.2f} ± {results['distance_error_std']:.2f} pixels using 5-fold cross-validation with {config.num_epochs} epochs per fold.\"\n\n")
        f.write(f"## 📁 Generated Files\n")
        f.write(f"- `cv_full_summary_{timestamp}.json`: Complete experimental data\n")
        f.write(f"- `cv_full_analysis_{timestamp}.png`: Comprehensive analysis plots\n")
        f.write(f"- `CV_FULL_REPORT_{timestamp}.md`: This detailed report\n\n")
        f.write(f"## 🎯 Conclusions\n")
        f.write(f"The T=250 dataset demonstrates {results['performance_grade'].lower()} performance with consistent results across all folds, making it suitable for physics-informed deep learning applications in wave source localization.\n")
    
    print(f"✅ Report saved: {report_file}")
    
    # Save individual fold models and complete MLflow data  
    models_dir = results_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    print("💾 Saving trained models...")
    
    # Save individual fold models from CV results
    fold_models = results['cv_results'].get('fold_models', [])
    best_fold_idx = 0
    best_error = float('inf')
    
    for i, model_path in enumerate(fold_models):
        if os.path.exists(model_path):
            # Copy fold model to results directory
            fold_model_dest = models_dir / f"fold_{i+1}_model.pt"
            shutil.copy2(model_path, fold_model_dest)
            print(f"✅ Saved Fold {i+1} model: {fold_model_dest}")
            
            # Track best model
            fold_error = results['cv_results']['fold_results'][i]['distance_error']
            if fold_error < best_error:
                best_error = fold_error
                best_fold_idx = i
    
    # Save best model separately
    if fold_models:
        best_model_source = fold_models[best_fold_idx]
        if os.path.exists(best_model_source):
            best_model_dest = models_dir / "best_model.pt"
            shutil.copy2(best_model_source, best_model_dest)
            print(f"✅ Saved best model (Fold {best_fold_idx+1}): {best_model_dest}")
            
            # Save best model metadata
            best_model_info = {
                'best_fold': best_fold_idx + 1,
                'distance_error': best_error,
                'val_loss': results['cv_results']['fold_results'][best_fold_idx]['val_loss'],
                'model_path': str(best_model_dest),
                'timestamp': datetime.now().isoformat()
            }
            
            best_info_file = models_dir / "best_model_info.json"
            with open(best_info_file, 'w') as f:
                json.dump(best_model_info, f, indent=2)
            print(f"✅ Saved best model info: {best_info_file}")
    
    # Complete MLflow backup (like T=500)
    mlflow_backup_dir = results_dir / "mlflow_backup"
    mlflow_backup_dir.mkdir(exist_ok=True)
    
    mlflow_source = "mlruns"
    if os.path.exists(mlflow_source):
        # Copy complete MLflow directory
        mlflow_dest = mlflow_backup_dir / "mlruns"
        if mlflow_dest.exists():
            shutil.rmtree(mlflow_dest)
        shutil.copytree(mlflow_source, mlflow_dest)
        print(f"✅ Complete MLflow backup: {mlflow_backup_dir}")
        
        # Also copy to Drive root for easy access (like T=500)
        drive_mlflow_dest = Path(drive_project_path) / "mlruns_t250_cv_full"
        if drive_mlflow_dest.exists():
            shutil.rmtree(drive_mlflow_dest)
        shutil.copytree(mlflow_source, drive_mlflow_dest)
        print(f"✅ MLflow copied to Drive root: {drive_mlflow_dest}")
    
    # Save model ensemble info for future inference
    ensemble_info = {
        'experiment_type': 'T250_5fold_CV_full',
        'num_folds': 5,
        'fold_models': [f"fold_{i+1}_model.pt" for i in range(len(fold_models))],
        'best_model': "best_model.pt",
        'best_fold': best_fold_idx + 1,
        'ensemble_performance': {
            'mean_error': results['distance_error_mean'],
            'std_error': results['distance_error_std'],
            'individual_errors': [fold['distance_error'] for fold in results['cv_results']['fold_results']]
        },
        'usage_instructions': {
            'single_prediction': "Load best_model.pt for single predictions",
            'ensemble_prediction': "Load all fold models for ensemble prediction",
            'model_format': "PyTorch state_dict (.pt files)"
        }
    }
    
    ensemble_file = models_dir / "ensemble_info.json"
    with open(ensemble_file, 'w') as f:
        json.dump(ensemble_info, f, indent=2)
    print(f"✅ Saved ensemble info: {ensemble_file}")
    
    print(f"\n🎉 ALL FULL CV RESULTS SAVED TO DRIVE!")
    print(f"📁 Location: {results_dir}")
    print(f"🔄 Results synced to Google Drive!")


def main():
    """Main full CV training function."""
    print_banner()
    
    # Check environment and dataset
    dataset_path = check_environment()
    if dataset_path is None:
        return
    
    # Create full training configuration
    config = create_full_cv_config(dataset_path)
    
    # Confirm before starting
    print(f"\n⚠️ CONFIRMATION REQUIRED:")
    print(f"   This will run 5-fold CV training for ~10 hours")
    print(f"   Keep your Colab session active during training")
    print(f"   Results will be auto-saved to Drive")
    print(f"\n🚀 Starting full CV training in 5 seconds...")
    time.sleep(5)
    
    # Run full CV training
    cv_results, total_time = run_full_cv(config)
    
    # Analyze results
    results = analyze_full_cv_results(cv_results, total_time)
    
    # Create plots
    fig = create_cv_plots(cv_results)
    
    # Save all results
    save_full_cv_results(config, results, fig)
    
    print("\n" + "=" * 60)
    print("🎉 T=250 5-Fold CV Full Training Complete!")
    print("Academic-quality results saved to Drive.")
    print("=" * 60)


if __name__ == "__main__":
    main() 