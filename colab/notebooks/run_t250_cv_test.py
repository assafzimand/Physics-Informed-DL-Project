#!/usr/bin/env python3
"""
T=250 5-Fold Cross-Validation TEST Script for Colab
Quick test with 2 epochs per fold to verify the flow before full training.
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
    print("🧪 T=250 5-Fold Cross-Validation TEST")
    print("=" * 60)
    print(f"🕐 Start: {datetime.now().strftime('%H:%M:%S')}")
    print("⏱️ Expected duration: ~20 minutes")
    print("🎯 2 epochs per fold × 5 folds = 10 total epochs")
    print("🔍 Goal: Verify CV flow before full training")
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
        print("⚠️ No GPU available - training will be slow!")
        
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


def create_test_cv_config(dataset_path):
    """Create test configuration for CV."""
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
        
        # TEST configuration - quick epochs
        num_epochs=2,  # TEST: Only 2 epochs per fold
        early_stopping_patience=10,  # Won't trigger with 2 epochs
        
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
        experiment_name="t250_cv_test",
        run_name=f"t250_cv_test_2epochs_{timestamp}",
        save_model_every_n_epochs=1,  # Save every epoch for test
        
        # Random seed for reproducibility
        random_seed=42
    )
    
    print("\n🔧 TEST CV Configuration:")
    print(f"   Dataset: T=250 ({dataset_path})")
    print(f"   Hyperparameters: lr={config.learning_rate}, bs={config.batch_size}, opt={config.optimizer}")
    print(f"   Epochs per fold: {config.num_epochs} (TEST MODE)")
    print(f"   Total training time: ~{config.num_epochs * 5 * 2} minutes")
    print(f"   Device: {config.device}")
    print(f"   Run name: {config.run_name}")
    
    return config


def run_cv_test(config):
    """Run the 5-fold CV test."""
    print("\n🚀 Starting T=250 5-Fold CV Test...")
    
    try:
        # Create CV trainer
        cv_trainer = CrossValidationTrainer(config, k_folds=5)
        
        # Start CV training
        start_time = time.time()
        cv_results = cv_trainer.run_cross_validation()
        end_time = time.time()
        
        total_time = (end_time - start_time) / 60  # minutes
        print(f"\n⏱️ CV test completed in {total_time:.1f} minutes")
        
        return cv_results, total_time
        
    except Exception as e:
        print(f"\n❌ CV test failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def analyze_cv_results(cv_results, total_time):
    """Analyze and display CV test results."""
    if cv_results is None:
        print("❌ No CV results available - training failed")
        return None
    
    print("\n" + "=" * 60)
    print("🎉 T=250 5-FOLD CV TEST COMPLETE!")
    print("=" * 60)
    
    print(f"⏱️ Total Time: {total_time:.1f} minutes")
    print(f"📊 Test Results (2 epochs per fold):")
    
    # Extract key metrics
    mean_distance_error = cv_results['mean_distance_error']
    std_distance_error = cv_results['std_distance_error']
    mean_val_loss = cv_results['mean_val_loss']
    std_val_loss = cv_results['std_val_loss']
    
    print(f"   Mean Distance Error: {mean_distance_error:.3f} ± {std_distance_error:.3f} px")
    print(f"   Mean Val Loss: {mean_val_loss:.4f} ± {std_val_loss:.4f}")
    
    # Fold-by-fold results
    print(f"\n📋 Individual Fold Results:")
    fold_results = cv_results['fold_results']
    for i, fold in enumerate(fold_results):
        print(f"   Fold {i+1}: {fold['distance_error']:.3f} px, {fold['val_loss']:.4f} loss")
    
    # Assessment for 2-epoch test
    print(f"\n📈 Test Assessment:")
    print(f"   ✅ Pipeline works: All 5 folds completed successfully")
    print(f"   ⏱️ Time per fold: {total_time/5:.1f} minutes")
    print(f"   📊 Results spread: {std_distance_error:.3f} px std deviation")
    
    # Projection to full training
    estimated_full_time = (total_time / 2) * 75  # Scale from 2 to 75 epochs
    print(f"\n🔮 Full Training Projection (75 epochs per fold):")
    print(f"   ⏱️ Estimated time: ~{estimated_full_time:.0f} minutes ({estimated_full_time/60:.1f} hours)")
    print(f"   📈 Expected improvement: Much better than {mean_distance_error:.3f} px")
    print(f"   🎯 Target: Similar to validation result (~2.2 px)")
    
    # Recommendation
    if mean_distance_error < 10.0:  # Reasonable for 2-epoch test
        recommendation = "✅ PROCEED: CV flow works, ready for full training"
    else:
        recommendation = "⚠️ INVESTIGATE: Results seem unusual for 2-epoch test"
    
    print(f"\n🎯 RECOMMENDATION:")
    print(f"   {recommendation}")
    
    return {
        'mean_distance_error': mean_distance_error,
        'std_distance_error': std_distance_error,
        'total_time': total_time,
        'estimated_full_time': estimated_full_time,
        'recommendation': recommendation,
        'cv_results': cv_results
    }


def save_test_results(config, results):
    """Save test results to Drive."""
    if results is None:
        return
    
    print("\n💾 Saving CV test results...")
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # Create test results directory in Drive  
    drive_project_path = '/content/drive/MyDrive/Physics_Informed_DL_Project'
    test_dir = Path(f"{drive_project_path}/results/t250_cv_test_{timestamp}")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Save test summary
    test_summary = {
        'timestamp': datetime.now().isoformat(),
        'experiment_type': 'T250_5fold_CV_test',
        'test_config': {
            'epochs_per_fold': config.num_epochs,
            'total_folds': 5,
            'dataset': 'wave_dataset_T250.h5'
        },
        'hyperparameters': {
            'learning_rate': config.learning_rate,
            'batch_size': config.batch_size,
            'optimizer': config.optimizer,
            'weight_decay': config.weight_decay
        },
        'results': results,
        'mlflow_experiment_name': config.experiment_name,
        'mlflow_run_name': config.run_name
    }
    
    # Save summary JSON
    summary_file = test_dir / f"cv_test_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(test_summary, f, indent=2)
    print(f"✅ Summary saved: {summary_file}")
    
    # Create test report
    report_file = test_dir / f"CV_TEST_REPORT_{timestamp}.md"
    with open(report_file, 'w') as f:
        f.write(f"# T=250 5-Fold CV Test Report\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## 🧪 Test Objective\n")
        f.write(f"Quick 5-fold CV test with 2 epochs per fold to verify pipeline before full training.\n\n")
        f.write(f"## 📊 Test Results\n")
        f.write(f"- **Mean Distance Error**: {results['mean_distance_error']:.3f} ± {results['std_distance_error']:.3f} px\n")
        f.write(f"- **Total Test Time**: {results['total_time']:.1f} minutes\n")
        f.write(f"- **Time per Fold**: {results['total_time']/5:.1f} minutes\n\n")
        f.write(f"## 🔮 Full Training Projection\n")
        f.write(f"- **Estimated Full Time**: {results['estimated_full_time']/60:.1f} hours\n")
        f.write(f"- **Expected Performance**: ~2.2 px (similar to validation)\n\n")
        f.write(f"## 🎯 Recommendation\n")
        f.write(f"{results['recommendation']}\n\n")
        f.write(f"## 🚀 Next Steps\n")
        f.write(f"If test passed, run full 75-epoch CV training:\n")
        f.write(f"- Create `run_t250_cv_full.py` script\n")
        f.write(f"- Set `num_epochs=75`\n")
        f.write(f"- Expected time: ~{results['estimated_full_time']/60:.1f} hours\n")
        f.write(f"- Expected results: Competitive with T=500 baseline\n\n")
        f.write(f"## 📁 Test Files\n")
        f.write(f"- `cv_test_summary_{timestamp}.json`: Complete test data\n")
        f.write(f"- `CV_TEST_REPORT_{timestamp}.md`: This report\n")
    
    print(f"✅ Report saved: {report_file}")
    
    print(f"\n🎉 TEST RESULTS SAVED TO DRIVE!")
    print(f"📁 Location: {test_dir}")
    print(f"🔄 Results synced to Google Drive!")


def main():
    """Main CV test function."""
    print_banner()
    
    # Check environment and dataset
    dataset_path = check_environment()
    if dataset_path is None:
        return
    
    # Create test configuration
    config = create_test_cv_config(dataset_path)
    
    # Run CV test
    cv_results, total_time = run_cv_test(config)
    
    # Analyze results
    results = analyze_cv_results(cv_results, total_time)
    
    # Save results
    save_test_results(config, results)
    
    print("\n" + "=" * 60)
    print("🧪 T=250 5-Fold CV Test Complete!")
    print("Check the recommendation above for next steps.")
    print("=" * 60)


if __name__ == "__main__":
    main() 