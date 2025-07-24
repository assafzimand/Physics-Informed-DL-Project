#!/usr/bin/env python3
"""
T=250 Extra Validation Script
Tests best T=250 trained model on dedicated T=250 validation dataset.
"""

import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import torch
from scipy import stats

# Add project root to path
sys.path.append('.')
sys.path.append('src')

from data.wave_dataset import WaveDataset
from models.wave_source_resnet import WaveSourceMiniResNet


def print_banner():
    """Print validation banner."""
    print("🔍 T=250 Extra Model Validation")
    print("=" * 60)
    print(f"🕐 Start: {datetime.now().strftime('%H:%M:%S')}")
    print("🎯 Testing best T=250 model on dedicated T=250 validation dataset")
    print("📊 Statistical analysis with histograms and error distributions")
    print("=" * 60)


def find_best_t250_model():
    """Find the best T=250 model from CV experiment."""
    experiment_dir = "experiments/t250_cv_full"
    print(f"🔍 Searching for T=250 CV models in: {experiment_dir}")
    
    # Look for models in the actual directory structure
    models_dir = Path(experiment_dir) / "data" / "models"
    analysis_dir = Path(experiment_dir) / "analysis"
    
    if not models_dir.exists():
        print(f"❌ Models directory not found: {models_dir}")
        return None, None
    
    # Look for fold models
    fold_models = list(models_dir.glob("*fold_*.pt*"))
    if not fold_models:
        print(f"❌ No fold models found in {models_dir}")
        return None, None
    
    print(f"✅ Found {len(fold_models)} fold models")
    
    # Try to find the best fold from CV results summary
    best_fold = None
    cv_summary_file = analysis_dir / "cv_results_summary.txt"
    
    if cv_summary_file.exists():
        print(f"📊 Reading CV results from: {cv_summary_file}")
        try:
            with open(cv_summary_file, 'r') as f:
                content = f.read()
                
            # Parse the individual fold errors line
            for line in content.split('\n'):
                if line.startswith('Individual Fold Errors:'):
                    # Extract the list of errors
                    errors_str = line.split('[')[1].split(']')[0]
                    errors = [float(x.strip()) for x in errors_str.split(',')]
                    
                    # Find the fold with minimum error (1-indexed)
                    best_fold = errors.index(min(errors)) + 1
                    min_error = min(errors)
                    
                    print(f"✅ Best fold determined from CV results: Fold {best_fold} ({min_error:.3f} px)")
                    break
                    
        except Exception as e:
            print(f"⚠️ Could not parse CV results: {e}")
    
    if not best_fold:
        print("⚠️ Could not determine best fold, using fold 1 as fallback")
        best_fold = 1
    
    # Find the best fold model
    best_model_path = None
    for model_path in fold_models:
        if f"fold_{best_fold}" in model_path.name:
            best_model_path = model_path
            break
    
    if best_model_path and best_model_path.exists():
        model_info = {
            'model_name': best_model_path.name,
            'fold': best_fold,
            'note': f'Best performing fold from CV results (fold {best_fold})'
        }
        print(f"✅ Selected best T=250 model: {best_model_path}")
        print(f"📊 Model info: {model_info}")
        return best_model_path, model_info
    else:
        print(f"❌ Could not find best fold model for fold {best_fold}")
        return None, None


def run_t250_validation():
    """Run validation on T=250 model and dataset."""
    print_banner()
    
    # Find best T=250 model
    model_path, model_info = find_best_t250_model()
    if model_path is None:
        print("❌ Could not find T=250 model")
        return
    
    # Load T=250 validation dataset
    dataset_path = "data/wave_dataset_T250_validation.h5"
    if not Path(dataset_path).exists():
        print(f"❌ T=250 validation dataset not found: {dataset_path}")
        return
    
    print(f"📊 Loading T=250 validation dataset: {dataset_path}")
    
    # Load dataset
    try:
        dataset = WaveDataset(dataset_path)
        print(f"✅ Loaded dataset with {len(dataset)} samples")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Load model
    print(f"🤖 Loading T=250 model: {model_path}")
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {device}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract model state dict and config
        if 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
            config = checkpoint.get('config', {})
        else:
            model_state_dict = checkpoint
            config = {}
        
        # Create model
        grid_size = config.get('grid_size', 128)
        model = WaveSourceMiniResNet(grid_size=grid_size)
        model.load_state_dict(model_state_dict)
        model.to(device)
        model.eval()
        
        print(f"✅ Model loaded successfully (grid_size: {grid_size})")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Run inference on all validation samples
    print("🚀 Running inference on validation samples...")
    
    predictions = []
    ground_truths = []
    distance_errors = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for i, (wave_data, target_coords) in enumerate(dataset):
            if i % 100 == 0:
                print(f"   Processing sample {i+1}/{len(dataset)}")
            
            # Add batch dimension and move to device
            wave_data = wave_data.unsqueeze(0).to(device)
            
            # Get prediction
            pred_coords = model(wave_data)
            pred_coords = pred_coords.cpu().numpy().squeeze()
            
            # Convert target to numpy
            target_coords = target_coords.numpy()
            
            # Calculate distance error
            distance_error = np.sqrt(np.sum((pred_coords - target_coords) ** 2))
            
            predictions.append(pred_coords)
            ground_truths.append(target_coords)
            distance_errors.append(distance_error)
    
    inference_time = time.time() - start_time
    print(f"✅ Inference completed in {inference_time:.2f}s")
    
    # Convert to arrays
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)
    distance_errors = np.array(distance_errors)
    
    # Calculate statistics
    mean_error = np.mean(distance_errors)
    std_error = np.std(distance_errors)
    median_error = np.median(distance_errors)
    min_error = np.min(distance_errors)
    max_error = np.max(distance_errors)
    
    print("\n" + "="*60)
    print("📊 T=250 VALIDATION RESULTS")
    print("="*60)
    print(f"🎯 Mean Distance Error: {mean_error:.3f} ± {std_error:.3f} px")
    print(f"📈 Median Distance Error: {median_error:.3f} px")
    print(f"📏 Min/Max Error: {min_error:.3f} / {max_error:.3f} px")
    print(f"📦 Total Samples: {len(distance_errors)}")
    print(f"⏱️ Inference Time: {inference_time:.2f}s")
    print("="*60)
    
    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    results_dir = Path(f"experiments/t250_cv_full/extra_validation/validation_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate comprehensive analysis plots
    create_analysis_plots(distance_errors, predictions, ground_truths, results_dir, model_info)
    
    # Save detailed results
    save_validation_results(distance_errors, predictions, ground_truths, model_info, 
                          results_dir, timestamp, inference_time)
    
    print(f"\n✅ T=250 validation completed successfully!")
    print(f"📁 Results saved to: {results_dir}")


def create_analysis_plots(distance_errors, predictions, ground_truths, results_dir, model_info):
    """Create comprehensive analysis plots matching T=500 layout."""
    print("\n📊 Generating analysis plots...")
    
    # Calculate component-wise errors
    x_errors = predictions[:, 0] - ground_truths[:, 0]
    y_errors = predictions[:, 1] - ground_truths[:, 1]
    
    # Calculate performance thresholds
    thresholds = [1.0, 2.0, 3.0]
    performance_stats = {}
    for threshold in thresholds:
        percentage = (np.sum(distance_errors <= threshold) / len(distance_errors)) * 100
        performance_stats[f'under_{threshold}px'] = percentage
    
    # Set up the plot style (matching T=500)
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('T=250 Model Extra Validation Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Distance Error Histogram
    axes[0, 0].hist(distance_errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(np.mean(distance_errors), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(distance_errors):.3f} px')
    axes[0, 0].axvline(np.median(distance_errors), color='orange', linestyle='--',
                       label=f'Median: {np.median(distance_errors):.3f} px')
    axes[0, 0].set_xlabel('Distance Error (px)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distance Error Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Q-Q Plot for Normality
    stats.probplot(distance_errors, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot (Normality Check)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: X vs Y Error Correlation
    axes[0, 2].scatter(x_errors, y_errors, alpha=0.6, s=20)
    axes[0, 2].axhline(0, color='red', linestyle='--', alpha=0.5)
    axes[0, 2].axvline(0, color='red', linestyle='--', alpha=0.5)
    axes[0, 2].set_xlabel('X Error')
    axes[0, 2].set_ylabel('Y Error')
    axes[0, 2].set_title('X vs Y Error Correlation')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Predictions vs Ground Truth
    axes[1, 0].scatter(ground_truths[:, 0], predictions[:, 0], alpha=0.6, s=20, label='X coordinate')
    axes[1, 0].scatter(ground_truths[:, 1], predictions[:, 1], alpha=0.6, s=20, label='Y coordinate')
    
    # Perfect prediction line
    min_coord = min(ground_truths.min(), predictions.min())
    max_coord = max(ground_truths.max(), predictions.max())
    axes[1, 0].plot([min_coord, max_coord], [min_coord, max_coord], 'r--', alpha=0.8, label='Perfect prediction')
    
    axes[1, 0].set_xlabel('Ground Truth')
    axes[1, 0].set_ylabel('Prediction')
    axes[1, 0].set_title('Predictions vs Ground Truth')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Error by Sample Index
    sample_indices = range(len(distance_errors))
    axes[1, 1].plot(sample_indices, distance_errors, alpha=0.7, linewidth=0.5)
    axes[1, 1].axhline(np.mean(distance_errors), color='red', linestyle='--', alpha=0.8)
    axes[1, 1].set_xlabel('Sample Index')
    axes[1, 1].set_ylabel('Distance Error (px)')
    axes[1, 1].set_title('Error by Sample')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Performance Summary
    summary_text = f"""T=250 Validation Summary

Statistics:
Mean Error: {np.mean(distance_errors):.3f} px
Std Deviation: {np.std(distance_errors):.3f} px
Median Error: {np.median(distance_errors):.3f} px

95% Confidence Interval:
{np.percentile(distance_errors, 2.5):.3f} - {np.percentile(distance_errors, 97.5):.3f} px

Performance Thresholds:
Under 1px: {performance_stats['under_1.0px']:.1f}%
Under 2px: {performance_stats['under_2.0px']:.1f}%
Under 3px: {performance_stats['under_3.0px']:.1f}%

Samples: {len(distance_errors)}
    """
    
    axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].set_xticks([])
    axes[1, 2].set_yticks([])
    axes[1, 2].set_title('Validation Summary')
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    plot_path = results_dir / f"validation_analysis_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Analysis plots saved: {plot_path}")
    plt.show()


def save_validation_results(distance_errors, predictions, ground_truths, model_info, 
                          results_dir, timestamp, inference_time):
    """Save detailed validation results to JSON."""
    print("💾 Saving validation results...")
    
    # Prepare results dictionary
    results = {
        'timestamp': timestamp,
        'model_info': model_info,
        'dataset_info': {
            'dataset_path': 'data/wave_dataset_T250_validation.h5',
            'T_value': 250,
            'num_samples': len(distance_errors),
            'description': 'T=250 dedicated validation dataset (500 samples)'
        },
        'performance_metrics': {
            'mean_distance_error': float(np.mean(distance_errors)),
            'std_distance_error': float(np.std(distance_errors)),
            'median_distance_error': float(np.median(distance_errors)),
            'min_distance_error': float(np.min(distance_errors)),
            'max_distance_error': float(np.max(distance_errors)),
            'percentile_25': float(np.percentile(distance_errors, 25)),
            'percentile_75': float(np.percentile(distance_errors, 75)),
            'percentile_95': float(np.percentile(distance_errors, 95)),
            'percentile_99': float(np.percentile(distance_errors, 99))
        },
        'statistical_tests': {},
        'runtime_info': {
            'inference_time_seconds': inference_time,
            'samples_per_second': len(distance_errors) / inference_time,
            'device_used': str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        }
    }
    
    # Statistical tests
    if len(distance_errors) <= 5000:
        shapiro_stat, shapiro_p = stats.shapiro(distance_errors)
        results['statistical_tests']['normality_test'] = {
            'test': 'Shapiro-Wilk',
            'statistic': float(shapiro_stat),
            'p_value': float(shapiro_p),
            'is_normal': bool(shapiro_p > 0.05)
        }
    
    # Save results
    results_file = results_dir / f"validation_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved: {results_file}")


if __name__ == "__main__":
    run_t250_validation() 