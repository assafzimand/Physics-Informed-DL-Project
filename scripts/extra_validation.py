#!/usr/bin/env python3
"""
Extra Validation Script
Tests best trained models on dedicated validation datasets with comprehensive statistical analysis.
"""

import sys
import os
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

from src.inference import WaveSourceInference
from src.data.wave_dataset import WaveDataset
from src.models.wave_source_resnet import WaveSourceMiniResNet


def print_banner():
    """Print validation banner."""
    print("🔍 Extra Model Validation")
    print("=" * 60)
    print(f"🕐 Start: {datetime.now().strftime('%H:%M:%S')}")
    print("🎯 Testing best models on dedicated validation datasets")
    print("📊 Statistical analysis with histograms and error distributions")
    print("=" * 60)


def find_best_model(experiment_dir):
    """Find the best model from CV experiment."""
    print(f"🔍 Searching for CV models in: {experiment_dir}")
    
    # Look for models in the actual directory structure
    models_dir = Path(experiment_dir) / "data" / "models"
    analysis_dir = Path(experiment_dir) / "analysis"
    
    if not models_dir.exists():
        print(f"❌ Models directory not found: {models_dir}")
        return None, None
    
    # Look for fold models
    fold_models = list(models_dir.glob("*fold_*_best.pth"))
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
        print("⚠️ Could not determine best fold, using fold 2 as fallback")
        best_fold = 2
    
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
        print(f"✅ Using best model: {best_model_path.name}")
        return str(best_model_path), model_info
    else:
        # Fallback to first available model
        model_path = fold_models[0]
        model_info = {
            'model_name': model_path.name,
            'fold': 'unknown',
            'note': 'Using first available fold model (could not find best fold model)'
        }
        print(f"⚠️ Using fallback model: {model_path.name}")
        return str(model_path), model_info


def load_validation_dataset(T_value):
    """Load validation dataset for specific T value."""
    dataset_path = f"data/wave_dataset_T{T_value}_validation.h5"
    
    print(f"📊 Loading T={T_value} validation dataset: {dataset_path}")
    
    if not Path(dataset_path).exists():
        print(f"❌ Validation dataset not found: {dataset_path}")
        print(f"💡 Run: python scripts/generate_validation_datasets.py")
        return None
    
    try:
        dataset = WaveDataset(
            hdf5_path=dataset_path,
            normalize_wave_fields=True,
            normalize_coordinates=False,
            grid_size=128
        )
        
        print(f"✅ Loaded {len(dataset)} validation samples")
        return dataset
        
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return None


def run_model_validation(model_path, dataset, T_value):
    """Run validation on dataset and return results."""
    print(f"🚀 Running validation for T={T_value}...")
    
    try:
        # Load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model instance
        model = WaveSourceMiniResNet(grid_size=128)
        
        # Load model weights
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        
        print(f"✅ Model loaded on {device}")
        
        # Run inference on all validation samples
        predictions = []
        ground_truths = []
        
        print(f"🔄 Processing {len(dataset)} samples...")
        
        with torch.no_grad():
            for i, (wave_field, coordinates) in enumerate(dataset):
                if i % 50 == 0:
                    print(f"   Progress: {i}/{len(dataset)} samples")
                
                # Prepare input
                wave_field = wave_field.unsqueeze(0).to(device)  # Add batch dimension
                
                # Get prediction
                pred_coords = model(wave_field)
                pred_coords = pred_coords.cpu().numpy().flatten()
                
                # Store results
                predictions.append(pred_coords)
                ground_truths.append(coordinates.numpy())
        
        predictions = np.array(predictions)
        ground_truths = np.array(ground_truths)
        
        print(f"✅ Validation complete: {len(predictions)} predictions")
        
        return predictions, ground_truths
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def calculate_error_statistics(predictions, ground_truths):
    """Calculate comprehensive error statistics."""
    print("📊 Calculating error statistics...")
    
    # Calculate distance errors
    distance_errors = np.sqrt(np.sum((predictions - ground_truths) ** 2, axis=1))
    
    # Calculate component-wise errors
    x_errors = predictions[:, 0] - ground_truths[:, 0]
    y_errors = predictions[:, 1] - ground_truths[:, 1]
    
    # Basic statistics
    stats_dict = {
        'distance_error': {
            'mean': float(np.mean(distance_errors)),
            'std': float(np.std(distance_errors)),
            'median': float(np.median(distance_errors)),
            'min': float(np.min(distance_errors)),
            'max': float(np.max(distance_errors)),
            'q25': float(np.percentile(distance_errors, 25)),
            'q75': float(np.percentile(distance_errors, 75))
        },
        'x_error': {
            'mean': float(np.mean(x_errors)),
            'std': float(np.std(x_errors)),
            'median': float(np.median(x_errors))
        },
        'y_error': {
            'mean': float(np.mean(y_errors)),
            'std': float(np.std(y_errors)),
            'median': float(np.median(y_errors))
        }
    }
    
    # Statistical tests
    # Normality test (Shapiro-Wilk)
    if len(distance_errors) <= 5000:  # Shapiro-Wilk works best for smaller samples
        shapiro_stat, shapiro_p = stats.shapiro(distance_errors)
        stats_dict['normality_test'] = {
            'test': 'Shapiro-Wilk',
            'statistic': float(shapiro_stat),
            'p_value': float(shapiro_p),
            'is_normal': bool(shapiro_p > 0.05)  # Convert to JSON-serializable bool
        }
    
    # Confidence intervals (95%)
    confidence_level = 0.95
    alpha = 1 - confidence_level
    df = len(distance_errors) - 1
    t_critical = stats.t.ppf(1 - alpha/2, df)
    
    mean_error = stats_dict['distance_error']['mean']
    std_error = stats_dict['distance_error']['std']
    margin_error = t_critical * (std_error / np.sqrt(len(distance_errors)))
    
    stats_dict['confidence_interval_95'] = {
        'lower': float(mean_error - margin_error),
        'upper': float(mean_error + margin_error),
        'margin': float(margin_error)
    }
    
    # Performance thresholds
    thresholds = [1.0, 2.0, 3.0, 5.0]
    stats_dict['performance_thresholds'] = {}
    for threshold in thresholds:
        percentage = (np.sum(distance_errors <= threshold) / len(distance_errors)) * 100
        stats_dict['performance_thresholds'][f'under_{threshold}px'] = float(percentage)
    
    print(f"✅ Statistics calculated: {mean_error:.3f} ± {std_error:.3f} px")
    
    return stats_dict, distance_errors, x_errors, y_errors


def create_validation_plots(distance_errors, x_errors, y_errors, predictions, ground_truths, T_value, stats_dict):
    """Create comprehensive validation plots."""
    print("📈 Creating validation plots...")
    
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'T={T_value} Model Extra Validation Analysis', fontsize=16, fontweight='bold')
    
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
    summary_text = f"""
T={T_value} Validation Summary

Statistics:
Mean Error: {stats_dict['distance_error']['mean']:.3f} px
Std Deviation: {stats_dict['distance_error']['std']:.3f} px
Median Error: {stats_dict['distance_error']['median']:.3f} px

95% Confidence Interval:
{stats_dict['confidence_interval_95']['lower']:.3f} - {stats_dict['confidence_interval_95']['upper']:.3f} px

Performance Thresholds:
Under 1px: {stats_dict['performance_thresholds']['under_1.0px']:.1f}%
Under 2px: {stats_dict['performance_thresholds']['under_2.0px']:.1f}%
Under 3px: {stats_dict['performance_thresholds']['under_3.0px']:.1f}%

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
    
    print("✅ Validation plots created")
    return fig


def save_validation_results(T_value, stats_dict, model_info, fig):
    """Save validation results to appropriate experiment folder."""
    print("💾 Saving validation results...")
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # Determine output directory based on T value
    if T_value == 250:
        base_dir = Path("experiments/t250_cv_full")
    elif T_value == 500:
        base_dir = Path("experiments/cv_full")
    else:
        base_dir = Path(f"experiments/t{T_value}_validation")
    
    # Create validation subdirectory
    validation_dir = base_dir / "extra_validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped results directory
    results_dir = validation_dir / f"validation_{timestamp}"
    results_dir.mkdir(exist_ok=True)
    
    # Save complete validation results
    validation_results = {
        'timestamp': datetime.now().isoformat(),
        'experiment_type': f'T{T_value}_extra_validation',
        'dataset': f'wave_dataset_T{T_value}_validation.h5',
        'model_info': model_info,
        'statistics': stats_dict,
        'validation_config': {
            'samples': len(stats_dict['distance_error']),  # Will be corrected
            'T_value': T_value,
            'validation_type': 'dedicated_validation_dataset'
        }
    }
    
    # Save results JSON
    results_file = results_dir / f"validation_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(validation_results, f, indent=2)
    print(f"✅ Results saved: {results_file}")
    
    # Save validation plots
    if fig is not None:
        plot_file = results_dir / f"validation_analysis_{timestamp}.png"
        fig.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"✅ Plots saved: {plot_file}")
    
    # Create detailed report
    report_file = results_dir / f"VALIDATION_REPORT_{timestamp}.md"
    with open(report_file, 'w') as f:
        f.write(f"# T={T_value} Extra Validation Report\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## 🎯 Validation Overview\n")
        f.write(f"Extra validation of T={T_value} best model on dedicated 500-sample validation dataset.\n\n")
        
        if model_info:
            f.write(f"### Model Information\n")
            f.write(f"- **Best Fold**: {model_info['fold']}\n")
            f.write(f"- **CV Performance**: {model_info['distance_error']:.3f} px\n")
            f.write(f"- **Model Path**: {model_info['model_path']}\n\n")
        
        f.write(f"## 📊 Validation Results\n")
        f.write(f"- **Mean Error**: {stats_dict['distance_error']['mean']:.3f} ± {stats_dict['distance_error']['std']:.3f} px\n")
        f.write(f"- **Median Error**: {stats_dict['distance_error']['median']:.3f} px\n")
        f.write(f"- **Min/Max Error**: {stats_dict['distance_error']['min']:.3f} / {stats_dict['distance_error']['max']:.3f} px\n")
        f.write(f"- **95% Confidence Interval**: [{stats_dict['confidence_interval_95']['lower']:.3f}, {stats_dict['confidence_interval_95']['upper']:.3f}] px\n\n")
        
        f.write(f"## 📈 Performance Breakdown\n")
        for threshold, percentage in stats_dict['performance_thresholds'].items():
            f.write(f"- **{threshold.replace('_', ' ').title()}**: {percentage:.1f}%\n")
        
        f.write(f"\n## 🔬 Statistical Analysis\n")
        if 'normality_test' in stats_dict:
            normal_text = "Yes" if stats_dict['normality_test']['is_normal'] else "No"
            f.write(f"- **Error Distribution Normal**: {normal_text} (p={stats_dict['normality_test']['p_value']:.4f})\n")
        
        f.write(f"- **X Error Bias**: {stats_dict['x_error']['mean']:.4f} ± {stats_dict['x_error']['std']:.4f}\n")
        f.write(f"- **Y Error Bias**: {stats_dict['y_error']['mean']:.4f} ± {stats_dict['y_error']['std']:.4f}\n")
        
        f.write(f"\n## 📁 Generated Files\n")
        f.write(f"- `validation_results_{timestamp}.json`: Complete statistical data\n")
        f.write(f"- `validation_analysis_{timestamp}.png`: Comprehensive analysis plots\n")
        f.write(f"- `VALIDATION_REPORT_{timestamp}.md`: This detailed report\n")
        
        f.write(f"\n## 🎯 Conclusions\n")
        mean_error = stats_dict['distance_error']['mean']
        if mean_error <= 2.0:
            conclusion = "Excellent performance on validation dataset"
        elif mean_error <= 3.0:
            conclusion = "Good performance on validation dataset"
        else:
            conclusion = "Performance needs improvement"
        
        f.write(f"The T={T_value} model demonstrates {conclusion.lower()} with a mean distance error of {mean_error:.3f} pixels.")
    
    print(f"✅ Report saved: {report_file}")
    print(f"\n💾 All validation results saved to: {results_dir}")
    
    return results_dir


def main():
    """Main validation function."""
    print_banner()
    
    # Define experiments to validate
    experiments = [
        # {
        #     'T_value': 250,
        #     'experiment_dir': 'experiments/t250_cv_full',
        #     'description': 'T=250 CV Full Training'
        # },
        {
            'T_value': 500,
            'experiment_dir': 'experiments/cv_full',
            'description': 'T=500 CV Full Training'
        }
    ]
    
    for exp in experiments:
        T_value = exp['T_value']
        experiment_dir = exp['experiment_dir']
        
        print(f"\n🔍 Validating {exp['description']}...")
        print("=" * 40)
        
        # Find best model
        model_path, model_info = find_best_model(experiment_dir)
        if not model_path:
            print(f"⚠️ Skipping T={T_value}: No model found")
            continue
        
        # Load validation dataset
        dataset = load_validation_dataset(T_value)
        if not dataset:
            print(f"⚠️ Skipping T={T_value}: No validation dataset")
            continue
        
        # Run validation
        predictions, ground_truths = run_model_validation(model_path, dataset, T_value)
        if predictions is None:
            print(f"⚠️ Skipping T={T_value}: Validation failed")
            continue
        
        # Calculate statistics
        stats_dict, distance_errors, x_errors, y_errors = calculate_error_statistics(predictions, ground_truths)
        
        # Create plots
        fig = create_validation_plots(distance_errors, x_errors, y_errors, predictions, ground_truths, T_value, stats_dict)
        
        # Save results
        results_dir = save_validation_results(T_value, stats_dict, model_info, fig)
        
        print(f"✅ T={T_value} validation complete: {stats_dict['distance_error']['mean']:.3f} ± {stats_dict['distance_error']['std']:.3f} px")
        
        plt.show()
    
    print(f"\n🎉 Extra validation complete for all models!")


if __name__ == "__main__":
    main() 