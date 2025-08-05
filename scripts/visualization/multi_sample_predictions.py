#!/usr/bin/env python3
"""
Multi-Sample Prediction Visualization Script
Creates comprehensive prediction comparison plots similar to the attached reference.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from datetime import datetime

# Add project root to path
sys.path.append('.')
sys.path.append('src')

from data.wave_dataset import WaveDataset
from models.wave_source_resnet import WaveSourceMiniResNet


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
                    
                    print(f"✅ Best fold determined: Fold {best_fold} ({min_error:.3f} px)")
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
        return best_model_path, model_info
    else:
        print(f"❌ Could not find best fold model for fold {best_fold}")
        return None, None


def load_model_and_dataset():
    """Load the best t250 model and validation dataset."""
    # Find best T=250 model
    model_path, model_info = find_best_t250_model()
    if model_path is None:
        print("❌ Could not find T=250 model")
        return None, None, None
    
    # Load T=250 validation dataset
    dataset_path = "data/wave_dataset_T250_validation.h5"
    if not Path(dataset_path).exists():
        print(f"❌ T=250 validation dataset not found: {dataset_path}")
        return None, None, None
    
    print(f"📊 Loading T=250 validation dataset: {dataset_path}")
    
    # Load dataset
    try:
        dataset = WaveDataset(dataset_path)
        print(f"✅ Loaded dataset with {len(dataset)} samples")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None, None, None
    
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
        
        return model, dataset, device
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def create_multi_sample_prediction_plot(model, dataset, device, sample_indices=None, save_path=None):
    """
    Create a comprehensive multi-sample prediction visualization.
    
    Args:
        model: Trained model
        dataset: Validation dataset
        device: Torch device
        sample_indices: List of sample indices to visualize (if None, use first 5)
        save_path: Path to save the plot
    """
    print("🎨 Creating multi-sample prediction visualization...")
    
    # Default to first 5 samples if not specified
    if sample_indices is None:
        sample_indices = [0, 1, 2, 3, 4]
    
    num_samples = len(sample_indices)
    
    # Create figure with 3 columns: Wave Field, Prediction Comparison, Coordinate Space
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    # Collect all predictions for statistics
    all_errors = []
    
    with torch.no_grad():
        for i, sample_idx in enumerate(sample_indices):
            # Get sample
            wave_data, target_coords = dataset[sample_idx]
            target_coords = target_coords.numpy()
            
            # Get prediction
            wave_input = wave_data.unsqueeze(0).to(device)
            pred_coords = model(wave_input)
            pred_coords = pred_coords.cpu().numpy().squeeze()
            
            # Calculate error
            error = np.sqrt(np.sum((pred_coords - target_coords) ** 2))
            all_errors.append(error)
            
            # Extract wave field for visualization (remove batch and channel dims)
            wave_field = wave_data.squeeze().numpy()
            
            # Column 1: Original Wave Field
            im1 = axes[i, 0].imshow(wave_field, cmap='RdBu_r', origin='lower', extent=[0, 127, 0, 127])
            axes[i, 0].set_title(f'Sample {sample_idx}\nWave Field')
            axes[i, 0].set_xlabel('X Position')
            axes[i, 0].set_ylabel('Y Position')
            
            # Column 2: Prediction Comparison (overlay on wave field)
            im2 = axes[i, 1].imshow(wave_field, cmap='RdBu_r', origin='lower', extent=[0, 127, 0, 127])
            
            # Mark true and predicted positions
            axes[i, 1].plot(target_coords[0], target_coords[1], 'x', markersize=12, 
                           markeredgewidth=3, color='red', label='True')
            axes[i, 1].plot(pred_coords[0], pred_coords[1], 'o', markersize=10, 
                           markerfacecolor='none', markeredgecolor='yellow', 
                           markeredgewidth=3, label='Predicted')
            
            axes[i, 1].set_title(f'Prediction Comparison\nError: {error:.2f} px')
            axes[i, 1].set_xlabel('X Position')
            axes[i, 1].set_ylabel('Y Position')
            axes[i, 1].legend()
            
            # Column 3: Coordinate Space (scatter plot style)
            axes[i, 2].set_xlim(-10, 140)
            axes[i, 2].set_ylim(-10, 140)
            
            # Draw grid background
            axes[i, 2].grid(True, alpha=0.3)
            axes[i, 2].set_facecolor('lightgray')
            
            # Mark positions
            axes[i, 2].plot(target_coords[0], target_coords[1], 'x', markersize=15, 
                           markeredgewidth=4, color='red', label='True')
            axes[i, 2].plot(pred_coords[0], pred_coords[1], 'o', markersize=12, 
                           markerfacecolor='none', markeredgecolor='black', 
                           markeredgewidth=3, label='Predicted')
            
            # Add coordinate text box
            coord_text = f'Sample {sample_idx}\n\n' \
                        f'True Position:\nx = {target_coords[0]:.1f}\ny = {target_coords[1]:.1f}\n\n' \
                        f'Predicted Position:\nx = {pred_coords[0]:.1f}\ny = {pred_coords[1]:.1f}\n\n' \
                        f'Error: {error:.2f} px\n\n' \
                        f'Individual Errors:\nΔx = {abs(pred_coords[0] - target_coords[0]):.2f}\n' \
                        f'Δy = {abs(pred_coords[1] - target_coords[1]):.2f}'
            
            axes[i, 2].text(0.98, 0.98, coord_text, transform=axes[i, 2].transAxes,
                           verticalalignment='top', horizontalalignment='right',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                           fontsize=9, fontfamily='monospace')
            
            axes[i, 2].set_title('Coordinate Space\nΔx: ±, Δy: ±')
            axes[i, 2].set_xlabel('X Position')
            axes[i, 2].set_ylabel('Y Position')
            axes[i, 2].legend()
            
            # Add colorbars for wave field plots
            if i == 0:  # Only add colorbar to first row
                plt.colorbar(im1, ax=axes[i, 0], fraction=0.046, pad=0.04)
                plt.colorbar(im2, ax=axes[i, 1], fraction=0.046, pad=0.04)
    
    # Add overall statistics as suptitle
    mean_error = np.mean(all_errors)
    std_error = np.std(all_errors)
    fig.suptitle(f'T=250 Model Multi-Sample Predictions\n' +
                f'Mean Error: {mean_error:.2f} ± {std_error:.2f} px | Samples: {num_samples}',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Multi-sample prediction plot saved: {save_path}")
    
    plt.show()
    
    return all_errors


def main():
    """Main function to create multi-sample prediction visualization."""
    print("🎯 T=250 Model Multi-Sample Prediction Visualization")
    print("=" * 60)
    
    # Load model and dataset
    model, dataset, device = load_model_and_dataset()
    if model is None:
        return
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    results_dir = Path(f"experiments/t250_visualization/multi_sample_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Ask user for sample selection
    print(f"\n📊 Dataset has {len(dataset)} samples available")
    print("Choose visualization option:")
    print("1. First 5 samples (default)")
    print("2. Specific sample indices")
    print("3. Random selection")
    
    choice = input("Enter choice (1-3, or press Enter for default): ").strip()
    
    sample_indices = None
    if choice == "2":
        indices_input = input("Enter sample indices (comma-separated, e.g., 10,25,50,100,200): ")
        try:
            sample_indices = [int(x.strip()) for x in indices_input.split(',')]
            # Validate indices
            max_idx = len(dataset) - 1
            sample_indices = [idx for idx in sample_indices if 0 <= idx <= max_idx]
            print(f"Using samples: {sample_indices}")
        except:
            print("Invalid input, using default...")
            sample_indices = None
    
    elif choice == "3":
        num_samples = input("How many random samples? (default 5): ").strip()
        try:
            num_samples = int(num_samples) if num_samples else 5
            sample_indices = np.random.choice(len(dataset), size=min(num_samples, len(dataset)), 
                                            replace=False).tolist()
            print(f"Using random samples: {sample_indices}")
        except:
            print("Invalid input, using default...")
            sample_indices = None
    
    # Default to first 5 samples
    if sample_indices is None:
        sample_indices = [0, 1, 2, 3, 4]
        print(f"Using default samples: {sample_indices}")
    
    # Create visualization
    save_path = results_dir / f"multi_sample_predictions_{timestamp}.png"
    errors = create_multi_sample_prediction_plot(model, dataset, device, 
                                               sample_indices, save_path)
    
    print(f"\n✅ Visualization completed!")
    print(f"📁 Results saved to: {results_dir}")
    print(f"📊 Sample errors: {[f'{e:.2f}' for e in errors]} px")


if __name__ == "__main__":
    main()