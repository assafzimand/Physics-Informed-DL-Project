#!/usr/bin/env python3
"""
Test Inference with Best CV Model
Tests our best cross-validation model (Fold 2: 1.663 px average error)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
import sys

# Add src to path
sys.path.append('src')
from inference import WaveSourceInference
from wave_simulation import Wave2DSimulator

def test_best_cv_model():
    """Test our best CV model with some example predictions."""
    print("🏆 Testing Best CV Model (Fold 2: 1.663 px)")
    print("=" * 50)
    
    # Path to our best model
    best_model_path = "experiments/cv_full/data/models/cv_full_5fold_75epochs_fold_2_best.pth"
    
    if not Path(best_model_path).exists():
        print(f"❌ Model not found: {best_model_path}")
        return
    
    # Initialize inference pipeline
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Using device: {device}")
    
    try:
        # Create inference object
        inference = WaveSourceInference(best_model_path, device)
        
        # Test with dataset samples
        dataset_path = "data/wave_dataset_T500.h5"
        
        if Path(dataset_path).exists():
            print(f"📂 Loading test samples from {dataset_path}")
            test_with_dataset_samples(inference, dataset_path)
        else:
            print("📊 Generating synthetic test samples")
            test_with_synthetic_samples(inference)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def test_with_dataset_samples(inference, dataset_path, num_samples=5):
    """Test with real dataset samples."""
    print(f"\n🧪 Testing with {num_samples} dataset samples...")
    
    # Load dataset
    with h5py.File(dataset_path, 'r') as f:
        wave_fields = f['wave_fields']
        source_coords = f['source_coords']
        
        # Get random test samples
        total_samples = wave_fields.shape[0]
        test_indices = np.random.choice(total_samples, size=num_samples, replace=False)
        
        errors = []
        
        plt.figure(figsize=(20, 4 * num_samples))
        
        for i, idx in enumerate(test_indices):
            # Get sample
            wave_field = wave_fields[idx]  # Already the final timestep
            true_coords = source_coords[idx]
            
            # Predict
            pred_coords = inference.predict_source(wave_field)
            
            # Calculate error
            error = np.sqrt((pred_coords[0] - true_coords[0])**2 + 
                          (pred_coords[1] - true_coords[1])**2)
            errors.append(error)
            
            # Plot results
            plt.subplot(num_samples, 4, i*4 + 1)
            plt.imshow(wave_field, cmap='RdBu_r', aspect='equal')
            plt.title(f'Sample {idx}\nWave Field')
            plt.colorbar()
            
            plt.subplot(num_samples, 4, i*4 + 2)
            plt.imshow(wave_field, cmap='RdBu_r', aspect='equal')
            plt.scatter(*true_coords, color='red', s=100, marker='x', linewidth=3, label='True')
            plt.scatter(*pred_coords, color='yellow', s=100, marker='o', linewidth=2, label='Predicted')
            plt.title(f'Prediction Comparison\nError: {error:.2f} px')
            plt.legend()
            
            plt.subplot(num_samples, 4, i*4 + 3)
            # Error visualization
            plt.plot([true_coords[0], pred_coords[0]], [true_coords[1], pred_coords[1]], 
                    'r--', linewidth=2, alpha=0.7)
            plt.scatter(*true_coords, color='red', s=150, marker='x', linewidth=3, label='True')
            plt.scatter(*pred_coords, color='yellow', s=150, marker='o', linewidth=2, 
                       edgecolor='black', label='Predicted')
            plt.xlim(0, 128)
            plt.ylim(0, 128)
            plt.grid(True, alpha=0.3)
            plt.title(f'Coordinate Space\nΔx: {pred_coords[0]-true_coords[0]:.1f}, Δy: {pred_coords[1]-true_coords[1]:.1f}')
            plt.legend()
            
            plt.subplot(num_samples, 4, i*4 + 4)
            # Show prediction details
            details_text = f"""Sample {idx}
            
True Position:
x = {true_coords[0]:.1f}
y = {true_coords[1]:.1f}

Predicted Position:
x = {pred_coords[0]:.1f}
y = {pred_coords[1]:.1f}

Error: {error:.2f} px

Individual Errors:
Δx = {abs(pred_coords[0]-true_coords[0]):.2f}
Δy = {abs(pred_coords[1]-true_coords[1]):.2f}"""
            
            plt.text(0.1, 0.5, details_text, fontsize=10, transform=plt.gca().transAxes,
                    verticalalignment='center', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue"))
            plt.axis('off')
            
            print(f"  Sample {idx}: True=({true_coords[0]:.1f}, {true_coords[1]:.1f}), "
                  f"Pred=({pred_coords[0]:.1f}, {pred_coords[1]:.1f}), Error={error:.2f} px")
        
        plt.tight_layout()
        plt.savefig('cv_best_model_inference_test.png', dpi=300, bbox_inches='tight')
        print(f"\n✅ Visualization saved: cv_best_model_inference_test.png")
        plt.show()
        
        # Summary statistics
        print(f"\n📊 Test Results Summary:")
        print(f"   Mean Error: {np.mean(errors):.3f} ± {np.std(errors):.3f} px")
        print(f"   Best Case: {np.min(errors):.3f} px")
        print(f"   Worst Case: {np.max(errors):.3f} px")
        print(f"   Expected (from CV): 1.663 px")

def test_with_synthetic_samples(inference, num_samples=3):
    """Test with synthetic wave field samples."""
    print(f"\n🧪 Testing with {num_samples} synthetic samples...")
    
    # Create wave simulator
    simulator = Wave2DSimulator(grid_size=128, wave_speed=16.7)
    
    plt.figure(figsize=(15, 5 * num_samples))
    errors = []
    
    for i in range(num_samples):
        # Generate random source position
        true_coords = (
            np.random.uniform(20, 108),  # Keep away from edges
            np.random.uniform(20, 108)
        )
        
        # Simulate wave propagation
        wave_field = simulator.simulate_wave_propagation(
            source_pos=true_coords,
            total_time=500,
            max_timesteps=500
        )
        
        # Use final timestep for prediction
        final_wave_field = wave_field[-1]
        
        # Predict
        pred_coords = inference.predict_source(final_wave_field)
        
        # Calculate error
        error = np.sqrt((pred_coords[0] - true_coords[0])**2 + 
                      (pred_coords[1] - true_coords[1])**2)
        errors.append(error)
        
        # Plot
        plt.subplot(num_samples, 3, i*3 + 1)
        plt.imshow(final_wave_field, cmap='RdBu_r', aspect='equal')
        plt.title(f'Synthetic Sample {i+1}\nWave Field')
        plt.colorbar()
        
        plt.subplot(num_samples, 3, i*3 + 2)
        plt.imshow(final_wave_field, cmap='RdBu_r', aspect='equal')
        plt.scatter(*true_coords, color='red', s=100, marker='x', linewidth=3, label='True')
        plt.scatter(*pred_coords, color='yellow', s=100, marker='o', linewidth=2, label='Predicted')
        plt.title(f'Prediction\nError: {error:.2f} px')
        plt.legend()
        
        plt.subplot(num_samples, 3, i*3 + 3)
        plt.plot([true_coords[0], pred_coords[0]], [true_coords[1], pred_coords[1]], 
                'r--', linewidth=2, alpha=0.7)
        plt.scatter(*true_coords, color='red', s=150, marker='x', linewidth=3, label='True')
        plt.scatter(*pred_coords, color='yellow', s=150, marker='o', linewidth=2, 
                   edgecolor='black', label='Predicted')
        plt.xlim(0, 128)
        plt.ylim(0, 128)
        plt.grid(True, alpha=0.3)
        plt.title(f'Error: {error:.2f} px')
        plt.legend()
        
        print(f"  Sample {i+1}: True=({true_coords[0]:.1f}, {true_coords[1]:.1f}), "
              f"Pred=({pred_coords[0]:.1f}, {pred_coords[1]:.1f}), Error={error:.2f} px")
    
    plt.tight_layout()
    plt.savefig('cv_best_model_synthetic_test.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Visualization saved: cv_best_model_synthetic_test.png")
    plt.show()
    
    # Summary
    print(f"\n📊 Synthetic Test Results:")
    print(f"   Mean Error: {np.mean(errors):.3f} ± {np.std(errors):.3f} px")
    print(f"   Expected (from CV): 1.663 px")

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    test_best_cv_model() 