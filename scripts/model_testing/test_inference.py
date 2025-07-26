"""
Test Script for Wave Source Localization Inference

This script:
1. Generates a new wave sample using T=500 timesteps (matching training data)
2. Uses the trained model to predict the source location
3. Visualizes the results with both real and predicted source locations
"""

import sys
import os
import numpy as np
import random

# Add parent directory for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.wave_simulation import Wave2DSimulator
from src.inference import load_inference_model


def create_test_sample(timesteps: int = 500) -> tuple:
    """
    Create a test wave sample using the same parameters as training data.
    
    Args:
        timesteps: Number of timesteps to simulate (should match training: T=500)
        
    Returns:
        Tuple of (wave_field, true_source_coordinates)
    """
    print(f"🌊 Generating test sample with T={timesteps} timesteps...")
    
    # Parameters matching the training dataset
    # These come from the dataset generator configuration
    GRID_SIZE = 128
    WAVE_SPEED = 0.8  # From dataset_generator.py main function
    DT = 0.03         # From dataset_generator.py main function  
    DX = 1.0
    
    # Create simulator with training parameters
    simulator = Wave2DSimulator(
        grid_size=GRID_SIZE,
        wave_speed=WAVE_SPEED,
        dt=DT,
        dx=DX
    )
    
    print("Simulator config:")
    print(f"  - Grid size: {GRID_SIZE}×{GRID_SIZE}")
    print(f"  - Wave speed: {WAVE_SPEED}")
    print(f"  - Time step: {DT}")
    print(f"  - CFL condition: {simulator.cfl_condition:.3f}")
    
    # Generate random source location (same logic as dataset generator)
    margin = 10  # Keep away from edges
    source_x = random.randint(margin, GRID_SIZE - margin - 1)
    source_y = random.randint(margin, GRID_SIZE - margin - 1)
    
    print(f"  - True source: ({source_x}, {source_y})")
    
    # Run wave simulation
    wave_field, _ = simulator.simulate(source_x, source_y, timesteps)
    
    print(f"  - Wave field range: [{wave_field.min():.3f}, {wave_field.max():.3f}]")
    print(f"✅ Test sample generated successfully!")
    
    return wave_field, (source_x, source_y)


def test_inference_pipeline():
    """
    Test the complete inference pipeline end-to-end.
    """
    print("🧠 Testing Wave Source Localization Inference Pipeline")
    print("=" * 60)
    
    try:
        # Step 1: Load the trained model
        print("\n📥 Loading trained model...")
        inference = load_inference_model("grid_search_001_quick_search_best.pth")
        
        # Step 2: Generate a test sample
        print("\n🎯 Generating test sample...")
        wave_field, true_source = create_test_sample(timesteps=500)
        
        # Step 3: Make prediction
        print("\n🔮 Making prediction...")
        predicted_source = inference.predict_source(wave_field)
        
        # Step 4: Calculate error
        true_x, true_y = true_source
        pred_x, pred_y = predicted_source
        distance_error = np.sqrt((true_x - pred_x)**2 + (true_y - pred_y)**2)
        
        print(f"\n📊 Results:")
        print(f"  - True source: ({true_x}, {true_y})")
        print(f"  - Predicted source: ({pred_x:.1f}, {pred_y:.1f})")
        print(f"  - Distance error: {distance_error:.2f} pixels")
        
        # Step 5: Visualize results
        print(f"\n📈 Creating visualization...")
        save_path = "results/inference_test_result.png"
        os.makedirs("results", exist_ok=True)
        
        error = inference.visualize_prediction(
            wave_field=wave_field,
            true_source=true_source,
            predicted_source=predicted_source,
            title="Inference Pipeline Test",
            save_path=save_path
        )
        
        # Step 6: Performance assessment
        print(f"\n🎯 Performance Assessment:")
        if distance_error < 5.0:
            print(f"✅ EXCELLENT: Error < 5 pixels")
        elif distance_error < 10.0:
            print(f"✅ GOOD: Error < 10 pixels")
        elif distance_error < 20.0:
            print(f"⚠️  FAIR: Error < 20 pixels")
        else:
            print(f"❌ POOR: Error > 20 pixels")
        
        print(f"\n🎉 Inference test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during inference test: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_combined_visualization(results_list, 
                                  save_path="results/combined_inference_test.png"):
    """
    Create a combined visualization showing all test samples in one plot.
    
    Args:
        results_list: List of result dictionaries with wave_field, sources, error
        save_path: Path to save the combined visualization
    """
    import matplotlib.pyplot as plt
    
    # Create a large figure with subplots
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('Wave Source Localization: 10 Test Samples', fontsize=16, fontweight='bold')
    
    for i, result in enumerate(results_list):
        row = i // 5
        col = i % 5
        ax = axes[row, col]
        
        # Plot the wave field
        im = ax.imshow(result['wave_field'], cmap='viridis', origin='lower')
        
        # Plot true source (red circle)
        true_x, true_y = result['true_source']
        ax.plot(true_x, true_y, 'ro', markersize=10, markeredgecolor='white', 
                markeredgewidth=2, label='True Source')
        
        # Plot predicted source (blue star)  
        pred_x, pred_y = result['predicted_source']
        ax.plot(pred_x, pred_y, 'b*', markersize=12, markeredgecolor='white',
                markeredgewidth=1, label='Predicted')
        
        # Add title with error
        ax.set_title(f'Sample {i+1}\\nError: {result["error"]:.1f}px', fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        
        # Add legend only on first subplot
        if i == 0:
            ax.legend(bbox_to_anchor=(0.02, 0.98), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"📊 Combined visualization saved to: {save_path}")


def test_multiple_samples(num_samples: int = 10):
    """
    Test the inference pipeline on multiple samples to get performance statistics.
    
    Args:
        num_samples: Number of test samples to generate and test
    """
    print(f"\n🔄 Testing inference on {num_samples} samples...")
    print("=" * 50)
    
    try:
        # Load model once
        inference = load_inference_model("grid_search_001_quick_search_epoch_050.pth")
        
        errors = []
        results = []
        
        for i in range(num_samples):
            print(f"\n📍 Sample {i+1}/{num_samples}")
            
            # Generate test sample
            wave_field, true_source = create_test_sample(timesteps=500)
            
            # Make prediction
            predicted_source = inference.predict_source(wave_field)
            
            # Calculate error
            true_x, true_y = true_source
            pred_x, pred_y = predicted_source
            distance_error = np.sqrt((true_x - pred_x)**2 + (true_y - pred_y)**2)
            
            errors.append(distance_error)
            results.append({
                'wave_field': wave_field,
                'true_source': true_source,
                'predicted_source': predicted_source,
                'error': distance_error
            })
            
            print(f"   True: ({true_x}, {true_y}) | Pred: ({pred_x:.1f}, {pred_y:.1f}) | Error: {distance_error:.2f}px")
        
        # Calculate statistics
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        min_error = np.min(errors)
        max_error = np.max(errors)
        
        print(f"\n📊 Performance Statistics:")
        print(f"  - Mean error: {mean_error:.2f} ± {std_error:.2f} pixels")
        print(f"  - Min error: {min_error:.2f} pixels")
        print(f"  - Max error: {max_error:.2f} pixels")
        print(f"  - Samples with error < 5px: {sum(1 for e in errors if e < 5.0)}/{num_samples}")
        print(f"  - Samples with error < 10px: {sum(1 for e in errors if e < 10.0)}/{num_samples}")
        
        # Create combined visualization
        print("\n📊 Creating combined visualization...")
        os.makedirs("results", exist_ok=True)
        create_combined_visualization(results)
        
        return results
        
    except Exception as e:
        print(f"❌ Error during multi-sample test: {e}")
        return None


if __name__ == "__main__":
    # Set random seed for reproducible results
    random.seed(42)
    np.random.seed(42)
    
    print("🧪 Wave Source Localization Inference Test Suite")
    print("=" * 60)
    
    # Multi-sample performance test with combined visualization
    print("\n🔬 Multi-Sample Inference Test (10 Samples)")
    results = test_multiple_samples(num_samples=10)
    
    if results:
        print(f"\n✅ Inference test completed successfully!")
    else:
        print(f"\n❌ Inference test failed") 