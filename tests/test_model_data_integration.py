"""
Demo: Model + Data Pipeline Integration

Shows how the WaveSourceMiniResNet model works with our data pipeline.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.wave_source_resnet import create_wave_source_model
from src.data.wave_dataset import create_dataloaders
import torch


def demo_model_data_integration():
    """Demonstrate model and data pipeline working together."""
    print("🌊 Wave Source Localization: Model + Data Integration Demo")
    print("=" * 65)
    
    # Check if dataset exists
    dataset_path = "data/wave_dataset_T250.h5"
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        print("Please generate datasets first using scripts/generate_full_datasets.py")
        return
    
    # Create model
    print("🧠 Creating Mini-ResNet model...")
    model = create_wave_source_model()
    model.eval()  # Set to evaluation mode
    print(f"✓ Model created with {model.get_num_parameters():,} parameters")
    
    # Create data loaders
    print("\n📊 Creating data loaders...")
    train_loader, val_loader = create_dataloaders(
        dataset_path,
        batch_size=4,
        validation_split=0.2,
        num_workers=0
    )
    
    # Get a batch from validation data
    print("\n🔍 Testing model with real data...")
    val_batch = next(iter(val_loader))
    wave_fields, true_coordinates = val_batch
    
    print(f"✓ Loaded batch with {wave_fields.shape[0]} samples")
    print(f"  - Wave fields shape: {wave_fields.shape}")
    print(f"  - True coordinates shape: {true_coordinates.shape}")
    
    # Run model prediction
    with torch.no_grad():
        predicted_coordinates = model(wave_fields)
    
    print(f"✓ Model prediction completed")
    print(f"  - Predictions shape: {predicted_coordinates.shape}")
    print(f"  - Predictions range: [{predicted_coordinates.min():.1f}, {predicted_coordinates.max():.1f}]")
    
    # Show sample predictions vs ground truth
    print(f"\n📋 Sample Predictions vs Ground Truth:")
    print("-" * 50)
    for i in range(min(4, wave_fields.shape[0])):
        true_x, true_y = true_coordinates[i]
        pred_x, pred_y = predicted_coordinates[i]
        error_x = abs(pred_x - true_x)
        error_y = abs(pred_y - true_y)
        
        print(f"Sample {i+1}:")
        print(f"  True:      ({true_x:.1f}, {true_y:.1f})")
        print(f"  Predicted: ({pred_x:.1f}, {pred_y:.1f})")
        print(f"  Error:     ({error_x:.1f}, {error_y:.1f})")
        print()
    
    # Show available activations for interpretability
    print("🔬 Available model activations for interpretability:")
    for name in model.activations.keys():
        activation = model.activations[name]
        print(f"  - {name}: {activation.shape}")
    
    print("✅ Integration demo completed successfully!")
    print("\n💡 The model and data pipeline are ready for training!")


if __name__ == "__main__":
    demo_model_data_integration() 