"""
Wave Source Localization Inference Pipeline

This module provides inference capabilities for the trained wave source localization model.
Loads a trained model and predicts source coordinates from wave field data.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import os
import sys

# Add parent directory for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.wave_source_resnet import create_wave_source_model
from src.wave_simulation import Wave2DSimulator


class WaveSourceInference:
    """
    Inference pipeline for wave source localization.
    """
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Initialize the inference pipeline.
        
        Args:
            model_path: Path to the trained model (.pth file)
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.device = torch.device(device)
        self.model_path = model_path
        
        # Training dataset normalization statistics (CRITICAL!)
        # These MUST match the statistics used during training
        self.wave_mean = 0.000460
        self.wave_std = 0.020842
        
        # Load the trained model
        self.model = self._load_model()
        self.model.eval()  # Set to evaluation mode
        
        print(f"✅ Loaded model from {model_path}")
        print(f"🔧 Using device: {self.device}")
        print(f"📊 Model parameters: {self.model.get_num_parameters():,}")
        print(f"📊 Normalization: mean={self.wave_mean:.6f}, std={self.wave_std:.6f}")
    
    def _load_model(self) -> torch.nn.Module:
        """Load the trained model from disk."""
        # Create model architecture
        model = create_wave_source_model(grid_size=128)
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Extract model state dict from checkpoint
        if 'model_state_dict' in checkpoint:
            # Full training checkpoint
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Just model weights
            model.load_state_dict(checkpoint)
        
        # Move to device
        model = model.to(self.device)
        
        return model
    
    def predict_source(self, wave_field: np.ndarray) -> Tuple[float, float]:
        """
        Predict source coordinates from a wave field.
        
        Args:
            wave_field: 2D numpy array of wave field data (128x128)
            
        Returns:
            Tuple of (x, y) predicted coordinates
        """
        # Validate input
        if wave_field.shape != (128, 128):
            raise ValueError(f"Expected wave field shape (128, 128), got {wave_field.shape}")
        
        # Apply training normalization (CRITICAL for correct predictions!)
        normalized_wave_field = (wave_field - self.wave_mean) / self.wave_std
        
        # Prepare input tensor
        # Shape: (1, 1, 128, 128) for batch_size=1, channels=1
        input_tensor = torch.from_numpy(normalized_wave_field).float()
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        input_tensor = input_tensor.to(self.device)
        
        # Run inference
        with torch.no_grad():
            predicted_coords = self.model(input_tensor)
        
        # Convert to numpy and extract coordinates
        predicted_coords = predicted_coords.cpu().numpy()[0]  # Remove batch dimension
        pred_x, pred_y = predicted_coords[0], predicted_coords[1]
        
        return float(pred_x), float(pred_y)
    
    def predict_with_confidence(self, wave_field: np.ndarray, num_samples: int = 10) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Predict source coordinates with uncertainty estimation using dropout sampling.
        
        Args:
            wave_field: 2D numpy array of wave field data (128x128)
            num_samples: Number of dropout samples for uncertainty estimation
            
        Returns:
            Tuple of ((mean_x, mean_y), (std_x, std_y))
        """
        # Enable dropout for uncertainty estimation
        self.model.train()
        
        predictions = []
        for _ in range(num_samples):
            pred_x, pred_y = self.predict_source(wave_field)
            predictions.append([pred_x, pred_y])
        
        # Set back to eval mode
        self.model.eval()
        
        # Calculate statistics
        predictions = np.array(predictions)
        mean_coords = np.mean(predictions, axis=0)
        std_coords = np.std(predictions, axis=0)
        
        return (float(mean_coords[0]), float(mean_coords[1])), (float(std_coords[0]), float(std_coords[1]))
    
    def visualize_prediction(self, wave_field: np.ndarray, true_source: Tuple[int, int], 
                           predicted_source: Tuple[float, float], 
                           title: str = "Wave Source Localization", 
                           save_path: Optional[str] = None):
        """
        Visualize the wave field with true and predicted source locations.
        
        Args:
            wave_field: 2D numpy array of wave field data
            true_source: True source coordinates (x, y)
            predicted_source: Predicted source coordinates (x, y)
            title: Title for the plot
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(12, 10))
        
        # Create the heatmap
        im = plt.imshow(wave_field, cmap='RdBu_r', origin='lower', 
                       extent=[0, 127, 0, 127])
        
        # Mark the true source location
        true_x, true_y = true_source
        plt.plot(true_x, true_y, 'o', markersize=12, 
                markerfacecolor='yellow', markeredgecolor='black', 
                markeredgewidth=3, label='True Source')
        
        # Mark the predicted source location
        pred_x, pred_y = predicted_source
        plt.plot(pred_x, pred_y, 's', markersize=12, 
                markerfacecolor='red', markeredgecolor='white', 
                markeredgewidth=3, label='Predicted Source')
        
        # Calculate and display error
        distance_error = np.sqrt((true_x - pred_x)**2 + (true_y - pred_y)**2)
        
        plt.colorbar(im, label='Wave Amplitude')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title(f'{title}\nPrediction Error: {distance_error:.2f} pixels')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add error information as text
        plt.text(0.02, 0.98, f'True: ({true_x:.1f}, {true_y:.1f})', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        plt.text(0.02, 0.90, f'Pred: ({pred_x:.1f}, {pred_y:.1f})', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.8))
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
        
        return distance_error


def load_inference_model(model_name: str = "grid_search_001_quick_search_best.pth") -> WaveSourceInference:
    """
    Convenient function to load a trained model for inference.
    
    Args:
        model_name: Name of the model file in the models/ directory
        
    Returns:
        WaveSourceInference object ready for predictions
    """
    model_path = os.path.join("models", model_name)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Use GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return WaveSourceInference(model_path, device)


if __name__ == "__main__":
    # Example usage
    print("🧠 Wave Source Localization Inference Pipeline")
    print("=" * 50)
    
    try:
        # Load the trained model
        inference = load_inference_model()
        
        print("\n✅ Inference pipeline ready!")
        print("Use inference.predict_source(wave_field) to make predictions")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}") 