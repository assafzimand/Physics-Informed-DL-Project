"""
Wave Source Localization Inference Pipeline

Provides a class-based, robust pipeline for loading a trained wave source
localization model and running predictions on new data. It automatically handles
loading the correct normalization statistics (mean/std) used during training,
ensuring consistency and preventing silent prediction errors.
"""

from pathlib import Path
from typing import Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.models.wave_source_resnet import create_wave_source_model
from src.common.normalization import infer_dataset_tag, load_training_stats
from src.common.paths import get_models_dir


class WaveSourceInference:
    """
    Inference pipeline for wave source localization.

    This class encapsulates the model loading, data normalization, and prediction logic.
    It infers the correct normalization stats from the model checkpoint, making it
    safe to use with models trained on different datasets (e.g., T250 vs. T500).

    Attributes:
        device: The torch device ('cpu' or 'cuda') the model is on.
        model_path: The path to the loaded model checkpoint.
        wave_mean: The mean value used for normalization, loaded from training stats.
        wave_std: The standard deviation used for normalization, loaded from training stats.
        model: The loaded PyTorch model instance.
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cpu",
        dataset_tag: str | None = None,
    ) -> None:
        """
        Initializes the inference pipeline.

        Args:
            model_path: Path to the trained model (.pth file).
            device: Device to run inference on ('cpu' or 'cuda').
            dataset_tag: Explicit dataset tag ('T250' or 'T500') to use for
                         normalization. If None, it will be inferred from the
                         model path.
        """
        self.device = torch.device(device)
        self.model_path = str(model_path)

        # Resolve training dataset normalization statistics dynamically
        if dataset_tag is None:
            model_tag = infer_dataset_tag(self.model_path)
        else:
            model_tag = dataset_tag.upper()

        self.wave_mean, self.wave_std = load_training_stats(model_tag)

        # Load the trained model
        self.model = self._load_model()
        self.model.eval()

        print(f"INFO: Loaded model from {self.model_path}")
        print(f"INFO: Using device: {self.device}")
        print(f"INFO: Model parameters: {self.model.get_num_parameters():,}")
        print(
            f"INFO: Normalization ({model_tag}): mean={self.wave_mean:.6f}, std={self.wave_std:.6f}"
        )

    def _load_model(self) -> torch.nn.Module:
        """Loads the trained model from disk and moves it to the target device."""
        model = create_wave_source_model(grid_size=128)
        checkpoint = torch.load(self.model_path, map_location=self.device)

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        return model.to(self.device)

    def predict_source(self, wave_field: np.ndarray) -> Tuple[float, float]:
        """
        Predicts source coordinates from a wave field.

        Args:
            wave_field: 2D numpy array of wave field data (128x128).

        Returns:
            Predicted (x, y) coordinates as floats.
        """
        if wave_field.shape != (128, 128):
            raise ValueError(
                f"Expected wave field shape (128, 128), got {wave_field.shape}"
            )

        normalized_wave_field = (wave_field - self.wave_mean) / self.wave_std

        input_tensor = (
            torch.from_numpy(normalized_wave_field).float().unsqueeze(0).unsqueeze(0)
        )
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad():
            predicted_coords = self.model(input_tensor)

        predicted_coords_np = predicted_coords.cpu().numpy()[0]
        pred_x, pred_y = float(predicted_coords_np[0]), float(predicted_coords_np[1])
        return pred_x, pred_y

    def visualize_prediction(
        self,
        wave_field: np.ndarray,
        true_source: Tuple[int, int],
        predicted_source: Tuple[float, float],
        title: str = "Wave Source Localization",
        save_path: Optional[str | Path] = None,
    ) -> float:
        """
        Visualizes a wave field with true and predicted source locations.

        Generates a 3-panel plot:
        1. The wave field as a heatmap.
        2. An overlay of the true source (yellow circle) and predicted source (red square).
        3. Text boxes with coordinates and a title displaying the Euclidean distance error.

        Args:
            wave_field: 2D numpy array of wave field data.
            true_source: True source coordinates (x, y).
            predicted_source: Predicted source coordinates (x, y).
            title: Plot title.
            save_path: Optional path to save the plot image.

        Returns:
            Euclidean distance error in pixels.
        """
        plt.figure(figsize=(12, 10))
        im = plt.imshow(
            wave_field, cmap="RdBu_r", origin="lower", extent=[0, 127, 0, 127]
        )

        true_x, true_y = true_source
        plt.plot(
            true_x,
            true_y,
            "o",
            markersize=12,
            markerfacecolor="yellow",
            markeredgecolor="black",
            markeredgewidth=3,
            label="True Source",
        )

        pred_x, pred_y = predicted_source
        plt.plot(
            pred_x,
            pred_y,
            "s",
            markersize=12,
            markerfacecolor="red",
            markeredgecolor="white",
            markeredgewidth=3,
            label="Predicted Source",
        )

        distance_error = float(np.sqrt((true_x - pred_x) ** 2 + (true_y - pred_y) ** 2))

        plt.colorbar(im, label="Wave Amplitude")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.title(f"{title}\nPrediction Error: {distance_error:.2f} pixels")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.text(
            0.02,
            0.98,
            f"True: ({true_x:.1f}, {true_y:.1f})",
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.8),
        )
        plt.text(
            0.02,
            0.90,
            f"Pred: ({pred_x:.1f}, {pred_y:.1f})",
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="red", alpha=0.8),
        )

        if save_path:
            plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
            print(f"INFO: Visualization saved to {save_path}")
        plt.show()
        return distance_error


def load_inference_model(model_name: str) -> WaveSourceInference:
    """
    Convenience function to load a trained model from the project's `models` directory.

    Args:
        model_name: Name of the model file (e.g., "best_model.pth").

    Returns:
        A WaveSourceInference instance ready for predictions.

    Raises:
        FileNotFoundError: If the model file does not exist in the `models` directory.
    """
    model_path = get_models_dir() / model_name
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found in models/ directory: {model_name}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return WaveSourceInference(model_path, device)


if __name__ == "__main__":
    # This is a simple demonstration of how to use the inference pipeline.
    # It requires a model to be present in the `models` directory.
    print("Wave Source Localization Inference Pipeline")
    print("=" * 50)
    try:
        # Example: replace with the actual name of your trained model
        inference = load_inference_model("grid_search_001_quick_search_best.pth")
        print("\nSUCCESS: Inference pipeline ready!")
        print("Use `inference.predict_source(wave_field)` to make predictions.")
    except FileNotFoundError as e:
        print("\nERROR: Could not load model for demonstration.")
        print(f"  - {e}")
        print(
            "  - Please ensure a trained model (e.g., 'best_model.pth') exists in the `models/` directory."
        )
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
