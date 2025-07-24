#!/usr/bin/env python3
"""
Feature Extraction for T=500 Model
Extracts activations from all stages for our 20 selected samples.
"""

import sys
import os
import numpy as np
import torch
import json
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append('.')
sys.path.append('src')

from models.wave_source_resnet import WaveSourceMiniResNet


class SimpleFeatureExtractor:
    def __init__(self, model_path):
        """Initialize feature extractor with T=500 model."""
        print(f"🤖 Loading T=500 model: {model_path}")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
            config = checkpoint.get('config', {})
        else:
            model_state_dict = checkpoint
            config = {}
        
        # Create model
        grid_size = config.get('grid_size', 128)
        self.model = WaveSourceMiniResNet(grid_size=grid_size)
        self.model.load_state_dict(model_state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Model loaded successfully (grid_size: {grid_size})")
        
        # Store activations - the model already has hooks set up!
        self.activations = {}
        
    def extract_activations(self, wave_input):
        """Extract activations from all stages."""
        # Clear previous activations
        self.model.activations.clear()
        
        # Run forward pass
        with torch.no_grad():
            wave_input = wave_input.unsqueeze(0).to(self.device)  # Add batch dimension
            prediction = self.model(wave_input)
        
        # Get activations (the model stores them automatically)
        activations = {}
        stage_names = ['basic_wave_features', 'complex_wave_patterns', 
                      'interference_patterns', 'source_localization']
        
        for i, stage_name in enumerate(stage_names):
            if stage_name in self.model.activations:
                # Get activation tensor [1, channels, height, width]
                activation = self.model.activations[stage_name].squeeze(0)  # Remove batch dim
                activations[f'stage_{i+1}'] = activation.cpu()
        
        # Also get stage 0 (initial processing) - need to hook it manually
        # For now, let's add stage 0 by running a partial forward
        with torch.no_grad():
            stage_0 = self.model.wave_input_processor(wave_input)
            activations['stage_0'] = stage_0.squeeze(0).cpu()
        
        return activations, prediction.cpu()
    
    def get_top_features(self, activation_tensor, n_features=9):
        """Get the n strongest feature maps from an activation tensor."""
        # activation_tensor shape: [channels, height, width]
        channels, height, width = activation_tensor.shape
        
        # Calculate mean activation strength per channel
        channel_strengths = activation_tensor.view(channels, -1).mean(dim=1)
        
        # Get indices of top n channels
        top_indices = torch.topk(channel_strengths, min(n_features, channels)).indices
        
        # Extract top feature maps
        top_features = activation_tensor[top_indices]
        
        return top_features, top_indices.tolist()
    
    def process_sample(self, sample_path, n_features=9):
        """Process a single sample and extract top features."""
        # Load sample data
        sample_data = np.load(sample_path)
        wave_field = torch.from_numpy(sample_data['wave_field']).float()
        coordinates = sample_data['coordinates']
        category = str(sample_data['category'])
        original_index = int(sample_data['original_index'])
        
        print(f"   Processing sample {sample_path.name}: {category} source at {coordinates}")
        
        # Extract activations
        activations, prediction = self.extract_activations(wave_field)
        
        # Get top features for each stage
        sample_features = {}
        for stage_name, activation in activations.items():
            top_features, top_indices = self.get_top_features(activation, n_features)
            sample_features[stage_name] = {
                'features': top_features,
                'indices': top_indices,
                'activation_shape': list(activation.shape)
            }
        
        # Prepare results
        results = {
            'sample_info': {
                'path': str(sample_path),
                'original_index': original_index,
                'category': category,
                'true_coordinates': coordinates.tolist(),
                'predicted_coordinates': prediction.squeeze().tolist()
            },
            'features': sample_features
        }
        
        return results


def process_all_samples(model_path, samples_dir, output_dir, n_features=9):
    """Process all samples and save feature extractions."""
    print("🚀 Starting feature extraction for all samples...")
    
    # Initialize extractor
    extractor = SimpleFeatureExtractor(model_path)
    
    # Create output directories
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create stage subdirectories
    for i in range(5):  # Stages 0-4
        stage_dir = output_dir / f"stage_{i}"
        stage_dir.mkdir(exist_ok=True)
    
    # Load sample metadata
    samples_dir = Path(samples_dir)
    metadata_file = samples_dir / "sample_info.json"
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    print(f"📊 Processing {metadata['total_samples']} samples...")
    
    # Process each sample
    all_results = []
    raw_data_dir = samples_dir / "raw_data"
    
    for sample_file in sorted(raw_data_dir.glob("*.npz")):
        print(f"🔄 {sample_file.name}")
        
        # Extract features for this sample
        results = extractor.process_sample(sample_file, n_features)
        all_results.append(results)
        
        # Save individual sample features
        sample_id = sample_file.stem  # e.g., "sample_00_idx_17"
        
        for stage_name, feature_data in results['features'].items():
            stage_dir = output_dir / stage_name
            feature_file = stage_dir / f"{sample_id}_features.npz"
            
            np.savez(feature_file,
                    features=feature_data['features'].numpy(),
                    indices=feature_data['indices'],
                    activation_shape=feature_data['activation_shape'],
                    sample_info=results['sample_info'])
    
    # Save summary results
    summary_file = output_dir / "extraction_summary.json"
    summary_data = {
        'model_path': model_path,
        'n_features_per_stage': n_features,
        'total_samples': len(all_results),
        'samples_processed': [r['sample_info'] for r in all_results],
        'stage_info': {
            'stage_0': 'Initial wave pattern extraction (32 channels)',
            'stage_1': 'Basic wave features (32 channels)', 
            'stage_2': 'Complex wave patterns (64 channels)',
            'stage_3': 'Interference patterns (128 channels)',
            'stage_4': 'Source localization (256 channels)'
        }
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\n✅ Feature extraction complete!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"📊 Extracted {n_features} top features per stage for {len(all_results)} samples")
    
    return all_results


def main():
    """Main function for feature extraction."""
    print("🔍 Starting T=500 feature extraction...")
    
    # Paths
    # NOTE: Model path is hardcoded to fold 2 - this happens to be the best fold for T=500
    # but should be updated to find best fold programmatically if further models are made
    model_path = "experiments/cv_full/data/models/cv_full_5fold_75epochs_fold_2_best.pth"  # Best T=500 model
    samples_dir = "experiments/feature_analysis/samples"
    output_dir = "experiments/feature_analysis/activations"
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print("💡 Make sure to use the correct path to your best T=500 model")
        return
    
    # Process all samples
    results = process_all_samples(model_path, samples_dir, output_dir, n_features=9)
    
    print(f"\n🎉 Feature extraction complete!")
    print(f"📊 Next step: Create visualization plots")


if __name__ == "__main__":
    main() 