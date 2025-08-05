#!/usr/bin/env python3
"""
Comprehensive Activation Maximization Analysis - All Layers

Automatically processes all Conv2D layers, finding top 10 active filters 
for each layer and optimizing them. This is a long-running process.
"""

import sys
from pathlib import Path
import torch
import time
from datetime import datetime, timedelta

sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from src.activation_maximization.simple_activation_max import (
    SimpleActivationMaximizer
)
from src.activation_maximization.layer_hooks import find_best_cv_model
from src.models.wave_source_resnet import create_wave_source_model
from src.data.wave_dataset import WaveDataset


def get_top_active_filters(model, device, layer, sample_tensor, top_k=10):
    """
    Find the top K most active filters in a given layer for a specific sample.
    
    Args:
        model: The trained model
        device: Torch device
        layer: The target layer module
        sample_tensor: Input sample tensor (1, 1, H, W)
        top_k: Number of top filters to return
        
    Returns:
        List of filter indices sorted by activation strength
    """
    print(f"🔍 Finding top {top_k} active filters...")
    
    # Hook to capture activations
    activations = {}
    
    def hook_fn(module, input, output):
        activations['target'] = output.detach()
    
    # Register hook
    hook = layer.register_forward_hook(hook_fn)
    
    try:
        # Forward pass
        model.eval()
        with torch.no_grad():
            _ = model(sample_tensor)
        
        # Get activations and compute filter-wise mean activation
        layer_output = activations['target']  # Shape: (1, C, H, W)
        
        # Compute mean activation per filter across spatial dimensions
        filter_activations = layer_output.mean(dim=(0, 2, 3))  # Shape: (C,)
        
        # Get top K filter indices
        top_indices = torch.argsort(filter_activations, descending=True)[:top_k].cpu().numpy()
        top_values = filter_activations[top_indices].cpu().numpy()
        
        print(f"📊 Top {top_k} filters:")
        for i, (idx, val) in enumerate(zip(top_indices, top_values)):
            print(f"  {i+1:2d}. Filter {idx:2d}: {val:.3f}")
        
        return top_indices.tolist()
    
    except Exception as e:
        print(f"❌ Error finding active filters: {e}")
        return []
    
    finally:
        hook.remove()


def estimate_total_time(num_layers, filters_per_layer, iterations_per_filter):
    """Estimate total processing time"""
    # Rough estimate: ~0.5 seconds per iteration based on previous runs
    total_iterations = num_layers * filters_per_layer * iterations_per_filter
    estimated_seconds = total_iterations * 0.5
    estimated_time = timedelta(seconds=estimated_seconds)
    
    print(f"📊 ANALYSIS SCOPE:")
    print(f"   - Layers to process: {num_layers}")
    print(f"   - Filters per layer: {filters_per_layer}")
    print(f"   - Iterations per filter: {iterations_per_filter}")
    print(f"   - Total iterations: {total_iterations:,}")
    print(f"   - Estimated time: {estimated_time}")
    print(f"   - Expected completion: {datetime.now() + estimated_time}")


def main():
    """Run comprehensive activation maximization analysis on all layers"""
    
    print("🚀 COMPREHENSIVE ACTIVATION MAXIMIZATION - ALL LAYERS")
    print("=" * 80)
    
    start_time = datetime.now()
    
    # Load model
    cv_results_path = (Path(__file__).parent.parent.parent / 
                      "experiments" / "cv_full")
    model_info = find_best_cv_model(cv_results_path)
    fold_id, error, model_path = model_info
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_wave_source_model(grid_size=128)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device).eval()
    
    print(f"✅ Loaded best model: Fold {fold_id}")
    print(f"🖥️ Using device: {device}")
    
    # Get all Conv2D layers
    print("\n🔧 Available Conv2D layers:")
    conv_layers = []
    for i, (name, module) in enumerate(model.named_modules()):
        if isinstance(module, torch.nn.Conv2d):
            conv_layers.append((i, name, module))
            print(f"  {len(conv_layers)-1:2d}: Layer {i:2d} - {name} "
                  f"({module.out_channels} filters)")
    
    # Analysis parameters
    TOP_K = 10
    ITERATIONS = 500
    LEARNING_RATE = 0.01
    
    # Estimate total time
    print(f"\n⏱️ TIME ESTIMATION:")
    estimate_total_time(len(conv_layers), TOP_K, ITERATIONS)
    
    # Load reference sample
    dataset_path = "data/wave_dataset_analysis_20samples.h5"
    dataset = WaveDataset(dataset_path, normalize_wave_fields=False)  # Load raw data
    sample_idx = 0
    wave_field, coordinates = dataset[sample_idx]
    sample_tensor = wave_field.to(device)
    
    # Create consistent initialization tensor for all filters (raw, non-normalized)
    init_tensor = sample_tensor.clone().detach()  # Shape: [1, 1, 128, 128] - RAW data
    
    print(f"\n🌊 Using sample {sample_idx} as reference and consistent initialization")
    print(f"📊 Sample coordinates: x={coordinates[0]:.1f}, y={coordinates[1]:.1f}")
    print(f"🎯 All {TOP_K} filters in all {len(conv_layers)} layers will use the SAME initialization")
    print(f"📊 Init tensor stats (RAW): mean={init_tensor.mean():.6f}, std={init_tensor.std():.6f}")
    
    # Process all layers
    total_layers = len(conv_layers)
    successful_layers = 0
    failed_layers = 0
    total_filters_processed = 0
    
    print(f"\n🎯 STARTING ANALYSIS OF {total_layers} LAYERS")
    print("=" * 80)
    
    for layer_num, (layer_idx, layer_name, target_layer) in enumerate(conv_layers):
        layer_start_time = datetime.now()
        
        print(f"\n📍 LAYER {layer_num + 1}/{total_layers}: "
              f"Layer {layer_idx} - {layer_name}")
        print(f"   ({target_layer.out_channels} filters)")
        
        # Create layer-specific save directory
        layer_save_dir = f"experiments/activation_maximization/comprehensive/layer_{layer_idx}"
        
        try:
            # Find top active filters for this layer (using normalized sample for activation measurement)
            # Create a temporary normalized dataset just for finding active filters
            temp_normalized_dataset = WaveDataset(dataset_path, normalize_wave_fields=True)
            normalized_sample, _ = temp_normalized_dataset[sample_idx]
            normalized_sample = normalized_sample.to(device)
            
            top_filters = get_top_active_filters(model, device, target_layer, 
                                               normalized_sample, top_k=TOP_K)
            
            if not top_filters:
                print(f"❌ No active filters found for layer {layer_idx}")
                failed_layers += 1
                continue
            
            # Setup maximizer for this layer
            maximizer = SimpleActivationMaximizer(model, device)
            maximizer.register_hook("target_layer", target_layer)
            
            try:
                print(f"🚀 Optimizing top {len(top_filters)} filters...")
                print(f"📁 Saving to: {layer_save_dir}")
                print(f"🎯 All filters will use SAME initialization tensor (RAW)")
                
                layer_successful_filters = 0
                
                for i, filter_idx in enumerate(top_filters):
                    filter_start_time = datetime.now()
                    
                    print(f"\n  🎯 Filter {filter_idx} ({i+1}/{len(top_filters)}) "
                          f"[Layer {layer_num+1}/{total_layers}]")
                    
                    try:
                        # Run optimization with consistent initialization
                        results = maximizer.optimize_filter(
                            layer_name="target_layer",
                            filter_idx=filter_idx,
                            iterations=ITERATIONS,
                            learning_rate=LEARNING_RATE,
                            skip_normalization=False,
                            use_real_data_init=False,  # IMPORTANT: Set to False when using init_tensor
                            init_tensor=init_tensor,  # Same RAW tensor for all filters!
                            save_dir=layer_save_dir
                        )
                        
                        final_activation = results['config']['final_activation']
                        filter_time = datetime.now() - filter_start_time
                        
                        print(f"     ✅ Final activation: {final_activation:.2f} "
                              f"(took {filter_time.total_seconds():.1f}s)")
                        
                        layer_successful_filters += 1
                        total_filters_processed += 1
                        
                    except Exception as e:
                        print(f"     ❌ Filter {filter_idx} failed: {str(e)[:100]}")
                        continue
                
                layer_time = datetime.now() - layer_start_time
                print(f"\n✅ Layer {layer_idx} complete: {layer_successful_filters}/{len(top_filters)} filters successful")
                print(f"   ⏱️ Layer time: {layer_time}")
                
                if layer_successful_filters > 0:
                    successful_layers += 1
                else:
                    failed_layers += 1
                    
            finally:
                maximizer.cleanup_hooks()
                
        except Exception as e:
            print(f"❌ Layer {layer_idx} failed completely: {e}")
            failed_layers += 1
            continue
        
        # Progress update
        elapsed_time = datetime.now() - start_time
        remaining_layers = total_layers - (layer_num + 1)
        avg_time_per_layer = elapsed_time / (layer_num + 1)
        estimated_remaining = avg_time_per_layer * remaining_layers
        
        print(f"\n📈 PROGRESS UPDATE:")
        print(f"   - Layers completed: {layer_num + 1}/{total_layers}")
        print(f"   - Successful layers: {successful_layers}")
        print(f"   - Failed layers: {failed_layers}")
        print(f"   - Total filters processed: {total_filters_processed}")
        print(f"   - Elapsed time: {elapsed_time}")
        print(f"   - Estimated remaining: {estimated_remaining}")
        print(f"   - Expected completion: {datetime.now() + estimated_remaining}")
    
    # Final summary
    total_time = datetime.now() - start_time
    
    print(f"\n🎉 ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"📊 FINAL SUMMARY:")
    print(f"   - Total layers processed: {total_layers}")
    print(f"   - Successful layers: {successful_layers}")
    print(f"   - Failed layers: {failed_layers}")
    print(f"   - Total filters processed: {total_filters_processed}")
    print(f"   - Total time: {total_time}")
    print(f"   - Average time per layer: {total_time / total_layers}")
    print(f"   - Results saved in: experiments/activation_maximization/comprehensive/")
    
    print(f"\n✨ All results are organized in layer-specific folders:")
    print(f"   experiments/activation_maximization/comprehensive/layer_{{N}}/")


if __name__ == "__main__":
    main() 