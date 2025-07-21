"""
Hyperparameter Grid Search for Wave Source Localization

Runs systematic grid search over hyperparameters with MLflow tracking.
"""

import os
import sys
import itertools
from typing import Dict, List, Any
import time

# Add parent directory for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from configs.training_config import (
    TrainingConfig, 
    HYPERPARAMETER_GRIDS, 
    create_config_from_dict,
    get_default_config
)
from src.training.trainer import train_model
import mlflow


def generate_hyperparameter_combinations(grid_name: str) -> List[Dict[str, Any]]:
    """
    Generate all combinations from a hyperparameter grid.
    
    Args:
        grid_name: Name of predefined grid or custom grid dict
        
    Returns:
        List of parameter combinations
    """
    if grid_name in HYPERPARAMETER_GRIDS:
        grid = HYPERPARAMETER_GRIDS[grid_name]
    else:
        raise ValueError(f"Unknown grid: {grid_name}. Available: {list(HYPERPARAMETER_GRIDS.keys())}")
    
    # Generate all combinations
    keys = list(grid.keys())
    values = list(grid.values())
    combinations = []
    
    for combination in itertools.product(*values):
        param_dict = dict(zip(keys, combination))
        combinations.append(param_dict)
    
    return combinations


def run_grid_search(dataset_name: str = "T500", 
                   grid_name: str = "quick_search",
                   max_epochs: int = 50,
                   experiment_name: str = "grid_search") -> List[Dict]:
    """
    Run hyperparameter grid search.
    
    Args:
        dataset_name: Dataset to use (T250 or T500)
        grid_name: Name of hyperparameter grid
        max_epochs: Maximum epochs per run
        experiment_name: MLflow experiment name
        
    Returns:
        List of results for each combination
    """
    print("🔍 Starting Hyperparameter Grid Search")
    print("=" * 60)
    
    # Get base configuration
    base_config = get_default_config(dataset_name)
    base_config.num_epochs = max_epochs
    base_config.experiment_name = experiment_name
    
    # Generate parameter combinations
    combinations = generate_hyperparameter_combinations(grid_name)
    total_combinations = len(combinations)
    
    print(f"📊 Dataset: {dataset_name}")
    print(f"🔧 Grid: {grid_name}")
    print(f"📈 Total combinations: {total_combinations}")
    print(f"⏰ Max epochs per run: {max_epochs}")
    print(f"🕐 Estimated time: {total_combinations * max_epochs * 0.5:.1f} minutes")
    print()
    
    results = []
    start_time = time.time()
    
    for i, params in enumerate(combinations):
        print(f"🚀 Running combination {i+1}/{total_combinations}")
        print(f"Parameters: {params}")
        
        # Create configuration for this combination
        config = create_config_from_dict(base_config, params)
        config.run_name = f"grid_search_{i+1:03d}_{grid_name}"
        
        try:
            # Train model
            run_start = time.time()
            history = train_model(config)
            run_time = time.time() - run_start
            
            # Get best validation loss
            best_val_loss = min(history['val_loss']) if history['val_loss'] else float('inf')
            final_train_loss = history['train_loss'][-1]
            final_val_distance_error = (history['val_distance_error'][-1] 
                                      if history['val_distance_error'] else float('inf'))
            
            # Store results
            result = {
                'combination_id': i + 1,
                'parameters': params,
                'best_val_loss': best_val_loss,
                'final_train_loss': final_train_loss,
                'final_val_distance_error': final_val_distance_error,
                'training_time': run_time,
                'epochs_completed': len(history['train_loss']),
                'config': config.to_dict()
            }
            results.append(result)
            
            print(f"✅ Completed in {run_time:.1f}s")
            print(f"   Best Val Loss: {best_val_loss:.6f}")
            print(f"   Final Distance Error: {final_val_distance_error:.2f} pixels")
            print()
            
        except Exception as e:
            print(f"❌ Failed: {str(e)}")
            result = {
                'combination_id': i + 1,
                'parameters': params,
                'error': str(e),
                'best_val_loss': float('inf'),
                'final_train_loss': float('inf'),
                'final_val_distance_error': float('inf'),
                'training_time': 0,
                'epochs_completed': 0
            }
            results.append(result)
            print()
    
    total_time = time.time() - start_time
    
    # Print summary
    print("🏁 Grid Search Complete!")
    print("=" * 60)
    print(f"⏰ Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"📊 Successful runs: {sum(1 for r in results if 'error' not in r)}/{total_combinations}")
    
    # Find best results
    successful_results = [r for r in results if 'error' not in r]
    if successful_results:
        best_result = min(successful_results, key=lambda x: x['best_val_loss'])
        
        print(f"\n🏆 Best Result:")
        print(f"   Combination: {best_result['combination_id']}")
        print(f"   Parameters: {best_result['parameters']}")
        print(f"   Best Val Loss: {best_result['best_val_loss']:.6f}")
        print(f"   Final Distance Error: {best_result['final_val_distance_error']:.2f} pixels")
        
        # Save best configuration
        best_config_path = f"configs/best_config_{dataset_name}_{grid_name}.py"
        with open(best_config_path, 'w') as f:
            f.write(f"# Best configuration from grid search\n")
            f.write(f"# Dataset: {dataset_name}, Grid: {grid_name}\n")
            f.write(f"# Best Val Loss: {best_result['best_val_loss']:.6f}\n")
            f.write(f"# Distance Error: {best_result['final_val_distance_error']:.2f} pixels\n\n")
            f.write(f"BEST_PARAMS = {best_result['parameters']}\n")
        
        print(f"💾 Best config saved to: {best_config_path}")
    
    return results


def run_quick_test():
    """Run a quick test with minimal parameters."""
    print("🧪 Running Quick Test")
    print("-" * 30)
    
    # Create minimal config for testing
    config = get_default_config("T500")
    config.num_epochs = 3
    config.batch_size = 8
    config.experiment_name = "quick_test"
    config.run_name = "quick_test_run"
    config.early_stopping_patience = 2
    
    print(f"Testing with {config.num_epochs} epochs...")
    
    try:
        history = train_model(config)
        print("✅ Quick test successful!")
        print(f"Final train loss: {history['train_loss'][-1]:.6f}")
        if history['val_loss']:
            print(f"Final val loss: {history['val_loss'][-1]:.6f}")
        return True
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hyperparameter Grid Search")
    parser.add_argument("--dataset", default="T500", choices=["T250", "T500"],
                       help="Dataset to use")
    parser.add_argument("--grid", default="quick_search", 
                       choices=list(HYPERPARAMETER_GRIDS.keys()),
                       help="Hyperparameter grid to use")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Maximum epochs per run")
    parser.add_argument("--experiment", default="grid_search",
                       help="MLflow experiment name")
    parser.add_argument("--test", action="store_true",
                       help="Run quick test instead of full grid search")
    
    args = parser.parse_args()
    
    if args.test:
        success = run_quick_test()
        sys.exit(0 if success else 1)
    else:
        results = run_grid_search(
            dataset_name=args.dataset,
            grid_name=args.grid,
            max_epochs=args.epochs,
            experiment_name=args.experiment
        )
        
        print(f"\n📈 View results in MLflow UI:")
        print(f"   cd {os.getcwd()}")
        print(f"   mlflow ui")
        print(f"   Open: http://localhost:5000") 