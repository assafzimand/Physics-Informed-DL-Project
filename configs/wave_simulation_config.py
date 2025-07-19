"""
Wave Simulation Configuration

Contains all the optimized parameters for wave simulation and dataset generation.
These parameters were determined through testing to provide stable, realistic wave behavior.
"""

# =============================================================================
# PHYSICS PARAMETERS
# =============================================================================

# Grid and spatial parameters
GRID_SIZE = 128                 # Size of simulation grid (128x128)
DX = 1.0                       # Spatial step size
DY = 1.0                       # Spatial step size (same as DX for square grid)

# Wave propagation parameters  
WAVE_SPEED = 16.7              # Wave propagation speed (optimized for CFL=0.5)
DT = 0.03                      # Time step size
CFL_TARGET = 0.5               # Target CFL condition (actual: wave_speed * dt / dx)

# Initial condition parameters
SOURCE_AMPLITUDE = 1.0         # Initial wave amplitude
SOURCE_WIDTH = 2.0             # Width of initial Gaussian pulse
SOURCE_MARGIN = 10             # Minimum distance from grid edges for sources

# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================

# Dataset generation parameters
TIMESTEPS_T1 = 250            # First dataset: shorter time evolution
TIMESTEPS_T2 = 500            # Second dataset: longer time evolution
NUM_SAMPLES_PER_DATASET = 2000 # Number of samples per timestep dataset

# Animation and visualization parameters
ANIMATION_SAVE_INTERVAL = 5    # Save every Nth timestep for animation
ANIMATION_FPS_INTERVAL = 30    # Milliseconds between animation frames (~33 FPS)

# =============================================================================
# FILE PATHS AND STORAGE
# =============================================================================

# Data storage
DATA_DIR = "data"
DATASET_FILENAME = "wave_datasets.h5"
VISUALIZATION_DIR = "results/sample_visualizations"
MODELS_DIR = "results/models"

# =============================================================================
# VALIDATION AND CHECKS
# =============================================================================

def validate_parameters():
    """
    Validate that the configuration parameters are consistent and safe.
    
    Returns:
        bool: True if parameters are valid, False otherwise
    """
    # Check CFL condition for stability
    cfl_actual = WAVE_SPEED * DT / DX
    cfl_limit_2d = 1.0 / (2**0.5)  # ~0.707 for 2D
    
    if cfl_actual > cfl_limit_2d:
        print(f"❌ CFL condition violated!")
        print(f"   Actual CFL: {cfl_actual:.3f}")
        print(f"   2D limit: {cfl_limit_2d:.3f}")
        return False
    
    # Check grid size is reasonable
    if GRID_SIZE < 64 or GRID_SIZE > 512:
        print(f"❌ Grid size {GRID_SIZE} may be unreasonable")
        return False
    
    # Check source margin
    if SOURCE_MARGIN >= GRID_SIZE // 4:
        print(f"❌ Source margin {SOURCE_MARGIN} too large for grid {GRID_SIZE}")
        return False
    
    # All checks passed
    print(f"✅ Configuration validation passed:")
    print(f"   Grid: {GRID_SIZE}×{GRID_SIZE}")
    print(f"   Wave speed: {WAVE_SPEED}")
    print(f"   CFL: {cfl_actual:.3f} (limit: {cfl_limit_2d:.3f})")
    print(f"   Timesteps: T1={TIMESTEPS_T1}, T2={TIMESTEPS_T2}")
    print(f"   Samples per dataset: {NUM_SAMPLES_PER_DATASET}")
    
    return True

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_simulator_params():
    """Get parameters for Wave2DSimulator initialization."""
    return {
        'grid_size': GRID_SIZE,
        'wave_speed': WAVE_SPEED,
        'dt': DT,
        'dx': DX
    }

def get_initial_condition_params():
    """Get parameters for initial wave conditions."""
    return {
        'source_amplitude': SOURCE_AMPLITUDE,
        'source_width': SOURCE_WIDTH
    }

def get_dataset_params():
    """Get parameters for dataset generation."""
    return {
        'timesteps': [TIMESTEPS_T1, TIMESTEPS_T2],
        'num_samples': NUM_SAMPLES_PER_DATASET,
        'source_margin': SOURCE_MARGIN
    }

def get_visualization_params():
    """Get parameters for visualization and animation."""
    return {
        'save_interval': ANIMATION_SAVE_INTERVAL,
        'animation_interval': ANIMATION_FPS_INTERVAL,
        'vis_dir': VISUALIZATION_DIR
    }

def get_storage_params():
    """Get parameters for data storage."""
    return {
        'data_dir': DATA_DIR,
        'filename': DATASET_FILENAME,
        'models_dir': MODELS_DIR
    }

# Validate configuration on import
if __name__ == "__main__":
    validate_parameters()
else:
    # Silent validation when imported
    is_valid = validate_parameters()
    if not is_valid:
        raise ValueError("Invalid wave simulation configuration!") 