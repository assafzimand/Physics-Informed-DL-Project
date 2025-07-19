"""
2D Wave Simulation for Physics-Informed Deep Learning Project

This module implements a 2D wave equation solver with reflecting boundary conditions
to generate training data for wave source localization.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import sys
import os

# Add configs directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'configs'))
import wave_simulation_config as config


class Wave2DSimulator:
    """
    2D Wave equation simulator with reflecting boundary conditions.
    
    Solves: ∂²u/∂t² = c²(∂²u/∂x² + ∂²u/∂y²)
    where u is wave amplitude, c is wave speed
    """
    
    def __init__(self, grid_size: int = None, wave_speed: float = None, dt: float = None, dx: float = None):
        """
        Initialize the wave simulator.
        
        Args:
            grid_size: Size of the square grid (grid_size x grid_size)
            wave_speed: Speed of wave propagation
            dt: Time step size
            dx: Spatial step size
        """
        self.grid_size = grid_size
        self.wave_speed = wave_speed
        self.dt = dt
        self.dx = dx
        
        # Stability condition for wave equation: c*dt/dx <= 1/sqrt(2) for 2D
        self.cfl_condition = wave_speed * dt / dx
        if self.cfl_condition > 1.0 / np.sqrt(2):
            raise ValueError(f"CFL condition violated: {self.cfl_condition:.3f} > {1.0/np.sqrt(2):.3f}")
        
        # Pre-compute coefficients for efficiency
        self.r_squared = (wave_speed * dt / dx) ** 2
        
    def create_initial_conditions(self, source_x: int, source_y: int, 
                                source_amplitude: float = 1.0, source_width: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create initial wave conditions with a Gaussian pulse at the source location.
        
        Args:
            source_x: X coordinate of wave source (0 to grid_size-1)
            source_y: Y coordinate of wave source (0 to grid_size-1)
            source_amplitude: Initial amplitude of the wave
            source_width: Width of the initial Gaussian pulse
            
        Returns:
            Tuple of (u_current, u_previous) - current and previous time step arrays
        """
        # Create coordinate grids
        x = np.arange(self.grid_size)
        y = np.arange(self.grid_size)
        X, Y = np.meshgrid(x, y)
        
        # Initial wave as Gaussian pulse centered at source
        distance_squared = (X - source_x)**2 + (Y - source_y)**2
        u_current = source_amplitude * np.exp(-distance_squared / (2 * source_width**2))
        
        # Assume zero initial velocity (u_previous = u_current for first step)
        u_previous = u_current.copy()
        
        return u_current, u_previous
    
    def apply_reflecting_boundaries(self, u: np.ndarray) -> np.ndarray:
        """
        Apply reflecting boundary conditions to the wave field.
        Sets gradient to zero at boundaries (Neumann boundary conditions).
        
        Args:
            u: Wave field array
            
        Returns:
            Wave field with reflecting boundaries applied
        """
        # Copy to avoid modifying original
        u_bc = u.copy()
        
        # Reflecting boundaries: ∂u/∂n = 0 at boundaries
        # This means the derivative normal to the boundary is zero
        u_bc[0, :] = u_bc[1, :]      # Top boundary
        u_bc[-1, :] = u_bc[-2, :]    # Bottom boundary
        u_bc[:, 0] = u_bc[:, 1]      # Left boundary
        u_bc[:, -1] = u_bc[:, -2]    # Right boundary
        
        return u_bc
    
    def simulate_step(self, u_current: np.ndarray, u_previous: np.ndarray) -> np.ndarray:
        """
        Perform one time step of the wave equation using finite differences.
        
        Uses the explicit finite difference scheme:
        u(t+dt) = 2*u(t) - u(t-dt) + r²[∇²u(t)]
        
        Args:
            u_current: Wave field at current time
            u_previous: Wave field at previous time
            
        Returns:
            Wave field at next time step
        """
        # Compute Laplacian using finite differences
        laplacian = np.zeros_like(u_current)
        
        # Interior points
        laplacian[1:-1, 1:-1] = (
            u_current[2:, 1:-1] + u_current[:-2, 1:-1] +      # d²u/dx²
            u_current[1:-1, 2:] + u_current[1:-1, :-2] -      # d²u/dy²
            4 * u_current[1:-1, 1:-1]                         # -4u(i,j)
        )
        
        # Update equation: u_new = 2*u_current - u_previous + r²*laplacian
        u_next = 2 * u_current - u_previous + self.r_squared * laplacian
        
        # Apply reflecting boundary conditions
        u_next = self.apply_reflecting_boundaries(u_next)
        
        return u_next
    
    def simulate(self, source_x: int, source_y: int, num_timesteps: int, 
                save_interval: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run complete wave simulation from t=0 to t=num_timesteps.
        
        Args:
            source_x: X coordinate of wave source
            source_y: Y coordinate of wave source  
            num_timesteps: Number of time steps to simulate
            save_interval: If provided, save intermediate steps every save_interval steps
            
        Returns:
            Tuple of (final_wave_field, history) where history contains intermediate steps if requested
        """
        # Initialize wave fields
        u_current, u_previous = self.create_initial_conditions(source_x, source_y)
        
        # Storage for intermediate results if requested
        history = []
        if save_interval is not None:
            history.append(u_current.copy())
        
        # Time stepping loop
        for step in range(num_timesteps):
            u_next = self.simulate_step(u_current, u_previous)
            
            # Update for next iteration
            u_previous = u_current
            u_current = u_next
            
            # Save intermediate result if requested
            if save_interval is not None and (step + 1) % save_interval == 0:
                history.append(u_current.copy())
        
        return u_current, np.array(history) if history else None
    
    def visualize_wave(self, wave_field: np.ndarray, source_x: int, source_y: int, 
                      title: str = "Wave Field", save_path: Optional[str] = None):
        """
        Visualize the wave field.
        
        Args:
            wave_field: 2D array of wave amplitudes
            source_x: X coordinate of original source
            source_y: Y coordinate of original source
            title: Title for the plot
            save_path: If provided, save the plot to this path
        """
        plt.figure(figsize=(10, 8))
        
        # Create the heatmap
        im = plt.imshow(wave_field, cmap='RdBu_r', origin='lower', 
                       extent=[0, self.grid_size-1, 0, self.grid_size-1])
        
        # Mark the source location
        plt.plot(source_x, source_y, 'ko', markersize=8, markerfacecolor='yellow', 
                markeredgecolor='black', markeredgewidth=2, label='Source')
        
        plt.colorbar(im, label='Wave Amplitude')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()


def test_wave_simulator():
    """Test the wave simulator with a simple example."""
    print("Testing Wave2D Simulator...")
    
    # Create simulator
    sim = Wave2DSimulator(grid_size=64, wave_speed=0.5, dt=0.1, dx=1.0)
    
    # Test with source in center
    source_x, source_y = 32, 32
    final_wave, _ = sim.simulate(source_x, source_y, num_timesteps=100)
    
    # Visualize result
    sim.visualize_wave(final_wave, source_x, source_y, 
                      title="Test Wave Simulation (t=100)")
    
    print(f"Simulation completed successfully!")
    print(f"Final wave field shape: {final_wave.shape}")
    print(f"Wave amplitude range: [{final_wave.min():.3f}, {final_wave.max():.3f}]")


if __name__ == "__main__":
    test_wave_simulator() 