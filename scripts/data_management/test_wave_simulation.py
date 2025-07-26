"""
Quick test script for wave simulation with animation
"""

import sys
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random

# Add src and configs directories to path (relative to tests/)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(os.path.join(project_root, 'src'))
sys.path.append(os.path.join(project_root, 'configs'))

from wave_simulation import Wave2DSimulator
import wave_simulation_config as config


def get_random_source_location():
    """Generate random source location anywhere in the grid (no margins)."""
    x = random.randint(0, config.GRID_SIZE - 1)
    y = random.randint(0, config.GRID_SIZE - 1)
    return x, y


def get_user_timesteps():
    """Get timesteps choice from user."""
    print("\nChoose simulation timesteps:")
    print("1. T=250 (shorter evolution)")
    print("2. T=500 (longer evolution)") 
    print("3. Custom timesteps")
    
    choice = input("Enter choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        return config.TIMESTEPS_T1
    elif choice == "2":
        return config.TIMESTEPS_T2
    elif choice == "3":
        while True:
            try:
                custom_t = int(input("Enter custom timesteps (e.g., 100): "))
                if custom_t > 0:
                    return custom_t
                else:
                    print("Please enter a positive number")
            except ValueError:
                print("Please enter a valid number")
    else:
        print("Invalid choice, using default T=250")
        return config.TIMESTEPS_T1


def animated_test():
    """Test with animated wave propagation - FULL SCALE"""
    print("🌊 Testing Wave Simulation with Animation (FULL SCALE)...")
    
    # Create simulator using config parameters
    sim_params = config.get_simulator_params()
    sim = Wave2DSimulator(**sim_params)
    
    print(f"Grid size: {sim.grid_size}×{sim.grid_size}")
    print(f"Wave speed: {sim.wave_speed}")
    print(f"Time step: {sim.dt}")
    print(f"CFL condition: {sim.cfl_condition:.3f} (should be < 0.707)")
    
    # Get user choice for timesteps
    num_steps = get_user_timesteps()
    
    # Generate random source location (no margins)
    source_x, source_y = get_random_source_location()
    print(f"Simulating wave from RANDOM source at ({source_x}, {source_y})")
    
    # Initialize wave fields using config parameters
    init_params = config.get_initial_condition_params()
    u_current, u_previous = sim.create_initial_conditions(source_x, source_y, **init_params)
    
    # Store wave history for animation using config
    wave_history = [u_current.copy()]
    vis_params = config.get_visualization_params()
    save_interval = vis_params['save_interval']
    print(f"Running {num_steps} time steps (saving every {save_interval} steps)...")
    
    for step in range(num_steps):
        u_next = sim.simulate_step(u_current, u_previous)
        u_previous = u_current
        u_current = u_next
        
        # Save every save_interval steps for animation
        if (step + 1) % save_interval == 0:
            wave_history.append(u_current.copy())
    
    print("Creating animation...")
    
    # Create animation
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Set up the plot
    wave_data = wave_history[0]
    vmin, vmax = np.min(wave_history), np.max(wave_history)
    
    im = ax.imshow(wave_data, cmap='RdBu_r', origin='lower', 
                   vmin=vmin, vmax=vmax, animated=True)
    
    # Mark source location
    ax.plot(source_x, source_y, 'ko', markersize=8, 
            markerfacecolor='yellow', markeredgecolor='black', markeredgewidth=2)
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(f'Wave Propagation Animation (T={num_steps})')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Wave Amplitude')
    
    # Add time counter
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                       fontsize=12, color='white', weight='bold',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
    
    def animate(frame):
        """Animation function"""
        im.set_array(wave_history[frame])
        actual_timestep = frame * save_interval
        time_text.set_text(f'Time step: {actual_timestep}')
        return [im, time_text]
    
    # Create animation (fast playback)
    anim = animation.FuncAnimation(fig, animate, frames=len(wave_history),
                                  interval=30, blit=True, repeat=True)
    
    print("🎬 Animation ready! Close the window when done watching.")
    plt.show()
    
    # Final statistics
    final_wave = wave_history[-1]
    print("✅ Animation completed!")
    print(f"Final wave field shape: {final_wave.shape}")
    print(f"Amplitude range: [{final_wave.min():.3f}, {final_wave.max():.3f}]")


def quick_test():
    """Static test with same parameters as animated version"""
    print("🌊 Testing Wave Simulation (Static - FULL SCALE)...")
    
    # Create simulator using config parameters - same as animated
    sim_params = config.get_simulator_params()
    sim = Wave2DSimulator(**sim_params)
    
    print(f"Grid size: {sim.grid_size}×{sim.grid_size}")
    print(f"Wave speed: {sim.wave_speed}")
    print(f"Time step: {sim.dt}")
    print(f"CFL condition: {sim.cfl_condition:.3f} (should be < 0.707)")
    
    # Get user choice for timesteps
    num_timesteps = get_user_timesteps()
    
    # Generate random source location (no margins) - same as animated
    source_x, source_y = get_random_source_location()
    print(f"Simulating wave from RANDOM source at ({source_x}, {source_y})")
    
    # Run full simulation using user-chosen timesteps
    final_wave, _ = sim.simulate(source_x, source_y, num_timesteps=num_timesteps)
    
    print("Simulation completed!")
    print(f"Wave field shape: {final_wave.shape}")
    print(f"Amplitude range: [{final_wave.min():.3f}, {final_wave.max():.3f}]")
    
    # Simple visualization
    plt.figure(figsize=(8, 6))
    plt.imshow(final_wave, cmap='RdBu_r', origin='lower')
    plt.colorbar(label='Wave Amplitude')
    plt.title(f'Test Wave Simulation (T={num_timesteps})')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.plot(source_x, source_y, 'ko', markersize=8, markerfacecolor='yellow')
    plt.show()
    
    print("✅ Test completed successfully!")


if __name__ == "__main__":
    print("Choose test mode:")
    print("1. Animated wave propagation (recommended)")
    print("2. Static final result")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1" or choice == "":
        animated_test()
    else:
        quick_test() 