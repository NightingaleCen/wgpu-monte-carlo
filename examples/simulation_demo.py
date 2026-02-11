#!/usr/bin/env python3
"""Simple Monte Carlo Simulation Example

Simulate a random walk and compute the mean displacement.
"""

import numpy as np
from wgpu_montecarlo import MonteCarloSimulator

# Create simulator
sim = MonteCarloSimulator()

# Start with 1000 particles at position 0
initial = np.zeros(1000, dtype=np.float32)

# Run simulation with default random walk
result = sim.run(initial, iterations=100, seed=42)

# Compute statistics
print(f"Mean displacement: {result.mean():.6f}")
print(f"Std deviation:     {result.std():.6f}")
print(f"Min position:      {result.min():.6f}")
print(f"Max position:      {result.max():.6f}")
