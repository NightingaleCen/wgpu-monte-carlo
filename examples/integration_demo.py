#!/usr/bin/env python3
"""Simple Monte Carlo Integration Example

Calculate the variance of a standard normal distribution.
Variance = E[X²] - E[X]²
"""

from wgpu_montecarlo import MonteCarloIntegrator, Distribution

# Create integrator
integrator = MonteCarloIntegrator()

# Standard normal distribution N(0, 1)
dist = Distribution.normal(mean=0.0, std=1.0)

# Calculate E[X] and E[X²]
result = integrator.integrate([lambda x: x, lambda x: x**2], dist, n_samples=100000000)

# Compute variance
mean = result.values[0]
variance = result.values[1] - mean**2

print(f"E[X]     = {mean:.6f}     (expected: 0.0)")
print(f"E[X²]    = {result.values[1]:.6f}  (expected: 1.0)")
print(f"Variance = {variance:.6f}  (expected: 1.0)")
