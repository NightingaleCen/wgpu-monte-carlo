#!/usr/bin/env python3
"""Simple Monte Carlo Integration Example

Calculate the variance of a standard normal distribution.
Variance = E[X²] - E[X]²
"""

from wgpu_montecarlo import MonteCarloIntegrator, Distribution

coeff_a = 1.0
coeff_b = 0.0

# Create integrator
integrator = MonteCarloIntegrator()

# Standard normal distribution N(0, 1)
dist = Distribution.normal(mean=0.0, std=1.0)

# Calculate E[X], E[X²], and E[a*X² + b*X]
funcs = [
    lambda x: x,
    lambda x: x**2,
    lambda x: coeff_a * x**2 + coeff_b * x,
]
result = integrator.integrate(funcs, dist, n_samples=100000000)

# Compute variance
mean = result.values[0]
variance = result.values[1] - mean**2

print(f"E[X]       = {result.values[0]:.6f}     (expected: 0.0)")
print(f"E[X²]      = {result.values[1]:.6f}  (expected: 1.0)")
print(f"Variance   = {variance:.6f}  (expected: 1.0)")
print(f"E[aX²+bX]  = {result.values[2]:.6f}  (expected: 1.0, a={coeff_a}, b={coeff_b})")
