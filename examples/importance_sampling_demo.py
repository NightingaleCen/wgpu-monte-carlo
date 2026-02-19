#!/usr/bin/env python3
"""Importance Sampling Example

Estimate E_p[f(X)] by sampling from a different proposal distribution q.
"""

import math
from wgpu_montecarlo import MonteCarloIntegrator, Distribution

# Create integrator
integrator = MonteCarloIntegrator()

# Target: N(0, 1), Proposal: N(0.5, 1.5)
target = Distribution.normal(0.0, 1.0)
proposal = Distribution.normal(0.5, 1.5)

# Compute E_p[X] and E_p[X²] using importance sampling
result = integrator.integrate_importance_sampling(
    [lambda x: x, lambda x: x**2],
    target,
    proposal,
    n_samples=10_000_000,
)

print(f"E_p[X]  = {result.values[0]:+.6f}  (expected: 0.0)")
print(f"E_p[X²] = {result.values[1]:.6f}  (expected: 1.0)")
