#!/usr/bin/env python3
"""MCMC (Metropolis-Hastings) Example

Use Markov Chain Monte Carlo to sample from a target distribution
when direct sampling is not available.
"""

from wgpu_montecarlo import MonteCarloIntegrator, Distribution

# Create integrator
integrator = MonteCarloIntegrator()

# Target: N(0, 1), Proposal: N(0, 2) - wider proposal
target = Distribution.normal(0.0, 1.0)
proposal = Distribution.normal(0.0, 2.0)

# Run MCMC with 4096 parallel chains
result = integrator.integrate_mcmc(
    [lambda x: x, lambda x: x**2],
    target,
    proposal,
    n_steps=10000,
    n_chains=4096,
    n_burnin=1000,
)

print(f"E[X]  = {result.values[0]:+.6f}  (expected: 0.0)")
print(f"E[X²] = {result.values[1]:.6f}  (expected: 1.0)")
