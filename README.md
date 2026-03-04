# WGPU Monte Carlo

GPU-accelerated Monte Carlo integration using WebGPU (wgpu) and PyO3. Calculate expected values E[f(X)] for millions of samples in parallel on your GPU.

## Features

- **GPU Acceleration**: WebGPU compute for massive parallel execution
- **Multi-Function Fusion**: Evaluate K functions on same samples in one GPU pass
- **Importance Sampling**: Efficient estimation for rare events using proposal distributions
- **MCMC**: Metropolis-Hastings algorithm for sampling from arbitrary distributions
- **Pythonic API**: Write Python functions, auto-transpiled to WGSL shaders
- **Native Distributions**: Uniform, Normal (Box-Muller), Exponential (analytical sampling)
- **Custom Distributions**: Define any distribution via PDF function or pre-computed tables
- **Cross-Platform**: Works on macOS, Linux, Windows

## Installation

Requires Python 3.11+, Rust 1.70+, and a GPU with WebGPU support.

```bash
git clone <repo-url>
cd wgpu-montecarlo
uv sync
uv run maturin develop
```

## Quick Start

```python
from wgpu_montecarlo import MonteCarloIntegrator, Distribution

# Calculate variance of standard normal
integrator = MonteCarloIntegrator()
dist = Distribution.normal(mean=0.0, std=1.0)

result = integrator.integrate(
    [lambda x: x, lambda x: x**2],  # E[X] and E[X²]
    dist,
    n_samples=1_000_000
)

mean = result.values[0]
variance = result.values[1] - mean**2
print(f"Variance = {variance:.6f}")  # ~1.0
```

## Usage

### Distributions

```python
# Built-in (analytical sampling)
dist = Distribution.normal(mean=0.0, std=1.0)
dist = Distribution.uniform(min=0.0, max=1.0)
dist = Distribution.exponential(lambda_param=2.0)

# Beta distribution (built-in)
dist = Distribution.beta(alpha=2.0, beta_param=5.0)

# Custom distribution from PDF function
import math
def my_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

dist = Distribution.from_pdf(my_pdf)  # Auto-detects support and generates CDF table

# Custom distribution from pre-computed table
import numpy as np
x_grid = np.linspace(0, 10, 512)
pdf_vals = np.exp(-x_grid)  # Exponential decay
dist = Distribution.from_pdf_table(x_grid, pdf_vals)
```

### Multiple Functions

All functions are evaluated on the **same random samples**:

```python
functions = [
    lambda x: x,           # E[X]
    lambda x: x**2,        # E[X²]
    lambda x: x > 0.5,     # P(X > 0.5)
]

result = integrator.integrate(functions, dist, n_samples=10_000_000)
```

### Importance Sampling

Estimate expectations under a target distribution p(x) by sampling from a proposal distribution q(x):

```python
from wgpu_montecarlo import MonteCarloIntegrator, Distribution

integrator = MonteCarloIntegrator()

# Target: N(0, 1), Proposal: N(0.5, 1.5)
target = Distribution.normal(0.0, 1.0)
proposal = Distribution.normal(0.5, 1.5)

# Compute E_p[X] and E_p[X²] using importance sampling
result = integrator.integrate_importance_sampling(
    [lambda x: x, lambda x: x**2],
    target,
    proposal,
    n_samples=10_000_000
)

print(f"E_p[X] = {result.values[0]:.6f}")   # ~0.0
print(f"E_p[X²] = {result.values[1]:.6f}")  # ~1.0
```

### MCMC (Metropolis-Hastings)

Sample from arbitrary distributions using Markov Chain Monte Carlo:

```python
from wgpu_montecarlo import MonteCarloIntegrator, Distribution

integrator = MonteCarloIntegrator()

# Target distribution to sample from
target = Distribution.normal(0.0, 1.0)

# Proposal distribution for MH steps (can be different from target)
proposal = Distribution.normal(0.0, 2.0)

# Run MCMC with parallel chains
result = integrator.integrate_mcmc(
    [lambda x: x, lambda x: x**2],
    target,
    proposal,
    n_steps=10000,    # Samples per chain after burn-in
    n_chains=256,     # Parallel chains (GPU threads)
    n_burnin=1000,    # Initial steps to discard
)

print(f"E[X]  = {result.values[0]:+.6f}")  # ~0.0
print(f"E[X²] = {result.values[1]:.6f}")  # ~1.0
```

MCMC works with any combination of:
- **Analytical distributions**: Normal, Uniform, Exponential
- **Custom distributions**: From PDF function or table

```python
# Custom target distribution
def my_pdf(x):
    return 0.5 * x if 0 <= x <= 2 else 0  # Triangular on [0, 2]

target = Distribution.from_pdf(my_pdf, support=(0.0, 2.0))
proposal = Distribution.uniform(0.0, 2.0)

result = integrator.integrate_mcmc([lambda x: x], target, proposal, n_steps=5000)
```
```


#### Supported PDF Types

Importance sampling works with any combination of:

1. **Transpilable PDFs**: PDFs using only transpiler-supported operations (math functions, arithmetic)
2. **Table-based PDFs**: Non-transpilable PDFs automatically use lookup tables

```python
# Example: Non-transpilable PDF uses table lookup
def complex_pdf(x):
    return float(int(x) % 2) * 0.5 + 0.1  # int() not transpilable

target = Distribution.from_pdf(complex_pdf, support=(0.0, 10.0))
proposal = Distribution.uniform(0.0, 10.0)

# Automatically falls back to PDF table
result = integrator.integrate_importance_sampling(
    [lambda x: 1.0], target, proposal, n_samples=1_000_000
)
```

### Custom PDF Distribution

Define any distribution by providing its PDF function. The library automatically:

1. Detects the effective support (where PDF > 0)
2. Generates a normalized CDF lookup table
3. Uses the table for GPU sampling

```python
import math

# Define any PDF
def my_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

# Auto-generates CDF table and support
dist = Distribution.from_pdf(my_pdf)

# Or specify support manually for bounded distributions
def beta_pdf(x):
    if 0 < x < 1:
        return 6 * x * (1 - x)  # Beta(2, 2) unnormalized
    return 0.0

dist = Distribution.from_pdf(beta_pdf, support=(0.0, 1.0))
```

## Transpiler

Python functions are transpiled to WGSL at runtime. Supported features:

| Python | WGSL |
|--------|------|
| `+`, `-`, `*`, `/`, `%` | Same operators |
| `**` | `pow(a, b)` |
| `>`, `<`, `>=`, `<=`, `==`, `!=` | Same operators |
| `sin`, `cos`, `tan`, `asin`, `acos`, `atan` | Same functions |
| `sinh`, `cosh`, `tanh` | Same functions |
| `sqrt`, `exp`, `exp2`, `log`, `log2` | Same functions |
| `floor`, `ceil`, `round`, `trunc`, `fract`, `sign` | Same functions |
| `min`, `max`, `clamp`, `mix`, `step`, `smoothstep` | Same functions |
| `pow`, `power`, `abs` | Same functions |
| `x if cond else y` | `select(else_val, then_val, cond)` |
| `and`, `or` | `&&`, `\|\|` |

Supported import styles: `math.sin(x)`, `np.sin(x)`, `from math import sin`.

**Not supported**: Loops, complex control flow, list/dict operations.

## Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run importance sampling tests
uv run pytest tests/test_importance_sampling.py -v

# Run MCMC tests
uv run pytest tests/test_mcmc.py -v
```

## Project Structure

```
├── python/wgpu_montecarlo/   # Python package
│   ├── __init__.py           # Public API
│   └── transpiler.py         # Python → WGSL
├── src/                      # Rust source
│   ├── lib.rs               # PyO3 bindings
│   ├── engine.rs            # wgpu compute
│   ├── distribution.rs      # WGSL distributions
│   └── shader_gen.rs        # Shader generation
├── examples/                # Example scripts
│   ├── integration_demo.py  # Basic integration
│   ├── importance_sampling_demo.py  # IS example
│   └── mcmc_demo.py        # MCMC example
└── tests/
```

## License

MIT
