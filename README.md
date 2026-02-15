# WGPU Monte Carlo

GPU-accelerated Monte Carlo integration using WebGPU (wgpu) and PyO3. Calculate expected values E[f(X)] for millions of samples in parallel on your GPU.

## Features

- **GPU Acceleration**: WebGPU compute for massive parallel execution
- **Multi-Function Fusion**: Evaluate K functions on same samples in one GPU pass
- **Pythonic API**: Write Python functions, auto-transpiled to WGSL shaders
- **Native Distributions**: Uniform, Normal (Box-Muller), Exponential
- **Table-Based Sampling**: Beta, Gamma, and arbitrary distributions via lookup tables
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
# Built-in
dist = Distribution.normal(mean=0.0, std=1.0)
dist = Distribution.uniform(min=0.0, max=1.0)
dist = Distribution.exponential(lambda_param=2.0)

# Table-based (requires scipy)
dist = Distribution.beta(alpha=2.0, beta=5.0)
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

### Custom Table Distribution

```python
import numpy as np
from scipy.stats import gamma

# Precompute inverse CDF
probs = np.linspace(0, 1, 2048, endpoint=False)
probs = np.clip(probs, 1e-7, 1 - 1e-7)
table = gamma.ppf(probs, 2.0, scale=1.0).astype(np.float32)

dist = Distribution.from_table(table)
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

Supported import styles: `math.sin(x)`, `np.sin(x)`, `from math import sin`.

## Testing

```bash
uv run pytest tests/ -v
```

## Project Structure

```
├── python/wgpu_montecarlo/   # Python package
│   ├── __init__.py           # Public API
│   └── transpiler.py         # Python → WGSL
├── src/                      # Rust source
│   ├── lib.rs               # PyO3 bindings
│   ├── engine.rs            # wgpu compute
│   └── distribution.rs      # WGSL distributions
└── tests/
```

## License

MIT
