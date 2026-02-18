# WGPU Monte Carlo

GPU-accelerated Monte Carlo integration using WebGPU (wgpu) and PyO3. Calculate expected values E[f(X)] for millions of samples in parallel on your GPU.

## Features

- **GPU Acceleration**: WebGPU compute for massive parallel execution
- **Multi-Function Fusion**: Evaluate K functions on same samples in one GPU pass
- **Pythonic API**: Write Python functions, auto-transpiled to WGSL shaders
- **Native Distributions**: Uniform, Normal (Box-Muller), Exponential (analytical sampling)
- **Custom Distributions**: Define any distribution via PDF function, auto-generates CDF table
- **Beta/Gamma**: Built-in support for common distributions
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

All distributions provide a unified `pdf(x)` interface for importance sampling.

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
