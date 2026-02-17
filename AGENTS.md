# AGENTS.md - Coding Guidelines for wgpu-montecarlo

## Project Overview

Hybrid Rust + Python GPU-accelerated Monte Carlo simulation library using:
- **Rust**: wgpu compute engine, PyO3 bindings (`src/`)
- **Python**: User API, Python→WGSL transpiler (`python/`)
- **Build**: Maturin (via uv) for Python extension compilation

## Build Commands

```bash
# Development build (creates editable Python package)
uv run maturin develop

# Release build (optimized)
uv run maturin develop --release

# Build wheel for distribution
uv run maturin build --release

# Clean and rebuild
rm -rf target/ python/wgpu_montecarlo/*.so && uv run maturin develop
```

## Test Commands

```bash
# Run all tests
uv run pytest tests/ -v

# Run single test file
uv run pytest tests/test_transpiler.py -v
uv run pytest tests/test_integration.py -v
uv run pytest tests/test_beta_distribution.py -v

# Run specific test class
uv run pytest tests/test_transpiler.py::TestTranspiler -v

# Run specific test method
uv run pytest tests/test_transpiler.py::TestTranspiler::test_simple_function -v

# Run with GPU tests (requires compatible GPU)
uv run pytest tests/test_integration.py -v

# Run transpiler tests only (no GPU required)
uv run pytest tests/test_transpiler.py -v
```

## Lint/Format Commands

```bash
# Format Rust code
cargo fmt

# Check Rust code
cargo check

# Python linting (project uses ruff via uv)
uv run ruff check python/

# Python formatting
uv run ruff format python/
```

## Code Style Guidelines

### Python Style

- **Imports**: Group as: stdlib → third-party → local. Use absolute imports.
- **Formatting**: Use ruff (PEP 8 compliant). Line length: 88-100 chars.
- **Types**: Use type hints for function signatures (`typing` module).
- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes, `SCREAMING_SNAKE_CASE` for constants.
- **Docstrings**: Use triple quotes with Args/Returns/Raises sections for public APIs.
- **Error Handling**: Raise specific exceptions (`ValueError`, `TypeError`, `RuntimeError`). Custom `TranspilerError` for transpiler.
- **Comments**: Keep comments concise and in English. Explain *what* the code does, not *why*. Focus on non-obvious parts.

### Rust Style

- **Imports**: Group as: std → third-party → crate → super/self. Use `use` statements.
- **Formatting**: Use `cargo fmt`. Standard Rust naming conventions.
- **Naming**: `snake_case` for functions/variables, `PascalCase` for types/structs/enums, `SCREAMING_SNAKE_CASE` for constants.
- **Error Handling**: Use `anyhow::Result` for internal errors, `PyResult` for Python-facing functions. Map errors with context.
- **Unsafe**: Minimize unsafe code; comment rationale when necessary.
- **Comments**: Keep comments concise and in English. Explain *what* the code does, not *why*.

### General Guidelines

- **Documentation**: Docstrings for all public Python APIs; `///` comments for public Rust items.
- **Testing**: Write tests for new transpiler features. Integration tests require GPU.
- **Git**: Follow conventional commits (feat:, fix:, docs:, test:, refactor:).
- **Python Version**: Supports 3.11+. Use modern Python features (match, union types with `|`).
- **Examples**: Keep examples minimal and focused. Demonstrate core functionality only. No verbose output or unrelated content.

## Project Structure

```
├── src/                     # Rust source code
│   ├── lib.rs              # PyO3 bindings, main module
│   ├── engine.rs           # wgpu compute engine
│   ├── distribution.rs     # WGSL distribution library
│   └── shader_gen.rs       # Shader code generation
├── python/wgpu_montecarlo/ # Python package
│   ├── __init__.py         # Public API (Integrator, Distribution, Simulator)
│   └── transpiler.py       # Python AST → WGSL transpiler
├── tests/                   # Test suite
│   ├── test_transpiler.py  # Transpiler unit tests (no GPU)
│   ├── test_integration.py # GPU integration tests
│   └── test_beta_distribution.py  # Distribution-specific tests
├── examples/               # Example scripts
├── pyproject.toml          # Python package config, tool settings
└── Cargo.toml             # Rust package config
```

## Transpiler Constraints

The Python→WGSL transpiler supports a restricted subset:
- Arithmetic: `+`, `-`, `*`, `/`, `%`, `**` (becomes `pow()`)
- Comparisons: `>`, `<`, `>=`, `<=`, `==`, `!=`
- Math functions: `sin`, `cos`, `sqrt`, `exp`, `log`, etc. (from `math` or `numpy` module)
- Conditionals: `x if cond else y` (becomes `select()`)
- Local variables: `var` declarations in WGSL
- Boolean results: Auto-converted to f32 via `select(0.0, 1.0, cond)`
- Constants: `math.pi`, `math.e`, `math.tau`, `numpy.euler_gamma`, etc.
- External variables: Automatically captured from globals and closures

**Not supported**: Loops, complex control flow, list/dict operations.

## Common Development Tasks

```bash
# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate

# Run an example
uv run python examples/integration_demo.py

# Verify installation
uv run pytest tests/ -v --tb=short
```

## GPU Requirements

- **macOS**: Metal 2.0+ support
- **Linux**: Vulkan drivers (mesa-vulkan-drivers, NVIDIA proprietary)
- **Windows**: DirectX 12 or Vulkan drivers

Set backend explicitly if needed: `WGPU_BACKEND=metal` or `WGPU_BACKEND=vulkan`

## Troubleshooting

- **"No GPU adapter found"**: Check GPU drivers, ensure WebGPU-compatible GPU
- **"Failed to create device"**: Update GPU drivers, check for GPU conflicts
- **Import errors**: Run `uv run maturin develop` to rebuild extension
- **WGSL compilation errors**: Ensure transpiler-supported Python subset only

## Dependencies

- **Required**: numpy, scipy (for distributions)
- **Dev**: pytest, maturin
- **Rust**: wgpu, pyo3, anyhow, bytemuck, pollster, ndarray

See `pyproject.toml` and `Cargo.toml` for versions.
