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
uv run pytest tests/test_distributions.py -v

# Run specific test class
uv run pytest tests/test_transpiler.py::TestTranspiler -v

# Run specific test method
uv run pytest tests/test_transpiler.py::TestTranspiler::test_simple_function -v

# Run with GPU tests (requires compatible GPU)
uv run pytest tests/test_integration.py -v

# Run transpiler tests only (no GPU required)
uv run pytest tests/test_transpiler.py -v
```

## Testing Guidelines

### Philosophy

**Tests exist to find problems, not to simply pass.** When a test fails, it often reveals a real issue that needs to be fixed in the implementation, not in the test itself.

### Writing Tests

- **Test the real behavior**: Tests should verify actual functionality, not just exercise code paths.
- **Cover edge cases**: Consider boundary conditions, invalid inputs, and unusual but valid use cases.
- **Use meaningful assertions**: Verify specific values, not just that "nothing crashed."
- **Name tests descriptively**: Test names should describe what is being tested and expected behavior.

### When Tests Fail

1. **Do NOT modify tests to make them pass** if the test is checking correct behavior.
2. **DO fix the implementation** if the test reveals a real bug.
3. **DO update tests** only if:
   - The original test was checking incorrect behavior
   - The API has intentionally changed
4. **Document known limitations** in test docstrings if certain edge cases cannot be handled.

### Test Categories

| Category | Description | GPU Required |
|----------|-------------|--------------|
| Unit Tests | Test individual functions/modules | No |
| Integration Tests | Test end-to-end functionality | Yes |
| Transpiler Tests | Test Python→WGSL conversion | No |
| Distribution Tests | Test probability distributions | Yes |

### Distribution Tests

When adding tests for `Distribution.from_pdf()`:
- Test automatic support detection for bounded distributions (e.g., Beta in (0,1))
- Test support detection for shifted distributions (e.g., N(100, 1))
- Test edge cases: PDF returns NaN, inf, negative values
- Test that user-specified `support` parameter works correctly
- Test numerical accuracy with different `table_size` values

Example:
```python
def test_bounded_support_auto_detection(self):
    """Test that bounded distribution (0, 1) is auto-detected without support param."""
    # PDF only non-zero in (0, 1)
    def pdf(x):
        return 6.0 * x * (1.0 - x) if 0 < x < 1 else 0.0

    # Should auto-detect without manual support
    dist = Distribution.from_pdf(pdf)
    assert dist is not None
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
- **Testing**: Write tests for new features. Tests should verify actual behavior, not just pass. Integration tests require GPU.
- **Git**: Follow conventional commits (feat:, fix:, docs:, test:, refactor:).
- **Python Version**: Supports 3.11+. Use modern Python features (match, union types with `|`).
- **Examples**: Keep examples minimal and focused. Demonstrate core functionality only. No verbose output or unrelated content.

## Project Structure

```
├── src/                     # Rust source code
│   ├── lib.rs              # PyO3 bindings, main module
│   ├── engine.rs           # wgpu compute engine
│   ├── distribution.rs      # WGSL distribution library
│   └── shader_gen.rs       # Shader code generation
├── python/wgpu_montecarlo/ # Python package
│   ├── __init__.py         # Public API (Integrator, Distribution, Simulator)
│   └── transpiler.py       # Python AST → WGSL transpiler
├── tests/                   # Test suite
│   ├── test_transpiler.py  # Transpiler unit tests (no GPU)
│   ├── test_integration.py # GPU integration tests
│   └── test_distributions.py  # Distribution tests (both analytical and custom)
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
