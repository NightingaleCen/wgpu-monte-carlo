"""Integration tests for wgpu-montecarlo."""

import pytest
import numpy as np

# Skip all tests if the Rust extension isn't built
try:
    from wgpu_montecarlo import MonteCarloSimulator, monte_carlo, transpile_function

    HAS_EXTENSION = True
except ImportError:
    HAS_EXTENSION = False


@pytest.mark.skipif(not HAS_EXTENSION, reason="Rust extension not built")
class TestIntegration:
    """Integration tests requiring the Rust extension."""

    def test_monte_carlo_simulator_init(self):
        """Test that we can create a simulator."""
        sim = MonteCarloSimulator()
        assert sim is not None

    def test_simple_simulation(self):
        """Test running a simple simulation."""
        sim = MonteCarloSimulator()

        # Create initial data
        initial = np.zeros(100, dtype=np.float32)

        # Run simulation with default step function
        result = sim.run(initial, iterations=10, seed=42)

        assert result is not None
        assert len(result) == 100
        assert result.dtype == np.float32

    def test_simulation_with_custom_function(self):
        """Test simulation with custom WGSL function."""
        sim = MonteCarloSimulator()

        # Define custom step function in WGSL
        wgsl_func = """
fn step(x: f32, rng: f32) -> f32 {
    return x + (rng - 0.5) * 0.1;
}
"""

        initial = np.zeros(1000, dtype=np.float32)
        result = sim.run_with_wgsl(initial, wgsl_func, iterations=100, seed=42)

        assert result is not None
        assert len(result) == 1000


class TestTranspilerIntegration:
    """Test transpiler without needing GPU."""

    def test_transpile_simple_function(self):
        """Test transpiling a simple function."""

        def step(x, rng):
            return x + rng * 0.1

        result = transpile_function(step)
        assert "fn step" in result
        assert "x + (rng * 0.1)" in result or "(x + (rng * 0.1))" in result

    def test_transpile_math_function(self):
        """Test transpiling with math functions."""
        import math

        def step(x, rng):
            return math.sin(x) * rng

        result = transpile_function(step)
        assert "sin(x)" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
