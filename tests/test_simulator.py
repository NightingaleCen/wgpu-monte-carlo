"""Tests for Monte Carlo Simulator."""

import pytest
import numpy as np

try:
    from wgpu_montecarlo import MonteCarloSimulator, monte_carlo

    HAS_EXTENSION = True
except ImportError:
    HAS_EXTENSION = False


@pytest.mark.skipif(not HAS_EXTENSION, reason="Rust extension not built")
class TestMonteCarloSimulator:
    """Test cases for MonteCarloSimulator."""

    def test_init(self):
        """Test simulator initialization."""
        sim = MonteCarloSimulator()
        assert sim is not None

    def test_simple_simulation(self):
        """Test running a simple simulation."""
        sim = MonteCarloSimulator()

        initial = np.zeros(100, dtype=np.float32)
        result = sim.run(initial, iterations=10, seed=42)

        assert result is not None
        assert len(result) == 100
        assert result.dtype == np.float32

    def test_simulation_with_python_function(self):
        """Test simulation with Python function."""
        sim = MonteCarloSimulator()

        def step(x, rng):
            return x + (rng - 0.5) * 0.1

        initial = np.zeros(1000, dtype=np.float32)
        result = sim.run(initial, step, iterations=100, seed=42)

        assert result is not None
        assert len(result) == 1000

    def test_simulation_with_wgsl_string(self):
        """Test simulation with WGSL string."""
        sim = MonteCarloSimulator()

        wgsl_func = """
fn step(x: f32, rng: f32) -> f32 {
    return x + (rng - 0.5) * 0.1;
}
"""

        initial = np.zeros(1000, dtype=np.float32)
        result = sim.run(initial, wgsl_func, iterations=100, seed=42)

        assert result is not None
        assert len(result) == 1000

    def test_set_step_function(self):
        """Test set_step_function method."""
        sim = MonteCarloSimulator()

        def step(x, rng):
            return x * 0.99 + (rng - 0.5) * 0.1

        sim.set_step_function(step)

        initial = np.zeros(100, dtype=np.float32)
        result = sim.run(initial, iterations=50, seed=42)

        assert result is not None
        assert len(result) == 100

    def test_run_with_wgsl(self):
        """Test run_with_wgsl method."""
        sim = MonteCarloSimulator()

        wgsl_func = "fn step(x: f32, rng: f32) -> f32 { return x + (rng - 0.5) * 0.1; }"

        initial = np.zeros(500, dtype=np.float32)
        result = sim.run_with_wgsl(initial, wgsl_func, iterations=50, seed=42)

        assert result is not None
        assert len(result) == 500

    def test_simulation_with_different_seeds(self):
        """Test that different seeds produce different results."""
        sim1 = MonteCarloSimulator()
        sim2 = MonteCarloSimulator()

        initial = np.zeros(100, dtype=np.float32)

        result1 = sim1.run(initial.copy(), iterations=100, seed=42)
        result2 = sim2.run(initial.copy(), iterations=100, seed=99)

        assert not np.allclose(result1, result2)

    def test_simulation_iterations(self):
        """Test that more iterations increase variance."""
        sim = MonteCarloSimulator()

        initial = np.zeros(1000, dtype=np.float32)

        result_few = sim.run(initial.copy(), iterations=10, seed=42)
        result_many = sim.run(initial.copy(), iterations=1000, seed=42)

        std_few = np.std(result_few)
        std_many = np.std(result_many)

        assert std_many > std_few


@pytest.mark.skipif(not HAS_EXTENSION, reason="Rust extension not built")
class TestMonteCarloConvenience:
    """Test convenience functions."""

    def test_monte_carlo_function(self):
        """Test monte_carlo convenience function."""

        def step(x, rng):
            return x + (rng - 0.5) * 0.1

        initial = np.zeros(100, dtype=np.float32)
        result = monte_carlo(initial, step, iterations=50, seed=42)

        assert result is not None
        assert len(result) == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
