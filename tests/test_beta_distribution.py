"""Test Beta distribution using lookup table method.

This test validates the table-based sampling architecture for
non-standard distributions that are difficult to sample directly in WGSL.
"""

import numpy as np
import pytest

try:
    from scipy.stats import beta as beta_dist

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from wgpu_montecarlo import MonteCarloIntegrator, Distribution, integrate

    HAS_EXTENSION = True
except ImportError:
    HAS_EXTENSION = False


@pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
@pytest.mark.skipif(not HAS_EXTENSION, reason="Rust extension not built")
class TestBetaDistribution:
    """Test Beta distribution using lookup table method."""

    def test_beta_2_5_lookup_table(self):
        """Test Beta(2.0, 5.0) distribution using inverse CDF lookup table.

        This validates that the table-based sampling mechanism correctly
        handles complex distributions that would be inefficient to sample
        directly in WGSL (due to thread divergence in rejection sampling).

        The test computes E[X], E[X²], and E[X³] and compares against
        analytical values for Beta(2, 5).
        """

        # Parameters
        alpha = 2.0
        beta_param = 5.0
        n_samples = 10_000_000  # 10 million samples for high precision

        # Precompute inverse CDF lookup table using scipy
        # TODO: Currently using 2048 elements. Consider 4096 for higher precision.
        table_size = 2048
        probabilities = np.linspace(0, 1, table_size, endpoint=False)

        # Clip to avoid 0 and 1 (undefined for Beta inverse CDF)
        probabilities = np.clip(probabilities, 1e-7, 1 - 1e-7)
        lookup_table = beta_dist.ppf(probabilities, alpha, beta_param).astype(
            np.float32
        )

        # Create distribution from lookup table
        dist = Distribution.from_table(lookup_table)

        # Define functions to integrate
        # E[X] for Beta(2, 5) = alpha / (alpha + beta) = 2/7 ≈ 0.2857
        # E[X²] = alpha * (alpha + 1) / ((alpha + beta) * (alpha + beta + 1))
        #       = 2 * 3 / (7 * 8) = 6/56 = 0.1071
        functions = [
            lambda x: x,  # First moment: E[X]
            lambda x: x**2,  # Second moment: E[X²]
            lambda x: x**3,  # Third moment: E[X³]
        ]

        # Run integration
        integrator = MonteCarloIntegrator()
        result = integrator.integrate(functions, dist, n_samples=n_samples, seed=42)

        # Analytical expected values
        expected_mean = alpha / (alpha + beta_param)  # 2/7
        expected_mean_sq = (
            alpha * (alpha + 1) / ((alpha + beta_param) * (alpha + beta_param + 1))
        )
        expected_mean_cubed = (
            alpha
            * (alpha + 1)
            * (alpha + 2)
            / (
                (alpha + beta_param)
                * (alpha + beta_param + 1)
                * (alpha + beta_param + 2)
            )
        )

        # Check results with 1% tolerance for Monte Carlo variance
        # With 10M samples, error should be approximately 1/sqrt(N) ≈ 0.0003
        tolerance = 0.01  # 1% tolerance (very conservative)

        assert abs(result.values[0] - expected_mean) < tolerance, (
            f"E[X] mismatch: got {result.values[0]:.6f}, expected {expected_mean:.6f}"
        )

        assert abs(result.values[1] - expected_mean_sq) < tolerance, (
            f"E[X²] mismatch: got {result.values[1]:.6f}, expected {expected_mean_sq:.6f}"
        )

        assert abs(result.values[2] - expected_mean_cubed) < tolerance, (
            f"E[X³] mismatch: got {result.values[2]:.6f}, expected {expected_mean_cubed:.6f}"
        )

    def test_beta_convenience_method(self):
        """Test Beta distribution using Distribution.beta() convenience method.

        This test validates the scipy-based auto-generation of lookup tables.
        """

        alpha = 3.0
        beta_param = 2.0
        n_samples = 5_000_000

        # Use convenience method (requires scipy)
        dist = Distribution.beta(alpha, beta_param, table_size=2048)

        # Compute mean and variance
        functions = [
            lambda x: x,
            lambda x: x**2,
        ]

        result = integrate(functions, dist, n_samples=n_samples, seed=123)

        # Analytical values for Beta(3, 2)
        expected_mean = alpha / (alpha + beta_param)  # 3/5 = 0.6
        expected_mean_sq = (
            alpha * (alpha + 1) / ((alpha + beta_param) * (alpha + beta_param + 1))
        )
        # = 3 * 4 / (5 * 6) = 12/30 = 0.4
        expected_variance = expected_mean_sq - expected_mean**2

        # Check within 2% tolerance
        assert abs(result.values[0] - expected_mean) < 0.02, (
            f"Mean mismatch: got {result.values[0]:.6f}, expected {expected_mean:.6f}"
        )

        computed_variance = result.values[1] - result.values[0] ** 2
        assert abs(computed_variance - expected_variance) < 0.02, (
            f"Variance mismatch: got {computed_variance:.6f}, expected {expected_variance:.6f}"
        )

    def test_table_vs_direct_sampling(self):
        """Compare table-based sampling accuracy against known values.

        For uniform distribution, we can compare table-based sampling
        against direct uniform sampling to validate the table mechanism.
        """

        n_samples = 1_000_000

        # Create uniform distribution using table method
        # Table with evenly spaced values from 0 to 1
        table = np.linspace(0, 1, 2048, endpoint=False).astype(np.float32)
        dist_table = Distribution.from_table(table)

        # Also test with direct uniform
        dist_direct = Distribution.uniform(0.0, 1.0)

        # Compute E[X] and E[X²] for both
        # Use named functions instead of lambdas to avoid inspect.getsource issues
        def f1(x):
            return x

        def f2(x):
            return x * x

        functions = [f1, f2]

        result_table = integrate(functions, dist_table, n_samples=n_samples, seed=42)
        result_direct = integrate(functions, dist_direct, n_samples=n_samples, seed=42)

        # For uniform [0, 1): E[X] = 0.5, E[X²] = 1/3 ≈ 0.333
        expected_mean = 0.5
        expected_mean_sq = 1.0 / 3.0

        # Check table method
        assert abs(result_table.values[0] - expected_mean) < 0.01, (
            f"Table method mean error: {result_table.values[0]:.6f}"
        )
        assert abs(result_table.values[1] - expected_mean_sq) < 0.01, (
            f"Table method E[X²] error: {result_table.values[1]:.6f}"
        )

        # Check direct method
        assert abs(result_direct.values[0] - expected_mean) < 0.01, (
            f"Direct method mean error: {result_direct.values[0]:.6f}"
        )
        assert abs(result_direct.values[1] - expected_mean_sq) < 0.01, (
            f"Direct method E[X²] error: {result_direct.values[1]:.6f}"
        )

        # Results should be similar (within 1% of each other)
        assert abs(result_table.values[0] - result_direct.values[0]) < 0.01, (
            "Table and direct methods disagree on mean"
        )
        assert abs(result_table.values[1] - result_direct.values[1]) < 0.01, (
            "Table and direct methods disagree on E[X²]"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
