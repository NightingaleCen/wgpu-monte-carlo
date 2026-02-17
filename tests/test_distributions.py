"""Tests for probability distributions."""

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


def f_identity(x):
    return x


def f_square(x):
    return x * x


def f_cube(x):
    return x * x * x


def f_quad(x):
    return x * x * x * x


@pytest.mark.skipif(not HAS_EXTENSION, reason="Rust extension not built")
class TestDistributionCreation:
    """Test distribution creation."""

    def test_uniform_creation(self):
        """Test uniform distribution creation."""
        dist = Distribution.uniform(min=0.0, max=1.0)
        assert dist.dist_type.name == "UNIFORM"
        assert dist.params["min"] == 0.0
        assert dist.params["max"] == 1.0

    def test_normal_creation(self):
        """Test normal distribution creation."""
        dist = Distribution.normal(mean=0.0, std=1.0)
        assert dist.dist_type.name == "NORMAL"
        assert dist.params["mean"] == 0.0
        assert dist.params["std"] == 1.0

    def test_exponential_creation(self):
        """Test exponential distribution creation."""
        dist = Distribution.exponential(lambda_param=2.0)
        assert dist.dist_type.name == "EXPONENTIAL"
        assert dist.params["lambda"] == 2.0

    def test_from_table_creation(self):
        """Test table-based distribution creation."""
        table = np.linspace(0, 1, 2048, endpoint=False).astype(np.float32)
        dist = Distribution.from_table(table)
        assert dist.dist_type.name == "TABLE"
        assert dist.params["table_size"] == 2048


@pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
@pytest.mark.skipif(not HAS_EXTENSION, reason="Rust extension not built")
class TestBetaDistribution:
    """Test Beta distribution using lookup table method."""

    def test_beta_2_5(self):
        """Test Beta(2.0, 5.0) distribution."""
        alpha = 2.0
        beta_param = 5.0
        n_samples = 10_000_000

        table_size = 2048
        probabilities = np.linspace(0, 1, table_size, endpoint=False)
        probabilities = np.clip(probabilities, 1e-7, 1 - 1e-7)
        lookup_table = beta_dist.ppf(probabilities, alpha, beta_param).astype(
            np.float32
        )

        dist = Distribution.from_table(lookup_table)

        functions = [f_identity, f_square, f_cube]

        integrator = MonteCarloIntegrator()
        result = integrator.integrate(functions, dist, n_samples=n_samples, seed=42)

        expected_mean = alpha / (alpha + beta_param)
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

        tolerance = 0.01

        assert abs(result.values[0] - expected_mean) < tolerance
        assert abs(result.values[1] - expected_mean_sq) < tolerance
        assert abs(result.values[2] - expected_mean_cubed) < tolerance

    def test_beta_convenience_method(self):
        """Test Beta distribution convenience method."""
        alpha = 3.0
        beta_param = 2.0
        n_samples = 5_000_000

        dist = Distribution.beta(alpha, beta_param, table_size=2048)

        functions = [f_identity, f_square]

        result = integrate(functions, dist, n_samples=n_samples, seed=123)

        expected_mean = alpha / (alpha + beta_param)
        expected_mean_sq = (
            alpha * (alpha + 1) / ((alpha + beta_param) * (alpha + beta_param + 1))
        )
        expected_variance = expected_mean_sq - expected_mean**2

        assert abs(result.values[0] - expected_mean) < 0.02
        computed_variance = result.values[1] - result.values[0] ** 2
        assert abs(computed_variance - expected_variance) < 0.02

    def test_table_vs_direct(self):
        """Compare table-based sampling with direct uniform sampling."""
        n_samples = 1_000_000

        table = np.linspace(0, 1, 2048, endpoint=False).astype(np.float32)
        dist_table = Distribution.from_table(table)
        dist_direct = Distribution.uniform(0.0, 1.0)

        functions = [f_identity, f_square]

        result_table = integrate(functions, dist_table, n_samples=n_samples, seed=42)
        result_direct = integrate(functions, dist_direct, n_samples=n_samples, seed=42)

        expected_mean = 0.5
        expected_mean_sq = 1.0 / 3.0

        assert abs(result_table.values[0] - expected_mean) < 0.01
        assert abs(result_table.values[1] - expected_mean_sq) < 0.01
        assert abs(result_direct.values[0] - expected_mean) < 0.01
        assert abs(result_direct.values[1] - expected_mean_sq) < 0.01


@pytest.mark.skipif(not HAS_EXTENSION, reason="Rust extension not built")
class TestUniformDistribution:
    """Test uniform distribution."""

    def test_uniform_mean(self):
        """Test E[X] for uniform distribution."""
        dist = Distribution.uniform(min=0.0, max=1.0)
        result = integrate([f_identity], dist, n_samples=10000000, seed=42)

        assert abs(result.values[0] - 0.5) < 0.01

    def test_uniform_variance(self):
        """Test Var[X] for uniform distribution."""
        dist = Distribution.uniform(min=0.0, max=1.0)
        result = integrate([f_identity, f_square], dist, n_samples=10000000, seed=42)

        mean = result.values[0]
        e_x2 = result.values[1]
        variance = e_x2 - mean**2

        expected_var = 1.0 / 12.0
        assert abs(variance - expected_var) < 0.01


@pytest.mark.skipif(not HAS_EXTENSION, reason="Rust extension not built")
class TestNormalDistribution:
    """Test normal distribution."""

    def test_normal_mean(self):
        """Test E[X] for standard normal."""
        dist = Distribution.normal(mean=0.0, std=1.0)
        result = integrate([f_identity], dist, n_samples=10000000, seed=42)

        assert abs(result.values[0]) < 0.01

    def test_normal_variance(self):
        """Test Var[X] for standard normal."""
        dist = Distribution.normal(mean=0.0, std=1.0)
        result = integrate([f_identity, f_square], dist, n_samples=10000000, seed=42)

        mean = result.values[0]
        e_x2 = result.values[1]
        variance = e_x2 - mean**2

        assert abs(variance - 1.0) < 0.01

    def test_normal_higher_moments(self):
        """Test E[X⁴] for standard normal."""
        dist = Distribution.normal(mean=0.0, std=1.0)
        result = integrate([f_quad], dist, n_samples=10000000, seed=42)

        assert abs(result.values[0] - 3.0) < 0.01

    def test_normal_with_mean_and_std(self):
        """Test normal distribution with non-zero mean."""
        mean = 5.0
        std = 2.0
        dist = Distribution.normal(mean=mean, std=std)

        result = integrate([f_identity, f_square], dist, n_samples=10000000, seed=42)

        computed_mean = result.values[0]
        e_x2 = result.values[1]
        computed_var = e_x2 - computed_mean**2

        assert abs(computed_mean - mean) < 0.01
        assert abs(computed_var - std * std) < 0.01


@pytest.mark.skipif(not HAS_EXTENSION, reason="Rust extension not built")
class TestExponentialDistribution:
    """Test exponential distribution."""

    def test_exponential_mean(self):
        """Test E[X] for exponential distribution."""
        lam = 2.0
        dist = Distribution.exponential(lambda_param=lam)
        result = integrate([f_identity], dist, n_samples=10000000, seed=42)

        expected_mean = 1.0 / lam
        assert abs(result.values[0] - expected_mean) < 0.01

    def test_exponential_variance(self):
        """Test Var[X] for exponential distribution."""
        lam = 2.0
        dist = Distribution.exponential(lambda_param=lam)

        result = integrate([f_identity, f_square], dist, n_samples=10000000, seed=42)

        mean = result.values[0]
        e_x2 = result.values[1]
        variance = e_x2 - mean**2

        expected_var = 1.0 / (lam * lam)
        assert abs(variance - expected_var) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
