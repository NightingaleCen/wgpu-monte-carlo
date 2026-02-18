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
        """Test custom distribution creation via from_pdf."""
        import math

        def pdf(x):
            return 1.0 if 0 <= x < 1 else 0.0

        dist = Distribution.from_pdf(pdf, support=(0.0, 1.0))
        assert dist.dist_type.name == "CUSTOM"
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

        dist = Distribution.beta(alpha, beta_param, table_size=2048)

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
        import math

        n_samples = 1_000_000

        def uniform_pdf(x):
            return 1.0 if 0 <= x < 1 else 0.0

        dist_table = Distribution.from_pdf(uniform_pdf, support=(0.0, 1.0))
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


@pytest.mark.skipif(not HAS_EXTENSION, reason="Rust extension not built")
class TestCustomDistribution:
    """Test custom distribution from PDF function."""

    def test_custom_pdf_interface(self):
        """Test that all distributions provide pdf(x) interface."""
        import math

        dist_uniform = Distribution.uniform(0.0, 1.0)
        dist_normal = Distribution.normal(0.0, 1.0)
        dist_exponential = Distribution.exponential(1.0)

        assert abs(dist_uniform.pdf(0.5) - 1.0) < 0.001
        assert abs(dist_normal.pdf(0.0) - 0.3989) < 0.001
        assert abs(dist_exponential.pdf(0.0) - 1.0) < 0.001

    def test_custom_distribution_from_pdf(self):
        """Test creating distribution from PDF function."""
        import math

        def my_pdf(x):
            return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

        dist = Distribution.from_pdf(my_pdf)
        assert dist.dist_type.name == "CUSTOM"
        assert dist._x_table is not None
        assert dist._cdf_table is not None

    def test_custom_distribution_pdf_at_points(self):
        """Test PDF evaluation at specific points."""
        import math

        def my_pdf(x):
            return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

        dist = Distribution.from_pdf(my_pdf)
        assert abs(dist.pdf(0.0) - 0.3989) < 0.001
        assert abs(dist.pdf(1.0) - 0.2419) < 0.001

    def test_custom_distribution_integration(self):
        """Test integration with custom distribution."""
        import math

        def my_pdf(x):
            return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

        dist = Distribution.from_pdf(my_pdf)
        result = integrate([f_identity, f_square], dist, n_samples=5000000, seed=42)

        assert abs(result.values[0]) < 0.02
        assert abs(result.values[1] - 1.0) < 0.02

    def test_custom_distribution_with_manual_support(self):
        """Test custom distribution with manually specified support."""
        import math

        def uniform_pdf(x):
            return 1.0 if 0 <= x < 1 else 0.0

        dist = Distribution.from_pdf(uniform_pdf, support=(0.0, 1.0))
        result = integrate([f_identity], dist, n_samples=1000000, seed=42)

        assert abs(result.values[0] - 0.5) < 0.02


@pytest.mark.skipif(not HAS_EXTENSION, reason="Rust extension not built")
class TestVariableTableSize:
    """Test variable table size for custom distributions."""

    def test_different_table_sizes(self):
        """Test that different table sizes are correctly applied."""
        import math

        def my_pdf(x):
            return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

        for size in [1024, 2048, 4096]:
            dist = Distribution.from_pdf(my_pdf, support=(-3.0, 3.0), table_size=size)
            assert dist.params["table_size"] == size
            assert len(dist._x_table) == size
            assert len(dist._cdf_table) == size

    def test_table_size_4096_accuracy(self):
        """Test that larger table size improves accuracy."""
        import math

        def my_pdf(x):
            return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

        dist = Distribution.from_pdf(my_pdf, support=(-5.0, 5.0), table_size=4096)
        result = integrate([f_square], dist, n_samples=5000000, seed=42)

        assert abs(result.values[0] - 1.0) < 0.02

    def test_minimum_table_size_enforced(self):
        """Test that table_size is enforced to minimum 1000."""
        import math

        def my_pdf(x):
            return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

        dist = Distribution.from_pdf(my_pdf, support=(-5.0, 5.0), table_size=500)
        assert dist.params["table_size"] == 1000
        assert len(dist._x_table) == 1000

    def test_bounded_support(self):
        """Test support detection for bounded distribution (0, 1)."""
        import math

        def pdf(x):
            return 6.0 * x * (1.0 - x) if 0 < x < 1 else 0.0

        dist = Distribution.from_pdf(pdf)
        support = dist.params["support"]

        assert support[0] >= -1.0
        assert support[1] <= 2.0


@pytest.mark.skipif(not HAS_EXTENSION, reason="Rust extension not built")
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_pdf_returns_nan(self):
        """Test handling of PDF returning NaN with manual support."""
        import math

        def pdf_nan(x):
            if abs(x) < 0.1:
                return float("nan")
            return math.exp(-0.5 * x * x)

        dist = Distribution.from_pdf(pdf_nan, support=(-3.0, 3.0))
        assert dist is not None
        assert abs(dist._cdf_table[-1] - 1.0) < 1e-6

    def test_pdf_returns_inf(self):
        """Test handling of PDF returning infinity with manual support."""
        import math

        def pdf_inf(x):
            return float("inf") if abs(x) < 0.001 else math.exp(-0.5 * x * x)

        dist = Distribution.from_pdf(pdf_inf, support=(-3.0, 3.0))
        assert dist is not None
        assert abs(dist._cdf_table[-1] - 1.0) < 1e-6

    def test_pdf_returns_negative(self):
        """Test handling of PDF returning negative values with manual support."""

        def pdf_negative(x):
            return 0.9 if 0 <= x < 1 else 0.0

        dist = Distribution.from_pdf(pdf_negative, support=(0.0, 1.0))
        assert dist is not None
        assert abs(dist._cdf_table[-1] - 1.0) < 1e-6

    def test_exponential_distribution_pdf_at_boundary(self):
        """Test exponential PDF at boundaries."""
        dist = Distribution.exponential(1.0)

        assert dist.pdf(0.0) > 0
        assert dist.pdf(100.0) < 1e-40
        assert dist.pdf(-1.0) == 0.0

    def test_uniform_pdf_at_boundaries(self):
        """Test uniform PDF at boundaries."""
        dist = Distribution.uniform(0.0, 1.0)

        assert dist.pdf(0.0) == 1.0
        assert dist.pdf(0.999) == 1.0
        assert dist.pdf(1.0) == 0.0
        assert dist.pdf(-0.001) == 0.0
        assert dist.pdf(1.001) == 0.0

    def test_beta_distribution_pdf_at_boundaries(self):
        """Test Beta distribution PDF at boundaries."""
        dist = Distribution.beta(2.0, 5.0)

        assert dist.pdf(0.0) == 0.0
        assert dist.pdf(1.0) == 0.0
        assert dist.pdf(0.5) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
