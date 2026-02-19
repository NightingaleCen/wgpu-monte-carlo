"""Tests for Importance Sampling functionality."""

import math
import numpy as np
import pytest

try:
    from wgpu_montecarlo import (
        MonteCarloIntegrator,
        Distribution,
        integrate_importance_sampling,
    )

    HAS_EXTENSION = True
except ImportError:
    HAS_EXTENSION = False


@pytest.mark.skipif(not HAS_EXTENSION, reason="Rust extension not built")
class TestImportanceSamplingBasic:
    """Basic functionality tests for importance sampling."""

    def test_identical_distributions(self):
        """When p == q, IS should give same result as direct integration."""
        integrator = MonteCarloIntegrator()
        target = Distribution.normal(0.0, 1.0)
        proposal = Distribution.normal(0.0, 1.0)

        result = integrator.integrate_importance_sampling(
            [lambda x: x], target, proposal, n_samples=1_000_000, seed=42
        )
        assert abs(result.values[0]) < 0.01

    def test_shifted_proposal(self):
        """Test IS with shifted proposal distribution.

        p = N(0, 1), q = N(1, 1)
        E_p[X^2] should still be 1.0
        """
        integrator = MonteCarloIntegrator()
        target = Distribution.normal(0.0, 1.0)
        proposal = Distribution.normal(1.0, 1.0)

        result = integrator.integrate_importance_sampling(
            [lambda x: x * x], target, proposal, n_samples=5_000_000, seed=42
        )
        assert abs(result.values[0] - 1.0) < 0.05

    def test_different_variance_proposal(self):
        """Test IS with different variance proposal.

        p = N(0, 1), q = N(0, 2)
        E_p[X^2] should be 1.0
        """
        integrator = MonteCarloIntegrator()
        target = Distribution.normal(0.0, 1.0)
        proposal = Distribution.normal(0.0, 2.0)

        result = integrator.integrate_importance_sampling(
            [lambda x: x * x], target, proposal, n_samples=5_000_000, seed=42
        )
        assert abs(result.values[0] - 1.0) < 0.1


@pytest.mark.skipif(not HAS_EXTENSION, reason="Rust extension not built")
class TestImportanceSamplingMixedDistributions:
    """Tests with mixed distribution types."""

    def test_normal_target_uniform_proposal(self):
        """Test normal target with uniform proposal."""
        integrator = MonteCarloIntegrator()
        target = Distribution.normal(0.5, 0.2)
        proposal = Distribution.uniform(0.0, 1.0)

        result = integrator.integrate_importance_sampling(
            [lambda x: x], target, proposal, n_samples=5_000_000, seed=42
        )
        assert abs(result.values[0] - 0.5) < 0.1

    def test_uniform_target_uniform_proposal(self):
        """Test uniform target with wider uniform proposal.

        p = Uniform(0, 0.5), q = Uniform(0, 1)
        E_p[X] = 0.25
        """
        integrator = MonteCarloIntegrator()
        target = Distribution.uniform(0.0, 0.5)
        proposal = Distribution.uniform(0.0, 1.0)

        result = integrator.integrate_importance_sampling(
            [lambda x: x], target, proposal, n_samples=5_000_000, seed=42
        )
        assert abs(result.values[0] - 0.25) < 0.05

    def test_exponential_mixture(self):
        """Test exponential target with different exponential proposal.

        p = Exp(2), q = Exp(1)
        E_p[X] = 1/2 = 0.5
        """
        integrator = MonteCarloIntegrator()
        target = Distribution.exponential(2.0)
        proposal = Distribution.exponential(1.0)

        result = integrator.integrate_importance_sampling(
            [lambda x: x], target, proposal, n_samples=5_000_000, seed=42
        )
        assert abs(result.values[0] - 0.5) < 0.1


@pytest.mark.skipif(not HAS_EXTENSION, reason="Rust extension not built")
class TestImportanceSamplingMultipleFunctions:
    """Tests with multiple functions."""

    def test_multiple_functions_same_weight(self):
        """Multiple functions share the same weight calculation."""
        integrator = MonteCarloIntegrator()
        target = Distribution.normal(0.0, 1.0)
        proposal = Distribution.normal(0.5, 1.5)

        result = integrator.integrate_importance_sampling(
            [lambda x: x, lambda x: x * x, lambda x: x * x * x],
            target,
            proposal,
            n_samples=5_000_000,
            seed=42,
        )

        assert abs(result.values[0]) < 0.1
        assert abs(result.values[1] - 1.0) < 0.1
        assert abs(result.values[2]) < 0.1

    def test_mixed_callable_and_wgsl(self):
        """Test with mixed callable and WGSL string functions."""
        integrator = MonteCarloIntegrator()
        target = Distribution.normal(0.0, 1.0)
        proposal = Distribution.normal(0.5, 1.0)

        result = integrator.integrate_importance_sampling(
            [lambda x: x, "fn f(x: f32) -> f32 { return x * x; }"],
            target,
            proposal,
            n_samples=5_000_000,
            seed=42,
        )

        assert abs(result.values[0]) < 0.1
        assert abs(result.values[1] - 1.0) < 0.1


@pytest.mark.skipif(not HAS_EXTENSION, reason="Rust extension not built")
class TestImportanceSamplingCustomPDF:
    """Tests with custom (transpilable) PDF distributions."""

    def test_custom_target_transpilable_pdf(self):
        """Test custom distribution with transpilable PDF as target.

        p = truncated exponential: exp(-x) on [0, 5]
        q = Uniform(0, 5)
        E_p[X] = integral of x*exp(-x) from 0 to inf / normalizer
               ≈ 1 - 6*exp(-5) for truncated
        """
        integrator = MonteCarloIntegrator()

        def truncated_exp_pdf(x: float) -> float:
            if (x >= 0) and (x < 5):
                return math.exp(-x)
            return 0.0

        target = Distribution.from_pdf(truncated_exp_pdf, support=(0.0, 5.0))
        proposal = Distribution.uniform(0.0, 5.0)

        result = integrator.integrate_importance_sampling(
            [lambda x: x], target, proposal, n_samples=5_000_000, seed=42
        )

        expected = 1.0 - 6.0 * math.exp(-5.0)
        assert abs(result.values[0] - expected) < 0.15

    def test_custom_proposal_transpilable_pdf(self):
        """Test custom distribution with transpilable PDF as proposal."""
        integrator = MonteCarloIntegrator()

        def truncated_exp_pdf(x: float) -> float:
            if (x >= 0) and (x < 5):
                return math.exp(-x)
            return 0.0

        target = Distribution.uniform(0.0, 5.0)
        proposal = Distribution.from_pdf(truncated_exp_pdf, support=(0.0, 5.0))

        result = integrator.integrate_importance_sampling(
            [lambda x: x], target, proposal, n_samples=5_000_000, seed=42
        )

        expected = 2.5
        assert abs(result.values[0] - expected) < 0.3

    def test_custom_both_transpilable_pdf(self):
        """Test both target and proposal with custom transpilable PDFs."""
        integrator = MonteCarloIntegrator()

        def pdf_shifted_exp(x: float) -> float:
            if (x >= 1) and (x < 6):
                return math.exp(-(x - 1))
            return 0.0

        def pdf_uniform(x: float) -> float:
            if (x >= 1) and (x < 6):
                return 0.2
            return 0.0

        target = Distribution.from_pdf(pdf_shifted_exp, support=(1.0, 6.0))
        proposal = Distribution.from_pdf(pdf_uniform, support=(1.0, 6.0))

        result = integrator.integrate_importance_sampling(
            [lambda x: 1.0], target, proposal, n_samples=5_000_000, seed=42
        )

        assert abs(result.values[0] - 1.0) < 0.1

    def test_custom_pdf_with_math_functions(self):
        """Test custom PDF using various math functions."""
        integrator = MonteCarloIntegrator()

        def gaussian_pdf(x: float) -> float:
            z = (x - 2.0) / 0.5
            return math.exp(-0.5 * z * z) / (0.5 * math.sqrt(2 * math.pi))

        target = Distribution.from_pdf(gaussian_pdf, support=(0.0, 4.0))
        proposal = Distribution.uniform(0.0, 4.0)

        result = integrator.integrate_importance_sampling(
            [lambda x: x], target, proposal, n_samples=5_000_000, seed=42
        )

        assert abs(result.values[0] - 2.0) < 0.2

    def test_custom_pdf_truncated_normal_moments(self):
        """Test moments with truncated normal custom PDF."""
        integrator = MonteCarloIntegrator()

        def truncated_normal_pdf(x: float) -> float:
            if (x >= -2) and (x < 2):
                z = x / 1.0
                return math.exp(-0.5 * z * z)
            return 0.0

        target = Distribution.from_pdf(truncated_normal_pdf, support=(-2.0, 2.0))
        proposal = Distribution.uniform(-2.0, 2.0)

        result = integrator.integrate_importance_sampling(
            [lambda x: x, lambda x: x * x],
            target,
            proposal,
            n_samples=5_000_000,
            seed=42,
        )

        assert abs(result.values[0]) < 0.1
        assert result.values[1] > 0

    def test_custom_pdf_with_power_function(self):
        """Test custom PDF using power function (x^a)."""
        integrator = MonteCarloIntegrator()

        def power_law_pdf(x: float) -> float:
            if (x >= 1) and (x < 10):
                return 0.1 * math.pow(x, -0.5)
            return 0.0

        target = Distribution.from_pdf(power_law_pdf, support=(1.0, 10.0))
        proposal = Distribution.uniform(1.0, 10.0)

        result = integrator.integrate_importance_sampling(
            [lambda x: x], target, proposal, n_samples=5_000_000, seed=42
        )

        assert result.values[0] > 1.0
        assert result.values[0] < 10.0


@pytest.mark.skipif(not HAS_EXTENSION, reason="Rust extension not built")
class TestImportanceSamplingWithPDFTables:
    """Tests for IS with PDF lookup tables (Phase 2)."""

    def test_non_transpilable_target_uses_table(self):
        """Non-transpilable target PDF should use table lookup instead of failing."""
        integrator = MonteCarloIntegrator()

        def non_transpilable_pdf(x: float) -> float:
            return float(int(x) % 2) * 0.5 + 0.1

        target = Distribution.from_pdf(non_transpilable_pdf, support=(0.0, 10.0))
        proposal = Distribution.uniform(0.0, 10.0)

        result = integrator.integrate_importance_sampling(
            [lambda x: 1.0], target, proposal, n_samples=1_000_000, seed=42
        )
        assert len(result.values) == 1

    def test_non_transpilable_proposal_uses_table(self):
        """Non-transpilable proposal PDF should use table lookup instead of failing."""
        integrator = MonteCarloIntegrator()

        def non_transpilable_pdf(x: float) -> float:
            return float(int(x) % 2) * 0.5 + 0.1

        target = Distribution.normal(0.5, 0.2)
        proposal = Distribution.from_pdf(non_transpilable_pdf, support=(0.0, 10.0))

        result = integrator.integrate_importance_sampling(
            [lambda x: 1.0], target, proposal, n_samples=1_000_000, seed=42
        )
        assert len(result.values) == 1

    def test_both_non_transpilable_uses_tables(self):
        """Both PDFs non-transpilable should use table lookup."""
        integrator = MonteCarloIntegrator()

        def pdf1(x: float) -> float:
            return float(int(x) % 2) * 0.5 + 0.1

        def pdf2(x: float) -> float:
            return float(int(x * 2) % 3) * 0.3 + 0.1

        target = Distribution.from_pdf(pdf1, support=(0.0, 10.0))
        proposal = Distribution.from_pdf(pdf2, support=(0.0, 10.0))

        result = integrator.integrate_importance_sampling(
            [lambda x: 1.0], target, proposal, n_samples=1_000_000, seed=42
        )
        assert len(result.values) == 1

    def test_from_pdf_table_api(self):
        """Test Distribution.from_pdf_table() API."""
        integrator = MonteCarloIntegrator()
        x_grid = np.linspace(0, 10, 512)
        pdf_vals = np.exp(-x_grid)
        dist = Distribution.from_pdf_table(x_grid, pdf_vals)

        proposal = Distribution.uniform(0, 10)
        result = integrator.integrate_importance_sampling(
            [lambda x: 1.0], dist, proposal, n_samples=1_000_000, seed=42
        )
        assert len(result.values) == 1

    def test_arbitrary_table_size(self):
        """Test with non-standard table sizes."""
        integrator = MonteCarloIntegrator()

        for size in [100, 500, 1000]:

            def my_pdf(x: float) -> float:
                return float(int(x * 10) % 7) * 0.1 + 0.05

            target = Distribution.from_pdf(my_pdf, support=(0.0, 10.0), table_size=size)
            proposal = Distribution.uniform(0.0, 10.0)

            result = integrator.integrate_importance_sampling(
                [lambda x: 1.0], target, proposal, n_samples=100_000, seed=42
            )
            assert len(result.values) == 1


@pytest.mark.skipif(not HAS_EXTENSION, reason="Rust extension not built")
class TestImportanceSamplingErrors:
    """Error handling tests."""

    def test_empty_functions_error(self):
        """Empty function list should raise error."""
        integrator = MonteCarloIntegrator()
        target = Distribution.normal(0.0, 1.0)
        proposal = Distribution.normal(0.0, 1.0)

        with pytest.raises(ValueError):
            integrator.integrate_importance_sampling(
                [], target, proposal, n_samples=1000
            )

    def test_invalid_function_type_error(self):
        """Invalid function type should raise error."""
        integrator = MonteCarloIntegrator()
        target = Distribution.normal(0.0, 1.0)
        proposal = Distribution.normal(0.0, 1.0)

        with pytest.raises(TypeError):
            integrator.integrate_importance_sampling(
                [123], target, proposal, n_samples=1000
            )


@pytest.mark.skipif(not HAS_EXTENSION, reason="Rust extension not built")
class TestConvenienceFunction:
    """Tests for the convenience function."""

    def test_integrate_importance_sampling_function(self):
        """Test top-level convenience function."""
        target = Distribution.normal(0.0, 1.0)
        proposal = Distribution.normal(0.0, 1.0)

        result = integrate_importance_sampling(
            [lambda x: x, lambda x: x * x],
            target,
            proposal,
            n_samples=1_000_000,
            seed=42,
        )

        assert len(result.values) == 2
        assert abs(result.values[0]) < 0.1
        assert abs(result.values[1] - 1.0) < 0.1

    def test_with_target_threads_parameter(self):
        """Test target_threads parameter is passed correctly."""
        target = Distribution.normal(0.0, 1.0)
        proposal = Distribution.normal(0.0, 1.0)

        result = integrate_importance_sampling(
            [lambda x: x],
            target,
            proposal,
            n_samples=100_000,
            seed=42,
            target_threads=32768,
        )

        assert abs(result.values[0]) < 0.1

    def test_convenience_with_custom_pdf(self):
        """Test convenience function with custom PDF."""
        import math

        def my_pdf(x: float) -> float:
            if (x >= 0) and (x < 2):
                return 0.5 * x
            return 0.0

        target = Distribution.from_pdf(my_pdf, support=(0.0, 2.0))
        proposal = Distribution.uniform(0.0, 2.0)

        result = integrate_importance_sampling(
            [lambda x: x], target, proposal, n_samples=5_000_000, seed=42
        )

        expected = 4.0 / 3.0
        assert abs(result.values[0] - expected) < 0.2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
