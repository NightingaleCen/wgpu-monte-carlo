"""Tests for MCMC (Markov Chain Monte Carlo) integration."""

import math
import numpy as np
import pytest

from wgpu_montecarlo import (
    MonteCarloIntegrator,
    Distribution,
    IntegrationResult,
    integrate_mcmc,
    HAS_RUST_EXTENSION,
)


@pytest.fixture
def integrator():
    """Create a MonteCarloIntegrator instance."""
    if not HAS_RUST_EXTENSION:
        pytest.skip("Rust extension not built")
    return MonteCarloIntegrator()


class TestLogPdfConversion:
    """Test PDF to log-PDF conversion functionality."""

    def test_normal_log_pdf_correctness(self):
        """Verify log-PDF conversion for normal distribution."""
        dist = Distribution.normal(0.0, 1.0)
        x_table, log_pdf_table = dist.get_log_pdf_table()

        # Check that we got arrays
        assert len(x_table) > 0
        assert len(log_pdf_table) == len(x_table)

        # Verify at x=0: log_pdf = -0.5 * log(2*pi) - log(1) = -0.9189...
        expected_log_pdf_at_0 = -0.5 * math.log(2 * math.pi)
        idx = np.argmin(np.abs(x_table))
        assert np.isclose(log_pdf_table[idx], expected_log_pdf_at_0, rtol=0.01)

    def test_log_pdf_zero_handling(self):
        """Test log-PDF handles zero PDF values correctly."""

        def step_pdf(x):
            return 1.0 if 0.0 <= x < 1.0 else 0.0

        dist = Distribution.from_pdf(step_pdf, support=(-1.0, 2.0))
        x_table, log_pdf_table = dist.get_log_pdf_table()

        # Values outside support should have low log-PDF
        outside_mask = (x_table < 0.0) | (x_table >= 1.0)
        assert np.all(log_pdf_table[outside_mask] < -50.0)

    def test_log_pdf_numerical_stability(self):
        """Test log-PDF handles extreme values."""

        # Normal with very small std - PDF values can be very large
        dist = Distribution.normal(0.0, 0.01)
        x_table, log_pdf_table = dist.get_log_pdf_table()

        # Should not have inf or nan
        assert np.all(np.isfinite(log_pdf_table))

    def test_log_pdf_negative_pdf_handling(self):
        """Test log-PDF handles negative PDF values (which are invalid)."""

        def bad_pdf(x):
            # Return negative for some values (invalid PDF)
            return -0.5 if x < 0 else 0.5

        dist = Distribution.from_pdf(bad_pdf, support=(-1.0, 1.0))
        x_table, log_pdf_table = dist.get_log_pdf_table()

        # Should not crash and should handle gracefully
        assert len(log_pdf_table) == len(x_table)

    def test_log_pdf_custom_min_value(self):
        """Test log-PDF with custom minimum log value."""
        dist = Distribution.normal(0.0, 1.0)
        custom_min = -200.0
        x_table, log_pdf_table = dist.get_log_pdf_table(min_log_value=custom_min)

        # All log-PDF values should be >= custom_min
        assert np.all(log_pdf_table >= custom_min)


@pytest.mark.skipif(not HAS_RUST_EXTENSION, reason="Rust extension not built")
class TestMcmcBasic:
    """Basic MCMC functionality tests."""

    def test_mcmc_normal_mean(self, integrator):
        """Test MCMC estimates normal distribution mean correctly."""
        target = Distribution.normal(0.0, 1.0)
        proposal = Distribution.normal(0.0, 1.0)  # Same as target = 100% accept

        result = integrator.integrate_mcmc(
            [lambda x: x],
            target,
            proposal,
            n_steps=5000,
            n_chains=256,
            n_burnin=500,
            seed=42,
        )

        # E[X] for standard normal is 0
        assert abs(result.values[0]) < 0.15

    def test_mcmc_normal_second_moment(self, integrator):
        """Test MCMC estimates E[X^2] for standard normal."""
        target = Distribution.normal(0.0, 1.0)
        proposal = Distribution.normal(0.0, 1.5)

        result = integrator.integrate_mcmc(
            [lambda x: x**2],
            target,
            proposal,
            n_steps=10000,
            n_chains=512,
            n_burnin=1000,
            seed=42,
        )

        # E[X^2] for standard normal is 1
        assert abs(result.values[0] - 1.0) < 0.15

    def test_mcmc_multiple_functions(self, integrator):
        """Test MCMC with multiple functions simultaneously."""
        target = Distribution.normal(0.0, 1.0)
        proposal = Distribution.normal(0.0, 1.0)

        result = integrator.integrate_mcmc(
            [lambda x: x, lambda x: x**2, lambda x: x**3],
            target,
            proposal,
            n_steps=5000,
            n_chains=256,
            n_burnin=500,
            seed=42,
        )

        assert result.n_functions == 3
        assert len(result.values) == 3

        # E[X] ≈ 0, E[X^2] ≈ 1, E[X^3] ≈ 0
        assert abs(result.values[0]) < 0.15  # E[X]
        assert abs(result.values[1] - 1.0) < 0.15  # E[X^2]
        assert abs(result.values[2]) < 0.2  # E[X^3]

    def test_mcmc_returns_integration_result(self, integrator):
        """Test that MCMC returns proper IntegrationResult."""
        target = Distribution.normal(0.0, 1.0)
        proposal = Distribution.normal(0.0, 1.0)

        result = integrator.integrate_mcmc(
            [lambda x: x],
            target,
            proposal,
            n_steps=1000,
            n_chains=64,
            n_burnin=100,
        )

        assert isinstance(result, IntegrationResult)
        assert result.n_samples == 1000 * 64  # n_steps * n_chains


@pytest.mark.skipif(not HAS_RUST_EXTENSION, reason="Rust extension not built")
class TestBurnIn:
    """Test burn-in phase effectiveness."""

    def test_zero_burnin_allowed(self, integrator):
        """Test that n_burnin=0 is allowed."""
        target = Distribution.normal(0.0, 1.0)
        proposal = Distribution.normal(0.0, 1.0)

        # Should not raise
        result = integrator.integrate_mcmc(
            [lambda x: x],
            target,
            proposal,
            n_steps=1000,
            n_chains=64,
            n_burnin=0,
            seed=42,
        )

        assert result.n_functions == 1

    def test_burnin_doesnt_affect_sample_count(self, integrator):
        """Test that burn-in doesn't count toward total samples."""
        target = Distribution.normal(0.0, 1.0)
        proposal = Distribution.normal(0.0, 1.0)

        result_no_burnin = integrator.integrate_mcmc(
            [lambda x: x],
            target,
            proposal,
            n_steps=1000,
            n_chains=64,
            n_burnin=0,
            seed=42,
        )

        result_with_burnin = integrator.integrate_mcmc(
            [lambda x: x],
            target,
            proposal,
            n_steps=1000,
            n_chains=64,
            n_burnin=1000,
            seed=42,
        )

        # Both should have same number of samples
        assert result_no_burnin.n_samples == result_with_burnin.n_samples


@pytest.mark.skipif(not HAS_RUST_EXTENSION, reason="Rust extension not built")
class TestProposalDistribution:
    """Test various proposal distribution scenarios."""

    def test_same_proposal_as_target(self, integrator):
        """When proposal == target, should match direct integration."""
        target = Distribution.normal(0.0, 1.0)
        proposal = Distribution.normal(0.0, 1.0)

        result = integrator.integrate_mcmc(
            [lambda x: x, lambda x: x**2],
            target,
            proposal,
            n_steps=10000,
            n_chains=256,
            n_burnin=500,
            seed=42,
        )

        # Should match expected values well (high acceptance rate)
        assert abs(result.values[0]) < 0.1
        assert abs(result.values[1] - 1.0) < 0.1

    def test_wider_proposal(self, integrator):
        """Test with wider proposal distribution."""
        target = Distribution.normal(0.0, 1.0)
        proposal = Distribution.normal(0.0, 2.0)  # Wider

        result = integrator.integrate_mcmc(
            [lambda x: x**2],
            target,
            proposal,
            n_steps=10000,
            n_chains=256,
            n_burnin=1000,
            seed=42,
        )

        # Should still estimate E[X^2] ≈ 1
        assert abs(result.values[0] - 1.0) < 0.2

    def test_uniform_proposal(self, integrator):
        """Test with uniform proposal distribution."""
        target = Distribution.normal(0.0, 1.0)
        proposal = Distribution.uniform(-5.0, 5.0)

        result = integrator.integrate_mcmc(
            [lambda x: x**2],
            target,
            proposal,
            n_steps=10000,
            n_chains=256,
            n_burnin=1000,
            seed=42,
        )

        # Should still converge, though with lower efficiency
        assert 0.5 < result.values[0] < 1.5


@pytest.mark.skipif(not HAS_RUST_EXTENSION, reason="Rust extension not built")
class TestMultiChain:
    """Test multi-chain parallel MCMC behavior."""

    def test_single_chain(self, integrator):
        """Test with single chain (n_chains=1)."""
        target = Distribution.normal(0.0, 1.0)
        proposal = Distribution.normal(0.0, 1.0)

        result = integrator.integrate_mcmc(
            [lambda x: x**2],
            target,
            proposal,
            n_steps=10000,
            n_chains=1,
            n_burnin=1000,
            seed=42,
        )

        # Single chain should still work
        assert abs(result.values[0] - 1.0) < 0.3

    def test_many_chains(self, integrator):
        """Test with many parallel chains."""
        target = Distribution.normal(0.0, 1.0)
        proposal = Distribution.normal(0.0, 1.0)

        result = integrator.integrate_mcmc(
            [lambda x: x**2],
            target,
            proposal,
            n_steps=1000,
            n_chains=4096,
            n_burnin=200,
            seed=42,
        )

        # Many chains should average well
        assert abs(result.values[0] - 1.0) < 0.1

    def test_reproducibility_with_seed(self, integrator):
        """Test that same seed produces same results."""
        target = Distribution.normal(0.0, 1.0)
        proposal = Distribution.normal(0.0, 1.0)

        result1 = integrator.integrate_mcmc(
            [lambda x: x**2],
            target,
            proposal,
            n_steps=1000,
            n_chains=64,
            n_burnin=100,
            seed=12345,
        )

        result2 = integrator.integrate_mcmc(
            [lambda x: x**2],
            target,
            proposal,
            n_steps=1000,
            n_chains=64,
            n_burnin=100,
            seed=12345,
        )

        np.testing.assert_array_almost_equal(result1.values, result2.values)


@pytest.mark.skipif(not HAS_RUST_EXTENSION, reason="Rust extension not built")
class TestCustomDistribution:
    """Test MCMC with custom (table-based) distributions."""

    def test_custom_target_distribution(self, integrator):
        """Test MCMC with custom target distribution."""

        def my_pdf(x):
            return 0.5 * (math.exp(-0.5 * (x - 2) ** 2) + math.exp(-0.5 * (x + 2) ** 2))

        target = Distribution.from_pdf(my_pdf, support=(-10.0, 10.0))
        proposal = Distribution.normal(0.0, 2.0)

        # For bimodal, estimate mean (should be 0)
        result = integrator.integrate_mcmc(
            [lambda x: x],
            target,
            proposal,
            n_steps=10000,
            n_chains=256,
            n_burnin=1000,
            seed=42,
        )

        # Mean of symmetric bimodal is 0
        assert abs(result.values[0]) < 0.5

    def test_beta_distribution(self, integrator):
        """Test MCMC with Beta distribution."""
        alpha, beta_param = 2.0, 5.0
        target = Distribution.beta(alpha, beta_param)
        proposal = Distribution.uniform(0.0, 1.0)

        result = integrator.integrate_mcmc(
            [lambda x: x],
            target,
            proposal,
            n_steps=10000,
            n_chains=256,
            n_burnin=1000,
            seed=42,
        )

        # E[X] for Beta(alpha, beta) = alpha / (alpha + beta)
        expected_mean = alpha / (alpha + beta_param)
        assert abs(result.values[0] - expected_mean) < 0.1


@pytest.mark.skipif(not HAS_RUST_EXTENSION, reason="Rust extension not built")
class TestErrorHandling:
    """Test error handling for invalid inputs."""

    def test_empty_function_list(self, integrator):
        """Test that empty function list raises ValueError."""
        target = Distribution.normal(0.0, 1.0)
        proposal = Distribution.normal(0.0, 1.0)

        with pytest.raises(ValueError, match="At least one function"):
            integrator.integrate_mcmc([], target, proposal)

    def test_zero_n_steps(self, integrator):
        """Test that n_steps=0 raises ValueError."""
        target = Distribution.normal(0.0, 1.0)
        proposal = Distribution.normal(0.0, 1.0)

        with pytest.raises(ValueError, match="n_steps must be positive"):
            integrator.integrate_mcmc(
                [lambda x: x], target, proposal, n_steps=0, n_chains=64, n_burnin=100
            )

    def test_zero_n_chains(self, integrator):
        """Test that n_chains=0 raises ValueError."""
        target = Distribution.normal(0.0, 1.0)
        proposal = Distribution.normal(0.0, 1.0)

        with pytest.raises(ValueError, match="n_chains must be positive"):
            integrator.integrate_mcmc(
                [lambda x: x], target, proposal, n_steps=100, n_chains=0, n_burnin=100
            )

    def test_negative_burnin(self, integrator):
        """Test that negative burn-in raises ValueError."""
        target = Distribution.normal(0.0, 1.0)
        proposal = Distribution.normal(0.0, 1.0)

        with pytest.raises(ValueError, match="n_burnin must be non-negative"):
            integrator.integrate_mcmc(
                [lambda x: x], target, proposal, n_steps=100, n_chains=64, n_burnin=-1
            )

    def test_invalid_function_type(self, integrator):
        """Test that invalid function type raises TypeError."""
        target = Distribution.normal(0.0, 1.0)
        proposal = Distribution.normal(0.0, 1.0)

        with pytest.raises(TypeError):
            integrator.integrate_mcmc([123], target, proposal)  # Not callable or string


@pytest.mark.skipif(not HAS_RUST_EXTENSION, reason="Rust extension not built")
class TestConvenienceFunction:
    """Test the integrate_mcmc convenience function."""

    def test_convenience_function_basic(self):
        """Test basic usage of integrate_mcmc convenience function."""
        target = Distribution.normal(0.0, 1.0)
        proposal = Distribution.normal(0.0, 1.0)

        result = integrate_mcmc(
            [lambda x: x**2],
            target,
            proposal,
            n_steps=5000,
            n_chains=256,
            n_burnin=500,
        )

        assert isinstance(result, IntegrationResult)
        assert abs(result.values[0] - 1.0) < 0.15
