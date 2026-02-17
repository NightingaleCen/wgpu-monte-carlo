"""Tests for Monte Carlo Integrator."""

import math
import pytest
import numpy as np

try:
    from wgpu_montecarlo import MonteCarloIntegrator, Distribution, integrate

    HAS_EXTENSION = True
except ImportError:
    HAS_EXTENSION = False


@pytest.mark.skipif(not HAS_EXTENSION, reason="Rust extension not built")
class TestMonteCarloIntegrator:
    """Test cases for MonteCarloIntegrator."""

    def test_init(self):
        """Test integrator initialization."""
        integrator = MonteCarloIntegrator()
        assert integrator is not None

    def test_single_function(self):
        """Test integration with a single function."""
        integrator = MonteCarloIntegrator()
        dist = Distribution.normal(mean=0.0, std=1.0)

        result = integrator.integrate([lambda x: x], dist, n_samples=1000000, seed=42)

        assert result is not None
        assert len(result.values) == 1
        assert abs(result.values[0]) < 0.1

    def test_multiple_functions(self):
        """Test integration with multiple functions."""
        integrator = MonteCarloIntegrator()
        dist = Distribution.normal(mean=0.0, std=1.0)

        funcs = [lambda x: x, lambda x: x**2, lambda x: x**3]
        result = integrator.integrate(funcs, dist, n_samples=1000000, seed=42)

        assert len(result.values) == 3
        assert abs(result.values[0]) < 0.1
        assert abs(result.values[1] - 1.0) < 0.1
        assert abs(result.values[2]) < 0.1

    def test_wgsl_string_function(self):
        """Test integration with WGSL string."""
        integrator = MonteCarloIntegrator()
        dist = Distribution.normal(mean=0.0, std=1.0)

        wgsl_func = "fn f(x: f32) -> f32 { return x * x; }"
        result = integrator.integrate([wgsl_func], dist, n_samples=1000000, seed=42)

        assert abs(result.values[0] - 1.0) < 0.1

    def test_mixed_callable_and_wgsl(self):
        """Test integration with mixed callable and WGSL functions."""
        integrator = MonteCarloIntegrator()
        dist = Distribution.normal(mean=0.0, std=1.0)

        funcs = [
            lambda x: x,
            "fn f(x: f32) -> f32 { return x * x; }",
        ]
        result = integrator.integrate(funcs, dist, n_samples=1000000, seed=42)

        assert len(result.values) == 2
        assert abs(result.values[0]) < 0.1
        assert abs(result.values[1] - 1.0) < 0.1

    def test_empty_functions_error(self):
        """Test that empty function list raises error."""
        integrator = MonteCarloIntegrator()
        dist = Distribution.normal(mean=0.0, std=1.0)

        with pytest.raises(ValueError):
            integrator.integrate([], dist, n_samples=1000)

    def test_invalid_function_type_error(self):
        """Test that invalid function type raises error."""
        integrator = MonteCarloIntegrator()
        dist = Distribution.normal(mean=0.0, std=1.0)

        with pytest.raises(TypeError):
            integrator.integrate([123], dist, n_samples=1000)


@pytest.mark.skipif(not HAS_EXTENSION, reason="Rust extension not built")
class TestInlineLambdas:
    """Test various inline lambda patterns (user input scenarios)."""

    def test_inline_lambdas_in_function_call(self):
        """Test lambdas written directly in function call."""
        integrator = MonteCarloIntegrator()
        dist = Distribution.normal(mean=0.0, std=1.0)

        result = integrator.integrate(
            [lambda x: x, lambda x: x**2], dist, n_samples=1000000, seed=42
        )

        assert abs(result.values[0]) < 0.1
        assert abs(result.values[1] - 1.0) < 0.1

    def test_inline_lambdas_four_functions(self):
        """Test four lambdas on same line."""
        integrator = MonteCarloIntegrator()
        dist = Distribution.normal(mean=0.0, std=1.0)

        result = integrator.integrate(
            [lambda x: x, lambda x: x**2, lambda x: x**3, lambda x: x**4],
            dist,
            n_samples=1000000,
            seed=42,
        )

        assert abs(result.values[0]) < 0.1
        assert abs(result.values[1] - 1.0) < 0.1
        assert abs(result.values[2]) < 0.1
        assert abs(result.values[3] - 3.0) < 0.1

    def test_tuple_unpacking_lambdas(self):
        """Test lambdas assigned via tuple unpacking."""
        f1, f2 = lambda x: x, lambda x: x**2

        integrator = MonteCarloIntegrator()
        dist = Distribution.normal(mean=0.0, std=1.0)

        result = integrator.integrate([f1, f2], dist, n_samples=1000000, seed=42)

        assert abs(result.values[0]) < 0.1
        assert abs(result.values[1] - 1.0) < 0.1

    def test_inline_lambdas_with_global_variables(self):
        """Test inline lambdas with global variable capture."""
        coeff = 2.0

        integrator = MonteCarloIntegrator()
        dist = Distribution.normal(mean=0.0, std=1.0)

        result = integrator.integrate(
            [lambda x: coeff * x, lambda x: coeff * x**2],
            dist,
            n_samples=1000000,
            seed=42,
        )

        assert abs(result.values[0]) < 0.1
        assert abs(result.values[1] - coeff) < 0.1

    def test_inline_lambdas_with_constants(self):
        """Test inline lambdas with math constants."""
        integrator = MonteCarloIntegrator()
        dist = Distribution.normal(mean=0.0, std=1.0)

        result = integrator.integrate(
            [lambda x: math.pi, lambda x: math.e], dist, n_samples=1000000, seed=42
        )

        assert abs(result.values[0] - math.pi) < 0.1
        assert abs(result.values[1] - math.e) < 0.1


@pytest.mark.skipif(not HAS_EXTENSION, reason="Rust extension not built")
class TestIntegrationAccuracy:
    """Test numerical accuracy of integrations."""

    def test_polynomial_expectation(self):
        """Test polynomial expectation E[aX² + bX + c]."""
        integrator = MonteCarloIntegrator()
        dist = Distribution.normal(mean=0.0, std=1.0)

        a, b, c = 1.0, 2.0, 3.0
        funcs = [lambda x: a * x * x + b * x + c]
        result = integrator.integrate(funcs, dist, n_samples=10000000, seed=42)

        expected = a * 1.0 + b * 0.0 + c
        assert abs(result.values[0] - expected) < 0.1

    def test_normal_mean_and_variance(self):
        """Test E[X] and Var[X] for normal distribution."""
        integrator = MonteCarloIntegrator()
        dist = Distribution.normal(mean=0.0, std=1.0)

        funcs = [lambda x: x, lambda x: x * x]
        result = integrator.integrate(funcs, dist, n_samples=10000000, seed=42)

        mean = result.values[0]
        e_x2 = result.values[1]
        variance = e_x2 - mean**2

        assert abs(mean) < 0.01
        assert abs(variance - 1.0) < 0.01

    def test_uniform_mean_and_variance(self):
        """Test E[X] and Var[X] for uniform distribution."""
        integrator = MonteCarloIntegrator()
        dist = Distribution.uniform(min=0.0, max=1.0)

        funcs = [lambda x: x, lambda x: x * x]
        result = integrator.integrate(funcs, dist, n_samples=10000000, seed=42)

        mean = result.values[0]
        e_x2 = result.values[1]
        variance = e_x2 - mean**2

        assert abs(mean - 0.5) < 0.01
        assert abs(variance - 1.0 / 12.0) < 0.01

    def test_exponential_mean_and_variance(self):
        """Test E[X] and Var[X] for exponential distribution."""
        integrator = MonteCarloIntegrator()
        lam = 2.0
        dist = Distribution.exponential(lambda_param=lam)

        funcs = [lambda x: x, lambda x: x * x]
        result = integrator.integrate(funcs, dist, n_samples=10000000, seed=42)

        expected_mean = 1.0 / lam
        expected_var = 1.0 / (lam * lam)

        mean = result.values[0]
        e_x2 = result.values[1]
        variance = e_x2 - mean**2

        assert abs(mean - expected_mean) < 0.01
        assert abs(variance - expected_var) < 0.01

    def test_moment_calculations(self):
        """Test higher moment calculations for N(0,1)."""
        integrator = MonteCarloIntegrator()
        dist = Distribution.normal(mean=0.0, std=1.0)

        funcs = [
            lambda x: x,
            lambda x: x * x,
            lambda x: x * x * x,
            lambda x: x * x * x * x,
        ]
        result = integrator.integrate(funcs, dist, n_samples=10000000, seed=42)

        assert abs(result.values[0]) < 0.01
        assert abs(result.values[1] - 1.0) < 0.01
        assert abs(result.values[2]) < 0.01
        assert abs(result.values[3] - 3.0) < 0.01

    def test_trigonometric_expectations(self):
        """Test E[sin(X)] and E[cos(X)] for uniform distribution."""
        integrator = MonteCarloIntegrator()
        dist = Distribution.uniform(min=0.0, max=2 * math.pi)

        funcs = [lambda x: math.sin(x), lambda x: math.cos(x)]
        result = integrator.integrate(funcs, dist, n_samples=10000000, seed=42)

        assert abs(result.values[0]) < 0.01
        assert abs(result.values[1]) < 0.01


@pytest.mark.skipif(not HAS_EXTENSION, reason="Rust extension not built")
class TestIntegrationWithGlobalVariables:
    """Test integration with global variables."""

    def test_global_variable_in_lambda(self):
        """Test global variable capture in lambda function."""
        a = 2.0
        b = 1.0

        integrator = MonteCarloIntegrator()
        dist = Distribution.normal(mean=0.0, std=1.0)

        func = lambda x: a * x + b
        result = integrator.integrate([func], dist, n_samples=10000000, seed=42)

        expected = a * 0.0 + b
        assert abs(result.values[0] - expected) < 0.01

    def test_global_variable_polynomial(self):
        """Test global variables in polynomial."""
        coeff_a = 1.0
        coeff_b = 2.0
        coeff_c = 3.0

        integrator = MonteCarloIntegrator()
        dist = Distribution.normal(mean=0.0, std=1.0)

        func = lambda x: coeff_a * x * x + coeff_b * x + coeff_c
        result = integrator.integrate([func], dist, n_samples=10000000, seed=42)

        expected = coeff_a * 1.0 + coeff_b * 0.0 + coeff_c
        assert abs(result.values[0] - expected) < 0.1

    def test_closure_capture(self):
        """Test closure variable capture."""

        def make_func(a, b):
            return lambda x: a * x + b

        func = make_func(2.0, 1.0)

        integrator = MonteCarloIntegrator()
        dist = Distribution.normal(mean=0.0, std=1.0)

        result = integrator.integrate([func], dist, n_samples=10000000, seed=42)

        expected = 2.0 * 0.0 + 1.0
        assert abs(result.values[0] - expected) < 0.01


@pytest.mark.skipif(not HAS_EXTENSION, reason="Rust extension not built")
class TestIntegrationWithConstants:
    """Test integration with mathematical constants."""

    def test_pi_constant(self):
        """Test math.pi constant in integration."""
        integrator = MonteCarloIntegrator()
        dist = Distribution.uniform(min=0.0, max=1.0)

        func = lambda x: math.pi * x
        result = integrator.integrate([func], dist, n_samples=10000000, seed=42)

        expected = math.pi * 0.5
        assert abs(result.values[0] - expected) < 0.01

    def test_e_constant(self):
        """Test math.e constant in integration."""
        integrator = MonteCarloIntegrator()
        dist = Distribution.normal(mean=0.0, std=1.0)

        func = lambda x: math.e
        result = integrator.integrate([func], dist, n_samples=10000000, seed=42)

        assert abs(result.values[0] - math.e) < 0.01


@pytest.mark.skipif(not HAS_EXTENSION, reason="Rust extension not built")
class TestIntegratorConvenience:
    """Test convenience functions."""

    def test_integrate_function(self):
        """Test integrate convenience function."""
        dist = Distribution.normal(mean=0.0, std=1.0)

        result = integrate(
            [lambda x: x, lambda x: x * x],
            dist,
            n_samples=1000000,
            seed=42,
        )

        assert len(result.values) == 2
        assert abs(result.values[0]) < 0.1
        assert abs(result.values[1] - 1.0) < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
