"""Tests for the WGSL transpiler."""

import pytest
import math
from wgpu_montecarlo.transpiler import PythonToWGSL, transpile_function, TranspilerError


class TestTranspiler:
    """Test cases for the Python to WGSL transpiler."""

    def test_simple_function(self):
        """Test transpiling a simple function."""

        def step(x, rng):
            return x + rng

        transpiler = PythonToWGSL()
        result = transpiler.transpile(step)

        assert "fn step(x: f32, rng: f32) -> f32" in result
        assert "return" in result

    def test_math_functions(self):
        """Test transpiling math function calls."""

        def step(x, rng):
            return math.sin(x) + math.cos(rng)

        transpiler = PythonToWGSL()
        result = transpiler.transpile(step)

        assert "sin(x)" in result
        assert "cos(rng)" in result

    def test_arithmetic_operations(self):
        """Test transpiling arithmetic operations."""

        def step(x, rng):
            a = x * 2.0
            b = a / 3.0
            c = b + rng
            return c - 1.0

        transpiler = PythonToWGSL()
        result = transpiler.transpile(step)

        assert "var a = (x * 2.0)" in result
        assert "var b = (a / 3.0)" in result
        assert "return (c - 1.0)" in result

    def test_power_operation(self):
        """Test transpiling power operator."""

        def step(x, rng):
            return x**2.0

        transpiler = PythonToWGSL()
        result = transpiler.transpile(step)

        assert "pow(x, 2.0)" in result

    def test_unary_operations(self):
        """Test transpiling unary operations."""

        def step(x, rng):
            return -x + +rng

        transpiler = PythonToWGSL()
        result = transpiler.transpile(step)

        assert "(-x)" in result or "-x" in result

    def test_local_variables(self):
        """Test handling of local variables."""

        def step(x, rng):
            temp = x * 0.5
            result = temp + rng
            return result

        transpiler = PythonToWGSL()
        result = transpiler.transpile(step)

        assert "var temp =" in result
        assert "var result =" in result

    def test_conditional_expression(self):
        """Test transpiling conditional expressions."""

        def step(x, rng):
            return x if x > 0 else rng

        transpiler = PythonToWGSL()
        result = transpiler.transpile(step)

        assert "select(" in result

    def test_comparison_operators(self):
        """Test transpiling comparison operators."""

        def step(x, rng):
            if x > rng:
                return x
            else:
                return rng

        transpiler = PythonToWGSL()
        result = transpiler.transpile(step)

        assert "if ((x > rng))" in result or "if (x > rng)" in result

    def test_empty_return(self):
        """Test that empty returns raise an error (not supported yet)."""

        def step(x, rng):
            return

        transpiler = PythonToWGSL()
        # Should not raise an error, should return "return;"
        result = transpiler.transpile(step)
        assert "return;" in result


class TestTranspileFunction:
    """Test the convenience transpile_function."""

    def test_convenience_function(self):
        """Test the convenience transpile_function."""

        def step(x, rng):
            return x + rng

        result = transpile_function(step)
        assert "fn step" in result


class TestLambdaTranspilation:
    """Test lambda function transpilation (requires Python 3.11+)."""

    def test_single_lambda(self):
        """Test transpiling a single lambda."""
        f = lambda x: x * 2
        result = transpile_function(f)
        assert "return (x * 2.0)" in result

    def test_lambda_returning_comparison(self):
        """Test lambda returning comparison (bool to f32 conversion)."""
        f = lambda x: x > 0.5
        result = transpile_function(f)
        assert "select(0.0, 1.0, (x > 0.5))" in result

    def test_multiple_lambdas_separate_lines(self):
        """Test multiple lambdas defined on separate lines."""
        f1 = lambda x: x
        f2 = lambda x: x**2
        f3 = lambda x: x**3

        results = [transpile_function(f) for f in [f1, f2, f3]]
        assert "return x;" in results[0]
        assert "pow(x, 2.0)" in results[1]
        assert "pow(x, 3.0)" in results[2]

    def test_multiple_lambdas_same_line(self):
        """Test multiple lambdas defined on the same line (Python 3.11+)."""
        f1, f2 = lambda x: x, lambda x: x**2

        r1 = transpile_function(f1)
        r2 = transpile_function(f2)

        # Ensure different code is generated for different lambdas
        assert "return x;" in r1, f"First lambda should return x, got: {r1}"
        assert "pow(x, 2.0)" in r2, f"Second lambda should return x**2, got: {r2}"
        assert r1 != r2, "Different lambdas should generate different code"


class TestBooleanConversion:
    """Test boolean expression to f32 conversion."""

    def test_function_returning_comparison(self):
        """Test regular function returning comparison."""

        def prob_gt_half(x):
            return x > 0.5

        result = transpile_function(prob_gt_half)
        assert "select(0.0, 1.0, (x > 0.5))" in result

    def test_function_returning_equality(self):
        """Test function returning equality comparison."""

        def is_zero(x):
            return x == 0.0

        result = transpile_function(is_zero)
        assert "select(0.0, 1.0, (x == 0.0))" in result

    def test_function_returning_less_than(self):
        """Test function returning less than comparison."""

        def is_negative(x):
            return x < 0.0

        result = transpile_function(is_negative)
        assert "select(0.0, 1.0, (x < 0.0))" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
