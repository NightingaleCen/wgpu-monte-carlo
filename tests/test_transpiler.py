"""Tests for the WGSL transpiler."""

import math
import numpy as np
import pytest
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


class TestNumpyImportHandling:
    """Test numpy import handling with various import styles."""

    def test_np_prefix(self):
        """Test np.sin(x) style import (import numpy as np)."""

        def step(x):
            return np.sin(x)

        result = transpile_function(step)
        assert "sin(x)" in result

    def test_np_prefix_multiple_functions(self):
        """Test multiple np. function calls."""

        def step(x, y):
            return np.sin(x) + np.cos(y)

        result = transpile_function(step)
        assert "sin(x)" in result
        assert "cos(y)" in result

    def test_numpy_prefix(self):
        """Test numpy.sin(x) style import (import numpy)."""

        def step(x):
            return numpy.sin(x)

        result = transpile_function(step)
        assert "sin(x)" in result

    def test_from_numpy_import(self):
        """Test from numpy import sin style."""

        def step(x):
            return sin(x)

        result = transpile_function(step)
        assert "sin(x)" in result

    def test_from_numpy_import_multiple(self):
        """Test from numpy import sin, cos style."""

        def step(x, y):
            return sin(x) + cos(y)

        result = transpile_function(step)
        assert "sin(x)" in result
        assert "cos(y)" in result

    def test_from_numpy_import_as(self):
        """Test from numpy import sin as np_sin style."""

        def step(x):
            return np_sin(x)

        result = transpile_function(step)
        assert "sin(x)" in result

    def test_mixed_math_and_numpy(self):
        """Test mixed imports: import numpy as np; from math import cos."""

        def step(x, y):
            return np.sin(x) + cos(y)

        result = transpile_function(step)
        assert "sin(x)" in result
        assert "cos(y)" in result

    def test_aliased_numpy_module(self):
        """Test import numpy as n style (custom alias).

        Note: Custom aliases like 'n' require explicit import in function source
        or adding to KNOWN_MODULE_ALIASES. This test verifies the behavior
        when alias is NOT in known aliases (should raise error).
        """

        def step(x):
            return n.sin(x)

        with pytest.raises(TranspilerError) as exc_info:
            transpile_function(step)
        assert "Unsupported module" in str(exc_info.value)

    def test_math_module_attribute(self):
        """Test math.sin(x) style (original behavior)."""

        def step(x):
            return math.sin(x)

        result = transpile_function(step)
        assert "sin(x)" in result

    def test_from_math_import(self):
        """Test from math import sin style."""

        def step(x):
            return sin(x)

        result = transpile_function(step)
        assert "sin(x)" in result

    def test_unsupported_module_error(self):
        """Test that unsupported modules raise appropriate error."""

        def step(x):
            return unknown_module.sin(x)

        with pytest.raises(TranspilerError) as exc_info:
            transpile_function(step)
        assert "Unsupported module" in str(exc_info.value)

    def test_numpy_sqrt(self):
        """Test np.sqrt function."""

        def step(x):
            return np.sqrt(x)

        result = transpile_function(step)
        assert "sqrt(x)" in result

    def test_numpy_exp(self):
        """Test np.exp function."""

        def step(x):
            return np.exp(x)

        result = transpile_function(step)
        assert "exp(x)" in result

    def test_numpy_power(self):
        """Test np.power function."""

        def step(x, y):
            return np.power(x, y)

        result = transpile_function(step)
        assert "pow(x, y)" in result


class TestFileLevelImports:
    """Test file-level import handling."""

    def test_file_level_import_numpy(self):
        """Test file-level 'import numpy as np'."""
        import numpy as np  # noqa: F401

        def step(x):
            return np.sin(x)

        result = transpile_function(step)
        assert "sin(x)" in result

    def test_file_level_import_math_as_mma(self):
        """Test file-level 'import math as mma'."""
        import math as mma  # noqa: F401

        def step(x):
            return mma.cos(x)

        result = transpile_function(step)
        assert "cos(x)" in result

    def test_file_level_import_numpy_direct(self):
        """Test file-level 'import numpy'."""
        import numpy  # noqa: F401

        def step(x):
            return numpy.sqrt(x)

        result = transpile_function(step)
        assert "sqrt(x)" in result

    def test_file_level_from_import(self):
        """Test file-level 'from math import sin'."""
        from math import sin  # noqa: F401

        def step(x):
            return sin(x)

        result = transpile_function(step)
        assert "sin(x)" in result

    def test_file_level_mixed_imports(self):
        """Test file-level mixed imports from different modules."""
        import numpy as np  # noqa: F401
        from math import cos  # noqa: F401

        def step(x, y):
            return np.sin(x) + cos(y)

        result = transpile_function(step)
        assert "sin(x)" in result
        assert "cos(y)" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
