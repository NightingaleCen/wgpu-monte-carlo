"""Tests for the WGSL transpiler."""

import math
import numpy as np
import pytest
from numpy import sin as np_sin
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
        assert "Undefined variable" in str(
            exc_info.value
        ) or "Unsupported module" in str(exc_info.value)

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
        assert "Undefined variable" in str(
            exc_info.value
        ) or "Unsupported module" in str(exc_info.value)

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


class TestConstants:
    """Test handling of mathematical constants from math and numpy modules."""

    def test_math_pi_attribute(self):
        """Test math.pi constant."""

        def step(x):
            return math.pi * x

        result = transpile_function(step)
        assert "3.141592653589793" in result

    def test_math_e_attribute(self):
        """Test math.e constant."""

        def step(x):
            return math.e**x

        result = transpile_function(step)
        assert "2.718281828459045" in result

    def test_math_tau_attribute(self):
        """Test math.tau constant."""

        def step(x):
            return math.tau * x

        result = transpile_function(step)
        assert "6.283185307179586" in result

    def test_math_inf_attribute(self):
        """Test math.inf constant."""

        def step(x):
            return x if x < math.inf else 0.0

        result = transpile_function(step)
        assert "1e300" in result

    def test_numpy_pi_attribute(self):
        """Test np.pi constant."""

        def step(x):
            return np.pi * x

        result = transpile_function(step)
        assert "3.141592653589793" in result

    def test_numpy_e_attribute(self):
        """Test np.e constant."""

        def step(x):
            return np.e * x

        result = transpile_function(step)
        assert "2.718281828459045" in result

    def test_numpy_euler_gamma_attribute(self):
        """Test np.euler_gamma constant."""

        def step(x):
            return np.euler_gamma * x

        result = transpile_function(step)
        assert "0.577215664901532" in result

    def test_from_math_import_pi(self):
        """Test 'from math import pi' style."""
        from math import pi

        def step(x):
            return pi * x

        result = transpile_function(step)
        assert "3.141592653589793" in result

    def test_from_math_import_e(self):
        """Test 'from math import e' style."""
        from math import e

        def step(x):
            return e * x

        result = transpile_function(step)
        assert "2.718281828459045" in result

    def test_from_numpy_import_pi(self):
        """Test 'from numpy import pi' style."""
        from numpy import pi

        def step(x):
            return pi * x

        result = transpile_function(step)
        assert "3.141592653589793" in result

    def test_from_math_import_as(self):
        """Test 'from math import pi as PI' style."""
        from math import pi as PI

        def step(x):
            return PI * x

        result = transpile_function(step)
        assert "3.141592653589793" in result

    def test_custom_alias_constant(self):
        """Test custom alias for math module with constant."""
        import math as mathh

        def step(x):
            return mathh.pi * x

        result = transpile_function(step)
        assert "3.141592653589793" in result

    def test_numpy_custom_alias_constant(self):
        """Test custom alias for numpy module with constant."""
        import numpy as npp

        def step(x):
            return npp.pi * x

        result = transpile_function(step)
        assert "3.141592653589793" in result

    def test_mixed_constant_and_function(self):
        """Test mixing constants and function calls."""

        def step(x):
            return math.sin(x) * math.pi + math.e

        result = transpile_function(step)
        assert "sin(x)" in result
        assert "3.141592653589793" in result
        assert "2.718281828459045" in result

    def test_multiple_constants(self):
        """Test multiple constants in one expression."""

        def step(x):
            return math.pi + math.e + math.tau

        result = transpile_function(step)
        assert "3.141592653589793" in result
        assert "2.718281828459045" in result
        assert "6.283185307179586" in result

    def test_unknown_constant_error(self):
        """Test that unknown constants raise appropriate error."""

        def step(x):
            return math.unknown_constant * x

        with pytest.raises(TranspilerError) as exc_info:
            transpile_function(step)
        assert "Unknown constant" in str(exc_info.value)


class TestGlobalVariables:
    """Test handling of external global variables."""

    def test_simple_global_variable(self):
        """Test simple global variable capture."""
        a = 1.5

        def step(x):
            return a * x

        result = transpile_function(step)
        assert "const a: f32 = 1.5" in result
        assert "a * x" in result

    def test_multiple_global_variables(self):
        """Test multiple global variables."""
        a = 1.0
        b = 2.0
        c = 3.0

        def step(x):
            return a * x * x + b * x + c

        result = transpile_function(step)
        assert "const a: f32 = 1.0" in result
        assert "const b: f32 = 2.0" in result
        assert "const c: f32 = 3.0" in result

    def test_global_variable_with_arithmetic(self):
        """Test global variable used in arithmetic."""
        factor = 0.5

        def step(x):
            return (x + factor) * 2.0

        result = transpile_function(step)
        assert "const factor: f32 = 0.5" in result

    def test_global_variable_integer(self):
        """Test integer global variable (converted to float)."""
        n = 5

        def step(x):
            return x * n

        result = transpile_function(step)
        assert "const n: f32 = 5.0" in result or "const n: f32 = 5" in result

    def test_global_variable_boolean(self):
        """Test boolean global variable."""
        flag = True

        def step(x):
            return x if flag else 0.0

        result = transpile_function(step)
        assert "const flag: f32 = 1.0" in result

    def test_local_variable_shadowing(self):
        """Test that local variable assignment shadows global."""
        a = 1.5

        def step(x):
            a = 2.0
            return a * x

        result = transpile_function(step)
        assert "const a" not in result
        assert "var a = 2.0" in result

    def test_lambda_with_global_variable(self):
        """Test lambda function with global variable."""
        scale = 3.0

        result = transpile_function(lambda x: scale * x)
        assert "const scale: f32 = 3.0" in result

    def test_lambda_multiple_global_variables(self):
        """Test lambda with multiple global variables."""
        a = 1.0
        b = 2.0

        result = transpile_function(lambda x: a * x + b)
        assert "const a: f32 = 1.0" in result
        assert "const b: f32 = 2.0" in result

    def test_global_variable_not_found_error(self):
        """Test error when global variable is not defined."""

        def step(x):
            return undefined_var * x

        with pytest.raises(TranspilerError) as exc_info:
            transpile_function(step)
        assert "not found" in str(exc_info.value) or "undefined_var" in str(
            exc_info.value
        )

    def test_unsupported_global_type_list(self):
        """Test error for unsupported list type."""
        my_list = [1, 2, 3]

        def step(x):
            return x + my_list[0]

        with pytest.raises(TranspilerError) as exc_info:
            transpile_function(step)
        assert "Unsupported external variable type" in str(exc_info.value)
        assert "list" in str(exc_info.value)

    def test_unsupported_global_type_dict(self):
        """Test error for unsupported dict type."""
        my_dict = {"key": 1.0}

        def step(x):
            return x + my_dict["key"]

        with pytest.raises(TranspilerError) as exc_info:
            transpile_function(step)
        assert "Unsupported external variable type" in str(exc_info.value)
        assert "dict" in str(exc_info.value)

    def test_global_variable_with_math_functions(self):
        """Test global variable combined with math functions."""
        coeff = 2.0

        def step(x):
            return math.sin(coeff * x)

        result = transpile_function(step)
        assert "const coeff: f32 = 2.0" in result
        assert "sin" in result

    def test_global_variable_negative(self):
        """Test negative global variable."""
        offset = -5.0

        def step(x):
            return x + offset

        result = transpile_function(step)
        assert "const offset: f32 = -5.0" in result

    def test_global_variable_scientific_notation(self):
        """Test global variable in scientific notation."""
        small = 1e-10

        def step(x):
            return x + small

        result = transpile_function(step)
        assert "const small: f32 = 1e-10" in result or "const small: f32 = " in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
