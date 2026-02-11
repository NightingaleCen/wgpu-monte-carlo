"""WGSL Transpiler - Converts Python functions to WGSL shader code."""

import ast
import inspect
import sys
import textwrap
from typing import Callable, Dict, Set

# Check Python version for lambda support
_PYTHON_SUPPORTS_LAMBDA_POSITIONS = sys.version_info >= (3, 11)
if not _PYTHON_SUPPORTS_LAMBDA_POSITIONS:
    import warnings

    warnings.warn(
        "Python < 3.11 detected. Lambda transpilation may fail for multiple "
        "lambdas defined on the same line. Consider upgrading to Python 3.11+ "
        "or defining each lambda on a separate line.",
        UserWarning,
        stacklevel=2,
    )


class TranspilerError(Exception):
    """Error raised during transpilation."""

    pass


class PythonToWGSL:
    """Transpiles a restricted subset of Python math functions to WGSL."""

    # TODO: Add support for numpy functions

    # Mapping of Python operators to WGSL operators
    OP_MAP: Dict[str, str] = {
        "Add": "+",
        "Sub": "-",
        "Mult": "*",
        "Div": "/",
        "Mod": "%",
        "Pow": "",  # Special case: handled separately
        "Gt": ">",
        "Lt": "<",
        "GtE": ">=",
        "LtE": "<=",
        "Eq": "==",
        "NotEq": "!=",
    }

    # Mapping of Python math functions to WGSL built-ins
    FUNC_MAP: Dict[str, str] = {
        "abs": "abs",
        "sin": "sin",
        "cos": "cos",
        "tan": "tan",
        "asin": "asin",
        "acos": "acos",
        "atan": "atan",
        "sinh": "sinh",
        "cosh": "cosh",
        "tanh": "tanh",
        "sqrt": "sqrt",
        "exp": "exp",
        "exp2": "exp2",
        "log": "log",
        "log2": "log2",
        "floor": "floor",
        "ceil": "ceil",
        "round": "round",
        "trunc": "trunc",
        "fract": "fract",
        "sign": "sign",
        "min": "min",
        "max": "max",
        "clamp": "clamp",
        "mix": "mix",
        "step": "step",
        "smoothstep": "smoothstep",
        "pow": "pow",
    }

    def __init__(self):
        self.indent_level = 0
        self.local_vars: Set[str] = set()

    def transpile(self, func: Callable) -> str:
        """
        Transpile a Python function to WGSL.

        Args:
            func: A Python function with a restricted subset of operations

        Returns:
            WGSL code string

        Raises:
            TranspilerError: If the function contains unsupported operations
        """
        # Check if this is a lambda function
        if func.__name__ == "<lambda>":
            return self._transpile_lambda(func)

        # Regular function transpilation
        try:
            source = inspect.getsource(func)
        except OSError as e:
            raise TranspilerError(f"Could not get source code: {e}")

        # Remove leading indentation (handles functions defined inside classes/methods)
        source = textwrap.dedent(source)

        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            raise TranspilerError(f"Invalid Python syntax: {e}")

        # Find the function definition
        func_def = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_def = node
                break

        if func_def is not None:
            return self._visit_function(func_def)
        else:
            raise TranspilerError("No function definition found")

    def _visit_function(self, node: ast.FunctionDef) -> str:
        """Visit a function definition and generate WGSL."""
        # Extract function name and parameters
        name = node.name
        params = [arg.arg for arg in node.args.args]

        # Build parameter list with types (all f32 for now)
        param_list = ", ".join(f"{p}: f32" for p in params)

        # Visit function body
        body_lines = []
        self.indent_level = 1
        for stmt in node.body:
            body_lines.extend(self._visit_statement(stmt))

        body_str = "\n    ".join(body_lines)

        return f"fn {name}({param_list}) -> f32 {{\n    {body_str}\n}}"

    def _transpile_lambda(self, func: Callable) -> str:
        """Transpile a lambda function, handling multiple lambdas on same line."""
        try:
            source = inspect.getsource(func)
        except OSError as e:
            raise TranspilerError(f"Could not get source code: {e}")

        # Calculate leading indentation before dedenting
        # This is needed because co_positions() returns positions in the original file
        leading_spaces = len(source) - len(source.lstrip())

        # Remove leading indentation
        source = textwrap.dedent(source)

        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            raise TranspilerError(f"Invalid Python syntax: {e}")

        # Find all lambda nodes
        lambdas = [node for node in ast.walk(tree) if isinstance(node, ast.Lambda)]

        if len(lambdas) == 0:
            raise TranspilerError("No lambda definition found")

        if len(lambdas) == 1:
            # Single lambda - transpile it directly
            return self._visit_lambda(lambdas[0])

        # Multiple lambdas on same line - need to match using code positions (Python 3.11+)
        if not _PYTHON_SUPPORTS_LAMBDA_POSITIONS:
            raise TranspilerError(
                "Multiple lambdas on the same line detected. "
                "Python < 3.11 cannot distinguish them. "
                "Please define each lambda on a separate line or upgrade to Python 3.11+."
            )

        # Use co_positions to find the correct lambda
        # The lambda's body position in co_positions should match the AST body's position
        target_positions = list(func.__code__.co_positions())
        if not target_positions:
            raise TranspilerError("Could not get code positions for lambda")

        # Get the first meaningful position (usually the second one, as first is often 0)
        # This gives us the column where the lambda body starts
        target_col = None
        for pos in target_positions:
            if pos[2] is not None and pos[2] > 0:
                target_col = pos[2]
                break

        if target_col is None:
            raise TranspilerError("Could not determine lambda position")

        # Adjust target_col to account for dedentation
        # co_positions() gives positions in the original file (with indentation)
        # AST gives positions in the dedented source
        target_col_adjusted = target_col - leading_spaces

        # Match by comparing the body column positions
        # target_col is from code positions (where the body expression starts)
        # AST lambda body also has col_offset (where the body expression starts in source)
        def get_body_start_col(lam):
            """Get the column where the lambda body starts."""
            body = lam.body
            if hasattr(body, "col_offset") and body.col_offset is not None:
                return body.col_offset
            # Fallback: dynamically calculate offset from lambda to body
            # This handles variable-length parameter names correctly
            # In practice, body.col_offset is always available in Python 3.8+
            raise TranspilerError(
                "Lambda body position information not available. "
                "Please upgrade to Python 3.8+ for lambda transpilation support."
            )

        best_match = min(
            lambdas, key=lambda lam: abs(get_body_start_col(lam) - target_col_adjusted)
        )

        return self._visit_lambda(best_match)

    def _visit_lambda(self, node: ast.Lambda) -> str:
        """Visit a lambda expression and generate WGSL."""
        # Lambda doesn't have a name, generate one
        import uuid

        name = f"user_func_{uuid.uuid4().hex[:8]}"
        params = [arg.arg for arg in node.args.args]

        # Build parameter list with types (all f32 for now)
        param_list = ", ".join(f"{p}: f32" for p in params)

        # Lambda body is a single expression
        body_expr = self._visit_expression(node.body)

        # If the expression is a comparison (returns bool), convert to f32
        if self._is_boolean_expression(node.body):
            body_expr = f"select(0.0, 1.0, {body_expr})"

        return f"fn {name}({param_list}) -> f32 {{\n    return {body_expr};\n}}"

    def _is_boolean_expression(self, node: ast.AST) -> bool:
        """Check if an expression returns a boolean value."""
        return isinstance(node, (ast.Compare, ast.BoolOp))

    def _wrap_boolean_to_f32(self, expr_code: str, node: ast.AST) -> str:
        """Wrap a boolean expression to convert it to f32."""
        if self._is_boolean_expression(node):
            return f"select(0.0, 1.0, {expr_code})"
        return expr_code

    def _visit_statement(self, node: ast.AST) -> list:
        """Visit a statement and return lines of WGSL code."""
        if isinstance(node, ast.Return):
            if node.value is None:
                return ["return;"]
            expr_code = self._visit_expression(node.value)
            # Convert boolean expressions to f32
            expr_code = self._wrap_boolean_to_f32(expr_code, node.value)
            return [f"return {expr_code};"]
        elif isinstance(node, ast.Assign):
            return self._visit_assignment(node)
        elif isinstance(node, ast.If):
            return self._visit_if(node)
        elif isinstance(node, ast.For):
            return self._visit_for(node)
        elif isinstance(node, ast.While):
            return self._visit_while(node)
        elif isinstance(node, ast.Expr):
            # Expression statement (e.g., function call for side effects)
            return [f"{self._visit_expression(node.value)};"]
        else:
            raise TranspilerError(f"Unsupported statement type: {type(node).__name__}")

    def _visit_assignment(self, node: ast.Assign) -> list:
        """Visit an assignment statement."""
        if len(node.targets) != 1:
            raise TranspilerError("Multiple assignment targets not supported")

        target = node.targets[0]
        if not isinstance(target, ast.Name):
            raise TranspilerError("Only simple variable assignment supported")

        var_name = target.id
        value = self._visit_expression(node.value)

        # Check if it's a new variable or reassigning
        if var_name not in self.local_vars:
            self.local_vars.add(var_name)
            return [f"var {var_name} = {value};"]
        else:
            return [f"{var_name} = {value};"]

    def _visit_if(self, node: ast.If) -> list:
        """Visit an if statement."""
        lines = []
        cond = self._visit_expression(node.test)
        lines.append(f"if ({cond}) {{")

        self.indent_level += 1
        for stmt in node.body:
            lines.extend(f"    {i}" for i in self._visit_statement(stmt))
        self.indent_level -= 1

        if node.orelse:
            lines.append("} else {")
            self.indent_level += 1
            for stmt in node.orelse:
                lines.extend(f"    {i}" for i in self._visit_statement(stmt))
            self.indent_level -= 1

        lines.append("}")
        return lines

    def _visit_for(self, node: ast.For) -> list:
        """Visit a for loop (limited to range-based loops)."""
        raise TranspilerError("For loops not yet implemented")

    def _visit_while(self, node: ast.While) -> list:
        lines = []
        cond = self._visit_expression(node.test)
        lines.append(f"while ({cond}) {{")

        self.indent_level += 1
        for stmt in node.body:
            lines.extend(f"    {i}" for i in self._visit_statement(stmt))
        self.indent_level -= 1

        lines.append("}")
        return lines

    def _visit_expression(self, node: ast.AST) -> str:
        """Visit an expression and return WGSL code."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return "true" if node.value else "false"
            elif isinstance(node.value, (int, float)):
                return str(float(node.value))
            else:
                raise TranspilerError(f"Unsupported constant type: {type(node.value)}")
        elif isinstance(node, ast.Num):  # Python < 3.8 compatibility
            return str(float(node.n))
        elif isinstance(node, ast.BinOp):
            return self._visit_binop(node)
        elif isinstance(node, ast.UnaryOp):
            return self._visit_unaryop(node)
        elif isinstance(node, ast.Call):
            return self._visit_call(node)
        elif isinstance(node, ast.IfExp):
            return self._visit_conditional(node)
        elif isinstance(node, ast.Compare):
            return self._visit_compare(node)
        else:
            raise TranspilerError(f"Unsupported expression type: {type(node).__name__}")

    def _visit_binop(self, node: ast.BinOp) -> str:
        """Visit a binary operation."""
        left = self._visit_expression(node.left)
        right = self._visit_expression(node.right)
        op_type = type(node.op).__name__

        if op_type == "Pow":
            return f"pow({left}, {right})"

        op = self.OP_MAP.get(op_type)
        if op is None:
            raise TranspilerError(f"Unsupported binary operator: {op_type}")

        return f"({left} {op} {right})"

    def _visit_unaryop(self, node: ast.UnaryOp) -> str:
        """Visit a unary operation."""
        operand = self._visit_expression(node.operand)
        op_type = type(node.op).__name__

        if op_type == "USub":
            return f"(-{operand})"
        elif op_type == "UAdd":
            return f"(+{operand})"
        else:
            raise TranspilerError(f"Unsupported unary operator: {op_type}")

    def _visit_call(self, node: ast.Call) -> str:
        """Visit a function call."""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            # Handle math.sin, etc.
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "math":
                func_name = node.func.attr
            else:
                raise TranspilerError(f"Unsupported attribute access: {node.func.attr}")
        else:
            raise TranspilerError("Unsupported function call")

        # Map to WGSL function
        wgsl_func = self.FUNC_MAP.get(func_name, func_name)

        # Build argument list
        args = [self._visit_expression(arg) for arg in node.args]
        args_str = ", ".join(args)

        return f"{wgsl_func}({args_str})"

    def _visit_conditional(self, node: ast.IfExp) -> str:
        """Visit a conditional expression (x if cond else y)."""
        test = self._visit_expression(node.test)
        body = self._visit_expression(node.body)
        orelse = self._visit_expression(node.orelse)
        return f"select({orelse}, {body}, {test})"

    def _visit_compare(self, node: ast.Compare) -> str:
        """Visit a comparison expression (e.g., x > 0)."""
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise TranspilerError("Only simple comparisons supported (e.g., x > y)")

        left = self._visit_expression(node.left)
        right = self._visit_expression(node.comparators[0])
        op_type = type(node.ops[0]).__name__

        op = self.OP_MAP.get(op_type)
        if op is None:
            raise TranspilerError(f"Unsupported comparison operator: {op_type}")

        return f"({left} {op} {right})"


def transpile_function(func: Callable) -> str:
    """Convenience function to transpile a Python function to WGSL."""
    transpiler = PythonToWGSL()
    return transpiler.transpile(func)
