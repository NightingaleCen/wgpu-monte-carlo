"""WGSL Transpiler - Converts Python functions to WGSL shader code."""

import ast
import inspect
import linecache
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


def _cache_source_file(filepath: str | None) -> None:
    """Cache source file content to prevent reading modified files during runtime.

    Once a source file is cached, subsequent reads will use the cached version
    even if the file has been modified. This ensures consistency between code
    positions (co_positions) and source code during transpilation.
    """
    if not filepath:
        return
    if filepath in linecache.cache:
        return
    try:
        with open(filepath, "r") as f:
            lines = f.readlines()
        size = sum(len(line) for line in lines)
        linecache.cache[filepath] = (size, None, lines, filepath)
    except OSError:
        pass


_cached_files: Set[str] = set()


def _ensure_source_cached(func: Callable) -> None:
    """Ensure source file is cached for the given function."""
    global _cached_files
    source_file = inspect.getsourcefile(func)
    if source_file and source_file not in _cached_files:
        _cache_source_file(source_file)
        _cached_files.add(source_file)


class TranspilerError(Exception):
    """Error raised during transpilation."""

    pass


class PythonToWGSL:
    """Transpiles a restricted subset of Python math functions to WGSL."""

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
        "power": "pow",
    }

    CONSTANTS_MAP: Dict[tuple, str] = {
        ("math", "pi"): "3.1415926535897932384626433832795",
        ("math", "e"): "2.7182818284590452353602874713527",
        ("math", "tau"): "6.283185307179586476925286766559",
        ("math", "inf"): "1e300",
        ("math", "nan"): "nan",
        ("numpy", "pi"): "3.1415926535897932384626433832795",
        ("numpy", "e"): "2.7182818284590452353602874713527",
        ("numpy", "tau"): "6.283185307179586476925286766559",
        ("numpy", "euler_gamma"): "0.577215664901532860606512090082",
        ("numpy", "inf"): "1e300",
        ("numpy", "nan"): "nan",
    }

    KNOWN_MODULE_ALIASES: Dict[str, str] = {
        "np": "numpy",
        "numpy": "numpy",
        "math": "math",
    }

    def __init__(self):
        self.indent_level = 0
        self.local_vars: Set[str] = set()
        self.imports: Dict[str, str] = {}
        self.module_aliases: Dict[str, str] = dict(self.KNOWN_MODULE_ALIASES)
        self.external_vars: Dict[str, float] = {}

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

        self._analyze_imports(tree)

        source_file = inspect.getsourcefile(func)
        if source_file:
            try:
                with open(source_file, "r") as f:
                    full_source = f.read()
                full_tree = ast.parse(full_source)
                self._analyze_file_imports(full_tree)
            except (OSError, SyntaxError):
                pass

        # Find the function definition
        func_def = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_def = node
                break

        if func_def is None:
            raise TranspilerError("No function definition found")

        # Collect function info and capture global variables
        params, used_names, assigned_names = self._collect_function_info(func_def)
        self._capture_external_vars(func, params, used_names, assigned_names)

        return self._visit_function(func_def)

    def _collect_function_info(self, node: ast.FunctionDef) -> tuple:
        """Collect parameter names, used names, and assigned names from a function."""
        params = {arg.arg for arg in node.args.args}

        class NameCollector(ast.NodeVisitor):
            def __init__(self):
                self.used: Set[str] = set()
                self.assigned: Set[str] = set()

            def visit_Name(self, n):
                self.used.add(n.id)
                self.generic_visit(n)

            def visit_Assign(self, n):
                for target in n.targets:
                    if isinstance(target, ast.Name):
                        self.assigned.add(target.id)
                    elif isinstance(target, ast.Tuple):
                        for elt in target.elts:
                            if isinstance(elt, ast.Name):
                                self.assigned.add(elt.id)
                self.generic_visit(n)

            def visit_For(self, n):
                if isinstance(n.target, ast.Name):
                    self.assigned.add(n.target.id)
                self.generic_visit(n)

        collector = NameCollector()
        for stmt in node.body:
            collector.visit(stmt)

        return params, collector.used, collector.assigned

    def _capture_external_vars(
        self,
        func: Callable,
        params: Set[str],
        used_names: Set[str],
        assigned_names: Set[str],
    ) -> None:
        """Capture external variables from function's globals and closure."""
        self.external_vars.clear()

        # Exclude parameters, local variables, imports, and builtins
        excluded = (
            params
            | assigned_names
            | set(self.imports.keys())
            | set(self.module_aliases.keys())
        )

        builtins = (
            set(dir(__builtins__))
            if isinstance(__builtins__, dict)
            else set(dir(__builtins__))
        )
        excluded |= builtins

        external_names = used_names - excluded

        # Extract closure variables (for nested functions)
        closure_vars = {}
        if func.__closure__:
            freevars = func.__code__.co_freevars
            closure_values = [c.cell_contents for c in func.__closure__]
            closure_vars = dict(zip(freevars, closure_values))

        unknown_names = []
        for name in external_names:
            value = None

            # Check closure first, then globals
            if name in closure_vars:
                value = closure_vars[name]
            elif name in func.__globals__:
                value = func.__globals__[name]

            if value is not None:
                # Skip callables (imported functions)
                if callable(value):
                    pass
                elif isinstance(value, bool):
                    self.external_vars[name] = 1.0 if value else 0.0
                elif isinstance(value, (int, float)):
                    self.external_vars[name] = float(value)
                elif isinstance(value, type(None)):
                    pass
                else:
                    raise TranspilerError(
                        f"Unsupported external variable type for '{name}': {type(value).__name__}. "
                        f"Only int, float, and bool are supported."
                    )
            else:
                unknown_names.append(name)

        if unknown_names:
            raise TranspilerError(
                f"Undefined variable(s): {', '.join(unknown_names)}. "
                f"Variables must be defined in global scope, imported, or passed as parameters."
            )

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

        # Add external variable constants at the beginning
        for var_name, var_value in self.external_vars.items():
            body_lines.append(f"const {var_name}: f32 = {var_value};")

        for stmt in node.body:
            body_lines.extend(self._visit_statement(stmt))

        body_str = "\n    ".join(body_lines)

        return f"fn {name}({param_list}) -> f32 {{\n    {body_str}\n}}"

    def _analyze_imports(self, tree: ast.AST) -> None:
        """Analyze import statements and build mapping tables."""
        self.imports.clear()
        self.module_aliases.clear()
        self.module_aliases.update(self.KNOWN_MODULE_ALIASES)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name
                    alias_name = alias.asname if alias.asname else module_name
                    self.module_aliases[alias_name] = module_name
            elif isinstance(node, ast.ImportFrom):
                module_name = node.module if node.module else ""
                for alias in node.names:
                    imported_name = alias.name
                    alias_name = alias.asname if alias.asname else imported_name
                    full_name = f"{module_name}.{imported_name}"
                    self.imports[alias_name] = full_name

    def _analyze_file_imports(self, tree: ast.AST) -> None:
        """Add file-level import statements without overwriting existing mappings."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name
                    alias_name = alias.asname if alias.asname else module_name
                    if alias_name not in self.module_aliases:
                        self.module_aliases[alias_name] = module_name
            elif isinstance(node, ast.ImportFrom):
                module_name = node.module if node.module else ""
                for alias in node.names:
                    imported_name = alias.name
                    alias_name = alias.asname if alias.asname else imported_name
                    full_name = f"{module_name}.{imported_name}"
                    if alias_name not in self.imports:
                        self.imports[alias_name] = full_name

    def _transpile_lambda(self, func: Callable) -> str:
        """Transpile a lambda function, handling multiple lambdas on same line."""
        _ensure_source_cached(func)

        try:
            source = inspect.getsource(func)
        except OSError as e:
            raise TranspilerError(f"Could not get source code: {e}")

        # Calculate leading indentation for position adjustment
        leading_spaces = len(source) - len(source.lstrip())
        source = textwrap.dedent(source)

        # Try to parse the source directly
        tree = None
        try:
            tree = ast.parse(source)
        except SyntaxError:
            # If parsing fails, the source might be a fragment
            # This happens when lambdas are passed as function arguments
            tree = self._try_parse_lambda_fragment(source)

        if tree is None:
            raise TranspilerError(
                "Could not parse lambda source. "
                "This may happen when lambdas are passed directly in function calls. "
                "Consider assigning the lambda to a variable first."
            )

        self._analyze_imports(tree)

        source_file = inspect.getsourcefile(func)
        if source_file:
            try:
                with open(source_file, "r") as f:
                    full_source = f.read()
                full_tree = ast.parse(full_source)
                self._analyze_file_imports(full_tree)
            except (OSError, SyntaxError):
                pass

        lambdas = [node for node in ast.walk(tree) if isinstance(node, ast.Lambda)]

        if len(lambdas) == 0:
            raise TranspilerError("No lambda definition found")

        if len(lambdas) == 1:
            return self._visit_lambda(lambdas[0], func)

        # Multiple lambdas on same line - match using code positions (Python 3.11+)
        if not _PYTHON_SUPPORTS_LAMBDA_POSITIONS:
            raise TranspilerError(
                "Multiple lambdas on the same line detected. "
                "Python < 3.11 cannot distinguish them. "
                "Please define each lambda on a separate line or upgrade to Python 3.11+."
            )

        target_positions = list(func.__code__.co_positions())
        if not target_positions:
            raise TranspilerError("Could not get code positions for lambda")

        # Find the column where the lambda body starts
        target_col = None
        for pos in target_positions:
            if pos[2] is not None and pos[2] > 0:
                target_col = pos[2]
                break

        if target_col is None:
            raise TranspilerError("Could not determine lambda position")

        # Adjust for dedentation
        target_col_adjusted = target_col - leading_spaces

        def get_body_start_col(lam):
            """Get the column where the lambda body starts."""
            body = lam.body
            if hasattr(body, "col_offset") and body.col_offset is not None:
                return body.col_offset
            raise TranspilerError(
                "Lambda body position information not available. "
                "Please upgrade to Python 3.8+ for lambda transpilation support."
            )

        # Match lambda by comparing body column positions
        best_match = min(
            lambdas, key=lambda lam: abs(get_body_start_col(lam) - target_col_adjusted)
        )

        return self._visit_lambda(best_match, func)

    def _try_parse_lambda_fragment(self, source: str) -> ast.AST | None:
        """Try to parse a lambda fragment that may be incomplete.

        This handles cases where inspect.getsource returns a fragment like:
        '[lambda x: x, lambda x: x**2], other_args, kwarg=val'
        """
        stripped = source.strip()

        # Case 1: List starting with '[' - extract until matching ']'
        if stripped.startswith("["):
            depth = 0
            for i, c in enumerate(stripped):
                if c == "[":
                    depth += 1
                elif c == "]":
                    depth -= 1
                    if depth == 0:
                        list_part = stripped[: i + 1]
                        try:
                            return ast.parse(list_part)
                        except SyntaxError:
                            break

        # Case 2: Tuple starting with '(' - extract until matching ')'
        if stripped.startswith("("):
            depth = 0
            for i, c in enumerate(stripped):
                if c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
                    if depth == 0:
                        tuple_part = stripped[: i + 1]
                        try:
                            return ast.parse(tuple_part)
                        except SyntaxError:
                            break

        # Case 3: Try wrapping in an assignment
        try:
            return ast.parse(f"__wrapper__ = {stripped}")
        except SyntaxError:
            pass

        return None

    def _collect_lambda_info(self, node: ast.Lambda) -> tuple:
        """Collect parameter names and used names from a lambda."""
        params = {arg.arg for arg in node.args.args}

        class NameCollector(ast.NodeVisitor):
            def __init__(self):
                self.used: Set[str] = set()

            def visit_Name(self, n):
                self.used.add(n.id)

        collector = NameCollector()
        collector.visit(node.body)

        return params, collector.used, set()

    def _visit_lambda(self, node: ast.Lambda, func: Callable) -> str:
        """Visit a lambda expression and generate WGSL."""
        import uuid

        # Lambda doesn't have a name, generate one
        name = f"user_func_{uuid.uuid4().hex[:8]}"
        params = [arg.arg for arg in node.args.args]

        # Collect external variables
        param_set, used_names, assigned_names = self._collect_lambda_info(node)
        self._capture_external_vars(func, param_set, used_names, assigned_names)

        # Build parameter list with types (all f32)
        param_list = ", ".join(f"{p}: f32" for p in params)

        body_lines = []
        for var_name, var_value in self.external_vars.items():
            body_lines.append(f"const {var_name}: f32 = {var_value};")

        # Lambda body is a single expression
        body_expr = self._visit_expression(node.body)

        # If the expression is a comparison (returns bool), convert to f32
        if self._is_boolean_expression(node.body):
            body_expr = f"select(0.0, 1.0, {body_expr})"

        body_lines.append(f"return {body_expr};")

        body_str = "\n    ".join(body_lines)

        return f"fn {name}({param_list}) -> f32 {{\n    {body_str}\n}}"

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
            return self._visit_name(node)
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return "true" if node.value else "false"
            elif isinstance(node.value, (int, float)):
                return str(float(node.value))
            else:
                raise TranspilerError(f"Unsupported constant type: {type(node.value)}")
        elif isinstance(node, ast.Constant):
            return str(float(node.value))
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
        elif isinstance(node, ast.BoolOp):
            return self._visit_boolop(node)
        elif isinstance(node, ast.Attribute):
            return self._visit_attribute(node)
        else:
            raise TranspilerError(f"Unsupported expression type: {type(node).__name__}")

    def _visit_name(self, node: ast.Name) -> str:
        """Visit a Name node, checking for imported constants."""
        name = node.id
        if name in self.imports:
            full_name = self.imports[name]
            parts = full_name.split(".")
            if len(parts) == 2:
                module, const = parts
                if (module, const) in self.CONSTANTS_MAP:
                    return self.CONSTANTS_MAP[(module, const)]
        return name

    def _visit_attribute(self, node: ast.Attribute) -> str:
        """Visit an Attribute node (e.g., math.pi, np.e)."""
        if isinstance(node.value, ast.Name):
            module_alias = node.value.id
            const_name = node.attr
            if module_alias in self.module_aliases:
                module_name = self.module_aliases[module_alias]
                key = (module_name, const_name)
                if key in self.CONSTANTS_MAP:
                    return self.CONSTANTS_MAP[key]
                else:
                    raise TranspilerError(
                        f"Unknown constant: {module_alias}.{const_name}. "
                        f"Available constants: {', '.join(f'{m}.{c}' for m, c in self.CONSTANTS_MAP.keys())}"
                    )
            elif module_alias in self.imports:
                full_import = self.imports[module_alias]
                module_name = full_import.split(".")[-1]
                key = (module_name, const_name)
                if key in self.CONSTANTS_MAP:
                    return self.CONSTANTS_MAP[key]
            else:
                raise TranspilerError(
                    f"Unsupported module: {module_alias}. "
                    f"Supported modules: {', '.join(sorted(set(self.module_aliases.values()) | set(k.split('.')[0] for k in self.imports.values())))}"
                )
        raise TranspilerError(f"Unsupported attribute access: {node.attr}")

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
        func_name = None

        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in self.imports:
                full_import = self.imports[func_name]
                func_name = full_import.split(".")[-1]
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                module_name = node.func.value.id
                if module_name in self.module_aliases:
                    func_name = node.func.attr
                elif module_name in self.imports:
                    full_import = self.imports[module_name]
                    func_name = full_import.split(".")[-1]
                else:
                    raise TranspilerError(
                        f"Unsupported module: {module_name}. "
                        f"Supported modules: {', '.join(sorted(set(self.module_aliases.values()) | set(k.split('.')[0] for k in self.imports.values())))}"
                    )
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

    def _visit_boolop(self, node: ast.BoolOp) -> str:
        """Visit a boolean operation (and/or)."""
        values = [self._visit_expression(v) for v in node.values]

        if isinstance(node.op, ast.And):
            return "(" + " && ".join(values) + ")"
        elif isinstance(node.op, ast.Or):
            return "(" + " || ".join(values) + ")"
        else:
            raise TranspilerError(
                f"Unsupported boolean operator: {type(node.op).__name__}"
            )


def transpile_function(func: Callable) -> str:
    """Convenience function to transpile a Python function to WGSL."""
    transpiler = PythonToWGSL()
    return transpiler.transpile(func)
