from enum import Enum, auto
from typing import List, Callable, Optional, Union
import numpy as np
from .transpiler import PythonToWGSL, transpile_function, TranspilerError

"""WGPU Monte Carlo - GPU-accelerated Monte Carlo simulations.

This library provides GPU-accelerated 1D Monte Carlo simulations using
WebGPU (wgpu) for compute shaders and PyO3 for Python bindings.

Example:
    >>> from wgpu_montecarlo import monte_carlo
    >>> import numpy as np
    >>>
    >>> def step(x, rng):
    ...     return x + (rng - 0.5) * 0.1
    ...
    >>> initial = np.zeros(1000, dtype=np.float32)
    >>> result = monte_carlo(initial, step, iterations=100, seed=42)
"""

"""WGPU Monte Carlo - GPU-accelerated Monte Carlo simulations and integration.

This library provides GPU-accelerated Monte Carlo simulations and integration using
WebGPU (wgpu) for compute shaders and PyO3 for Python bindings.

Example (Simulation Mode):
    >>> from wgpu_montecarlo import monte_carlo
    >>> import numpy as np
    >>>
    >>> def step(x, rng):
    ...     return x + (rng - 0.5) * 0.1
    ...
    >>> initial = np.zeros(1000, dtype=np.float32)
    >>> result = monte_carlo(initial, step, iterations=100, seed=42)

Example (Integration Mode):
    >>> from wgpu_montecarlo import MonteCarloIntegrator, Distribution
    >>>
    >>> # Calculate E[X], E[X²], E[exp(X)] for X ~ Normal(0, 1)
    >>> integrator = MonteCarloIntegrator()
    >>> dist = Distribution.normal(mean=0.0, std=1.0)
    >>> funcs = [lambda x: x, lambda x: x**2, lambda x: __import__('math').exp(x)]
    >>> result = integrator.integrate(funcs, dist, n_samples=10_000_000)
    >>> print(f"E[X] = {result.values[0]:.6f}")  # ~0.0
    >>> print(f"E[X²] = {result.values[1]:.6f}")  # ~1.0
"""

try:
    from ._core import (
        MonteCarloSimulator as _RustMonteCarloSimulator,
        MonteCarloIntegrator as _RustMonteCarloIntegrator,
    )

    HAS_RUST_EXTENSION = True
except ImportError:
    HAS_RUST_EXTENSION = False
    _RustMonteCarloSimulator = None
    _RustMonteCarloIntegrator = None

__version__ = "0.1.0"

__all__ = [
    "MonteCarloSimulator",
    "MonteCarloIntegrator",
    "Distribution",
    "IntegrationResult",
    "PythonToWGSL",
    "transpile_function",
    "TranspilerError",
    "monte_carlo",
    "integrate",
]


class MonteCarloSimulator:
    """GPU-accelerated Monte Carlo simulator for time evolution.

    This class provides GPU-accelerated 1D Monte Carlo simulations using
    WebGPU (wgpu) for compute shaders. It supports user-defined step
    functions written in Python (automatically transpiled to WGSL) or
    raw WGSL code.

    Key Features:
        - Automatic transpilation of Python functions to WGSL
        - Support for custom WGSL code strings
        - Efficient GPU-accelerated parallel execution
        - Configurable iteration count and random seed

    Example:
        >>> from wgpu_montecarlo import MonteCarloSimulator
        >>> import numpy as np
        >>>
        >>> # Define step function (will be transpiled to WGSL)
        >>> def step(x, rng):
        ...     return x * 0.99 + (rng - 0.5) * 0.1
        >>>
        >>> simulator = MonteCarloSimulator()
        >>> initial = np.zeros(10000, dtype=np.float32)
        >>> result = simulator.run(initial, step, iterations=1000, seed=42)
        >>> print(f"Mean final value: {result.mean():.6f}")
    """

    def __init__(self):
        """Initialize the Monte Carlo simulator.

        Raises:
            ImportError: If the Rust extension is not built.
        """
        if not HAS_RUST_EXTENSION:
            raise ImportError(
                "The Rust extension is not built. Please run: maturin develop --uv"
            )

        self._simulator = _RustMonteCarloSimulator()
        self._wgsl_code = None

    def set_step_function(self, step_fn: Union[Callable, str]) -> "MonteCarloSimulator":
        """Set the step function for the simulation.

        The step function defines how each particle evolves from one iteration
        to the next. It should have the signature: step(x: f32, rng: f32) -> f32

        Args:
            step_fn: Python function or WGSL code string defining the step.
                    If callable, it will be transpiled to WGSL automatically.
                    If string, it's assumed to be valid WGSL code.

        Returns:
            self (for method chaining)

        Raises:
            TypeError: If step_fn is neither callable nor a string.

        Example:
            >>> simulator = MonteCarloSimulator()
            >>>
            >>> # Using a Python function
            >>> def my_step(x, rng):
            ...     return x + (rng - 0.5) * 0.1
            >>> simulator.set_step_function(my_step)
        """
        if callable(step_fn):
            transpiler = PythonToWGSL()
            self._wgsl_code = transpiler.transpile(step_fn)
        elif isinstance(step_fn, str):
            self._wgsl_code = step_fn
        else:
            raise TypeError("step_fn must be callable or WGSL string")

        return self

    def run(
        self,
        initial: np.ndarray,
        step_fn: Optional[Union[Callable, str]] = None,
        iterations: int = 1000,
        seed: int = 42,
    ) -> np.ndarray:
        """Run the Monte Carlo simulation.

        Args:
            initial: numpy array of float32 values (initial state)
            step_fn: Optional Python function or WGSL code string. If provided,
                    it overrides any step function set via set_step_function().
                    If None, uses the default random walk behavior.
            iterations: Number of Monte Carlo iterations to run (default: 1000)
            seed: Random number generator seed (default: 42)

        Returns:
            numpy array of float32 values (final state)

        Raises:
            RuntimeError: If GPU initialization or execution fails
            TypeError: If initial is not a numpy array of float32

        Example:
            >>> simulator = MonteCarloSimulator()
            >>> initial = np.zeros(10000, dtype=np.float32)
            >>>
            >>> # Run with explicit step function
            >>> result = simulator.run(initial, step_fn=lambda x, r: x + (r - 0.5) * 0.1)
            >>>
            >>> # Or using pre-configured step function
            >>> simulator.set_step_function(lambda x, r: x * 0.99 + (r - 0.5) * 0.1)
            >>> result = simulator.run(initial, iterations=5000)
        """
        import numpy as np

        # Ensure input is float32
        if initial.dtype != np.float32:
            initial = initial.astype(np.float32)

        # Handle step function
        wgsl_code = self._wgsl_code
        if step_fn is not None:
            if callable(step_fn):
                transpiler = PythonToWGSL()
                wgsl_code = transpiler.transpile(step_fn)
            elif isinstance(step_fn, str):
                wgsl_code = step_fn
            else:
                raise TypeError("step_fn must be callable or WGSL string")

        # Set the step function if provided
        if wgsl_code is not None:
            self._simulator.set_user_function(wgsl_code)

        # Run simulation
        return self._simulator.run(initial, iterations, seed)

    def run_with_wgsl(
        self,
        initial: np.ndarray,
        wgsl_function: str,
        iterations: int = 1000,
        seed: int = 42,
    ) -> np.ndarray:
        """Run simulation with explicit WGSL function code.

        This is a lower-level interface that accepts raw WGSL code.

        Args:
            initial: numpy array of float32 values
            wgsl_function: WGSL function code as string
            iterations: Number of iterations (default: 1000)
            seed: RNG seed (default: 42)

        Returns:
            numpy array of final values

        Example:
            >>> simulator = MonteCarloSimulator()
            >>> wgsl = "fn step(x: f32, rng: f32) -> f32 { return x + (rng - 0.5) * 0.1; }"
            >>> initial = np.zeros(1000, dtype=np.float32)
            >>> result = simulator.run_with_wgsl(initial, wgsl, iterations=500)
        """
        import numpy as np

        if initial.dtype != np.float32:
            initial = initial.astype(np.float32)

        return self._simulator.run_with_function(
            initial, iterations, seed, wgsl_function
        )


def monte_carlo(initial, step_fn=None, iterations=1000, seed=42):
    """
    Run a GPU-accelerated 1D Monte Carlo simulation.

    Args:
        initial: numpy array of float32 values (initial state)
        step_fn: Python function or WGSL code string defining the step(x, rng) -> f32
                If None, uses default random walk.
        iterations: Number of Monte Carlo iterations to run
        seed: Random number generator seed

    Returns:
        numpy array of float32 values (final state)

    Raises:
        RuntimeError: If GPU initialization or execution fails
        ImportError: If the Rust extension is not built

    Example:
        >>> import numpy as np
        >>> from wgpu_montecarlo import monte_carlo
        >>>
        >>> # Define step function (will be transpiled to WGSL)
        >>> def step(x, rng):
        ...     # Simple mean-reverting random walk
        ...     return x * 0.99 + (rng - 0.5) * 0.1
        ...
        >>> initial = np.zeros(10000, dtype=np.float32)
        >>> result = monte_carlo(initial, step, iterations=1000)
    """
    if not HAS_RUST_EXTENSION:
        raise ImportError(
            "The Rust extension is not built. Please run: maturin develop --uv"
        )

    import numpy as np

    # Ensure input is float32
    if initial.dtype != np.float32:
        initial = initial.astype(np.float32)

    # Create simulator and run
    simulator = MonteCarloSimulator()
    return simulator.run(initial, step_fn, iterations, seed)


def simulate(initial, wgsl_function=None, iterations=1000, seed=42):
    """
    Run simulation with explicit WGSL function code.

    This is a lower-level interface that accepts raw WGSL code.

    Args:
        initial: numpy array of float32 values
        wgsl_function: WGSL function code as string (fn step(x: f32, rng: f32) -> f32)
        iterations: Number of iterations
        seed: RNG seed

    Returns:
        numpy array of final values
    """
    if not HAS_RUST_EXTENSION:
        raise ImportError(
            "The Rust extension is not built. Please run: maturin develop --uv"
        )

    import numpy as np

    if initial.dtype != np.float32:
        initial = initial.astype(np.float32)

    simulator = MonteCarloSimulator()

    return simulator.run_with_wgsl(initial, wgsl_function or "", iterations, seed)


# ============================================================================
# Monte Carlo Integration (Expected Value Calculation)
# ============================================================================


class DistributionType(Enum):
    """Supported probability distributions for Monte Carlo integration."""

    UNIFORM = auto()
    NORMAL = auto()
    EXPONENTIAL = auto()
    CUSTOM = auto()


def _find_support(
    pdf: Callable,
    threshold_ratio: float = 1e-5,
    max_hard_limit: float = 10000.0,
) -> tuple:
    """Automatically detect the effective support of a PDF.

    Algorithm: Locate - Peak Find - Expand

    Phase 1 (Locate):
        - Dense coverage [-4, 4] with step 0.5: catches bounded distributions like Beta
        - Exponential coverage [-1024, 1024]: handles shifted/heavy-tailed distributions

    Phase 2 (Peak Find): Local hill climbing to find peak
    Phase 3 (Expand): Expand from peak until pdf drops below threshold

    Args:
        pdf: PDF function
        threshold_ratio: Relative truncation threshold (relative to peak)
        max_hard_limit: Safety hard limit for expansion

    Returns:
        (x_min, x_max): Support boundaries

    Raises:
        ValueError: If PDF is zero everywhere in scanned range
    """
    import math

    # Search grid design:
    # - Dense coverage [-4, 4] with step 0.5: catches bounded distributions like Beta
    # - Exponential coverage [-1024, 1024]: handles shifted/heavy-tailed distributions
    points = set()
    for i in range(-8, 9):
        points.add(i * 0.5)
    for exp in range(4, 11):
        points.add(2**exp)
        points.add(-(2**exp))
    scan_points = sorted(points)

    first_nonzero_x = None
    first_nonzero_val = None
    for x in scan_points:
        try:
            val = pdf(x)
            if val > 0 and math.isfinite(val):
                first_nonzero_x = x
                first_nonzero_val = val
                break
        except (ValueError, TypeError, OverflowError):
            continue

    if first_nonzero_x is None:
        raise ValueError(
            "PDF is zero everywhere in scanned range [-4, 4] (step=0.5) and [-1024, 1024] (exponential).\n"
            "This may happen if your distribution is:\n"
            "  - Bounded and located outside [-4, 4] (e.g., Uniform(10, 10.1))\n"
            "  - Heavily shifted (e.g., N(1000, 1)) but not detected by exponential scan\n\n"
            "Solution: Manually specify the support parameter:\n"
            "  dist = Distribution.from_pdf(your_pdf, support=(x_min, x_max))\n\n"
            "Example for Uniform(5, 10):\n"
            "  def my_pdf(x):\n"
            "      return 0.2 if 5 <= x < 10 else 0.0\n"
            "  dist = Distribution.from_pdf(my_pdf, support=(5.0, 10.0))"
        )

    peak_x = first_nonzero_x
    peak_val = first_nonzero_val

    step_size = 1.0
    max_climb_iterations = 100
    for _ in range(max_climb_iterations):
        left_val = (
            pdf(peak_x - step_size) if peak_x - step_size > -max_hard_limit else 0
        )
        right_val = (
            pdf(peak_x + step_size) if peak_x + step_size < max_hard_limit else 0
        )

        if left_val > peak_val:
            peak_x = peak_x - step_size
            peak_val = left_val
        elif right_val > peak_val:
            peak_x = peak_x + step_size
            peak_val = right_val
        else:
            step_size /= 2
            if step_size < 1e-6:
                break

    threshold = peak_val * threshold_ratio

    x_min = peak_x
    step = 0.1
    while x_min > -max_hard_limit:
        try:
            val = pdf(x_min - step)
            if val <= 0 or val < threshold:
                x_min = x_min - step
                break
            x_min = x_min - step
            step *= 2
        except (ValueError, TypeError, OverflowError):
            break

    x_max = peak_x
    step = 0.1
    while x_max < max_hard_limit:
        try:
            val = pdf(x_max + step)
            if val <= 0 or val < threshold:
                x_max = x_max + step
                break
            x_max = x_max + step
            step *= 2
        except (ValueError, TypeError, OverflowError):
            break

    return x_min, x_max


def _compute_cdf_table(
    pdf: Callable,
    x_min: float,
    x_max: float,
    n_points: int = 2048,
) -> tuple:
    """Compute normalized CDF lookup table on support.

    Uses trapezoidal rule for numerical integration and enforces
    normalization to ensure CDF endpoint is exactly 1.0.

    Args:
        pdf: PDF function
        x_min, x_max: Support boundaries
        n_points: Number of grid points (minimum 1000)

    Returns:
        (x_grid, cdf_values): Normalized CDF lookup tables

    Raises:
        ValueError: If PDF integral is zero
    """
    n_points = max(n_points, 1000)

    x_grid = np.linspace(x_min, x_max, n_points)
    pdf_values = np.array([pdf(x) for x in x_grid])

    pdf_values = np.nan_to_num(pdf_values, nan=0.0, posinf=0.0, neginf=0.0)
    pdf_values = np.clip(pdf_values, 0, None)

    dx = (x_max - x_min) / (n_points - 1)
    cdf_values = np.zeros(n_points)
    cdf_values[1:] = np.cumsum((pdf_values[:-1] + pdf_values[1:]) / 2) * dx
    cdf_values[0] = 0.0

    total = cdf_values[-1]
    if total <= 0:
        raise ValueError(
            "PDF integral is zero. Please check the PDF function or support range."
        )
    cdf_values = cdf_values / total

    return x_grid, cdf_values


class Distribution:
    """Configuration for a probability distribution.

    This class provides factory methods for creating different probability
    distributions that can be used with MonteCarloIntegrator.

    All distributions provide a unified pdf(x) interface for importance sampling.

    Examples:
        >>> # Uniform distribution U(0, 1)
        >>> dist = Distribution.uniform(min=0.0, max=1.0)

        >>> # Normal distribution N(0, 1)
        >>> dist = Distribution.normal(mean=0.0, std=1.0)

        >>> # Exponential distribution with lambda=2.0
        >>> dist = Distribution.exponential(lambda_param=2.0)

        >>> # Custom distribution from PDF function
        >>> import math
        >>> def my_pdf(x):
        ...     return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
        >>> dist = Distribution.from_pdf(my_pdf)

        >>> # Beta distribution (convenience method)
        >>> dist = Distribution.beta(alpha=2.0, beta_param=5.0)
    """

    def __init__(
        self,
        dist_type: DistributionType,
        params: dict,
        pdf_func: Callable[[float], float],
        x_table: Optional[np.ndarray] = None,
        cdf_table: Optional[np.ndarray] = None,
    ):
        self.dist_type = dist_type
        self.params = params
        self._pdf_func = pdf_func
        self._x_table = x_table
        self._cdf_table = cdf_table

    def pdf(self, x: float) -> float:
        """Evaluate PDF at point x."""
        return self._pdf_func(x)

    @staticmethod
    def uniform(min: float = 0.0, max: float = 1.0) -> "Distribution":
        """Create uniform distribution U(min, max).

        Uses analytical sampling (linear transformation).

        Args:
            min: Minimum value (inclusive)
            max: Maximum value (exclusive)

        Returns:
            Distribution configured for uniform sampling
        """
        width = max - min

        def pdf(x: float) -> float:
            return 1.0 / width if min <= x < max else 0.0

        return Distribution(
            dist_type=DistributionType.UNIFORM,
            params={"min": min, "max": max},
            pdf_func=pdf,
        )

    @staticmethod
    def normal(mean: float = 0.0, std: float = 1.0) -> "Distribution":
        """Create normal distribution N(mean, std).

        Uses Box-Muller transform on GPU for high-quality normal samples.

        Args:
            mean: Distribution mean
            std: Standard deviation

        Returns:
            Distribution configured for normal sampling
        """
        import math

        sqrt_2pi = math.sqrt(2 * math.pi)

        def pdf(x: float) -> float:
            z = (x - mean) / std
            return math.exp(-0.5 * z * z) / (std * sqrt_2pi)

        return Distribution(
            dist_type=DistributionType.NORMAL,
            params={"mean": mean, "std": std},
            pdf_func=pdf,
        )

    @staticmethod
    def exponential(lambda_param: float = 1.0) -> "Distribution":
        """Create exponential distribution Exp(lambda).

        Uses inverse transform sampling on GPU.

        Args:
            lambda_param: Rate parameter (1/mean)

        Returns:
            Distribution configured for exponential sampling
        """
        import math

        def pdf(x: float) -> float:
            return lambda_param * math.exp(-lambda_param * x) if x >= 0 else 0.0

        return Distribution(
            dist_type=DistributionType.EXPONENTIAL,
            params={"lambda": lambda_param},
            pdf_func=pdf,
        )

    @staticmethod
    def from_pdf(
        pdf_func: Callable[[float], float],
        support: Optional[tuple] = None,
        table_size: int = 2048,
    ) -> "Distribution":
        """Create custom distribution from PDF function.

        Automatically detects support and generates CDF lookup table.

        Args:
            pdf_func: PDF function accepting float, returning float
            support: Optional (x_min, x_max) tuple to skip auto-detection
            table_size: Number of points in lookup table (default: 2048)

        Returns:
            Distribution configured for table-based sampling

        Raises:
            ValueError: If PDF is invalid or integral is zero

        Example:
            >>> import math
            >>> def my_pdf(x):
            ...     return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
            >>> dist = Distribution.from_pdf(my_pdf)
        """
        if not callable(pdf_func):
            raise TypeError("pdf_func must be callable")

        if support is not None:
            x_min, x_max = support
        else:
            x_min, x_max = _find_support(pdf_func)

        x_table, cdf_table = _compute_cdf_table(pdf_func, x_min, x_max, table_size)
        actual_size = len(x_table)

        return Distribution(
            dist_type=DistributionType.CUSTOM,
            params={"table_size": actual_size, "support": (x_min, x_max)},
            pdf_func=pdf_func,
            x_table=x_table.astype(np.float32),
            cdf_table=cdf_table.astype(np.float32),
        )

    @staticmethod
    def beta(alpha: float, beta_param: float, table_size: int = 2048) -> "Distribution":
        """Create Beta distribution from PDF function.

        Uses from_pdf internally with known support [0, 1].

        Args:
            alpha: First shape parameter
            beta_param: Second shape parameter
            table_size: Number of points in lookup table (default: 2048)

        Returns:
            Distribution configured for Beta sampling

        Raises:
            ImportError: If scipy is not installed
        """
        try:
            from scipy.special import beta as beta_fn

            B = beta_fn(alpha, beta_param)

            def pdf(x: float) -> float:
                if 0 < x < 1:
                    return (x ** (alpha - 1)) * ((1 - x) ** (beta_param - 1)) / B
                return 0.0

            return Distribution.from_pdf(pdf, support=(0.0, 1.0), table_size=table_size)
        except ImportError:
            raise ImportError(
                "scipy is required for Beta distribution. Install with: pip install scipy"
            )


class IntegrationResult:
    """Results from Monte Carlo integration.

    This class holds the expected values computed by MonteCarloIntegrator.
    Values are ordered according to the function list passed to integrate().

    Attributes:
        values: numpy array of expected values (one per function)
        n_samples: Total number of Monte Carlo samples used
        n_functions: Number of functions integrated

    Example:
        >>> result = integrator.integrate([f1, f2, f3], dist, n_samples=1e7)
        >>> print(f"E[f1(X)] = {result.values[0]:.6f}")
        >>> print(f"E[f2(X)] = {result.values[1]:.6f}")
        >>> print(f"E[f3(X)] = {result.values[2]:.6f}")
    """

    def __init__(self, values: np.ndarray, n_samples: int, n_functions: int):
        self.values = np.array(
            values, dtype=np.float64
        )  # Use double precision for results
        self.n_samples = n_samples
        self.n_functions = n_functions

    def __repr__(self):
        return f"IntegrationResult(values={self.values}, n_samples={self.n_samples})"

    def __getitem__(self, idx):
        """Allow indexing: result[0] returns E[f_0(X)]"""
        return self.values[idx]

    def __len__(self):
        """Return number of functions"""
        return self.n_functions


class MonteCarloIntegrator:
    """GPU-accelerated Monte Carlo integrator for expected value calculation.

    This class supports fusing multiple functions into a single GPU pass for
    efficient computation of E[f_1(X)], E[f_2(X)], ..., E[f_K(X)].

    Key Features:
        - Fused multi-function evaluation (all functions evaluated on same samples)
        - Native GPU sampling (Uniform, Normal, Exponential, Table-based)
        - Smart workload partitioning (~65k threads, configurable)
        - CPU-based reduction for final expected values

    Example:
        >>> from wgpu_montecarlo import MonteCarloIntegrator, Distribution
        >>>
        >>> integrator = MonteCarloIntegrator()
        >>> dist = Distribution.normal(mean=0.0, std=1.0)
        >>>
        >>> # Define functions to integrate
        >>> funcs = [
        ...     lambda x: x,           # E[X]
        ...     lambda x: x**2,        # E[X²]
        ...     lambda x: x**4,        # E[X⁴]
        ... ]
        >>>
        >>> result = integrator.integrate(funcs, dist, n_samples=10_000_000)
        >>> print(f"Variance = {result.values[1] - result.values[0]**2:.6f}")  # Should be ~1.0
    """

    def __init__(self, target_threads: Optional[int] = None):
        """Initialize the Monte Carlo integrator.

        Args:
            target_threads: Target number of GPU threads (default: 65536).
                          Increase for larger GPUs, decrease for smaller.
        """
        if not HAS_RUST_EXTENSION:
            raise ImportError(
                "The Rust extension is not built. Please run: maturin develop --uv"
            )

        self._integrator = _RustMonteCarloIntegrator()
        self._target_threads = target_threads

    def integrate(
        self,
        functions: List[Union[Callable, str]],
        distribution: Distribution,
        n_samples: int = 1_000_000,
        seed: int = 42,
    ) -> IntegrationResult:
        """Compute expected values E[f(X)] for multiple functions.

        This method evaluates all functions on the same random samples,
        minimizing memory bandwidth and RNG overhead.

        Args:
            functions: List of Python callables or WGSL code strings.
                      Each function should accept a single f32 argument.
            distribution: Distribution to sample from (use Distribution factory methods).
            n_samples: Total number of Monte Carlo samples (default: 1_000_000).
            seed: Random seed for reproducibility (default: 42).

        Returns:
            IntegrationResult containing expected values and metadata.

        Raises:
            ImportError: If the Rust extension is not built.
            ValueError: If functions list is empty or distribution is invalid.
            RuntimeError: If GPU execution fails.

        Example:
            >>> integrator = MonteCarloIntegrator()
            >>> dist = Distribution.normal(mean=0.0, std=1.0)
            >>>
            >>> # Compute mean and variance in one pass
            >>> funcs = [lambda x: x, lambda x: x**2]
            >>> result = integrator.integrate(funcs, dist, n_samples=10_000_000)
            >>>
            >>> mean = result.values[0]
            >>> variance = result.values[1] - mean**2
            >>> print(f"Mean = {mean:.6f}, Variance = {variance:.6f}")
        """
        if len(functions) == 0:
            raise ValueError("At least one function is required")

        # Transpile Python functions to WGSL
        wgsl_functions = []
        for func in functions:
            if callable(func):
                wgsl_code = transpile_function(func)
                wgsl_functions.append(wgsl_code)
            elif isinstance(func, str):
                # Assume it's already WGSL code
                wgsl_functions.append(func)
            else:
                raise TypeError(
                    f"Function must be callable or WGSL string, got {type(func)}"
                )

        # Prepare CDF tables for custom distributions
        x_table = None
        cdf_table = None
        if distribution.dist_type == DistributionType.CUSTOM:
            if distribution._x_table is not None:
                x_table = distribution._x_table
            if distribution._cdf_table is not None:
                cdf_table = distribution._cdf_table

        # Convert distribution type to string
        dist_type_str = distribution.dist_type.name.lower()

        # Call Rust backend
        values = self._integrator.integrate(
            wgsl_functions,
            dist_type_str,
            distribution.params,
            n_samples,
            seed,
            x_table,
            cdf_table,
            self._target_threads,
        )

        return IntegrationResult(
            values=values,
            n_samples=n_samples,
            n_functions=len(functions),
        )


def integrate(
    functions: List[Union[Callable, str]],
    distribution: Distribution,
    n_samples: int = 1_000_000,
    seed: int = 42,
    target_threads: Optional[int] = None,
) -> IntegrationResult:
    """Convenience function for Monte Carlo integration.

    This is a shorthand for creating a MonteCarloIntegrator and calling integrate().

    Args:
        functions: List of Python callables or WGSL code strings.
        distribution: Distribution to sample from.
        n_samples: Total number of Monte Carlo samples.
        seed: Random seed.
        target_threads: Optional target thread count (default: 65536).

    Returns:
        IntegrationResult containing expected values.

    Example:
        >>> from wgpu_montecarlo import integrate, Distribution
        >>>
        >>> # Calculate E[X] and E[X²] for standard normal
        >>> result = integrate(
        ...     [lambda x: x, lambda x: x**2],
        ...     Distribution.normal(0.0, 1.0),
        ...     n_samples=10_000_000
        ... )
        >>> print(f"E[X] = {result[0]:.6f}")  # ~0.0
        >>> print(f"E[X²] = {result[1]:.6f}")  # ~1.0
    """
    integrator = MonteCarloIntegrator(target_threads=target_threads)
    return integrator.integrate(functions, distribution, n_samples, seed)
