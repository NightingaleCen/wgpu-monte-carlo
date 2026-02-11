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
    TABLE = auto()  # Lookup table-based sampling


class Distribution:
    """Configuration for a probability distribution.

    This class provides factory methods for creating different probability
    distributions that can be used with MonteCarloIntegrator.

    Examples:
        >>> # Uniform distribution U(0, 1)
        >>> dist = Distribution.uniform(min=0.0, max=1.0)

        >>> # Normal distribution N(0, 1)
        >>> dist = Distribution.normal(mean=0.0, std=1.0)

        >>> # Exponential distribution with lambda=2.0
        >>> dist = Distribution.exponential(lambda_param=2.0)

        >>> # Table-based distribution (e.g., for Beta, Gamma)
        >>> import numpy as np
        >>> from scipy.stats import beta
        >>> # Precompute inverse CDF for Beta(2, 5)
        >>> probs = np.linspace(0, 1, 2048, endpoint=False)
        >>> probs = np.clip(probs, 1e-7, 1 - 1e-7)
        >>> table = beta.ppf(probs, 2.0, 5.0).astype(np.float32)
        >>> dist = Distribution.from_table(table)
    """

    def __init__(self, dist_type: DistributionType, params: dict):
        self.dist_type = dist_type
        self.params = params

    @staticmethod
    def uniform(min: float = 0.0, max: float = 1.0) -> "Distribution":
        """Create uniform distribution U(min, max).

        Args:
            min: Minimum value (inclusive)
            max: Maximum value (exclusive)

        Returns:
            Distribution configured for uniform sampling
        """
        return Distribution(DistributionType.UNIFORM, {"min": min, "max": max})

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
        return Distribution(DistributionType.NORMAL, {"mean": mean, "std": std})

    @staticmethod
    def exponential(lambda_param: float = 1.0) -> "Distribution":
        """Create exponential distribution Exp(lambda).

        Uses inverse transform sampling on GPU.

        Args:
            lambda_param: Rate parameter (1/mean)

        Returns:
            Distribution configured for exponential sampling
        """
        return Distribution(DistributionType.EXPONENTIAL, {"lambda": lambda_param})

    @staticmethod
    def from_table(table_data: np.ndarray) -> "Distribution":
        """Create distribution from precomputed inverse CDF lookup table.

        This enables arbitrary distributions (Beta, Gamma, empirical, etc.)
        by precomputing the inverse CDF on the CPU using scipy.

        The lookup table should contain quantiles: for each p in [0, 1),
        table[i] = F^{-1}(p) where F is the CDF.

        TODO: Currently uses table size as provided (typically 2048 elements).
        Consider using 4096 or 8192 for higher precision in future versions.

        Args:
            table_data: 1D numpy array of float32 values representing inverse CDF

        Returns:
            Distribution configured for table-based sampling

        Example:
            >>> from scipy.stats import beta
            >>> probs = np.linspace(0, 1, 2048, endpoint=False)
            >>> table = beta.ppf(probs, 2.0, 5.0).astype(np.float32)
            >>> dist = Distribution.from_table(table)
        """
        return Distribution(
            DistributionType.TABLE, {"table": table_data, "table_size": len(table_data)}
        )

    @staticmethod
    def beta(alpha: float, beta_param: float, table_size: int = 2048) -> "Distribution":
        """Create Beta distribution using lookup table method.

        This is a convenience method that automatically generates the lookup
        table using scipy.stats.beta.ppf if scipy is available.

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
            from scipy.stats import beta as beta_dist

            probs = np.linspace(0, 1, table_size, endpoint=False)
            probs = np.clip(probs, 1e-7, 1 - 1e-7)
            table = beta_dist.ppf(probs, alpha, beta_param).astype(np.float32)
            return Distribution.from_table(table)
        except ImportError:
            raise ImportError(
                "scipy is required for automatic Beta distribution generation. "
                "Install with: pip install scipy"
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

        # Prepare lookup table if needed
        lookup_table = None
        if distribution.dist_type == DistributionType.TABLE:
            if "table" in distribution.params:
                lookup_table = distribution.params["table"]

        # Convert distribution type to string
        dist_type_str = distribution.dist_type.name.lower()

        # Call Rust backend
        values = self._integrator.integrate(
            wgsl_functions,
            dist_type_str,
            distribution.params,
            n_samples,
            seed,
            lookup_table,
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
