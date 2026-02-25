from enum import Enum, auto
from typing import Callable, List, Optional, Tuple, Union
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
        MonteCarloIntegrator as _RustMonteCarloIntegrator,
    )

    HAS_RUST_EXTENSION = True
except ImportError:
    HAS_RUST_EXTENSION = False
    _RustMonteCarloIntegrator = None

__version__ = "0.2.0"

__all__ = [
    "MonteCarloIntegrator",
    "Distribution",
    "IntegrationResult",
    "PythonToWGSL",
    "transpile_function",
    "TranspilerError",
    "integrate",
    "integrate_importance_sampling",
]


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
        pdf_table: Optional[np.ndarray] = None,
    ):
        self.dist_type = dist_type
        self.params = params
        self._pdf_func = pdf_func
        self._x_table = x_table
        self._cdf_table = cdf_table
        self._pdf_table = pdf_table

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
            return 1.0 / width if (min <= x) and (x < max) else 0.0

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

        # "std" is a reserved keyword in WGSL so we rename
        # while keeping the original parameter name for the user.
        sigma = std
        sqrt_2pi = np.sqrt(2 * np.pi)

        def pdf(x: float) -> float:
            z = (x - mean) / sigma
            return np.exp(-0.5 * z * z) / (sigma * sqrt_2pi)

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
    def from_pdf_table(
        x_table: Union[np.ndarray, list],
        pdf_table: Union[np.ndarray, list],
        cdf_table: Optional[Union[np.ndarray, list]] = None,
    ) -> "Distribution":
        """Create distribution from pre-computed PDF lookup table.

        Useful when PDF values are already computed (e.g., from experimental data
        or external numerical libraries).

        Args:
            x_table: Grid points where PDF is evaluated (must be sorted ascending)
            pdf_table: PDF values at each grid point (must be non-negative)
            cdf_table: Optional pre-computed CDF values. If None, CDF is computed
                      by trapezoidal integration of PDF.

        Returns:
            Distribution configured for table-based sampling and PDF lookup

        Raises:
            ValueError: If arrays have invalid shapes, x_table not sorted,
                       or pdf_table contains negative values.

        Example:
            >>> import numpy as np
            >>> x = np.linspace(0, 10, 2048)
            >>> pdf = np.exp(-x)  # Exponential decay
            >>> dist = Distribution.from_pdf_table(x, pdf)
        """
        x_arr = np.asarray(x_table, dtype=np.float32)
        pdf_arr = np.asarray(pdf_table, dtype=np.float32)

        if len(x_arr.shape) != 1 or len(pdf_arr.shape) != 1:
            raise ValueError("x_table and pdf_table must be 1D arrays")

        if len(x_arr) != len(pdf_arr):
            raise ValueError("x_table and pdf_table must have the same length")

        if len(x_arr) < 2:
            raise ValueError("Tables must have at least 2 points")

        if not np.all(np.diff(x_arr) > 0):
            raise ValueError("x_table must be sorted in ascending order")

        if np.any(pdf_arr < 0):
            raise ValueError("pdf_table must contain non-negative values")

        table_size = len(x_arr)
        x_min, x_max = float(x_arr[0]), float(x_arr[-1])

        if cdf_table is not None:
            cdf_arr = np.asarray(cdf_table, dtype=np.float32)
            if len(cdf_arr) != table_size:
                raise ValueError("cdf_table must have same length as x_table")
        else:
            # compute CDF from PDF
            cdf_arr = np.zeros(table_size, dtype=np.float32)
            for i in range(1, table_size):
                dx = x_arr[i] - x_arr[i - 1]
                cdf_arr[i] = cdf_arr[i - 1] + 0.5 * (pdf_arr[i] + pdf_arr[i - 1]) * dx
            if cdf_arr[-1] > 0:
                cdf_arr = cdf_arr / cdf_arr[-1]

        x_min_f, x_max_f = float(x_min), float(x_max)
        pdf_copy = pdf_arr.copy()

        def pdf_func(x: float) -> float:
            if x < x_min_f or x > x_max_f:
                return 0.0
            idx = np.searchsorted(x_arr, x)
            if idx == 0:
                return float(pdf_copy[0])
            if idx >= table_size:
                return float(pdf_copy[-1])
            t = (x - x_arr[idx - 1]) / (x_arr[idx] - x_arr[idx - 1])
            return float((1 - t) * pdf_copy[idx - 1] + t * pdf_copy[idx])

        return Distribution(
            dist_type=DistributionType.CUSTOM,
            params={"table_size": table_size, "support": (x_min, x_max)},
            pdf_func=pdf_func,
            x_table=x_arr,
            cdf_table=cdf_arr,
            pdf_table=pdf_arr,
        )

    def get_or_compute_pdf_table(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (x_table, pdf_table), computing if necessary.

        For distributions created from analytical PDF functions, this method
        computes the PDF values on the x_grid (reusing existing x_table if available).

        Returns:
            Tuple of (x_table, pdf_table) as numpy float32 arrays
        """
        if self._pdf_table is not None and self._x_table is not None:
            return self._x_table, self._pdf_table

        if self._x_table is None:
            support = self.params.get("support", (-5.0, 5.0))
            table_size = self.params.get("table_size", 2048)
            x_min, x_max = support
            self._x_table = np.linspace(x_min, x_max, table_size, dtype=np.float32)

        self._pdf_table = np.array(
            [self._pdf_func(float(x)) for x in self._x_table], dtype=np.float32
        )
        return self._x_table, self._pdf_table


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

    def integrate_importance_sampling(
        self,
        functions: List[Union[Callable, str]],
        target_distribution: Distribution,
        proposal_distribution: Distribution,
        n_samples: int = 1_000_000,
        seed: int = 42,
    ) -> IntegrationResult:
        """Compute E_p[f(X)] using importance sampling.

        Uses proposal distribution q(x) to sample, then applies importance
        weights p(x)/q(x) to estimate expectations under target distribution p.

        Formula: E_p[f(X)] ≈ (1/N) Σ f(x_i) * p(x_i) / q(x_i) where x_i ~ q

        Args:
            functions: List of Python callables or WGSL code strings.
            target_distribution: Target distribution p(x) to compute expectations
                under. Must be analytical (uniform, normal, exponential).
            proposal_distribution: Proposal distribution q(x) to sample from.
                Must be analytical (uniform, normal, exponential).
            n_samples: Total number of Monte Carlo samples (default: 1_000_000).
            seed: Random seed for reproducibility (default: 42).

        Returns:
            IntegrationResult containing E_p[f_i(X)] values.

        Raises:
            ImportError: If the Rust extension is not built.
            NotImplementedError: If target or proposal distribution is custom
                (PDF table-based). This feature is planned for Phase 2.
            ValueError: If functions list is empty or distribution is invalid.
            RuntimeError: If GPU execution fails.

        Example:
            >>> integrator = MonteCarloIntegrator()
            >>> target = Distribution.normal(0.0, 1.0)
            >>> proposal = Distribution.normal(0.5, 1.5)
            >>> result = integrator.integrate_importance_sampling(
            ...     [lambda x: x, lambda x: x**2],
            ...     target, proposal,
            ...     n_samples=10_000_000
            ... )
        """
        if len(functions) == 0:
            raise ValueError("At least one function is required")

        # Try to transpile PDFs
        p_pdf_wgsl = None
        p_can_transpile = True
        try:
            p_pdf_wgsl = transpile_function(target_distribution._pdf_func)
        except TranspilerError:
            p_can_transpile = False

        q_pdf_wgsl = None
        q_can_transpile = True
        try:
            q_pdf_wgsl = transpile_function(proposal_distribution._pdf_func)
        except TranspilerError:
            q_can_transpile = False

        if p_can_transpile and q_can_transpile:
            # if both PDFs are transpilable
            assert p_pdf_wgsl is not None and q_pdf_wgsl is not None
            return self._integrate_is_transpiled(
                functions,
                target_distribution,
                proposal_distribution,
                p_pdf_wgsl,
                q_pdf_wgsl,
                n_samples,
                seed,
            )
        else:
            # if at least one PDF needs table lookup
            return self._integrate_is_with_tables(
                functions,
                target_distribution,
                proposal_distribution,
                p_pdf_wgsl,
                q_pdf_wgsl,
                p_can_transpile,
                q_can_transpile,
                n_samples,
                seed,
            )

    def _integrate_is_transpiled(
        self,
        functions: List[Union[Callable, str]],
        target_distribution: Distribution,
        proposal_distribution: Distribution,
        p_pdf_wgsl: str,
        q_pdf_wgsl: str,
        n_samples: int,
        seed: int,
    ) -> IntegrationResult:
        """IS implementation when both PDFs are transpilable."""
        weighted_wgsls = []
        for i, func in enumerate(functions):
            if callable(func):
                f_wgsl = transpile_function(func)
            elif isinstance(func, str):
                f_wgsl = func
            else:
                raise TypeError(
                    f"Function must be callable or WGSL string, got {type(func)}"
                )

            p_renamed = _rename_wgsl_function(p_pdf_wgsl, f"_is_pdf_p_{i}")
            q_renamed = _rename_wgsl_function(q_pdf_wgsl, f"_is_pdf_q_{i}")
            f_renamed = _rename_wgsl_function(f_wgsl, f"_is_f_orig_{i}")

            wrapper_name = f"_is_wrapper_{i}"
            weighted = f"""
fn {wrapper_name}(x: f32) -> f32 {{
    let f_val = _is_f_orig_{i}(x);
    let p = _is_pdf_p_{i}(x);
    let q = _is_pdf_q_{i}(x);
    return f_val * p / q;
}}

{p_renamed}
{q_renamed}
{f_renamed}
"""
            weighted_wgsls.append(weighted)

        return self.integrate(weighted_wgsls, proposal_distribution, n_samples, seed)

    def _integrate_is_with_tables(
        self,
        functions: List[Union[Callable, str]],
        target_distribution: Distribution,
        proposal_distribution: Distribution,
        p_pdf_wgsl: Optional[str],
        q_pdf_wgsl: Optional[str],
        p_can_transpile: bool,
        q_can_transpile: bool,
        n_samples: int,
        seed: int,
    ) -> IntegrationResult:
        """IS implementation using PDF tables for non-transpilable distributions."""
        weighted_wgsls = []
        target_x_table = None
        target_pdf_table = None
        proposal_x_table = None
        proposal_pdf_table = None

        for i, func in enumerate(functions):
            if callable(func):
                f_wgsl = transpile_function(func)
            elif isinstance(func, str):
                f_wgsl = func
            else:
                raise TypeError(
                    f"Function must be callable or WGSL string, got {type(func)}"
                )

            f_renamed = _rename_wgsl_function(f_wgsl, f"_is_f_orig_{i}")

            # Build PDF calls
            if p_can_transpile and p_pdf_wgsl is not None:
                p_renamed = _rename_wgsl_function(p_pdf_wgsl, f"_is_pdf_p_{i}")
                p_call = f"_is_pdf_p_{i}(x)"
                p_code = p_renamed
            else:
                p_call = "pdf_target_from_table(x)"
                p_code = ""
                # Get PDF table for target
                if target_x_table is None:
                    target_x_table, target_pdf_table = (
                        target_distribution.get_or_compute_pdf_table()
                    )

            if q_can_transpile and q_pdf_wgsl is not None:
                q_renamed = _rename_wgsl_function(q_pdf_wgsl, f"_is_pdf_q_{i}")
                q_call = f"_is_pdf_q_{i}(x)"
                q_code = q_renamed
            else:
                q_call = "pdf_proposal_from_table(x)"
                q_code = ""
                # Get PDF table for proposal
                if proposal_x_table is None:
                    proposal_x_table, proposal_pdf_table = (
                        proposal_distribution.get_or_compute_pdf_table()
                    )

            wrapper_name = f"_is_wrapper_{i}"
            weighted = f"""
fn {wrapper_name}(x: f32) -> f32 {{
    let f_val = _is_f_orig_{i}(x);
    let p = {p_call};
    let q = {q_call};
    return f_val * p / q;
}}

{p_code}
{q_code}
{f_renamed}
"""
            weighted_wgsls.append(weighted)

        # Prepare CDF tables for proposal distribution
        x_table = None
        cdf_table = None
        if proposal_distribution.dist_type == DistributionType.CUSTOM:
            if proposal_distribution._x_table is not None:
                x_table = proposal_distribution._x_table
            if proposal_distribution._cdf_table is not None:
                cdf_table = proposal_distribution._cdf_table

        dist_type_str = proposal_distribution.dist_type.name.lower()

        # Call Rust backend with PDF tables
        values = self._integrator.integrate_is_tables(
            weighted_wgsls,
            dist_type_str,
            proposal_distribution.params,
            n_samples,
            seed,
            x_table,
            cdf_table,
            target_x_table,
            target_pdf_table,
            proposal_x_table,
            proposal_pdf_table,
            self._target_threads,
        )

        return IntegrationResult(
            values=values,
            n_samples=n_samples,
            n_functions=len(functions),
        )


def _rename_wgsl_function(wgsl_code: str, new_name: str) -> str:
    """Helper function to rename a WGSL function definition.

    Args:
        wgsl_code: WGSL function code (e.g., "fn foo(x: f32) -> f32 { ... }")
        new_name: New function name

    Returns:
        WGSL code with renamed function
    """
    import re

    return re.sub(r"fn\s+\w+\s*\(", f"fn {new_name}(", wgsl_code, count=1)


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


def integrate_importance_sampling(
    functions: List[Union[Callable, str]],
    target_distribution: Distribution,
    proposal_distribution: Distribution,
    n_samples: int = 1_000_000,
    seed: int = 42,
    target_threads: Optional[int] = None,
) -> IntegrationResult:
    """Convenience function for importance sampling integration.

    This is a shorthand for creating a MonteCarloIntegrator and calling
    integrate_importance_sampling().

    Args:
        functions: List of Python callables or WGSL code strings.
        target_distribution: Target distribution p(x).
        proposal_distribution: Proposal distribution q(x) to sample from.
        n_samples: Total number of Monte Carlo samples.
        seed: Random seed.
        target_threads: Optional target thread count (default: 65536).

    Returns:
        IntegrationResult containing E_p[f_i(X)] values.

    Example:
        >>> from wgpu_montecarlo import integrate_importance_sampling, Distribution
        >>>
        >>> target = Distribution.normal(0.0, 1.0)
        >>> proposal = Distribution.normal(0.5, 1.5)
        >>> result = integrate_importance_sampling(
        ...     [lambda x: x, lambda x: x**2],
        ...     target, proposal,
        ...     n_samples=10_000_000
        ... )
        >>> print(f"E_p[X] = {result[0]:.6f}")
    """
    integrator = MonteCarloIntegrator(target_threads=target_threads)
    return integrator.integrate_importance_sampling(
        functions, target_distribution, proposal_distribution, n_samples, seed
    )
