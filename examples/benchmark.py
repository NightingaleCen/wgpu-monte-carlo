import wgpu_montecarlo as wmc
import numpy as np
import time
from matplotlib import pyplot as plt
from numpy import exp, sin, cos


def f1(x):
    b = exp(sin(x)) + cos(exp(x))
    return x / b


SAMPLE_SIZES = [1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000]

functions = [f1]

gpu_times = []
manual_times = []
numpy_times = []

integrator = wmc.MonteCarloIntegrator()

# Warm up GPU
integrator.integrate(functions, wmc.Distribution.normal(0.0, 1.0), n_samples=1000)


for N_SAMPLES in SAMPLE_SIZES:
    print(f"\n{'=' * 60}")
    print(f"Testing with {N_SAMPLES:,} samples")
    print(f"{'=' * 60}")

    # GPU-accelerated integration
    start_gpu = time.time()
    result = integrator.integrate(
        functions, wmc.Distribution.normal(0.0, 1.0), n_samples=N_SAMPLES
    )
    end_gpu = time.time()
    gpu_time = end_gpu - start_gpu
    gpu_times.append(gpu_time)
    print(f"GPU Monte Carlo results: {result.values}")
    print(f"GPU execution time: {gpu_time:.6f} seconds")

    # Simple Monte Carlo integration using for loop
    start_manual = time.time()
    manual_sums = [0.0 for _ in functions]
    for i in range(N_SAMPLES):
        x = np.random.normal(0.0, 1.0)
        for j, func in enumerate(functions):
            manual_sums[j] += func(x)
    manual_means = [s / N_SAMPLES for s in manual_sums]
    end_manual = time.time()
    manual_time = end_manual - start_manual
    manual_times.append(manual_time)
    print(f"Manual Monte Carlo results: {manual_means}")
    print(f"Manual execution time: {manual_time:.6f} seconds")

    # NumPy vectorized Monte Carlo integration
    start_numpy = time.time()
    x_samples = np.random.normal(0.0, 1.0, N_SAMPLES)
    numpy_means = []
    for func in functions:
        values = np.array([func(x) for x in x_samples])
        numpy_means.append(np.mean(values))
    end_numpy = time.time()
    numpy_time = end_numpy - start_numpy
    numpy_times.append(numpy_time)
    print(f"NumPy Monte Carlo results: {numpy_means}")
    print(f"NumPy execution time: {numpy_time:.6f} seconds")

    # Comparison
    print(f"\nSpeedup (GPU vs Manual): {manual_time / gpu_time:.2f}x")
    print(f"Speedup (GPU vs NumPy): {numpy_time / gpu_time:.2f}x")

plt.figure(figsize=(8, 6), dpi=100, layout="constrained")
plt.loglog(SAMPLE_SIZES, gpu_times, "o-", label="GPU", linewidth=2, markersize=8)
plt.loglog(
    SAMPLE_SIZES,
    manual_times,
    "s-",
    label="Manual (for loop)",
    linewidth=2,
    markersize=8,
)
plt.loglog(SAMPLE_SIZES, numpy_times, "^-", label="NumPy", linewidth=2, markersize=8)

plt.xlabel("Number of Samples", fontsize=12)
plt.ylabel("Execution Time (seconds)", fontsize=12)
plt.title("Monte Carlo Integration Performance Comparison", fontsize=14)
plt.legend(fontsize=11)
plt.show()
