use numpy::{PyArray1, PyReadonlyArray1, ToPyArray};
use pyo3::prelude::*;

mod distribution;
mod engine;
mod shader_gen;

use distribution::DistributionParams;
use engine::{ComputeEngine, DistributionParamsBuffer};
use shader_gen::generate_compute_shader;
use shader_gen::{generate_integration_shader, IntegrationShaderConfig};

/// Monte Carlo Simulator (Path-Dependent Simulation)
#[pyclass]
struct MonteCarloSimulator {
    engine: ComputeEngine,
    user_function: Option<String>,
}

#[pymethods]
impl MonteCarloSimulator {
    #[new]
    fn new() -> PyResult<Self> {
        let engine = pollster::block_on(ComputeEngine::new()).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to initialize GPU: {}", e))
        })?;

        Ok(Self {
            engine,
            user_function: None,
        })
    }

    fn set_user_function(&mut self, wgsl_code: &str) {
        self.user_function = Some(wgsl_code.to_string());
    }

    fn run<'py>(
        &mut self,
        py: Python<'py>,
        initial: PyReadonlyArray1<f32>,
        iterations: u32,
        seed: u32,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let input_data = initial.as_slice().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to read input array: {}", e))
        })?;

        // Setup simulation
        self.engine
            .setup_simulation(input_data, iterations, seed)
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to setup simulation: {}",
                    e
                ))
            })?;

        // Generate shader code
        let shader_code = if let Some(ref user_fn) = self.user_function {
            generate_compute_shader(user_fn, 64)
        } else {
            shader_gen::generate_simple_shader(64)
        };

        // Create compute pipeline
        self.engine
            .create_compute_pipeline(&shader_code)
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to create compute pipeline: {}",
                    e
                ))
            })?;

        // Execute
        let result = self.engine.execute().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Execution failed: {}", e))
        })?;

        Ok(result.to_pyarray(py))
    }

    fn run_with_function<'py>(
        &mut self,
        py: Python<'py>,
        initial: PyReadonlyArray1<f32>,
        iterations: u32,
        seed: u32,
        wgsl_function: &str,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        self.set_user_function(wgsl_function);
        self.run(py, initial, iterations, seed)
    }
}

/// Monte Carlo Integrator for Expected Value Calculation
/// Supports fusing multiple functions into a single GPU pass
#[pyclass]
struct MonteCarloIntegrator {
    engine: ComputeEngine,
}

#[pymethods]
impl MonteCarloIntegrator {
    #[new]
    fn new() -> PyResult<Self> {
        let engine = pollster::block_on(ComputeEngine::new()).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to initialize GPU: {}", e))
        })?;

        Ok(Self { engine })
    }

    /// Integrate multiple functions simultaneously
    ///
    /// Args:
    ///     functions: List of WGSL function code strings
    ///     dist_type: Distribution type ("uniform", "normal", "exponential", "table")
    ///     dist_params: Distribution parameters as dict
    ///     n_samples: Total number of Monte Carlo samples
    ///     seed: Random seed
    ///     lookup_table: Optional lookup table for table-based distributions
    ///     target_threads: Optional target thread count (defaults to 65536)
    ///
    /// Returns:
    ///     Vec<f32> of expected values (one per function)
    fn integrate<'py>(
        &mut self,
        py: Python<'py>,
        functions: Vec<String>,
        dist_type: &str,
        dist_params: &Bound<'_, pyo3::types::PyDict>,
        n_samples: u64,
        seed: u32,
        lookup_table: Option<PyReadonlyArray1<f32>>,
        target_threads: Option<u32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let k = functions.len();
        if k == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "At least one function is required",
            ));
        }

        // Parse distribution parameters
        let dist_params_buffer = Self::parse_dist_params(dist_type, dist_params)?;

        // Convert lookup table if provided
        let table_data: Option<Vec<f32>> =
            lookup_table.map(|arr| arr.as_slice().unwrap_or(&[]).to_vec());
        let table_ref: Option<&[f32]> = table_data.as_deref();

        // Setup integration
        let config = self
            .engine
            .setup_integration(
                n_samples,
                k,
                &dist_params_buffer,
                table_ref,
                seed,
                target_threads,
            )
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to setup integration: {}",
                    e
                ))
            })?;

        // Create shader config
        let shader_config = IntegrationShaderConfig {
            user_functions: functions,
            dist_params: Self::convert_dist_params(&dist_params_buffer),
            workgroup_size: config.workgroup_size,
        };

        // Generate shader
        let shader_code = generate_integration_shader(&shader_config, config.loops_per_thread);

        // Create pipeline
        self.engine
            .create_integration_pipeline(&shader_code)
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to create integration pipeline: {}",
                    e
                ))
            })?;

        // Execute
        let thread_results = self
            .engine
            .execute_integration(config.workgroup_count)
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Integration execution failed: {}",
                    e
                ))
            })?;

        // CPU-based reduction: average all thread results for each function
        let n_threads = thread_results.len() / k;
        let mut final_results = vec![0.0f32; k];

        for func_idx in 0..k {
            let sum: f32 = (0..n_threads)
                .map(|thread_idx| thread_results[thread_idx * k + func_idx])
                .sum();
            final_results[func_idx] = sum / n_threads as f32;
        }

        Ok(final_results.to_pyarray(py))
    }
}

impl MonteCarloIntegrator {
    /// Parse distribution parameters from Python dict
    fn parse_dist_params(
        dist_type: &str,
        params: &Bound<'_, pyo3::types::PyDict>,
    ) -> PyResult<DistributionParamsBuffer> {
        match dist_type {
            "uniform" => {
                let min = params
                    .get_item("min")?
                    .and_then(|v| v.extract::<f32>().ok())
                    .unwrap_or(0.0);
                let max = params
                    .get_item("max")?
                    .and_then(|v| v.extract::<f32>().ok())
                    .unwrap_or(1.0);
                Ok(DistributionParamsBuffer {
                    param1: min,
                    param2: max,
                    dist_type: 0,
                    table_size: 0,
                })
            }
            "normal" => {
                let mean = params
                    .get_item("mean")?
                    .and_then(|v| v.extract::<f32>().ok())
                    .unwrap_or(0.0);
                let std = params
                    .get_item("std")?
                    .and_then(|v| v.extract::<f32>().ok())
                    .unwrap_or(1.0);
                Ok(DistributionParamsBuffer {
                    param1: mean,
                    param2: std,
                    dist_type: 1,
                    table_size: 0,
                })
            }
            "exponential" => {
                let lambda = params
                    .get_item("lambda")?
                    .and_then(|v| v.extract::<f32>().ok())
                    .unwrap_or(1.0);
                Ok(DistributionParamsBuffer {
                    param1: lambda,
                    param2: 0.0,
                    dist_type: 2,
                    table_size: 0,
                })
            }
            "table" => {
                let table_size = params
                    .get_item("table_size")?
                    .and_then(|v| v.extract::<u32>().ok())
                    .unwrap_or(2048);
                Ok(DistributionParamsBuffer {
                    param1: 0.0,
                    param2: 0.0,
                    dist_type: 3,
                    table_size,
                })
            }
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown distribution type: {}. Use 'uniform', 'normal', 'exponential', or 'table'",
                dist_type
            ))),
        }
    }

    /// Convert to DistributionParams for shader generation
    fn convert_dist_params(buffer: &DistributionParamsBuffer) -> DistributionParams {
        use distribution::DistributionType;

        let dist_type = match buffer.dist_type {
            0 => DistributionType::Uniform,
            1 => DistributionType::Normal,
            2 => DistributionType::Exponential,
            3 => DistributionType::Table,
            _ => DistributionType::Uniform,
        };

        DistributionParams {
            dist_type,
            param1: buffer.param1,
            param2: buffer.param2,
            table_size: buffer.table_size,
        }
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MonteCarloSimulator>()?;
    m.add_class::<MonteCarloIntegrator>()?;
    Ok(())
}
