use numpy::{PyArray1, PyReadonlyArray1, ToPyArray};
use pyo3::prelude::*;

mod distribution;
mod engine;
mod shader_gen;

use distribution::DistributionType;
use engine::{ComputeEngine, DistributionParamsBuffer};
use shader_gen::{
    generate_integration_shader, generate_integration_shader_with_pdf_tables,
    IntegrationShaderConfig, IntegrationShaderConfigWithPdfTables,
};

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
    ///     dist_type: Distribution type ("uniform", "normal", "exponential", "custom")
    ///     dist_params: Distribution parameters as dict
    ///     n_samples: Total number of Monte Carlo samples
    ///     seed: Random seed
    ///     x_table: Optional x values for custom CDF table
    ///     cdf_table: Optional CDF values for custom distribution
    ///     target_threads: Optional target thread count (defaults to 65536)
    ///
    /// Returns:
    ///     Vec<f32> of expected values (one per function)
    #[pyo3(signature = (functions, dist_type, dist_params, n_samples, seed, x_table=None, cdf_table=None, target_threads=None))]
    fn integrate<'py>(
        &mut self,
        py: Python<'py>,
        functions: Vec<String>,
        dist_type: &str,
        dist_params: &Bound<'_, pyo3::types::PyDict>,
        n_samples: u64,
        seed: u32,
        x_table: Option<PyReadonlyArray1<f32>>,
        cdf_table: Option<PyReadonlyArray1<f32>>,
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

        // Convert CDF tables if provided (for custom distributions)
        let x_table_data: Option<Vec<f32>> =
            x_table.map(|arr| arr.as_slice().unwrap_or(&[]).to_vec());
        let x_table_ref: Option<&[f32]> = x_table_data.as_deref();

        let cdf_table_data: Option<Vec<f32>> =
            cdf_table.map(|arr| arr.as_slice().unwrap_or(&[]).to_vec());
        let cdf_table_ref: Option<&[f32]> = cdf_table_data.as_deref();

        // Setup integration
        let config = self
            .engine
            .setup_integration(
                n_samples,
                k,
                &dist_params_buffer,
                x_table_ref,
                cdf_table_ref,
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
            dist_type: Self::convert_dist_type(&dist_params_buffer),
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

    /// Integrate with PDF tables for importance sampling
    ///
    /// Args:
    ///     functions: List of WGSL function code strings
    ///     dist_type: Proposal distribution type
    ///     dist_params: Proposal distribution parameters as dict
    ///     n_samples: Total number of Monte Carlo samples
    ///     seed: Random seed
    ///     x_table: Optional x values for proposal CDF table
    ///     cdf_table: Optional CDF values for proposal distribution
    ///     target_x_table: Optional x values for target PDF table
    ///     target_pdf_table: Optional target PDF values
    ///     proposal_x_table: Optional x values for proposal PDF table
    ///     proposal_pdf_table: Optional proposal PDF values
    ///     target_threads: Optional target thread count
    #[pyo3(signature = (
        functions, dist_type, dist_params, n_samples, seed,
        x_table=None, cdf_table=None,
        target_x_table=None, target_pdf_table=None,
        proposal_x_table=None, proposal_pdf_table=None,
        target_threads=None
    ))]
    fn integrate_is_tables<'py>(
        &mut self,
        py: Python<'py>,
        functions: Vec<String>,
        dist_type: &str,
        dist_params: &Bound<'_, pyo3::types::PyDict>,
        n_samples: u64,
        seed: u32,
        x_table: Option<PyReadonlyArray1<f32>>,
        cdf_table: Option<PyReadonlyArray1<f32>>,
        target_x_table: Option<PyReadonlyArray1<f32>>,
        target_pdf_table: Option<PyReadonlyArray1<f32>>,
        proposal_x_table: Option<PyReadonlyArray1<f32>>,
        proposal_pdf_table: Option<PyReadonlyArray1<f32>>,
        target_threads: Option<u32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let k = functions.len();
        if k == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "At least one function is required",
            ));
        }

        let dist_params_buffer = Self::parse_dist_params(dist_type, dist_params)?;

        // Convert all table data
        let x_table_data: Option<Vec<f32>> =
            x_table.map(|arr| arr.as_slice().unwrap_or(&[]).to_vec());
        let cdf_table_data: Option<Vec<f32>> =
            cdf_table.map(|arr| arr.as_slice().unwrap_or(&[]).to_vec());
        let target_x_data: Option<Vec<f32>> =
            target_x_table.map(|arr| arr.as_slice().unwrap_or(&[]).to_vec());
        let target_pdf_data: Option<Vec<f32>> =
            target_pdf_table.map(|arr| arr.as_slice().unwrap_or(&[]).to_vec());
        let proposal_x_data: Option<Vec<f32>> =
            proposal_x_table.map(|arr| arr.as_slice().unwrap_or(&[]).to_vec());
        let proposal_pdf_data: Option<Vec<f32>> =
            proposal_pdf_table.map(|arr| arr.as_slice().unwrap_or(&[]).to_vec());

        // Setup integration with PDF tables
        let config = self
            .engine
            .setup_integration_with_pdf_tables(
                n_samples,
                k,
                &dist_params_buffer,
                x_table_data.as_deref(),
                cdf_table_data.as_deref(),
                target_x_data.as_deref(),
                target_pdf_data.as_deref(),
                proposal_x_data.as_deref(),
                proposal_pdf_data.as_deref(),
                seed,
                target_threads,
            )
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to setup integration with PDF tables: {}",
                    e
                ))
            })?;

        // Get PDF table config for shader generation
        let pdf_config = self.engine.get_pdf_table_config();

        // Create shader config with PDF tables
        let shader_config = IntegrationShaderConfigWithPdfTables {
            user_functions: functions,
            dist_type: Self::convert_dist_type(&dist_params_buffer),
            workgroup_size: config.workgroup_size,
            pdf_table_config: pdf_config,
        };

        // Generate shader
        let shader_code =
            generate_integration_shader_with_pdf_tables(&shader_config, config.loops_per_thread);

        // Create pipeline with PDF tables
        self.engine
            .create_integration_pipeline_with_pdf_tables(&shader_code)
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to create integration pipeline with PDF tables: {}",
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

        // reduction
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
            "custom" => {
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
                "Unknown distribution type: {}. Use 'uniform', 'normal', 'exponential', or 'custom'",
                dist_type
            ))),
        }
    }

    /// Convert to DistributionType for shader generation
    fn convert_dist_type(buffer: &DistributionParamsBuffer) -> DistributionType {
        match buffer.dist_type {
            0 => DistributionType::Uniform,
            1 => DistributionType::Normal,
            2 => DistributionType::Exponential,
            3 => DistributionType::Custom,
            _ => DistributionType::Uniform,
        }
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MonteCarloIntegrator>()?;
    Ok(())
}
