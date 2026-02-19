//! Shader Generation for Multi-Function Monte Carlo Integration
//!
//! Generates WGSL compute shaders that can evaluate multiple functions
//! on the same random samples in a single GPU pass.

use crate::distribution::{
    generate_distribution_library, generate_pdf_table_bindings, generate_pdf_table_functions,
    generate_rng_code, generate_sample_code, DistributionType, PdfTableConfig,
};

/// Configuration for integration shader generation
pub struct IntegrationShaderConfig {
    pub user_functions: Vec<String>,
    pub dist_type: DistributionType,
    pub workgroup_size: u32,
}

/// Extended configuration for integration shader with PDF table support
pub struct IntegrationShaderConfigWithPdfTables {
    pub user_functions: Vec<String>,
    pub dist_type: DistributionType,
    pub workgroup_size: u32,
    pub pdf_table_config: PdfTableConfig,
}

/// Generate a multi-function integration compute shader
///
/// The shader will:
/// 1. Spawn N threads (e.g., 65536)
/// 2. Each thread loops M times (loops_per_thread)
/// 3. Each iteration generates a random sample from the specified distribution
/// 4. All K user functions are evaluated on the same sample
/// 5. Results are accumulated in K separate registers
/// 6. Final results written to output buffer in flattened layout
pub fn generate_integration_shader(
    config: &IntegrationShaderConfig,
    _loops_per_thread: u32,
) -> String {
    let k = config.user_functions.len();
    let dist_lib = generate_distribution_library();
    let user_funcs_code = generate_user_functions(&config.user_functions);
    let accumulator_init = generate_accumulator_init(k);
    let sample_code = generate_sample_code(config.dist_type);
    let rng_code = generate_rng_code(config.dist_type);
    let accumulation_code = generate_accumulation_code(k);
    let output_write_code = generate_output_write(k);

    format!(
        r#"struct IntegrationParams {{
    n_threads: u32,
    loops_per_thread: u32,
    seed: u32,
    k_functions: u32,
    _padding: u32,
}}

struct DistParams {{
    param1: f32,
    param2: f32,
    dist_type: u32,
    table_size: u32,
}}

@group(0) @binding(0)
var<uniform> params: IntegrationParams;

@group(0) @binding(1)
var<uniform> dist_params: DistParams;

@group(0) @binding(2)
var<storage, read> lookup_table: array<f32>;

@group(0) @binding(3)
var<storage, read> x_table: array<f32>;

@group(0) @binding(4)
var<storage, read_write> output_buffer: array<f32>;

{distribution_library}

{user_functions}

@compute @workgroup_size({workgroup_size})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let idx = global_id.x;
    
    if (idx >= params.n_threads) {{
        return;
    }}
    
    // Initialize K accumulators in registers (fast local memory)
    {accumulator_init}
    
    // Main integration loop
    for (var i = 0u; i < params.loops_per_thread; i = i + 1u) {{
        // Generate random sample from distribution
        {rng_code}
        let sample = {sample_code};
        
        // Evaluate all K functions on the same sample, accumulate
        {accumulation_code}
    }}
    
    // Write results to output buffer (flattened layout: output[thread_id * K + metric_id])
    let base_idx = idx * params.k_functions;
    {output_write_code}
}}
"#,
        distribution_library = dist_lib,
        user_functions = user_funcs_code,
        workgroup_size = config.workgroup_size,
        accumulator_init = accumulator_init,
        rng_code = rng_code,
        sample_code = sample_code,
        accumulation_code = accumulation_code,
        output_write_code = output_write_code,
    )
}

/// Generate a multi-function integration compute shader with PDF table support
///
/// This is an extended version that supports importance sampling with PDF lookup tables.
/// The PDF tables are stored in interleaved format: [table_size, x0, pdf0, x1, pdf1, ...]
pub fn generate_integration_shader_with_pdf_tables(
    config: &IntegrationShaderConfigWithPdfTables,
    _loops_per_thread: u32,
) -> String {
    let k = config.user_functions.len();
    let dist_lib = generate_distribution_library();
    let pdf_bindings = generate_pdf_table_bindings(&config.pdf_table_config);
    let pdf_functions = generate_pdf_table_functions(&config.pdf_table_config);
    let user_funcs_code = generate_user_functions(&config.user_functions);
    let accumulator_init = generate_accumulator_init(k);
    let sample_code = generate_sample_code(config.dist_type);
    let rng_code = generate_rng_code(config.dist_type);
    let accumulation_code = generate_accumulation_code(k);
    let output_write_code = generate_output_write(k);

    format!(
        r#"struct IntegrationParams {{
    n_threads: u32,
    loops_per_thread: u32,
    seed: u32,
    k_functions: u32,
    _padding: u32,
}}

struct DistParams {{
    param1: f32,
    param2: f32,
    dist_type: u32,
    table_size: u32,
}}

@group(0) @binding(0)
var<uniform> params: IntegrationParams;

@group(0) @binding(1)
var<uniform> dist_params: DistParams;

@group(0) @binding(2)
var<storage, read> lookup_table: array<f32>;

@group(0) @binding(3)
var<storage, read> x_table: array<f32>;

@group(0) @binding(4)
var<storage, read_write> output_buffer: array<f32>;

{pdf_bindings}

{distribution_library}

{pdf_functions}

{user_functions}

@compute @workgroup_size({workgroup_size})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let idx = global_id.x;
    
    if (idx >= params.n_threads) {{
        return;
    }}
    
    // Initialize K accumulators in registers (fast local memory)
    {accumulator_init}
    
    // Main integration loop
    for (var i = 0u; i < params.loops_per_thread; i = i + 1u) {{
        // Generate random sample from distribution
        {rng_code}
        let sample = {sample_code};
        
        // Evaluate all K functions on the same sample, accumulate
        {accumulation_code}
    }}
    
    // Write results to output buffer (flattened layout: output[thread_id * K + metric_id])
    let base_idx = idx * params.k_functions;
    {output_write_code}
}}
"#,
        pdf_bindings = pdf_bindings,
        distribution_library = dist_lib,
        pdf_functions = pdf_functions,
        user_functions = user_funcs_code,
        workgroup_size = config.workgroup_size,
        accumulator_init = accumulator_init,
        rng_code = rng_code,
        sample_code = sample_code,
        accumulation_code = accumulation_code,
        output_write_code = output_write_code,
    )
}

/// Generate user function wrappers with standardized naming
/// Functions are renamed from whatever the user provided to user_func_0, user_func_1, etc.
fn generate_user_functions(functions: &[String]) -> String {
    functions
        .iter()
        .enumerate()
        .map(|(i, func_code)| {
            // Auto-rename the function to user_func_{i}
            // This assumes the function definition starts with "fn function_name("
            let renamed = rename_function(func_code, i);
            format!("// User function {}\n{}", i, renamed)
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Rename a function definition from whatever it is to user_func_{index}
fn rename_function(func_code: &str, index: usize) -> String {
    // Simple regex-like replacement: find "fn name(" and replace with "fn user_func_i("
    // This handles cases where the function might be named "step", "f", "kernel", etc.
    let new_name = format!("user_func_{}", index);

    // Find the first occurrence of "fn " followed by an identifier and "("
    if let Some(fn_pos) = func_code.find("fn ") {
        let after_fn = &func_code[fn_pos + 3..];
        if let Some(paren_pos) = after_fn.find('(') {
            let before_fn = &func_code[..fn_pos + 3];
            let after_paren = &func_code[fn_pos + 3 + paren_pos..];
            return format!("{}{}{}", before_fn, new_name, after_paren);
        }
    }

    // If we can't parse it, just prepend the rename comment
    format!("// Auto-renamed to {}\n{}", new_name, func_code)
}

/// Generate K accumulator variable declarations
fn generate_accumulator_init(k: usize) -> String {
    (0..k)
        .map(|i| format!("var acc_{} = 0.0;", i))
        .collect::<Vec<_>>()
        .join("\n    ")
}

/// Generate accumulation code that evaluates all K functions
fn generate_accumulation_code(k: usize) -> String {
    (0..k)
        .map(|i| {
            format!(
                "// Cast to f32 to handle boolean or other return types\n        acc_{i} += f32(user_func_{i}(sample));",
                i = i
            )
        })
        .collect::<Vec<_>>()
        .join("\n        ")
}

/// Generate output write code for flattened buffer layout
fn generate_output_write(k: usize) -> String {
    (0..k)
        .map(|i| {
            format!(
                "output_buffer[base_idx + {i}u] = acc_{i} / f32(params.loops_per_thread);",
                i = i
            )
        })
        .collect::<Vec<_>>()
        .join("\n    ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distribution::DistributionType;

    #[test]
    fn test_single_function_shader() {
        let func = "fn compute(x: f32) -> f32 {\n    return x * x;\n}".to_string();
        let config = IntegrationShaderConfig {
            user_functions: vec![func],
            dist_type: DistributionType::Uniform,
            workgroup_size: 256,
        };

        let shader = generate_integration_shader(&config, 100);
        assert!(shader.contains("struct IntegrationParams"));
        assert!(shader.contains("user_func_0"));
        assert!(shader.contains("var acc_0 = 0.0"));
        assert!(shader.contains("acc_0 += f32(user_func_0(sample))"));
        assert!(shader.contains("base_idx + 0u"));
    }

    #[test]
    fn test_multi_function_shader() {
        let funcs = vec![
            "fn f1(x: f32) -> f32 { return x; }".to_string(),
            "fn f2(x: f32) -> f32 { return x * x; }".to_string(),
            "fn f3(x: f32) -> f32 { return x * x * x; }".to_string(),
        ];
        let config = IntegrationShaderConfig {
            user_functions: funcs,
            dist_type: DistributionType::Normal,
            workgroup_size: 256,
        };

        let shader = generate_integration_shader(&config, 100);

        assert!(shader.contains("user_func_0"));
        assert!(shader.contains("user_func_1"));
        assert!(shader.contains("user_func_2"));

        assert!(shader.contains("var acc_0 = 0.0"));
        assert!(shader.contains("var acc_1 = 0.0"));
        assert!(shader.contains("var acc_2 = 0.0"));

        assert!(shader.contains("acc_0 += f32(user_func_0(sample))"));
        assert!(shader.contains("acc_1 += f32(user_func_1(sample))"));
        assert!(shader.contains("acc_2 += f32(user_func_2(sample))"));

        assert!(shader.contains("base_idx + 0u"));
        assert!(shader.contains("base_idx + 1u"));
        assert!(shader.contains("base_idx + 2u"));
    }

    #[test]
    fn test_function_rename() {
        let code = "fn step(x: f32) -> f32 { return x + 1.0; }";
        let renamed = rename_function(code, 5);
        assert!(renamed.contains("fn user_func_5(x: f32)"));
        assert!(!renamed.contains("fn step(x: f32)"));
    }

    #[test]
    fn test_accumulator_generation() {
        let init_1 = generate_accumulator_init(1);
        assert_eq!(init_1, "var acc_0 = 0.0;");

        let init_3 = generate_accumulator_init(3);
        assert!(init_3.contains("var acc_0 = 0.0;"));
        assert!(init_3.contains("var acc_1 = 0.0;"));
        assert!(init_3.contains("var acc_2 = 0.0;"));
    }

    #[test]
    fn test_shader_uniform_sampling_in_output() {
        let config = IntegrationShaderConfig {
            user_functions: vec!["fn f(x: f32) -> f32 { return x; }".to_string()],
            dist_type: DistributionType::Uniform,
            workgroup_size: 256,
        };

        let shader = generate_integration_shader(&config, 100);
        assert!(shader.contains("sample_uniform"));
    }

    #[test]
    fn test_shader_normal_sampling_in_output() {
        let config = IntegrationShaderConfig {
            user_functions: vec!["fn f(x: f32) -> f32 { return x; }".to_string()],
            dist_type: DistributionType::Normal,
            workgroup_size: 256,
        };

        let shader = generate_integration_shader(&config, 100);
        assert!(shader.contains("sample_normal_box_muller"));
    }

    #[test]
    fn test_shader_exponential_sampling_in_output() {
        let config = IntegrationShaderConfig {
            user_functions: vec!["fn f(x: f32) -> f32 { return x; }".to_string()],
            dist_type: DistributionType::Exponential,
            workgroup_size: 256,
        };

        let shader = generate_integration_shader(&config, 100);
        assert!(shader.contains("sample_exponential"));
    }

    #[test]
    fn test_shader_table_sampling_in_output() {
        let config = IntegrationShaderConfig {
            user_functions: vec!["fn f(x: f32) -> f32 { return x; }".to_string()],
            dist_type: DistributionType::Custom,
            workgroup_size: 256,
        };

        let shader = generate_integration_shader(&config, 100);
        assert!(shader.contains("sample_from_cdf_table"));
        assert!(shader.contains("lookup_table"));
        assert!(shader.contains("x_table"));
    }
}

// ============================================================================
// Legacy Simulation Shaders (for backward compatibility)
// ============================================================================

/// Generate compute shader for path-dependent simulation
///
/// DEPRECATED: Use generate_integration_shader for new code.
/// Kept for backward compatibility with existing MonteCarloSimulator API.
pub fn generate_compute_shader(user_function: &str, workgroup_size: u32) -> String {
    format!(
        r#"struct SimulationParams {{
    n: u32,
    iterations: u32,
    seed: u32,
    _padding: u32,
}}

@group(0) @binding(0)
var<uniform> params: SimulationParams;

@group(0) @binding(1)
var<storage, read> input_buffer: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output_buffer: array<f32>;

// PCG Hash random number generator
fn pcg_hash(v: u32) -> u32 {{
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}}

// Generate random float in [0.0, 1.0)
fn random_float(seed: u32, idx: u32, iter: u32) -> f32 {{
    let combined = seed + idx * 7199369u + iter * 15485863u;
    let hashed = pcg_hash(combined);
    return f32(hashed) / 4294967295.0;
}}

// User-defined step function
{user_function}

@compute @workgroup_size({workgroup_size})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let idx = global_id.x;
    
    // Guard against out-of-bounds access
    if (idx >= params.n) {{
        return;
    }}
    
    // Load initial value
    var x = input_buffer[idx];
    
    // Run simulation iterations
    for (var i = 0u; i < params.iterations; i = i + 1u) {{
        // Generate random number for this iteration
        let rng = random_float(params.seed, idx, i);
        
        // Apply user step function
        x = step(x, rng);
    }}
    
    // Store result
    output_buffer[idx] = x;
}}
"#,
        user_function = user_function,
        workgroup_size = workgroup_size
    )
}

/// Generate simple default simulation shader
///
/// DEPRECATED: Use generate_integration_shader for new code.
/// Kept for backward compatibility with existing MonteCarloSimulator API.
pub fn generate_simple_shader(workgroup_size: u32) -> String {
    format!(
        r#"struct SimulationParams {{
    n: u32,
    iterations: u32,
    seed: u32,
    _padding: u32,
}}

@group(0) @binding(0)
var<uniform> params: SimulationParams;

@group(0) @binding(1)
var<storage, read> input_buffer: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output_buffer: array<f32>;

// PCG Hash random number generator
fn pcg_hash(v: u32) -> u32 {{
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}}

// Generate random float in [0.0, 1.0)
fn random_float(seed: u32, idx: u32, iter: u32) -> f32 {{
    let combined = seed + idx * 7199369u + iter * 15485863u;
    let hashed = pcg_hash(combined);
    return f32(hashed) / 4294967295.0;
}}

// Default step function: simple random walk
fn step(x: f32, rng: f32) -> f32 {{
    // Random walk with mean reversion
    let delta = (rng - 0.5) * 0.1;
    return x + delta;
}}

@compute @workgroup_size({workgroup_size})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let idx = global_id.x;
    
    if (idx >= params.n) {{
        return;
    }}
    
    var x = input_buffer[idx];
    
    for (var i = 0u; i < params.iterations; i = i + 1u) {{
        let rng = random_float(params.seed, idx, i);
        x = step(x, rng);
    }}
    
    output_buffer[idx] = x;
}}
"#,
        workgroup_size = workgroup_size
    )
}
