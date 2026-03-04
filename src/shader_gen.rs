//! Shader Generation for Multi-Function Monte Carlo Integration
//!
//! Generates WGSL compute shaders that can evaluate multiple functions
//! on the same random samples in a single GPU pass.

use crate::distribution::{
    generate_distribution_library, generate_mcmc_log_pdf_bindings, generate_mcmc_log_pdf_functions,
    generate_pdf_table_bindings, generate_pdf_table_functions, generate_rng_code,
    generate_sample_code, DistributionType, MhLogPdfConfig, PdfTableConfig,
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

/// Configuration for MCMC shader generation
pub struct McmcShaderConfig {
    pub user_functions: Vec<String>,
    pub proposal_dist_type: DistributionType,
    pub target_dist_type: DistributionType,
    pub workgroup_size: u32,
    pub log_pdf_config: MhLogPdfConfig,
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

/// Generate MCMC accumulation code that evaluates at current_x
fn generate_mcmc_accumulation_code(k: usize) -> String {
    (0..k)
        .map(|i| format!("acc_{i} += f32(user_func_{i}(current_x));", i = i))
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

/// Generate MCMC compute shader for Metropolis-Hastings algorithm
///
/// The shader will:
/// 1. Spawn N chains (one per thread)
/// 2. Each chain performs burn-in iterations
/// 3. Each chain collects samples and accumulates function values
/// 4. Results are written to output buffer
pub fn generate_mcmc_shader(config: &McmcShaderConfig) -> String {
    let k = config.user_functions.len();
    let dist_lib = generate_distribution_library();
    let log_pdf_bindings = generate_mcmc_log_pdf_bindings(&config.log_pdf_config);
    let log_pdf_functions = generate_mcmc_log_pdf_functions(&config.log_pdf_config);
    let user_funcs_code = generate_user_functions(&config.user_functions);
    let accumulator_init = generate_accumulator_init(k);
    let mcmc_accumulation_code = generate_mcmc_accumulation_code(k);
    let output_write_code = generate_mcmc_output_write(k);

    // Generate MCMC-specific sampling code
    let init_sample_code = generate_mcmc_init_sample_code(config.proposal_dist_type);

    // Generate log-PDF evaluation code for target distribution
    // For initialization, evaluate at current_x
    let target_log_pdf_code_init = if config.log_pdf_config.has_target_log_pdf_table {
        "log_pdf_target_from_table(current_x)".to_string()
    } else {
        generate_log_pdf_code_for_dist(config.target_dist_type, "target_dist_params")
            .replace("x", "current_x")
    };

    // For step function, we'll generate a version that evaluates at proposal_x
    let target_log_pdf_code_for_step = if config.log_pdf_config.has_target_log_pdf_table {
        "log_pdf_target_from_table(x)".to_string()
    } else {
        generate_log_pdf_code_for_dist(config.target_dist_type, "target_dist_params")
    };

    let mcmc_step_code = generate_mcmc_step_code(
        config.proposal_dist_type,
        &target_log_pdf_code_for_step,
        &config.log_pdf_config,
    );

    format!(
        r#"struct McmcParams {{
    n_chains: u32,
    n_steps: u32,
    n_burnin: u32,
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
var<uniform> params: McmcParams;

@group(0) @binding(1)
var<uniform> proposal_dist_params: DistParams;

@group(0) @binding(2)
var<uniform> target_dist_params: DistParams;

@group(0) @binding(3)
var<storage, read> lookup_table: array<f32>;

@group(0) @binding(4)
var<storage, read> x_table: array<f32>;

@group(0) @binding(5)
var<storage, read_write> output_buffer: array<f32>;

{log_pdf_bindings}

{distribution_library}

{log_pdf_functions}

{user_functions}

// MCMC chain state (per-thread private variables)
var<private> current_x: f32;
var<private> current_log_p: f32;

{mcmc_step_code}

@compute @workgroup_size({workgroup_size})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let idx = global_id.x;
    
    if (idx >= params.n_chains) {{
        return;
    }}
    
    // Initialize chain state with a sample from proposal distribution
    {init_sample_code}
    current_log_p = {target_log_pdf_eval};
    
    // Burn-in phase
    for (var i = 0u; i < params.n_burnin; i = i + 1u) {{
        mcmc_step(params.seed, idx, i + 1u);
    }}
    
    // Sampling phase: initialize K accumulators
    {accumulator_init}
    
    // Main MCMC loop
    for (var i = 0u; i < params.n_steps; i = i + 1u) {{
        let iter = i + params.n_burnin + 1u;
        mcmc_step(params.seed, idx, iter);
        
        // Accumulate function values at current state
        {accumulation_code}
    }}
    
    // Write results to output buffer
    let base_idx = idx * params.k_functions;
    {output_write_code}
}}
"#,
        log_pdf_bindings = log_pdf_bindings,
        distribution_library = dist_lib,
        log_pdf_functions = log_pdf_functions,
        user_functions = user_funcs_code,
        workgroup_size = config.workgroup_size,
        accumulator_init = accumulator_init,
        init_sample_code = init_sample_code,
        target_log_pdf_eval = target_log_pdf_code_init,
        mcmc_step_code = mcmc_step_code,
        accumulation_code = mcmc_accumulation_code,
        output_write_code = output_write_code,
    )
}

/// Generate initialization sample code for MCMC
fn generate_mcmc_init_sample_code(dist_type: DistributionType) -> String {
    match dist_type {
        DistributionType::Uniform => {
            r#"let init_rng = random_uniform(params.seed, idx, 0u);
    current_x = sample_uniform(init_rng, proposal_dist_params.param1, proposal_dist_params.param2);"#.to_string()
        }
        DistributionType::Normal => {
            r#"current_x = sample_normal_box_muller(params.seed, idx, 0u, proposal_dist_params.param1, proposal_dist_params.param2);"#.to_string()
        }
        DistributionType::Exponential => {
            r#"let init_rng = random_uniform(params.seed, idx, 0u);
    current_x = sample_exponential(init_rng, proposal_dist_params.param1);"#.to_string()
        }
        DistributionType::Custom => {
            r#"let init_rng = random_uniform(params.seed, idx, 0u);
    current_x = sample_from_cdf_table(init_rng, proposal_dist_params.table_size);"#.to_string()
        }
    }
}

/// Generate MCMC step function code
fn generate_mcmc_step_code(
    proposal_dist_type: DistributionType,
    target_log_pdf_eval: &str,
    proposal_log_pdf_config: &MhLogPdfConfig,
) -> String {
    // Replace 'x' with 'proposal_x' in target_log_pdf_eval since we're evaluating at the proposal
    let target_log_pdf_at_proposal = target_log_pdf_eval.replace("x", "proposal_x");

    // Generate proposal sampling code for the step function
    let proposal_code = match proposal_dist_type {
        DistributionType::Uniform => {
            r#"let step_rng = random_uniform(seed, idx, iter + 1000000u);
    let proposal_x = sample_uniform(step_rng, proposal_dist_params.param1, proposal_dist_params.param2);"#
        }
        DistributionType::Normal => {
            r#"let proposal_x = sample_normal_box_muller(seed, idx, iter + 1000000u, proposal_dist_params.param1, proposal_dist_params.param2);"#
        }
        DistributionType::Exponential => {
            r#"let step_rng = random_uniform(seed, idx, iter + 1000000u);
    let proposal_x = sample_exponential(step_rng, proposal_dist_params.param1);"#
        }
        DistributionType::Custom => {
            r#"let step_rng = random_uniform(seed, idx, iter + 1000000u);
    let proposal_x = sample_from_cdf_table(step_rng, proposal_dist_params.table_size);"#
        }
    };

    // Generate proposal log-PDF code for MH correction
    // For independent proposal, MH acceptance is:
    // log_alpha = log_p_target(proposal_x) + log_q_proposal(current_x) - log_p_target(current_x) - log_q_proposal(proposal_x)
    let (proposal_log_pdf_at_proposal, proposal_log_pdf_at_current) =
        if proposal_log_pdf_config.has_proposal_log_pdf_table {
            (
                "log_pdf_proposal_from_table(proposal_x)".to_string(),
                "log_pdf_proposal_from_table(current_x)".to_string(),
            )
        } else {
            let proposal_log_pdf_code =
                generate_log_pdf_code_for_dist(proposal_dist_type, "proposal_dist_params");
            (
                proposal_log_pdf_code.replace("x", "proposal_x"),
                proposal_log_pdf_code.replace("x", "current_x"),
            )
        };

    format!(
        r#"fn mcmc_step(seed: u32, idx: u32, iter: u32) {{
    // Generate proposal from proposal distribution
    // For now, we use independent proposal (not dependent on current_x)
    {proposal_code}
    
    // Compute log-PDF of proposal and current under target distribution
    let proposal_log_p_target = {target_log_pdf_at_proposal};
    
    // Compute log-PDF of proposal and current under proposal distribution (for MH correction)
    let proposal_log_q = {proposal_log_pdf_at_proposal};
    let current_log_q = {proposal_log_pdf_at_current};
    
    // Metropolis-Hastings acceptance ratio for independent proposal
    // log_alpha = log_p_target(proposal) + log_q_proposal(current) - log_p_target(current) - log_q_proposal(proposal)
    let log_alpha = proposal_log_p_target + current_log_q - current_log_p - proposal_log_q;
    
    // Accept/reject
    let u = random_uniform(seed + 999999u, idx, iter);
    if (log(u) < log_alpha) {{
        // Accept proposal
        current_x = proposal_x;
        current_log_p = proposal_log_p_target;
    }}
    // Otherwise, keep current state (reject)
}}
"#
    )
}

/// Generate log-PDF code for a distribution type
/// Returns a simple expression that can be used inline
fn generate_log_pdf_code_for_dist(dist_type: DistributionType, params_name: &str) -> String {
    match dist_type {
        DistributionType::Uniform => {
            // Uniform distribution log-PDF: -log(max - min) inside support
            format!(
                "select(-100.0, -log({params}.param2 - {params}.param1), ({params}.param1 <= x && x < {params}.param2))",
                params = params_name
            )
        }
        DistributionType::Normal => {
            // Normal distribution log-PDF: -0.5 * ((x - mean) / std)^2 - log(std * sqrt(2*pi))
            format!(
                "(-0.5 * pow((x - {params}.param1) / {params}.param2, 2.0) - log({params}.param2 * 2.50662827463))",
                params = params_name
            )
        }
        DistributionType::Exponential => {
            // Exponential distribution log-PDF: log(lambda) - lambda * x for x >= 0
            format!(
                "select(-100.0, log({params}.param1) - {params}.param1 * x, (x >= 0.0))",
                params = params_name
            )
        }
        DistributionType::Custom => {
            // Custom distribution uses table lookup
            "log_pdf_target_from_table(x)".to_string()
        }
    }
}

/// Generate output write code for MCMC (per-chain averages)
fn generate_mcmc_output_write(k: usize) -> String {
    (0..k)
        .map(|i| format!("output_buffer[base_idx + {i}u] = acc_{i} / f32(params.n_steps);"))
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
