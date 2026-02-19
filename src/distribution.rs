//! WGSL Distribution Library for Monte Carlo Integration
//!
//! This module provides WGSL code generation for various probability distributions.
//! All distributions generate samples on the GPU from uniform random numbers.

/// Distribution types supported by the integrator
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistributionType {
    Uniform,
    Normal,
    Exponential,
    Custom,
}

/// Configuration for PDF table functions
#[derive(Debug, Clone)]
pub struct PdfTableConfig {
    pub has_target_pdf_table: bool,
    pub has_proposal_pdf_table: bool,
    pub target_table_size: u32,
    pub proposal_table_size: u32,
}

impl Default for PdfTableConfig {
    fn default() -> Self {
        Self {
            has_target_pdf_table: false,
            has_proposal_pdf_table: false,
            target_table_size: 0,
            proposal_table_size: 0,
        }
    }
}

/// Generate the WGSL distribution library code
pub fn generate_distribution_library() -> String {
    r#"// ========================================
// WGSL Distribution Library
// ========================================

// PCG Hash random number generator
fn pcg_hash(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Generate uniform random float in [0.0, 1.0)
fn random_uniform(seed: u32, idx: u32, iter: u32) -> f32 {
    let combined = seed + idx * 7199369u + iter * 15485863u;
    let hashed = pcg_hash(combined);
    return f32(hashed) / 4294967295.0;
}

// ========================================
// Distribution Sampling Functions
// ========================================

// Uniform distribution: scale [0,1) to [min, max)
fn sample_uniform(rng: f32, min: f32, max: f32) -> f32 {
    return min + rng * (max - min);
}

// Normal distribution: Box-Muller Transform
// Uses 2 uniform random numbers to generate 2 normal samples
// We cache one and return the other
var<private> normal_cached: f32 = 0.0;
var<private> has_cached_normal: bool = false;

fn sample_normal_box_muller(seed: u32, idx: u32, iter: u32, mean: f32, sigma: f32) -> f32 {
    if (has_cached_normal) {
        has_cached_normal = false;
        return mean + sigma * normal_cached;
    }
    
    // Generate two independent uniform random numbers
    let u1 = random_uniform(seed, idx, iter * 2u);
    let u2 = random_uniform(seed, idx, iter * 2u + 1u);
    
    // Box-Muller transform
    // z0 = sqrt(-2 * ln(u1)) * cos(2*pi*u2)
    // z1 = sqrt(-2 * ln(u1)) * sin(2*pi*u2)
    let r = sqrt(-2.0 * log(u1));
    let theta = 6.283185307179586 * u2;  // 2 * PI
    
    let z0 = r * cos(theta);
    let z1 = r * sin(theta);
    
    // Cache z1 for next call
    normal_cached = z1;
    has_cached_normal = true;
    
    return mean + sigma * z0;
}

// Exponential distribution: Inverse Transform Sampling
// PDF: f(x) = lambda * exp(-lambda * x) for x >= 0
// CDF: F(x) = 1 - exp(-lambda * x)
// Inverse CDF: x = -ln(1 - u) / lambda = -ln(u) / lambda (since 1-u is also uniform)
fn sample_exponential(rng: f32, lambda: f32) -> f32 {
    // Avoid log(0) by ensuring rng > 0
    let u = max(rng, 1.0e-7);
    return -log(u) / lambda;
}

// Custom distribution sampling using CDF lookup table
// Uses binary search and linear interpolation
fn sample_from_cdf_table(rng: f32, table_size: u32) -> f32 {
    // Binary search for rng in cdf_table
    var low = 0u;
    var high = table_size - 1u;
    
    for (var j = 0u; j < 12u; j++) {
        if (low >= high) { break; }
        let mid = (low + high) / 2u;
        if (lookup_table[mid] < rng) {
            low = mid + 1u;
        } else {
            high = mid;
        }
    }
    
    // Linear interpolation
    let idx_low = max(low, 1u) - 1u;
    let idx_high = min(low, table_size - 1u);
    
    let cdf_low = lookup_table[idx_low];
    let cdf_high = lookup_table[idx_high];
    let x_low = x_table[idx_low];
    let x_high = x_table[idx_high];
    
    if (cdf_high - cdf_low < 1.0e-10) {
        return x_low;
    }
    
    let t = (rng - cdf_low) / (cdf_high - cdf_low);
    return mix(x_low, x_high, t);
}

// ========================================
// PDF Lookup Functions for Importance Sampling
// ========================================

// PDF lookup helper macro for interleaved table
// Interleaved format: [table_size, x0, pdf0, x1, pdf1, ..., xn, pdfn]
// This function is NOT used directly - see pdf_target_from_table and pdf_proposal_from_table
"#
    .to_string()
}

/// Generate WGSL code for PDF table wrapper functions
/// Each wrapper directly accesses its own global buffer
pub fn generate_pdf_table_functions(config: &PdfTableConfig) -> String {
    let mut functions = String::new();

    if config.has_target_pdf_table {
        functions.push_str(&format!(
            r#"
// Target PDF from interleaved table
// Format: [table_size, x0, pdf0, x1, pdf1, ...]
fn pdf_target_from_table(x: f32) -> f32 {{
    let table_size = u32(target_pdf_data[0u]);
    
    // Boundary check
    let x_min = target_pdf_data[1u];
    let x_max = target_pdf_data[1u + (table_size - 1u) * 2u];
    if (x < x_min) || (x > x_max) {{
        return 0.0;
    }}
    
    // Binary search for x position
    var low = 0u;
    var high = table_size - 1u;
    
    for (var j = 0u; j < 16u; j++) {{
        if (low >= high) {{ break; }}
        let mid = (low + high) / 2u;
        let x_mid = target_pdf_data[1u + mid * 2u];
        if (x_mid < x) {{
            low = mid + 1u;
        }} else {{
            high = mid;
        }}
    }}
    
    // Clamp to valid interpolation range
    low = max(1u, low) - 1u;
    low = min(low, table_size - 2u);
    
    // Linear interpolation
    let x_low = target_pdf_data[1u + low * 2u];
    let x_high = target_pdf_data[1u + (low + 1u) * 2u];
    let pdf_low = target_pdf_data[2u + low * 2u];
    let pdf_high = target_pdf_data[2u + (low + 1u) * 2u];
    
    let dx = x_high - x_low;
    if (dx < 1.0e-10) {{
        return pdf_low;
    }}
    
    let t = (x - x_low) / dx;
    return mix(pdf_low, pdf_high, t);
}}
"#
        ));
    }

    if config.has_proposal_pdf_table {
        functions.push_str(&format!(
            r#"
// Proposal PDF from interleaved table
// Format: [table_size, x0, pdf0, x1, pdf1, ...]
fn pdf_proposal_from_table(x: f32) -> f32 {{
    let table_size = u32(proposal_pdf_data[0u]);
    
    // Boundary check
    let x_min = proposal_pdf_data[1u];
    let x_max = proposal_pdf_data[1u + (table_size - 1u) * 2u];
    if (x < x_min) || (x > x_max) {{
        return 0.0;
    }}
    
    // Binary search for x position
    var low = 0u;
    var high = table_size - 1u;
    
    for (var j = 0u; j < 16u; j++) {{
        if (low >= high) {{ break; }}
        let mid = (low + high) / 2u;
        let x_mid = proposal_pdf_data[1u + mid * 2u];
        if (x_mid < x) {{
            low = mid + 1u;
        }} else {{
            high = mid;
        }}
    }}
    
    // Clamp to valid interpolation range
    low = max(1u, low) - 1u;
    low = min(low, table_size - 2u);
    
    // Linear interpolation
    let x_low = proposal_pdf_data[1u + low * 2u];
    let x_high = proposal_pdf_data[1u + (low + 1u) * 2u];
    let pdf_low = proposal_pdf_data[2u + low * 2u];
    let pdf_high = proposal_pdf_data[2u + (low + 1u) * 2u];
    
    let dx = x_high - x_low;
    if (dx < 1.0e-10) {{
        return pdf_low;
    }}
    
    let t = (x - x_low) / dx;
    return mix(pdf_low, pdf_high, t);
}}
"#
        ));
    }

    functions
}

/// Generate WGSL bind group declarations for PDF tables
pub fn generate_pdf_table_bindings(config: &PdfTableConfig) -> String {
    let mut bindings = String::new();
    let mut binding_idx = 5u32;

    if config.has_target_pdf_table {
        let data_size = 1 + config.target_table_size * 2;
        bindings.push_str(&format!(
            r#"
@group(0) @binding({binding_idx})
var<storage, read> target_pdf_data: array<f32, {data_size}>;
"#
        ));
        binding_idx += 1;
    }

    if config.has_proposal_pdf_table {
        let data_size = 1 + config.proposal_table_size * 2;
        bindings.push_str(&format!(
            r#"
@group(0) @binding({binding_idx})
var<storage, read> proposal_pdf_data: array<f32, {data_size}>;
"#
        ));
    }

    bindings
}

/// Generate WGSL code for sampling from a specific distribution
pub fn generate_sample_code(dist_type: DistributionType) -> String {
    match dist_type {
        DistributionType::Uniform => {
            "sample_uniform(rng, dist_params.param1, dist_params.param2)".to_string()
        }
        DistributionType::Normal => {
            "sample_normal_box_muller(params.seed, idx, i, dist_params.param1, dist_params.param2)"
                .to_string()
        }
        DistributionType::Exponential => "sample_exponential(rng, dist_params.param1)".to_string(),
        DistributionType::Custom => {
            "sample_from_cdf_table(rng, dist_params.table_size)".to_string()
        }
    }
}

/// Generate WGSL code for RNG generation based on distribution type
pub fn generate_rng_code(dist_type: DistributionType) -> String {
    match dist_type {
        DistributionType::Normal => "".to_string(),
        _ => "let rng = random_uniform(params.seed, idx, i);".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distribution_library_generation() {
        let lib = generate_distribution_library();
        assert!(lib.contains("pcg_hash"));
        assert!(lib.contains("sample_uniform"));
        assert!(lib.contains("sample_normal_box_muller"));
        assert!(lib.contains("sample_exponential"));
        assert!(lib.contains("sample_from_cdf_table"));
    }

    #[test]
    fn test_distribution_type_variants() {
        let _ = DistributionType::Uniform;
        let _ = DistributionType::Normal;
        let _ = DistributionType::Exponential;
        let _ = DistributionType::Custom;
    }

    #[test]
    fn test_distribution_type_debug() {
        let uniform = format!("{:?}", DistributionType::Uniform);
        assert!(uniform.contains("Uniform"));

        let normal = format!("{:?}", DistributionType::Normal);
        assert!(normal.contains("Normal"));
    }

    #[test]
    fn test_distribution_type_partial_eq() {
        assert_eq!(DistributionType::Uniform, DistributionType::Uniform);
        assert_ne!(DistributionType::Uniform, DistributionType::Normal);
    }

    #[test]
    fn test_sample_code_uniform() {
        let code = generate_sample_code(DistributionType::Uniform);
        assert!(code.contains("sample_uniform"));
    }

    #[test]
    fn test_sample_code_normal() {
        let code = generate_sample_code(DistributionType::Normal);
        assert!(code.contains("sample_normal_box_muller"));
    }

    #[test]
    fn test_sample_code_exponential() {
        let code = generate_sample_code(DistributionType::Exponential);
        assert!(code.contains("sample_exponential"));
    }

    #[test]
    fn test_sample_code_custom() {
        let code = generate_sample_code(DistributionType::Custom);
        assert!(code.contains("sample_from_cdf_table"));
    }

    #[test]
    fn test_rng_code_normal() {
        let code = generate_rng_code(DistributionType::Normal);
        assert!(code.is_empty());
    }

    #[test]
    fn test_rng_code_other() {
        let code = generate_rng_code(DistributionType::Uniform);
        assert!(code.contains("random_uniform"));

        let code = generate_rng_code(DistributionType::Exponential);
        assert!(code.contains("random_uniform"));

        let code = generate_rng_code(DistributionType::Custom);
        assert!(code.contains("random_uniform"));
    }
}
