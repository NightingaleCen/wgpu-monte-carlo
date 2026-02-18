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
"#
    .to_string()
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
