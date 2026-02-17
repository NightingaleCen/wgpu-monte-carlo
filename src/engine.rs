use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::{Buffer, Device, Queue};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SimulationParams {
    pub n: u32,          // Array length
    pub iterations: u32, // Number of iterations
    pub seed: u32,       // RNG seed
    pub _padding: u32,   // Ensure 16-byte alignment
}

/// Parameters for Monte Carlo integration
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct IntegrationParams {
    pub n_threads: u32,        // Number of GPU threads to launch
    pub loops_per_thread: u32, // Samples per thread
    pub seed: u32,             // RNG seed
    pub k_functions: u32,      // Number of functions being integrated
    pub _padding: u32,         // Ensure 16-byte alignment
}

/// Parameters for probability distributions
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct DistributionParamsBuffer {
    pub param1: f32,     // min/mean/lambda
    pub param2: f32,     // max/std/unused
    pub dist_type: u32,  // 0=uniform, 1=normal, 2=exponential, 3=table
    pub table_size: u32, // For table-based distributions
}

/// Dispatch configuration for smart workload partitioning
#[derive(Debug, Clone)]
pub struct DispatchConfig {
    pub workgroup_size: u32,
    pub workgroup_count: u32,
    pub loops_per_thread: u32,
    pub total_threads: u32,
}

pub struct ComputeEngine {
    device: Arc<Device>,
    queue: Arc<Queue>,
    params_buffer: Option<Buffer>,
    dist_params_buffer: Option<Buffer>,
    input_buffer: Option<Buffer>,
    output_buffer: Option<Buffer>,
    staging_buffer: Option<Buffer>,
    lookup_table_buffer: Option<Buffer>,
    compute_pipeline: Option<wgpu::ComputePipeline>,
    bind_group: Option<wgpu::BindGroup>,
    workgroup_size: u32,
}

impl ComputeEngine {
    pub async fn new() -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .context("Failed to find a suitable GPU adapter")?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Compute Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .context("Failed to create device")?;

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            params_buffer: None,
            dist_params_buffer: None,
            input_buffer: None,
            output_buffer: None,
            staging_buffer: None,
            lookup_table_buffer: None,
            compute_pipeline: None,
            bind_group: None,
            workgroup_size: 64,
        })
    }

    pub fn set_workgroup_size(&mut self, size: u32) {
        self.workgroup_size = size;
    }

    pub fn setup_simulation(
        &mut self,
        input_data: &[f32],
        iterations: u32,
        seed: u32,
    ) -> Result<()> {
        let n = input_data.len() as u32;
        let buffer_size = (input_data.len() * std::mem::size_of::<f32>()) as u64;

        // Create simulation parameters buffer
        let params = SimulationParams {
            n,
            iterations,
            seed,
            _padding: 0,
        };

        self.params_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Params Buffer"),
            size: std::mem::size_of::<SimulationParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        self.queue.write_buffer(
            self.params_buffer.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&[params]),
        );

        // Create input buffer (read-only storage)
        self.input_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Input Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        self.queue.write_buffer(
            self.input_buffer.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(input_data),
        );

        // Create output buffer (read-write storage)
        self.output_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));

        // Create staging buffer for CPU readback
        self.staging_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        Ok(())
    }

    pub fn create_compute_pipeline(&mut self, shader_code: &str) -> Result<()> {
        let shader_module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Compute Shader"),
                source: wgpu::ShaderSource::Wgsl(shader_code.into()),
            });

        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Bind Group Layout"),
                    entries: &[
                        // Params (uniform)
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Input buffer (read-only storage)
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Output buffer (read-write storage)
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        self.compute_pipeline = Some(self.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("Compute Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            },
        ));

        // Create bind group (only if buffers exist)
        if self.params_buffer.is_some()
            && self.input_buffer.is_some()
            && self.output_buffer.is_some()
        {
            self.bind_group = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Bind Group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.params_buffer.as_ref().unwrap().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.input_buffer.as_ref().unwrap().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.output_buffer.as_ref().unwrap().as_entire_binding(),
                    },
                ],
            }));
        }

        Ok(())
    }

    pub fn execute(&mut self) -> Result<Vec<f32>> {
        let pipeline = self
            .compute_pipeline
            .as_ref()
            .context("Pipeline not created")?;
        let bind_group = self.bind_group.as_ref().context("Bind group not created")?;
        let n = self
            .input_buffer
            .as_ref()
            .map(|b| b.size() as u32 / 4)
            .context("Input buffer not initialized")?;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Compute Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, bind_group, &[]);

            let dispatch_count = (n + self.workgroup_size - 1) / self.workgroup_size;
            compute_pass.dispatch_workgroups(dispatch_count, 1, 1);
        }

        // Copy output to staging buffer
        encoder.copy_buffer_to_buffer(
            self.output_buffer.as_ref().unwrap(),
            0,
            self.staging_buffer.as_ref().unwrap(),
            0,
            self.output_buffer.as_ref().unwrap().size(),
        );

        // Submit commands
        self.queue.submit(std::iter::once(encoder.finish()));

        // Read back results
        let staging_buffer = self.staging_buffer.as_ref().unwrap();
        let buffer_slice = staging_buffer.slice(..);

        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        self.device.poll(wgpu::Maintain::Wait);

        rx.recv()
            .context("Failed to receive map async result")?
            .context("Failed to map buffer")?;

        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        Ok(result)
    }

    pub fn get_device(&self) -> &Device {
        &self.device
    }

    pub fn get_queue(&self) -> &Queue {
        &self.queue
    }

    // ========================================
    // Integration Mode Methods
    // ========================================

    /// Calculate optimal dispatch configuration for Monte Carlo integration
    ///
    /// Given a total number of samples, this calculates:
    /// - How many threads to launch (target ~65k to saturate GPU)
    /// - How many samples each thread should process
    /// - Workgroup count and size
    ///
    /// Exposed as tunable parameter for advanced users
    pub fn calculate_dispatch_config(
        &self,
        n_samples: u64,
        target_threads: Option<u32>,
    ) -> DispatchConfig {
        // Default to ~65k threads to saturate GPU without hitting limits
        // Users can override this via target_threads parameter
        let target = target_threads.unwrap_or(65536);
        let workgroup_size: u32 = 256; // Common optimal size

        let workgroup_count = (target + workgroup_size - 1) / workgroup_size;
        let total_threads = workgroup_count * workgroup_size;

        // Calculate how many samples each thread should process
        // Ceiling division to ensure we get at least n_samples total
        let loops_per_thread =
            ((n_samples + total_threads as u64 - 1) / total_threads as u64) as u32;

        DispatchConfig {
            workgroup_size,
            workgroup_count,
            loops_per_thread,
            total_threads,
        }
    }

    /// Setup integration buffers and parameters
    ///
    /// This creates the necessary GPU buffers for Monte Carlo integration.
    /// Unlike simulation mode, there's no input buffer (samples generated on GPU).
    pub fn setup_integration(
        &mut self,
        n_samples: u64,
        k_functions: usize,
        dist_params: &DistributionParamsBuffer,
        lookup_table: Option<&[f32]>,
        seed: u32,
        target_threads: Option<u32>,
    ) -> Result<DispatchConfig> {
        let config = self.calculate_dispatch_config(n_samples, target_threads);

        // Create integration params uniform buffer
        let integration_params = IntegrationParams {
            n_threads: config.total_threads,
            loops_per_thread: config.loops_per_thread,
            seed,
            k_functions: k_functions as u32,
            _padding: 0,
        };

        self.params_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Integration Params Buffer"),
            size: std::mem::size_of::<IntegrationParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        self.queue.write_buffer(
            self.params_buffer.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&[integration_params]),
        );

        // Create distribution params uniform buffer
        self.dist_params_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Distribution Params Buffer"),
            size: std::mem::size_of::<DistributionParamsBuffer>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        self.queue.write_buffer(
            self.dist_params_buffer.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&[*dist_params]),
        );

        // Create lookup table buffer if provided (for table-based distributions)
        // Otherwise create a small dummy buffer (required by the shader layout)
        if let Some(table_data) = lookup_table {
            self.lookup_table_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Lookup Table Buffer"),
                size: (table_data.len() * std::mem::size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));

            self.queue.write_buffer(
                self.lookup_table_buffer.as_ref().unwrap(),
                0,
                bytemuck::cast_slice(table_data),
            );
        } else {
            // Create a dummy 1-element buffer for non-table distributions
            self.lookup_table_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Dummy Lookup Table Buffer"),
                size: std::mem::size_of::<f32>() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));

            self.queue.write_buffer(
                self.lookup_table_buffer.as_ref().unwrap(),
                0,
                bytemuck::cast_slice(&[0.0f32]),
            );
        }

        // Output buffer: size = total_threads * k_functions (flattened layout)
        let output_size =
            (config.total_threads as usize * k_functions * std::mem::size_of::<f32>()) as u64;
        self.output_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Integration Output Buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));

        // Staging buffer for CPU readback
        self.staging_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Integration Staging Buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        Ok(config)
    }

    /// Create compute pipeline for integration mode
    ///
    /// This sets up the bind group layout with 4 bindings:
    /// - binding 0: IntegrationParams (uniform)
    /// - binding 1: DistributionParams (uniform)
    /// - binding 2: Lookup table (storage, optional)
    /// - binding 3: Output buffer (storage)
    pub fn create_integration_pipeline(&mut self, shader_code: &str) -> Result<()> {
        let shader_module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Integration Compute Shader"),
                source: wgpu::ShaderSource::Wgsl(shader_code.into()),
            });

        // Bind group layout for integration mode
        let entries = vec![
            // Params (uniform) - binding 0
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Distribution params (uniform) - binding 1
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Lookup table (storage, read-only) - binding 2
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Output buffer (storage, read-write) - binding 3
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ];

        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Integration Bind Group Layout"),
                    entries: &entries,
                });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Integration Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        self.compute_pipeline = Some(self.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("Integration Compute Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            },
        ));

        // Create bind group
        let mut bind_entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: self.params_buffer.as_ref().unwrap().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: self
                    .dist_params_buffer
                    .as_ref()
                    .unwrap()
                    .as_entire_binding(),
            },
        ];

        // Add lookup table binding (always present - dummy for non-table distributions)
        bind_entries.push(wgpu::BindGroupEntry {
            binding: 2,
            resource: self
                .lookup_table_buffer
                .as_ref()
                .unwrap()
                .as_entire_binding(),
        });

        bind_entries.push(wgpu::BindGroupEntry {
            binding: 3,
            resource: self.output_buffer.as_ref().unwrap().as_entire_binding(),
        });

        self.bind_group = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Integration Bind Group"),
            layout: &bind_group_layout,
            entries: &bind_entries,
        }));

        Ok(())
    }

    /// Execute integration and return thread results
    ///
    /// Returns raw thread results. CPU-based reduction to final expected values
    /// should be done by the caller.
    pub fn execute_integration(&mut self, workgroup_count: u32) -> Result<Vec<f32>> {
        let pipeline = self
            .compute_pipeline
            .as_ref()
            .context("Pipeline not created")?;
        let bind_group = self.bind_group.as_ref().context("Bind group not created")?;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Integration Compute Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Integration Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        // Copy output to staging buffer
        encoder.copy_buffer_to_buffer(
            self.output_buffer.as_ref().unwrap(),
            0,
            self.staging_buffer.as_ref().unwrap(),
            0,
            self.output_buffer.as_ref().unwrap().size(),
        );

        // Submit commands
        self.queue.submit(std::iter::once(encoder.finish()));

        // Read back results
        let staging_buffer = self.staging_buffer.as_ref().unwrap();
        let buffer_slice = staging_buffer.slice(..);

        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        self.device.poll(wgpu::Maintain::Wait);

        rx.recv()
            .context("Failed to receive map async result")?
            .context("Failed to map buffer")?;

        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        Ok(result)
    }
}
