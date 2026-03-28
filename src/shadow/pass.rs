use bevy_ecs::prelude::*;
use bytemuck::{Pod, Zeroable};
use modul_render::{Operation, OperationBuilder, RenderTargetSource};
use wgpu::{
    Buffer, BufferDescriptor, BufferUsages, CommandEncoder, Device, TextureFormat,
    TextureUsages,
};

use super::gpu::ShadowGpuBuffers;

// --- Resources ---

#[derive(Resource)]
pub struct ShadowConfig {
    pub scale_denominator: u32,
    pub debug_overlay: bool,
}

impl Default for ShadowConfig {
    fn default() -> Self {
        Self {
            scale_denominator: 3,
            debug_overlay: true,
        }
    }
}

#[derive(Resource)]
pub struct SunDirection(pub [f32; 3]);

impl Default for SunDirection {
    fn default() -> Self {
        // ~25 degrees above horizon (65 degrees from vertical)
        let d = glam::Vec3::new(0.3, 0.47, 0.5).normalize();
        Self(d.to_array())
    }
}

#[derive(Resource)]
pub struct PreviousFrameData {
    /// The VP/offset that matches the depth buffer the shadow pass will read.
    pub inv_view_proj: [[f32; 4]; 4],
    pub view_proj: [[f32; 4]; 4],
    pub chunk_offset: [i32; 3],
    pub valid: bool,
    /// Staging: current frame's data, becomes "active" next frame.
    next_inv_view_proj: [[f32; 4]; 4],
    next_view_proj: [[f32; 4]; 4],
    next_chunk_offset: [i32; 3],
    next_valid: bool,
}

impl Default for PreviousFrameData {
    fn default() -> Self {
        Self {
            view_proj: [[0.0; 4]; 4],
            inv_view_proj: [[0.0; 4]; 4],
            chunk_offset: [0; 3],
            valid: false,
            next_view_proj: [[0.0; 4]; 4],
            next_inv_view_proj: [[0.0; 4]; 4],
            next_chunk_offset: [0; 3],
            next_valid: false,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct ShadowPassUniform {
    pub inv_view_proj: [[f32; 4]; 4],
    pub chunk_offset: [i32; 3],
    pub _pad0: i32,
    pub sun_direction: [f32; 3],
    pub max_ray_distance: f32,
    pub lod_count: u32,
    pub grid_size: u32,
    pub _pad1: [u32; 2],
}

// --- Shadow Pass Resources ---

#[derive(Resource)]
pub struct ShadowPassResources {
    pub shadow_mask_texture: wgpu::Texture,
    pub shadow_mask_view: wgpu::TextureView,
    pub shadow_mask_sampler: wgpu::Sampler,
    pub shadow_uniform_buffer: Buffer,
    pub shadow_pipeline: wgpu::RenderPipeline,
    pub depth_bind_group_layout: wgpu::BindGroupLayout,
    pub uniform_bind_group_layout: wgpu::BindGroupLayout,
    pub uniform_bind_group: wgpu::BindGroup,
    pub current_size: (u32, u32),
}

fn create_shadow_mask_texture(
    device: &Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Shadow mask"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: TextureFormat::R8Unorm,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    (texture, view)
}

fn create_depth_bind_group(
    device: &Device,
    layout: &wgpu::BindGroupLayout,
    depth_view: &wgpu::TextureView,
    depth_sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Shadow depth BG"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(depth_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(depth_sampler),
            },
        ],
    })
}

impl ShadowPassResources {
    pub fn new(
        device: &Device,
        shadow_gpu: &ShadowGpuBuffers,
        width: u32,
        height: u32,
        scale: u32,
    ) -> Self {
        let sw = width / scale;
        let sh = height / scale;

        let (shadow_mask_texture, shadow_mask_view) = create_shadow_mask_texture(device, sw, sh);

        let shadow_mask_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Shadow mask sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Uniform bind group (group 0)
        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Shadow uniform BG layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZero::new(
                            std::mem::size_of::<ShadowPassUniform>() as u64,
                        ),
                    },
                    count: None,
                }],
            });

        let shadow_uniform_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Shadow uniform"),
            size: std::mem::size_of::<ShadowPassUniform>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shadow uniform BG"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: shadow_uniform_buffer.as_entire_binding(),
            }],
        });

        // Depth bind group (group 1)
        let depth_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Shadow depth BG layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        // Pipeline
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Shadow pipeline layout"),
            bind_group_layouts: &[
                &uniform_bind_group_layout,
                &depth_bind_group_layout,
                &shadow_gpu.bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shadow shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shadow.wgsl").into()),
        });

        let shadow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Shadow pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_shadow"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_shadow"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: TextureFormat::R8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        });

        Self {
            shadow_mask_texture,
            shadow_mask_view,
            shadow_mask_sampler,
            shadow_uniform_buffer,
            shadow_pipeline,
            depth_bind_group_layout,
            uniform_bind_group_layout,
            uniform_bind_group,
            current_size: (sw, sh),
        }
    }
}

impl ShadowPassResources {
    pub fn resize(&mut self, device: &Device, width: u32, height: u32, scale: u32) {
        let sw = (width / scale).max(1);
        let sh = (height / scale).max(1);
        if (sw, sh) == self.current_size {
            return;
        }
        let (tex, view) = create_shadow_mask_texture(device, sw, sh);
        self.shadow_mask_texture = tex;
        self.shadow_mask_view = view;
        self.current_size = (sw, sh);
    }
}

// --- Systems ---

pub fn update_previous_frame_data(
    camera: Res<crate::Camera>,
    mut prev: ResMut<PreviousFrameData>,
) {
    // Promote staging → active (now matches the depth buffer from last frame)
    prev.inv_view_proj = prev.next_inv_view_proj;
    prev.view_proj = prev.next_view_proj;
    prev.chunk_offset = prev.next_chunk_offset;
    prev.valid = prev.next_valid;

    // Store current frame's data in staging (will be used next frame)
    let uniform = camera.0.uniform();
    prev.next_view_proj = uniform.view_proj;
    prev.next_inv_view_proj = invert_mat4(&uniform.view_proj);
    prev.next_chunk_offset = uniform.chunk_offset;
    prev.next_valid = true;
}

fn invert_mat4(m: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    // Gaussian elimination for 4x4 matrix inverse
    let mut a = *m;
    let mut inv = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];

    for col in 0..4 {
        // Find pivot
        let mut max_val = 0.0f32;
        let mut max_row = col;
        for row in col..4 {
            let v = a[col][row].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }

        // Swap rows
        if max_row != col {
            for c in 0..4 {
                let tmp = a[c][col];
                a[c][col] = a[c][max_row];
                a[c][max_row] = tmp;
                let tmp = inv[c][col];
                inv[c][col] = inv[c][max_row];
                inv[c][max_row] = tmp;
            }
        }

        let pivot = a[col][col];
        if pivot.abs() < 1e-12 {
            return inv; // singular
        }

        // Scale pivot row
        for c in 0..4 {
            a[c][col] /= pivot;
            inv[c][col] /= pivot;
        }

        // Eliminate column
        for row in 0..4 {
            if row == col {
                continue;
            }
            let factor = a[col][row];
            for c in 0..4 {
                a[c][row] -= factor * a[c][col];
                inv[c][row] -= factor * inv[c][col];
            }
        }
    }

    inv
}

// --- Operations ---

pub struct ShadowTraceOperation;

impl Operation for ShadowTraceOperation {
    fn run(&mut self, world: &mut World, command_encoder: &mut CommandEncoder) {
        let prev = world.resource::<PreviousFrameData>();
        if !prev.valid {
            return;
        }

        // Resize shadow mask if window size changed
        {
            let main_window = world
                .query_filtered::<bevy_ecs::prelude::Entity, bevy_ecs::prelude::With<modul_core::MainWindow>>()
                .single(world);
            let size = world
                .get::<modul_render::SurfaceRenderTarget>(main_window)
                .map(|rt| modul_render::RenderTarget::size(rt));
            if let Some((w, h)) = size {
                let scale = world.resource::<ShadowConfig>().scale_denominator;
                world.resource_scope(|world, mut device: bevy_ecs::prelude::Mut<modul_core::DeviceRes>| {
                    world.resource_mut::<ShadowPassResources>().resize(&device.0, w, h, scale);
                });
            }
        }

        // Extract uniform data from resources (release borrows before query)
        let uniform = {
            let prev = world.resource::<PreviousFrameData>();
            let sun = world.resource::<SunDirection>();
            let grid = world.resource::<crate::shadow::grid::ShadowGrid>();
            ShadowPassUniform {
                inv_view_proj: prev.inv_view_proj,
                chunk_offset: prev.chunk_offset,
                _pad0: 0,
                sun_direction: sun.0,
                max_ray_distance: 512.0,
                lod_count: grid.lod_count,
                grid_size: grid.grid_size,
                _pad1: [0; 2],
            }
        };

        // Upload uniform
        {
            let queue = &world.resource::<modul_core::QueueRes>().0;
            let shadow_res = world.resource::<ShadowPassResources>();
            queue.write_buffer(
                &shadow_res.shadow_uniform_buffer,
                0,
                bytemuck::bytes_of(&uniform),
            );
        }

        // Get depth texture view from the surface render target
        let main_window = world
            .query_filtered::<bevy_ecs::prelude::Entity, bevy_ecs::prelude::With<modul_core::MainWindow>>()
            .single(world);

        let rt = world.get::<modul_render::SurfaceRenderTarget>(main_window);
        let Some(rt) = rt else { return };
        let Some(depth_view) = modul_render::RenderTarget::depth_stencil_view(rt) else {
            return;
        };

        let device = &world.resource::<modul_core::DeviceRes>().0;
        let shadow_res = world.resource::<ShadowPassResources>();

        let depth_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Shadow depth sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let depth_bind_group = create_depth_bind_group(
            device,
            &shadow_res.depth_bind_group_layout,
            depth_view,
            &depth_sampler,
        );

        let shadow_gpu = world.resource::<ShadowGpuBuffers>();

        // Render pass on the shadow mask texture
        let mut pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Shadow trace pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &shadow_res.shadow_mask_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        pass.set_pipeline(&shadow_res.shadow_pipeline);
        pass.set_bind_group(0, &shadow_res.uniform_bind_group, &[]);
        pass.set_bind_group(1, &depth_bind_group, &[]);
        pass.set_bind_group(2, &shadow_gpu.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

pub struct ShadowTraceOperationBuilder;

impl OperationBuilder for ShadowTraceOperationBuilder {
    fn reading(&self) -> Vec<RenderTargetSource> {
        Vec::new()
    }
    fn writing(&self) -> Vec<RenderTargetSource> {
        Vec::new()
    }
    fn finish(self, _world: &World, _device: &Device) -> impl Operation + 'static {
        ShadowTraceOperation
    }
}

// --- Debug Overlay ---
// For Phase 3: draws the shadow mask as a red/white overlay on top of the main render target.

pub struct ShadowDebugOverlay {
    pub target: RenderTargetSource,
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl Operation for ShadowDebugOverlay {
    fn run(&mut self, world: &mut World, command_encoder: &mut CommandEncoder) {
        let config = world.resource::<ShadowConfig>();
        if !config.debug_overlay {
            return;
        }

        let shadow_res = world.resource::<ShadowPassResources>();

        let bind_group = world
            .resource::<modul_core::DeviceRes>()
            .0
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Shadow debug BG"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&shadow_res.shadow_mask_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&shadow_res.shadow_mask_sampler),
                    },
                ],
            });

        let Some(mut rt) = self.target.get_mut(world) else {
            return;
        };
        let Some(mut pass) = rt.begin_ending_pass(command_encoder) else {
            return;
        };

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

pub struct ShadowDebugOverlayBuilder {
    pub target: RenderTargetSource,
}

impl OperationBuilder for ShadowDebugOverlayBuilder {
    fn reading(&self) -> Vec<RenderTargetSource> {
        Vec::new()
    }
    fn writing(&self) -> Vec<RenderTargetSource> {
        vec![self.target]
    }
    fn finish(self, _world: &World, device: &Device) -> impl Operation + 'static {
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Shadow debug BG layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Shadow debug pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shadow debug shader"),
            source: wgpu::ShaderSource::Wgsl(SHADOW_DEBUG_WGSL.into()),
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Shadow debug pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_debug"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_debug"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: TextureFormat::Bgra8UnormSrgb, // will be overridden by modul
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent::OVER,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        });

        ShadowDebugOverlay {
            target: self.target,
            pipeline,
            bind_group_layout,
        }
    }
}

const SHADOW_DEBUG_WGSL: &str = "
@group(0) @binding(0)
var shadow_mask: texture_2d<f32>;
@group(0) @binding(1)
var shadow_sampler: sampler;

struct DebugVaryings {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_debug(@builtin(vertex_index) vi: u32) -> DebugVaryings {
    var out: DebugVaryings;
    let uv = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
    out.position = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(uv.x, 1.0 - uv.y);
    return out;
}

@fragment
fn fs_debug(in: DebugVaryings) -> @location(0) vec4<f32> {
    let shadow_val = textureSample(shadow_mask, shadow_sampler, in.uv).r;
    // Red tint for shadow, transparent for lit
    let alpha = (1.0 - shadow_val) * 0.5;
    return vec4<f32>(1.0, 0.0, 0.0, alpha);
}
";
