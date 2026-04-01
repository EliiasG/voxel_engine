use bevy_ecs::prelude::*;
use modul_asset::AssetWorldExt;
use modul_render::{
    BindGroupLayoutDef, Operation, OperationBuilder, RenderTarget, RenderTargetSource,
};
use wgpu::{
    CommandEncoder, Device, TextureFormat, TextureUsages,
};

use crate::render;

// --- Resources ---

#[derive(Resource)]
pub struct TaaResources {
    /// Scene color texture — voxels render here instead of the surface
    pub scene_texture: wgpu::Texture,
    pub scene_view: wgpu::TextureView,
    /// History ping-pong: [0] and [1]
    pub history_textures: [(wgpu::Texture, wgpu::TextureView); 2],
    pub ping_pong_index: u32,
    /// Sampler for reading scene and history textures (linear filter)
    pub sampler: wgpu::Sampler,
    /// TAA resolve pipeline (fullscreen triangle, 2 color targets)
    pub resolve_pipeline: wgpu::RenderPipeline,
    /// Bind group layout for TAA inputs (group 1)
    pub taa_bind_group_layout: wgpu::BindGroupLayout,
    /// Current texture size
    pub current_size: (u32, u32),
    /// Surface format (for texture recreation on resize)
    pub surface_format: TextureFormat,
    /// Previous frame's jittered view_proj (for CameraUniform.prev_jittered_view_proj)
    pub prev_jittered_view_proj: [[f32; 4]; 4],
    pub prev_chunk_offset: [i32; 3],
    pub prev_valid: bool,
}

fn create_color_texture(
    device: &Device,
    width: u32,
    height: u32,
    format: TextureFormat,
    label: &str,
) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

impl TaaResources {
    pub fn new(device: &Device, surface_format: TextureFormat, width: u32, height: u32) -> Self {
        let (scene_texture, scene_view) =
            create_color_texture(device, width, height, surface_format, "TAA scene color");
        let history_textures = [
            create_color_texture(device, width, height, surface_format, "TAA history 0"),
            create_color_texture(device, width, height, surface_format, "TAA history 1"),
        ];

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("TAA sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let taa_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("TAA input BG layout"),
                entries: &[
                    // 0: scene color
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
                    // 1: sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // 2: history texture (previous resolved)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // 3: depth texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });

        let camera_layout =
            device.create_bind_group_layout(render::CameraBGLayout::LAYOUT);
        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("TAA resolve pipeline layout"),
                bind_group_layouts: &[&camera_layout, &taa_bind_group_layout],
                push_constant_ranges: &[],
            });

        // Compose shader: camera library (group 0) + TAA bindings + TAA shader
        let camera_wgsl = render::CameraBGLayout::LIBRARY
            .replace("#BIND_GROUP", "0");
        let taa_bindings = "\
@group(1) @binding(0) var scene_color: texture_2d<f32>;
@group(1) @binding(1) var taa_sampler: sampler;
@group(1) @binding(2) var history_tex: texture_2d<f32>;
@group(1) @binding(3) var depth_tex: texture_depth_2d;
";
        let taa_shader_src = include_str!("shaders/taa.wgsl");
        let full_source = format!("{camera_wgsl}\n{taa_bindings}\n{taa_shader_src}");

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("TAA resolve shader"),
            source: wgpu::ShaderSource::Wgsl(full_source.into()),
        });

        let resolve_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("TAA resolve pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_taa"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_taa"),
                    targets: &[
                        // Attachment 0: surface (display)
                        Some(wgpu::ColorTargetState {
                            format: surface_format,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                        // Attachment 1: history (for next frame)
                        Some(wgpu::ColorTargetState {
                            format: surface_format,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                    ],
                    compilation_options: Default::default(),
                }),
                multiview: None,
                cache: None,
            });

        Self {
            scene_texture,
            scene_view,
            history_textures,
            ping_pong_index: 0,
            sampler,
            resolve_pipeline,
            taa_bind_group_layout,
            current_size: (width, height),
            surface_format,
            prev_jittered_view_proj: [[0.0; 4]; 4],
            prev_chunk_offset: [0; 3],
            prev_valid: false,
        }
    }

    pub fn resize(&mut self, device: &Device, width: u32, height: u32) {
        if self.current_size == (width, height) {
            return;
        }
        let (scene_texture, scene_view) =
            create_color_texture(device, width, height, self.surface_format, "TAA scene color");
        self.scene_texture = scene_texture;
        self.scene_view = scene_view;
        self.history_textures = [
            create_color_texture(device, width, height, self.surface_format, "TAA history 0"),
            create_color_texture(device, width, height, self.surface_format, "TAA history 1"),
        ];
        self.current_size = (width, height);
        self.prev_valid = false;
    }

    pub fn prev_history_view(&self) -> &wgpu::TextureView {
        &self.history_textures[1 - self.ping_pong_index as usize].1
    }

    pub fn current_history_view(&self) -> &wgpu::TextureView {
        &self.history_textures[self.ping_pong_index as usize].1
    }

    pub fn swap(&mut self) {
        self.ping_pong_index = 1 - self.ping_pong_index;
    }
}

// --- TaaVoxelDrawOperation ---
// Renders voxels to the scene texture (using surface depth), bypassing modul's render target.

pub struct TaaVoxelDrawOperation;

impl Operation for TaaVoxelDrawOperation {
    fn run(&mut self, world: &mut World, command_encoder: &mut CommandEncoder) {
        // Resize scene + history if window size changed
        {
            let main_window_entity = world
                .query_filtered::<Entity, With<modul_core::MainWindow>>()
                .single(world);
            let surface_rt = world
                .get::<modul_render::SurfaceRenderTarget>(main_window_entity)
                .unwrap();
            let (w, h) = RenderTarget::size(surface_rt);
            world.resource_scope(|world, mut taa_res: Mut<TaaResources>| {
                let device = &world.resource::<modul_core::DeviceRes>().0;
                taa_res.resize(device, w, h);
            });
        }

        // Extract pointers we need before the render pass (avoids borrow conflicts)
        let surface_format;
        let scene_view_ptr: *const wgpu::TextureView;
        let depth_view_ptr: *const wgpu::TextureView;
        {
            let taa_res = world.resource::<TaaResources>();
            surface_format = taa_res.surface_format;
            scene_view_ptr = &taa_res.scene_view as *const _;
        }
        {
            let main_window_entity = world
                .query_filtered::<Entity, With<modul_core::MainWindow>>()
                .single(world);
            let surface_rt = world
                .get::<modul_render::SurfaceRenderTarget>(main_window_entity)
                .unwrap();
            depth_view_ptr = surface_rt.depth_stencil_view().unwrap() as *const _;
        }
        // SAFETY: The views live in World resources/components, which are stable during run().
        let scene_view = unsafe { &*scene_view_ptr };
        let depth_view = unsafe { &*depth_view_ptr };

        // Create render pass targeting scene_color + surface depth
        let mut pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("TAA voxel draw pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: scene_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.1, g: 0.15, b: 0.25, a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(0.0), // Reversed-Z: 0.0 = far/infinity
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        // Get compatible voxel pipeline for the scene format
        let voxel_pipeline = world.resource::<render::VoxelPipeline>();
        let pipeline_id = if world.get_resource::<render::Wireframe>().is_some_and(|w| w.0) {
            voxel_pipeline.wireframe
        } else {
            voxel_pipeline.fill
        };

        world.asset_scope(
            pipeline_id,
            |world, pipeline_man: &mut modul_render::RenderPipelineManager| {
                let params = modul_render::PipelineParameters {
                    color_format: Some(surface_format),
                    depth_stencil_format: Some(TextureFormat::Depth32Float),
                    sample_count: 1,
                };
                let pipeline = pipeline_man.get(world, &params);
                pass.set_pipeline(pipeline);

                let camera_bg = &world.resource::<render::CameraBindGroup>().bind_group;
                let gpu = world.resource::<render::GpuBuffers>();
                let device = &world.resource::<modul_core::DeviceRes>().0;
                let shadow_res = world.resource::<crate::render::shadow::pass::ShadowPassResources>();
                let shadow_mask_bg = render::create_shadow_mask_bind_group(device, shadow_res);

                let atmo_res = world.resource::<crate::render::atmosphere::AtmosphereResources>();
                let atmo_bg_ptr = &atmo_res.bind_group as *const wgpu::BindGroup;
                let atmo_bg = unsafe { &*atmo_bg_ptr };

                pass.set_bind_group(0, camera_bg, &[]);
                pass.set_bind_group(2, &shadow_mask_bg, &[]);
                pass.set_bind_group(3, atmo_bg, &[]);

                render::draw_voxel_geometry(&mut pass, gpu);
            },
        );
    }
}

pub struct TaaVoxelDrawOperationBuilder;

impl OperationBuilder for TaaVoxelDrawOperationBuilder {
    fn reading(&self) -> Vec<RenderTargetSource> { Vec::new() }
    fn writing(&self) -> Vec<RenderTargetSource> { Vec::new() }
    fn finish(self, _world: &World, _device: &Device) -> impl Operation + 'static {
        TaaVoxelDrawOperation
    }
}

// --- TaaResolveOperation ---
// Fullscreen triangle: reads scene + depth + history[prev], MRT writes to surface + history[current].

pub struct TaaResolveOperation {
    pub surface_entity: Entity,
}

impl Operation for TaaResolveOperation {
    fn run(&mut self, world: &mut World, command_encoder: &mut CommandEncoder) {
        // Handle resize
        {
            let surface_rt = world
                .get::<modul_render::SurfaceRenderTarget>(self.surface_entity)
                .unwrap();
            let (w, h) = RenderTarget::size(surface_rt);
            world.resource_scope(
                |world, mut taa_res: Mut<TaaResources>| {
                    let device = &world.resource::<modul_core::DeviceRes>().0;
                    taa_res.resize(device, w, h);
                },
            );
        }

        // Extract all view pointers before creating bind group / render pass
        let scene_view_ptr: *const wgpu::TextureView;
        let prev_history_ptr: *const wgpu::TextureView;
        let current_history_ptr: *const wgpu::TextureView;
        let sampler_ptr: *const wgpu::Sampler;
        let pipeline_ptr: *const wgpu::RenderPipeline;
        let bg_layout_ptr: *const wgpu::BindGroupLayout;
        let surface_view_ptr: *const wgpu::TextureView;
        let depth_view_ptr: *const wgpu::TextureView;
        {
            let taa_res = world.resource::<TaaResources>();
            scene_view_ptr = &taa_res.scene_view as *const _;
            prev_history_ptr = taa_res.prev_history_view() as *const _;
            current_history_ptr = taa_res.current_history_view() as *const _;
            sampler_ptr = &taa_res.sampler as *const _;
            pipeline_ptr = &taa_res.resolve_pipeline as *const _;
            bg_layout_ptr = &taa_res.taa_bind_group_layout as *const _;
        }
        {
            let surface_rt = world
                .get::<modul_render::SurfaceRenderTarget>(self.surface_entity)
                .unwrap();
            surface_view_ptr = RenderTarget::texture_view(surface_rt).unwrap() as *const _;
            depth_view_ptr = surface_rt.depth_stencil_view().unwrap() as *const _;
        }
        // SAFETY: All views/resources live in World storage, stable for the duration of run().
        let scene_view = unsafe { &*scene_view_ptr };
        let prev_history_view = unsafe { &*prev_history_ptr };
        let current_history_view = unsafe { &*current_history_ptr };
        let taa_sampler = unsafe { &*sampler_ptr };
        let taa_pipeline = unsafe { &*pipeline_ptr };
        let taa_bg_layout = unsafe { &*bg_layout_ptr };
        let surface_view = unsafe { &*surface_view_ptr };
        let depth_view = unsafe { &*depth_view_ptr };

        // Build TAA input bind group
        let device = &world.resource::<modul_core::DeviceRes>().0;
        let taa_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("TAA input BG"),
            layout: taa_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(scene_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(taa_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(prev_history_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
            ],
        });

        let camera_bg = &world.resource::<render::CameraBindGroup>().bind_group;

        // MRT render pass: surface + history[current]
        let mut pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("TAA resolve pass"),
            color_attachments: &[
                // Attachment 0: surface (display)
                Some(wgpu::RenderPassColorAttachment {
                    view: surface_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                }),
                // Attachment 1: history[current] (for next frame)
                Some(wgpu::RenderPassColorAttachment {
                    view: current_history_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                }),
            ],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        pass.set_pipeline(taa_pipeline);
        pass.set_bind_group(0, camera_bg, &[]);
        pass.set_bind_group(1, &taa_bind_group, &[]);
        pass.draw(0..3, 0..1);
        drop(pass);

        // Swap ping-pong for next frame
        world.resource_mut::<TaaResources>().swap();
    }
}

pub struct TaaResolveOperationBuilder {
    pub surface_entity: Entity,
}

impl OperationBuilder for TaaResolveOperationBuilder {
    fn reading(&self) -> Vec<RenderTargetSource> { Vec::new() }
    fn writing(&self) -> Vec<RenderTargetSource> {
        vec![RenderTargetSource::Surface(self.surface_entity)]
    }
    fn finish(self, _world: &World, _device: &Device) -> impl Operation + 'static {
        TaaResolveOperation { surface_entity: self.surface_entity }
    }
}

// --- Init system ---

/// Initializes TAA resources.
pub fn init_taa(
    mut commands: Commands,
    device: Res<modul_core::DeviceRes>,
    surface_fmt: Res<modul_core::SurfaceFormat>,
) {
    let taa_res = TaaResources::new(&device.0, surface_fmt.0, 800, 600);
    commands.insert_resource(taa_res);
}
