use bevy_ecs::prelude::*;
use modul_render::{BindGroupLayoutDef, Operation, OperationBuilder, RenderTarget, RenderTargetSource};
use modul_core::wgpu;
use modul_core::wgpu::{CommandEncoder, Device, TextureFormat, TextureUsages};

use crate::render;

// --- Resources ---

#[derive(Resource)]
pub struct WboitResources {
    /// Accumulation texture (Rgba16Float): stores weighted color sum + weight sum
    pub accum_texture: wgpu::Texture,
    pub accum_view: wgpu::TextureView,
    /// Revealage texture (R8Unorm): stores product of (1 - alpha)
    pub revealage_texture: wgpu::Texture,
    pub revealage_view: wgpu::TextureView,
    /// Resolve pipeline (fullscreen triangle)
    pub resolve_pipeline: wgpu::RenderPipeline,
    /// Bind group layout for resolve inputs
    pub resolve_bind_group_layout: wgpu::BindGroupLayout,
    /// Current size
    pub current_size: (u32, u32),
}

fn create_accum_texture(
    device: &Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("WBOIT accumulation"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: TextureFormat::Rgba16Float,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

fn create_revealage_texture(
    device: &Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("WBOIT revealage"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: TextureFormat::R8Unorm,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

impl WboitResources {
    pub fn new(device: &Device, surface_format: TextureFormat, width: u32, height: u32) -> Self {
        let (accum_texture, accum_view) = create_accum_texture(device, width, height);
        let (revealage_texture, revealage_view) = create_revealage_texture(device, width, height);

        // Resolve bind group layout: accum + revealage textures
        let resolve_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("WBOIT resolve BG layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });

        let resolve_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("WBOIT resolve layout"),
            bind_group_layouts: &[Some(&resolve_bind_group_layout)],
            immediate_size: 0,
        });

        let resolve_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("WBOIT resolve shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/wboit_resolve.wgsl").into(),
            ),
        });

        // Resolve pipeline: alpha-blends transparent result over opaque scene
        let resolve_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("WBOIT resolve pipeline"),
            layout: Some(&resolve_layout),
            vertex: wgpu::VertexState {
                module: &resolve_shader,
                entry_point: Some("vs_resolve"),
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
                module: &resolve_shader,
                entry_point: Some("fs_resolve"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    // Standard alpha blending: transparent over opaque
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
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
        });

        Self {
            accum_texture,
            accum_view,
            revealage_texture,
            revealage_view,
            resolve_pipeline,
            resolve_bind_group_layout,
            current_size: (width, height),
        }
    }

    pub fn resize(&mut self, device: &Device, width: u32, height: u32) {
        if self.current_size == (width, height) {
            return;
        }
        let (accum_texture, accum_view) = create_accum_texture(device, width, height);
        let (revealage_texture, revealage_view) = create_revealage_texture(device, width, height);
        self.accum_texture = accum_texture;
        self.accum_view = accum_view;
        self.revealage_texture = revealage_texture;
        self.revealage_view = revealage_view;
        self.current_size = (width, height);
    }
}

// --- Transparent Pipeline ---

/// Transparent voxel pipeline (WBOIT accumulation). Uses same vertex format as opaque,
/// different fragment shader, different blend states, no depth write.
#[derive(Resource)]
pub struct TransparentVoxelPipeline {
    pub pipeline: wgpu::RenderPipeline,
}

pub fn init_transparent_pipeline(
    device: &Device,
    _surface_format: TextureFormat,
) -> TransparentVoxelPipeline {
    // Build shader: camera BG (0) + metadata BG (1) + atlas BG (2) + shadow mask BG (3) + atmosphere BG (4)
    // + sky_sample + lighting + vertex + transparent fragment
    let camera_wgsl = render::CameraBGLayout::LIBRARY.replace("#BIND_GROUP", "0");
    let metadata_wgsl = render::MetadataBGLayout::LIBRARY.replace("#BIND_GROUP", "1");
    let atlas_wgsl = render::TextureAtlasBGLayout::LIBRARY.replace("#BIND_GROUP", "2");
    let shadow_mask_wgsl = render::ShadowMaskBGLayout::LIBRARY.replace("#BIND_GROUP", "3");
    let atmosphere_wgsl = render::atmosphere::AtmosphereBGLayout::LIBRARY.replace("#BIND_GROUP", "4");
    let sky_sample_wgsl = include_str!("shaders/sky_sample.wgsl");
    let fog_wgsl = include_str!("shaders/fog.wgsl");
    let lighting_wgsl = include_str!("shaders/lighting.wgsl");
    let vertex_wgsl = include_str!("shaders/voxel_vertex.wgsl");
    let transparent_wgsl = include_str!("shaders/voxel_transparent.wgsl");

    let full_source = format!(
        "{camera_wgsl}\n{metadata_wgsl}\n{atlas_wgsl}\n{shadow_mask_wgsl}\n{atmosphere_wgsl}\n\
         {sky_sample_wgsl}\n{fog_wgsl}\n{lighting_wgsl}\n{vertex_wgsl}\n{transparent_wgsl}"
    );

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Transparent voxel shader"),
        source: wgpu::ShaderSource::Wgsl(full_source.into()),
    });

    let camera_layout = device.create_bind_group_layout(render::CameraBGLayout::LAYOUT);
    let metadata_layout = device.create_bind_group_layout(render::MetadataBGLayout::LAYOUT);
    let atlas_layout = device.create_bind_group_layout(render::TextureAtlasBGLayout::LAYOUT);
    let shadow_mask_layout = device.create_bind_group_layout(render::ShadowMaskBGLayout::LAYOUT);
    let atmosphere_layout = device.create_bind_group_layout(render::atmosphere::AtmosphereBGLayout::LAYOUT);

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Transparent voxel pipeline layout"),
        bind_group_layouts: &[
            Some(&camera_layout),
            Some(&metadata_layout),
            Some(&atlas_layout),
            Some(&shadow_mask_layout),
            Some(&atmosphere_layout),
        ],
        immediate_size: 0,
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Transparent voxel pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<crate::chunk::FaceData>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Uint8x4,
                        offset: 0,
                        shader_location: 0,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Uint8x4,
                        offset: 4,
                        shader_location: 1,
                    },
                ],
            }],
            compilation_options: Default::default(),
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None, // No backface culling for transparent geometry
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: TextureFormat::Depth32Float,
            depth_write_enabled: Some(false), // Read depth, don't write
            depth_compare: Some(wgpu::CompareFunction::GreaterEqual), // Reverse-Z
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_transparent"),
            targets: &[
                // Target 0: Accumulation (Rgba16Float) — additive blending
                Some(wgpu::ColorTargetState {
                    format: TextureFormat::Rgba16Float,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                // Target 1: Revealage (R8Unorm) — multiplicative: dst * (1 - src)
                Some(wgpu::ColorTargetState {
                    format: TextureFormat::R8Unorm,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::Zero,
                            dst_factor: wgpu::BlendFactor::OneMinusSrc,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                }),
            ],
            compilation_options: Default::default(),
        }),
        multiview_mask: None,
        cache: None,
    });

    TransparentVoxelPipeline { pipeline }
}

// --- Operations ---

/// Draws transparent voxel geometry into WBOIT accumulation + revealage textures.
/// Reads opaque depth buffer for depth testing (no depth write).
pub struct TransparentDrawOperation;

impl Operation for TransparentDrawOperation {
    fn run(&mut self, world: &mut World, command_encoder: &mut CommandEncoder) {
        let gpu = world.resource::<render::GpuBuffers>();
        if gpu.transparent_draws.is_empty() {
            return;
        }

        // Get surface depth view
        let main_window_entity = world
            .query_filtered::<Entity, With<modul_core::MainWindow>>()
            .single(world)
            .expect("main window not spawned");

        let depth_view_ptr: *const wgpu::TextureView;
        {
            let surface_rt = world
                .get::<modul_render::SurfaceRenderTarget>(main_window_entity)
                .unwrap();
            depth_view_ptr = surface_rt.depth_stencil_view().unwrap() as *const _;
        }

        // Resize WBOIT textures if needed
        {
            let surface_rt = world
                .get::<modul_render::SurfaceRenderTarget>(main_window_entity)
                .unwrap();
            let (w, h) = RenderTarget::size(surface_rt);
            world.resource_scope(|world, mut wboit_res: Mut<WboitResources>| {
                let device = &world.resource::<modul_core::RenderContext>().device;
                wboit_res.resize(device, w, h);
            });
        }

        let wboit_res = world.resource::<WboitResources>();
        let accum_view_ptr = &wboit_res.accum_view as *const _;
        let revealage_view_ptr = &wboit_res.revealage_view as *const _;

        // SAFETY: Views live in World resources, stable during run().
        let depth_view = unsafe { &*depth_view_ptr };
        let accum_view = unsafe { &*accum_view_ptr };
        let revealage_view = unsafe { &*revealage_view_ptr };

        let mut pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Transparent voxel draw pass"),
            color_attachments: &[
                // Accumulation: clear to 0
                Some(wgpu::RenderPassColorAttachment {
                    view: accum_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                // Revealage: clear to 1 (fully transparent = no coverage)
                Some(wgpu::RenderPassColorAttachment {
                    view: revealage_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 1.0, g: 1.0, b: 1.0, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                }),
            ],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load, // Keep opaque depth
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        let trans_pipeline = world.resource::<TransparentVoxelPipeline>();
        pass.set_pipeline(&trans_pipeline.pipeline);

        let camera_bg = &world.resource::<render::CameraBindGroup>().bind_group;
        let atlas_bg = &world.resource::<render::TextureAtlasBindGroup>().bind_group;
        let device = &world.resource::<modul_core::RenderContext>().device;
        let shadow_res = world.resource::<render::shadow::pass::ShadowPassResources>();
        let shadow_mask_bg = render::create_shadow_mask_bind_group(device, shadow_res);
        let atmo_res = world.resource::<render::atmosphere::AtmosphereResources>();
        let atmo_bg_ptr = &atmo_res.bind_group as *const wgpu::BindGroup;
        let atmo_bg = unsafe { &*atmo_bg_ptr };

        pass.set_bind_group(0, camera_bg, &[]);
        pass.set_bind_group(2, atlas_bg, &[]);
        pass.set_bind_group(3, &shadow_mask_bg, &[]);
        pass.set_bind_group(4, atmo_bg, &[]);

        let gpu = world.resource::<render::GpuBuffers>();
        render::draw_transparent_geometry(&mut pass, gpu);
    }
}

pub struct TransparentDrawOperationBuilder;

impl OperationBuilder for TransparentDrawOperationBuilder {
    fn reading(&self) -> Vec<RenderTargetSource> { Vec::new() }
    fn writing(&self) -> Vec<RenderTargetSource> { Vec::new() }
    fn finish(self, _world: &World, _device: &Device) -> impl Operation + 'static {
        TransparentDrawOperation
    }
}

/// Composites WBOIT result over the scene (opaque) framebuffer.
pub struct WboitResolveOperation;

impl Operation for WboitResolveOperation {
    fn run(&mut self, world: &mut World, command_encoder: &mut CommandEncoder) {
        let gpu = world.resource::<render::GpuBuffers>();
        if gpu.transparent_draws.is_empty() {
            return;
        }

        let taa_enabled = world.resource::<render::taa::TaaEnabled>().0;
        let main_window_entity = world
            .query_filtered::<Entity, With<modul_core::MainWindow>>()
            .single(world)
            .expect("main window not spawned");

        // Color target: scene texture (TAA on) or surface (TAA off)
        let color_view_ptr: *const wgpu::TextureView;
        if taa_enabled {
            let taa_res = world.resource::<render::taa::TaaResources>();
            color_view_ptr = &taa_res.scene_view as *const _;
        } else {
            let surface_rt = world
                .get::<modul_render::SurfaceRenderTarget>(main_window_entity)
                .unwrap();
            color_view_ptr = RenderTarget::texture_view(surface_rt).unwrap() as *const _;
        }
        let color_view = unsafe { &*color_view_ptr };

        let wboit_res = world.resource::<WboitResources>();
        let resolve_pipeline_ptr = &wboit_res.resolve_pipeline as *const _;
        let bg_layout_ptr = &wboit_res.resolve_bind_group_layout as *const _;
        let accum_view_ptr = &wboit_res.accum_view as *const _;
        let revealage_view_ptr = &wboit_res.revealage_view as *const _;

        let resolve_pipeline = unsafe { &*resolve_pipeline_ptr };
        let bg_layout = unsafe { &*bg_layout_ptr };
        let accum_view = unsafe { &*accum_view_ptr };
        let revealage_view = unsafe { &*revealage_view_ptr };

        let device = &world.resource::<modul_core::RenderContext>().device;
        let resolve_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("WBOIT resolve BG"),
            layout: bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(accum_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(revealage_view),
                },
            ],
        });

        let mut pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("WBOIT resolve pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load, // Keep opaque scene
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        pass.set_pipeline(resolve_pipeline);
        pass.set_bind_group(0, &resolve_bg, &[]);
        pass.draw(0..3, 0..1);
    }
}

pub struct WboitResolveOperationBuilder;

impl OperationBuilder for WboitResolveOperationBuilder {
    fn reading(&self) -> Vec<RenderTargetSource> { Vec::new() }
    fn writing(&self) -> Vec<RenderTargetSource> { Vec::new() }
    fn finish(self, _world: &World, _device: &Device) -> impl Operation + 'static {
        WboitResolveOperation
    }
}

// --- Init system ---

pub fn init_wboit(
    mut commands: Commands,
    ctx: Res<modul_core::RenderContext>,
    surface_fmt: Res<modul_core::SurfaceFormat>,
) {
    let wboit_res = WboitResources::new(&ctx.device, surface_fmt.0, 800, 600);
    let trans_pipeline = init_transparent_pipeline(&ctx.device, surface_fmt.0);
    commands.insert_resource(wboit_res);
    commands.insert_resource(trans_pipeline);
}
