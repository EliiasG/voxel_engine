pub mod atmosphere;
pub mod shadow;
pub mod taa;

use std::collections::HashMap;

use bevy_ecs::prelude::*;
use bytemuck::{Pod, Zeroable};
use glam::IVec3;
use modul_asset::{AssetId, Assets};
use modul_render::{
    BindGroupLayoutDef, DirectRenderPipelineResourceProvider, GenericDepthStencilState,
    GenericFragmentState, GenericMultisampleState, GenericRenderPipelineDescriptor,
    GenericVertexBufferLayout, GenericVertexState, Operation, OperationBuilder,
    RenderPipelineManager, RenderTargetSource,
};
use wgpu::{
    BlendState, Buffer, BufferDescriptor, BufferUsages, ColorWrites, CommandEncoder,
    CompareFunction, DepthBiasState, Device, FrontFace, PipelineLayout, PipelineLayoutDescriptor,
    PolygonMode, PrimitiveState, PrimitiveTopology, ShaderModule, ShaderModuleDescriptor,
    ShaderSource, StencilState, VertexFormat, VertexStepMode,
};

use crate::chunk::meshing::ChunkFaces;
use crate::chunk::{ChunkLod, ChunkPos, FaceData, LoadedChunkIndex, NUM_DIRECTIONS, DIR_OFFSETS};

pub const PAGE_SIZE: usize = 96;
pub const PAGES_PER_SLAB: usize = 174763; // 128 MB face data per slab
const MAX_INDIRECT: usize = 1024 * 1024; // max draw args across all slabs

// --- GPU Types ---

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PageMetadata {
    pub chunk_x: i32,
    pub chunk_y: i32,
    pub chunk_z: i32,
    pub direction_and_lod: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct DrawIndirectArgs {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub first_vertex: u32,
    pub first_instance: u32,
}

// --- Bind Group Layout Providers ---

pub struct CameraBGLayout;

impl BindGroupLayoutDef for CameraBGLayout {
    const LAYOUT: &'static wgpu::BindGroupLayoutDescriptor<'static> =
        &wgpu::BindGroupLayoutDescriptor {
            label: Some("Camera BG Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: std::num::NonZero::new(
                        std::mem::size_of::<crate::camera::CameraUniform>() as u64,
                    ),
                },
                count: None,
            }],
        };

    const LIBRARY: &'static str = "\
struct CameraUniform {
    view_proj: mat4x4<f32>,
    chunk_offset: vec3<i32>,
    _pad: i32,
    screen_size: vec2<f32>,
    jitter_offset: vec2<f32>,
    inv_view_proj: mat4x4<f32>,
    prev_jittered_view_proj: mat4x4<f32>,
    prev_chunk_offset: vec3<i32>,
    frame_index: u32,
    camera_local_pos: vec3<f32>,
    _pad4: f32,
};

@group(#BIND_GROUP) @binding(0)
var<uniform> camera: CameraUniform;
";
}

pub struct MetadataBGLayout;

impl BindGroupLayoutDef for MetadataBGLayout {
    const LAYOUT: &'static wgpu::BindGroupLayoutDescriptor<'static> =
        &wgpu::BindGroupLayoutDescriptor {
            label: Some("Metadata BG Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: std::num::NonZero::new(
                        std::mem::size_of::<PageMetadata>() as u64,
                    ),
                },
                count: None,
            }],
        };

    const LIBRARY: &'static str = "\
struct PageMetadata {
    chunk_x: i32,
    chunk_y: i32,
    chunk_z: i32,
    direction_and_lod: u32,
};

@group(#BIND_GROUP) @binding(0)
var<storage, read> metadata: array<PageMetadata>;
";
}

pub struct ShadowMaskBGLayout;

impl BindGroupLayoutDef for ShadowMaskBGLayout {
    const LAYOUT: &'static wgpu::BindGroupLayoutDescriptor<'static> =
        &wgpu::BindGroupLayoutDescriptor {
            label: Some("Shadow Mask BG Layout"),
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
            ],
        };

    const LIBRARY: &'static str = "\
@group(#BIND_GROUP) @binding(0)
var shadow_mask: texture_2d<f32>;
@group(#BIND_GROUP) @binding(1)
var shadow_sampler: sampler;
@group(#BIND_GROUP) @binding(2)
var shadow_normal: texture_2d<f32>;
";
}

// --- Slab ---

pub struct Slab {
    pub face_buffer: Buffer,
    pub metadata_buffer: Buffer,
    pub metadata_bind_group: wgpu::BindGroup,
    free_list: Vec<u32>,
}

impl Slab {
    fn new(device: &Device, metadata_bg_layout: &wgpu::BindGroupLayout, index: usize) -> Self {
        let face_buffer = device.create_buffer(&BufferDescriptor {
            label: Some(&format!("Face buffer slab {index}")),
            size: PAGES_PER_SLAB as u64 * PAGE_SIZE as u64 * std::mem::size_of::<FaceData>() as u64,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let metadata_buffer = device.create_buffer(&BufferDescriptor {
            label: Some(&format!("Metadata buffer slab {index}")),
            size: PAGES_PER_SLAB as u64 * std::mem::size_of::<PageMetadata>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let metadata_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("Metadata BG slab {index}")),
            layout: metadata_bg_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: metadata_buffer.as_entire_binding(),
            }],
        });

        Self {
            face_buffer,
            metadata_buffer,
            metadata_bind_group,
            free_list: (0..PAGES_PER_SLAB as u32).rev().collect(),
        }
    }

    fn allocate(&mut self) -> Option<u32> {
        self.free_list.pop()
    }

    fn deallocate(&mut self, page: u32) {
        self.free_list.push(page);
    }

    fn used_count(&self) -> usize {
        PAGES_PER_SLAB - self.free_list.len()
    }

    fn is_full(&self) -> bool {
        self.free_list.is_empty()
    }
}

// --- Resources ---

/// Draw range within the indirect buffer for one slab+LOD combination.
pub struct SlabLodDraw {
    pub slab_index: usize,
    pub offset: u64,
    pub count: u32,
}

#[derive(Resource)]
pub struct GpuBuffers {
    pub slabs: Vec<Slab>,
    pub metadata_bind_group_layout: wgpu::BindGroupLayout,
    pub indirect_buffer: Buffer,
    pub draws: Vec<SlabLodDraw>,
    pub frustum_culled: u32,
}

impl GpuBuffers {
    fn add_slab(&mut self, device: &Device) -> usize {
        let index = self.slabs.len();
        self.slabs.push(Slab::new(device, &self.metadata_bind_group_layout, index));
        println!("Created slab {index} ({}MB each)", PAGES_PER_SLAB * PAGE_SIZE * 8 / (1024 * 1024));
        index
    }
}

/// Allocates pages across slabs. Grows by adding new slabs on demand.
#[derive(Resource)]
pub struct PageAllocator {
    // Thin wrapper -- actual free lists live in slabs
}

impl PageAllocator {
    pub fn new() -> Self {
        Self {}
    }

    /// Allocate a page, returns (slab_index, page_index_within_slab).
    pub fn allocate(gpu: &mut GpuBuffers, device: &Device) -> (usize, u32) {
        // Try existing slabs
        for (i, slab) in gpu.slabs.iter_mut().enumerate() {
            if let Some(page) = slab.allocate() {
                return (i, page);
            }
        }
        // All full -- create new slab
        let i = gpu.add_slab(device);
        let page = gpu.slabs[i].allocate().expect("fresh slab should have pages");
        (i, page)
    }

    pub fn deallocate(gpu: &mut GpuBuffers, slab_index: usize, page_index: u32) {
        gpu.slabs[slab_index].deallocate(page_index);
    }

    pub fn total_used(gpu: &GpuBuffers) -> usize {
        gpu.slabs.iter().map(|s| s.used_count()).sum()
    }

    pub fn total_capacity(gpu: &GpuBuffers) -> usize {
        gpu.slabs.len() * PAGES_PER_SLAB
    }
}

pub struct AllocatedPage {
    pub slab_index: u16,
    pub page_index: u32,
    pub face_count: u32,
}

pub struct DirectionPages {
    pub pages: Vec<AllocatedPage>,
    pub standard_faces: u32,
    pub total_faces: u32,
}

pub struct ChunkRenderEntry {
    pub chunk_pos: IVec3,
    pub lod: u8,
    pub directions: [DirectionPages; NUM_DIRECTIONS],
}

#[derive(Resource, Default)]
pub struct ChunkRenderData {
    pub entries: HashMap<Entity, ChunkRenderEntry>,
}

#[derive(Resource)]
pub struct CameraBindGroup {
    pub buffer: Buffer,
    pub bind_group: wgpu::BindGroup,
}

#[derive(Resource)]
pub struct Wireframe(pub bool);

// --- Geometry Pipeline ---

/// Pipeline set for a geometry type: full (lit), wireframe, and normal-only variants.
pub struct GeometryPipeline {
    pub fill: AssetId<RenderPipelineManager>,
    pub wireframe: AssetId<RenderPipelineManager>,
    pub normal: AssetId<RenderPipelineManager>,
}

/// Builder for creating a [`GeometryPipeline`] from geometry-specific shaders and bind groups.
/// Automatically appends shadow mask + atmosphere bind groups and shared lighting snippets
/// for the full pipeline, and fs_normal for the normal-only pipeline.
pub struct GeometryPipelineBuilder<'a> {
    label: &'a str,
    vertex_source: &'a str,
    material_source: &'a str,
    bind_group_libraries: Vec<String>,
    bind_group_layouts: Vec<wgpu::BindGroupLayout>,
    vertex_buffers: Vec<GenericVertexBufferLayout>,
}

impl<'a> GeometryPipelineBuilder<'a> {
    pub fn new(label: &'a str) -> Self {
        Self {
            label,
            vertex_source: "",
            material_source: "",
            bind_group_libraries: Vec::new(),
            bind_group_layouts: Vec::new(),
            vertex_buffers: Vec::new(),
        }
    }

    /// Set the vertex shader source (shared between full and normal variants).
    pub fn vertex_shader(mut self, source: &'a str) -> Self {
        self.vertex_source = source;
        self
    }

    /// Set the material evaluation + fs_main source (full pipeline only).
    pub fn material_shader(mut self, source: &'a str) -> Self {
        self.material_source = source;
        self
    }

    /// Add a bind group (layout + WGSL library). Groups are numbered starting at 0 in the
    /// order they are added. Shadow mask and atmosphere are appended automatically for the
    /// full pipeline.
    pub fn add_bind_group(mut self, device: &Device, def_layout: &wgpu::BindGroupLayoutDescriptor, library: &str) -> Self {
        let group_index = self.bind_group_libraries.len();
        self.bind_group_libraries.push(library.replace("#BIND_GROUP", &group_index.to_string()));
        self.bind_group_layouts.push(device.create_bind_group_layout(def_layout));
        self
    }

    /// Add a vertex buffer layout.
    pub fn vertex_buffer(mut self, layout: GenericVertexBufferLayout) -> Self {
        self.vertex_buffers.push(layout);
        self
    }

    /// Build the geometry pipeline set (fill, wireframe, normal).
    pub fn build(
        self,
        device: &Device,
        pipelines: &mut Assets<RenderPipelineManager>,
        shaders: &mut Assets<ShaderModule>,
        layouts: &mut Assets<PipelineLayout>,
    ) -> GeometryPipeline {
        let geometry_bg_count = self.bind_group_libraries.len();
        let geometry_bg_wgsl: String = self.bind_group_libraries.join("\n");

        // Full shader: geometry BGs + shadow mask BG + atmosphere BG + shared snippets + vertex + material
        let shadow_mask_index = geometry_bg_count;
        let atmosphere_index = geometry_bg_count + 1;
        let shadow_mask_wgsl = ShadowMaskBGLayout::LIBRARY.replace("#BIND_GROUP", &shadow_mask_index.to_string());
        let atmosphere_wgsl = atmosphere::AtmosphereBGLayout::LIBRARY.replace("#BIND_GROUP", &atmosphere_index.to_string());
        let sky_sample_wgsl = include_str!("shaders/sky_sample.wgsl");
        let lighting_wgsl = include_str!("shaders/lighting.wgsl");
        let full_source = format!(
            "{geometry_bg_wgsl}\n{shadow_mask_wgsl}\n{atmosphere_wgsl}\n{sky_sample_wgsl}\n{lighting_wgsl}\n{}\n{}",
            self.vertex_source, self.material_source,
        );
        let full_shader = shaders.add(device.create_shader_module(ShaderModuleDescriptor {
            label: Some(&format!("{} shader (full)", self.label)),
            source: ShaderSource::Wgsl(full_source.into()),
        }));

        // Normal shader: geometry BGs + vertex + fs_normal
        let fs_normal_src = include_str!("shaders/fs_normal.wgsl");
        let normal_source = format!("{geometry_bg_wgsl}\n{}\n{fs_normal_src}", self.vertex_source);
        let normal_shader = shaders.add(device.create_shader_module(ShaderModuleDescriptor {
            label: Some(&format!("{} shader (normal)", self.label)),
            source: ShaderSource::Wgsl(normal_source.into()),
        }));

        // Full layout: geometry BGs + shadow mask + atmosphere
        let shadow_mask_layout = device.create_bind_group_layout(ShadowMaskBGLayout::LAYOUT);
        let atmosphere_layout = device.create_bind_group_layout(atmosphere::AtmosphereBGLayout::LAYOUT);
        let mut full_layouts: Vec<&wgpu::BindGroupLayout> = self.bind_group_layouts.iter().collect();
        full_layouts.push(&shadow_mask_layout);
        full_layouts.push(&atmosphere_layout);
        let full_layout = layouts.add(device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some(&format!("{} pipeline layout (full)", self.label)),
            bind_group_layouts: &full_layouts,
            push_constant_ranges: &[],
        }));

        // Normal layout: geometry BGs only
        let normal_bg_layouts: Vec<&wgpu::BindGroupLayout> = self.bind_group_layouts.iter().collect();
        let normal_layout = layouts.add(device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some(&format!("{} pipeline layout (normal)", self.label)),
            bind_group_layouts: &normal_bg_layouts,
            push_constant_ranges: &[],
        }));

        let make_desc = |shader, layout, polygon_mode, label: &str, frag_entry: &str| {
            GenericRenderPipelineDescriptor {
                resource_provider: Box::new(DirectRenderPipelineResourceProvider {
                    layout,
                    vertex_shader_module: shader,
                    fragment_shader_module: shader,
                }),
                label: Some(label.into()),
                vertex_state: GenericVertexState {
                    entry_point: "vs_main".into(),
                    buffers: self.vertex_buffers.clone(),
                },
                primitive: PrimitiveState {
                    topology: PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    unclipped_depth: false,
                    polygon_mode,
                    conservative: false,
                },
                depth_stencil: Some(GenericDepthStencilState {
                    depth_write_enable: true,
                    depth_compare: CompareFunction::GreaterEqual,
                    stencil: StencilState::default(),
                    bias: DepthBiasState::default(),
                }),
                multisample: GenericMultisampleState {
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                fragment: Some(GenericFragmentState {
                    entry_point: frag_entry.into(),
                    target_blend: Some(BlendState::REPLACE),
                    target_color_writes: ColorWrites::ALL,
                }),
            }
        };

        let fill = pipelines.add(RenderPipelineManager::new(make_desc(
            full_shader, full_layout, PolygonMode::Fill,
            &format!("{} fill pipeline", self.label), "fs_main",
        )));
        let wireframe = pipelines.add(RenderPipelineManager::new(make_desc(
            full_shader, full_layout, PolygonMode::Line,
            &format!("{} wireframe pipeline", self.label), "fs_main",
        )));
        let normal = pipelines.add(RenderPipelineManager::new(make_desc(
            normal_shader, normal_layout, PolygonMode::Fill,
            &format!("{} normal pipeline", self.label), "fs_normal",
        )));

        GeometryPipeline { fill, wireframe, normal }
    }
}

// --- Voxel Pipeline (uses GeometryPipeline) ---

#[derive(Resource)]
pub struct VoxelPipeline(pub GeometryPipeline);

// --- Initialization ---

pub fn create_gpu_buffers(device: &Device) -> GpuBuffers {
    let metadata_bind_group_layout =
        device.create_bind_group_layout(MetadataBGLayout::LAYOUT);

    let indirect_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("Indirect buffer"),
        size: MAX_INDIRECT as u64 * std::mem::size_of::<DrawIndirectArgs>() as u64,
        usage: BufferUsages::INDIRECT | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut gpu = GpuBuffers {
        slabs: Vec::new(),
        metadata_bind_group_layout,
        indirect_buffer,
        draws: Vec::new(),
        frustum_culled: 0,
    };

    // Start with one slab
    gpu.add_slab(device);

    gpu
}

pub fn create_camera_bind_group(device: &Device) -> CameraBindGroup {
    let layout = device.create_bind_group_layout(CameraBGLayout::LAYOUT);

    let buffer = device.create_buffer(&BufferDescriptor {
        label: Some("Camera uniform"),
        size: std::mem::size_of::<crate::camera::CameraUniform>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Camera BG"),
        layout: &layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });

    CameraBindGroup { buffer, bind_group }
}

pub fn init_voxel_pipeline(
    device: &Device,
    pipelines: &mut Assets<RenderPipelineManager>,
    shaders: &mut Assets<ShaderModule>,
    layouts: &mut Assets<PipelineLayout>,
) -> VoxelPipeline {
    let pipeline = GeometryPipelineBuilder::new("Voxel")
        .vertex_shader(include_str!("shaders/voxel_vertex.wgsl"))
        .material_shader(include_str!("shaders/voxel.wgsl"))
        .add_bind_group(device, CameraBGLayout::LAYOUT, CameraBGLayout::LIBRARY)
        .add_bind_group(device, MetadataBGLayout::LAYOUT, MetadataBGLayout::LIBRARY)
        .vertex_buffer(GenericVertexBufferLayout {
            array_stride: std::mem::size_of::<FaceData>() as u64,
            step_mode: VertexStepMode::Instance,
            attributes: vec![
                wgpu::VertexAttribute {
                    format: VertexFormat::Uint8x4,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: VertexFormat::Uint8x4,
                    offset: 4,
                    shader_location: 1,
                },
            ],
        })
        .build(device, pipelines, shaders, layouts);

    VoxelPipeline(pipeline)
}

// --- Synchronize System ---

fn is_fully_covered(chunk_pos: IVec3, lod: u8, index: &LoadedChunkIndex) -> bool {
    if lod == 0 {
        return false;
    }
    let child_lod = lod - 1;
    for dx in 0..2i32 {
        for dy in 0..2i32 {
            for dz in 0..2i32 {
                let child = chunk_pos * 2 + IVec3::new(dx, dy, dz);
                if !index.0.contains(&(child, child_lod)) {
                    return false;
                }
            }
        }
    }
    true
}

pub fn synchronize_gpu(
    mut commands: Commands,
    query: Query<(Entity, &ChunkPos, &ChunkLod, &ChunkFaces)>,
    mut render_data: ResMut<ChunkRenderData>,
    mut loaded_index: ResMut<LoadedChunkIndex>,
    _allocator: Res<PageAllocator>,
    mut gpu: ResMut<GpuBuffers>,
    device: Res<modul_core::DeviceRes>,
    queue: Res<modul_core::QueueRes>,
    config: Res<crate::chunk::loading::LoadConfig>,
    debug: Res<crate::DebugMode>,
    cam_query: Query<(&crate::camera::Position, &crate::camera::Camera), With<crate::camera::MainCamera>>,
) {
    for (entity, pos, lod, faces) in query.iter() {
        // Deallocate old pages
        if let Some(old) = render_data.entries.remove(&entity) {
            for dir_pages in &old.directions {
                for page in &dir_pages.pages {
                    PageAllocator::deallocate(&mut gpu, page.slab_index as usize, page.page_index);
                }
            }
        }

        let meta_base = PageMetadata {
            chunk_x: pos.0.x,
            chunk_y: pos.0.y,
            chunk_z: pos.0.z,
            direction_and_lod: 0,
        };

        let directions = std::array::from_fn(|dir| {
            let dir_faces = &faces.0[dir];
            let standard_count = dir_faces.standard.len() as u32;
            let total_count = standard_count + dir_faces.border.len() as u32;

            let combined: Vec<FaceData> = dir_faces
                .standard
                .iter()
                .chain(dir_faces.border.iter())
                .copied()
                .collect();

            let mut pages = Vec::new();
            for face_chunk in combined.chunks(PAGE_SIZE) {
                let (slab_idx, page_idx) = PageAllocator::allocate(&mut gpu, &device.0);

                let slab = &gpu.slabs[slab_idx];
                let face_offset =
                    page_idx as u64 * PAGE_SIZE as u64 * std::mem::size_of::<FaceData>() as u64;
                queue
                    .0
                    .write_buffer(&slab.face_buffer, face_offset, bytemuck::cast_slice(face_chunk));

                let dir_meta = PageMetadata {
                    direction_and_lod: (dir as u32) | ((lod.0 as u32) << 8),
                    ..meta_base
                };
                let meta_offset = page_idx as u64 * std::mem::size_of::<PageMetadata>() as u64;
                queue
                    .0
                    .write_buffer(&slab.metadata_buffer, meta_offset, bytemuck::bytes_of(&dir_meta));

                pages.push(AllocatedPage {
                    slab_index: slab_idx as u16,
                    page_index: page_idx,
                    face_count: face_chunk.len() as u32,
                });
            }

            DirectionPages {
                pages,
                standard_faces: standard_count,
                total_faces: total_count,
            }
        });

        loaded_index.0.insert((pos.0, lod.0));

        render_data.entries.insert(
            entity,
            ChunkRenderEntry {
                chunk_pos: pos.0,
                lod: lod.0,
                directions,
            },
        );
        commands.entity(entity).remove::<ChunkFaces>();
    }

    // Rebuild indirect buffer: group draw args by (slab, lod)
    {
        let lod_count = config.lod_count as usize;
        let slab_count = gpu.slabs.len();

        // Compute frustum planes for culling (frozen in debug mode)
        let (frustum_planes, frustum_chunk_offset, cam_world) = if let Some(ref f) = debug.frozen {
            (f.planes, f.chunk_pos, f.camera_world)
        } else if let Ok((pos, cam)) = cam_query.get_single() {
            (
                crate::camera::extract_frustum_planes(&cam.view_proj),
                IVec3::from_array(cam.chunk_offset),
                pos.0,
            )
        } else {
            return;
        };

        // args_per_slab_lod[slab][lod] = Vec<DrawIndirectArgs>
        let mut args_per_slab_lod: Vec<Vec<Vec<DrawIndirectArgs>>> = (0..slab_count)
            .map(|_| (0..lod_count).map(|_| Vec::new()).collect())
            .collect();

        let mut frustum_culled = 0u32;

        for entry in render_data.entries.values() {
            if entry.lod > 0
                && is_fully_covered(entry.chunk_pos, entry.lod, &loaded_index)
            {
                continue;
            }

            // Frustum culling: test chunk AABB against frustum planes
            let lod_scale = 1i32 << entry.lod;
            let rel = entry.chunk_pos * lod_scale - frustum_chunk_offset;
            let cs = crate::chunk::CHUNK_SIZE as f32;
            let min = [rel.x as f32 * cs, rel.y as f32 * cs, rel.z as f32 * cs];
            let extent = lod_scale as f32 * cs;
            let max = [min[0] + extent, min[1] + extent, min[2] + extent];
            if !crate::camera::is_aabb_in_frustum(&frustum_planes, min, max) {
                frustum_culled += 1;
                continue;
            }

            // World-space chunk bounds for direction culling
            let cs_d = crate::chunk::CHUNK_SIZE as f64;
            let lod_scale_d = lod_scale as f64;
            let chunk_min_w = [
                entry.chunk_pos.x as f64 * lod_scale_d * cs_d,
                entry.chunk_pos.y as f64 * lod_scale_d * cs_d,
                entry.chunk_pos.z as f64 * lod_scale_d * cs_d,
            ];
            let w_extent = lod_scale_d * cs_d;
            let chunk_max_w = [
                chunk_min_w[0] + w_extent,
                chunk_min_w[1] + w_extent,
                chunk_min_w[2] + w_extent,
            ];

            let lod = (entry.lod as usize).min(lod_count - 1);

            for (dir, dir_pages) in entry.directions.iter().enumerate() {
                if dir_pages.total_faces == 0 {
                    continue;
                }

                // Direction culling: skip direction groups whose faces are
                // all back-facing (camera is on the opposite side of the chunk).
                let backface = match dir {
                    0 => cam_world[0] <= chunk_min_w[0], // +X faces, cam to -X
                    1 => cam_world[0] >= chunk_max_w[0], // -X faces, cam to +X
                    2 => cam_world[1] <= chunk_min_w[1], // +Y faces, cam below
                    3 => cam_world[1] >= chunk_max_w[1], // -Y faces, cam above
                    4 => cam_world[2] <= chunk_min_w[2], // +Z faces, cam to -Z
                    5 => cam_world[2] >= chunk_max_w[2], // -Z faces, cam to +Z
                    _ => false,
                };
                if backface {
                    continue;
                }

                let neighbor_pos = entry.chunk_pos + DIR_OFFSETS[dir];
                let neighbor_covered = entry.lod > 0
                    && is_fully_covered(neighbor_pos, entry.lod, &loaded_index);
                let face_limit = if neighbor_covered {
                    dir_pages.total_faces
                } else {
                    dir_pages.standard_faces
                };

                if face_limit == 0 {
                    continue;
                }

                let mut faces_remaining = face_limit;
                for page in &dir_pages.pages {
                    if faces_remaining == 0 {
                        break;
                    }
                    let count = page.face_count.min(faces_remaining);
                    let slab = page.slab_index as usize;

                    // Grow slab_lod vec if new slabs were added
                    while args_per_slab_lod.len() <= slab {
                        args_per_slab_lod.push((0..lod_count).map(|_| Vec::new()).collect());
                    }

                    args_per_slab_lod[slab][lod].push(DrawIndirectArgs {
                        vertex_count: 6,
                        instance_count: count,
                        first_vertex: 0,
                        first_instance: page.page_index * PAGE_SIZE as u32,
                    });
                    faces_remaining -= count;
                }
            }
        }

        // Write all args to the indirect buffer, recording (slab, offset, count) per group.
        // Draw order: LOD 0 across all slabs, then LOD 1, etc.
        let mut draws = Vec::new();
        let mut offset = 0u64;
        let stride = std::mem::size_of::<DrawIndirectArgs>() as u64;

        for lod in 0..lod_count {
            for (slab_idx, slab_lods) in args_per_slab_lod.iter().enumerate() {
                let args = &slab_lods[lod];
                if args.is_empty() {
                    continue;
                }
                let byte_offset = offset;
                queue
                    .0
                    .write_buffer(&gpu.indirect_buffer, byte_offset, bytemuck::cast_slice(args));
                draws.push(SlabLodDraw {
                    slab_index: slab_idx,
                    offset: byte_offset,
                    count: args.len() as u32,
                });
                offset += args.len() as u64 * stride;
            }
        }

        gpu.draws = draws;
        gpu.frustum_culled = frustum_culled;
    }
}

// --- Operations ---

pub struct ClearAll {
    pub render_target: RenderTargetSource,
}

impl Operation for ClearAll {
    fn run(&mut self, world: &mut World, _command_encoder: &mut CommandEncoder) {
        if let Some(mut rt) = self.render_target.get_mut(world) {
            rt.schedule_clear_color();
            rt.schedule_clear_depth();
        }
    }
}

impl OperationBuilder for ClearAll {
    fn reading(&self) -> Vec<RenderTargetSource> { Vec::new() }
    fn writing(&self) -> Vec<RenderTargetSource> { Vec::new() }
    fn finish(self, _world: &World, _device: &Device) -> impl Operation + 'static { self }
}

/// Shared voxel draw loop: iterates slabs, sets vertex buffers and metadata bind groups,
/// issues multi_draw_indirect calls. The caller must set the pipeline and any other bind
/// groups (camera, shadow mask, atmosphere) before calling this.
pub fn draw_voxel_geometry(pass: &mut wgpu::RenderPass, gpu: &GpuBuffers) {
    let mut current_slab = usize::MAX;
    for draw in &gpu.draws {
        if draw.slab_index != current_slab {
            current_slab = draw.slab_index;
            let slab = &gpu.slabs[current_slab];
            pass.set_vertex_buffer(0, slab.face_buffer.slice(..));
            pass.set_bind_group(1, &slab.metadata_bind_group, &[]);
        }
        pass.multi_draw_indirect(&gpu.indirect_buffer, draw.offset, draw.count);
    }
}

// --- Init system ---

/// Initializes core render resources: GPU buffers, camera bind group, voxel pipelines.
pub fn init_render(
    mut commands: Commands,
    device: Res<modul_core::DeviceRes>,
    mut shaders: ResMut<Assets<ShaderModule>>,
    mut layouts: ResMut<Assets<PipelineLayout>>,
    mut pipelines: ResMut<Assets<RenderPipelineManager>>,
) {
    let gpu_buffers = create_gpu_buffers(&device.0);
    let camera_bg = create_camera_bind_group(&device.0);
    let voxel_pipeline = init_voxel_pipeline(&device.0, &mut pipelines, &mut shaders, &mut layouts);

    commands.insert_resource(gpu_buffers);
    commands.insert_resource(camera_bg);
    commands.insert_resource(voxel_pipeline);
    commands.insert_resource(PageAllocator::new());
    commands.insert_resource(Wireframe(false));
}

/// Create a shadow mask bind group from the current shadow pass state.
pub fn create_shadow_mask_bind_group(
    device: &wgpu::Device,
    shadow_res: &shadow::pass::ShadowPassResources,
) -> wgpu::BindGroup {
    let layout = device.create_bind_group_layout(ShadowMaskBGLayout::LAYOUT);
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Shadow mask BG"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(shadow_res.prev_view()),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&shadow_res.shadow_mask_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&shadow_res.shadow_normal_view),
            },
        ],
    })
}
