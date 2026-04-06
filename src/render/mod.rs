pub mod atmosphere;
pub mod shadow;
pub mod taa;
pub mod wboit;

use std::collections::{HashMap, HashSet};

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

use crate::chunk::meshing::{ChunkFaces, TransparentChunkFaces};
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

pub struct TextureAtlasBGLayout;

impl BindGroupLayoutDef for TextureAtlasBGLayout {
    const LAYOUT: &'static wgpu::BindGroupLayoutDescriptor<'static> =
        &wgpu::BindGroupLayoutDescriptor {
            label: Some("Texture Atlas BG Layout"),
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
        };

    const LIBRARY: &'static str = "\
@group(#BIND_GROUP) @binding(0)
var atlas_texture: texture_2d<f32>;
@group(#BIND_GROUP) @binding(1)
var atlas_sampler: sampler;
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
    pub transparent_draws: Vec<SlabLodDraw>,
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
    pub transparent_directions: [DirectionPages; NUM_DIRECTIONS],
}

/// Pre-built draw args for one (chunk, direction) pair.
/// Camera-independent — backface culling is stored separately and updated incrementally.
pub struct CachedDraw {
    pub entity: Entity,
    pub chunk_pos: IVec3,
    pub chunk_lod: u8,
    pub dir: u8,
    /// Entry should be removed (chunk re-meshed, unloaded, or parent fully covered).
    /// Skipped during frustum cull, removed during compaction.
    pub dead: bool,
    /// Currently backface-culled by camera position. Updated on camera chunk crossings.
    pub backface_culled: bool,
    pub rel_chunk_pos: IVec3,
    pub lod_scale: i32,
    pub slab_index: usize,
    pub lod: usize,
    pub args: Vec<DrawIndirectArgs>,
}

/// Cached draw list: rebuilt on player chunk change or chunk upload/unload.
/// Per-frame work is just frustum culling + copying pre-built args.
#[derive(Resource, Default)]
pub struct DrawCache {
    pub entries: Vec<CachedDraw>,
    /// Player chunk when cache was last built.
    pub last_camera_chunk: Option<IVec3>,
    /// Incremented when chunks are uploaded or removed.
    pub generation: u64,
    /// Generation when cache was last built.
    pub cached_generation: u64,
}

/// Draw cache for transparent geometry. Same pattern as opaque DrawCache.
#[derive(Resource, Default)]
pub struct TransparentDrawCache(pub DrawCache);

#[derive(Resource, Default)]
pub struct ChunkRenderData {
    pub dirty: bool,
    pub entries: HashMap<Entity, ChunkRenderEntry>,
    /// (Entity, chunk_pos, lod) removed this frame — drained by update_draw_cache.
    /// Used to drop cache entries AND re-add parent LOD chunks that became uncovered.
    pub removed_entities: Vec<(Entity, IVec3, u8)>,
}

#[derive(Resource)]
pub struct CameraBindGroup {
    pub buffer: Buffer,
    pub bind_group: wgpu::BindGroup,
}

#[derive(Resource)]
pub struct TextureAtlasBindGroup {
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
        let fog_wgsl = include_str!("shaders/fog.wgsl");
        let lighting_wgsl = include_str!("shaders/lighting.wgsl");
        let full_source = format!(
            "{geometry_bg_wgsl}\n{shadow_mask_wgsl}\n{atmosphere_wgsl}\n{sky_sample_wgsl}\n{fog_wgsl}\n{lighting_wgsl}\n{}\n{}",
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
        transparent_draws: Vec::new(),
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
        .add_bind_group(device, TextureAtlasBGLayout::LAYOUT, TextureAtlasBGLayout::LIBRARY)
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

/// Upload one set of direction faces to GPU pages. Returns DirectionPages per direction.
fn upload_direction_faces(
    dir_faces_array: &[crate::chunk::meshing::DirFaces; NUM_DIRECTIONS],
    meta_base: &PageMetadata,
    lod: u8,
    gpu: &mut GpuBuffers,
    device: &Device,
    queue: &wgpu::Queue,
) -> [DirectionPages; NUM_DIRECTIONS] {
    std::array::from_fn(|dir| {
        let dir_faces = &dir_faces_array[dir];
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
            let (slab_idx, page_idx) = PageAllocator::allocate(gpu, device);

            let slab = &gpu.slabs[slab_idx];
            let face_offset =
                page_idx as u64 * PAGE_SIZE as u64 * std::mem::size_of::<FaceData>() as u64;
            queue.write_buffer(&slab.face_buffer, face_offset, bytemuck::cast_slice(face_chunk));

            let dir_meta = PageMetadata {
                direction_and_lod: (dir as u32) | ((lod as u32) << 8),
                ..*meta_base
            };
            let meta_offset = page_idx as u64 * std::mem::size_of::<PageMetadata>() as u64;
            queue.write_buffer(&slab.metadata_buffer, meta_offset, bytemuck::bytes_of(&dir_meta));

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
    })
}

/// Deallocate all pages in a set of DirectionPages.
fn deallocate_direction_pages(directions: &[DirectionPages; NUM_DIRECTIONS], gpu: &mut GpuBuffers) {
    for dir_pages in directions {
        for page in &dir_pages.pages {
            PageAllocator::deallocate(gpu, page.slab_index as usize, page.page_index);
        }
    }
}

/// Drains ChunkUnloadQueue, deallocates GPU pages and removes shadow grid entries.
/// Runs before synchronize_gpu in the Synchronize stage.
pub fn cleanup_unloaded_chunks(
    mut unload_queue: ResMut<crate::chunk::ChunkUnloadQueue>,
    mut render_data: ResMut<ChunkRenderData>,
    mut gpu: ResMut<GpuBuffers>,
    mut shadow_grid: ResMut<shadow::grid::ShadowGrid>,
    mut bitmask_pool: ResMut<shadow::grid::BitmaskPool>,
    mut color_pool: ResMut<shadow::grid::TransparentColorPool>,
) {
    let _timer = crate::SysTimer::new(&crate::TIMING_CLEANUP_US);
    for unload in unload_queue.0.drain(..) {
        if let Some(entry) = render_data.entries.remove(&unload.entity) {
            render_data.dirty = true;
            render_data.removed_entities.push((unload.entity, unload.pos, unload.lod));
            deallocate_direction_pages(&entry.directions, &mut gpu);
            deallocate_direction_pages(&entry.transparent_directions, &mut gpu);
        }

        shadow::grid::remove_chunk_from_grid(
            &mut shadow_grid,
            &mut bitmask_pool,
            unload.pos,
            unload.lod,
        );
        shadow::grid::remove_chunk_transparent_data(
            &mut shadow_grid,
            &mut color_pool,
            unload.pos,
            unload.lod,
        );
    }
}

/// Compute backface cull state for a (rel_chunk_pos, lod_scale, dir) triple given camera world position.
/// Camera-dependent — recomputed on chunk crossings.
fn compute_backface(rel_chunk_pos: IVec3, lod_scale: i32, dir: u8, cam_world: [f64; 3]) -> bool {
    let cs_d = crate::chunk::CHUNK_SIZE as f64;
    let chunk_min_w = [
        rel_chunk_pos.x as f64 * cs_d,
        rel_chunk_pos.y as f64 * cs_d,
        rel_chunk_pos.z as f64 * cs_d,
    ];
    let w_extent = lod_scale as f64 * cs_d;
    let chunk_max_w = [
        chunk_min_w[0] + w_extent,
        chunk_min_w[1] + w_extent,
        chunk_min_w[2] + w_extent,
    ];
    match dir {
        0 => cam_world[0] <= chunk_min_w[0],
        1 => cam_world[0] >= chunk_max_w[0],
        2 => cam_world[1] <= chunk_min_w[1],
        3 => cam_world[1] >= chunk_max_w[1],
        4 => cam_world[2] <= chunk_min_w[2],
        5 => cam_world[2] >= chunk_max_w[2],
        _ => false,
    }
}

/// Build a cached draw entry for one (chunk, direction) pair.
/// Returns None only if there are no faces to draw — backface culling is stored
/// separately so the entry remains live and can be re-evaluated on camera moves.
fn build_draw_for_direction(
    entity: Entity,
    entry: &ChunkRenderEntry,
    directions: &[DirectionPages; NUM_DIRECTIONS],
    dir: usize,
    cam_world: [f64; 3],
    loaded_index: &LoadedChunkIndex,
    lod_count: usize,
    do_backface_cull: bool,
) -> Option<CachedDraw> {
    let dir_pages = &directions[dir];
    if dir_pages.total_faces == 0 {
        return None;
    }

    let lod_scale = 1i32 << entry.lod;
    let lod = (entry.lod as usize).min(lod_count - 1);
    let rel_chunk_pos = entry.chunk_pos * lod_scale;

    let neighbor_pos = entry.chunk_pos + DIR_OFFSETS[dir];
    let neighbor_covered = entry.lod > 0
        && is_fully_covered(neighbor_pos, entry.lod, loaded_index);
    let face_limit = if neighbor_covered {
        dir_pages.total_faces
    } else {
        dir_pages.standard_faces
    };
    if face_limit == 0 {
        return None;
    }

    let mut args = Vec::new();
    let mut faces_remaining = face_limit;
    let mut slab_index = 0;
    for page in &dir_pages.pages {
        if faces_remaining == 0 {
            break;
        }
        let count = page.face_count.min(faces_remaining);
        slab_index = page.slab_index as usize;
        args.push(DrawIndirectArgs {
            vertex_count: 6,
            instance_count: count,
            first_vertex: 0,
            first_instance: page.page_index * PAGE_SIZE as u32,
        });
        faces_remaining -= count;
    }

    if args.is_empty() {
        return None;
    }

    let backface_culled = if do_backface_cull {
        compute_backface(rel_chunk_pos, lod_scale, dir as u8, cam_world)
    } else {
        false
    };

    Some(CachedDraw {
        entity,
        chunk_pos: entry.chunk_pos,
        chunk_lod: entry.lod,
        dir: dir as u8,
        dead: false,
        backface_culled,
        rel_chunk_pos,
        lod_scale,
        slab_index,
        lod,
        args,
    })
}

/// Build cached draw entries for a single chunk's render data.
/// Returns empty if the chunk is fully covered by finer LOD.
/// `directions` selects which face set (opaque or transparent) to read from.
/// `do_backface_cull` should be false for transparent geometry (visible from both sides).
fn build_draws_for_entry(
    entity: Entity,
    entry: &ChunkRenderEntry,
    directions: &[DirectionPages; NUM_DIRECTIONS],
    cam_world: [f64; 3],
    loaded_index: &LoadedChunkIndex,
    lod_count: usize,
    do_backface_cull: bool,
) -> Vec<CachedDraw> {
    if entry.lod > 0 && is_fully_covered(entry.chunk_pos, entry.lod, loaded_index) {
        return Vec::new();
    }

    let mut draws = Vec::new();
    for dir in 0..NUM_DIRECTIONS {
        if let Some(draw) = build_draw_for_direction(entity, entry, directions, dir, cam_world, loaded_index, lod_count, do_backface_cull) {
            draws.push(draw);
        }
    }
    draws
}

pub fn synchronize_gpu(
    mut commands: Commands,
    query: Query<(Entity, &ChunkPos, &ChunkLod, &ChunkFaces, &TransparentChunkFaces)>,
    mut render_data: ResMut<ChunkRenderData>,
    mut loaded_index: ResMut<LoadedChunkIndex>,
    _allocator: Res<PageAllocator>,
    mut gpu: ResMut<GpuBuffers>,
    mut draw_cache: ResMut<DrawCache>,
    mut trans_draw_cache: ResMut<TransparentDrawCache>,
    device: Res<modul_core::DeviceRes>,
    queue: Res<modul_core::QueueRes>,
    lod_maps: Res<crate::chunk::LodChunkMaps>,
    debug: Res<crate::DebugMode>,
    cam_query: Query<(&crate::camera::Position, &crate::camera::Camera), With<crate::camera::MainCamera>>,
) {
    let _t_upload = std::time::Instant::now();
    let mut uploaded_entities: Vec<Entity> = Vec::new();
    for (entity, pos, lod, faces, trans_faces) in query.iter() {
        // Deallocate old pages (both opaque and transparent)
        if let Some(old) = render_data.entries.remove(&entity) {
            deallocate_direction_pages(&old.directions, &mut gpu);
            deallocate_direction_pages(&old.transparent_directions, &mut gpu);
        }

        let meta_base = PageMetadata {
            chunk_x: pos.0.x,
            chunk_y: pos.0.y,
            chunk_z: pos.0.z,
            direction_and_lod: 0,
        };

        let directions = upload_direction_faces(&faces.0, &meta_base, lod.0, &mut gpu, &device.0, &queue.0);
        let transparent_directions = upload_direction_faces(&trans_faces.0, &meta_base, lod.0, &mut gpu, &device.0, &queue.0);

        loaded_index.0.insert((pos.0, lod.0));

        render_data.entries.insert(
            entity,
            ChunkRenderEntry {
                chunk_pos: pos.0,
                lod: lod.0,
                directions,
                transparent_directions,
            },
        );
        uploaded_entities.push(entity);
        commands.entity(entity).remove::<ChunkFaces>();
        commands.entity(entity).remove::<TransparentChunkFaces>();
    }

    // Mark that chunks changed so both draw caches rebuild.
    if !uploaded_entities.is_empty() || render_data.dirty {
        render_data.dirty = false;
        draw_cache.generation += 1;
        trans_draw_cache.0.generation += 1;
    }

    crate::TIMING_SYNC_UPLOAD_US.fetch_max(_t_upload.elapsed().as_micros() as u32, std::sync::atomic::Ordering::Relaxed);
    let _t_draws = std::time::Instant::now();

    // --- Determine camera state ---
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

    let camera_chunk = crate::camera::chunk_pos_from_world(cam_world);

    let lod_count = lod_maps.maps.len();

    /// Update one draw cache (opaque or transparent). `get_directions` extracts the
    /// relevant DirectionPages from a ChunkRenderEntry.
    /// `do_backface_cull` should be false for transparent geometry.
    fn update_draw_cache(
        cache: &mut DrawCache,
        render_data: &ChunkRenderData,
        uploaded_entities: &[Entity],
        removed_entities: &[(Entity, IVec3, u8)],
        camera_chunk: IVec3,
        cam_world: [f64; 3],
        loaded_index: &LoadedChunkIndex,
        lod_count: usize,
        lod_maps: &crate::chunk::LodChunkMaps,
        get_directions: fn(&ChunkRenderEntry) -> &[DirectionPages; NUM_DIRECTIONS],
        do_backface_cull: bool,
    ) {
        let cam_changed = cache.last_camera_chunk != Some(camera_chunk);
        let gen_changed = cache.cached_generation != cache.generation;

        if !cam_changed && !gen_changed {
            return;
        }

        // Compute newly_covered (parent LODs that became fully covered) BEFORE compaction.
        let mut newly_covered: HashSet<(IVec3, u8)> = HashSet::new();
        if gen_changed {
            for &entity in uploaded_entities {
                if let Some(entry) = render_data.entries.get(&entity) {
                    let parent_lod = entry.lod + 1;
                    if (parent_lod as usize) < lod_count {
                        let parent_pos = IVec3::new(
                            entry.chunk_pos.x.div_euclid(2),
                            entry.chunk_pos.y.div_euclid(2),
                            entry.chunk_pos.z.div_euclid(2),
                        );
                        if is_fully_covered(parent_pos, parent_lod, loaded_index) {
                            newly_covered.insert((parent_pos, parent_lod));
                        }
                    }
                }
            }
        }

        // Compute newly_uncovered (parent LODs that lost coverage because a child unloaded).
        // These need to be re-added to the cache since they were previously dropped via newly_covered.
        let mut newly_uncovered: HashSet<(IVec3, u8)> = HashSet::new();
        if gen_changed {
            for &(_, pos, lod) in removed_entities {
                let parent_lod = lod + 1;
                if (parent_lod as usize) < lod_count {
                    let parent_pos = IVec3::new(
                        pos.x.div_euclid(2),
                        pos.y.div_euclid(2),
                        pos.z.div_euclid(2),
                    );
                    if !is_fully_covered(parent_pos, parent_lod, loaded_index) {
                        newly_uncovered.insert((parent_pos, parent_lod));
                    }
                }
            }
        }

        // Build dead-entity sets for O(1) lookup during compaction.
        let upload_set: HashSet<Entity> = if gen_changed {
            uploaded_entities.iter().copied().collect()
        } else {
            HashSet::new()
        };
        let removed_set: HashSet<Entity> = if gen_changed {
            removed_entities.iter().map(|&(e, _, _)| e).collect()
        } else {
            HashSet::new()
        };

        // Single O(N) pass: drop dead entries, drop newly-stale entries, update backface flags.
        // Also tracks which entities survive so we can avoid duplicates when re-adding parents.
        let mut surviving_entities: HashSet<Entity> = HashSet::new();
        cache.entries.retain_mut(|cached| {
            if cached.dead {
                return false;
            }
            if gen_changed
                && (upload_set.contains(&cached.entity) || removed_set.contains(&cached.entity))
            {
                return false;
            }
            if !newly_covered.is_empty()
                && newly_covered.contains(&(cached.chunk_pos, cached.chunk_lod))
            {
                return false;
            }
            if cam_changed && do_backface_cull {
                cached.backface_culled = compute_backface(
                    cached.rel_chunk_pos, cached.lod_scale, cached.dir, cam_world,
                );
            }
            surviving_entities.insert(cached.entity);
            true
        });

        if gen_changed {
            cache.cached_generation = cache.generation;

            // Add new entries for uploaded chunks
            for &entity in uploaded_entities {
                if let Some(entry) = render_data.entries.get(&entity) {
                    cache.entries.extend(build_draws_for_entry(
                        entity, entry, get_directions(entry), cam_world, loaded_index, lod_count, do_backface_cull,
                    ));
                }
            }

            // Newly-uncovered LOD parents: a child unloaded, parent is no longer fully covered,
            // so it needs to be drawn. Add its entries if not already in the cache.
            for &(parent_pos, parent_lod) in &newly_uncovered {
                let lod_idx = parent_lod as usize;
                if lod_idx >= lod_maps.maps.len() { continue; }
                let parent_entity = match lod_maps.maps[lod_idx].get(&parent_pos) {
                    Some(&e) => e,
                    None => continue,
                };
                if surviving_entities.contains(&parent_entity) {
                    continue;
                }
                if let Some(parent_entry) = render_data.entries.get(&parent_entity) {
                    cache.entries.extend(build_draws_for_entry(
                        parent_entity, parent_entry, get_directions(parent_entry),
                        cam_world, loaded_index, lod_count, do_backface_cull,
                    ));
                }
            }

            // Newly-covered LOD parents: their neighbors' face_limits change, so rebuild
            // those neighbor entries (the border face count depends on whether the neighbor
            // is fully covered).
            for &(covered_pos, covered_lod) in &newly_covered {
                let lod_idx = covered_lod as usize;
                if lod_idx >= lod_maps.maps.len() { continue; }
                for dir in 0..NUM_DIRECTIONS {
                    let neighbor_pos = covered_pos + DIR_OFFSETS[dir];
                    let opposite_dir = (dir ^ 1) as u8;

                    if is_fully_covered(neighbor_pos, covered_lod, loaded_index) { continue; }

                    let neighbor_entity = match lod_maps.maps[lod_idx].get(&neighbor_pos) {
                        Some(&e) => e,
                        None => continue,
                    };
                    let neighbor_entry = match render_data.entries.get(&neighbor_entity) {
                        Some(e) => e,
                        None => continue,
                    };

                    if let Some(cached) = cache.entries.iter_mut().find(|c| {
                        !c.dead && c.entity == neighbor_entity && c.dir == opposite_dir
                    }) {
                        if let Some(updated) = build_draw_for_direction(
                            neighbor_entity, neighbor_entry, get_directions(neighbor_entry),
                            opposite_dir as usize, cam_world, loaded_index, lod_count, do_backface_cull,
                        ) {
                            cached.args = updated.args;
                            cached.backface_culled = updated.backface_culled;
                        }
                    } else if let Some(new_draw) = build_draw_for_direction(
                        neighbor_entity, neighbor_entry, get_directions(neighbor_entry),
                        opposite_dir as usize, cam_world, loaded_index, lod_count, do_backface_cull,
                    ) {
                        cache.entries.push(new_draw);
                    }
                }
            }
        }

        if cam_changed {
            cache.last_camera_chunk = Some(camera_chunk);
        }
    }

    // --- Update opaque draw cache ---
    {
        let _t = std::time::Instant::now();
        update_draw_cache(
            &mut draw_cache, &render_data, &uploaded_entities, &render_data.removed_entities,
            camera_chunk, cam_world, &loaded_index, lod_count, &lod_maps,
            |e| &e.directions, true,
        );
        crate::TIMING_DRAW_CACHE_OPAQUE_US.fetch_max(_t.elapsed().as_micros() as u32, std::sync::atomic::Ordering::Relaxed);
    }

    // --- Update transparent draw cache (no backface cull — visible from both sides) ---
    {
        let _t = std::time::Instant::now();
        update_draw_cache(
            &mut trans_draw_cache.0, &render_data, &uploaded_entities, &render_data.removed_entities,
            camera_chunk, cam_world, &loaded_index, lod_count, &lod_maps,
            |e| &e.transparent_directions, false,
        );
        crate::TIMING_DRAW_CACHE_TRANS_US.fetch_max(_t.elapsed().as_micros() as u32, std::sync::atomic::Ordering::Relaxed);
    }
    render_data.removed_entities.clear();
    crate::TIMING_DRAW_CACHE_ENTRIES.fetch_max(
        (draw_cache.entries.len() + trans_draw_cache.0.entries.len()) as u32,
        std::sync::atomic::Ordering::Relaxed,
    );

    // --- Per-frame: frustum cull cached entries and write to indirect buffer ---
    /// Frustum cull a draw cache and collect indirect draw args grouped by slab+lod.
    fn frustum_cull_cache(
        cache: &DrawCache,
        frustum_planes: &[[f32; 4]; 6],
        frustum_chunk_offset: IVec3,
        lod_count: usize,
        slab_count: usize,
    ) -> (Vec<Vec<Vec<DrawIndirectArgs>>>, u32) {
        let mut args_per_slab_lod: Vec<Vec<Vec<DrawIndirectArgs>>> = (0..slab_count)
            .map(|_| (0..lod_count).map(|_| Vec::new()).collect())
            .collect();
        let mut frustum_culled = 0u32;
        let cs = crate::chunk::CHUNK_SIZE as f32;

        for cached in &cache.entries {
            if cached.dead || cached.backface_culled { continue; }
            let rel = cached.rel_chunk_pos - frustum_chunk_offset;
            let min = [rel.x as f32 * cs, rel.y as f32 * cs, rel.z as f32 * cs];
            let extent = cached.lod_scale as f32 * cs;
            let max = [min[0] + extent, min[1] + extent, min[2] + extent];
            if !crate::camera::is_aabb_in_frustum(frustum_planes, min, max) {
                frustum_culled += 1;
                continue;
            }

            let slab = cached.slab_index;
            while args_per_slab_lod.len() <= slab {
                args_per_slab_lod.push((0..lod_count).map(|_| Vec::new()).collect());
            }
            args_per_slab_lod[slab][cached.lod].extend_from_slice(&cached.args);
        }

        (args_per_slab_lod, frustum_culled)
    }

    /// Write frustum-culled args into indirect buffer, returning SlabLodDraw list and next offset.
    fn write_draws_to_indirect(
        args_per_slab_lod: &[Vec<Vec<DrawIndirectArgs>>],
        lod_count: usize,
        queue: &wgpu::Queue,
        indirect_buffer: &Buffer,
        start_offset: u64,
    ) -> (Vec<SlabLodDraw>, u64) {
        let stride = std::mem::size_of::<DrawIndirectArgs>() as u64;
        let mut draws = Vec::new();
        let mut offset = start_offset;

        for lod in 0..lod_count {
            for (slab_idx, slab_lods) in args_per_slab_lod.iter().enumerate() {
                let args = &slab_lods[lod];
                if args.is_empty() {
                    continue;
                }
                queue.write_buffer(indirect_buffer, offset, bytemuck::cast_slice(args));
                draws.push(SlabLodDraw {
                    slab_index: slab_idx,
                    offset,
                    count: args.len() as u32,
                });
                offset += args.len() as u64 * stride;
            }
        }

        (draws, offset)
    }

    {
        let lod_count = lod_maps.maps.len();
        let slab_count = gpu.slabs.len();

        // Opaque
        let _t_cull = std::time::Instant::now();
        let (opaque_args, frustum_culled) = frustum_cull_cache(
            &draw_cache, &frustum_planes, frustum_chunk_offset, lod_count, slab_count,
        );
        let (trans_args, trans_culled) = frustum_cull_cache(
            &trans_draw_cache.0, &frustum_planes, frustum_chunk_offset, lod_count, slab_count,
        );
        crate::TIMING_FRUSTUM_CULL_US.fetch_max(_t_cull.elapsed().as_micros() as u32, std::sync::atomic::Ordering::Relaxed);

        let _t_write = std::time::Instant::now();
        let (opaque_draws, next_offset) = write_draws_to_indirect(
            &opaque_args, lod_count, &queue.0, &gpu.indirect_buffer, 0,
        );
        let (trans_draws, _) = write_draws_to_indirect(
            &trans_args, lod_count, &queue.0, &gpu.indirect_buffer, next_offset,
        );
        crate::TIMING_WRITE_INDIRECT_US.fetch_max(_t_write.elapsed().as_micros() as u32, std::sync::atomic::Ordering::Relaxed);

        gpu.draws = opaque_draws;
        gpu.transparent_draws = trans_draws;
        gpu.frustum_culled = frustum_culled + trans_culled;
    }

    crate::TIMING_SYNC_DRAWS_US.fetch_max(_t_draws.elapsed().as_micros() as u32, std::sync::atomic::Ordering::Relaxed);
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

/// Draw loop for transparent geometry. Same as opaque but uses transparent_draws.
pub fn draw_transparent_geometry(pass: &mut wgpu::RenderPass, gpu: &GpuBuffers) {
    let mut current_slab = usize::MAX;
    for draw in &gpu.transparent_draws {
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

fn downsample_half(src: &modul_texture::Image) -> modul_texture::Image {
    let dst_w = (src.width / 2).max(1);
    let dst_h = (src.height / 2).max(1);
    let mut dst_data = Vec::with_capacity((dst_w * dst_h * 4) as usize);
    let stride = (src.width * 4) as usize;

    for y in 0..dst_h {
        for x in 0..dst_w {
            let sx = (x * 2) as usize;
            let sy = (y * 2) as usize;
            for c in 0..4usize {
                let i00 = sy * stride + sx * 4 + c;
                let i10 = sy * stride + (sx + 1) * 4 + c;
                let i01 = (sy + 1) * stride + sx * 4 + c;
                let i11 = (sy + 1) * stride + (sx + 1) * 4 + c;
                let avg = (src.data[i00] as u16
                    + src.data[i10] as u16
                    + src.data[i01] as u16
                    + src.data[i11] as u16
                    + 2) / 4;
                dst_data.push(avg as u8);
            }
        }
    }

    modul_texture::Image { data: dst_data, width: dst_w, height: dst_h }
}

fn generate_mip_chain(base: modul_texture::Image) -> modul_texture::MipMapImage {
    let level_count = (base.width.max(base.height).ilog2() + 1) as usize;
    let mut levels = Vec::with_capacity(level_count);
    levels.push(base);
    for _ in 1..level_count {
        let prev = levels.last().unwrap();
        if prev.width <= 1 && prev.height <= 1 {
            break;
        }
        levels.push(downsample_half(prev));
    }
    modul_texture::MipMapImage::with_images(levels)
}

fn create_texture_atlas_bind_group(device: &Device, queue: &wgpu::Queue) -> TextureAtlasBindGroup {
    let image = modul_texture::Image::load_from_path("assets/textures/my_tiles1.png")
        .expect("failed to load atlas texture");

    let mipmap = generate_mip_chain(image);
    let base = &mipmap.levels()[0];

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Atlas texture"),
        size: wgpu::Extent3d {
            width: base.width,
            height: base.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: mipmap.level_count() as u32,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    mipmap.write_to_texture(queue, wgpu::Origin3d::ZERO, &texture);

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("Atlas sampler"),
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    let layout = device.create_bind_group_layout(TextureAtlasBGLayout::LAYOUT);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Atlas BG"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });

    TextureAtlasBindGroup { bind_group }
}

/// Initializes core render resources: GPU buffers, camera bind group, voxel pipelines.
pub fn init_render(
    mut commands: Commands,
    device: Res<modul_core::DeviceRes>,
    queue: Res<modul_core::QueueRes>,
    mut shaders: ResMut<Assets<ShaderModule>>,
    mut layouts: ResMut<Assets<PipelineLayout>>,
    mut pipelines: ResMut<Assets<RenderPipelineManager>>,
) {
    let gpu_buffers = create_gpu_buffers(&device.0);
    let camera_bg = create_camera_bind_group(&device.0);
    let atlas_bg = create_texture_atlas_bind_group(&device.0, &queue.0);
    let voxel_pipeline = init_voxel_pipeline(&device.0, &mut pipelines, &mut shaders, &mut layouts);

    commands.insert_resource(gpu_buffers);
    commands.insert_resource(camera_bg);
    commands.insert_resource(atlas_bg);
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
