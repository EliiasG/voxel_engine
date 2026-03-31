use std::collections::HashMap;

use bevy_ecs::prelude::*;
use bytemuck::{Pod, Zeroable};
use glam::IVec3;
use modul_asset::{AssetId, AssetWorldExt, Assets};
use modul_render::{
    BindGroupLayoutProvider, DirectRenderPipelineResourceProvider, GenericDepthStencilState,
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

impl BindGroupLayoutProvider for CameraBGLayout {
    fn layout(&self) -> wgpu::BindGroupLayoutDescriptor<'_> {
        static ENTRIES: [wgpu::BindGroupLayoutEntry; 1] = [wgpu::BindGroupLayoutEntry {
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
        }];
        wgpu::BindGroupLayoutDescriptor {
            label: Some("Camera BG Layout"),
            entries: &ENTRIES,
        }
    }

    fn library(&self) -> &str {
        "\
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
"
    }
}

pub struct MetadataBGLayout;

impl BindGroupLayoutProvider for MetadataBGLayout {
    fn layout(&self) -> wgpu::BindGroupLayoutDescriptor<'_> {
        static ENTRIES: [wgpu::BindGroupLayoutEntry; 1] = [wgpu::BindGroupLayoutEntry {
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
        }];
        wgpu::BindGroupLayoutDescriptor {
            label: Some("Metadata BG Layout"),
            entries: &ENTRIES,
        }
    }

    fn library(&self) -> &str {
        "\
struct PageMetadata {
    chunk_x: i32,
    chunk_y: i32,
    chunk_z: i32,
    direction_and_lod: u32,
};

@group(#BIND_GROUP) @binding(0)
var<storage, read> metadata: array<PageMetadata>;
"
    }
}

pub struct ShadowMaskBGLayout;

impl BindGroupLayoutProvider for ShadowMaskBGLayout {
    fn layout(&self) -> wgpu::BindGroupLayoutDescriptor<'_> {
        static ENTRIES: [wgpu::BindGroupLayoutEntry; 3] = [
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
        ];
        wgpu::BindGroupLayoutDescriptor {
            label: Some("Shadow Mask BG Layout"),
            entries: &ENTRIES,
        }
    }

    fn library(&self) -> &str {
        "\
@group(#BIND_GROUP) @binding(0)
var shadow_mask: texture_2d<f32>;
@group(#BIND_GROUP) @binding(1)
var shadow_sampler: sampler;
@group(#BIND_GROUP) @binding(2)
var shadow_normal: texture_2d<f32>;
"
    }
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
pub struct VoxelPipeline {
    pub fill: AssetId<RenderPipelineManager>,
    pub wireframe: AssetId<RenderPipelineManager>,
    pub normal_fill: AssetId<RenderPipelineManager>,
}

#[derive(Resource)]
pub struct Wireframe(pub bool);

// --- Initialization ---

pub fn create_gpu_buffers(device: &Device) -> GpuBuffers {
    let metadata_bind_group_layout =
        device.create_bind_group_layout(&MetadataBGLayout.layout());

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
    let layout = device.create_bind_group_layout(&CameraBGLayout.layout());

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

// --- Pipeline ---

fn create_pipeline_desc(
    shader: AssetId<ShaderModule>,
    layout: AssetId<PipelineLayout>,
    polygon_mode: PolygonMode,
    label: &str,
    frag_entry: &str,
) -> GenericRenderPipelineDescriptor {
    GenericRenderPipelineDescriptor {
        resource_provider: Box::new(DirectRenderPipelineResourceProvider {
            layout,
            vertex_shader_module: shader,
            fragment_shader_module: shader,
        }),
        label: Some(label.into()),
        vertex_state: GenericVertexState {
            entry_point: "vs_main".into(),
            buffers: vec![GenericVertexBufferLayout {
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
            }],
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
}

pub fn init_pipelines(
    device: &Device,
    pipelines: &mut Assets<RenderPipelineManager>,
    shaders: &mut Assets<ShaderModule>,
    layouts: &mut Assets<PipelineLayout>,
) -> VoxelPipeline {
    // Compose shader from bind group libraries + main shader
    let camera_wgsl = CameraBGLayout.library().replace("#BIND_GROUP", "0");
    let metadata_wgsl = MetadataBGLayout.library().replace("#BIND_GROUP", "1");
    let shadow_mask_wgsl = ShadowMaskBGLayout.library().replace("#BIND_GROUP", "2");
    let atmosphere_wgsl = crate::atmosphere::AtmosphereBGLayout.library().replace("#BIND_GROUP", "3");
    let full_source = format!(
        "{camera_wgsl}\n{metadata_wgsl}\n{shadow_mask_wgsl}\n{atmosphere_wgsl}\n{}",
        include_str!("voxel.wgsl")
    );

    let shader = shaders.add(device.create_shader_module(ShaderModuleDescriptor {
        label: Some("Voxel shader"),
        source: ShaderSource::Wgsl(full_source.into()),
    }));

    // Create pipeline layout from bind group layout providers
    let camera_layout = device.create_bind_group_layout(&CameraBGLayout.layout());
    let metadata_layout = device.create_bind_group_layout(&MetadataBGLayout.layout());
    let shadow_mask_layout = device.create_bind_group_layout(&ShadowMaskBGLayout.layout());
    let atmosphere_layout = device.create_bind_group_layout(&crate::atmosphere::AtmosphereBGLayout.layout());
    let layout = layouts.add(device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("Voxel pipeline layout"),
        bind_group_layouts: &[&camera_layout, &metadata_layout, &shadow_mask_layout, &atmosphere_layout],
        push_constant_ranges: &[],
    }));

    let fill = pipelines.add(RenderPipelineManager::new(create_pipeline_desc(
        shader, layout, PolygonMode::Fill, "Voxel fill pipeline", "fs_main",
    )));
    let wireframe = pipelines.add(RenderPipelineManager::new(create_pipeline_desc(
        shader, layout, PolygonMode::Line, "Voxel wireframe pipeline", "fs_main",
    )));
    let normal_fill = pipelines.add(RenderPipelineManager::new(create_pipeline_desc(
        shader, layout, PolygonMode::Fill, "Voxel normal pipeline", "fs_normal",
    )));

    VoxelPipeline { fill, wireframe, normal_fill }
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
    camera: Res<crate::Camera>,
    debug: Res<crate::DebugMode>,
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
        let (frustum_planes, frustum_chunk_offset) = match debug.frozen {
            Some(ref f) => (f.planes, f.chunk_pos),
            None => {
                let u = camera.0.uniform();
                (
                    crate::camera::extract_frustum_planes(&u.view_proj),
                    IVec3::from_array(u.chunk_offset),
                )
            }
        };

        // args_per_slab_lod[slab][lod] = Vec<DrawIndirectArgs>
        let mut args_per_slab_lod: Vec<Vec<Vec<DrawIndirectArgs>>> = (0..slab_count)
            .map(|_| (0..lod_count).map(|_| Vec::new()).collect())
            .collect();

        let mut frustum_culled = 0u32;
        let cam_world = match debug.frozen {
            Some(ref f) => f.camera_world,
            None => camera.0.position,
        };

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

pub struct VoxelDrawOperation {
    pub target: RenderTargetSource,
}

impl Operation for VoxelDrawOperation {
    fn run(&mut self, world: &mut World, command_encoder: &mut CommandEncoder) {
        let voxel_pipeline = world.resource::<VoxelPipeline>();
        let pipeline_id = if world.get_resource::<Wireframe>().is_some_and(|w| w.0) {
            voxel_pipeline.wireframe
        } else {
            voxel_pipeline.fill
        };

        world.asset_scope(pipeline_id, |world, pipeline_man: &mut RenderPipelineManager| {
            let Some(pipeline) = pipeline_man.get_compatible(self.target, world) else {
                return;
            };
            let Some(mut rt) = self.target.get_mut(world) else {
                return;
            };
            let Some(mut pass) = rt.begin_ending_pass(command_encoder) else {
                return;
            };

            pass.set_pipeline(pipeline);

            let camera_bg = &world.resource::<CameraBindGroup>().bind_group;
            let gpu = world.resource::<GpuBuffers>();
            let shadow_res = world.resource::<crate::shadow::pass::ShadowPassResources>();

            // Create shadow mask bind group from current accumulated texture
            let shadow_mask_layout = world.resource::<modul_core::DeviceRes>().0
                .create_bind_group_layout(&ShadowMaskBGLayout.layout());
            let shadow_mask_bg = world.resource::<modul_core::DeviceRes>().0
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Shadow mask BG"),
                    layout: &shadow_mask_layout,
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
                });

            pass.set_bind_group(0, camera_bg, &[]);
            pass.set_bind_group(2, &shadow_mask_bg, &[]);

            // Draw: iterate slab+LOD groups (already ordered LOD 0 first)
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
        });
    }
}

pub struct VoxelDrawOperationBuilder {
    pub target: RenderTargetSource,
}

impl OperationBuilder for VoxelDrawOperationBuilder {
    fn reading(&self) -> Vec<RenderTargetSource> { Vec::new() }
    fn writing(&self) -> Vec<RenderTargetSource> { vec![self.target] }
    fn finish(self, _world: &World, _device: &Device) -> impl Operation + 'static {
        VoxelDrawOperation { target: self.target }
    }
}
