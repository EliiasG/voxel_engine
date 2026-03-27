use std::collections::HashMap;

use bevy_ecs::prelude::*;
use bytemuck::{Pod, Zeroable};
use glam::IVec3;
use modul_asset::{AssetId, AssetWorldExt, Assets};
use modul_render::{
    DirectRenderPipelineResourceProvider, GenericDepthStencilState, GenericFragmentState,
    GenericMultisampleState, GenericRenderPipelineDescriptor, GenericVertexBufferLayout,
    GenericVertexState, Operation, OperationBuilder, RenderPipelineManager, RenderTargetSource,
};
use wgpu::{
    BlendState, Buffer, BufferDescriptor, BufferUsages, ColorWrites, CommandEncoder,
    CompareFunction, DepthBiasState, Device, FrontFace, PipelineLayout, PipelineLayoutDescriptor,
    PolygonMode, PrimitiveState, PrimitiveTopology, ShaderModule, ShaderModuleDescriptor,
    ShaderSource, StencilState, VertexFormat, VertexStepMode,
};

use crate::chunk::meshing::ChunkFaces;
use crate::chunk::{ChunkLod, ChunkPos, FaceData, LoadedChunkIndex};

pub const PAGE_SIZE: usize = 512;
pub const MAX_PAGES: usize = 65536; // ~256 MB face buffer

// --- GPU Types ---

/// Per-page metadata. Integer chunk_pos + packed direction/lod.
/// Written once at upload, never needs updating.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PageMetadata {
    pub chunk_x: i32,
    pub chunk_y: i32,
    pub chunk_z: i32,
    /// Bits 0-7: direction (0-5), bits 8-15: lod level
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

// --- Resources ---

pub struct LodDrawRange {
    pub offset: u64,
    pub count: u32,
}

#[derive(Resource)]
pub struct GpuBuffers {
    pub face_buffer: Buffer,
    pub metadata_buffer: Buffer,
    pub indirect_buffer: Buffer,
    pub metadata_bind_group: wgpu::BindGroup,
    pub metadata_bind_group_layout: wgpu::BindGroupLayout,
    pub lod_draws: Vec<LodDrawRange>,
}

#[derive(Resource)]
pub struct PageAllocator {
    free_list: Vec<u32>,
}

impl PageAllocator {
    pub fn new() -> Self {
        Self {
            free_list: (0..MAX_PAGES as u32).rev().collect(),
        }
    }

    pub fn allocate(&mut self) -> Option<u32> {
        self.free_list.pop()
    }

    pub fn deallocate(&mut self, page: u32) {
        self.free_list.push(page);
    }

    pub fn used_count(&self) -> usize {
        MAX_PAGES - self.free_list.len()
    }
}

pub struct AllocatedPage {
    pub page_index: u32,
    pub face_count: u32,
}

pub struct ChunkRenderEntry {
    pub chunk_pos: IVec3,
    pub lod: u8,
    pub pages: Vec<AllocatedPage>,
}

/// Tracks which pages belong to which chunk entity.
#[derive(Resource, Default)]
pub struct ChunkRenderData {
    pub entries: HashMap<Entity, ChunkRenderEntry>,
}

#[derive(Resource)]
pub struct CameraBindGroup {
    pub buffer: Buffer,
    pub bind_group: wgpu::BindGroup,
    pub layout: wgpu::BindGroupLayout,
}

#[derive(Resource)]
pub struct VoxelPipeline {
    pub fill: AssetId<RenderPipelineManager>,
    pub wireframe: AssetId<RenderPipelineManager>,
}

#[derive(Resource)]
pub struct Wireframe(pub bool);

// --- Initialization ---

pub fn create_gpu_buffers(device: &Device) -> GpuBuffers {
    let face_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("Face buffer"),
        size: MAX_PAGES as u64 * PAGE_SIZE as u64 * std::mem::size_of::<FaceData>() as u64,
        usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let metadata_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("Metadata buffer"),
        size: MAX_PAGES as u64 * std::mem::size_of::<PageMetadata>() as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let indirect_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("Indirect buffer"),
        size: MAX_PAGES as u64 * std::mem::size_of::<DrawIndirectArgs>() as u64,
        usage: BufferUsages::INDIRECT | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let metadata_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        });

    let metadata_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Metadata BG"),
        layout: &metadata_bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: metadata_buffer.as_entire_binding(),
        }],
    });

    GpuBuffers {
        face_buffer,
        metadata_buffer,
        indirect_buffer,
        metadata_bind_group,
        metadata_bind_group_layout,
        lod_draws: Vec::new(),
    }
}

pub fn create_camera_bind_group(device: &Device) -> CameraBindGroup {
    use crate::camera::CameraUniform;

    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Camera BG Layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: std::num::NonZero::new(
                    std::mem::size_of::<CameraUniform>() as u64,
                ),
            },
            count: None,
        }],
    });

    let buffer = device.create_buffer(&BufferDescriptor {
        label: Some("Camera uniform"),
        size: std::mem::size_of::<CameraUniform>() as u64,
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

    CameraBindGroup {
        buffer,
        bind_group,
        layout,
    }
}

// --- Pipeline ---

fn create_pipeline_desc(
    shader: AssetId<ShaderModule>,
    layout: AssetId<PipelineLayout>,
    polygon_mode: PolygonMode,
    label: &str,
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
            depth_compare: CompareFunction::Less,
            stencil: StencilState::default(),
            bias: DepthBiasState::default(),
        }),
        multisample: GenericMultisampleState {
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        fragment: Some(GenericFragmentState {
            entry_point: "fs_main".into(),
            target_blend: Some(BlendState::REPLACE),
            target_color_writes: ColorWrites::ALL,
        }),
    }
}

pub fn init_pipelines(
    device: &Device,
    camera_bg: &CameraBindGroup,
    gpu_buffers: &GpuBuffers,
    shaders: &mut Assets<ShaderModule>,
    layouts: &mut Assets<PipelineLayout>,
    pipelines: &mut Assets<RenderPipelineManager>,
) -> VoxelPipeline {
    let shader = shaders.add(device.create_shader_module(ShaderModuleDescriptor {
        label: Some("Voxel shader"),
        source: ShaderSource::Wgsl(include_str!("voxel.wgsl").into()),
    }));

    let layout = layouts.add(device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("Voxel pipeline layout"),
        bind_group_layouts: &[&camera_bg.layout, &gpu_buffers.metadata_bind_group_layout],
        push_constant_ranges: &[],
    }));

    let fill = pipelines.add(RenderPipelineManager::new(create_pipeline_desc(
        shader, layout, PolygonMode::Fill, "Voxel fill pipeline",
    )));
    let wireframe = pipelines.add(RenderPipelineManager::new(create_pipeline_desc(
        shader, layout, PolygonMode::Line, "Voxel wireframe pipeline",
    )));

    VoxelPipeline { fill, wireframe }
}

// --- Synchronize System ---

/// Check if a LOD N chunk is fully covered by LOD N-1 children.
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

/// Uploads ChunkFaces to GPU pages and rebuilds the per-LOD indirect draw buffer.
pub fn synchronize_gpu(
    mut commands: Commands,
    query: Query<(Entity, &ChunkPos, &ChunkLod, &ChunkFaces)>,
    mut render_data: ResMut<ChunkRenderData>,
    mut loaded_index: ResMut<LoadedChunkIndex>,
    mut allocator: ResMut<PageAllocator>,
    mut gpu: ResMut<GpuBuffers>,
    queue: Res<modul_core::QueueRes>,
    config: Res<crate::chunk::loading::LoadConfig>,
) {
    for (entity, pos, lod, faces) in query.iter() {
        // Deallocate old pages for this chunk
        if let Some(old) = render_data.entries.remove(&entity) {
            for page in &old.pages {
                allocator.deallocate(page.page_index);
            }
        }

        let mut new_pages = Vec::new();

        for (dir, dir_faces) in faces.0.iter().enumerate() {
            if dir_faces.is_empty() {
                continue;
            }

            for face_chunk in dir_faces.chunks(PAGE_SIZE) {
                let Some(page_idx) = allocator.allocate() else {
                    eprintln!("Out of pages!");
                    break;
                };

                let face_offset =
                    page_idx as u64 * PAGE_SIZE as u64 * std::mem::size_of::<FaceData>() as u64;
                queue
                    .0
                    .write_buffer(&gpu.face_buffer, face_offset, bytemuck::cast_slice(face_chunk));

                let meta = PageMetadata {
                    chunk_x: pos.0.x,
                    chunk_y: pos.0.y,
                    chunk_z: pos.0.z,
                    direction_and_lod: (dir as u32) | ((lod.0 as u32) << 8),
                };
                let meta_offset = page_idx as u64 * std::mem::size_of::<PageMetadata>() as u64;
                queue
                    .0
                    .write_buffer(&gpu.metadata_buffer, meta_offset, bytemuck::bytes_of(&meta));

                new_pages.push(AllocatedPage {
                    page_index: page_idx,
                    face_count: face_chunk.len() as u32,
                });
            }
        }

        // Mark chunk as rendered in index (even if zero faces).
        // This drives LOD visibility: parent LOD hidden only when children are actually on GPU.
        loaded_index.0.insert((pos.0, lod.0));

        render_data.entries.insert(
            entity,
            ChunkRenderEntry {
                chunk_pos: pos.0,
                lod: lod.0,
                pages: new_pages,
            },
        );
        commands.entity(entity).remove::<ChunkFaces>();
    }

    // Rebuild indirect buffer grouped by LOD (every frame -- cheap, keeps LOD visibility correct)
    {
        let lod_count = config.lod_count as usize;
        let mut args_per_lod: Vec<Vec<DrawIndirectArgs>> =
            (0..lod_count).map(|_| Vec::new()).collect();

        for entry in render_data.entries.values() {
            // Skip LOD N pages fully covered by LOD N-1
            if entry.lod > 0
                && is_fully_covered(entry.chunk_pos, entry.lod, &loaded_index)
            {
                continue;
            }

            let lod = (entry.lod as usize).min(lod_count - 1);
            for page in &entry.pages {
                args_per_lod[lod].push(DrawIndirectArgs {
                    vertex_count: 6,
                    instance_count: page.face_count,
                    first_vertex: 0,
                    first_instance: page.page_index * PAGE_SIZE as u32,
                });
            }
        }

        // Write to indirect buffer sequentially, record per-LOD ranges
        let mut lod_draws = Vec::new();
        let mut offset = 0u64;
        let stride = std::mem::size_of::<DrawIndirectArgs>() as u64;

        for args in &args_per_lod {
            let byte_offset = offset;
            if !args.is_empty() {
                queue
                    .0
                    .write_buffer(&gpu.indirect_buffer, byte_offset, bytemuck::cast_slice(args));
            }
            lod_draws.push(LodDrawRange {
                offset: byte_offset,
                count: args.len() as u32,
            });
            offset += args.len() as u64 * stride;
        }

        gpu.lod_draws = lod_draws;
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
            pass.set_bind_group(0, camera_bg, &[]);
            pass.set_bind_group(1, &gpu.metadata_bind_group, &[]);
            pass.set_vertex_buffer(0, gpu.face_buffer.slice(..));

            // Draw LOD 0 first (writes depth), then LOD 1, 2, ... (depth-rejected if behind)
            for draw in &gpu.lod_draws {
                if draw.count > 0 {
                    pass.multi_draw_indirect(&gpu.indirect_buffer, draw.offset, draw.count);
                }
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
