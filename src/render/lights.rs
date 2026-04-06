//! Light buffers, bind group, and per-frame upload for slice 1 of the
//! clustered lighting system.
//!
//! Slice 1 keeps everything as flat arrays without clustering: chunk-owned
//! lights are rebuilt into one global buffer whenever the set of loaded
//! chunks changes, and entity-attached dynamic lights are uploaded fresh
//! every frame. The fragment shader walks both buffers linearly. This is
//! intentionally slow but visually verifiable; slice 2 will introduce
//! frustum culling + the cluster build pass.

use std::collections::HashMap;

use bevy_ecs::prelude::*;
use bytemuck::{Pod, Zeroable};
use modul_render::{BindGroupLayoutDef, Operation, OperationBuilder, RenderTargetSource};
use modul_core::wgpu;
use modul_core::wgpu::{
    Buffer, BufferDescriptor, BufferUsages, CommandEncoder, Device,
};

use crate::camera::{MainCamera, Position};
use crate::chunk::meshing::{ChunkLocalLight, ChunkSimpleLights};
use crate::chunk::{ChunkPos, CHUNK_SIZE, MAX_SIMPLE_LIGHT_RANGE};

/// Maximum chunk-owned simple lights across all loaded chunks.
pub const MAX_CHUNK_LIGHTS: usize = 65_536;

/// Maximum dynamic (ECS-owned) simple lights uploaded per frame.
pub const MAX_DYNAMIC_LIGHTS: usize = 1024;

// --- Cluster grid ---
//
// Standard Doom-2016 layout: 16x9 screen tiles × 24 log-Z depth slices.
// Per-cluster atomic counters in `cluster_index_counts`, with each
// cluster owning a fixed slice of `cluster_index_list` of size
// `MAX_LIGHTS_PER_CLUSTER`. Indices are encoded as
// `(buffer_id << 28) | local_index` so a single list serves both
// chunk-owned and dynamic light buffers.

pub const NUM_CLUSTERS_X: u32 = 16;
pub const NUM_CLUSTERS_Y: u32 = 9;
pub const NUM_CLUSTERS_Z: u32 = 24;
pub const NUM_CLUSTERS: u32 = NUM_CLUSTERS_X * NUM_CLUSTERS_Y * NUM_CLUSTERS_Z;
pub const MAX_LIGHTS_PER_CLUSTER: u32 = 256;

/// Log-Z near plane for clustering. Should match the camera near plane.
pub const CLUSTER_NEAR: f32 = 0.1;
/// Log-Z far plane for clustering. Beyond this, the last depth slice
/// extends to infinity. 1024 voxels is far enough for any v1 lights
/// (max range = 32) and concentrates resolution near the camera.
pub const CLUSTER_FAR: f32 = 1024.0;

/// Buffer-id encoding bits in the cluster index list.
pub const CLUSTER_INDEX_BITS: u32 = 28;
pub const CLUSTER_INDEX_MASK: u32 = (1 << CLUSTER_INDEX_BITS) - 1;
pub const BUFFER_ID_CHUNK: u32 = 0;
pub const BUFFER_ID_DYNAMIC: u32 = 1;

// --- GPU types ---

/// Layout matches WGSL `SimpleLight` (see `lights_common.wgsl`). Total 64
/// bytes; offsets are computed against std430 vec3 alignment rules so the
/// Rust and WGSL views are bitwise compatible.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Default, Debug)]
pub struct GpuSimpleLight {
    pub chunk_pos: [i32; 3],   // 0..12
    pub range: f32,            // 12..16
    pub local_pos: [f32; 3],   // 16..28
    pub intensity: f32,        // 28..32
    pub color: [f32; 3],       // 32..44
    pub inner_cos: f32,        // 44..48
    pub direction: [f32; 3],   // 48..60
    pub outer_cos: f32,        // 60..64
}

const _: () = assert!(std::mem::size_of::<GpuSimpleLight>() == 64);

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Default)]
pub struct LightCounts {
    pub chunk_light_count: u32,
    pub dynamic_light_count: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

// --- Light component ---

/// Marker + parameters for an entity-attached dynamic light. Uploaded
/// fresh every frame from a single ECS query.
#[derive(Component, Clone, Copy)]
pub struct Light {
    pub color: glam::Vec3,
    pub intensity: f32,
    pub range: f32,
    pub kind: LightKind,
}

#[derive(Clone, Copy)]
pub enum LightKind {
    Point,
    /// Spot light with cone smoothing. `direction` is in world space.
    Spot {
        direction: glam::Vec3,
        inner_cos: f32,
        outer_cos: f32,
    },
}

// --- Bind group layout ---

pub struct LightsBGLayout;

impl BindGroupLayoutDef for LightsBGLayout {
    const LAYOUT: &'static wgpu::BindGroupLayoutDescriptor<'static> =
        &wgpu::BindGroupLayoutDescriptor {
            label: Some("Lights BG Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZero::new(
                            std::mem::size_of::<GpuSimpleLight>() as u64,
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZero::new(
                            std::mem::size_of::<GpuSimpleLight>() as u64,
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZero::new(
                            std::mem::size_of::<LightCounts>() as u64,
                        ),
                    },
                    count: None,
                },
                // Cluster index counts (one u32 per cluster). Read in the
                // fragment shader, atomic read/write in the compute pass.
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZero::new(4),
                    },
                    count: None,
                },
                // Cluster index list (flat array of encoded indices).
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZero::new(4),
                    },
                    count: None,
                },
            ],
        };

    const LIBRARY: &'static str = "\
struct SimpleLight {
    chunk_pos: vec3<i32>,
    range: f32,
    local_pos: vec3<f32>,
    intensity: f32,
    color: vec3<f32>,
    inner_cos: f32,
    direction: vec3<f32>,
    outer_cos: f32,
};

struct LightCounts {
    chunk_light_count: u32,
    dynamic_light_count: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(#BIND_GROUP) @binding(0)
var<storage, read> chunk_lights: array<SimpleLight>;
@group(#BIND_GROUP) @binding(1)
var<storage, read> dynamic_lights: array<SimpleLight>;
@group(#BIND_GROUP) @binding(2)
var<uniform> light_counts: LightCounts;
@group(#BIND_GROUP) @binding(3)
var<storage, read> cluster_index_counts: array<u32>;
@group(#BIND_GROUP) @binding(4)
var<storage, read> cluster_index_list: array<u32>;
";
}

/// Compute-side bind group layout for the cluster build pass. Same
/// buffers as `LightsBGLayout`, but cluster buffers are read_write so
/// the shader can do atomic appends. Lights are read-only.
pub struct ClusterBuildBGLayout;

impl BindGroupLayoutDef for ClusterBuildBGLayout {
    const LAYOUT: &'static wgpu::BindGroupLayoutDescriptor<'static> =
        &wgpu::BindGroupLayoutDescriptor {
            label: Some("Cluster Build BG Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZero::new(
                            std::mem::size_of::<GpuSimpleLight>() as u64,
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZero::new(
                            std::mem::size_of::<GpuSimpleLight>() as u64,
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZero::new(
                            std::mem::size_of::<LightCounts>() as u64,
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZero::new(4),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZero::new(4),
                    },
                    count: None,
                },
            ],
        };

    const LIBRARY: &'static str = "";
}

// --- Resources ---

#[derive(Resource)]
pub struct LightBuffers {
    pub chunk_lights_buffer: Buffer,
    pub dynamic_lights_buffer: Buffer,
    pub counts_buffer: Buffer,
    pub cluster_index_counts_buffer: Buffer,
    pub cluster_index_list_buffer: Buffer,
    /// Fragment-side bind group: lights + clusters as read-only.
    pub bind_group: wgpu::BindGroup,
    /// Compute-side bind group: same buffers but clusters are read_write
    /// so the cluster build pass can do atomic appends.
    pub compute_bind_group: wgpu::BindGroup,
}

#[derive(Resource)]
pub struct ClusterBuildPipeline {
    pub pipeline: wgpu::ComputePipeline,
}

/// CPU-side store of chunk-owned lights, keyed by chunk entity. Lights
/// are stored in their final GPU layout (with `chunk_pos` baked in) so a
/// dirty rebuild just concatenates entries into the GPU buffer.
#[derive(Resource, Default)]
pub struct ChunkLightStore {
    pub entries: HashMap<Entity, Vec<GpuSimpleLight>>,
    pub dirty: bool,
}

impl ChunkLightStore {
    pub fn remove(&mut self, entity: Entity) {
        if self.entries.remove(&entity).is_some() {
            self.dirty = true;
        }
    }
}

// --- Init ---

pub fn create_light_buffers(device: &Device) -> LightBuffers {
    let chunk_lights_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("Chunk simple lights"),
        size: (MAX_CHUNK_LIGHTS * std::mem::size_of::<GpuSimpleLight>()) as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let dynamic_lights_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("Dynamic simple lights"),
        size: (MAX_DYNAMIC_LIGHTS * std::mem::size_of::<GpuSimpleLight>()) as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let counts_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("Light counts"),
        size: std::mem::size_of::<LightCounts>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let cluster_index_counts_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("Cluster index counts"),
        size: (NUM_CLUSTERS as usize * std::mem::size_of::<u32>()) as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let cluster_index_list_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("Cluster index list"),
        size: (NUM_CLUSTERS as usize
            * MAX_LIGHTS_PER_CLUSTER as usize
            * std::mem::size_of::<u32>()) as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let layout = device.create_bind_group_layout(LightsBGLayout::LAYOUT);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Lights BG"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: chunk_lights_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: dynamic_lights_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: counts_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: cluster_index_counts_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: cluster_index_list_buffer.as_entire_binding(),
            },
        ],
    });

    let compute_layout = device.create_bind_group_layout(ClusterBuildBGLayout::LAYOUT);
    let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Cluster Build BG"),
        layout: &compute_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: chunk_lights_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: dynamic_lights_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: counts_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: cluster_index_counts_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: cluster_index_list_buffer.as_entire_binding(),
            },
        ],
    });

    LightBuffers {
        chunk_lights_buffer,
        dynamic_lights_buffer,
        counts_buffer,
        cluster_index_counts_buffer,
        cluster_index_list_buffer,
        bind_group,
        compute_bind_group,
    }
}

pub fn create_cluster_build_pipeline(
    device: &Device,
    camera_bg_layout: &wgpu::BindGroupLayout,
) -> ClusterBuildPipeline {
    let compute_bg_layout = device.create_bind_group_layout(ClusterBuildBGLayout::LAYOUT);

    let camera_wgsl = crate::render::CameraBGLayout::LIBRARY.replace("#BIND_GROUP", "0");
    let cluster_wgsl = include_str!("shaders/cluster_build.wgsl");
    let source = format!("{camera_wgsl}\n{cluster_wgsl}");

    // SAFETY: cluster build's loops are bounded by `light_counts.*` (CPU-written
    // to actual buffer content) and `cluster_id < NUM_CLUSTERS` is enforced by
    // the early-out at the top of the entry point.
    let module = unsafe { device.create_shader_module_trusted(
        wgpu::ShaderModuleDescriptor {
            label: Some("Cluster build shader"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        },
        wgpu::ShaderRuntimeChecks::unchecked(),
    ) };

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Cluster build pipeline layout"),
        bind_group_layouts: &[Some(camera_bg_layout), Some(&compute_bg_layout)],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Cluster build pipeline"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("cs_cluster_build"),
        compilation_options: Default::default(),
        cache: None,
    });

    ClusterBuildPipeline { pipeline }
}

pub fn init_lights(mut commands: Commands, ctx: Res<modul_core::RenderContext>) {
    let buffers = create_light_buffers(&ctx.device);
    let camera_bg_layout = ctx
        .device
        .create_bind_group_layout(crate::render::CameraBGLayout::LAYOUT);
    let cluster_pipeline = create_cluster_build_pipeline(&ctx.device, &camera_bg_layout);
    commands.insert_resource(buffers);
    commands.insert_resource(cluster_pipeline);
    commands.insert_resource(ChunkLightStore::default());
}

// --- Conversion helpers ---

fn make_chunk_light(local: &ChunkLocalLight, chunk_pos: glam::IVec3) -> GpuSimpleLight {
    GpuSimpleLight {
        chunk_pos: chunk_pos.to_array(),
        range: local.range.min(MAX_SIMPLE_LIGHT_RANGE),
        local_pos: local.local_pos.to_array(),
        intensity: local.intensity,
        color: local.color.to_array(),
        inner_cos: local.inner_cos,
        direction: local.direction.to_array(),
        outer_cos: local.outer_cos,
    }
}

// --- Synchronize ---

/// Top-level light synchronizer. Runs in the Synchronize stage and:
///   1. Inspects the unload queue (without draining) and drops chunk
///      lights for entities about to be cleaned up.
///   2. Ingests new `ChunkSimpleLights` components from the mesher,
///      bakes `chunk_pos` into each light, and stores per-entity.
///   3. Rebuilds the GPU chunk lights buffer when the store is dirty.
///   4. Uploads dynamic ECS-owned lights every frame, camera-relative.
///   5. Writes the counts uniform.
///
/// Must run *before* `cleanup_unloaded_chunks` (which drains the queue).
pub fn synchronize_lights(
    mut commands: Commands,
    mut store: ResMut<ChunkLightStore>,
    buffers: Res<LightBuffers>,
    ctx: Res<modul_core::RenderContext>,
    unload_queue: Res<crate::chunk::ChunkUnloadQueue>,
    chunk_lights_query: Query<(Entity, &ChunkPos, &ChunkSimpleLights)>,
    cam_query: Query<&crate::camera::Camera, With<MainCamera>>,
    light_query: Query<(&Position, &Light, Option<&crate::camera::Rotation>)>,
) {
    let _timer = crate::SysTimer::new(&crate::TIMING_SYNC_LIGHTS_US);
    // 1. Drop chunk lights for entities being unloaded this frame.
    for unload in &unload_queue.0 {
        store.remove(unload.entity);
    }

    // 2. Ingest new chunk lights from the mesher.
    for (entity, pos, lights) in chunk_lights_query.iter() {
        if lights.0.is_empty() {
            if store.entries.remove(&entity).is_some() {
                store.dirty = true;
            }
        } else {
            let baked: Vec<GpuSimpleLight> = lights
                .0
                .iter()
                .map(|l| make_chunk_light(l, pos.0))
                .collect();
            store.entries.insert(entity, baked);
            store.dirty = true;
        }
        commands.entity(entity).remove::<ChunkSimpleLights>();
    }

    // 3. Rebuild chunk lights buffer if dirty.
    let chunk_count = if store.dirty {
        let mut flat: Vec<GpuSimpleLight> = Vec::new();
        for chunk in store.entries.values() {
            if flat.len() >= MAX_CHUNK_LIGHTS {
                break;
            }
            let remaining = MAX_CHUNK_LIGHTS - flat.len();
            if chunk.len() > remaining {
                flat.extend_from_slice(&chunk[..remaining]);
                break;
            }
            flat.extend_from_slice(chunk);
        }
        if !flat.is_empty() {
            ctx.queue.write_buffer(
                &buffers.chunk_lights_buffer,
                0,
                bytemuck::cast_slice(&flat),
            );
        }
        store.dirty = false;
        flat.len() as u32
    } else {
        store
            .entries
            .values()
            .map(|v| v.len() as u32)
            .sum::<u32>()
            .min(MAX_CHUNK_LIGHTS as u32)
    };

    // 4. Upload dynamic lights from ECS.
    let dynamic_count = upload_dynamic_lights(&cam_query, &light_query, &buffers, &ctx);

    // 5. Write counts uniform.
    let counts = LightCounts {
        chunk_light_count: chunk_count,
        dynamic_light_count: dynamic_count,
        _pad0: 0,
        _pad1: 0,
    };
    ctx.queue
        .write_buffer(&buffers.counts_buffer, 0, bytemuck::bytes_of(&counts));
}

fn upload_dynamic_lights(
    cam_query: &Query<&crate::camera::Camera, With<MainCamera>>,
    light_query: &Query<(&Position, &Light, Option<&crate::camera::Rotation>)>,
    buffers: &LightBuffers,
    ctx: &modul_core::RenderContext,
) -> u32 {
    let Ok(cam) = cam_query.single() else { return 0 };
    let cam_chunk = glam::IVec3::from_array(cam.chunk_offset);
    let cam_chunk_origin = glam::DVec3::new(
        cam_chunk.x as f64 * CHUNK_SIZE as f64,
        cam_chunk.y as f64 * CHUNK_SIZE as f64,
        cam_chunk.z as f64 * CHUNK_SIZE as f64,
    );

    let mut buf: Vec<GpuSimpleLight> = Vec::new();
    for (pos, light, rot) in light_query.iter() {
        if buf.len() >= MAX_DYNAMIC_LIGHTS {
            break;
        }
        let local = (glam::DVec3::from_array(pos.0) - cam_chunk_origin).as_vec3();
        let (direction, inner_cos, outer_cos) = match light.kind {
            LightKind::Point => (glam::Vec3::ZERO, -1.0, -1.0),
            LightKind::Spot { direction, inner_cos, outer_cos } => {
                let world_dir = if let Some(r) = rot {
                    (r.0 * direction).normalize_or_zero()
                } else {
                    direction.normalize_or_zero()
                };
                (world_dir, inner_cos, outer_cos)
            }
        };
        // `local` is already in camera-chunk-relative voxel space, so we
        // store the camera's chunk offset as `chunk_pos`. The shader's
        // rebase (`light.chunk_pos - camera.chunk_offset`) then collapses
        // to zero and `local_pos` is used directly.
        buf.push(GpuSimpleLight {
            chunk_pos: cam_chunk.to_array(),
            range: light.range.min(MAX_SIMPLE_LIGHT_RANGE),
            local_pos: local.to_array(),
            intensity: light.intensity,
            color: light.color.to_array(),
            inner_cos,
            direction: direction.to_array(),
            outer_cos,
        });
    }

    if !buf.is_empty() {
        ctx.queue.write_buffer(
            &buffers.dynamic_lights_buffer,
            0,
            bytemuck::cast_slice(&buf),
        );
    }
    buf.len() as u32
}

// --- Cluster build operation ---

/// Compute pass that runs once per frame, before the voxel draw, to
/// rebuild the per-cluster light index lists. Clears
/// `cluster_index_counts` to zero, then dispatches the cluster build
/// shader (one thread per cluster, 64 threads per workgroup).
pub struct ClusterBuildOperation;

impl Operation for ClusterBuildOperation {
    fn run(&mut self, world: &mut World, command_encoder: &mut CommandEncoder) {
        let _timer = crate::SysTimer::new(&crate::TIMING_CLUSTER_BUILD_US);
        let buffers = world.resource::<LightBuffers>();
        let pipeline_res = world.resource::<ClusterBuildPipeline>();

        // Clear the per-cluster atomic counters. The list itself is left
        // dirty — fragment-side reads only walk up to the count.
        command_encoder.clear_buffer(
            &buffers.cluster_index_counts_buffer,
            0,
            None,
        );

        let camera_bg_ptr =
            &world.resource::<crate::render::CameraBindGroup>().bind_group as *const wgpu::BindGroup;
        let compute_bg_ptr = &buffers.compute_bind_group as *const wgpu::BindGroup;
        let pipeline_ptr = &pipeline_res.pipeline as *const wgpu::ComputePipeline;
        // SAFETY: all bind groups + the pipeline live in World resources
        // and are stable for the duration of run().
        let camera_bg = unsafe { &*camera_bg_ptr };
        let compute_bg = unsafe { &*compute_bg_ptr };
        let pipeline = unsafe { &*pipeline_ptr };

        let mut pass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Cluster build pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, camera_bg, &[]);
        pass.set_bind_group(1, compute_bg, &[]);
        // 64 threads per workgroup, one thread per cluster.
        let workgroup_count = (NUM_CLUSTERS + 63) / 64;
        pass.dispatch_workgroups(workgroup_count, 1, 1);
    }
}

pub struct ClusterBuildOperationBuilder;

impl OperationBuilder for ClusterBuildOperationBuilder {
    fn reading(&self) -> Vec<RenderTargetSource> {
        Vec::new()
    }
    fn writing(&self) -> Vec<RenderTargetSource> {
        Vec::new()
    }
    fn finish(self, _world: &World, _device: &Device) -> impl Operation + 'static {
        ClusterBuildOperation
    }
}
