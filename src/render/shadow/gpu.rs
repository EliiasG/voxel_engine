use bevy_ecs::prelude::*;
use modul_render::BindGroupLayoutDef;
use wgpu::{Buffer, BufferDescriptor, BufferUsages, Device};

use super::bitmask::ChunkBitmask;
use super::grid::{BitmaskPool, LodInfo, ShadowGrid};

const INITIAL_BITMASK_CAPACITY: u32 = 8192;
const BITMASK_SLOT_SIZE: u64 = std::mem::size_of::<ChunkBitmask>() as u64; // 4104 bytes

pub struct ShadowAccelBGLayout;

impl BindGroupLayoutDef for ShadowAccelBGLayout {
    const LAYOUT: &'static wgpu::BindGroupLayoutDescriptor<'static> =
        &wgpu::BindGroupLayoutDescriptor {
            label: Some("Shadow Accel BG Layout"),
            entries: &[
                // binding 0: LodInfo uniform array
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZero::new(
                            std::mem::size_of::<LodInfo>() as u64 * 6,
                        ),
                    },
                    count: None,
                },
                // binding 1: grid storage (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZero::new(4), // at least one u32
                    },
                    count: None,
                },
                // binding 2: bitmask storage (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZero::new(BITMASK_SLOT_SIZE),
                    },
                    count: None,
                },
            ],
        };

    const LIBRARY: &'static str = "";
}

#[derive(Resource)]
pub struct ShadowGpuBuffers {
    pub lod_info_buffer: Buffer,
    pub grid_buffer: Buffer,
    pub bitmask_buffer: Buffer,
    pub bitmask_capacity: u32,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
}

impl ShadowGpuBuffers {
    pub fn new(device: &Device, grid: &ShadowGrid) -> Self {
        let bind_group_layout = device.create_bind_group_layout(ShadowAccelBGLayout::LAYOUT);

        let lod_info_size = std::mem::size_of::<LodInfo>() as u64 * grid.lod_count as u64;
        let lod_info_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Shadow LOD info"),
            size: lod_info_size,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let grid_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Shadow grid"),
            size: grid.grid_data.len() as u64 * 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bitmask_capacity = INITIAL_BITMASK_CAPACITY;
        let bitmask_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Shadow bitmask pool"),
            size: bitmask_capacity as u64 * BITMASK_SLOT_SIZE,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = Self::create_bind_group(
            device,
            &bind_group_layout,
            &lod_info_buffer,
            &grid_buffer,
            &bitmask_buffer,
        );

        Self {
            lod_info_buffer,
            grid_buffer,
            bitmask_buffer,
            bitmask_capacity,
            bind_group_layout,
            bind_group,
        }
    }

    fn create_bind_group(
        device: &Device,
        layout: &wgpu::BindGroupLayout,
        lod_info: &Buffer,
        grid: &Buffer,
        bitmask: &Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shadow Accel BG"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: lod_info.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bitmask.as_entire_binding(),
                },
            ],
        })
    }

    fn grow_bitmask_buffer(&mut self, device: &Device, needed: u32) {
        let mut new_capacity = self.bitmask_capacity;
        while new_capacity < needed {
            new_capacity *= 2;
        }
        self.bitmask_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Shadow bitmask pool"),
            size: new_capacity as u64 * BITMASK_SLOT_SIZE,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.bitmask_capacity = new_capacity;
        self.bind_group = Self::create_bind_group(
            device,
            &self.bind_group_layout,
            &self.lod_info_buffer,
            &self.grid_buffer,
            &self.bitmask_buffer,
        );
    }
}

/// Uploads shadow grid and bitmask data to GPU. Runs in Synchronize.
pub fn synchronize_shadow_buffers(
    mut shadow_gpu: ResMut<ShadowGpuBuffers>,
    mut grid: ResMut<ShadowGrid>,
    mut pool: ResMut<BitmaskPool>,
    device: Res<modul_core::DeviceRes>,
    queue: Res<modul_core::QueueRes>,
) {
    // Check if bitmask buffer needs to grow
    let slots_needed = pool.slots.len() as u32;
    if slots_needed > shadow_gpu.bitmask_capacity {
        shadow_gpu.grow_bitmask_buffer(&device.0, slots_needed);
        // Re-upload all existing slots
        for (i, slot) in pool.slots.iter().enumerate() {
            let offset = i as u64 * BITMASK_SLOT_SIZE;
            queue
                .0
                .write_buffer(&shadow_gpu.bitmask_buffer, offset, bytemuck::bytes_of(slot));
        }
        pool.dirty_slots.clear();
    }

    // Upload dirty bitmask slots
    for &slot_idx in &pool.dirty_slots {
        let offset = slot_idx as u64 * BITMASK_SLOT_SIZE;
        let slot = &pool.slots[slot_idx as usize];
        queue
            .0
            .write_buffer(&shadow_gpu.bitmask_buffer, offset, bytemuck::bytes_of(slot));
    }
    pool.dirty_slots.clear();

    // Upload grid data if dirty
    if grid.dirty {
        queue.0.write_buffer(
            &shadow_gpu.lod_info_buffer,
            0,
            bytemuck::cast_slice(&grid.lod_infos),
        );
        queue.0.write_buffer(
            &shadow_gpu.grid_buffer,
            0,
            bytemuck::cast_slice(&grid.grid_data),
        );
        grid.dirty = false;
    }
}
