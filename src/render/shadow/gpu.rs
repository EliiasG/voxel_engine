use bevy_ecs::prelude::*;
use modul_render::BindGroupLayoutDef;
use modul_core::wgpu;
use modul_core::wgpu::{Buffer, BufferDescriptor, BufferUsages, Device};

use crate::chunk::ChunkBitmask;
use super::grid::{BitmaskPool, LodInfo, ShadowGrid, TransparentColorPool, TransparentColorRegion};

const INITIAL_BITMASK_CAPACITY: u32 = 8192;
const BITMASK_SLOT_SIZE: u64 = std::mem::size_of::<ChunkBitmask>() as u64; // 4104 bytes
const INITIAL_COLOR_POOL_CAPACITY: u32 = 1024;
const COLOR_REGION_SIZE: u64 = std::mem::size_of::<TransparentColorRegion>() as u64; // 512 bytes
const ABSORPTION_ENTRY_COUNT: usize = 254;
const ABSORPTION_BUFFER_SIZE: u64 = (ABSORPTION_ENTRY_COUNT * 16) as u64; // 254 * vec4<f32>

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
                // binding 3: transparent indirection storage (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZero::new(4), // at least one u32
                    },
                    count: None,
                },
                // binding 4: transparent color pool storage (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZero::new(COLOR_REGION_SIZE),
                    },
                    count: None,
                },
                // binding 5: absorption coefficients storage (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZero::new(ABSORPTION_BUFFER_SIZE),
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
    pub indirection_buffer: Buffer,
    pub color_pool_buffer: Buffer,
    pub color_pool_capacity: u32,
    pub absorption_buffer: Buffer,
    pub absorption_dirty: bool,
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

        let indirection_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Shadow indirection"),
            size: grid.indirection_data.len() as u64 * 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let color_pool_capacity = INITIAL_COLOR_POOL_CAPACITY;
        let color_pool_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Shadow color pool"),
            size: (color_pool_capacity as u64 * COLOR_REGION_SIZE).max(COLOR_REGION_SIZE),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let absorption_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Shadow absorption coeffs"),
            size: ABSORPTION_BUFFER_SIZE,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = Self::create_bind_group(
            device,
            &bind_group_layout,
            &lod_info_buffer,
            &grid_buffer,
            &bitmask_buffer,
            &indirection_buffer,
            &color_pool_buffer,
            &absorption_buffer,
        );

        Self {
            lod_info_buffer,
            grid_buffer,
            bitmask_buffer,
            bitmask_capacity,
            indirection_buffer,
            color_pool_buffer,
            color_pool_capacity,
            absorption_buffer,
            absorption_dirty: true,
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
        indirection: &Buffer,
        color_pool: &Buffer,
        absorption: &Buffer,
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
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: indirection.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: color_pool.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: absorption.as_entire_binding(),
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
        self.rebind(device);
    }

    fn grow_color_pool_buffer(&mut self, device: &Device, needed: u32) {
        let mut new_capacity = self.color_pool_capacity;
        while new_capacity < needed {
            new_capacity *= 2;
        }
        self.color_pool_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Shadow color pool"),
            size: new_capacity as u64 * COLOR_REGION_SIZE,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.color_pool_capacity = new_capacity;
        self.rebind(device);
    }

    fn rebind(&mut self, device: &Device) {
        self.bind_group = Self::create_bind_group(
            device,
            &self.bind_group_layout,
            &self.lod_info_buffer,
            &self.grid_buffer,
            &self.bitmask_buffer,
            &self.indirection_buffer,
            &self.color_pool_buffer,
            &self.absorption_buffer,
        );
    }
}

/// Compute absorption coefficients from block tint colors.
fn compute_absorption_coefficients() -> Vec<[f32; 4]> {
    (0..ABSORPTION_ENTRY_COUNT)
        .map(|i| {
            let tint = crate::chunk::transparent_tint(i as u8);
            [
                -(tint[0].ln()),
                -(tint[1].ln()),
                -(tint[2].ln()),
                0.0,
            ]
        })
        .collect()
}

/// Uploads shadow grid and bitmask data to GPU. Runs in Synchronize.
pub fn synchronize_shadow_buffers(
    mut shadow_gpu: ResMut<ShadowGpuBuffers>,
    mut grid: ResMut<ShadowGrid>,
    mut pool: ResMut<BitmaskPool>,
    mut color_pool: ResMut<TransparentColorPool>,
    ctx: Res<modul_core::RenderContext>,
) {
    let _timer = crate::SysTimer::new(&crate::TIMING_SHADOW_SYNC_US);
    // Check if bitmask buffer needs to grow
    let slots_needed = pool.slots.len() as u32;
    if slots_needed > shadow_gpu.bitmask_capacity {
        shadow_gpu.grow_bitmask_buffer(&ctx.device, slots_needed);
        // Re-upload all existing slots
        for (i, slot) in pool.slots.iter().enumerate() {
            let offset = i as u64 * BITMASK_SLOT_SIZE;
            ctx.queue
                .write_buffer(&shadow_gpu.bitmask_buffer, offset, bytemuck::bytes_of(slot));
        }
        pool.dirty_slots.clear();
    }

    // Upload dirty bitmask slots
    for &slot_idx in &pool.dirty_slots {
        let offset = slot_idx as u64 * BITMASK_SLOT_SIZE;
        let slot = &pool.slots[slot_idx as usize];
        ctx.queue
            .write_buffer(&shadow_gpu.bitmask_buffer, offset, bytemuck::bytes_of(slot));
    }
    pool.dirty_slots.clear();

    // Check if color pool buffer needs to grow
    let color_slots_needed = color_pool.slots.len() as u32;
    if color_slots_needed > shadow_gpu.color_pool_capacity {
        shadow_gpu.grow_color_pool_buffer(&ctx.device, color_slots_needed);
        // Re-upload all existing color pool slots
        for (i, slot) in color_pool.slots.iter().enumerate() {
            let offset = i as u64 * COLOR_REGION_SIZE;
            ctx.queue
                .write_buffer(&shadow_gpu.color_pool_buffer, offset, bytemuck::bytes_of(slot));
        }
        color_pool.dirty_slots.clear();
    }

    // Upload dirty color pool slots
    for &slot_idx in &color_pool.dirty_slots {
        let offset = slot_idx as u64 * COLOR_REGION_SIZE;
        let slot = &color_pool.slots[slot_idx as usize];
        ctx.queue
            .write_buffer(&shadow_gpu.color_pool_buffer, offset, bytemuck::bytes_of(slot));
    }
    color_pool.dirty_slots.clear();

    // Upload absorption coefficients once
    if shadow_gpu.absorption_dirty {
        let coeffs = compute_absorption_coefficients();
        ctx.queue.write_buffer(
            &shadow_gpu.absorption_buffer,
            0,
            bytemuck::cast_slice(&coeffs),
        );
        shadow_gpu.absorption_dirty = false;
    }

    // Upload lod infos if origins changed
    if grid.lod_infos_dirty {
        ctx.queue.write_buffer(
            &shadow_gpu.lod_info_buffer,
            0,
            bytemuck::cast_slice(&grid.lod_infos),
        );
        grid.lod_infos_dirty = false;
    }

    // Upload grid data if dirty (full upload — only ~157KB, not worth per-cell)
    if grid.grid_dirty {
        ctx.queue.write_buffer(
            &shadow_gpu.grid_buffer,
            0,
            bytemuck::cast_slice(&grid.grid_data),
        );
        grid.grid_dirty = false;
    }

    // Upload only dirty indirection cells (each cell = 64 u32s = 256 bytes)
    if !grid.indirection_dirty_cells.is_empty() {
        grid.indirection_dirty_cells.sort_unstable();
        grid.indirection_dirty_cells.dedup();
        for &cell_idx in &grid.indirection_dirty_cells {
            let byte_offset = cell_idx as u64 * 64 * 4;
            let start = cell_idx * 64;
            let end = start + 64;
            ctx.queue.write_buffer(
                &shadow_gpu.indirection_buffer,
                byte_offset,
                bytemuck::cast_slice(&grid.indirection_data[start..end]),
            );
        }
        grid.indirection_dirty_cells.clear();
    }
}
