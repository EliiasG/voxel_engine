use std::collections::HashMap;

use bevy_ecs::prelude::*;
use bytemuck::{Pod, Zeroable};
use glam::IVec3;

use crate::chunk::lod_chunk_pos;

use crate::chunk::{self, ChunkBitmask, ChunkBitmaskResult};

pub const GRID_EMPTY: u32 = 0xFFFFFFFF;
pub const GRID_SOLID: u32 = 0xFFFFFFFE;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LodInfo {
    pub grid_origin: [i32; 3],
    pub grid_size: u32,
    pub lod_scale: u32,
    pub _pad: [u32; 3],
}

#[derive(Resource)]
pub struct ShadowGrid {
    pub lod_infos: Vec<LodInfo>,
    pub grid_data: Vec<u32>,
    pub grid_size: u32,
    pub lod_count: u32,
    pub dirty: bool,
    /// Source of truth: (chunk_pos, lod) → grid value (slot index or sentinel).
    /// Grid data is rebuilt from this whenever origins change.
    chunk_values: HashMap<(IVec3, u8), u32>,
    /// Per-chunk transparent color indirection (4x4x4 = 64 u32 entries per chunk).
    /// Parallel to grid_data: indirection_data[grid_index * 64 + region] = color pool slot or GRID_EMPTY.
    pub indirection_data: Vec<u32>,
    pub indirection_dirty: bool,
    /// Source of truth for indirection, rebuilt on origin change.
    chunk_indirections: HashMap<(IVec3, u8), [u32; 64]>,
    /// Last camera chunk seen — drives origin rebuilds.
    last_camera_chunk: Option<IVec3>,
}

impl ShadowGrid {
    pub fn new(end_radius: u32, lod_count: u32) -> Self {
        let grid_size = end_radius * 2 + 1;
        let entries_per_lod = (grid_size * grid_size * grid_size) as usize;

        let lod_infos = (0..lod_count)
            .map(|lod| LodInfo {
                grid_origin: [0; 3],
                grid_size,
                lod_scale: 1 << lod,
                _pad: [0; 3],
            })
            .collect();

        let total_cells = entries_per_lod * lod_count as usize;
        let grid_data = vec![GRID_EMPTY; total_cells];
        let indirection_data = vec![GRID_EMPTY; total_cells * 64];

        Self {
            lod_infos,
            grid_data,
            grid_size,
            lod_count,
            dirty: true,
            chunk_values: HashMap::new(),
            indirection_data,
            indirection_dirty: true,
            chunk_indirections: HashMap::new(),
            last_camera_chunk: None,
        }
    }

    fn grid_index(&self, lod: u8, chunk_pos: IVec3) -> Option<usize> {
        let info = &self.lod_infos[lod as usize];
        let local = chunk_pos - IVec3::from(info.grid_origin);
        let s = self.grid_size as i32;
        if local.x < 0 || local.y < 0 || local.z < 0 || local.x >= s || local.y >= s || local.z >= s
        {
            return None;
        }
        let entries_per_lod = (self.grid_size * self.grid_size * self.grid_size) as usize;
        let lod_offset = lod as usize * entries_per_lod;
        let flat = local.x as usize
            + local.y as usize * self.grid_size as usize
            + local.z as usize * self.grid_size as usize * self.grid_size as usize;
        Some(lod_offset + flat)
    }

    pub fn get(&self, lod: u8, chunk_pos: IVec3) -> Option<u32> {
        self.grid_index(lod, chunk_pos).map(|i| self.grid_data[i])
    }

    pub fn set(&mut self, lod: u8, chunk_pos: IVec3, value: u32) {
        if let Some(i) = self.grid_index(lod, chunk_pos) {
            self.grid_data[i] = value;
            self.dirty = true;
        }
    }

    /// Recompute grid origins based on camera chunk position.
    /// If any origin changes, rebuild the entire grid from the chunk_values map.
    pub fn rebuild_origins(&mut self, camera_chunk: IVec3, end_radius: u32) {
        let radius = end_radius as i32;
        let mut changed = false;
        for lod in 0..self.lod_count {
            let lod_cam = lod_chunk_pos(camera_chunk, lod);
            let info = &mut self.lod_infos[lod as usize];
            let new_origin = (lod_cam - IVec3::splat(radius)).to_array();
            if info.grid_origin != new_origin {
                info.grid_origin = new_origin;
                changed = true;
            }
        }
        if changed {
            // Clear grid and repopulate from the canonical map
            self.grid_data.fill(GRID_EMPTY);
            self.indirection_data.fill(GRID_EMPTY);
            for (&(pos, lod), &value) in &self.chunk_values {
                if let Some(i) = self.grid_index(lod, pos) {
                    self.grid_data[i] = value;
                }
            }
            for (&(pos, lod), indirection) in &self.chunk_indirections {
                if let Some(i) = self.grid_index(lod, pos) {
                    let base = i * 64;
                    self.indirection_data[base..base + 64].copy_from_slice(indirection);
                }
            }
            self.dirty = true;
            self.indirection_dirty = true;
        }
    }
}

#[derive(Resource)]
pub struct BitmaskPool {
    pub slots: Vec<ChunkBitmask>,
    pub free_list: Vec<u32>,
    pub dirty_slots: Vec<u32>,
}

impl BitmaskPool {
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            free_list: Vec::new(),
            dirty_slots: Vec::new(),
        }
    }

    pub fn allocate(&mut self, bitmask: ChunkBitmask) -> u32 {
        let slot = if let Some(slot) = self.free_list.pop() {
            self.slots[slot as usize] = bitmask;
            slot
        } else {
            let slot = self.slots.len() as u32;
            self.slots.push(bitmask);
            slot
        };
        self.dirty_slots.push(slot);
        slot
    }

    pub fn deallocate(&mut self, slot: u32) {
        self.free_list.push(slot);
    }
}

/// Per-8x8x8-region transparent color data (512 bytes = 8*8*8).
/// Byte values: 0-253 = transparent color index, 254 = air, 255 = opaque.
pub const TRANSPARENT_AIR: u8 = 254;
pub const TRANSPARENT_OPAQUE: u8 = 255;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct TransparentColorRegion {
    pub data: [u8; 512],
}

#[derive(Resource)]
pub struct TransparentColorPool {
    pub slots: Vec<TransparentColorRegion>,
    pub free_list: Vec<u32>,
    pub dirty_slots: Vec<u32>,
}

impl TransparentColorPool {
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            free_list: Vec::new(),
            dirty_slots: Vec::new(),
        }
    }

    pub fn allocate(&mut self, region: TransparentColorRegion) -> u32 {
        let slot = if let Some(slot) = self.free_list.pop() {
            self.slots[slot as usize] = region;
            slot
        } else {
            let slot = self.slots.len() as u32;
            self.slots.push(region);
            slot
        };
        self.dirty_slots.push(slot);
        slot
    }

    pub fn deallocate(&mut self, slot: u32) {
        self.free_list.push(slot);
    }
}

/// Called when a chunk finishes generation. Updates the grid and pool.
pub fn update_grid_for_chunk(
    grid: &mut ShadowGrid,
    pool: &mut BitmaskPool,
    chunk_pos: IVec3,
    lod: u8,
    result: &ChunkBitmaskResult,
) {
    // Deallocate old slot if this chunk already had one
    if let Some(&old_value) = grid.chunk_values.get(&(chunk_pos, lod)) {
        if old_value != GRID_EMPTY && old_value != GRID_SOLID {
            pool.deallocate(old_value);
        }
    }

    let value = match result {
        ChunkBitmaskResult::AllAir => GRID_EMPTY,
        ChunkBitmaskResult::AllSolid => GRID_SOLID,
        ChunkBitmaskResult::Partial(bitmask) => pool.allocate(*bitmask),
    };
    grid.chunk_values.insert((chunk_pos, lod), value);
    grid.set(lod, chunk_pos, value);
}

/// Called when a chunk is unloaded. Frees the pool slot if any.
pub fn remove_chunk_from_grid(
    grid: &mut ShadowGrid,
    pool: &mut BitmaskPool,
    chunk_pos: IVec3,
    lod: u8,
) {
    if let Some(old_value) = grid.chunk_values.remove(&(chunk_pos, lod)) {
        if old_value != GRID_EMPTY && old_value != GRID_SOLID {
            pool.deallocate(old_value);
        }
    }
    grid.set(lod, chunk_pos, GRID_EMPTY);
}

/// Called when a chunk is unloaded. Frees transparent color pool slots.
pub fn remove_chunk_transparent_data(
    grid: &mut ShadowGrid,
    color_pool: &mut TransparentColorPool,
    chunk_pos: IVec3,
    lod: u8,
) {
    if let Some(indirection) = grid.chunk_indirections.remove(&(chunk_pos, lod)) {
        for &slot in &indirection {
            if slot != GRID_EMPTY {
                color_pool.deallocate(slot);
            }
        }
        if let Some(i) = grid.grid_index(lod, chunk_pos) {
            let base = i * 64;
            for j in 0..64 {
                grid.indirection_data[base + j] = GRID_EMPTY;
            }
            grid.indirection_dirty = true;
        }
    }
}

/// Builds transparent color indirection + pool entries for a single chunk.
fn update_chunk_transparent_data(
    grid: &mut ShadowGrid,
    color_pool: &mut TransparentColorPool,
    chunk_pos: IVec3,
    lod: u8,
    storage: &chunk::ChunkStorage,
) {
    // Deallocate old transparent data
    if let Some(old_indirection) = grid.chunk_indirections.remove(&(chunk_pos, lod)) {
        for &slot in &old_indirection {
            if slot != GRID_EMPTY {
                color_pool.deallocate(slot);
            }
        }
    }

    // Quick check: does this chunk have any transparent blocks?
    let has_transparent = match storage {
        chunk::ChunkStorage::Filled(block) => chunk::is_transparent(*block),
        chunk::ChunkStorage::Paletted { palette, .. } => {
            palette.iter().any(|&b| chunk::is_transparent(b))
        }
    };

    if !has_transparent {
        // Clear indirection in grid data
        if let Some(i) = grid.grid_index(lod, chunk_pos) {
            let base = i * 64;
            for j in 0..64 {
                grid.indirection_data[base + j] = GRID_EMPTY;
            }
            grid.indirection_dirty = true;
        }
        return;
    }

    // Build indirection table + color regions
    let mut indirection = [GRID_EMPTY; 64];

    for rz in 0..4usize {
        for ry in 0..4usize {
            for rx in 0..4usize {
                let region_idx = rx + ry * 4 + rz * 16;
                let mut has_transparent_in_region = false;
                let mut region = TransparentColorRegion {
                    data: [TRANSPARENT_AIR; 512],
                };

                for dz in 0..8usize {
                    for dy in 0..8usize {
                        for dx in 0..8usize {
                            let x = rx * 8 + dx;
                            let y = ry * 8 + dy;
                            let z = rz * 8 + dz;
                            let block = storage.get(x, y, z);
                            let local_idx = dx + dy * 8 + dz * 64;

                            if block == chunk::AIR {
                                // already TRANSPARENT_AIR
                            } else if chunk::is_opaque(block) {
                                region.data[local_idx] = TRANSPARENT_OPAQUE;
                            } else {
                                region.data[local_idx] = chunk::block_props(block).color_index;
                                has_transparent_in_region = true;
                            }
                        }
                    }
                }

                if has_transparent_in_region {
                    let slot = color_pool.allocate(region);
                    indirection[region_idx] = slot;
                }
            }
        }
    }

    // Store indirection
    grid.chunk_indirections.insert((chunk_pos, lod), indirection);
    if let Some(i) = grid.grid_index(lod, chunk_pos) {
        let base = i * 64;
        grid.indirection_data[base..base + 64].copy_from_slice(&indirection);
        grid.indirection_dirty = true;
    }
}

/// Rebuilds bitmasks for changed chunks and updates the shadow grid.
/// Reacts to ChunkChangedQueue — covers both newly generated and player-modified chunks.
pub fn process_chunk_bitmasks(
    changed: Res<crate::chunk::ChunkChangedQueue>,
    mut shadow_grid: ResMut<ShadowGrid>,
    mut bitmask_pool: ResMut<BitmaskPool>,
    chunk_data_query: Query<&crate::chunk::ChunkData>,
) {
    for change in &changed.0 {
        if let Ok(data) = chunk_data_query.get(change.entity) {
            let bitmask_result = crate::chunk::build_bitmask(&data.0);
            update_grid_for_chunk(
                &mut shadow_grid,
                &mut bitmask_pool,
                change.pos,
                change.lod,
                &bitmask_result,
            );
        }
    }
}

/// Builds transparent color data for changed chunks.
pub fn process_chunk_transparent_colors(
    changed: Res<crate::chunk::ChunkChangedQueue>,
    mut shadow_grid: ResMut<ShadowGrid>,
    mut color_pool: ResMut<TransparentColorPool>,
    chunk_data_query: Query<&crate::chunk::ChunkData>,
) {
    for change in &changed.0 {
        if let Ok(data) = chunk_data_query.get(change.entity) {
            update_chunk_transparent_data(
                &mut shadow_grid,
                &mut color_pool,
                change.pos,
                change.lod,
                &data.0,
            );
        }
    }
}

/// Rebuilds shadow grid origins when the camera moves to a new chunk.
pub fn update_shadow_grid_origins(
    mut shadow_grid: ResMut<ShadowGrid>,
    debug: Res<crate::DebugMode>,
    cam_query: Query<&crate::camera::Position, With<crate::camera::MainCamera>>,
    source_query: Query<&crate::chunk::demand::ChunkSource>,
) {
    let camera_chunk = if let Some(ref frozen) = debug.frozen {
        frozen.chunk_pos
    } else if let Ok(pos) = cam_query.get_single() {
        crate::camera::chunk_pos(pos)
    } else {
        return;
    };

    if shadow_grid.last_camera_chunk != Some(camera_chunk) {
        shadow_grid.last_camera_chunk = Some(camera_chunk);
        if let Some(source) = source_query.iter().next() {
            shadow_grid.rebuild_origins(camera_chunk, source.end_radius);
        }
    }
}
