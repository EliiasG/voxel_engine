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
    /// Precomputed origin.rem_euclid(grid_size) per axis — used for wrapping grid indexing.
    pub origin_wrap: [u32; 3],
}

#[derive(Resource)]
pub struct ShadowGrid {
    pub lod_infos: Vec<LodInfo>,
    pub grid_data: Vec<u32>,
    pub grid_size: u32,
    pub lod_count: u32,
    /// Grid data changed — full upload (grid_data is only ~157KB, not worth per-cell).
    pub grid_dirty: bool,
    /// LodInfos changed (origin shift) — need uniform re-upload.
    pub lod_infos_dirty: bool,
    /// Source of truth: (chunk_pos, lod) → grid value (slot index or sentinel).
    chunk_values: HashMap<(IVec3, u8), u32>,
    /// Per-chunk transparent color indirection (4x4x4 = 64 u32 entries per chunk).
    /// Parallel to grid_data: indirection_data[grid_index * 64 + region] = color pool slot or GRID_EMPTY.
    pub indirection_data: Vec<u32>,
    /// Individual grid cells whose indirection changed (index into grid_data; each covers 64 u32s).
    pub indirection_dirty_cells: Vec<usize>,
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
                origin_wrap: [0; 3],
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
            grid_dirty: true,
            lod_infos_dirty: true,
            chunk_values: HashMap::new(),
            indirection_data,
            indirection_dirty_cells: Vec::new(),
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
        let gs = self.grid_size;
        let su = gs as usize;
        let entries_per_lod = su * su * su;
        let lod_offset = lod as usize * entries_per_lod;
        let ow = info.origin_wrap;
        let wx = ((ow[0] + local.x as u32) % gs) as usize;
        let wy = ((ow[1] + local.y as u32) % gs) as usize;
        let wz = ((ow[2] + local.z as u32) % gs) as usize;
        let flat = wx + wy * su + wz * su * su;
        Some(lod_offset + flat)
    }

    /// Compute wrapped flat index without bounds checking.
    /// `local` must be in [0, grid_size) per axis.
    fn wrapped_flat(grid_size: u32, lod: u8, origin_wrap: [u32; 3], local: [u32; 3]) -> usize {
        let su = grid_size as usize;
        let entries_per_lod = su * su * su;
        let lod_offset = lod as usize * entries_per_lod;
        let wx = ((origin_wrap[0] + local[0]) % grid_size) as usize;
        let wy = ((origin_wrap[1] + local[1]) % grid_size) as usize;
        let wz = ((origin_wrap[2] + local[2]) % grid_size) as usize;
        lod_offset + wx + wy * su + wz * su * su
    }

    pub fn get(&self, lod: u8, chunk_pos: IVec3) -> Option<u32> {
        self.grid_index(lod, chunk_pos).map(|i| self.grid_data[i])
    }

    pub fn set(&mut self, lod: u8, chunk_pos: IVec3, value: u32) {
        if let Some(i) = self.grid_index(lod, chunk_pos) {
            self.grid_data[i] = value;
            self.grid_dirty = true;
        }
    }

    /// Incrementally update grid origins when the camera moves to a new chunk.
    /// Uses wrapping indexing: only clears and refills newly-exposed edge slices.
    pub fn update_origins(&mut self, camera_chunk: IVec3, end_radius: u32) {
        let radius = end_radius as i32;
        let s = self.grid_size as i32;

        for lod in 0..self.lod_count {
            let lod_cam = lod_chunk_pos(camera_chunk, lod);
            let old_origin = IVec3::from(self.lod_infos[lod as usize].grid_origin);
            let new_origin = lod_cam - IVec3::splat(radius);

            if old_origin == new_origin {
                continue;
            }

            let delta = new_origin - old_origin;

            // Update origin + wrap offset (grid_index needs these for bounds checks + wrapping)
            self.lod_infos[lod as usize].grid_origin = new_origin.to_array();
            self.lod_infos[lod as usize].origin_wrap = [
                new_origin.x.rem_euclid(s) as u32,
                new_origin.y.rem_euclid(s) as u32,
                new_origin.z.rem_euclid(s) as u32,
            ];
            self.lod_infos_dirty = true;

            // If delta too large (teleport), full rebuild for this LOD
            if delta.x.abs() >= s || delta.y.abs() >= s || delta.z.abs() >= s {
                self.full_rebuild_lod(lod as u8);
                continue;
            }

            // Incremental: clear and fill only newly exposed slices per axis
            for axis in 0..3usize {
                let d = delta[axis];
                if d == 0 { continue; }

                let (start, end) = if d > 0 {
                    (old_origin[axis] + s, old_origin[axis] + s + d)
                } else {
                    (new_origin[axis], old_origin[axis])
                };

                for coord in start..end {
                    self.clear_and_fill_slice(lod as u8, axis, coord);
                }
            }

            self.grid_dirty = true;
        }
    }

    /// Clear one 2D slice of the grid and refill from HashMaps.
    fn clear_and_fill_slice(&mut self, lod: u8, axis: usize, coord: i32) {
        let s = self.grid_size as i32;
        let gs = self.grid_size;
        let su = gs as usize;
        let entries_per_lod = su * su * su;
        let lod_offset = lod as usize * entries_per_lod;
        let info = &self.lod_infos[lod as usize];
        let origin = IVec3::from(info.grid_origin);
        let ow = info.origin_wrap;

        // Wrapped coordinate of the slice along the axis
        let local_c = (coord - origin[axis]) as u32; // in [0, s)
        let wc = ((ow[axis] + local_c) % gs) as usize;

        // Clear the entire slice at this wrapped coordinate
        for u in 0..su {
            for v in 0..su {
                let (wx, wy, wz) = match axis {
                    0 => (wc, u, v),
                    1 => (u, wc, v),
                    _ => (u, v, wc),
                };
                let flat = lod_offset + wx + wy * su + wz * su * su;
                self.grid_data[flat] = GRID_EMPTY;
                let base = flat * 64;
                self.indirection_data[base..base + 64].fill(GRID_EMPTY);
                self.indirection_dirty_cells.push(flat);
            }
        }

        // Fill from HashMaps for positions in the new range
        for u in 0..s {
            for v in 0..s {
                let (chunk_pos, local) = match axis {
                    0 => (
                        IVec3::new(coord, origin.y + u, origin.z + v),
                        [local_c, u as u32, v as u32],
                    ),
                    1 => (
                        IVec3::new(origin.x + u, coord, origin.z + v),
                        [u as u32, local_c, v as u32],
                    ),
                    _ => (
                        IVec3::new(origin.x + u, origin.y + v, coord),
                        [u as u32, v as u32, local_c],
                    ),
                };

                let flat = Self::wrapped_flat(gs, lod, ow, local);

                if let Some(&value) = self.chunk_values.get(&(chunk_pos, lod)) {
                    self.grid_data[flat] = value;
                }
                if let Some(indirection) = self.chunk_indirections.get(&(chunk_pos, lod)) {
                    let base = flat * 64;
                    self.indirection_data[base..base + 64].copy_from_slice(indirection);
                }
            }
        }
    }

    /// Full clear + repopulate for a single LOD (used on teleport / first frame).
    fn full_rebuild_lod(&mut self, lod: u8) {
        let s = self.grid_size as i32;
        let gs = self.grid_size;
        let su = gs as usize;
        let entries_per_lod = su * su * su;
        let lod_offset = lod as usize * entries_per_lod;

        self.grid_data[lod_offset..lod_offset + entries_per_lod].fill(GRID_EMPTY);
        let ind_start = lod_offset * 64;
        self.indirection_data[ind_start..ind_start + entries_per_lod * 64].fill(GRID_EMPTY);

        let info = &self.lod_infos[lod as usize];
        let origin = IVec3::from(info.grid_origin);
        let ow = info.origin_wrap;

        for (&(pos, l), &value) in &self.chunk_values {
            if l != lod { continue; }
            let local = pos - origin;
            if local.x < 0 || local.y < 0 || local.z < 0
                || local.x >= s || local.y >= s || local.z >= s
            {
                continue;
            }
            let flat = Self::wrapped_flat(gs, lod, ow, [local.x as u32, local.y as u32, local.z as u32]);
            self.grid_data[flat] = value;
        }
        for (&(pos, l), indirection) in &self.chunk_indirections {
            if l != lod { continue; }
            let local = pos - origin;
            if local.x < 0 || local.y < 0 || local.z < 0
                || local.x >= s || local.y >= s || local.z >= s
            {
                continue;
            }
            let flat = Self::wrapped_flat(gs, lod, ow, [local.x as u32, local.y as u32, local.z as u32]);
            let base = flat * 64;
            self.indirection_data[base..base + 64].copy_from_slice(indirection);
        }

        // Mark all cells in this LOD as dirty for indirection upload
        for i in lod_offset..lod_offset + entries_per_lod {
            self.indirection_dirty_cells.push(i);
        }
        self.grid_dirty = true;
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
            grid.indirection_dirty_cells.push(i);
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
            grid.indirection_dirty_cells.push(i);
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
        grid.indirection_dirty_cells.push(i);
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
    let _timer = crate::SysTimer::new(&crate::TIMING_SHADOW_ORIGINS_US);
    let camera_chunk = if let Some(ref frozen) = debug.frozen {
        frozen.chunk_pos
    } else if let Ok(pos) = cam_query.single() {
        crate::camera::chunk_pos(pos)
    } else {
        return;
    };

    if shadow_grid.last_camera_chunk != Some(camera_chunk) {
        shadow_grid.last_camera_chunk = Some(camera_chunk);
        if let Some(source) = source_query.iter().next() {
            shadow_grid.update_origins(camera_chunk, source.end_radius);
        }
    }
}
