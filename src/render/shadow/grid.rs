use std::collections::HashMap;

use bevy_ecs::prelude::*;
use bytemuck::{Pod, Zeroable};
use glam::IVec3;

use crate::chunk::loading::LoadConfig;
use crate::chunk::lod_chunk_pos;

use super::bitmask::{ChunkBitmask, ChunkBitmaskResult};

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
}

impl ShadowGrid {
    pub fn new(config: &LoadConfig) -> Self {
        let grid_size = config.end_radius * 2 + 1;
        let entries_per_lod = (grid_size * grid_size * grid_size) as usize;
        let lod_count = config.lod_count;

        let lod_infos = (0..lod_count)
            .map(|lod| LodInfo {
                grid_origin: [0; 3],
                grid_size,
                lod_scale: 1 << lod,
                _pad: [0; 3],
            })
            .collect();

        let grid_data = vec![GRID_EMPTY; entries_per_lod * lod_count as usize];

        Self {
            lod_infos,
            grid_data,
            grid_size,
            lod_count,
            dirty: true,
            chunk_values: HashMap::new(),
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
    pub fn rebuild_origins(&mut self, camera_chunk: IVec3, config: &LoadConfig) {
        let radius = config.end_radius as i32;
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
            for (&(pos, lod), &value) in &self.chunk_values {
                if let Some(i) = self.grid_index(lod, pos) {
                    self.grid_data[i] = value;
                }
            }
            self.dirty = true;
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

/// Called when a chunk finishes generation. Updates the grid and pool.
pub fn update_grid_for_chunk(
    grid: &mut ShadowGrid,
    pool: &mut BitmaskPool,
    chunk_pos: IVec3,
    lod: u8,
    result: ChunkBitmaskResult,
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
        ChunkBitmaskResult::Partial(bitmask) => pool.allocate(bitmask),
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
