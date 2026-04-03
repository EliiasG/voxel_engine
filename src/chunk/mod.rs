pub mod demand;
pub mod generation;
pub mod loading;
pub mod meshing;

use std::sync::Arc;

use bevy_ecs::prelude::*;
use bytemuck::{Pod, Zeroable};
use glam::IVec3;

pub const CHUNK_SIZE: usize = 32;
pub const CHUNK_SIZE_2: usize = CHUNK_SIZE * CHUNK_SIZE;
pub const CHUNK_SIZE_3: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;

pub type BlockId = u32;
pub const AIR: BlockId = 0;
pub const STONE: BlockId = 1;
pub const DIRT: BlockId = 2;
pub const GRASS: BlockId = 3;
pub const GLASS: BlockId = 4;
pub const WATER: BlockId = 5;

/// Per-block-type properties for rendering.
pub struct BlockProperties {
    /// Block is transparent (rays/light pass through, rendered with WBOIT).
    pub is_transparent: bool,
    /// Index into the absorption coefficient table (0-253). Only meaningful when is_transparent.
    pub color_index: u8,
}

/// Look up properties for a block type.
#[inline]
pub fn block_props(block: BlockId) -> BlockProperties {
    match block {
        GLASS => BlockProperties { is_transparent: true, color_index: 0 },
        WATER => BlockProperties { is_transparent: true, color_index: 1 },
        _ => BlockProperties { is_transparent: false, color_index: 0 },
    }
}

/// Returns true if the block is opaque (not air, not transparent).
#[inline]
pub fn is_opaque(block: BlockId) -> bool {
    block != AIR && !block_props(block).is_transparent
}

/// Returns true if the block is transparent (not air, but light passes through).
#[inline]
pub fn is_transparent(block: BlockId) -> bool {
    block != AIR && block_props(block).is_transparent
}

pub const NUM_DIRECTIONS: usize = 6;
pub const DIR_POS_X: usize = 0;
pub const DIR_NEG_X: usize = 1;
pub const DIR_POS_Y: usize = 2;
pub const DIR_NEG_Y: usize = 3;
pub const DIR_POS_Z: usize = 4;
pub const DIR_NEG_Z: usize = 5;

pub const DIR_OFFSETS: [IVec3; 6] = [
    IVec3::X,     // +X
    IVec3::NEG_X, // -X
    IVec3::Y,     // +Y
    IVec3::NEG_Y, // -Y
    IVec3::Z,     // +Z
    IVec3::NEG_Z, // -Z
];

// --- Face Data (8 bytes, matches GPU vertex layout) ---

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct FaceData {
    pub x: u8,
    pub y: u8,
    pub z: u8,
    pub w: u8,
    pub h: u8,
    pub material: [u8; 3],
}

// --- Palette-compressed chunk storage ---

#[derive(Clone)]
pub enum ChunkStorage {
    Filled(BlockId),
    Paletted {
        palette: Vec<BlockId>,
        data: Vec<u64>,
        bits_per_entry: u32,
    },
}

impl ChunkStorage {
    pub fn new_filled(block: BlockId) -> Self {
        ChunkStorage::Filled(block)
    }

    pub fn from_flat_array(blocks: &[BlockId]) -> Self {
        assert_eq!(blocks.len(), CHUNK_SIZE_3);

        let mut palette = Vec::new();
        for &block in blocks {
            if !palette.contains(&block) {
                palette.push(block);
            }
        }

        if palette.len() == 1 {
            return ChunkStorage::Filled(palette[0]);
        }

        let bits = bits_for_count(palette.len());
        let num_words = packed_words(bits);
        let mut data = vec![0u64; num_words];

        for (i, &block) in blocks.iter().enumerate() {
            let palette_idx = palette.iter().position(|&b| b == block).unwrap();
            write_packed(&mut data, bits, i, palette_idx as u64);
        }

        ChunkStorage::Paletted {
            palette,
            data,
            bits_per_entry: bits,
        }
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize, z: usize) -> BlockId {
        let index = x + y * CHUNK_SIZE + z * CHUNK_SIZE_2;
        match self {
            ChunkStorage::Filled(block) => *block,
            ChunkStorage::Paletted {
                palette,
                data,
                bits_per_entry,
            } => {
                let pi = read_packed(data, *bits_per_entry, index);
                palette[pi as usize]
            }
        }
    }

    pub fn set(&mut self, x: usize, y: usize, z: usize, block: BlockId) {
        let index = x + y * CHUNK_SIZE + z * CHUNK_SIZE_2;
        match self {
            ChunkStorage::Filled(current) => {
                if block == *current {
                    return;
                }
                let old = *current;
                let palette = vec![old, block];
                let bits: u32 = 1;
                let num_words = packed_words(bits);
                let mut data = vec![0u64; num_words];
                write_packed(&mut data, bits, index, 1);
                *self = ChunkStorage::Paletted {
                    palette,
                    data,
                    bits_per_entry: bits,
                };
            }
            ChunkStorage::Paletted {
                palette,
                data,
                bits_per_entry,
            } => {
                let palette_idx = match palette.iter().position(|&b| b == block) {
                    Some(idx) => idx as u64,
                    None => {
                        palette.push(block);
                        let new_idx = (palette.len() - 1) as u64;
                        let needed = bits_for_count(palette.len());
                        if needed > *bits_per_entry {
                            widen_data(data, *bits_per_entry, needed);
                            *bits_per_entry = needed;
                        }
                        new_idx
                    }
                };
                write_packed(data, *bits_per_entry, index, palette_idx);
            }
        }
    }
}

// Variable bit-width: 1, 2, 4, 8, 16 bits per entry
fn bits_for_count(count: usize) -> u32 {
    if count <= 2 {
        1
    } else if count <= 4 {
        2
    } else if count <= 16 {
        4
    } else if count <= 256 {
        8
    } else {
        16
    }
}

fn packed_words(bits: u32) -> usize {
    let total_bits = CHUNK_SIZE_3 as u64 * bits as u64;
    ((total_bits + 63) / 64) as usize
}

#[inline]
fn read_packed(data: &[u64], bits: u32, index: usize) -> u64 {
    let bit_offset = index as u64 * bits as u64;
    let word = (bit_offset / 64) as usize;
    let bit = (bit_offset % 64) as u32;
    let mask = (1u64 << bits) - 1;
    let mut value = (data[word] >> bit) & mask;
    if bit + bits > 64 {
        let overflow = bit + bits - 64;
        value |= (data[word + 1] & ((1u64 << overflow) - 1)) << (bits - overflow);
    }
    value
}

#[inline]
fn write_packed(data: &mut [u64], bits: u32, index: usize, value: u64) {
    let bit_offset = index as u64 * bits as u64;
    let word = (bit_offset / 64) as usize;
    let bit = (bit_offset % 64) as u32;
    let mask = (1u64 << bits) - 1;
    data[word] = (data[word] & !(mask << bit)) | ((value & mask) << bit);
    if bit + bits > 64 {
        let overflow = bit + bits - 64;
        let upper_mask = (1u64 << overflow) - 1;
        data[word + 1] = (data[word + 1] & !upper_mask) | ((value >> (bits - overflow)) & upper_mask);
    }
}

fn widen_data(data: &mut Vec<u64>, old_bits: u32, new_bits: u32) {
    let num_words = packed_words(new_bits);
    let mut new_data = vec![0u64; num_words];
    for i in 0..CHUNK_SIZE_3 {
        let value = read_packed(data, old_bits, i);
        write_packed(&mut new_data, new_bits, i, value);
    }
    *data = new_data;
}

// --- Components ---

#[derive(Component, Clone)]
pub struct ChunkData(pub Arc<ChunkStorage>);

#[derive(Component)]
pub struct ChunkPos(pub IVec3);

#[derive(Component)]
pub struct ChunkLod(pub u8);

// --- Resources ---

#[derive(Resource, Default)]
pub struct ChunkMap {
    map: std::collections::HashMap<IVec3, Entity>,
}

impl ChunkMap {
    pub fn insert(&mut self, pos: IVec3, entity: Entity) {
        self.map.insert(pos, entity);
    }

    pub fn get(&self, pos: &IVec3) -> Option<&Entity> {
        self.map.get(pos)
    }

    pub fn remove(&mut self, pos: &IVec3) -> Option<Entity> {
        self.map.remove(pos)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&IVec3, &Entity)> {
        self.map.iter()
    }
}

/// One ChunkMap per LOD level.
#[derive(Resource)]
pub struct LodChunkMaps {
    pub maps: Vec<ChunkMap>,
}

impl LodChunkMaps {
    pub fn new(lod_count: usize) -> Self {
        Self {
            maps: (0..lod_count).map(|_| ChunkMap::default()).collect(),
        }
    }
}

/// Tracks which (chunk_pos, lod) pairs have been uploaded to GPU.
/// Used for LOD visibility checks (is a region fully covered by finer LOD?).
#[derive(Resource, Default)]
pub struct LoadedChunkIndex(pub std::collections::HashSet<(IVec3, u8)>);

pub struct ChunkChange {
    pub entity: Entity,
    pub pos: IVec3,
    pub lod: u8,
}

/// Change queue. Generation pushes, consumers iterate, cleared at end of frame.
#[derive(Resource, Default)]
pub struct ChunkChangedQueue(pub Vec<ChunkChange>);

pub struct ChunkUnload {
    pub entity: Entity,
    pub pos: IVec3,
    pub lod: u8,
}

/// Unload queue. Loading pushes, render-side cleanup drains.
#[derive(Resource, Default)]
pub struct ChunkUnloadQueue(pub Vec<ChunkUnload>);

/// Convert a LOD 0 chunk coordinate to a LOD N chunk coordinate.
pub fn lod_chunk_pos(lod0_pos: IVec3, lod: u32) -> IVec3 {
    if lod == 0 {
        return lod0_pos;
    }
    let scale = 1i32 << lod;
    IVec3::new(
        lod0_pos.x.div_euclid(scale),
        lod0_pos.y.div_euclid(scale),
        lod0_pos.z.div_euclid(scale),
    )
}

// --- Occupancy bitmask (chunk data, used by shadow system) ---

/// Per-chunk occupancy bitmask for shadow ray tracing.
/// Two levels: a coarse 4x4x4 mask and a fine 32x32x32 mask.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct ChunkBitmask {
    pub coarse: u64,
    pub fine: [u64; 512],
}

pub enum ChunkBitmaskResult {
    AllAir,
    AllSolid,
    Partial(ChunkBitmask),
}

/// Build a bitmask from chunk storage.
///
/// Fine mask: only opaque blocks set a bit (transparent = air for ray traversal).
/// Coarse mask: set for regions containing opaque OR transparent blocks.
pub fn build_bitmask(storage: &ChunkStorage) -> ChunkBitmaskResult {
    match storage {
        ChunkStorage::Filled(block) => {
            if *block == AIR {
                ChunkBitmaskResult::AllAir
            } else if is_opaque(*block) {
                ChunkBitmaskResult::AllSolid
            } else {
                // Filled with transparent block: coarse all set, fine all clear
                let coarse = !0u64; // all 64 bits set
                let fine = [0u64; 512];
                ChunkBitmaskResult::Partial(ChunkBitmask { coarse, fine })
            }
        }
        ChunkStorage::Paletted { .. } => {
            let mut fine = [0u64; 512];
            let mut coarse = 0u64;
            let mut opaque_count = 0u32;
            let mut non_air_count = 0u32;

            for z in 0..CHUNK_SIZE {
                for y in 0..CHUNK_SIZE {
                    for x in 0..CHUNK_SIZE {
                        let block = storage.get(x, y, z);
                        if block == AIR {
                            continue;
                        }
                        non_air_count += 1;

                        // Coarse bit: set for any non-air block (opaque or transparent)
                        let rx = x / 8;
                        let ry = y / 8;
                        let rz = z / 8;
                        let coarse_idx = rx + ry * 4 + rz * 16;
                        coarse |= 1u64 << coarse_idx;

                        // Fine bit: only set for opaque blocks
                        if is_opaque(block) {
                            opaque_count += 1;
                            let fine_idx = x + y * CHUNK_SIZE + z * CHUNK_SIZE * CHUNK_SIZE;
                            fine[fine_idx / 64] |= 1u64 << (fine_idx % 64);
                        }
                    }
                }
            }

            if non_air_count == 0 {
                ChunkBitmaskResult::AllAir
            } else if opaque_count == CHUNK_SIZE_3 as u32 {
                ChunkBitmaskResult::AllSolid
            } else {
                ChunkBitmaskResult::Partial(ChunkBitmask { coarse, fine })
            }
        }
    }
}

/// Clears the chunk changed queue. Scheduled after all consumers have run.
pub fn clear_chunk_changed_queue(mut changed: ResMut<ChunkChangedQueue>) {
    changed.0.clear();
}
