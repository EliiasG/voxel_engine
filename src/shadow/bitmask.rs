use bytemuck::{Pod, Zeroable};

use crate::chunk::{ChunkStorage, CHUNK_SIZE, CHUNK_SIZE_3, AIR};

/// Per-chunk occupancy bitmask for shadow ray tracing.
/// Two levels: a coarse 4x4x4 mask and a fine 32x32x32 mask.
/// Size: 8 + 512 * 8 = 4104 bytes per slot.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct ChunkBitmask {
    /// 4x4x4 = 64 bits. Each bit represents an 8^3 sub-region.
    /// Bit index: rx + ry * 4 + rz * 16.
    pub coarse: u64,
    /// 32x32x32 = 32768 bits packed into 512 u64s.
    /// Bit index: x + y * 32 + z * 1024. Word = index / 64, bit = index % 64.
    pub fine: [u64; 512],
}

pub enum ChunkBitmaskResult {
    /// Chunk is entirely air. Grid sentinel 0xFFFFFFFF.
    AllAir,
    /// Chunk is entirely solid. Grid sentinel 0xFFFFFFFE.
    AllSolid,
    /// Mixed — needs a pool slot.
    Partial(ChunkBitmask),
}

/// Build a bitmask from chunk storage.
pub fn build_bitmask(storage: &ChunkStorage) -> ChunkBitmaskResult {
    match storage {
        ChunkStorage::Filled(block) => {
            if *block == AIR {
                ChunkBitmaskResult::AllAir
            } else {
                ChunkBitmaskResult::AllSolid
            }
        }
        ChunkStorage::Paletted { .. } => {
            let mut fine = [0u64; 512];
            let mut coarse = 0u64;
            let mut solid_count = 0u32;

            for z in 0..CHUNK_SIZE {
                for y in 0..CHUNK_SIZE {
                    for x in 0..CHUNK_SIZE {
                        if storage.get(x, y, z) != AIR {
                            solid_count += 1;
                            let fine_idx = x + y * CHUNK_SIZE + z * CHUNK_SIZE * CHUNK_SIZE;
                            fine[fine_idx / 64] |= 1u64 << (fine_idx % 64);

                            let rx = x / 8;
                            let ry = y / 8;
                            let rz = z / 8;
                            let coarse_idx = rx + ry * 4 + rz * 16;
                            coarse |= 1u64 << coarse_idx;
                        }
                    }
                }
            }

            if solid_count == 0 {
                ChunkBitmaskResult::AllAir
            } else if solid_count == CHUNK_SIZE_3 as u32 {
                ChunkBitmaskResult::AllSolid
            } else {
                ChunkBitmaskResult::Partial(ChunkBitmask { coarse, fine })
            }
        }
    }
}

/// Test a single voxel bit in the fine mask.
pub fn test_voxel(bitmask: &ChunkBitmask, x: u32, y: u32, z: u32) -> bool {
    let idx = (x + y * 32 + z * 1024) as usize;
    (bitmask.fine[idx / 64] >> (idx % 64)) & 1 != 0
}

/// Test a coarse region bit (8^3 sub-region).
pub fn test_coarse(bitmask: &ChunkBitmask, rx: u32, ry: u32, rz: u32) -> bool {
    let idx = rx + ry * 4 + rz * 16;
    (bitmask.coarse >> idx) & 1 != 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::ChunkStorage;

    #[test]
    fn test_build_bitmask_all_air() {
        let storage = ChunkStorage::new_filled(AIR);
        assert!(matches!(build_bitmask(&storage), ChunkBitmaskResult::AllAir));
    }

    #[test]
    fn test_build_bitmask_all_solid() {
        let storage = ChunkStorage::new_filled(crate::chunk::STONE);
        assert!(matches!(build_bitmask(&storage), ChunkBitmaskResult::AllSolid));
    }

    #[test]
    fn test_build_bitmask_partial() {
        let mut blocks = vec![AIR; CHUNK_SIZE_3];
        // Place a single solid block at (5, 10, 20)
        blocks[5 + 10 * CHUNK_SIZE + 20 * CHUNK_SIZE * CHUNK_SIZE] = crate::chunk::STONE;
        let storage = ChunkStorage::from_flat_array(&blocks);

        let result = build_bitmask(&storage);
        let bitmask = match result {
            ChunkBitmaskResult::Partial(b) => b,
            _ => panic!("expected Partial"),
        };

        assert!(test_voxel(&bitmask, 5, 10, 20));
        assert!(!test_voxel(&bitmask, 0, 0, 0));
        assert!(!test_voxel(&bitmask, 5, 10, 19));

        // Coarse region (5/8=0, 10/8=1, 20/8=2) should be set
        assert!(test_coarse(&bitmask, 0, 1, 2));
        // Other coarse regions should not be set
        assert!(!test_coarse(&bitmask, 0, 0, 0));
        assert!(!test_coarse(&bitmask, 3, 3, 3));
    }

    #[test]
    fn test_voxel_bit_roundtrip() {
        let mut blocks = vec![AIR; CHUNK_SIZE_3];
        // Set a few known positions
        let positions = [(0, 0, 0), (31, 31, 31), (16, 8, 24), (7, 7, 7), (8, 8, 8)];
        for &(x, y, z) in &positions {
            blocks[x + y * CHUNK_SIZE + z * CHUNK_SIZE * CHUNK_SIZE] = crate::chunk::STONE;
        }
        let storage = ChunkStorage::from_flat_array(&blocks);
        let bitmask = match build_bitmask(&storage) {
            ChunkBitmaskResult::Partial(b) => b,
            _ => panic!("expected Partial"),
        };

        for z in 0..CHUNK_SIZE as u32 {
            for y in 0..CHUNK_SIZE as u32 {
                for x in 0..CHUNK_SIZE as u32 {
                    let expected = positions.contains(&(x as usize, y as usize, z as usize));
                    assert_eq!(
                        test_voxel(&bitmask, x, y, z),
                        expected,
                        "mismatch at ({x}, {y}, {z})"
                    );
                }
            }
        }
    }

    #[test]
    fn test_coarse_bit_consistency() {
        // Build a chunk with scattered blocks
        let mut blocks = vec![AIR; CHUNK_SIZE_3];
        for z in (0..32).step_by(7) {
            for y in (0..32).step_by(11) {
                for x in (0..32).step_by(5) {
                    blocks[x + y * CHUNK_SIZE + z * CHUNK_SIZE * CHUNK_SIZE] = crate::chunk::STONE;
                }
            }
        }
        let storage = ChunkStorage::from_flat_array(&blocks);
        let bitmask = match build_bitmask(&storage) {
            ChunkBitmaskResult::Partial(b) => b,
            _ => panic!("expected Partial"),
        };

        // Every fine bit that is set must have its coarse bit set
        for z in 0..32u32 {
            for y in 0..32u32 {
                for x in 0..32u32 {
                    if test_voxel(&bitmask, x, y, z) {
                        assert!(
                            test_coarse(&bitmask, x / 8, y / 8, z / 8),
                            "fine bit set at ({x},{y},{z}) but coarse ({},{},{}) not set",
                            x / 8,
                            y / 8,
                            z / 8
                        );
                    }
                }
            }
        }

        // Every coarse bit that is set must have at least one fine bit set in its region
        for rz in 0..4u32 {
            for ry in 0..4u32 {
                for rx in 0..4u32 {
                    if test_coarse(&bitmask, rx, ry, rz) {
                        let mut found = false;
                        for dz in 0..8u32 {
                            for dy in 0..8u32 {
                                for dx in 0..8u32 {
                                    if test_voxel(&bitmask, rx * 8 + dx, ry * 8 + dy, rz * 8 + dz)
                                    {
                                        found = true;
                                    }
                                }
                            }
                        }
                        assert!(
                            found,
                            "coarse bit set at ({rx},{ry},{rz}) but no fine bits in region"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_paletted_all_air() {
        // A paletted storage where all blocks happen to be air
        let blocks = vec![AIR; CHUNK_SIZE_3];
        let storage = ChunkStorage::from_flat_array(&blocks);
        // from_flat_array with all same block returns Filled, but let's test the logic
        assert!(matches!(build_bitmask(&storage), ChunkBitmaskResult::AllAir));
    }
}
