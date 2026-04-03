pub use crate::chunk::ChunkBitmask;

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
    use crate::chunk::{build_bitmask, ChunkBitmaskResult, ChunkStorage, CHUNK_SIZE, CHUNK_SIZE_3, AIR};

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

        assert!(test_coarse(&bitmask, 0, 1, 2));
        assert!(!test_coarse(&bitmask, 0, 0, 0));
        assert!(!test_coarse(&bitmask, 3, 3, 3));
    }

    #[test]
    fn test_voxel_bit_roundtrip() {
        let mut blocks = vec![AIR; CHUNK_SIZE_3];
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
        let blocks = vec![AIR; CHUNK_SIZE_3];
        let storage = ChunkStorage::from_flat_array(&blocks);
        assert!(matches!(build_bitmask(&storage), ChunkBitmaskResult::AllAir));
    }
}
