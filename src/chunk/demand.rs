use bevy_ecs::prelude::*;
use glam::IVec3;

use super::lod_chunk_pos;

/// Declares that this entity is a source of chunk loading demand.
/// Attach to any entity with a `Position` component (typically the camera).
#[derive(Component, Clone)]
pub struct ChunkSource {
    /// Radius of the innermost (highest-priority) batch, in chunk coords per LOD.
    pub start_radius: u32,
    /// Radius increment per priority segment.
    pub step: u32,
    /// Maximum loading radius (outermost shell).
    pub end_radius: u32,
    /// Number of LOD levels to load.
    pub lod_count: u32,
}

impl Default for ChunkSource {
    fn default() -> Self {
        Self {
            start_radius: 4,
            step: 2,
            end_radius: 8,
            lod_count: 8,
        }
    }
}

/// Priority-ordered list of chunks this source wants loaded.
///
/// `segments[0]` = outermost shell (lowest priority).
/// `segments.last()` = innermost region (highest priority).
///
/// Each entry is `(chunk_pos_at_lod, lod)`.
#[derive(Component, Default)]
pub struct ChunkLoadList {
    pub segments: Vec<Vec<(IVec3, u8)>>,
}

/// Builds `ChunkLoadList` from each entity's `Position` + `ChunkSource`.
pub fn update_chunk_demand(
    debug: Res<crate::DebugMode>,
    mut query: Query<(
        &crate::camera::Position,
        &ChunkSource,
        &mut ChunkLoadList,
    )>,
) {
    for (pos, source, mut load_list) in query.iter_mut() {
        let camera_chunk = if let Some(ref frozen) = debug.frozen {
            frozen.chunk_pos
        } else {
            crate::camera::chunk_pos(pos)
        };

        // Build segments from outermost (low priority) to innermost (high priority).
        // Shells: [end_radius..end_radius-step+1], [end_radius-step..end_radius-2*step+1], ..., [start_radius..0]
        let mut segments = Vec::new();

        let mut outer = source.end_radius as i32;
        while outer > 0 {
            // -1 for the innermost segment so we include max_d == 0 (the origin)
            let inner = if outer as u32 <= source.start_radius {
                -1
            } else {
                (outer - source.step as i32).max(0)
            };

            // One segment per LOD within this shell (high LOD numbers first = low priority).
            for lod in (0..source.lod_count).rev() {
                let origin = lod_chunk_pos(camera_chunk, lod);
                let r = outer;
                let mut segment = Vec::new();
                for x in -r..=r {
                    for y in -r..=r {
                        for z in -r..=r {
                            let d = IVec3::new(x, y, z).abs();
                            let max_d = d.x.max(d.y).max(d.z);
                            if max_d > r || max_d <= inner {
                                continue;
                            }
                            segment.push((origin + IVec3::new(x, y, z), lod as u8));
                        }
                    }
                }
                if !segment.is_empty() {
                    segments.push(segment);
                }
            }

            if inner == 0 {
                break;
            }
            outer = inner;
        }

        load_list.segments = segments;
    }
}
