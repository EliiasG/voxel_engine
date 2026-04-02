use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use bevy_ecs::prelude::*;
use glam::IVec3;

use super::demand::ChunkLoadList;
use super::generation::{ChunkGenerator, GenPool};
use super::*;

/// Central chunk lifecycle manager.
///
/// Tracks all chunk entities, manages refcounting across multiple sources,
/// and schedules generation via round-robin fairness across sources.
#[derive(Resource, Default)]
pub struct ChunkLoader {
    /// All chunks with spawned entities: (pos, lod) → entity.
    loaded: HashMap<(IVec3, u8), Entity>,
    /// Chunks submitted to the generator, awaiting results.
    in_flight: HashSet<(IVec3, u8)>,
    /// Chunks whose generation completed (have ChunkData).
    completed: HashSet<(IVec3, u8)>,
    /// Cached desired set — only rebuilt when a ChunkLoadList changes.
    desired: HashSet<(IVec3, u8)>,
    /// Round-robin index across sources.
    round_robin_index: usize,
    /// Last camera chunk seen — drives shadow grid origin rebuilds.
    last_camera_chunk: Option<IVec3>,
}


/// Main chunk loading system.
///
/// Runs every frame: polls generation results, diffs desired vs loaded,
/// spawns/despawns entities, and round-robin submits to the generator.
pub fn update_chunk_loading(
    mut loader: ResMut<ChunkLoader>,
    generator: Res<GenPool>,
    mut lod_maps: ResMut<LodChunkMaps>,
    mut loaded_index: ResMut<LoadedChunkIndex>,
    mut render_data: ResMut<crate::render::ChunkRenderData>,
    mut gpu: ResMut<crate::render::GpuBuffers>,
    mut shadow_grid: ResMut<crate::render::shadow::grid::ShadowGrid>,
    mut bitmask_pool: ResMut<crate::render::shadow::grid::BitmaskPool>,
    mut changed: ResMut<ChunkChangedQueue>,
    mut commands: Commands,
    entity_check: Query<()>,
    load_lists: Query<(Entity, &ChunkLoadList), Changed<ChunkLoadList>>,
    all_load_lists: Query<(Entity, &ChunkLoadList)>,
    cam_query: Query<&crate::camera::Position, With<crate::camera::MainCamera>>,
    source_query: Query<&super::demand::ChunkSource>,
    debug: Res<crate::DebugMode>,
) {
    // --- Phase 1: Poll generation results ---
    let results = generator.poll();
    for result in results {
        let key = (result.pos, result.lod);
        loader.in_flight.remove(&key);
        loader.completed.insert(key);

        if entity_check.get(result.entity).is_ok() {
            crate::render::shadow::grid::update_grid_for_chunk(
                &mut shadow_grid,
                &mut bitmask_pool,
                result.pos,
                result.lod,
                result.bitmask,
            );
            commands
                .entity(result.entity)
                .insert(ChunkData(Arc::new(result.storage)));
            changed.0.push(ChunkChange {
                pos: result.pos,
                lod: result.lod,
            });
        }
    }

    // --- Phase 2: Determine camera chunk, rebuild shadow origins if moved ---
    let camera_chunk = if let Some(ref frozen) = debug.frozen {
        frozen.chunk_pos
    } else if let Ok(pos) = cam_query.get_single() {
        crate::camera::chunk_pos(pos)
    } else {
        return;
    };

    if loader.last_camera_chunk != Some(camera_chunk) {
        loader.last_camera_chunk = Some(camera_chunk);

        // Find the largest end_radius from any source for shadow grid
        if let Some(source) = source_query.iter().next() {
            shadow_grid.rebuild_origins(camera_chunk, source.end_radius);
        }
    }

    // --- Phase 3: Rebuild desired set only when a ChunkLoadList changed ---
    let lists_changed = load_lists.iter().count() > 0;
    if lists_changed {
        loader.desired.clear();
        for (_, list) in all_load_lists.iter() {
            for segment in &list.segments {
                for &(pos, lod) in segment {
                    loader.desired.insert((pos, lod));
                }
            }
        }
    }

    // --- Phase 4: Unload chunks no longer desired ---
    let to_remove: Vec<(IVec3, u8)> = loader
        .loaded
        .keys()
        .filter(|k| !loader.desired.contains(k))
        .cloned()
        .collect();

    for key in &to_remove {
        let (pos, lod) = *key;
        if let Some(entity) = loader.loaded.remove(key) {
            // Deallocate GPU pages
            if let Some(entry) = render_data.entries.remove(&entity) {
                for dir_pages in &entry.directions {
                    for page in &dir_pages.pages {
                        crate::render::PageAllocator::deallocate(
                            &mut gpu,
                            page.slab_index as usize,
                            page.page_index,
                        );
                    }
                }
            }

            // Remove from shadow grid
            crate::render::shadow::grid::remove_chunk_from_grid(
                &mut shadow_grid,
                &mut bitmask_pool,
                pos,
                lod,
            );

            // Remove from spatial maps
            lod_maps.maps[lod as usize].remove(&pos);
            loaded_index.0.remove(&(pos, lod));

            commands.entity(entity).despawn();
        }
        loader.in_flight.remove(key);
        loader.completed.remove(key);
    }

    if !to_remove.is_empty() {
        // Group by LOD for logging
        let mut counts: HashMap<u8, usize> = HashMap::new();
        for &(_, lod) in &to_remove {
            *counts.entry(lod).or_insert(0) += 1;
        }
        for (lod, count) in counts {
            println!("Unloaded {count} LOD {lod} chunks");
        }
    }

    // --- Phase 5: Spawn entities for newly desired chunks ---
    if lists_changed {
        let to_spawn: Vec<(IVec3, u8)> = loader
            .desired
            .iter()
            .filter(|k| !loader.loaded.contains_key(k))
            .cloned()
            .collect();

        for (pos, lod) in &to_spawn {
            let entity = commands.spawn((ChunkPos(*pos), ChunkLod(*lod))).id();
            lod_maps.maps[*lod as usize].insert(*pos, entity);
            loader.loaded.insert((*pos, *lod), entity);
        }

        if !to_spawn.is_empty() {
            println!("Spawned {} new chunk entities", to_spawn.len());
        }
    }

    // --- Phase 6: Round-robin generation submission ---
    let capacity = generator.capacity();
    if capacity == 0 {
        return;
    }

    // Build per-source pending lists (only chunks needing generation).
    // Each source's segments are filtered to exclude completed and in-flight chunks.
    let mut active_loaders: Vec<(Entity, Vec<Vec<(IVec3, u8)>>)> = Vec::new();
    for (source_entity, list) in all_load_lists.iter() {
        let mut remaining_segments: Vec<Vec<(IVec3, u8)>> = Vec::new();
        for segment in &list.segments {
            let filtered: Vec<(IVec3, u8)> = segment
                .iter()
                .filter(|&&(pos, lod)| {
                    let key = (pos, lod);
                    loader.loaded.contains_key(&key)
                        && !loader.in_flight.contains(&key)
                        && !loader.completed.contains(&key)
                })
                .cloned()
                .collect();
            if !filtered.is_empty() {
                remaining_segments.push(filtered);
            }
        }
        if !remaining_segments.is_empty() {
            active_loaders.push((source_entity, remaining_segments));
        }
    }

    if active_loaders.is_empty() {
        return;
    }

    // Round-robin: process highest-priority segment from each loader in turn.
    let mut to_submit: Vec<(Entity, IVec3, u8)> = Vec::new();
    let mut remaining_capacity = capacity;
    let mut index = loader.round_robin_index % active_loaders.len().max(1);

    loop {
        if active_loaders.is_empty() || remaining_capacity == 0 {
            break;
        }

        if index >= active_loaders.len() {
            index = 0;
        }

        let (_, ref mut segments) = active_loaders[index];
        // Last segment = highest priority
        if let Some(segment) = segments.last_mut() {
            while remaining_capacity > 0 && !segment.is_empty() {
                let (pos, lod) = segment.pop().unwrap();
                // Look up the entity we spawned for this chunk
                if let Some(&entity) = loader.loaded.get(&(pos, lod)) {
                    to_submit.push((entity, pos, lod));
                    remaining_capacity -= 1;
                }
            }
            if segment.is_empty() {
                segments.pop();
                if segments.is_empty() {
                    active_loaders.remove(index);
                    // Don't increment index — next loader slides into this slot
                    continue;
                }
            }
        }

        index += 1;
    }

    loader.round_robin_index = index;

    if !to_submit.is_empty() {
        // Mark as in-flight
        for &(_, pos, lod) in &to_submit {
            loader.in_flight.insert((pos, lod));
        }

        let requests: Vec<(Entity, IVec3, u8)> = to_submit;
        generator.submit(&requests);
    }
}
