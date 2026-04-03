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
    /// Priority-ordered queue of chunks awaiting generation submission.
    /// Highest priority at the end (pop from back).
    pending_generation: Vec<(Entity, IVec3, u8)>,
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
    mut changed: ResMut<ChunkChangedQueue>,
    mut unload_queue: ResMut<ChunkUnloadQueue>,
    mut commands: Commands,
    entity_check: Query<()>,
    load_lists: Query<(Entity, &ChunkLoadList), Changed<ChunkLoadList>>,
    all_load_lists: Query<(Entity, &ChunkLoadList)>,
) {
    let _timer = crate::SysTimer::new(&crate::TIMING_LOADING_US);
    // --- Phase 1: Poll generation results ---
    let results = generator.poll();
    for result in results {
        let key = (result.pos, result.lod);
        loader.in_flight.remove(&key);
        loader.completed.insert(key);

        if entity_check.get(result.entity).is_ok() {
            commands
                .entity(result.entity)
                .insert(ChunkData(Arc::new(result.storage)));
            changed.0.push(ChunkChange {
                entity: result.entity,
                pos: result.pos,
                lod: result.lod,
            });
        }
    }

    // --- Phase 2: Rebuild desired set only when a ChunkLoadList changed ---
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

    // --- Phase 3: Unload chunks no longer desired ---
    let to_remove: Vec<(IVec3, u8)> = if lists_changed {
        loader
            .loaded
            .keys()
            .filter(|k| !loader.desired.contains(k))
            .cloned()
            .collect()
    } else {
        Vec::new()
    };

    for key in &to_remove {
        let (pos, lod) = *key;
        if let Some(entity) = loader.loaded.remove(key) {
            // Push to unload queue for render-side cleanup
            unload_queue.0.push(ChunkUnload { entity, pos, lod });

            // Remove from spatial maps
            lod_maps.maps[lod as usize].remove(&pos);
            loaded_index.0.remove(&(pos, lod));

            commands.entity(entity).despawn();
        }
        loader.in_flight.remove(key);
        loader.completed.remove(key);
    }

    if !to_remove.is_empty() {
        let mut counts: HashMap<u8, usize> = HashMap::new();
        for &(_, lod) in &to_remove {
            *counts.entry(lod).or_insert(0) += 1;
        }
        for (lod, count) in counts {
            println!("Unloaded {count} LOD {lod} chunks");
        }
    }

    // --- Phase 4: Spawn entities for newly desired chunks ---
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

        // Rebuild pending generation queue from segments (priority order).
        // Segments are ordered outer→inner (low→high priority).
        // We iterate in order so highest priority ends up at the back (popped first).
        loader.pending_generation.clear();
        for (_, list) in all_load_lists.iter() {
            for segment in &list.segments {
                for &(pos, lod) in segment {
                    let key = (pos, lod);
                    if let Some(&entity) = loader.loaded.get(&key) {
                        if !loader.in_flight.contains(&key) && !loader.completed.contains(&key) {
                            loader.pending_generation.push((entity, pos, lod));
                        }
                    }
                }
            }
        }
    }

    // --- Phase 5: Submit from pending queue ---
    let capacity = generator.capacity();
    if capacity == 0 || loader.pending_generation.is_empty() {
        return;
    }

    let count = capacity.min(loader.pending_generation.len());
    let start = loader.pending_generation.len() - count;
    let to_submit: Vec<(Entity, IVec3, u8)> = loader.pending_generation.drain(start..).collect();

    for &(_, pos, lod) in &to_submit {
        loader.in_flight.insert((pos, lod));
    }
    generator.submit(&to_submit);
}
