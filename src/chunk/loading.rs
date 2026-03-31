use std::collections::VecDeque;

use bevy_ecs::prelude::*;
use glam::IVec3;

use super::*;

#[derive(Resource, Clone)]
pub struct LoadConfig {
    pub start_radius: u32,
    pub step: u32,
    pub end_radius: u32,
    pub lod_count: u32,
}

impl Default for LoadConfig {
    fn default() -> Self {
        Self {
            start_radius: 4,
            step: 2,
            end_radius: 8,
            lod_count: 8,
        }
    }
}

struct LoadJob {
    lod: u32,
    radius: u32,
}

struct ActiveJob {
    pending_entities: Vec<Entity>,
}

#[derive(Resource, Default)]
pub struct Loader {
    jobs: VecDeque<LoadJob>,
    current_job: Option<ActiveJob>,
    last_origin: Option<IVec3>,
}

fn build_job_queue(config: &LoadConfig) -> VecDeque<LoadJob> {
    let mut jobs = VecDeque::new();
    let mut radius = config.start_radius;
    while radius <= config.end_radius {
        for lod in 0..config.lod_count {
            jobs.push_back(LoadJob { lod, radius });
        }
        radius += config.step;
    }
    jobs
}

/// Main loader system: tracks camera, unloads distant chunks, spawns new ones via job queue.
pub fn update_loader(
    camera: Res<crate::Camera>,
    debug: Res<crate::DebugMode>,
    config: Res<LoadConfig>,
    mut loader: ResMut<Loader>,
    mut lod_maps: ResMut<LodChunkMaps>,
    mut loaded_index: ResMut<LoadedChunkIndex>,
    mut render_data: ResMut<crate::render::ChunkRenderData>,
    mut gpu: ResMut<crate::render::GpuBuffers>,
    mut shadow_grid: ResMut<crate::shadow::grid::ShadowGrid>,
    mut bitmask_pool: ResMut<crate::shadow::grid::BitmaskPool>,
    mut commands: Commands,
    chunk_data_query: Query<(), With<ChunkData>>,
) {
    let camera_chunk = if let Some(ref frozen) = debug.frozen {
        frozen.chunk_pos
    } else {
        camera.0.chunk_pos()
    };

    // Detect camera chunk change
    let origin_changed = loader.last_origin != Some(camera_chunk);
    if origin_changed {
        loader.last_origin = Some(camera_chunk);

        // Unload chunks outside end_radius for each LOD
        for lod in 0..config.lod_count {
            let lod_cam = lod_chunk_pos(camera_chunk, lod);
            let max_r = config.end_radius as i32;

            let to_remove: Vec<(IVec3, Entity)> = lod_maps.maps[lod as usize]
                .iter()
                .filter(|(pos, _)| {
                    let d = (**pos - lod_cam).abs();
                    d.x > max_r || d.y > max_r || d.z > max_r
                })
                .map(|(pos, entity)| (*pos, *entity))
                .collect();

            for (pos, entity) in &to_remove {
                lod_maps.maps[lod as usize].remove(pos);
                loaded_index.0.remove(&(*pos, lod as u8));
                if let Some(entry) = render_data.entries.remove(entity) {
                    for dir_pages in &entry.directions {
                        for page in &dir_pages.pages {
                            crate::render::PageAllocator::deallocate(
                                &mut gpu, page.slab_index as usize, page.page_index,
                            );
                        }
                    }
                }
                crate::shadow::grid::remove_chunk_from_grid(
                    &mut shadow_grid, &mut bitmask_pool, *pos, lod as u8,
                );
                commands.entity(*entity).despawn();
            }

            if !to_remove.is_empty() {
                println!(
                    "Unloaded {} LOD {} chunks",
                    to_remove.len(),
                    lod
                );
            }
        }

        // Update shadow grid origins for new camera position
        shadow_grid.rebuild_origins(camera_chunk, &config);

        // Rebuild job queue from current position
        loader.jobs = build_job_queue(&config);
        loader.current_job = None;
    }

    // Process jobs
    loop {
        // Check if current job is done (all entities have ChunkData)
        if let Some(ref active) = loader.current_job {
            let all_done = active
                .pending_entities
                .iter()
                .all(|&e| chunk_data_query.get(e).is_ok());
            if !all_done {
                return;
            }
        }
        loader.current_job = None;

        // Pop next job
        let Some(job) = loader.jobs.pop_front() else {
            return;
        };

        let lod = job.lod as u8;
        let origin = lod_chunk_pos(camera_chunk, job.lod);
        let radius = job.radius as i32;

        let mut pending = Vec::new();

        for x in -radius..=radius {
            for y in -radius..=radius {
                for z in -radius..=radius {
                    let pos = origin + IVec3::new(x, y, z);
                    if lod_maps.maps[lod as usize].get(&pos).is_some() {
                        continue;
                    }
                    let entity = commands
                        .spawn((ChunkPos(pos), ChunkLod(lod), NeedsGeneration))
                        .id();
                    lod_maps.maps[lod as usize].insert(pos, entity);
                    pending.push(entity);
                }
            }
        }

        if !pending.is_empty() {
            println!(
                "Loading LOD {} radius {}: {} chunks",
                lod,
                job.radius,
                pending.len()
            );
            loader.current_job = Some(ActiveJob {
                pending_entities: pending,
            });
            return;
        }
        // No-op job (all chunks already loaded), skip to next
    }
}
