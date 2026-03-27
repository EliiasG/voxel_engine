use std::sync::Arc;

use bevy_ecs::prelude::*;
use crossbeam_channel::{Receiver, Sender};
use glam::IVec3;

use super::{
    ChunkChange, ChunkChangedQueue, ChunkData, ChunkLod, ChunkPos, ChunkStorage,
    NeedsGeneration, AIR, CHUNK_SIZE, CHUNK_SIZE_2, CHUNK_SIZE_3, STONE,
};

struct GenRequest {
    entity: Entity,
    pos: IVec3,
    lod: u8,
}

struct GenResult {
    entity: Entity,
    pos: IVec3,
    lod: u8,
    storage: ChunkStorage,
}

/// Channel-based worker pool for chunk generation.
#[derive(Resource)]
pub struct GenPool {
    tx: Sender<GenRequest>,
    rx: Receiver<GenResult>,
}

impl GenPool {
    pub fn new() -> Self {
        let (req_tx, req_rx) = crossbeam_channel::unbounded::<GenRequest>();
        let (res_tx, res_rx) = crossbeam_channel::unbounded::<GenResult>();

        let num_threads = std::thread::available_parallelism()
            .map(|n| (n.get() / 2).max(1))
            .unwrap_or(2);

        for i in 0..num_threads {
            let req_rx = req_rx.clone();
            let res_tx = res_tx.clone();
            std::thread::Builder::new()
                .name(format!("gen-worker-{i}"))
                .spawn(move || {
                    while let Ok(req) = req_rx.recv() {
                        let storage = generate_terrain(req.pos, req.lod);
                        let _ = res_tx.send(GenResult {
                            entity: req.entity,
                            pos: req.pos,
                            lod: req.lod,
                            storage,
                        });
                    }
                })
                .expect("failed to spawn gen worker");
        }

        println!("Gen pool: {num_threads} threads");
        Self { tx: req_tx, rx: res_rx }
    }
}

/// Sends generation requests for all NeedsGeneration entities.
pub fn start_generation(
    mut commands: Commands,
    query: Query<(Entity, &ChunkPos, &ChunkLod), With<NeedsGeneration>>,
    pool: Res<GenPool>,
) {
    for (entity, pos, lod) in query.iter() {
        let _ = pool.tx.send(GenRequest {
            entity,
            pos: pos.0,
            lod: lod.0,
        });
        commands.entity(entity).remove::<NeedsGeneration>();
    }
}

/// Drains completed generation results from the worker pool.
pub fn poll_generation(
    mut commands: Commands,
    pool: Res<GenPool>,
    mut changed: ResMut<ChunkChangedQueue>,
    entity_check: Query<()>,
) {
    while let Ok(result) = pool.rx.try_recv() {
        if entity_check.get(result.entity).is_ok() {
            commands
                .entity(result.entity)
                .insert(ChunkData(Arc::new(result.storage)));
            changed.0.push(ChunkChange {
                pos: result.pos,
                lod: result.lod,
            });
        }
    }
}

fn generate_terrain(chunk_pos: IVec3, lod: u8) -> ChunkStorage {
    use fastnoise_lite::{FastNoiseLite, FractalType, NoiseType};

    let mut noise1 = FastNoiseLite::new();
    noise1.set_noise_type(Some(NoiseType::Perlin));
    noise1.set_fractal_type(Some(FractalType::FBm));
    noise1.set_fractal_octaves(Some(7));
    noise1.set_fractal_lacunarity(Some(2.0));
    noise1.set_fractal_gain(Some(0.5));

    let mut noise2 = FastNoiseLite::new();
    noise2.set_noise_type(Some(NoiseType::Perlin));
    noise2.set_fractal_type(Some(FractalType::FBm));
    noise2.set_fractal_octaves(Some(16));
    noise2.set_fractal_lacunarity(Some(2.0));
    noise2.set_fractal_gain(Some(0.5));

    let lod_scale = 1i32 << lod;
    let scale1 = 0.025;
    let scale2 = 0.5;
    let amplitude1 = 40000.0f32;
    let amplitude2 = 200.0f32;

    let wx0 = chunk_pos.x * CHUNK_SIZE as i32 * lod_scale;
    let wy0 = chunk_pos.y * CHUNK_SIZE as i32 * lod_scale;
    let wz0 = chunk_pos.z * CHUNK_SIZE as i32 * lod_scale;

    let height_at = |wx: f32, wz: f32| -> i32 {
        let n1 = (noise1.get_noise_2d(wx * scale1, wz * scale1) + 1.0) * 0.5;
        let n2 = (noise2.get_noise_2d(wx * scale2, wz * scale2) + 1.0) * 0.5;
        (n1.powi(8) * amplitude1 + n2 * amplitude2) as i32
    };

    let mut blocks = vec![AIR; CHUNK_SIZE_3];
    let mut all_same = true;
    let first_block = if wy0 <= height_at(wx0 as f32, wz0 as f32) {
        STONE
    } else {
        AIR
    };

    for z in 0..CHUNK_SIZE {
        for x in 0..CHUNK_SIZE {
            let wx = (wx0 + x as i32 * lod_scale) as f32;
            let wz = (wz0 + z as i32 * lod_scale) as f32;
            let height = height_at(wx, wz);

            for y in 0..CHUNK_SIZE {
                let wy = wy0 + y as i32 * lod_scale;
                let block = if wy <= height { STONE } else { AIR };
                if block != first_block {
                    all_same = false;
                }
                blocks[x + y * CHUNK_SIZE + z * CHUNK_SIZE_2] = block;
            }
        }
    }

    if all_same {
        ChunkStorage::new_filled(first_block)
    } else {
        ChunkStorage::from_flat_array(&blocks)
    }
}
