use std::sync::atomic::{AtomicUsize, Ordering};

use bevy_ecs::prelude::*;
use crossbeam_channel::{Receiver, Sender};
use glam::IVec3;

use super::{
    ChunkStorage, AIR, CHUNK_SIZE, CHUNK_SIZE_2, CHUNK_SIZE_3, DIRT, GRASS, STONE,
};
use crate::render::shadow::bitmask::{self, ChunkBitmaskResult};

pub struct GenResult {
    pub entity: Entity,
    pub pos: IVec3,
    pub lod: u8,
    pub storage: ChunkStorage,
    pub bitmask: ChunkBitmaskResult,
}

/// Trait for chunk generation backends.
/// The loading system calls these methods to submit work and collect results.
pub trait ChunkGenerator {
    /// How many more requests can be accepted right now.
    fn capacity(&self) -> usize;

    /// Submit generation requests. Each tuple is (entity, chunk_pos, lod).
    fn submit(&self, requests: &[(Entity, IVec3, u8)]);

    /// Drain all completed results available this frame.
    fn poll(&self) -> Vec<GenResult>;
}

struct GenRequest {
    entity: Entity,
    pos: IVec3,
    lod: u8,
}

/// Channel-based worker pool for chunk generation.
#[derive(Resource)]
pub struct GenPool {
    tx: Option<Sender<GenRequest>>,
    rx: Receiver<GenResult>,
    in_flight: AtomicUsize,
    max_in_flight: usize,
    workers: Vec<std::thread::JoinHandle<()>>,
}

impl GenPool {
    pub fn new() -> Self {
        let (req_tx, req_rx) = crossbeam_channel::unbounded::<GenRequest>();
        let (res_tx, res_rx) = crossbeam_channel::unbounded::<GenResult>();

        let num_threads = std::thread::available_parallelism()
            .map(|n| (n.get() / 2).max(1))
            .unwrap_or(2);

        let mut workers = Vec::with_capacity(num_threads);
        for i in 0..num_threads {
            let req_rx = req_rx.clone();
            let res_tx = res_tx.clone();
            let handle = std::thread::Builder::new()
                .name(format!("gen-worker-{i}"))
                .spawn(move || {
                    while let Ok(req) = req_rx.recv() {
                        let storage = generate_terrain(req.pos, req.lod);
                        let bitmask = bitmask::build_bitmask(&storage);
                        let _ = res_tx.send(GenResult {
                            entity: req.entity,
                            pos: req.pos,
                            lod: req.lod,
                            storage,
                            bitmask,
                        });
                    }
                })
                .expect("failed to spawn gen worker");
            workers.push(handle);
        }

        let max_in_flight = num_threads * 4;
        println!("Gen pool: {num_threads} threads, max in-flight: {max_in_flight}");
        Self {
            tx: Some(req_tx),
            rx: res_rx,
            in_flight: AtomicUsize::new(0),
            max_in_flight,
            workers,
        }
    }
}

impl Drop for GenPool {
    fn drop(&mut self) {
        // Drop the sender so workers' recv() returns Err and they exit.
        self.tx.take();
        for handle in self.workers.drain(..) {
            let _ = handle.join();
        }
    }
}

impl ChunkGenerator for GenPool {
    fn capacity(&self) -> usize {
        let current = self.in_flight.load(Ordering::Relaxed);
        self.max_in_flight.saturating_sub(current)
    }

    fn submit(&self, requests: &[(Entity, IVec3, u8)]) {
        let Some(tx) = &self.tx else { return };
        for &(entity, pos, lod) in requests {
            let _ = tx.send(GenRequest { entity, pos, lod });
        }
        self.in_flight.fetch_add(requests.len(), Ordering::Relaxed);
    }

    fn poll(&self) -> Vec<GenResult> {
        let mut results = Vec::new();
        while let Ok(result) = self.rx.try_recv() {
            results.push(result);
        }
        if !results.is_empty() {
            self.in_flight.fetch_sub(results.len(), Ordering::Relaxed);
        }
        results
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
    let h0 = height_at(wx0 as f32, wz0 as f32);
    let first_block = if wy0 > h0 {
        AIR
    } else if wy0 >= h0 {
        GRASS
    } else if wy0 >= h0 - 3 {
        DIRT
    } else {
        STONE
    };

    for z in 0..CHUNK_SIZE {
        for x in 0..CHUNK_SIZE {
            let wx = (wx0 + x as i32 * lod_scale) as f32;
            let wz = (wz0 + z as i32 * lod_scale) as f32;
            let height = height_at(wx, wz);

            for y in 0..CHUNK_SIZE {
                let wy = wy0 + y as i32 * lod_scale;
                let block = if wy > height {
                    AIR
                } else if wy >= height - 0 {
                    GRASS
                } else if wy >= height - 3 {
                    DIRT
                } else {
                    STONE
                };
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
