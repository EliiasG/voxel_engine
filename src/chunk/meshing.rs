use std::sync::Arc;

use bevy_ecs::prelude::*;
use crossbeam_channel::{Receiver, Sender};

use super::*;

#[derive(Component)]
#[component(storage = "SparseSet")]
pub struct NeedsRemesh;

#[derive(Component)]
#[component(storage = "SparseSet")]
pub struct ChunkFaces(pub [Vec<FaceData>; NUM_DIRECTIONS]);

struct MeshRequest {
    entity: Entity,
    storage: Arc<ChunkStorage>,
    neighbors: [Option<Arc<ChunkStorage>>; 6],
}

struct MeshResult {
    entity: Entity,
    faces: [Vec<FaceData>; NUM_DIRECTIONS],
}

/// Channel-based worker pool for chunk meshing.
#[derive(Resource)]
pub struct MeshPool {
    tx: Sender<MeshRequest>,
    rx: Receiver<MeshResult>,
}

impl MeshPool {
    pub fn new() -> Self {
        let (req_tx, req_rx) = crossbeam_channel::unbounded::<MeshRequest>();
        let (res_tx, res_rx) = crossbeam_channel::unbounded::<MeshResult>();

        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        for i in 0..num_threads {
            let req_rx = req_rx.clone();
            let res_tx = res_tx.clone();
            std::thread::Builder::new()
                .name(format!("mesh-worker-{i}"))
                .spawn(move || {
                    while let Ok(req) = req_rx.recv() {
                        let face_buffers = extract_face_buffers(&req.storage, &req.neighbors);
                        let faces = greedy_mesh_buffers(&face_buffers);
                        let _ = res_tx.send(MeshResult {
                            entity: req.entity,
                            faces,
                        });
                    }
                })
                .expect("failed to spawn mesh worker");
        }

        println!("Mesh pool: {num_threads} threads");
        Self { tx: req_tx, rx: res_rx }
    }
}

/// Consumes ChunkChangedQueue, marks affected chunks + neighbors with NeedsRemesh.
/// Neighbors are only marked if they've already been uploaded (in LoadedChunkIndex),
/// avoiding a remesh cascade during initial loading. The old mesh stays on GPU
/// until the new one is uploaded, so no visual holes.
pub fn resolve_changes(
    mut commands: Commands,
    mut changed: ResMut<ChunkChangedQueue>,
    lod_maps: Res<LodChunkMaps>,
    loaded_index: Res<LoadedChunkIndex>,
) {
    for change in changed.0.drain(..) {
        let map = &lod_maps.maps[change.lod as usize];
        if let Some(&entity) = map.get(&change.pos) {
            commands.entity(entity).insert(NeedsRemesh);
        }
        for offset in &DIR_OFFSETS {
            let neighbor_pos = change.pos + *offset;
            if loaded_index.0.contains(&(neighbor_pos, change.lod)) {
                if let Some(&entity) = map.get(&neighbor_pos) {
                    commands.entity(entity).insert(NeedsRemesh);
                }
            }
        }
    }
}

/// Sends mesh requests for all NeedsRemesh + ChunkData entities.
pub fn start_meshing(
    mut commands: Commands,
    query: Query<(Entity, &ChunkPos, &ChunkLod, &ChunkData), With<NeedsRemesh>>,
    lod_maps: Res<LodChunkMaps>,
    chunk_data_query: Query<&ChunkData>,
    pool: Res<MeshPool>,
) {
    for (entity, pos, lod, data) in query.iter() {
        let map = &lod_maps.maps[lod.0 as usize];
        let neighbors: [Option<Arc<ChunkStorage>>; 6] = std::array::from_fn(|dir| {
            let neighbor_pos = pos.0 + DIR_OFFSETS[dir];
            map.get(&neighbor_pos)
                .and_then(|&e| chunk_data_query.get(e).ok())
                .map(|cd| cd.0.clone())
        });

        let _ = pool.tx.send(MeshRequest {
            entity,
            storage: data.0.clone(),
            neighbors,
        });
        commands.entity(entity).remove::<NeedsRemesh>();
    }
}

/// Drains completed mesh results from the worker pool.
pub fn poll_meshing(
    mut commands: Commands,
    pool: Res<MeshPool>,
    entity_check: Query<()>,
) {
    while let Ok(result) = pool.rx.try_recv() {
        if entity_check.get(result.entity).is_ok() {
            commands.entity(result.entity).insert(ChunkFaces(result.faces));
        }
    }
}

fn get_neighbor_block(
    storage: &Option<Arc<ChunkStorage>>,
    x: usize,
    y: usize,
    z: usize,
) -> BlockId {
    match storage {
        Some(s) => s.get(x, y, z),
        None => AIR,
    }
}

/// Stage 1: Extract 6 face buffers (one per direction).
fn extract_face_buffers(
    storage: &ChunkStorage,
    neighbors: &[Option<Arc<ChunkStorage>>; 6],
) -> [Vec<BlockId>; NUM_DIRECTIONS] {
    let mut buffers: [Vec<BlockId>; NUM_DIRECTIONS] =
        std::array::from_fn(|_| vec![AIR; CHUNK_SIZE_3]);

    for z in 0..CHUNK_SIZE {
        for y in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                let block = storage.get(x, y, z);
                if block == AIR {
                    continue;
                }
                let idx = x + y * CHUNK_SIZE + z * CHUNK_SIZE_2;

                // +X
                let nb = if x < CHUNK_SIZE - 1 {
                    storage.get(x + 1, y, z)
                } else {
                    get_neighbor_block(&neighbors[DIR_POS_X], 0, y, z)
                };
                if nb == AIR {
                    buffers[DIR_POS_X][idx] = block;
                }

                // -X
                let nb = if x > 0 {
                    storage.get(x - 1, y, z)
                } else {
                    get_neighbor_block(&neighbors[DIR_NEG_X], CHUNK_SIZE - 1, y, z)
                };
                if nb == AIR {
                    buffers[DIR_NEG_X][idx] = block;
                }

                // +Y
                let nb = if y < CHUNK_SIZE - 1 {
                    storage.get(x, y + 1, z)
                } else {
                    get_neighbor_block(&neighbors[DIR_POS_Y], x, 0, z)
                };
                if nb == AIR {
                    buffers[DIR_POS_Y][idx] = block;
                }

                // -Y
                let nb = if y > 0 {
                    storage.get(x, y - 1, z)
                } else {
                    get_neighbor_block(&neighbors[DIR_NEG_Y], x, CHUNK_SIZE - 1, z)
                };
                if nb == AIR {
                    buffers[DIR_NEG_Y][idx] = block;
                }

                // +Z
                let nb = if z < CHUNK_SIZE - 1 {
                    storage.get(x, y, z + 1)
                } else {
                    get_neighbor_block(&neighbors[DIR_POS_Z], x, y, 0)
                };
                if nb == AIR {
                    buffers[DIR_POS_Z][idx] = block;
                }

                // -Z
                let nb = if z > 0 {
                    storage.get(x, y, z - 1)
                } else {
                    get_neighbor_block(&neighbors[DIR_NEG_Z], x, y, CHUNK_SIZE - 1)
                };
                if nb == AIR {
                    buffers[DIR_NEG_Z][idx] = block;
                }
            }
        }
    }

    buffers
}

/// Stage 2: Greedy meshing.
fn greedy_mesh_buffers(buffers: &[Vec<BlockId>; NUM_DIRECTIONS]) -> [Vec<FaceData>; NUM_DIRECTIONS] {
    const CS: usize = CHUNK_SIZE;
    const CS2: usize = CHUNK_SIZE_2;

    const D_STRIDES: [usize; 6] = [1, 1, CS, CS, CS2, CS2];
    const U_STRIDES: [usize; 6] = [CS, CS2, CS2, 1, 1, CS];
    const V_STRIDES: [usize; 6] = [CS2, CS, 1, CS2, CS, 1];

    std::array::from_fn(|dir| {
        let buffer = &buffers[dir];
        let ds = D_STRIDES[dir];
        let us = U_STRIDES[dir];
        let vs = V_STRIDES[dir];

        let mut faces = Vec::new();
        let mut consumed = [false; CHUNK_SIZE_2];

        for d in 0..CHUNK_SIZE {
            consumed.fill(false);

            for u in 0..CHUNK_SIZE {
                for v in 0..CHUNK_SIZE {
                    let grid_idx = u * CHUNK_SIZE + v;
                    if consumed[grid_idx] {
                        continue;
                    }

                    let block = buffer[d * ds + u * us + v * vs];
                    if block == AIR {
                        continue;
                    }

                    let mut hv = 1usize;
                    while v + hv < CHUNK_SIZE {
                        let ni = u * CHUNK_SIZE + (v + hv);
                        if consumed[ni] || buffer[d * ds + u * us + (v + hv) * vs] != block {
                            break;
                        }
                        hv += 1;
                    }

                    let mut hu = 1usize;
                    'expand_u: while u + hu < CHUNK_SIZE {
                        for dv in 0..hv {
                            let ni = (u + hu) * CHUNK_SIZE + (v + dv);
                            if consumed[ni]
                                || buffer[d * ds + (u + hu) * us + (v + dv) * vs] != block
                            {
                                break 'expand_u;
                            }
                        }
                        hu += 1;
                    }

                    for du in 0..hu {
                        for dv in 0..hv {
                            consumed[(u + du) * CHUNK_SIZE + (v + dv)] = true;
                        }
                    }

                    let (x, y, z) = match dir {
                        DIR_POS_X => (d, u, v),
                        DIR_NEG_X => (d, v, u),
                        DIR_POS_Y => (v, d, u),
                        DIR_NEG_Y => (u, d, v),
                        DIR_POS_Z => (u, v, d),
                        DIR_NEG_Z => (v, u, d),
                        _ => unreachable!(),
                    };

                    faces.push(FaceData {
                        x: x as u8,
                        y: y as u8,
                        z: z as u8,
                        w: hu as u8,
                        h: hv as u8,
                        material: [0; 3],
                    });
                }
            }
        }

        faces
    })
}
