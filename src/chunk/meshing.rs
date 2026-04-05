use std::sync::Arc;

use bevy_ecs::prelude::*;
use crossbeam_channel::{Receiver, Sender};

use super::*;

#[derive(Component)]
#[component(storage = "SparseSet")]
pub struct NeedsRemesh;

/// Per-direction face data: standard faces (always drawn) + border faces
/// (only drawn when same-LOD neighbor in this direction is hidden by finer LOD).
pub struct DirFaces {
    pub standard: Vec<FaceData>,
    pub border: Vec<FaceData>,
}

#[derive(Component)]
#[component(storage = "SparseSet")]
pub struct ChunkFaces(pub [DirFaces; NUM_DIRECTIONS]);

/// Transparent faces, separate from opaque. Same standard/border split per direction.
#[derive(Component)]
#[component(storage = "SparseSet")]
pub struct TransparentChunkFaces(pub [DirFaces; NUM_DIRECTIONS]);

struct MeshRequest {
    entity: Entity,
    storage: Arc<ChunkStorage>,
    neighbors: [Option<Arc<ChunkStorage>>; 6],
}

struct MeshResult {
    entity: Entity,
    opaque_faces: [DirFaces; NUM_DIRECTIONS],
    transparent_faces: [DirFaces; NUM_DIRECTIONS],
    mesh_us: u32,
}

/// Channel-based worker pool for chunk meshing.
#[derive(Resource)]
pub struct MeshPool {
    tx: Option<Sender<MeshRequest>>,
    rx: Receiver<MeshResult>,
    workers: Vec<std::thread::JoinHandle<()>>,
}

impl MeshPool {
    pub fn new() -> Self {
        let (req_tx, req_rx) = crossbeam_channel::unbounded::<MeshRequest>();
        let (res_tx, res_rx) = crossbeam_channel::unbounded::<MeshResult>();

        let num_threads = std::thread::available_parallelism()
            .map(|n| (n.get() / 2).max(1))
            .unwrap_or(2);

        let mut workers = Vec::with_capacity(num_threads);
        for i in 0..num_threads {
            let req_rx = req_rx.clone();
            let res_tx = res_tx.clone();
            let handle = std::thread::Builder::new()
                .name(format!("mesh-worker-{i}"))
                .spawn(move || {
                    while let Ok(req) = req_rx.recv() {
                        let t0 = std::time::Instant::now();
                        let (mut opaque, mut transparent) = mesh_chunk(&req.storage, &req.neighbors);
                        // Drop border faces for directions with no standard faces.
                        // If there's no visible surface in a direction, border faces
                        // are pure overhead (the chunk is buried on that side).
                        for dir_faces in &mut opaque {
                            if dir_faces.standard.is_empty() {
                                dir_faces.border.clear();
                            }
                        }
                        for dir_faces in &mut transparent {
                            if dir_faces.standard.is_empty() {
                                dir_faces.border.clear();
                            }
                        }
                        let mesh_us = t0.elapsed().as_micros() as u32;
                        let _ = res_tx.send(MeshResult {
                            entity: req.entity,
                            opaque_faces: opaque,
                            transparent_faces: transparent,
                            mesh_us,
                        });
                    }
                })
                .expect("failed to spawn mesh worker");
            workers.push(handle);
        }

        println!("Mesh pool: {num_threads} threads");
        Self { tx: Some(req_tx), rx: res_rx, workers }
    }
}

impl Drop for MeshPool {
    fn drop(&mut self) {
        self.tx.take();
        for handle in self.workers.drain(..) {
            let _ = handle.join();
        }
    }
}

/// Reads ChunkChangedQueue, marks affected chunks + neighbors with NeedsRemesh.
/// Does not drain — queue is cleared separately at end of frame.
pub fn resolve_changes(
    mut commands: Commands,
    changed: Res<ChunkChangedQueue>,
    lod_maps: Res<LodChunkMaps>,
    chunk_data_query: Query<(), With<ChunkData>>,
) {
    let _timer = crate::SysTimer::new(&crate::TIMING_RESOLVE_US);
    for change in &changed.0 {
        let map = &lod_maps.maps[change.lod as usize];
        if let Some(&entity) = map.get(&change.pos) {
            commands.entity(entity).insert(NeedsRemesh);
        }
        for offset in &DIR_OFFSETS {
            let neighbor_pos = change.pos + *offset;
            if let Some(&entity) = map.get(&neighbor_pos) {
                if chunk_data_query.get(entity).is_ok() {
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
    let _timer = crate::SysTimer::new(&crate::TIMING_START_MESH_US);
    let empty_dir_faces = || std::array::from_fn(|_| DirFaces {
        standard: Vec::new(),
        border: Vec::new(),
    });

    for (entity, pos, lod, data) in query.iter() {
        // Skip meshing for all-air chunks -- no faces possible
        if let ChunkStorage::Filled(AIR) = &*data.0 {
            commands.entity(entity)
                .insert((ChunkFaces(empty_dir_faces()), TransparentChunkFaces(empty_dir_faces())))
                .remove::<NeedsRemesh>();
            continue;
        }

        let map = &lod_maps.maps[lod.0 as usize];
        let neighbors: [Option<Arc<ChunkStorage>>; 6] = std::array::from_fn(|dir| {
            let neighbor_pos = pos.0 + DIR_OFFSETS[dir];
            map.get(&neighbor_pos)
                .and_then(|&e| chunk_data_query.get(e).ok())
                .map(|cd| cd.0.clone())
        });

        let Some(tx) = &pool.tx else { continue };
        let _ = tx.send(MeshRequest {
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
    let _timer = crate::SysTimer::new(&crate::TIMING_POLL_MESH_US);
    let mut max_worker_us = 0u32;
    let mut count = 0u32;
    while let Ok(result) = pool.rx.try_recv() {
        max_worker_us = max_worker_us.max(result.mesh_us);
        count += 1;
        if entity_check.get(result.entity).is_ok() {
            commands.entity(result.entity).insert((
                ChunkFaces(result.opaque_faces),
                TransparentChunkFaces(result.transparent_faces),
            ));
        }
    }
    crate::TIMING_MESH_WORKER_MAX_US.store(max_worker_us, std::sync::atomic::Ordering::Relaxed);
    crate::TIMING_MESH_WORKER_COUNT.store(count, std::sync::atomic::Ordering::Relaxed);
}

// --- Binary Greedy Meshing internals ---
// Face culling via bitmask column operations, greedy merge via bit manipulation.
// Y-axis columns (u64, bits 1-32 = voxels y=0..31, bits 0/33 = neighbor padding).

const CS: usize = CHUNK_SIZE;      // 32
const CS_P: usize = CS + 2;        // 34 (padded x/z grid)
const CS_P2: usize = CS_P * CS_P;  // 1156
const VALID: u64 = (1u64 << CS) - 1; // lower 32 bits

/// Map (layer, forward, bit) to block (x, y, z) per direction.
/// For X/Z-normal faces: columns along Y → bit=y, forward=z or x.
/// For Y-normal faces: bit=z, forward=x.
#[inline]
fn face_block_pos(dir: usize, layer: usize, fwd: usize, bit: usize) -> (usize, usize, usize) {
    match dir {
        DIR_POS_X | DIR_NEG_X => (layer, bit, fwd),
        DIR_POS_Y | DIR_NEG_Y => (fwd, layer, bit),
        DIR_POS_Z | DIR_NEG_Z => (fwd, bit, layer),
        _ => unreachable!(),
    }
}

/// Map (bit_width, fwd_height) to FaceData (w, h) per direction.
/// w = extent along u axis, h = extent along v axis (as defined by the shader).
#[inline]
fn face_wh(dir: usize, bit_w: u32, fwd_h: u32) -> (u32, u32) {
    match dir {
        DIR_POS_X | DIR_POS_Y | DIR_NEG_Z => (bit_w, fwd_h),
        DIR_NEG_X | DIR_NEG_Y | DIR_POS_Z => (fwd_h, bit_w),
        _ => unreachable!(),
    }
}

// --- AO ---

fn is_solid_at(
    x: i32, y: i32, z: i32,
    storage: &ChunkStorage,
    neighbors: &[Option<Arc<ChunkStorage>>; 6],
) -> bool {
    let cs = CHUNK_SIZE as i32;
    if x >= 0 && x < cs && y >= 0 && y < cs && z >= 0 && z < cs {
        return is_opaque(storage.get(x as usize, y as usize, z as usize));
    }
    let (nx, dir_x) = if x < 0 { (x + cs, Some(DIR_NEG_X)) } else if x >= cs { (x - cs, Some(DIR_POS_X)) } else { (x, None) };
    let (ny, dir_y) = if y < 0 { (y + cs, Some(DIR_NEG_Y)) } else if y >= cs { (y - cs, Some(DIR_POS_Y)) } else { (y, None) };
    let (nz, dir_z) = if z < 0 { (z + cs, Some(DIR_NEG_Z)) } else if z >= cs { (z - cs, Some(DIR_POS_Z)) } else { (z, None) };
    let active_dir = match (dir_x, dir_y, dir_z) {
        (Some(d), None, None) | (None, Some(d), None) | (None, None, Some(d)) => d,
        _ => return false,
    };
    match &neighbors[active_dir] {
        Some(n) => is_opaque(n.get(nx as usize, ny as usize, nz as usize)),
        None => false,
    }
}

const AO_NORMAL: [[i32; 3]; 6] = [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]];
const AO_TAN_U: [[i32; 3]; 6] = [[0,1,0],[0,0,1],[0,0,1],[1,0,0],[1,0,0],[0,1,0]];
const AO_TAN_V: [[i32; 3]; 6] = [[0,0,1],[0,1,0],[1,0,0],[0,0,1],[0,1,0],[1,0,0]];

fn compute_ao(
    dir: usize, x: usize, y: usize, z: usize,
    storage: &ChunkStorage, neighbors: &[Option<Arc<ChunkStorage>>; 6],
) -> u8 {
    let n = AO_NORMAL[dir]; let tu = AO_TAN_U[dir]; let tv = AO_TAN_V[dir];
    let ax = x as i32 + n[0]; let ay = y as i32 + n[1]; let az = z as i32 + n[2];
    let mut ao_byte: u8 = 0;
    for corner in 0..4u8 {
        let us = if corner & 1 == 0 { -1i32 } else { 1 };
        let vs = if corner & 2 == 0 { -1i32 } else { 1 };
        let side_u = is_solid_at(ax + us*tu[0], ay + us*tu[1], az + us*tu[2], storage, neighbors);
        let side_v = is_solid_at(ax + vs*tv[0], ay + vs*tv[1], az + vs*tv[2], storage, neighbors);
        let ao_val = if side_u && side_v { 0u8 } else {
            let diag = is_solid_at(ax + us*tu[0] + vs*tv[0], ay + us*tu[1] + vs*tv[1], az + us*tu[2] + vs*tv[2], storage, neighbors);
            3 - (side_u as u8 + side_v as u8 + diag as u8)
        };
        ao_byte |= ao_val << (corner * 2);
    }
    ao_byte
}

// --- Column building ---

/// Fill a padding column from a neighbor chunk boundary.
/// Sets bits in opaque_col and trans_cols for the given column index.
fn fill_padding_col(
    opaque_cols: &mut [u64; CS_P2],
    trans_cols: &mut [Vec<u64>],
    trans_types: &[BlockId],
    ci: usize,
    neighbor: &Option<Arc<ChunkStorage>>,
    get_block: impl Fn(usize) -> BlockId,
) {
    match neighbor {
        Some(_) => {
            for y in 0..CS {
                let block = get_block(y);
                if block == AIR { continue; }
                let bit = 1u64 << (y + 1);
                if is_opaque(block) {
                    opaque_cols[ci] |= bit;
                } else if let Some(ti) = trans_types.iter().position(|&t| t == block) {
                    trans_cols[ti][ci] |= bit;
                }
            }
        }
        None => {
            // Opaque: leave as 0 (air) so opaque faces stay standard for LOD transitions.
            // Transparent: fill all types so transparent faces are suppressed (no overlap).
            for tc in trans_cols.iter_mut() {
                tc[ci] = VALID << 1;
            }
        }
    }
}

/// Build Y-axis columns for opaque blocks and per-transparent-type.
fn build_columns(
    storage: &ChunkStorage,
    neighbors: &[Option<Arc<ChunkStorage>>; 6],
    trans_types: &[BlockId],
) -> ([u64; CS_P2], Vec<Vec<u64>>) {
    let mut opaque_cols = [0u64; CS_P2];
    let mut trans_cols: Vec<Vec<u64>> = (0..trans_types.len()).map(|_| vec![0u64; CS_P2]).collect();

    // Interior columns
    for z in 0..CS { for x in 0..CS {
        let ci = (x + 1) * CS_P + (z + 1);
        for y in 0..CS {
            let block = storage.get(x, y, z);
            if block == AIR { continue; }
            let bit = 1u64 << (y + 1);
            if is_opaque(block) {
                opaque_cols[ci] |= bit;
            } else if let Some(ti) = trans_types.iter().position(|&t| t == block) {
                trans_cols[ti][ci] |= bit;
            }
        }
    }}

    // X padding
    for z in 0..CS {
        let zp = z + 1;
        // x=-1 (xp=0) from DIR_NEG_X at x=CS-1
        fill_padding_col(&mut opaque_cols, &mut trans_cols, trans_types,
            0 * CS_P + zp, &neighbors[DIR_NEG_X], |y| neighbors[DIR_NEG_X].as_ref().map_or(STONE, |n| n.get(CS - 1, y, z)));
        // x=CS (xp=CS+1) from DIR_POS_X at x=0
        fill_padding_col(&mut opaque_cols, &mut trans_cols, trans_types,
            (CS + 1) * CS_P + zp, &neighbors[DIR_POS_X], |y| neighbors[DIR_POS_X].as_ref().map_or(STONE, |n| n.get(0, y, z)));
    }

    // Z padding
    for x in 0..CS {
        let xp = x + 1;
        // z=-1 (zp=0) from DIR_NEG_Z at z=CS-1
        fill_padding_col(&mut opaque_cols, &mut trans_cols, trans_types,
            xp * CS_P + 0, &neighbors[DIR_NEG_Z], |y| neighbors[DIR_NEG_Z].as_ref().map_or(STONE, |n| n.get(x, y, CS - 1)));
        // z=CS (zp=CS+1) from DIR_POS_Z at z=0
        fill_padding_col(&mut opaque_cols, &mut trans_cols, trans_types,
            xp * CS_P + (CS + 1), &neighbors[DIR_POS_Z], |y| neighbors[DIR_POS_Z].as_ref().map_or(STONE, |n| n.get(x, y, 0)));
    }

    // Y padding (bits 0 and 33 of each interior column)
    for z in 0..CS { for x in 0..CS {
        let ci = (x + 1) * CS_P + (z + 1);
        // bit 0: DIR_NEG_Y neighbor at y=CS-1
        match &neighbors[DIR_NEG_Y] {
            Some(n) => {
                let block = n.get(x, CS - 1, z);
                if block != AIR {
                    if is_opaque(block) {
                        opaque_cols[ci] |= 1;
                    } else if let Some(ti) = trans_types.iter().position(|&t| t == block) {
                        trans_cols[ti][ci] |= 1;
                    }
                }
            }
            None => {
                for tc in trans_cols.iter_mut() { tc[ci] |= 1; }
            }
        }
        // bit 33: DIR_POS_Y neighbor at y=0
        match &neighbors[DIR_POS_Y] {
            Some(n) => {
                let block = n.get(x, 0, z);
                if block != AIR {
                    let bit33 = 1u64 << (CS + 1);
                    if is_opaque(block) {
                        opaque_cols[ci] |= bit33;
                    } else if let Some(ti) = trans_types.iter().position(|&t| t == block) {
                        trans_cols[ti][ci] |= bit33;
                    }
                }
            }
            None => {
                let bit33 = 1u64 << (CS + 1);
                for tc in trans_cols.iter_mut() { tc[ci] |= bit33; }
            }
        }
    }}

    (opaque_cols, trans_cols)
}

// --- Face mask building ---

/// Build face bitmask rows for one layer of a lateral direction (+X/-X/+Z/-Z).
/// Returns u32 rows indexed by forward position, bits = y positions.
fn build_lateral_face_rows(
    my_cols: &[u64; CS_P2],
    nb_cols: &[u64; CS_P2],
    layer: usize,
    dir: usize,
    rows: &mut [u32; CS],
) {
    for fwd in 0..CS {
        let (xp, zp, nb_xp, nb_zp) = match dir {
            DIR_POS_X => (layer + 1, fwd + 1, layer + 2, fwd + 1),
            DIR_NEG_X => (layer + 1, fwd + 1, layer, fwd + 1),
            DIR_POS_Z => (fwd + 1, layer + 1, fwd + 1, layer + 2),
            DIR_NEG_Z => (fwd + 1, layer + 1, fwd + 1, layer),
            _ => unreachable!(),
        };
        let my = my_cols[xp * CS_P + zp] >> 1;
        let nb = nb_cols[nb_xp * CS_P + nb_zp] >> 1;
        rows[fwd] = (my & !nb & VALID) as u32;
    }
}

/// Build face bitmask rows for one layer of a Y-normal direction (+Y/-Y).
/// Returns u32 rows indexed by forward=x, bits = z positions.
fn build_y_face_rows(
    my_cols: &[u64; CS_P2],
    nb_cols: &[u64; CS_P2],
    layer: usize,
    dir: usize,
    rows: &mut [u32; CS],
) {
    let bit_y = (layer + 1) as u64;
    for x in 0..CS {
        let mut row = 0u32;
        for z in 0..CS {
            let ci = (x + 1) * CS_P + (z + 1);
            let my_set = my_cols[ci] & (1u64 << bit_y) != 0;
            let nb_bit = match dir {
                DIR_POS_Y => bit_y + 1, // y+1 neighbor
                DIR_NEG_Y => bit_y - 1, // y-1 neighbor
                _ => unreachable!(),
            };
            let nb_set = nb_cols[ci] & (1u64 << nb_bit) != 0;
            if my_set && !nb_set {
                row |= 1u32 << z;
            }
        }
        rows[x] = row;
    }
}

/// Build "my block" rows (all positions where this block type exists at this layer).
/// Used for computing border faces = my_block & ~standard.
fn build_my_block_rows(
    my_cols: &[u64],
    layer: usize,
    dir: usize,
    rows: &mut [u32; CS],
) {
    match dir {
        DIR_POS_X | DIR_NEG_X => {
            let xp = layer + 1;
            for fwd in 0..CS {
                rows[fwd] = ((my_cols[xp * CS_P + (fwd + 1)] >> 1) & VALID) as u32;
            }
        }
        DIR_POS_Z | DIR_NEG_Z => {
            let zp = layer + 1;
            for fwd in 0..CS {
                rows[fwd] = ((my_cols[(fwd + 1) * CS_P + zp] >> 1) & VALID) as u32;
            }
        }
        DIR_POS_Y | DIR_NEG_Y => {
            let bit_y = (layer + 1) as u64;
            for x in 0..CS {
                let mut row = 0u32;
                for z in 0..CS {
                    if my_cols[(x + 1) * CS_P + (z + 1)] & (1u64 << bit_y) != 0 {
                        row |= 1u32 << z;
                    }
                }
                rows[x] = row;
            }
        }
        _ => unreachable!(),
    }
}

// --- Greedy merge ---

/// Greedy merge one layer of face bitmask rows into FaceData quads.
fn greedy_merge_layer(
    dir: usize,
    layer: usize,
    face_rows: &[u32; CS],
    storage: &ChunkStorage,
    neighbors: &[Option<Arc<ChunkStorage>>; 6],
    known_type: Option<BlockId>,
    faces: &mut Vec<FaceData>,
) {
    // Pre-compute AO for all visible faces
    let mut ao_cache = [0u8; CS * CS];
    for fwd in 0..CS {
        let mut bits = face_rows[fwd];
        while bits != 0 {
            let bit = bits.trailing_zeros() as usize;
            let (x, y, z) = face_block_pos(dir, layer, fwd, bit);
            ao_cache[fwd * CS + bit] = compute_ao(dir, x, y, z, storage, neighbors);
            bits &= bits - 1;
        }
    }

    let mut forward_merged = [0u32; CS];

    for fwd in 0..CS {
        let mut row = face_rows[fwd];
        let next_row = if fwd + 1 < CS { face_rows[fwd + 1] } else { 0 };

        while row != 0 {
            let bit_pos = row.trailing_zeros() as usize;
            let (bx, by, bz) = face_block_pos(dir, layer, fwd, bit_pos);
            let block = known_type.unwrap_or_else(|| storage.get(bx, by, bz));
            let ao = ao_cache[fwd * CS + bit_pos];

            // Try forward merge
            if next_row & (1u32 << bit_pos) != 0 {
                let (nx, ny, nz) = face_block_pos(dir, layer, fwd + 1, bit_pos);
                let next_block = known_type.unwrap_or_else(|| storage.get(nx, ny, nz));
                let next_ao = ao_cache[(fwd + 1) * CS + bit_pos];
                if next_block == block && next_ao == ao {
                    forward_merged[bit_pos] += 1;
                    row &= !(1u32 << bit_pos);
                    continue;
                }
            }

            // Lateral merge along bit axis
            let fm = forward_merged[bit_pos];
            let mut width = 1u32;
            for b in (bit_pos + 1)..CS {
                if row & (1u32 << b) == 0 { break; }
                if forward_merged[b] != fm { break; }
                if ao_cache[fwd * CS + b] != ao { break; }
                if known_type.is_none() {
                    let (lx, ly, lz) = face_block_pos(dir, layer, fwd, b);
                    if storage.get(lx, ly, lz) != block { break; }
                }
                forward_merged[b] = 0;
                width += 1;
            }

            // Clear merged bits
            let mask = if width == 32 { !0u32 } else { ((1u32 << width) - 1) << bit_pos };
            row &= !mask;

            // Emit quad
            let fwd_start = fwd as u32 - fm;
            let fwd_height = fm + 1;
            let (fx, fy, fz) = face_block_pos(dir, layer, fwd_start as usize, bit_pos);
            let (w, h) = face_wh(dir, width, fwd_height);

            faces.push(FaceData {
                x: fx as u8, y: fy as u8, z: fz as u8,
                w: w as u8, h: h as u8,
                material: [ao, block as u8, 0],
            });

            forward_merged[bit_pos] = 0;
        }
    }
}

// --- Main entry ---

fn mesh_chunk(
    storage: &ChunkStorage,
    neighbors: &[Option<Arc<ChunkStorage>>; 6],
) -> ([DirFaces; NUM_DIRECTIONS], [DirFaces; NUM_DIRECTIONS]) {
    let mut opaque_out = std::array::from_fn(|_| DirFaces { standard: Vec::new(), border: Vec::new() });
    let mut trans_out = std::array::from_fn(|_| DirFaces { standard: Vec::new(), border: Vec::new() });

    // Find transparent types in this chunk
    let trans_types: Vec<BlockId> = match storage {
        ChunkStorage::Filled(b) => if is_transparent(*b) { vec![*b] } else { vec![] },
        ChunkStorage::Paletted { palette, .. } => palette.iter().copied().filter(|&b| is_transparent(b)).collect(),
    };

    // Build Y-axis columns
    let (opaque_cols, trans_cols) = build_columns(storage, neighbors, &trans_types);

    // Process each face direction
    for dir in 0..NUM_DIRECTIONS {
        let boundary_layer = match dir {
            DIR_POS_X | DIR_POS_Y | DIR_POS_Z => CS - 1,
            _ => 0,
        };

        for layer in 0..CS {
            let at_boundary = layer == boundary_layer;

            // --- Opaque faces ---
            let mut opaque_face_rows = [0u32; CS];
            match dir {
                DIR_POS_X | DIR_NEG_X | DIR_POS_Z | DIR_NEG_Z => {
                    build_lateral_face_rows(&opaque_cols, &opaque_cols, layer, dir, &mut opaque_face_rows);
                }
                DIR_POS_Y | DIR_NEG_Y => {
                    build_y_face_rows(&opaque_cols, &opaque_cols, layer, dir, &mut opaque_face_rows);
                }
                _ => unreachable!(),
            }

            if at_boundary {
                let mut my_rows = [0u32; CS];
                build_my_block_rows(&opaque_cols, layer, dir, &mut my_rows);
                let mut border_rows = [0u32; CS];
                for i in 0..CS {
                    border_rows[i] = my_rows[i] & !opaque_face_rows[i];
                }
                greedy_merge_layer(dir, layer, &opaque_face_rows, storage, neighbors, None, &mut opaque_out[dir].standard);
                greedy_merge_layer(dir, layer, &border_rows, storage, neighbors, None, &mut opaque_out[dir].border);
            } else {
                greedy_merge_layer(dir, layer, &opaque_face_rows, storage, neighbors, None, &mut opaque_out[dir].standard);
            }

            // --- Transparent faces (per type) ---
            for (ti, &block_type) in trans_types.iter().enumerate() {
                let mut trans_face_rows = [0u32; CS];
                // Transparent face: my type T, neighbor not opaque AND not same type T
                // Combined "hiding" columns = opaque + same-type-T
                // We build a combined column array on the fly per row
                match dir {
                    DIR_POS_X | DIR_NEG_X | DIR_POS_Z | DIR_NEG_Z => {
                        for fwd in 0..CS {
                            let (xp, zp, nb_xp, nb_zp) = match dir {
                                DIR_POS_X => (layer + 1, fwd + 1, layer + 2, fwd + 1),
                                DIR_NEG_X => (layer + 1, fwd + 1, layer, fwd + 1),
                                DIR_POS_Z => (fwd + 1, layer + 1, fwd + 1, layer + 2),
                                DIR_NEG_Z => (fwd + 1, layer + 1, fwd + 1, layer),
                                _ => unreachable!(),
                            };
                            let my = trans_cols[ti][xp * CS_P + zp] >> 1;
                            let nb_hide = (opaque_cols[nb_xp * CS_P + nb_zp] | trans_cols[ti][nb_xp * CS_P + nb_zp]) >> 1;
                            trans_face_rows[fwd] = (my & !nb_hide & VALID) as u32;
                        }
                    }
                    DIR_POS_Y | DIR_NEG_Y => {
                        let bit_y = (layer + 1) as u64;
                        for x in 0..CS {
                            let mut row = 0u32;
                            for z in 0..CS {
                                let ci = (x + 1) * CS_P + (z + 1);
                                let my_set = trans_cols[ti][ci] & (1u64 << bit_y) != 0;
                                let nb_bit = match dir {
                                    DIR_POS_Y => bit_y + 1,
                                    DIR_NEG_Y => bit_y - 1,
                                    _ => unreachable!(),
                                };
                                let nb_hide = (opaque_cols[ci] | trans_cols[ti][ci]) & (1u64 << nb_bit) != 0;
                                if my_set && !nb_hide {
                                    row |= 1u32 << z;
                                }
                            }
                            trans_face_rows[x] = row;
                        }
                    }
                    _ => unreachable!(),
                }

                if at_boundary {
                    let mut my_rows = [0u32; CS];
                    build_my_block_rows(&trans_cols[ti], layer, dir, &mut my_rows);
                    let mut border_rows = [0u32; CS];
                    for i in 0..CS {
                        border_rows[i] = my_rows[i] & !trans_face_rows[i];
                    }
                    greedy_merge_layer(dir, layer, &trans_face_rows, storage, neighbors, Some(block_type), &mut trans_out[dir].standard);
                    greedy_merge_layer(dir, layer, &border_rows, storage, neighbors, Some(block_type), &mut trans_out[dir].border);
                } else {
                    greedy_merge_layer(dir, layer, &trans_face_rows, storage, neighbors, Some(block_type), &mut trans_out[dir].standard);
                }
            }
        }
    }

    (opaque_out, trans_out)
}
