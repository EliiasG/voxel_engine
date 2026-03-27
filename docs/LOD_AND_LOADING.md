# LOD & Dynamic Loading

## Overview

Chunks are loaded progressively around the camera across multiple LOD levels. LOD 0 is full detail, LOD N has 2^N scale (each voxel represents 2^N world blocks). All LOD levels share the same 32^3 chunk format, paged rendering system, and meshing pipeline.

## LOD Levels

| LOD | Scale | Chunk world extent | Voxel size |
|-----|-------|--------------------|------------|
| 0   | 1x    | 32 blocks          | 1 block    |
| 1   | 2x    | 64 blocks          | 2 blocks   |
| 2   | 4x    | 128 blocks         | 4 blocks   |
| N   | 2^N   | 32 * 2^N blocks    | 2^N blocks |

Max LOD count ~8-12 in practice. LOD 8 = 256x scale, each chunk covers 8192 blocks.

## Data Structures

### LodChunkMaps (Resource)

```rust
struct LodChunkMaps {
    maps: Vec<ChunkMap>,  // one per LOD level
}
```

Spatial lookup per LOD level. `ChunkMap` is `HashMap<IVec3, Entity>` as before. LOD 0 map is the existing `ChunkMap`.

### ChunkLod (Component)

```rust
#[derive(Component)]
struct ChunkLod(pub u8);
```

On every chunk entity. Used by generation (scale factor), meshing (neighbor lookup from correct map), and rendering (shader scaling).

### LoadOrigin (Resource)

```rust
struct LoadOrigin {
    chunk_pos: IVec3,  // camera's LOD 0 chunk coordinate
}
```

Updated each frame from camera position.

## Loader

### Configuration

```rust
struct LoadConfig {
    start_radius: u32,  // e.g., 4 -- initial load radius (in chunks at each LOD)
    step: u32,          // e.g., 2 -- radius increment per expansion round
    end_radius: u32,    // e.g., 16 -- max radius per LOD
    lod_count: u32,     // e.g., 8
}
```

### Job Queue

The loader maintains a priority queue (VecDeque) of `LoadJob`s, ordered by priority. Each job represents "ensure all chunks within this radius at this LOD are loaded."

```rust
struct LoadJob {
    lod: u32,
    radius: u32,
}
```

On startup (or camera teleport), the queue is populated:

```
(lod=0, radius=4), (lod=1, radius=4), ..., (lod=N, radius=4),
(lod=0, radius=6), (lod=1, radius=6), ..., (lod=N, radius=6),
...
(lod=0, radius=16), (lod=1, radius=16), ..., (lod=N, radius=16)
```

### Job Execution

The loader processes one job at a time (front of queue). For the current job:

1. Compute the set of chunk positions within the radius (sphere in LOD-N chunk space, centered on camera's LOD-N chunk coordinate)
2. For each position not already in the LOD's ChunkMap: spawn entity with `ChunkPos`, `ChunkLod`, `NeedsGeneration`
3. Wait until all spawned chunks in this job have `ChunkData` (generation complete)
4. Pop the job, advance to next

"Wait" means: each frame, check if all entities spawned for this job have `ChunkData`. If not, do nothing. The generation system handles the actual async work independently.

### Camera Movement

When camera enters a new LOD 0 chunk:

1. Update `LoadOrigin`
2. **Unload**: for each LOD level, remove chunks outside `end_radius` from origin. Despawn entities, deallocate pages.
3. **Requeue**: rebuild the job queue from the new origin. Jobs for already-loaded radii are no-ops (their chunks already exist).

In-flight generation/meshing tasks for unloaded chunks produce stale results -- the poll systems should check entity validity before applying.

### Coordinate Mapping

Camera is at world position `P` (f64). Its LOD 0 chunk is `floor(P / 32)`.

For LOD N, the camera's chunk coordinate is `floor(P / (32 * 2^N))`.

A LOD N chunk at position `(cx, cy, cz)` covers world blocks `[cx * 32 * 2^N .. (cx+1) * 32 * 2^N)` per axis.

## Generation

`generate_terrain(chunk_pos: IVec3, lod: u8) -> ChunkStorage`

Samples terrain at LOD resolution. World coordinate for voxel (x, y, z) in a LOD N chunk:
```
wx = chunk_pos.x * CHUNK_SIZE * 2^lod + x * 2^lod
wy = chunk_pos.y * CHUNK_SIZE * 2^lod + y * 2^lod
wz = chunk_pos.z * CHUNK_SIZE * 2^lod + z * 2^lod
```

The height function is evaluated at `(wx, wz)`. Block is solid if `wy <= height`.

## Meshing

Same two-stage pipeline (extraction + greedy). Neighbors come from the same LOD level's ChunkMap. The output `FaceData` is identical across LOD levels -- the shader handles scaling.

For neighbor boundary checks: look up adjacent chunk in `LodChunkMaps.maps[lod]`. If not loaded, treat boundary as air (same as current behavior).

## Rendering

### PageMetadata

16 bytes, unchanged size:

```rust
struct PageMetadata {
    chunk_x: i32,
    chunk_y: i32,
    chunk_z: i32,
    direction_and_lod: u32,  // bits 0-7: direction (0-5), bits 8-15: lod level
}
```

Written once at upload. Integer chunk positions never go stale. LOD is static per page.

### Shader

```wgsl
let direction = meta.direction_and_lod & 0xFFu;
let lod = (meta.direction_and_lod >> 8u) & 0xFFu;
let lod_scale = 1i << lod;

// Camera-relative origin (integer math, exact)
let rel_chunk = chunk_pos * lod_scale - camera.chunk_offset;
let chunk_origin = vec3<f32>(rel_chunk * CHUNK_SIZE);

// Local position scaled by LOD
let local_pos = (pos + offset + tangent_u * u_factor * w + tangent_v * v_factor * h) * f32(lod_scale);

let rel_pos = chunk_origin + local_pos;
```

All integer until final cast. Camera-relative rendering works unchanged.

### Indirect Buffer & Draw Order

When rebuilding the indirect buffer each frame, group draw args by LOD level:

```
args_per_lod: [Vec<DrawIndirectArgs>; lod_count]

for each page:
    lod = page.lod
    if lod > 0 and all children at lod-1 are loaded:
        skip (fully covered by finer LOD)
    else:
        args_per_lod[lod].push(args)
```

Write all vecs to the indirect buffer sequentially. Record offset + count per LOD.

Draw LOD 0 first, then LOD 1, etc:

```
for lod in 0..lod_count:
    if count[lod] > 0:
        multi_draw_indirect(buffer, offset[lod], count[lod])
```

LOD 0 writes depth first, so higher LOD fragments behind LOD 0 geometry are depth-rejected. Reduces overdraw.

### LOD Visibility

A LOD N page is skipped (not drawn) when ALL 2x2x2 child chunks at LOD N-1 covering the same region are loaded and have pages. This prevents full overdraw.

Partial coverage is allowed: if only some LOD 0 children exist, both the LOD 0 pages AND the LOD 1 page are drawn. Depth testing handles the overlap correctly. As more LOD 0 chunks load in, the LOD 1 page eventually gets fully occluded.

When LOD 0 chunks unload (camera moved away), the LOD 1 page automatically becomes visible on the next indirect rebuild.

This lookup requires a spatial index of "which (pos, lod) chunks have pages." A `HashSet<(IVec3, u32)>` maintained alongside `ChunkRenderData` is sufficient.

## Unloading

When a chunk is unloaded:
1. Remove from `LodChunkMaps`
2. Deallocate pages via `PageAllocator` (already supported)
3. Remove from `ChunkRenderData`
4. Despawn entity

Stale face data in the GPU buffer is harmless -- no draw args reference it after page deallocation.

## System Ordering

```
Redraw (before RenderSystemSet):
    loader::update_origin        -- detect camera chunk change, unload, requeue
    loader::process_jobs         -- spawn NeedsGeneration entities for current job
    apply_deferred
    generation::poll_generation  -- apply completed gen results
    generation::start_generation -- spawn async gen tasks
    apply_deferred
    meshing::resolve_changes     -- changed queue -> NeedsRemesh
    apply_deferred
    meshing::poll_meshing        -- apply completed mesh results
    meshing::start_meshing       -- spawn async mesh tasks

Synchronize:
    render::synchronize_gpu      -- upload faces, rebuild indirect buffer per LOD
    render::update_camera        -- upload camera uniform
```

## Not In Scope (Yet)

- Frustum / backface culling (orthogonal, adds to indirect rebuild)
- LOD bias based on distance (all LODs use same chunk size, scale handles it)
- Async chunk unloading
- Cross-LOD meshing seams (accept minor visual discontinuities at LOD boundaries for now)
