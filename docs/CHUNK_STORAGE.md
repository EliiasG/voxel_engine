# Chunk Storage & World Architecture

Design for how voxels are stored and managed before rendering.

## Voxel Storage (Palette Compression)

`BlockId = u32` globally (~4 billion block types). Chunk size is 32^3.

```rust
enum ChunkStorage {
    Filled(BlockId),                // uniform (e.g. Filled(0) for all air)
    Paletted {
        palette: SmallVec<[BlockId; 8]>,
        indices: BitVec,            // packed indices into palette
    },
}
```

Palette width is variable based on number of types in the chunk:
- 2 types -> 1 bit per voxel (4 KB)
- 4 types -> 2 bits (8 KB)
- 16 types -> 4 bits (16 KB)
- 256 types -> 8 bits (32 KB, same as flat array)

Palette widens when new types are added. Can compact when types are removed.

`ChunkData` wraps storage in an `Arc` for cheap sharing with async tasks:

```rust
#[derive(Component, Clone)]
struct ChunkData(Arc<ChunkStorage>);
```

Chunk data is immutable once created. Modifications replace the entire Arc. In-flight async tasks (meshing, etc.) hold their own Arc reference and see consistent data -- no locks, no races. Stale results are simply superseded by the next remesh.

## World Structure (Hybrid ECS + HashMap)

`HashMap<IVec3, Entity>` as a Bevy `Resource` for spatial lookups. Chunk data lives on ECS entities as components.

**Table-stored components** (stable, rarely added/removed):
- `ChunkPos(IVec3)`
- `ChunkData(Arc<ChunkStorage>)` -- added once after generation, replaced on modification
- Mesh-related components (page allocations, etc.)

**SparseSet components** (frequently toggled, no archetype churn):
- `NeedsGeneration` -- marker for chunks awaiting terrain generation
- `NeedsRemesh` -- private to `chunk::meshing`, marker for chunks awaiting meshing
- `ChunkFaces([Vec<Face>; 6])` -- transient, added by meshing, consumed and removed by render

## Chunk Lifecycle

1. **Loader** -- creates entity with `ChunkPos` + `NeedsGeneration`, inserts into HashMap
2. **Generator** -- queries `With<NeedsGeneration>`, generates terrain, adds `ChunkData`, removes `NeedsGeneration`, fires `ChunkChanged` event
3. **Change resolver** (in `chunk::meshing`) -- consumes `ChunkChanged` events, determines affected chunks (self + neighbors based on bounds), adds `NeedsRemesh` to those entities
4. **Mesher** (in `chunk::meshing`) -- queries `With<NeedsRemesh>`, clones Arc references for the chunk + 6 neighbors, spawns async task, removes `NeedsRemesh`. On completion, adds `ChunkFaces` component to entity.
5. **Render** -- queries entities with `ChunkFaces`, uploads faces to GPU pages, adds render data to entity, removes `ChunkFaces`
6. **Unloader** -- removes entity from HashMap, despawns it

## Two-Stage Meshing Pipeline

Meshing is split into face extraction and face optimization, decoupled by an intermediate buffer format.

### Stage 1: Face Extraction

Iterates all voxels in the chunk. For each exposed surface, emits a 1x1 face into a `[BlockId; 32*32*32]` buffer for that direction (6 buffers total). A face exists when the voxel is solid and its neighbor in that direction is air. Neighbor chunks are consulted at chunk boundaries.

The value in each slot is the BlockId of the face (0 = no face).

This is a pure function of `ChunkData` + the 6 neighbor `ChunkData` (via Arc clones).

### Stage 2: Greedy Meshing

Takes the 6 face buffers from stage 1. For each direction, sweeps 32 slices of 32x32 grids, merging adjacent same-type faces into larger quads. Outputs `Vec<Face>` per direction (where Face is the 8-byte packed format from PAGED_RENDERING.md).

### Intermediate Buffer Format

```
[BlockId; CHUNK_SIZE^3] per direction, 6 total
```

- Fixed size, predictable, stack-friendly (6 * 32^3 * 4 bytes = 768 KB with u32 BlockId)
- Index maps directly to voxel position: buffer[x + y*32 + z*1024]
- Value = BlockId of the face at that position, or 0 for no face
- Stage 2 can be swapped (naive 1x1 output, greedy, future algorithms) without touching extraction

### Output

Stage 2 produces `[Vec<Face>; 6]` -- one vec per direction, added to the entity as a `ChunkFaces` component. This component is transient: the render module queries it, uploads faces to GPU pages per PAGED_RENDERING.md, then removes the component. No CPU-side face data is retained long-term.

## Change Propagation

```rust
struct ChunkChanged {
    pos: IVec3,
    bounds: Option<UVec3Range>,
}
```

Fired as a Bevy event by any system that modifies voxel data. One dedicated system consumes these, resolves which chunks need remeshing (the chunk itself + affected neighbors if the edit touched a chunk boundary), and adds `NeedsRemesh` markers. This centralizes invalidation logic.

Other future systems (lighting, pathfinding) can also subscribe to `ChunkChanged` independently.

## LOD

Not designed in detail yet. Planned approach:
- An array of N `HashMap<IVec3, Entity>` (one per LOD level)
- A `ChunkLod(u8)` component on chunk entities
- Each LOD level has 2^N scale factor

To be designed once base rendering works.

## Async

Generation and meshing run on async task pools. Async tasks receive Arc-cloned chunk data, avoiding locks. Results are polled each frame.

## Module Structure

- **`chunk`** -- `ChunkStorage`, `ChunkData`, `BlockId`, `ChunkMap`, `ChunkChanged` event
- **`chunk::generation`** -- terrain generation, produces `ChunkData`
- **`chunk::loading`** -- entity lifecycle, HashMap management (spawn/despawn, NeedsGeneration)
- **`chunk::meshing`** -- change resolver, `NeedsRemesh` (private), face extraction, greedy meshing, adds `ChunkFaces` component
- **`render`** -- queries `ChunkFaces`, uploads to GPU pages, adds render data to entity, removes `ChunkFaces`
- **`camera`** -- camera controller

Meshing and render are decoupled via the `ChunkFaces` component. A server build can omit both `chunk::meshing` and `render`.

## Deferred

- Block registry (BlockId -> properties like solid/transparent/texture)
- Serialization/persistence
- LOD detail design
