# Server / Client Architecture

Design notes for splitting the engine into a simulation "server" and a render "client".
This is a future refactor — not yet implemented.

## Motivation

After fixing the worst chunk-crossing hitches (shadow uploads, draw cache rebuild),
the remaining bottleneck on chunk crossings is `update_chunk_demand` (~14ms) and
`update_chunk_loading` (~22ms). These are CPU-bound orchestration costs:
HashSet diffing, entity spawn/despawn, generation queue management.

Optimizing these in-place would help, but they're fundamentally **bursty**: idle most
frames, then spike on every chunk crossing. The proper architectural fix is to move
all chunk-loading work off the render thread entirely.

This also opens the door to multiplayer, since the same code path that streams chunks
locally via `Arc` can stream them over the network via serialization.

## High-level structure

Two `World`s living in the same process (or different processes for multiplayer):

- **Server world** — owns canonical game state, runs at fixed tick rate (~20 TPS)
- **Client world** — owns presentation state, runs at frame rate (60-200+ FPS)

They communicate via channels (local) or network (multiplayer). Same event protocol
in both cases — local IPC is just `crossbeam_channel<Event>`, network is the same
event types with a serialization layer.

### What lives where

**Server world owns:**
- World generation (`GenPool`, noise/biome code)
- Canonical `ChunkData` (the source of truth for block contents)
- Chunk loader bookkeeping (`ChunkLoader`, server-side `LoadedChunkIndex`, server-side `LodChunkMaps`)
- Demand calculation (`update_chunk_demand`, `update_chunk_loading`)
- Player position (authoritative game state)
- Server-side simulation (water, fire, plants, redstone — none today, but the architecture supports it)
- Block change validation (anti-cheat for multiplayer; trivial for SP)

**Client world owns:**
- GPU buffers, draw caches (`ChunkRenderData`, `DrawCache`, `GpuBuffers`)
- Shadow grid + bitmask pool (`ShadowGrid`, `BitmaskPool`, `TransparentColorPool`)
- Mesh worker pool (`MeshPool`)
- Camera (presentation, can interpolate between server ticks)
- `Arc<ChunkData>` for chunks the client cares about (see "Memory" below)
- Collision detection (reads local `ChunkData`)
- Block raycasting / interaction (reads local `ChunkData`)
- Particle effects, animations, etc.

**Two parallel maps, not shared entities:**
- Server has its own `LodChunkMaps: HashMap<(IVec3, u8), Entity>`
- Client has its own `LodChunkMaps: HashMap<(IVec3, u8), Entity>`
- They are kept in sync via events
- The link between worlds is the `(chunk_pos, lod)` tuple, **never** the `Entity` ID
- Each world's entity IDs are private and unrelated

## Event protocol

All cross-world communication uses operations, not state. Sending whole chunks back
to the server breaks down the moment there's any concurrent writer (multiple players,
server-side simulation, even just async generation). Operations compose; state replacement
doesn't.

```rust
enum BlockOp {
    Place(Vec<(IVec3, BlockId)>),
    Break(Vec<IVec3>),
}

enum ClientToServer {
    /// Camera/player position update — used for demand calculation
    PlayerMoved { pos: DVec3 },

    /// Player wants to apply this op (block placement, breaking, etc.)
    Op { seq: u64, op: BlockOp },

    /// Client needs ChunkData for this chunk (was previously dropped)
    RequestChunkData { pos: IVec3, lod: u8 },
}

enum ServerToClient {
    /// New chunk available — full data
    ChunkLoaded { pos: IVec3, lod: u8, data: Arc<ChunkData> },

    /// Chunk is no longer in view — drop it
    ChunkUnloaded { pos: IVec3, lod: u8 },

    /// A block op happened (from this client, another client, or server simulation)
    /// For ops originated by this client, this is also the ack.
    OpApplied { seq: Option<u64>, op: BlockOp },

    /// Server rejected an op (anti-cheat, etc.) — client must roll back
    OpRejected { seq: u64 },
}
```

`seq` is a monotonic counter per client. Used to match ack/reject with pending ops.

For local single-player, `Arc::clone()` is essentially free. For network multiplayer,
the same enum gets serialized — chunks compressed with `flate2`, ops are tiny.

## Why operations, not state

Concurrent edit scenario:

```
T0: Players A and B both have Arc<ChunkData> with state S0
T1: A places stone at (5,5,5)  — A's local state = S0 + stone@(5,5,5)
T1: B places wood at (10,10,10) — B's local state = S0 + wood@(10,10,10)
T2: A sends "here's my chunk" → server stores S0+stone
T3: B sends "here's my chunk" → server stores S0+wood
                                ^^ A's stone is lost
```

A and B were editing **different positions**. Both edits are valid. But whole-chunk-replace
forces one to win.

With ops: server applies them in order, both edits land, last-writer-wins handles the
edge case where they touched the same position. This is the standard approach in every
multiplayer voxel game (Minecraft, Terraria, etc.).

Even in single-player, the moment the server runs *any* background simulation, the
race exists between server and client. Operation-based avoids it from day one.

## Multi-block ops and chunk dispatch

A `Place(Vec<(IVec3, BlockId)>)` can span multiple chunks. The server dispatches each
position to its containing chunk. The client must re-mesh all affected chunks.

```rust
fn affected_chunks(op: &BlockOp) -> HashSet<(IVec3, u8)> {
    // For each block position, compute chunk_pos = pos.div_euclid(CHUNK_SIZE)
    // Include neighbor chunks if the block is at position 0 or 31 along any axis
    // (face culling depends on the neighbor block)
}
```

Border-block edits affect the neighbor chunk's mesh too, so the affected set must include
neighbors when edits touch chunk edges.

## Big ops are server-side

Operations like explosions, structure generation, large fill commands are originated
by the server, not the client. The server computes the affected blocks and broadcasts
a single `OpApplied` to all clients. Client doesn't need to know it was an explosion —
it just receives "these blocks are gone".

## Client-side prediction

The 20 TPS server vs 100+ FPS client mismatch means the client cannot wait for the
server to confirm a block placement — that's up to 50ms perceived latency, which is
unacceptable. Even single-player needs prediction.

### Client flow

```rust
struct PendingOp {
    seq: u64,
    op: BlockOp,
    rollback: Vec<(IVec3, BlockId)>, // previous block state at each modified position
}

// On player click:
let op = build_op_from_click();
let rollback = read_current_state(&op);  // for potential reconciliation
apply_locally(&op);                       // mutate local Arc<ChunkData>
schedule_remesh(affected_chunks(&op));
let seq = next_seq(); next_seq += 1;
pending.push_back(PendingOp { seq, op: op.clone(), rollback });
send(ClientToServer::Op { seq, op });
```

### Server flow

Single-player: always accept, broadcast back as `OpApplied { seq: Some(seq), op }`.
Multiplayer: validate (anti-cheat, region permissions), then accept or reject.

### Client receives

- `OpApplied { seq: Some(seq), op }` — server confirmed our op. Drop matching entry from `pending`.
- `OpApplied { seq: None, op }` — op from another player or server simulation. Apply locally + re-mesh.
- `OpRejected { seq }` — walk `pending` from `seq` onwards, replay rollbacks in reverse, re-mesh.

In single-player, `OpRejected` never fires — the rollback path is dead code until multiplayer.
The pending list clears within ~1-2 server ticks as acks arrive.

### Applying locally

```rust
let chunk_data = Arc::make_mut(&mut self.chunks[&chunk_pos].data);
for (pos, block) in op_blocks_in_this_chunk {
    chunk_data.set(pos, block);
}
```

`Arc::make_mut` is the right primitive: clone-on-write. If the server still holds the
same `Arc` (likely), it clones the inner data; if not, it mutates in place. Either way
the client gets a private mutable copy without affecting the server's view.

## LODs are frozen after generation

LOD > 0 chunks are derived from world generation, not from base LOD 0 blocks. Once generated,
they don't update on block placement. Players modify blocks at LOD 0, and the visual mismatch
at distance is invisible.

This means:
- **LOD 0 chunks**: read+write on the client (collision, placement, meshing, shadow)
- **LOD > 0 chunks**: read-only on the client (meshing, shadow only)

Same data type (`Arc<ChunkData>`), different usage. Same code path on the client side.

## Memory cost

Naive approach: client holds `Arc<ChunkData>` for every loaded chunk across all LODs.
With ~5K chunks per LOD × 8 LODs × ~4KB compressed per chunk = **80-160 MB**.

This is comparable to Minecraft's actual memory cost for similar view distances. Not
catastrophic, but not great.

### The reduction strategy

**Drop `ChunkData` after consumption** for chunks outside interaction range:

```
Server sends Arc<ChunkData> → Client receives
  ↓
Client schedules mesh + bitmask + transparent-color worker job
  ↓
Worker reads ChunkData, produces mesh + bitmask + indirection
  ↓
Client uploads mesh to GPU, stores bitmask in pool
  ↓
For LOD 0 within interaction range: keep the Arc (collision, modification)
For everything else: drop the Arc — server still holds the canonical copy
```

After this:
- Interaction range (~500 LOD 0 chunks near player): ~500 × 4KB = **~2 MB ChunkData**
- All other chunks: 0 ChunkData on client, mesh on GPU, bitmask in pool
- Bitmask pool: ~8-10K active × 4KB = **~40 MB** (already exists today)

**Total client-side ChunkData drops from ~80-160 MB to ~2 MB.**

If the client later needs `ChunkData` for a chunk it dropped (e.g., player walked into
interaction range), it requests it via `RequestChunkData { pos, lod }`. Rare, amortized.

The server still holds canonical copies, so total process memory is unchanged — but the
data is no longer duplicated across worlds.

### Variable-bit-width palette compression

Worth verifying that `ChunkStorage` uses **variable-bit-width** palette indices (4 bits
when ≤16 unique blocks, 5 bits when ≤32, etc.) rather than fixed 1-byte indices. Fixed
1-byte gives 32KB per chunk; variable 4-bit gives 4KB. That's a 2× memory cut for free
if not already done.

## Camera & player

- **Player position**: server (authoritative game state)
- **Camera**: client (presentation, can interpolate between 20 TPS server ticks for smooth movement)

For a free-fly debug camera, decouple from player. For a real player camera, interpolate
between consecutive `PlayerMoved` updates so the camera moves smoothly even though the
underlying position only updates 20 times per second.

## Server tick handling

Two strategies for processing ops on the server:

**A. Tick-boundary processing.** Server drains the op queue once per tick. Worst-case
op latency is one tick (~50ms), but with client-side prediction this is invisible. Simpler.

**B. Eager processing.** Server has a tight loop that drains the op queue between ticks.
Op latency is sub-millisecond. Slightly more code.

Start with A. Switch to B only if it becomes a problem.

The 20 TPS tick is for **simulation work** (demand calculation, generation orchestration,
world simulation). Op processing can be either tick-bound or eager — they're separate.

## Open questions / future work

1. **Modul framework support for multi-world**: does `modul` have first-class SubApp support,
   or does this need to be bolted on? If not, the simplest path is a bare `std::thread` running
   a separate `bevy_ecs::World` with its own `Schedule`, communicating via channels.

2. **Cancellation / kill switch**: when the camera teleports, the chunk thread might be
   mid-update with stale work. Need a way to invalidate in-progress jobs.

3. **Initial sync**: first connection (or initial load on game start) needs to stream a lot
   of chunks. Probably batched, possibly compressed.

4. **Server-side simulation**: not needed for current scope but the architecture supports it.
   Water flow, plant growth, redstone, etc. would all run as server tick work and emit
   `BlockOp` events to broadcast.

## Implementation order

When the time comes to actually build this, three possible starting points:

**A. Top-down**: define channel types, the two-`World` setup, event protocol. Compile with
empty stubs. Fill in piece by piece.

**B. Bottom-up**: move just `update_chunk_demand` and `update_chunk_loading` to a background
thread. No World split yet, just channels. Validate the perf benefit, then expand.

**C. Refactor first**: identify which resources/components are server-side vs client-side
in the current code. Add marker types or split into separate modules. The actual world
split becomes mechanical.

Option B has the lowest risk and validates the approach early. Recommended starting point.

## What this architecture does NOT solve

- Generation latency itself (a chunk still takes time to generate; this just moves where the
  waiting happens)
- GPU upload bandwidth (still need budgeted uploads to avoid sync_up spikes)
- Mesh worker contention (existing worker pools still apply)

The split is specifically about **decoupling the chunk-loading orchestration from the render
loop**, so that bursty load work doesn't translate into frame stutters.

## Priority

**Not the next thing to do.** Bugs and clustered lighting come first:

1. Bug fixes (always first — clean baseline before any refactor)
2. Clustered lighting (additive, render-side, doesn't conflict with the future split)
3. Server architecture (this document) — when it's the only big thing on the plate

Lighting doesn't conflict with this refactor because lights are inherently client-side
(GPU buffers, shaders, light extraction during meshing). Building lighting first means
less code to migrate when the split happens.
