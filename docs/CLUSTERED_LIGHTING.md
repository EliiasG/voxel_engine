# Clustered Lighting

Design notes for the clustered light rendering system. Not yet implemented.

## Goals

- Many simultaneous lights from voxel content (torches, lava, glowing ore, neon walls).
- Many simultaneous lights from gameplay entities (player headlamp, enemy lanterns, drones, projectiles).
- Stable performance — clusters are the standard escape from the O(pixels × lights) shading cost.
- Reuse existing engine machinery (paged GPU buffers, camera-relative coords, chunk lifecycle, frustum cull).
- Sun + sun shadow stay as the existing directional pass; cluster lights are pure additive contribution on top.

Out of scope for v1:

- Per-light shadows. Cluster lights do not occlude each other or cast shadow. Adding even one
  shadow-casting point light is a separate large project (virtual shadow maps or per-light view
  rendering). The sun's existing compute-RT shadow mask is the only shadow-casting light.
- Custom block models. "Special" emissive blocks like torches are still cube-shaped voxels in v1
  — they just additionally register a light. Custom geometry comes later, the light contribution
  code path stays the same.
- Per-instance block facing. Wall-mounted variants are separate `BlockId`s for v1.

---

## Light populations and lifecycles

The architecturally important split is **static vs dynamic**, not LTC vs simple. Three populations,
two lifecycles:

| Population | Source | Lifecycle | Light type produced |
|---|---|---|---|
| Greedy-mesh emissive faces | Lava lake, glowing ore vein, neon strip | Static — chunk-owned | LTC area |
| "Special" emissive blocks | Torch, lantern, candle, glowstone | Static — chunk-owned | Simple (point/spot) |
| Entity-attached lights | Player headlamp, enemy lantern, drone light, projectile glow | Dynamic — ECS-owned | Simple (point/spot) |

The first two are extracted when a chunk loads, freed when it unloads, and re-extracted when blocks
in the chunk are placed/removed. They live in **paged GPU buffers** that reuse the existing
geometry paged allocator (`PAGE_SIZE = 96` per-page entries, free-list slot allocation,
per-chunk page assignment).

The third is uploaded fresh each frame from a single ECS query. Fixed cap of **1024** entries.

The lifecycle split is the design's load-bearing decision. Chunk-owned lights are GPU-resident and
**never re-uploaded per frame** in the steady state — a city of 50 000 torches costs zero per-frame
upload bandwidth, only the cluster build pass needs to read them. ECS-owned lights pay per-frame
upload but are bounded small.

---

## Light types

### LTC area light

One per merged emissive quad from the greedy mesher. Linearly Transformed Cosines (Heitz 2016).

| Field | Type | Notes |
|---|---|---|
| `chunk_pos` | `ivec3` | Chunk this light belongs to (used for camera-relative rebasing). |
| `local_pos` | `vec3<f32>` | Quad min corner, in chunk-local voxel space. |
| `edge_u` | `vec3<f32>` | First edge vector along the merged quad. |
| `edge_v` | `vec3<f32>` | Second edge vector along the merged quad. |
| `normal` | `vec3<f32>` | Face normal (axis-aligned in v1, free direction allowed by the format). |
| `color` | `vec3<f32>` | Linear RGB. |
| `intensity` | `f32` | Multiplied into `color` at eval time. Separate so artists can tweak without renormalizing. |
| `flags` | `u32` | Bit 0 = two-sided. Other bits reserved. |

LTC eval requires two precomputed 64×64 RGBA16F LUTs (M-matrix + amplitude/Fresnel). Shipped as
binary assets via `include_bytes!` next to the atlas, like the fog LUT. Standard files from the
public LTC repository — no runtime baking needed.

### Simple light (point and spot, single struct)

| Field | Type | Notes |
|---|---|---|
| `chunk_pos` | `ivec3` | Static lights only. Dynamic lights store this as zeros and use `local_pos` directly relative to camera chunk origin. |
| `local_pos` | `vec3<f32>` | Light position (post-`offset` for chunk lights). |
| `direction` | `vec3<f32>` | Spot direction. Unused for points. |
| `color` | `vec3<f32>` | Linear RGB. |
| `intensity` | `f32` | Multiplied into `color` at eval time. |
| `range` | `f32` | Hard cutoff distance. Clamped to `MAX_SIMPLE_LIGHT_RANGE` at registration. |
| `inner_cos` | `f32` | Cone smoothing. For points: set to `-1.0` (full sphere). |
| `outer_cos` | `f32` | Cone cutoff. For points: set to `-1.0`. |

One struct for both kinds halves cluster bookkeeping. The branch in shader cost is negligible
because spot vs point can be computed branchlessly via the `inner_cos`/`outer_cos` values
themselves.

Falloff:

```
let d = length(to_light);
let atten_dist = clamp(1.0 - (d / range), 0.0, 1.0);
let atten_dist_sq = atten_dist * atten_dist;
let radial = 1.0 / max(d * d, 0.01);
let spot_dot = dot(normalize(-to_light), direction);
let atten_cone = smoothstep(outer_cos, inner_cos, spot_dot);
let atten = atten_dist_sq * radial * atten_cone;
```

For points, `atten_cone` is always `1.0` because `spot_dot` is unused (the light has no direction)
— set `inner_cos = outer_cos = -1.0` so `smoothstep` collapses.

---

## Coordinate space

Same camera-relative discipline as the rest of the renderer (see `feedback_camera_relative.md`
and the recent fog bug). The CPU camera is f64; everything that goes to the GPU is `chunk_pos:
ivec3 + local_offset: vec3<f32>` and rebased per-frame against `camera.chunk_offset`.

### Chunk-owned lights (LTC and static simple)

Stored as `chunk_pos + local_pos`, mirroring `PageMetadata` for geometry pages. The shader
computes the camera-relative position as:

```wgsl
let rel_chunk = light.chunk_pos - camera.chunk_offset;
let cam_relative = vec3<f32>(rel_chunk * CHUNK_SIZE) + light.local_pos;
let to_light = cam_relative - surface.world_pos;
```

This means light positions are stable across `chunk_offset` changes — a chunk owning a torch
doesn't need to know which chunk the camera is in. The light's GPU representation never changes
once written.

### Dynamic ECS-owned lights

Computed on the CPU each frame as a single f64 subtraction:

```rust
let camera_chunk_origin_f64 = (camera_chunk_offset.as_dvec3()) * CHUNK_SIZE_F64;
let local = (entity_position.0 - camera_chunk_origin_f64).as_vec3();
```

Stored with `chunk_pos = camera_chunk_offset` and `local_pos = local`. The shader's rebasing
math then collapses to `rel_chunk = 0` and `local_pos` is used directly. No per-light integer
chunk bookkeeping needed for dynamic lights — they're per-frame uploads anyway, and the
camera chunk offset is the same across all of them.

---

## Block registry — `LightSpec`

Extend `block_props()` in `src/chunk/mod.rs`:

```rust
pub struct BlockProperties {
    pub is_transparent: bool,
    pub color_index: u8,
    pub light: Option<LightSpec>,
}

pub enum LightSpec {
    /// Block contributes a simple point or spot light extracted by the chunk light pass.
    Simple {
        offset: Vec3,         // local offset from block-min corner, in voxels (0..1 per axis usually)
        color: Vec3,
        intensity: f32,
        range: f32,           // clamped to MAX_SIMPLE_LIGHT_RANGE at extraction
        kind: SimpleLightKind,
    },
    /// Block's emissive faces feed the LTC light path via the greedy mesher.
    /// No separate extraction — handled in the meshing pass.
    LtcFromMesh {
        color: Vec3,
        intensity: f32,
        two_sided: bool,
    },
}

pub enum SimpleLightKind {
    Point,
    Spot {
        direction: Vec3,
        inner_cos: f32,
        outer_cos: f32,
    },
}
```

Block variations (orientation, fluid level, etc.) — register as separate `BlockId`s. A wall torch
is `TorchNorth`, `TorchEast`, etc., each with its own pre-baked offset and spot direction. This
avoids designing a per-instance block-state system in v1. Acceptable while variation count stays
under ~100 per concept.

---

## Chunk light extraction

Two passes, both run when a chunk's mesh is generated/regenerated:

### LTC extraction (piggybacks on greedy meshing)

The greedy mesher already iterates merged faces. For each face whose source `BlockId` has
`LightSpec::LtcFromMesh`, write an LTC entry into the chunk's allocated LTC pages alongside the
geometry pages. Same iteration, no extra scan.

Emissive faces also **skip AO during meshing** (treat AO as uniform 0). This prevents AO
boundaries from splitting emissive faces and gives maximally-merged quads — a single LTC light
per lava lake instead of dozens.

The merged quads serve double duty: rendered geometry AND LTC light source. They share the same
position/edge/normal data, just written into two parallel buffer streams.

### Simple light extraction (separate pass)

Linear scan over the chunk's block grid after meshing completes. For each block with
`LightSpec::Simple`, emit one entry at `block_local + spec.offset` into the chunk's simple-light
pages. Single-threaded, ~50 µs per chunk on commodity hardware. No neighbor lookups, no SIMD,
no bitmask machinery.

```rust
for (idx, block_id) in chunk.iter_blocks() {
    let Some(LightSpec::Simple { offset, color, intensity, range, kind }) = block_props(block_id).light else {
        continue;
    };
    let local = local_pos_from_index(idx) + offset;
    let range = range.min(MAX_SIMPLE_LIGHT_RANGE);
    simple_lights.push(SimpleLight { local_pos: local, color, intensity, range, kind });
}
```

The output goes through the same paged allocator as geometry. Each chunk owns one or more pages
in the global `simple_lights_buffer`, with a per-page metadata entry containing `chunk_pos`.

### Re-extraction on edits

Block placement/removal triggers a re-mesh of the affected chunk (existing pattern). The two
light passes re-run inside the same job and re-write their pages. No additional invalidation
machinery needed.

---

## GPU buffer layout

Three buffers, all using the same paged allocator pattern as the existing geometry buffer:

| Buffer | Allocator | Page size | Purpose |
|---|---|---|---|
| `ltc_lights_buffer` | Paged free-list | Match geometry (96 entries) | All chunk-owned LTC lights |
| `simple_lights_buffer` | Paged free-list | Match geometry (96 entries) | All chunk-owned simple lights |
| `dynamic_lights_buffer` | Single fixed-size, ring or rewrite | 1024 entries | ECS-owned per-frame lights |

Each paged buffer has a parallel `LightPageMetadata` SSBO with one entry per page:

```rust
#[repr(C)]
struct LightPageMetadata {
    chunk_pos: [i32; 3],
    light_count: u32,         // <= PAGE_SIZE; tail of page may be unused
}
```

The shader rebases `chunk_pos` against `camera.chunk_offset` exactly like geometry pages.

Index encoding into per-cluster lists uses the high bits of the index to identify the source
buffer:

```
bits 31..28 = buffer id (0=LTC, 1=simple chunk, 2=simple dynamic)
bits 27..0  = local index into that buffer
```

This lets a single per-cluster index list serve all three buffers without sorting by type.

---

## Frustum culling — two layers

### Why per-chunk light culling is not the same as per-chunk mesh culling

The existing CPU chunk frustum cull rejects chunks outside the camera frustum. For meshes that's
correct — a chunk behind the camera contributes zero pixels.

For **lights** that's wrong. A torch in a chunk just behind the camera still illuminates the
wall in front of you. Mesh culling alone would cause shadow popping every time the camera turns.

Light pages need to be culled with an **expanded** frustum test: the chunk's AABB inflated by
`MAX_SIMPLE_LIGHT_RANGE` in all directions. "Is any light in this chunk close enough to the
frustum to possibly affect a visible pixel?"

### Layer 1 — CPU per-chunk light cull

```rust
const MAX_SIMPLE_LIGHT_RANGE: f32 = 32.0;  // 1 chunk

for chunk in loaded_chunks {
    let mesh_visible = frustum_intersects(chunk.aabb);
    let lights_visible = frustum_intersects(chunk.aabb.expanded(MAX_SIMPLE_LIGHT_RANGE));

    if mesh_visible {
        visible_mesh_pages.extend(chunk.mesh_pages);
    }
    if lights_visible {
        visible_light_pages.extend(chunk.ltc_pages);
        visible_light_pages.extend(chunk.simple_pages);
    }
}
```

Piggybacks on the existing chunk iteration in the cull system. One extra plane test per chunk,
one extra output list. Negligible CPU cost. Can dramatically reduce the input set the GPU
cluster build has to scan — in a city with 100 k torches loaded, only the lights from chunks
within ~32 voxels of the visible region make it to the GPU.

LTC lights technically have tighter bounds than `MAX_SIMPLE_LIGHT_RANGE` (they're rectangles,
not point sources), but using the same expanded test for both is correct (LTC bounds are smaller
or equal) and keeps the cull loop branch-free.

Dynamic lights skip layer 1 entirely. Max 1024 of them — feed them all directly to the cluster
build.

### Layer 2 — GPU cluster build

A compute pass takes the layer-1-culled light pages plus the dynamic light buffer and writes
per-cluster light index lists. This **is** fine-grained frustum culling — clusters are
sub-frustums, and a light that doesn't intersect any cluster definitionally doesn't intersect
any visible pixel.

#### Cluster topology

- **16 × 9 screen tiles × 24 log-Z depth slices = 3 456 clusters.**
- Standard Doom-2016 layout. Tile aspect roughly matches 16:9 displays.
- Log-Z slicing: depth slice `i` covers the depth range `near × (far/near)^(i/24)` to `near × (far/near)^((i+1)/24)`. Concentrates resolution near the camera where it matters.
- All numbers configurable; these are starting points. Likely fine for the project's foreseeable
  scale (thousands of lights, not hundreds of thousands).

#### Compute pass structure

One workgroup per cluster (or one thread per cluster, depending on contention). Each thread:

1. Computes its cluster's frustum AABB in view space.
2. Iterates the visible LTC pages, simple chunk pages, and dynamic lights.
3. For each light, computes its bounding sphere/box and tests against the cluster AABB.
4. On hit, atomically appends the encoded index `(buffer_id << 28) | local_index` into the
   cluster's own pre-allocated slice of the index list, using a **per-cluster** atomic counter.

Two SSBOs out:

- `cluster_index_counts` — one `u32` per cluster. Incremented atomically on hit. Bounded by
  `MAX_LIGHTS_PER_CLUSTER` (starting value: 256). Overflow hits are dropped — profile and bump
  the cap if it ever triggers.
- `cluster_index_list` — flat array sized `NUM_CLUSTERS * MAX_LIGHTS_PER_CLUSTER`. Cluster `i`
  owns the slice `[i * MAX_LIGHTS_PER_CLUSTER, (i + 1) * MAX_LIGHTS_PER_CLUSTER)`.

At `3 456 clusters × 256 × 4 B`, the index list is ~3.5 MB — trivial. Per-cluster counters
distribute atomic traffic across 3 456 counters instead of one, which avoids pathological
contention by construction without needing workgroup shared memory gymnastics.

The fragment shader computes `start_offset = cluster_id * MAX_LIGHTS_PER_CLUSTER` directly from
the cluster id — no separate offsets buffer needed.

The fragment shader looks up its cluster from `clip_position` and reads the index list:

```wgsl
let cluster_id = compute_cluster(in.clip_position);
let count = min(cluster_index_counts[cluster_id], MAX_LIGHTS_PER_CLUSTER);
let start = cluster_id * MAX_LIGHTS_PER_CLUSTER;
for (var i = 0u; i < count; i++) {
    let encoded = cluster_index_list[start + i];
    let buffer_id = encoded >> 28u;
    let local_idx = encoded & 0x0FFFFFFFu;
    // dispatch by buffer_id and accumulate contribution
}
```

#### Cluster build budget

For ~10 k visible lights × 3 456 clusters = ~35 M tests per frame. GPU-trivial. Even 100 k visible
lights is feasible if needed; layer 1 is what keeps the input set manageable.

---

## Shading

The fragment shader runs after the existing sun + shadow + AO pipeline in `lighting.wgsl`. The
clustered contribution is **purely additive** on top.

```wgsl
fn apply_lighting(surface: Surface) -> vec4<f32> {
    // ... existing sun + shadow + AO + sky_light contribution ...
    var lit = surface.base_color * (ambient_term + sun_diffuse);

    // Clustered light accumulation
    let cluster_id = compute_cluster(surface.clip_position);
    let cluster = cluster_index_offsets[cluster_id];
    for (var i = 0u; i < cluster.count; i++) {
        let encoded = cluster_index_list[cluster.start_offset + i];
        let buffer_id = encoded >> 28u;
        let local_idx = encoded & 0x0FFFFFFFu;
        if (buffer_id == 0u) {
            lit += eval_ltc(ltc_lights[local_idx], surface);
        } else if (buffer_id == 1u) {
            lit += eval_simple(simple_lights[local_idx], surface);
        } else {
            lit += eval_simple(dynamic_lights[local_idx], surface);
        }
    }

    let final_color = apply_fog(lit, surface.world_pos);
    return vec4<f32>(final_color, 1.0);
}
```

Cluster lights are **not shadowed**. They contribute pure additive light filtered by surface
normal (NdotL). This is the major v1 simplification.

### LTC eval

Standard Heitz LTC integration:

1. Sample the LTC LUTs at `(NdotV, roughness)` to get the inverse transformation matrix `M⁻¹`.
2. Transform the rectangle's four corners by `M⁻¹` into the cosine-distribution space.
3. Compute the polygon-clipping integral over the unit hemisphere.
4. Multiply by the amplitude term + Fresnel term.

For voxel surfaces with effectively-Lambert response, roughness can be hard-coded or omitted
entirely (just use the pure-cosine LUT slice). This simplification halves the LUT cost.

### Simple eval

```wgsl
fn eval_simple(light: SimpleLight, surface: Surface) -> vec3<f32> {
    let rel_chunk = light.chunk_pos - camera.chunk_offset;
    let light_pos = vec3<f32>(rel_chunk * CHUNK_SIZE) + light.local_pos;
    let to_light = light_pos - surface.world_pos;
    let d = length(to_light);
    if (d > light.range) { return vec3<f32>(0.0); }
    let L = to_light / d;
    let ndotl = max(dot(surface.normal, L), 0.0);
    let dist_atten = clamp(1.0 - d / light.range, 0.0, 1.0);
    let dist_atten_sq = dist_atten * dist_atten;
    let radial = 1.0 / max(d * d, 0.01);
    let spot_dot = dot(-L, light.direction);
    let cone_atten = smoothstep(light.outer_cos, light.inner_cos, spot_dot);
    return light.color * light.intensity * ndotl * dist_atten_sq * radial * cone_atten;
}
```

Branch-free for point vs spot — points set `inner_cos = outer_cos = -1.0` and `cone_atten`
collapses to `1.0`.

---

## Implementation slices

Three visually-verifiable checkpoints. Each builds on the previous and is mergeable independently.

### Slice 1 — Simple lights end-to-end (no clustering, no LTC)

Goal: get one new light type rendering correctly with both static and dynamic sources.

- Add `light: Option<LightSpec>` to `BlockProperties`. Add `TORCH` `BlockId` (visually still a
  glowing cube — defer custom models). Register a `LightSpec::Simple` for it with a center offset.
- Implement chunk simple-light extraction pass. Add `SimpleLightPages` component alongside
  `ChunkFaces`. Reuse the existing paged allocator for `simple_lights_buffer`.
- Add `Light` ECS component + system that fills `dynamic_lights_buffer` per frame. Test entity:
  a debug headlamp parented to the camera.
- Wire simple light shading into `lighting.wgsl`. **No clustering yet** — fragment shader loops
  over the entire flat global light buffer (LTC + cluster passes are stubbed). This is slow but
  immediately verifiable.
- Add new bind group for the lights buffers. Update opaque + transparent pipelines to bind it.
- **Visual checkpoint**: place torches in a cave, walk around with the headlamp. Both should
  illuminate walls correctly with smooth attenuation. Performance will be bad with many lights —
  this is expected and will be fixed in slice 2.

Files touched (estimated):

- `src/chunk/mod.rs` — `LightSpec` types, `block_props` extension, `TORCH` constant.
- `src/chunk/loading.rs` — call simple light extraction after meshing.
- `src/chunk/lighting.rs` (new) — extraction pass.
- `src/render/lights.rs` (new) — paged buffer + bind group + dynamic upload system.
- `src/render/mod.rs` — register new bind group, add to geometry pipeline builder.
- `src/render/wboit.rs` — add new bind group to transparent pipeline.
- `src/render/shaders/lights_common.wgsl` (new) — light struct definitions, `eval_simple()`.
- `src/render/shaders/lighting.wgsl` — call clustered loop after sun contribution.

### Slice 2 — Frustum culling and cluster build

Goal: make slice 1's lighting fast enough for thousands of static lights.

- Add `MAX_SIMPLE_LIGHT_RANGE = 32` constant. Clamp light ranges at extraction.
- CPU layer 1 cull: extend the existing chunk frustum cull to produce a parallel
  `visible_light_pages` list using the expanded AABB. Negligible code addition.
- GPU cluster build compute pass: 16 × 9 × 24 clusters. One workgroup per cluster.
  Atomically appends light indices to `cluster_index_list` with `(buffer_id << 28) | local_idx`
  encoding.
- Fragment shader switches from "loop all global lights" to "loop my cluster's index list".
- **Visual checkpoint**: identical visuals to slice 1, but now scales to thousands of static
  lights without dying. Profile against slice 1 with a stress-test scene.

Files touched:

- `src/render/cull.rs` — extend chunk cull with expanded AABB pass.
- `src/render/lights.rs` — cluster build dispatch + buffers.
- `src/render/shaders/cluster_build.wgsl` (new) — compute pass.
- `src/render/shaders/lighting.wgsl` — switch to clustered loop.

### Slice 3 — LTC area lights

Goal: replace flat lava-lake lighting with area-light irradiance.

- Add `LtcFromMesh` variant to `LightSpec`. Add `LAVA` `BlockId` with this variant. Register
  caves/biomes to spawn it.
- Hook the greedy mesher to write LTC entries when emitting an emissive face. Skip AO for
  emissive blocks (uniform AO=0) so faces stay maximally merged.
- Add `ltc_lights_buffer` (paged, same allocator). Wire its bind group entry.
- Ship Heitz LTC LUTs (M-matrix + amplitude). Load at startup, bind to atmosphere or new bind
  group. Two 64×64 RGBA16F textures.
- Implement `eval_ltc()` in `lights_common.wgsl`. Add the `buffer_id == 0` branch to the
  fragment shader's cluster loop.
- **Visual checkpoint**: lava lake illuminates the cavern ceiling above with full irradiance,
  not just nearest-point falloff.

Files touched:

- `src/chunk/meshing.rs` — emit LTC entries on emissive face merge; AO skip for emissive.
- `src/chunk/generation.rs` — spawn lava in caves.
- `src/render/lights.rs` — LTC paged buffer.
- `src/render/ltc_lut.rs` (new) — LUT load.
- `src/render/shaders/lights_common.wgsl` — `eval_ltc()`.

---

## Prerequisites and dependencies

| Prerequisite | Status |
|---|---|
| Block placement (for testing) | Done — commit 9693e99 |
| `BlockProperties` registry | Exists at `src/chunk/mod.rs:34`, needs `light` field |
| Paged GPU allocator | Done — `src/render/mod.rs:28+`, reuse pattern |
| Paged lights buffers | New, but pattern-identical to existing geometry buffers |
| Greedy mesher emissive flag | Mesher knows opaque/transparent today; needs emissive output |
| LTC LUTs | Need to ship as binary assets (Heitz public files) |
| ECS `Light` component + system | New |
| Cluster build compute pipeline | New |

---

## Resolved decisions

- **Light buffer page size** — **8 entries per page** (not 96 like geometry). Most chunks have
  few lights; 96-slot pages would be pathologically sparse. The allocator is the same code path,
  just parameterized on page size.
- **LTC roughness** — **hard-coded constant**. Voxel surfaces are Lambert; skip the LUT
  roughness axis entirely. Halves the LUT cost and simplifies eval. Revisit only if PBR
  materials are ever added.
- **Light intensity HDR range** — deferred to a tuning pass once slice 1 is on-screen.
  Interacts with the tonemapper; can only be dialed in visually.
- **MAX_SIMPLE_LIGHT_RANGE** — **global 32-voxel cap**, no per-chunk override. Gameplay-driven
  long-range lights (beacons) are not a v1 concern.
- **Light buffer growth strategy** — existing doubling-slab allocator is fine for v1. Revisit
  only if profiling shows pathological sparsity.

## Cluster build contention — decision ladder

Per-cluster atomic counters with pre-allocated slices (described above) handle the v1 scale by
construction. If profiling later shows stalls, escalate in order:

1. **v1 baseline:** per-cluster atomic counters, fixed cap `MAX_LIGHTS_PER_CLUSTER = 256`,
   overflow drops extra hits.
2. **If a cluster genuinely overflows the cap in gameplay:** bump the cap, or switch to
   two-pass (count → exclusive prefix scan → write) for exact sizing.
3. **If atomic contention within a hot cluster bottlenecks the build pass:** fall back to
   workgroup-shared-memory local lists with a compaction copy. Adds a spill path and barrier
   logic — only worth it if measurements demand it.

---

## Why this design

A few rejected alternatives for posterity:

- **Forward+ tile shading instead of clustered.** Clustered subsumes Forward+ (Forward+ is
  effectively 2D clusters with one Z slice). Once you accept cluster bookkeeping, going from 2D
  to 3D is free, and 3D dramatically improves indoor scenes where lights are stacked in depth.
- **Per-light shadows.** Considered and rejected for v1. Even one shadow caster doubles the
  rendering cost; multiple shadow casters require either virtual shadow maps (massive complexity)
  or per-light passes (linear in light count). Distance falloff alone gives the correct
  qualitative behavior for additive lights.
- **One unified lights buffer instead of three.** Considered. Rejected because LTC and simple
  lights have different field layouts and very different lifecycles. Three smaller, type-pure
  buffers are clearer and let the cluster build use type bits in indices to dispatch shading.
- **Closest-point area lights instead of LTC.** The previous version of this plan started with
  closest-point math as a v1 simplification. With the current scope (tens of thousands of
  emissive blocks possible), going straight to LTC is worth the extra complexity — closest-point
  is wrong precisely in the case voxel scenes care most about (large emissive surfaces lighting
  nearby geometry, e.g., cavern ceilings above lava).
- **Per-frame upload of all chunk lights.** Considered as a v1 simplification (skip the paged
  GPU allocator). Rejected because in steady state (tens of thousands of static torches in a
  city), the per-frame upload cost is wasted — those lights never change. The paged allocator is
  the right answer the first time.
