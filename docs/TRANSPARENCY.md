# Transparency rendering + colored shadows

Two phases: Phase B (transparent rendering via WBOIT) first, Phase A (colored shadows) second.

## Key design decisions

- **WBOIT** for multi-layer transparency (water tanks, stacked glass)
- **No shadows ON transparent surfaces** — only opaque surfaces behind show (colored) shadows
- **Beer-Lambert absorption** in log space: accumulate `vec3` absorption per transparent voxel hit, `exp(-total)` once per ray. Distance-correct via DDA t values, scaled by `lod_scale`
- **Absorption coefficients** precomputed on CPU (`-ln(tint)`) and uploaded as a uniform array of 254 `vec3<f32>` entries

## Per-8x8x8 region color buffer (dual-purpose)

- 4×4×4 = 64 `u32` indirection per chunk, `u32::MAX` = no transparent blocks
- Pool entries: 8×8×8 × 1 byte = 512B per region
- Byte values: 0-253 = transparent color index, 254 = air, 255 = opaque
- When indirection valid: color buffer replaces fine bitmask for that region (one read per voxel step)
- When `u32::MAX`: use fine bitmask as normal
- Indirection cached at region entry, re-read on region transition

## Shadow trace logic

```
if !coarse_bit → skip region
cached_indirection = indirection[region]
if cached_indirection != u32::MAX:
    byte = color_pool[cached_indirection + local_idx]
    254 = air (continue), 255 = opaque (stop), else = accumulate absorption
else:
    fine bitmask check (stop if set)
```

## Bitmask changes

- Fine mask: transparent blocks = AIR (no bit, ray passes through)
- Coarse mask: set for regions with opaque OR transparent blocks
- Color data built in render system (not chunk system), removed from entity after upload

## Buffer management

Same pattern as BitmaskPool: CPU-side Vec + free list + dirty tracking, GPU storage buffer doubles when needed.

---

## Phase B: Transparent geometry rendering

### B1. Block properties
- Add transparent flag + color index per `BlockId`
- Define initial transparent block types (WATER, GLASS)

### B2. Meshing changes (`src/chunk/meshing.rs`)
- `mesh_chunk()` separates transparent faces from opaque
- New component (e.g., `TransparentChunkFaces`) alongside existing `ChunkFaces`
- Transparent faces still get standard/border split per direction

### B3. GPU upload (`src/render/mod.rs`)
- Upload transparent faces to separate pages (same slab system, separate tracking)
- Remove `TransparentChunkFaces` component after upload

### B4. Transparent render pass
- Runs after opaque pass
- Reads opaque depth buffer (depth test, no depth write)
- Same vertex format as opaque pipeline
- Different blend state for WBOIT accumulation

### B5. WBOIT implementation
- Two render targets: accumulation (`Rgba16Float`) + revealage (`R8Unorm`)
- Transparent fragment shader outputs:
  - Accumulation: `vec4(color.rgb * color.a * w, color.a * w)` where `w` = depth-based weight
  - Revealage: `color.a`
- Fullscreen resolve pass composites over opaque framebuffer:
  - `final = (1 - revealage) * opaque + accum.rgb / max(accum.a, 0.001)`

### B6. Draw cache integration
- Separate `DrawCache` (or section) for transparent geometry
- Same incremental update pattern as opaque cache

---

## Phase A: Colored shadows

### A1. Transparent color data structures
- `TransparentColorPool`: `Vec<[u8; 512]>` (8×8×8 regions) + free list + dirty tracking
  - Same pattern as `BitmaskPool` (`src/render/shadow/grid.rs:118-150`)
  - Each byte: 0-253 = transparent color, 254 = air, 255 = opaque
- Per-chunk indirection: 4×4×4 = 64 `u32` entries alongside grid data
  - `u32::MAX` = no transparent blocks in region (use fine bitmask)
  - Otherwise = pool slot index (use color buffer instead of fine bitmask)
- GPU storage buffer: starts small, doubles when needed

### A2. Build in render system
- New system `build_transparent_color_data()` reads `ChunkData`/`ChunkStorage`
- Builds indirection table + allocates pool entries for regions containing transparent blocks
- Fills each byte: air → 254, opaque → 255, transparent → color index 0-253
- Runs at synchronize time, removes source component after upload
- Deallocates pool entries on chunk unload

### A3. Bitmask interaction
- `build_bitmask()` fine mask: transparent blocks = AIR (no fine bit set)
- `build_bitmask()` coarse mask: set for regions containing opaque OR transparent blocks
- Indirection value cached per region transition — determines per-voxel data source

### A4. Shadow ray tracing changes (`src/render/shaders/shadow.wgsl`)
- Add `total_absorption: vec3<f32>` accumulator
- Per transparent voxel hit: `total_absorption += absorption_rgb * distance * f32(lod_scale)`
  - `distance` = `t_next - t_current` from DDA (path length through voxel)
  - `absorption_rgb` looked up from uniform array indexed by color byte (0-253)
- End of ray: `color = exp(-total_absorption)`
- Bind transparent indirection + color pool buffers to shadow compute pipeline
- Absorption coefficients precomputed on CPU (`-ln(tint)`), uploaded as uniform array of 254 `vec3<f32>`

### A5. Shadow output format
- `.rgb` = colored shadow multiplier (was: `.r` = scalar)
- `.a` = normal height (was: `.g`)
- `vec3(1,1,1)` = lit, `vec3(0,0,0)` = shadowed, `vec3(0.7,0,0.7)` = purple shadow

### A6. Lighting shader (`src/render/shaders/lighting.wgsl`)
- Edge-aware upscaling operates on `vec3` shadow color
- Normal height reads from `.a` instead of `.g`
- `surface_color * shadow_color` instead of `* shadow_scalar`

---

## Verification

### Phase B
- `cargo build` + `cargo test`
- Visual: place transparent blocks, verify WBOIT rendering (no sorting artifacts, multiple layers)

### Phase A
- Visual: colored shadows through glass blocks
- Visual: multiple transparent layers (water tank scenario)
- Verify no shadow ON transparent surfaces, only colored shadow on opaque behind
