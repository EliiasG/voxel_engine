# Rendering Architecture

## Overview

Each chunk is rendered using 6 vertex buffers (one per face direction: +X, -X, +Y, -Y, +Z, -Z). Faces are stored as instanced quads with greedy mesh dimensions. LOD is a per-chunk uniform that scales all positions and sizes.

---

## Face Layout (64 bits / 8 bytes)

| Field    | Size   | Description                     |
|----------|--------|---------------------------------|
| X        | 1 byte | Chunk-local X coordinate (0-255)|
| Y        | 1 byte | Chunk-local Y coordinate (0-255)|
| Z        | 1 byte | Chunk-local Z coordinate (0-255)|
| W        | 1 byte | Greedy mesh width (0-255)       |
| H        | 1 byte | Greedy mesh height (0-255)      |
| Material | 3 bytes| Material data (TBD)             |

**Notes:**
- 8 bits per coordinate is generous for 32-block chunks but allows experimentation with larger chunk sizes
- Normal is implicit from which of the 6 buffers the face belongs to
- Material bits reserved for future use (texture ID, AO, lighting, flags)

---

## LOD System

LOD is a per-chunk uniform, not stored per face.

| LOD | Scale | Chunk World Extent (32-block chunk) |
|-----|-------|-------------------------------------|
| 0   | 1x    | 32 units                            |
| 1   | 2x    | 64 units                            |
| 2   | 4x    | 128 units                           |
| 3   | 8x    | 256 units                           |
| n   | 2^n   | 32 * 2^n units                      |

**Shader scaling:**
```glsl
scale = 1 << lod;
world_pos  = chunk_origin + local_pos * scale;
world_size = face_size * scale;
```

All chunks have identical block dimensions (32x32x32, size TBD). Higher LOD chunks cover more world space with the same vertex count.

---

## Per-Chunk Data

| Data           | Storage          | Description                          |
|----------------|------------------|--------------------------------------|
| Vertex Buffers | 6 GPU buffers    | One per face direction               |
| Chunk Origin   | Uniform/Push     | World position of chunk corner       |
| LOD            | Uniform/Push     | Scale factor exponent (0, 1, 2, ...) |

---

## Buffer Organization

```
Chunk
├── Buffer +X  (faces pointing +X direction)
├── Buffer -X  (faces pointing -X direction)
├── Buffer +Y  (faces pointing +Y direction)
├── Buffer -Y  (faces pointing -Y direction)
├── Buffer +Z  (faces pointing +Z direction)
└── Buffer -Z  (faces pointing -Z direction)
```

**Benefits:**
- Normal implicit from buffer (no storage cost)
- Backface culling at buffer level (skip entire buffer if facing away)
- Uniform shader state per draw call

---

## Rendering Pipeline

1. **Frustum cull** chunks against view frustum
2. **Determine visible buffers** per chunk based on camera direction (backface cull)
3. **Issue indirect draw** commands for visible buffers
4. **Vertex shader** scales position/size by LOD, computes world position
5. **Fragment shader** samples material, applies lighting

---

## Draw Call Structure

Per visible chunk, up to 6 draw calls (one per direction buffer, skip backfacing):

```
For each visible chunk:
    For each direction buffer:
        If buffer faces camera:
            Draw instanced (face count from buffer)
            Push constants: chunk_origin, lod
```

With indirect rendering, draw commands are queued without CPU knowing exact face counts.

---

## Backface Culling (Buffer Level)

Camera view direction determines which buffers to skip entirely:

| Camera Facing | Skip Buffer |
|---------------|-------------|
| +X            | -X          |
| -X            | +X          |
| +Y            | -Y          |
| -Y            | +Y          |
| +Z            | -Z          |
| -Z            | +Z          |

In practice, camera faces a direction so 3 opposite buffers can be skipped. Edge cases (camera exactly axis-aligned) may draw up to 3 directions.

---

## Open Questions

- [ ] Optimal chunk size (32? 64? benchmark)
- [ ] Material bit layout (texture ID, AO, light, flags)
- [ ] Index buffer vs pure instancing
- [ ] Double/triple buffering strategy for streaming chunks
- [ ] Compute shader for indirect draw command generation
