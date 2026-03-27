# Voxel Engine Optimizations

Reference document for optimization techniques. Implementation details to follow.

---

## Rendering Architecture

### 6 Buffers Per Chunk (One Per Face Direction)
- Separate vertex buffers for +X, -X, +Y, -Y, +Z, -Z faces
- **Benefits:**
  - No need to store normal per vertex (implicit from buffer)
  - Entire buffers can be skipped based on camera direction (backface culling at buffer level)
  - Better GPU batching (uniform shader state per buffer)

### Instanced Rendering
- Store block faces as instanced vertices
- Single draw call per buffer
- GPU generates final vertex positions from instance data

### Indirect Rendering
- GPU fills draw commands, CPU doesn't stall waiting for counts
- Essential for dynamic chunk visibility
- CPU queues commands without knowing exact face counts
- Pairs with compute shaders for GPU-driven culling

---

## Mesh Generation

### Greedy Meshing
- Combine adjacent coplanar faces into larger quads
- Drastically reduces triangle count
- Faces merged along X and Y axes per slice
- Trade-off: more complex mesh generation, but far fewer triangles to render

### Removing Obstructed Faces
- Don't generate faces between two solid blocks
- Only emit faces adjacent to air/transparent blocks

---

## Data Layout

### Bit-Packed Vertices
- Pack vertex data into 32 or 64-bit integers
- Example 32-bit layout: 10 bits X, 10 bits Y, 10 bits Z, 2 bits spare
- Example 64-bit layout: 20 bits X, 20 bits Y, 20 bits Z + greedy dims + material/AO
- With 6 direction buffers, no normal bits needed

### Single-Dimensional Arrays
- Use flat arrays instead of `[x][y][z]` multi-dimensional
- Index via bitshifts: `data[x + (y << 5) + (z << 10)]` for 32-wide chunks
- Enables compiler optimizations and better cache locality

### Chunk-Local Coordinates
- Store positions relative to chunk origin (0-31 range for 32-block chunks)
- Reduces bits needed per vertex
- Chunk world position applied via uniform/push constant

---

## Memory & Access Patterns

### Neighboring Chunk References
- Cache references to 6 adjacent chunks before mesh generation
- Avoids repeated global map lookups for edge block checks
- Significant speedup for chunk boundary face culling

### Direct Buffer Writing
- Write vertices directly to GPU-mapped buffer
- Avoid intermediate arrays and copies
- Periodic capacity checks, grow buffer as needed

### Bitwise Coordinate Conversion
- Use shifts and masks instead of division/modulo
- `chunk_index = pos >> 5` instead of `pos / 32`
- `local_pos = pos & 0x1F` instead of `pos % 32`

---

## Culling

### Backface Culling (Buffer-Level)
- With 6 direction buffers, skip entire buffers based on camera facing
- Camera facing +X → don't draw -X buffer at all

### Frustum Culling
- Per-chunk AABB test against view frustum
- Skip chunks entirely outside view
- Can be GPU-driven with indirect rendering

### Occlusion Culling (Advanced)
- Skip chunks hidden behind terrain
- Hierarchical Z-buffer or software rasterization approaches
- More complex, evaluate if needed

---

## GPU Considerations

### Vertex Alignment
- Use 32-bit or 64-bit vertices (power of 2)
- Avoid 48-bit or other awkward sizes
- GPU vertex fetch prefers aligned data

### Buffer Streaming
- Double or triple buffer for chunks being updated
- Avoid stalls from writing buffers in use

---

## Summary: Optimization Chain

```
Obstructed Face Removal  →  Fewer faces generated
Greedy Meshing           →  Faces merged into larger quads
6 Direction Buffers      →  No normals stored, easy backface cull
Bit-Packed Vertices      →  Minimal memory per vertex
Instanced Rendering      →  One draw call per buffer
Indirect Rendering       →  GPU-driven, no CPU stalls
Frustum/Occlusion Cull   →  Skip invisible chunks entirely
```

---

## References

- [Vercidium Blog - Voxel World Optimisations](https://vercidium.com/blog/voxel-world-optimisations/)
- [Vercidium - Further Optimisations](https://vercidium.com/blog/further-voxel-world-optimisations/)
- [Vercidium GitHub - Voxel Mesh Generation](https://github.com/Vercidium/voxel-mesh-generation)
- [Vercidium Video - I Optimised My Game Engine Up To 12000 FPS](https://www.youtube.com/watch?v=40JzyaOYJeY)
