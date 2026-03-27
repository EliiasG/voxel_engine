# Demo Spec

Single chunk proof-of-concept. Render one 32x32x32 chunk of procedural terrain.

## Features

- Procedural terrain (sine-wave hills)
- Internal face culling (only emit faces adjacent to air)
- Face data supports W x H dimensions (greedy meshing ready, not implemented)
- 6 direction buffers per chunk
- Instanced quad rendering (6 vertices per face instance)
- Orbiting camera
- Directional lighting with position-based coloring
- Depth testing

## Not In Scope

- Multiple chunks
- LOD
- Frustum / occlusion culling
- Buffer-level backface culling
- Indirect rendering
- Material system
- Player input
