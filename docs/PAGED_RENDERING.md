# Paged Indirect Rendering

## Overview

One large vertex buffer holds all face data across all chunks, divided into fixed-size **pages** of 512 faces. Each page belongs to one chunk × one direction. A chunk with many faces in one direction simply uses multiple pages.

A parallel **metadata storage buffer** has one entry per page (direction + chunk origin), indexed by `page_index = instance_index / PAGE_SIZE`.

Each frame, the CPU iterates all active pages, performs culling, and writes a compact **indirect buffer** containing only visible draw args. A single `multi_draw_indirect` call renders everything.

**Page allocation** is a simple free list — load a chunk = grab pages, unload = return them.

---

## Face Data Layout (8 bytes)

Stored in the vertex buffer. No direction or chunk origin — those come from page metadata.

| Field    | Size    | Offset |
|----------|---------|--------|
| X        | 1 byte  | 0      |
| Y        | 1 byte  | 1      |
| Z        | 1 byte  | 2      |
| W        | 1 byte  | 3      |
| H        | 1 byte  | 4      |
| Material | 3 bytes | 5      |

Read in the shader as two `Uint8x4` vertex attributes:
- `data0`: X, Y, Z, W
- `data1`: H, Mat0, Mat1, Mat2

---

## Page Metadata (16 bytes)

One entry per page in a storage buffer. Indexed by `page_index = instance_index / PAGE_SIZE`.

| Field     | Type   | Size     |
|-----------|--------|----------|
| Origin X  | f32    | 4 bytes  |
| Origin Y  | f32    | 4 bytes  |
| Origin Z  | f32    | 4 bytes  |
| Direction | u32    | 4 bytes  |

---

## Indirect Draw Args (16 bytes)

One entry per visible page. Rebuilt each frame.

| Field          | Value                       |
|----------------|-----------------------------|
| vertex_count   | 6                           |
| instance_count | face count in page         |
| first_vertex   | 0                           |
| first_instance | page_index × PAGE_SIZE    |

---

## GPU Buffers

| Buffer   | Size                                  | Usage              |
|----------|---------------------------------------|--------------------|
| Face     | MAX_PAGES × PAGE_SIZE × 8 bytes    | Vertex + Copy Dst  |
| Metadata | MAX_PAGES × 16 bytes                 | Storage + Copy Dst |
| Indirect | MAX_PAGES × 16 bytes                 | Indirect + Copy Dst|

---

## Bind Groups

| Group | Binding | Content                     |
|-------|---------|-----------------------------|
| 0     | 0       | Camera uniform (mat4×4)     |
| 1     | 0       | Metadata storage buffer     |

---

## Frame Loop

1. CPU iterates active pages
2. Frustum cull (per page, using chunk AABB)
3. Backface cull (per page, using direction)
4. Write surviving draw args to compact buffer
5. Upload indirect buffer
6. One `multi_draw_indirect` call

---

## Chunk Load / Unload

**Load:**
1. Generate terrain, build face buffers (6 directions)
2. For each direction, split faces into pages of 512
3. Allocate page index from free list for each page
4. Upload face data and metadata to GPU at page offsets
5. Record page indices in chunk entry

**Unload:**
1. Return all page indices to free list
2. Stale face data is harmless — no draw args will reference it
