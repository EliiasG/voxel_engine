// Shadow ray trace shader
// Reads previous frame's depth buffer, reconstructs world position,
// traces a ray toward the sun through the bitmask acceleration structure.
// Outputs: 0.0 = shadow, 1.0 = lit.

const CHUNK_SIZE: i32 = 32;
const GRID_EMPTY: u32 = 0xFFFFFFFFu;
const GRID_SOLID: u32 = 0xFFFFFFFEu;
const MAX_STEPS: u32 = 512u;

struct ShadowUniform {
    inv_view_proj: mat4x4<f32>,
    chunk_offset: vec3<i32>,
    _pad0: i32,
    sun_direction: vec3<f32>,
    max_ray_distance: f32,
    lod_count: u32,
    grid_size: u32,
    _pad1: vec2<u32>,
};

struct LodInfo {
    grid_origin_x: i32,
    grid_origin_y: i32,
    grid_origin_z: i32,
    grid_size: u32,
    lod_scale: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0)
var<uniform> shadow: ShadowUniform;

@group(1) @binding(0)
var depth_tex: texture_depth_2d;
@group(1) @binding(1)
var depth_sampler: sampler;

@group(2) @binding(0)
var<uniform> lod_infos: array<LodInfo, 6>;
@group(2) @binding(1)
var<storage, read> grid: array<u32>;
@group(2) @binding(2)
var<storage, read> bitmask_data: array<u32>;

struct ShadowVaryings {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_shadow(@builtin(vertex_index) vi: u32) -> ShadowVaryings {
    var out: ShadowVaryings;
    let uv = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
    out.position = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(uv.x, 1.0 - uv.y);
    return out;
}

// Read a bit from the fine bitmask (32x32x32 = 32768 bits = 1024 u32s per slot)
fn test_fine_bit(slot: u32, x: u32, y: u32, z: u32) -> bool {
    let bit_index = x + y * 32u + z * 1024u;
    let word_index = bit_index / 32u;
    let bit = bit_index % 32u;
    // Layout: slot * (2 + 1024) u32s. First 2 u32s = coarse (u64 as 2xu32), then 1024 u32s = fine
    let base = slot * 1026u;
    return (bitmask_data[base + 2u + word_index] & (1u << bit)) != 0u;
}

// Read a bit from the coarse bitmask (4x4x4 = 64 bits = 2 u32s per slot)
fn test_coarse_bit(slot: u32, rx: u32, ry: u32, rz: u32) -> bool {
    let bit_index = rx + ry * 4u + rz * 16u;
    let base = slot * 1026u;
    if (bit_index < 32u) {
        return (bitmask_data[base] & (1u << bit_index)) != 0u;
    } else {
        return (bitmask_data[base + 1u] & (1u << (bit_index - 32u))) != 0u;
    }
}

// Look up the grid entry for a chunk position at a given LOD
fn grid_lookup(lod: u32, chunk_pos: vec3<i32>) -> u32 {
    let info = lod_infos[lod];
    let local = chunk_pos - vec3<i32>(info.grid_origin_x, info.grid_origin_y, info.grid_origin_z);
    let s = i32(info.grid_size);
    if (any(local < vec3<i32>(0)) || any(local >= vec3<i32>(s))) {
        return GRID_EMPTY;
    }
    let entries_per_lod = info.grid_size * info.grid_size * info.grid_size;
    let lod_offset = lod * entries_per_lod;
    let flat = u32(local.x) + u32(local.y) * info.grid_size + u32(local.z) * info.grid_size * info.grid_size;
    return grid[lod_offset + flat];
}

// Trace ray through a single chunk's bitmask. Returns true if hit.
fn trace_chunk_bitmask(
    slot: u32,
    ray_pos: vec3<f32>,
    ray_dir: vec3<f32>,
    chunk_origin: vec3<f32>,
    voxel_size: f32,
) -> bool {
    // Position in chunk-local voxel coords (0..32)
    let local_start = (ray_pos - chunk_origin) / voxel_size;

    // DDA through 4x4x4 coarse regions (each 8 voxels wide)
    let coarse_size = 8.0;
    let inv_dir = 1.0 / ray_dir;

    var coarse_pos = vec3<i32>(floor(local_start / coarse_size));
    let step = vec3<i32>(sign(ray_dir));
    let step_f = vec3<f32>(sign(ray_dir));

    // Distance to next coarse boundary
    var t_max = (vec3<f32>(
        select(f32(coarse_pos.x), f32(coarse_pos.x + 1), ray_dir.x >= 0.0),
        select(f32(coarse_pos.y), f32(coarse_pos.y + 1), ray_dir.y >= 0.0),
        select(f32(coarse_pos.z), f32(coarse_pos.z + 1), ray_dir.z >= 0.0),
    ) * coarse_size - local_start) * inv_dir;

    let t_delta = abs(vec3<f32>(coarse_size) * inv_dir);

    for (var i = 0u; i < 24u; i++) {
        if (any(coarse_pos < vec3<i32>(0)) || any(coarse_pos >= vec3<i32>(4))) {
            break;
        }

        if (test_coarse_bit(slot, u32(coarse_pos.x), u32(coarse_pos.y), u32(coarse_pos.z))) {
            // DDA through individual voxels in this 8^3 region
            let region_origin = vec3<f32>(coarse_pos) * coarse_size;
            let fine_start = clamp(local_start + ray_dir * max(0.0, min(t_max.x, min(t_max.y, t_max.z)) - max(t_delta.x, max(t_delta.y, t_delta.z))), region_origin, region_origin + vec3<f32>(7.999));

            var fine_pos = vec3<i32>(floor(max(local_start, region_origin)));
            // Clamp to region bounds
            fine_pos = clamp(fine_pos, coarse_pos * 8, coarse_pos * 8 + 7);

            var ft_max = (vec3<f32>(
                select(f32(fine_pos.x), f32(fine_pos.x + 1), ray_dir.x >= 0.0),
                select(f32(fine_pos.y), f32(fine_pos.y + 1), ray_dir.y >= 0.0),
                select(f32(fine_pos.z), f32(fine_pos.z + 1), ray_dir.z >= 0.0),
            ) - local_start) * inv_dir;

            let ft_delta = abs(inv_dir);
            let region_max = coarse_pos * 8 + 8;

            for (var j = 0u; j < 32u; j++) {
                if (any(fine_pos < coarse_pos * 8) || any(fine_pos >= region_max)) {
                    break;
                }

                if (test_fine_bit(slot, u32(fine_pos.x), u32(fine_pos.y), u32(fine_pos.z))) {
                    return true;
                }

                // Advance DDA
                if (ft_max.x < ft_max.y && ft_max.x < ft_max.z) {
                    fine_pos.x += step.x;
                    ft_max.x += ft_delta.x;
                } else if (ft_max.y < ft_max.z) {
                    fine_pos.y += step.y;
                    ft_max.y += ft_delta.y;
                } else {
                    fine_pos.z += step.z;
                    ft_max.z += ft_delta.z;
                }
            }
        }

        // Advance coarse DDA
        if (t_max.x < t_max.y && t_max.x < t_max.z) {
            coarse_pos.x += step.x;
            t_max.x += t_delta.x;
        } else if (t_max.y < t_max.z) {
            coarse_pos.y += step.y;
            t_max.y += t_delta.y;
        } else {
            coarse_pos.z += step.z;
            t_max.z += t_delta.z;
        }
    }

    return false;
}

@fragment
fn fs_shadow(in: ShadowVaryings) -> @location(0) vec4<f32> {
    let depth = textureSample(depth_tex, depth_sampler, in.uv);

    // Sky pixels are always lit (reversed-Z: sky = 0.0)
    if (depth <= 0.0) {
        return vec4<f32>(1.0);
    }

    // Reconstruct world position from depth
    let ndc = vec4<f32>(
        in.uv.x * 2.0 - 1.0,
        (1.0 - in.uv.y) * 2.0 - 1.0,  // flip Y for wgpu NDC
        depth,
        1.0,
    );
    let clip = shadow.inv_view_proj * ndc;
    let cam_rel_pos = clip.xyz / clip.w;

    // World position = camera-relative + chunk_offset * CHUNK_SIZE
    let world_pos = cam_rel_pos + vec3<f32>(shadow.chunk_offset * CHUNK_SIZE);

    let ray_dir = normalize(shadow.sun_direction);

    // Determine starting LOD from world_pos (before offset)
    var current_lod = 0u;
    for (var l = 0u; l < shadow.lod_count; l++) {
        let info = lod_infos[l];
        let lod_scale = i32(info.lod_scale);
        let chunk_pos = vec3<i32>(floor(world_pos / f32(CHUNK_SIZE * lod_scale)));
        let local = chunk_pos - vec3<i32>(info.grid_origin_x, info.grid_origin_y, info.grid_origin_z);
        let s = i32(info.grid_size);
        if (all(local >= vec3<i32>(0)) && all(local < vec3<i32>(s))) {
            current_lod = l;
            break;
        }
    }

    let start_voxel_size = f32(1u << current_lod);
    let dist = length(cam_rel_pos);
    let bias = mix(0.1, 1.5, clamp(dist / 500.0, 0.0, 1.0));
    var pos = world_pos + ray_dir * bias * start_voxel_size;
    var total_dist = 0.0;

    // Main ray march loop
    for (var step_count = 0u; step_count < MAX_STEPS; step_count++) {
        if (total_dist > shadow.max_ray_distance) {
            break;
        }

        let info = lod_infos[current_lod];
        let lod_scale = i32(info.lod_scale);
        let chunk_world_size = f32(CHUNK_SIZE * lod_scale);

        // Current chunk position at this LOD
        let chunk_pos = vec3<i32>(floor(pos / chunk_world_size));

        // Check if we're within this LOD's grid
        let local = chunk_pos - vec3<i32>(info.grid_origin_x, info.grid_origin_y, info.grid_origin_z);
        let s = i32(info.grid_size);
        if (any(local < vec3<i32>(0)) || any(local >= vec3<i32>(s))) {
            // Try stepping up to next LOD
            if (current_lod + 1u < shadow.lod_count) {
                current_lod += 1u;
                continue;
            }
            break;  // Outside all grids — lit
        }

        let slot = grid_lookup(current_lod, chunk_pos);

        if (slot == GRID_SOLID) {
            return vec4<f32>(0.0, 0.0, 0.0, 1.0);  // shadow
        }

        if (slot != GRID_EMPTY) {
            // Partial chunk — trace through bitmask
            let chunk_origin = vec3<f32>(chunk_pos * CHUNK_SIZE * lod_scale);
            let voxel_size = f32(lod_scale);
            if (trace_chunk_bitmask(slot, pos, ray_dir, chunk_origin, voxel_size)) {
                return vec4<f32>(0.0, 0.0, 0.0, 1.0);  // shadow
            }
        }

        // DDA step to next chunk boundary
        let chunk_min = vec3<f32>(chunk_pos) * chunk_world_size;
        let chunk_max = chunk_min + vec3<f32>(chunk_world_size);
        let inv_dir = 1.0 / ray_dir;

        let t_exit = vec3<f32>(
            select((chunk_min.x - pos.x) * inv_dir.x, (chunk_max.x - pos.x) * inv_dir.x, ray_dir.x >= 0.0),
            select((chunk_min.y - pos.y) * inv_dir.y, (chunk_max.y - pos.y) * inv_dir.y, ray_dir.y >= 0.0),
            select((chunk_min.z - pos.z) * inv_dir.z, (chunk_max.z - pos.z) * inv_dir.z, ray_dir.z >= 0.0),
        );

        let t_step = max(0.001, min(t_exit.x, min(t_exit.y, t_exit.z)));
        pos += ray_dir * (t_step + 0.01);
        total_dist += t_step + 0.01;
    }

    return vec4<f32>(1.0, 1.0, 1.0, 1.0);  // lit
}
