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
    frame_index: u32,
    camera_moving: u32,
    prev_view_proj: mat4x4<f32>,
    prev_chunk_offset: vec3<i32>,
    _pad2: i32,
    shadow_tex_size: vec2<f32>,
    _pad3: vec2<f32>,
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

@group(3) @binding(0)
var prev_accum_tex: texture_2d<f32>;
@group(3) @binding(1)
var prev_accum_sampler: sampler;
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
// Flat DDA through 32x32x32 with O(1) coarse region skipping.
fn trace_chunk_bitmask(
    slot: u32,
    ray_pos: vec3<f32>,
    ray_dir: vec3<f32>,
    chunk_origin: vec3<f32>,
    voxel_size: f32,
) -> bool {
    let local = (ray_pos - chunk_origin) / voxel_size;
    let inv_dir = 1.0 / ray_dir;
    let step = vec3<i32>(sign(ray_dir));

    var vpos = vec3<i32>(floor(local));
    vpos = clamp(vpos, vec3<i32>(0), vec3<i32>(31));

    var t_max = (vec3<f32>(
        select(f32(vpos.x), f32(vpos.x + 1), ray_dir.x >= 0.0),
        select(f32(vpos.y), f32(vpos.y + 1), ray_dir.y >= 0.0),
        select(f32(vpos.z), f32(vpos.z + 1), ray_dir.z >= 0.0),
    ) - local) * inv_dir;

    let t_delta = abs(inv_dir);

    for (var i = 0u; i < 96u; i++) {
        if (any(vpos < vec3<i32>(0)) || any(vpos >= vec3<i32>(32))) {
            break;
        }

        let coarse = vec3<u32>(vpos) / 8u;
        if (!test_coarse_bit(slot, coarse.x, coarse.y, coarse.z)) {
            // Empty 8x8x8 region — jump to exit face in O(1)
            let exit_face = vec3<f32>(
                select(f32(coarse.x * 8u), f32((coarse.x + 1u) * 8u), ray_dir.x >= 0.0),
                select(f32(coarse.y * 8u), f32((coarse.y + 1u) * 8u), ray_dir.y >= 0.0),
                select(f32(coarse.z * 8u), f32((coarse.z + 1u) * 8u), ray_dir.z >= 0.0),
            );
            let t_exit = (exit_face - local) * inv_dir;
            let t_skip = min(t_exit.x, min(t_exit.y, t_exit.z));

            let new_pos = local + ray_dir * (t_skip + 0.001);
            vpos = vec3<i32>(floor(new_pos));
            vpos = clamp(vpos, vec3<i32>(0), vec3<i32>(31));

            t_max = (vec3<f32>(
                select(f32(vpos.x), f32(vpos.x + 1), ray_dir.x >= 0.0),
                select(f32(vpos.y), f32(vpos.y + 1), ray_dir.y >= 0.0),
                select(f32(vpos.z), f32(vpos.z + 1), ray_dir.z >= 0.0),
            ) - local) * inv_dir;
            continue;
        }

        if (test_fine_bit(slot, u32(vpos.x), u32(vpos.y), u32(vpos.z))) {
            return true;
        }

        if (t_max.x < t_max.y && t_max.x < t_max.z) {
            vpos.x += step.x;
            t_max.x += t_delta.x;
        } else if (t_max.y < t_max.z) {
            vpos.y += step.y;
            t_max.y += t_delta.y;
        } else {
            vpos.z += step.z;
            t_max.z += t_delta.z;
        }
    }

    return false;
}

// 4-sample Halton(2,3) jitter pattern
fn jitter_offset(frame: u32) -> vec2<f32> {
    switch frame % 4u {
        case 0u: { return vec2<f32>(0.0, 0.0); }
        case 1u: { return vec2<f32>(0.5, 0.333); }
        case 2u: { return vec2<f32>(0.25, 0.667); }
        case 3u: { return vec2<f32>(0.75, 0.111); }
        default: { return vec2<f32>(0.0, 0.0); }
    }
}

@fragment
fn fs_shadow(in: ShadowVaryings) -> @location(0) vec4<f32> {
    // Jitter UV by sub-pixel offset
    let jitter = (jitter_offset(shadow.frame_index) - 0.5) / shadow.shadow_tex_size;
    let jittered_uv = in.uv + jitter;
    let depth = textureSample(depth_tex, depth_sampler, jittered_uv);

    // Trace result: 0.0 = shadow, 1.0 = lit
    var trace_result = 1.0;

    // Sky pixels are always lit (reversed-Z: sky = 0.0)
    if (depth > 0.0) {
        // Reconstruct world position from depth (use jittered UV for NDC)
        let ndc = vec4<f32>(
            jittered_uv.x * 2.0 - 1.0,
            (1.0 - jittered_uv.y) * 2.0 - 1.0,
            depth,
            1.0,
        );
        let clip = shadow.inv_view_proj * ndc;
        let cam_rel_pos = clip.xyz / clip.w;
        let world_pos = cam_rel_pos + vec3<f32>(shadow.chunk_offset * CHUNK_SIZE);

        let ray_dir = normalize(shadow.sun_direction);

        {
            // Determine starting LOD
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
            let t = clamp(dist / 500.0, 0.0, 1.0);
            let bias = mix(0.025, 15.0, t * t);
            var pos = world_pos + ray_dir * bias * start_voxel_size;
            var total_dist = 0.0;
            var hit = false;

            for (var step_count = 0u; step_count < MAX_STEPS; step_count++) {
                if (total_dist > shadow.max_ray_distance || hit) {
                    break;
                }

                let info = lod_infos[current_lod];
                let lod_scale = i32(info.lod_scale);
                let chunk_world_size = f32(CHUNK_SIZE * lod_scale);
                let chunk_pos = vec3<i32>(floor(pos / chunk_world_size));
                let local = chunk_pos - vec3<i32>(info.grid_origin_x, info.grid_origin_y, info.grid_origin_z);
                let s = i32(info.grid_size);

                if (any(local < vec3<i32>(0)) || any(local >= vec3<i32>(s))) {
                    if (current_lod + 1u < shadow.lod_count) {
                        current_lod += 1u;
                        continue;
                    }
                    break;
                }

                let slot = grid_lookup(current_lod, chunk_pos);

                if (slot == GRID_SOLID) {
                    hit = true;
                } else if (slot != GRID_EMPTY) {
                    let chunk_origin = vec3<f32>(chunk_pos * CHUNK_SIZE * lod_scale);
                    let voxel_size = f32(lod_scale);
                    if (trace_chunk_bitmask(slot, pos, ray_dir, chunk_origin, voxel_size)) {
                        hit = true;
                    }
                }

                if (!hit) {
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
            }

            if (hit) {
                trace_result = 0.0;
            }
        }
    }

    // Temporal accumulation — disabled during camera motion to avoid blur
    var result = trace_result;
    if (shadow.camera_moving == 0u && depth > 0.0) {
        let prev_sample = textureSample(prev_accum_tex, prev_accum_sampler, in.uv).r;
        result = mix(prev_sample, trace_result, 0.25);
    }

    return vec4<f32>(result, result, result, 1.0);
}
