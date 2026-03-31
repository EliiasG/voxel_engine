// Shadow ray trace compute shader
// Reads depth buffer, reconstructs world position,
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
    scale_factor: f32,
    _pad3: f32,
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
@group(1) @binding(2)
var normal_tex: texture_2d<f32>;

@group(2) @binding(0)
var<uniform> lod_infos: array<LodInfo, 6>;
@group(2) @binding(1)
var<storage, read> grid: array<u32>;
@group(2) @binding(2)
var<storage, read> bitmask_data: array<u32>;

@group(3) @binding(0)
var prev_accum_tex: texture_2d<f32>;
@group(3) @binding(1)
var prev_accum_sampler: sampler;
@group(3) @binding(2)
var output_tex: texture_storage_2d<rgba16float, write>;

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

@compute @workgroup_size(8, 8)
fn cs_shadow(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel = gid.xy;
    let dims = vec2<u32>(shadow.shadow_tex_size);
    if (pixel.x >= dims.x || pixel.y >= dims.y) {
        return;
    }

    let uv = (vec2<f32>(pixel) + 0.5) / shadow.shadow_tex_size;

    // Checkerboard: only trace half the pixels per frame, reuse previous for the other half
    let checker = (pixel.x / 2u + pixel.y / 2u + shadow.frame_index) % 2u;

    // Jitter UV by sub-pixel offset for temporal anti-aliasing
    let jitter = (jitter_offset(shadow.frame_index) - 0.5) / shadow.shadow_tex_size;
    let jittered_uv = uv + jitter;
    let depth = textureSampleLevel(depth_tex, depth_sampler, jittered_uv, 0u);

    // Reconstruct camera-relative position for depth output and early-outs
    var normal_height = 0.0;
    var cam_rel_early = vec3<f32>(0.0);
    if (depth > 0.0) {
        let ndc_e = vec4<f32>(
            jittered_uv.x * 2.0 - 1.0,
            (1.0 - jittered_uv.y) * 2.0 - 1.0,
            depth, 1.0,
        );
        let clip_e = shadow.inv_view_proj * ndc_e;
        cam_rel_early = clip_e.xyz / clip_e.w;
        // Store height along surface normal — distinguishes parallel surfaces at different positions
        let face_n = textureSampleLevel(normal_tex, depth_sampler, jittered_uv, 0.0).xyz * 2.0 - 1.0;
        normal_height = dot(cam_rel_early, face_n);
    }

    // Skipped pixels: return previous frame's value directly
    if (checker != 0u && depth > 0.0) {
        var prev_uv = uv;
        if (shadow.camera_moving != 0u) {
            let chunk_shift = vec3<f32>((shadow.chunk_offset - shadow.prev_chunk_offset) * CHUNK_SIZE);
            let prev_clip = shadow.prev_view_proj * vec4<f32>(cam_rel_early + chunk_shift, 1.0);
            let prev_ndc = prev_clip.xyz / prev_clip.w;
            prev_uv = vec2<f32>(prev_ndc.x * 0.5 + 0.5, 1.0 - (prev_ndc.y * 0.5 + 0.5));
            if (any(prev_uv < vec2<f32>(0.0)) || any(prev_uv > vec2<f32>(1.0))) {
                prev_uv = uv;
            }
        }
        let prev = textureSampleLevel(prev_accum_tex, prev_accum_sampler, prev_uv, 0.0);
        textureStore(output_tex, pixel, vec4<f32>(prev.r, prev.g, 0.0, 0.0));
        return;
    }

    // Back-facing early-out: surfaces facing away from sun are always in shadow
    if (depth > 0.0) {
        let face_normal_early = textureSampleLevel(normal_tex, depth_sampler, jittered_uv, 0.0).xyz * 2.0 - 1.0;
        let ndotl_early = dot(face_normal_early, normalize(shadow.sun_direction));
        if (ndotl_early <= 0.0) {
            textureStore(output_tex, pixel, vec4<f32>(0.0, normal_height, 0.0, 0.0));
            return;
        }
    }

    // Trace result: 0.0 = shadow, 1.0 = lit
    var trace_result = 1.0;
    var cam_rel_pos = vec3<f32>(0.0);

    // Sky pixels are always lit (reversed-Z: sky = 0.0)
    if (depth > 0.0) {
        // Reconstruct world position from depth
        let ndc = vec4<f32>(
            jittered_uv.x * 2.0 - 1.0,
            (1.0 - jittered_uv.y) * 2.0 - 1.0,
            depth,
            1.0,
        );
        let clip = shadow.inv_view_proj * ndc;
        cam_rel_pos = clip.xyz / clip.w;
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

            // Sample face normal from geometry pass (encoded as n*0.5+0.5)
            let face_normal = textureSampleLevel(normal_tex, depth_sampler, jittered_uv, 0.0).xyz * 2.0 - 1.0;
            let ndotl = abs(dot(face_normal, ray_dir));

            // Directional bias (along ray), scaled by downscale factor
            let sf = shadow.scale_factor;
            let dir_bias = mix(0.01 * sf, 1.0 * sf, t * t);
            // Normal offset: push ray origin out of surface, scales with distance like dir_bias
            let tn = clamp(dist / 200.0, 0.0, 1.0);
            let normal_bias = mix(0.05 * sf, 1.0 * sf, tn * tn);
            let normal_push = face_normal * (1.0 - ndotl) * start_voxel_size * normal_bias;
            var pos = world_pos + normal_push + ray_dir * dir_bias * start_voxel_size;
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

    // Temporal accumulation with reprojection
    var result = trace_result;
    if (depth > 0.0) {
        var prev_uv = uv;
        var reprojection_valid = true;

        if (shadow.camera_moving != 0u) {
            // Reproject current world position into previous frame's screen space
            let chunk_shift = vec3<f32>((shadow.chunk_offset - shadow.prev_chunk_offset) * CHUNK_SIZE);
            let prev_cam_rel = cam_rel_pos + chunk_shift;
            let prev_clip = shadow.prev_view_proj * vec4<f32>(prev_cam_rel, 1.0);
            let prev_ndc = prev_clip.xyz / prev_clip.w;
            prev_uv = vec2<f32>(prev_ndc.x * 0.5 + 0.5, 1.0 - (prev_ndc.y * 0.5 + 0.5));
            reprojection_valid = all(prev_uv >= vec2<f32>(0.0)) && all(prev_uv <= vec2<f32>(1.0));
        }

        if (reprojection_valid) {
            let prev_sample = textureSampleLevel(prev_accum_tex, prev_accum_sampler, prev_uv, 0.0).r;
            let dist = length(cam_rel_pos);
            let t = clamp((dist - 30.0) / 170.0, 0.0, 1.0); // 30..200 range
            // Moving: aggressive blend so near shadows respond fast
            // Static: gentle blend so jitter converges smoothly without visible flicker
            var blend = mix(0.6, 0.2, t);
            if (shadow.camera_moving == 0u) {
                blend = mix(0.12, 0.04, t);
            }
            result = mix(prev_sample, trace_result, blend);
        }
    }

    textureStore(output_tex, pixel, vec4<f32>(result, normal_height, 0.0, 0.0));
}
