// Simple light evaluation, used by the opaque voxel pipeline.
// Concatenated after the lights bind group library, before lighting.wgsl.
// Slice 2: clustered loop. The cluster build compute pass writes a flat
// per-cluster light index list; this code looks up the surface fragment's
// cluster from clip-space coordinates and walks only that cluster's slice.

const CHUNK_SIZE_LIGHTS: i32 = 32;

const NUM_CLUSTERS_X: u32 = 16u;
const NUM_CLUSTERS_Y: u32 = 9u;
const NUM_CLUSTERS_Z: u32 = 24u;
const NUM_CLUSTERS: u32 = 3456u;
const MAX_LIGHTS_PER_CLUSTER: u32 = 256u;
const CLUSTER_NEAR: f32 = 0.1;
const CLUSTER_FAR: f32 = 1024.0;
const CLUSTER_INDEX_MASK: u32 = 0x0FFFFFFFu;
const BUFFER_ID_CHUNK: u32 = 0u;
const BUFFER_ID_DYNAMIC: u32 = 1u;

fn eval_simple_light(light: SimpleLight, world_pos: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    // Camera-relative light position. `chunk_pos` is integer chunks; rebase
    // against the camera's chunk offset before adding the local fractional
    // offset. Dynamic lights store chunk_pos = (0,0,0) and bake everything
    // into local_pos already.
    let rel_chunk = light.chunk_pos - camera.chunk_offset;
    let light_pos = vec3<f32>(rel_chunk * CHUNK_SIZE_LIGHTS) + light.local_pos;
    let to_light = light_pos - world_pos;
    let d2 = dot(to_light, to_light);
    let range_sq = light.range * light.range;
    if (d2 > range_sq) {
        return vec3<f32>(0.0);
    }
    let d = sqrt(max(d2, 1e-6));
    let L = to_light / d;
    let ndotl = max(dot(normal, L), 0.0);
    if (ndotl <= 0.0) {
        return vec3<f32>(0.0);
    }
    // Linear distance attenuation, smoothed: (1 - d/range)^2.
    let dist_atten = clamp(1.0 - d / light.range, 0.0, 1.0);
    let dist_atten_sq = dist_atten * dist_atten;
    // Inverse-square radial term, with a floor to avoid divide-by-zero
    // singularities for very close points.
    let radial = 1.0 / max(d2, 0.01);
    // Spot cone attenuation. Points have inner_cos = outer_cos = -1, which
    // collapses smoothstep to 1.0 (full sphere).
    let spot_dot = dot(-L, light.direction);
    let cone_atten = smoothstep(light.outer_cos, light.inner_cos, spot_dot);

    return light.color * light.intensity * ndotl * dist_atten_sq * radial * cone_atten;
}

/// Compute which cluster a fragment belongs to from its clip-space
/// position. Mirrors the cluster build pass's depth slicing.
fn compute_cluster_id(clip_position: vec4<f32>) -> u32 {
    // Tile from screen XY (clip_position.xy is window-space pixels in
    // wgpu's @builtin(position): origin at top-left, y increases down).
    // The cluster build operates in NDC, where y increases up. We invert
    // the Y axis here so cluster_y=0 maps to the bottom of the screen in
    // both passes — matching the compute pass's `ndc_y_min = -1 + 2*cy/N`.
    let pixel = clip_position.xy;
    let screen = camera.screen_size;
    let tile_x = clamp(
        u32(pixel.x / screen.x * f32(NUM_CLUSTERS_X)),
        0u, NUM_CLUSTERS_X - 1u
    );
    let tile_y = clamp(
        u32((1.0 - pixel.y / screen.y) * f32(NUM_CLUSTERS_Y)),
        0u, NUM_CLUSTERS_Y - 1u
    );

    // Linear depth from reverse-Z NDC z. The project's perspective matrix
    // makes ndc_z = near / d, so d = near / ndc_z.
    let ndc_z = clip_position.z;
    let d = max(CLUSTER_NEAR / max(ndc_z, 1e-6), CLUSTER_NEAR);
    // Inverse of the cluster build's `d = near * (far/near)^(z/N)`.
    let log_ratio = log(CLUSTER_FAR / CLUSTER_NEAR);
    let slice_f = log(d / CLUSTER_NEAR) / log_ratio * f32(NUM_CLUSTERS_Z);
    let tile_z = clamp(u32(slice_f), 0u, NUM_CLUSTERS_Z - 1u);

    return tile_z * NUM_CLUSTERS_X * NUM_CLUSTERS_Y + tile_y * NUM_CLUSTERS_X + tile_x;
}

/// Walk this fragment's cluster light index list and return the sum of
/// per-light contributions. Camera-relative inputs.
fn accumulate_simple_lights(
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    clip_position: vec4<f32>,
) -> vec3<f32> {
    var sum = vec3<f32>(0.0);

    let cluster_id = compute_cluster_id(clip_position);
    let raw_count = cluster_index_counts[cluster_id];
    let count = min(raw_count, MAX_LIGHTS_PER_CLUSTER);
    let start = cluster_id * MAX_LIGHTS_PER_CLUSTER;

    for (var i: u32 = 0u; i < count; i = i + 1u) {
        let encoded = cluster_index_list[start + i];
        let buffer_id = encoded >> 28u;
        let local_idx = encoded & CLUSTER_INDEX_MASK;
        if (buffer_id == BUFFER_ID_CHUNK) {
            sum = sum + eval_simple_light(chunk_lights[local_idx], world_pos, normal);
        } else {
            sum = sum + eval_simple_light(dynamic_lights[local_idx], world_pos, normal);
        }
    }

    return sum;
}
