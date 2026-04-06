// Simple light evaluation, used by the opaque voxel pipeline.
// Concatenated after the lights bind group library, before lighting.wgsl.
// Slice 1: flat global loop over chunk and dynamic light buffers — no
// clustering, no frustum cull. Slow but correct.

const CHUNK_SIZE_LIGHTS: i32 = 32;

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

/// Walk the chunk and dynamic light buffers and return the sum of their
/// contributions for a surface. Camera-relative inputs.
fn accumulate_simple_lights(world_pos: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    var sum = vec3<f32>(0.0);

    let chunk_n = light_counts.chunk_light_count;
    for (var i: u32 = 0u; i < chunk_n; i = i + 1u) {
        sum = sum + eval_simple_light(chunk_lights[i], world_pos, normal);
    }

    let dyn_n = light_counts.dynamic_light_count;
    for (var i: u32 = 0u; i < dyn_n; i = i + 1u) {
        sum = sum + eval_simple_light(dynamic_lights[i], world_pos, normal);
    }

    return sum;
}
