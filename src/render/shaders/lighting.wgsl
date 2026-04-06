// Shared lighting pipeline. Requires camera, shadow mask, and atmosphere bind groups,
// plus sky_sample.wgsl to be concatenated before this file.

struct Surface {
    base_color: vec3<f32>,
    normal: vec3<f32>,
    world_pos: vec3<f32>,
    clip_position: vec4<f32>,
    ao: f32,
};

fn apply_lighting(surface: Surface) -> vec4<f32> {
    let light_dir = atmosphere.sun_direction;
    let raw_ndotl = dot(surface.normal, light_dir);
    let ndotl = max(raw_ndotl, 0.0);

    // Edge-aware shadow upscale: 3x3 neighborhood with hard accept/reject
    let shadow_uv = surface.clip_position.xy / camera.screen_size;
    let shadow_dims = vec2<f32>(textureDimensions(shadow_mask));
    let frag_height = dot(surface.world_pos, surface.normal);

    let center_texel = shadow_uv * shadow_dims - 0.5;
    let center_i = vec2<i32>(round(center_texel));
    let dims_i = vec2<i32>(shadow_dims);

    var total_shadow = vec3<f32>(0.0);
    var total_weight = 0.0;
    var best_shadow = vec3<f32>(0.0);
    var best_dist = 999.0;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let tc = clamp(center_i + vec2<i32>(dx, dy), vec2<i32>(0), dims_i - 1);
            let ss = textureLoad(shadow_mask, tc, 0);
            let n = textureLoad(shadow_normal, tc, 0).xyz * 2.0 - 1.0;
            let height_diff = abs(ss.a - frag_height);
            if (dot(n, surface.normal) > 0.9 && height_diff < 0.3) {
                let d = vec2<f32>(tc) + 0.5 - (center_texel + 0.5);
                let w = 1.0 / (1.0 + dot(d, d));
                total_shadow += ss.rgb * w;
                total_weight += w;
            }
            let spatial_d = length(vec2<f32>(tc) - center_texel);
            if (spatial_d < best_dist) {
                best_dist = spatial_d;
                best_shadow = ss.rgb;
            }
        }
    }

    var shadow_color: vec3<f32>;
    if (total_weight > 0.001) {
        shadow_color = total_shadow / total_weight;
    } else {
        shadow_color = best_shadow;
    }

    // Faces pointing away from sun are always in shadow regardless of mask
    let shadow = select(shadow_color, vec3<f32>(0.0), raw_ndotl <= 0.0);

    // Modulate lighting by time of day
    let day = 1.0 - atmosphere.night_factor;
    let sky_light = max(surface.normal.y * 0.5 + 0.5, 0.0) * mix(0.04, 0.15, day);

    let ambient = mix(0.08, 0.25, day);
    let diffuse = 0.7 * ndotl * shadow * day;
    let ao = mix(0.4, 1.0, surface.ao);
    let ambient_term = ambient * ao + sky_light * ao;
    var lit_color = surface.base_color * (vec3<f32>(ambient_term) + diffuse);

    // Additive simple light contribution (chunk-owned + dynamic). Slice 2
    // looks up this fragment's cluster and walks only the lights that
    // touch it.
    let light_contrib = accumulate_simple_lights(
        surface.world_pos, surface.normal, surface.clip_position,
    );
    lit_color = lit_color + surface.base_color * light_contrib;

    // Exponential distance fog
    let final_color = apply_fog(lit_color, surface.world_pos);

    return vec4<f32>(final_color, 1.0);
}
