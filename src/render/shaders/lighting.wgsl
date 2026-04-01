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

    var total_shadow = 0.0;
    var total_weight = 0.0;
    var best_shadow = 0.0;
    var best_dist = 999.0;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let tc = clamp(center_i + vec2<i32>(dx, dy), vec2<i32>(0), dims_i - 1);
            let ss = textureLoad(shadow_mask, tc, 0);
            let n = textureLoad(shadow_normal, tc, 0).xyz * 2.0 - 1.0;
            let height_diff = abs(ss.g - frag_height);
            if (dot(n, surface.normal) > 0.9 && height_diff < 0.3) {
                let d = vec2<f32>(tc) + 0.5 - (center_texel + 0.5);
                let w = 1.0 / (1.0 + dot(d, d));
                total_shadow += ss.r * w;
                total_weight += w;
            }
            let spatial_d = length(vec2<f32>(tc) - center_texel);
            if (spatial_d < best_dist) {
                best_dist = spatial_d;
                best_shadow = ss.r;
            }
        }
    }

    var shadow_val: f32;
    if (total_weight > 0.001) {
        shadow_val = total_shadow / total_weight;
    } else {
        shadow_val = best_shadow;
    }

    // Faces pointing away from sun are always in shadow regardless of mask
    let shadow = select(shadow_val, 0.0, raw_ndotl <= 0.0);

    // Modulate lighting by time of day
    let day = 1.0 - atmosphere.night_factor;
    let sky_light = max(surface.normal.y * 0.5 + 0.5, 0.0) * 0.15 * day;

    let effective_shadow = mix(0.0, 1.0, shadow);
    let ambient = mix(0.04, 0.25, day);
    let diffuse = 0.7 * ndotl * effective_shadow * day;
    let ao = mix(0.4, 1.0, surface.ao);
    let lit_color = surface.base_color * (ambient * ao + sky_light * ao + diffuse);

    // Exponential distance fog
    let view_dir = normalize(surface.world_pos - camera.camera_local_pos);
    let fog_color = sample_fog_color(view_dir);
    let dist = length(surface.world_pos);
    let fog_factor = 1.0 - exp(-dist * atmosphere.fog_density);
    let final_color = mix(lit_color, fog_color, fog_factor);

    return vec4<f32>(final_color, 1.0);
}
