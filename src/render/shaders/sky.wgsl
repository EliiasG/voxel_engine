// sky_sample.wgsl is concatenated before this file, providing PI and night_sky_gradient().

struct SkyVaryings {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_sky(@builtin(vertex_index) vi: u32) -> SkyVaryings {
    var out: SkyVaryings;
    let uv = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
    // Output at depth 0.0 in clip space (reversed-Z far plane) so depth test Equal passes
    // only where the depth buffer is still the clear value (no geometry).
    out.position = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(uv.x, 1.0 - uv.y);
    return out;
}

@fragment
fn fs_sky(in: SkyVaryings) -> @location(0) vec4<f32> {
    // Reconstruct view direction: unproject near-plane point, subtract camera position
    let ndc_xy = vec2<f32>(
        in.uv.x * 2.0 - 1.0,
        (1.0 - in.uv.y) * 2.0 - 1.0,
    );
    let clip = camera.inv_view_proj * vec4<f32>(ndc_xy, 1.0, 1.0);
    let near_point = clip.xyz / clip.w;
    let view_dir = normalize(near_point - camera.camera_local_pos);

    // Sky LUT coordinates: (relative yaw to sun, elevation)
    let sun_yaw = atan2(atmosphere.sun_direction.x, atmosphere.sun_direction.z);
    let view_yaw = atan2(view_dir.x, view_dir.z);
    let rel_yaw = view_yaw - sun_yaw;
    let elevation = asin(clamp(view_dir.y, -1.0, 1.0));

    let lut_u = rel_yaw / (2.0 * PI) + 0.5; // [-π, π] → [0, 1], wraps via Repeat sampler

    // Above-horizon sky (clamp LUT elevation just above horizon to avoid ground-hit dark band)
    let sky_elev = max(elevation, 0.02);
    let lut_v = sky_elev / PI + 0.5;
    let sky_color = textureSample(sky_lut, sky_sampler, vec3<f32>(lut_u, lut_v, atmosphere.lut_w)).rgb;
    let night_grad = night_sky_gradient(sky_elev) * atmosphere.night_factor;
    let sky_with_night = max(sky_color, night_grad);

    // Sun disk at true 3D position
    let cos_sun_angle = dot(view_dir, atmosphere.sun_direction);
    let sun_edge = cos(atmosphere.sun_angular_radius);
    let sun_disk = smoothstep(sun_edge - 0.0005, sun_edge, cos_sun_angle) * atmosphere.sun_intensity;

    // Moon disk (opposite the sun)
    let moon_dir = -atmosphere.sun_direction;
    let cos_moon = dot(view_dir, moon_dir);
    let moon_edge = cos(atmosphere.sun_angular_radius * 2.0);
    let moon_disk = smoothstep(moon_edge - 0.001, moon_edge, cos_moon)
        * 0.3 * atmosphere.night_factor;
    let moon_color = vec3<f32>(0.7, 0.8, 1.0) * moon_disk;

    let above_color = sky_with_night + vec3<f32>(sun_disk) + moon_color;

    // Below-horizon: fog LUT + subtle night horizon glow
    let fog_color = textureSample(fog_lut, sky_sampler, vec3<f32>(lut_u, 0.56, atmosphere.lut_w)).rgb;
    let below_t = clamp(-elevation / (PI * 0.15), 0.0, 1.0);
    let night_below = mix(vec3<f32>(0.004, 0.006, 0.018), vec3<f32>(0.001, 0.001, 0.004), below_t);
    let below_color = max(fog_color, night_below * atmosphere.night_factor);

    // Smooth horizon blend instead of hard if/else
    let horizon_blend = smoothstep(-0.05, 0.05, elevation);
    var final_color = mix(below_color, above_color, horizon_blend);

    return vec4<f32>(final_color, 1.0);
}
