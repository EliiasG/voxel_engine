// Shared sky/fog color sampling. Requires atmosphere bind group declarations.

const PI: f32 = 3.14159265359;

fn night_sky_gradient(elevation: f32) -> vec3<f32> {
    let zenith = vec3<f32>(0.003, 0.005, 0.018);
    let horizon = vec3<f32>(0.010, 0.015, 0.045);
    let t = clamp(elevation / (PI * 0.5), 0.0, 1.0);
    return mix(horizon, zenith, t);
}

/// Sample fog/sky color for a given view direction, blending LUT with night gradient.
fn sample_fog_color(view_dir: vec3<f32>) -> vec3<f32> {
    let sun_yaw = atan2(atmosphere.sun_direction.x, atmosphere.sun_direction.z);
    let fog_yaw = atan2(view_dir.x, view_dir.z);
    let fog_rel_yaw = fog_yaw - sun_yaw;
    let fog_u = fog_rel_yaw / (2.0 * PI) + 0.5;
    let fog_elevation = asin(clamp(view_dir.y, 0.15, 1.0) * 0.8);
    let fog_v = fog_elevation / PI + 0.5;
    let fog_direct = textureSample(fog_lut, sky_sampler, vec3<f32>(fog_u, fog_v, atmosphere.lut_w)).rgb;
    // Yaw-independent fog: fixed perpendicular-to-sun samples (u=0.25/0.75 = ±90° from sun)
    let fog_uniform = (
        textureSample(fog_lut, sky_sampler, vec3<f32>(0.25, fog_v, atmosphere.lut_w)).rgb +
        textureSample(fog_lut, sky_sampler, vec3<f32>(0.75, fog_v, atmosphere.lut_w)).rgb
    ) * 0.5;
    // Below horizon: fade from directional to uniform to prevent spoke artifact at nadir
    let below_t = smoothstep(0.0, 0.3, -view_dir.y);
    let fog_lut_color = mix(fog_direct, fog_uniform, below_t);
    // At night the LUT goes near-black; use the night gradient as a floor so fog matches the sky
    let night_grad = night_sky_gradient(fog_elevation) * atmosphere.night_factor;
    return max(fog_lut_color, night_grad);
}
