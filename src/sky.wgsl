const PI: f32 = 3.14159265359;

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

    var final_color: vec3<f32>;
    if elevation < 0.0 {
        // Below horizon: use fog LUT (no glow) at fixed elevation — matches fog color
        final_color = textureSample(fog_lut, sky_sampler, vec3<f32>(lut_u, 0.56, atmosphere.lut_w)).rgb;
    } else {
        // Above horizon: full sky with Mie glow + sun disk
        let lut_v = elevation / PI + 0.5;
        let sky_color = textureSample(sky_lut, sky_sampler, vec3<f32>(lut_u, lut_v, atmosphere.lut_w)).rgb;

        let cos_sun_angle = dot(view_dir, atmosphere.sun_direction);
        let sun_edge = cos(atmosphere.sun_angular_radius);
        let sun_disk = smoothstep(sun_edge - 0.0005, sun_edge, cos_sun_angle) * atmosphere.sun_intensity;

        final_color = sky_color + vec3<f32>(sun_disk);
    }

    return vec4<f32>(final_color, 1.0);
}
