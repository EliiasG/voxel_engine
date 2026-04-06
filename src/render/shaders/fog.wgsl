// Shared exponential distance fog.
//
// Requires:
//   - camera bind group (camera.camera_local_pos)
//   - atmosphere bind group (atmosphere.fog_density, fog_lut, sky_sampler)
//   - sky_sample.wgsl concatenated before this file (sample_fog_color)
//
// world_pos must be in camera-chunk-origin-relative space (i.e. the
// voxel_vertex.wgsl world_pos output, NOT absolute world coordinates).

fn apply_fog(color: vec3<f32>, world_pos: vec3<f32>) -> vec3<f32> {
    let to_frag = world_pos - camera.camera_local_pos;
    let dist = length(to_frag);
    let view_dir = to_frag / max(dist, 1e-6);
    let fog_color = sample_fog_color(view_dir);
    let fog_factor = 1.0 - exp(-dist * atmosphere.fog_density);
    return mix(color, fog_color, fog_factor);
}
