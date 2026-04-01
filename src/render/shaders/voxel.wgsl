// Voxel material evaluation and fragment entry point.
// Concatenated after: bind groups, sky_sample.wgsl, lighting.wgsl, voxel_vertex.wgsl

fn evaluate_material(in: VertexOutput) -> Surface {
    let abs_y = in.world_pos.y + f32(camera.chunk_offset.y * CHUNK_SIZE);
    let height_color = clamp(abs_y / 500.0, 0.0, 1.0);
    let base_color = vec3<f32>(
        0.3 + 0.4 * (1.0 - height_color),
        0.4 + 0.5 * height_color,
        0.2 + 0.2 * height_color,
    );

    return Surface(base_color, in.normal, in.world_pos, in.clip_position, in.ao);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let surface = evaluate_material(in);
    return apply_lighting(surface);
}
