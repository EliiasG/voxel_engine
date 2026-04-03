// Transparent voxel fragment shader (WBOIT accumulation).
// Concatenated after: bind groups, sky_sample.wgsl, lighting.wgsl, voxel_vertex.wgsl
// Outputs to two render targets: accumulation (Rgba16Float) + revealage (R8Unorm).

fn transparent_color(material_id: u32) -> vec4<f32> {
    switch material_id {
        // GLASS = 4: light blue tint
        case 4u: { return vec4<f32>(0.6, 0.8, 0.9, 0.3); }
        // WATER = 5: blue-green tint
        case 5u: { return vec4<f32>(0.2, 0.5, 0.7, 0.5); }
        default: { return vec4<f32>(1.0, 1.0, 1.0, 0.5); }
    }
}

struct WboitOutput {
    @location(0) accumulation: vec4<f32>,
    @location(1) revealage: f32,
};

@fragment
fn fs_transparent(in: VertexOutput) -> WboitOutput {
    let tint = transparent_color(in.material_id);

    // Apply basic lighting to the transparent surface
    let light_dir = atmosphere.sun_direction;
    let ndotl = max(dot(in.normal, light_dir), 0.0);
    let day = 1.0 - atmosphere.night_factor;
    let ambient = mix(0.08, 0.25, day);
    let sky_light = max(in.normal.y * 0.5 + 0.5, 0.0) * mix(0.04, 0.15, day);
    let diffuse = 0.7 * ndotl * day;
    let lit_color = tint.rgb * (ambient + sky_light + diffuse);

    let alpha = tint.a;

    // WBOIT weight function: depth-based weight for order independence
    // McGuire & Bavoil 2013, equation 10
    let d = in.clip_position.z; // reverse-Z: near=1, far=0
    let w = alpha * max(1e-2, 3e3 * d * d * d);

    var out: WboitOutput;
    out.accumulation = vec4<f32>(lit_color * alpha * w, alpha * w);
    out.revealage = alpha;
    return out;
}
