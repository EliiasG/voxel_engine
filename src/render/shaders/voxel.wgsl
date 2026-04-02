// Voxel material evaluation and fragment entry point.
// Concatenated after: bind groups, sky_sample.wgsl, lighting.wgsl, voxel_vertex.wgsl

// Atlas is a 2x2 grid of 512x512 textures within a 1024x1024 image.
// Each quadrant is selected by material_id + face direction.
//   (0,0) = dirt       (0.5,0) = grass top
//   (0,0.5) = stone    (0.5,0.5) = grass side

fn get_atlas_offset(material_id: u32, direction: u32) -> vec2<f32> {
    switch material_id {
        // STONE = 1
        case 1u: { return vec2<f32>(0.0, 0.5); }
        // DIRT = 2
        case 2u: { return vec2<f32>(0.0, 0.0); }
        // GRASS = 3
        case 3u: {
            if direction == 2u { // +Y (top)
                return vec2<f32>(0.5, 0.0);
            }
            if direction == 3u { // -Y (bottom)
                return vec2<f32>(0.0, 0.0); // dirt underneath
            }
            return vec2<f32>(0.5, 0.5); // sides
        }
        default: { return vec2<f32>(0.0, 0.0); }
    }
}

// Triplanar projection: project world position onto 2D based on face direction.
fn triplanar_project(pos: vec3<f32>, direction: u32) -> vec2<f32> {
    switch direction {
        case 0u: { return vec2<f32>(-pos.z, -pos.y); } // +X
        case 1u: { return vec2<f32>( pos.z, -pos.y); } // -X
        case 2u: { return vec2<f32>( pos.x,  pos.z); } // +Y
        case 3u: { return vec2<f32>(-pos.x,  pos.z); } // -Y
        case 4u: { return vec2<f32>( pos.x, -pos.y); } // +Z
        case 5u: { return vec2<f32>(-pos.x, -pos.y); } // -Z
        default: { return vec2<f32>(0.0, 0.0); }
    }
}

fn evaluate_material(in: VertexOutput) -> Surface {
    // Reconstruct absolute world position (block coordinates)
    let abs_pos = in.world_pos + vec3<f32>(camera.chunk_offset * CHUNK_SIZE);

    // Triplanar project to 2D face coordinates
    let proj = triplanar_project(abs_pos, in.direction);

    // Tile size: texture repeats every tile_size blocks
    let tile_size = 64.0;
    var uv = proj / tile_size;
    // Wrap to [0,1) — fract handles negatives correctly in WGSL
    uv = fract(uv);

    // Scale into atlas quadrant (each is 0.5 x 0.5) and offset
    let atlas_offset = get_atlas_offset(in.material_id, in.direction);
    let final_uv = uv * 0.5 + atlas_offset;

    let base_color = textureSample(atlas_texture, atlas_sampler, final_uv).rgb;

    return Surface(base_color, in.normal, in.world_pos, in.clip_position, in.ao);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let surface = evaluate_material(in);
    return apply_lighting(surface);
}
