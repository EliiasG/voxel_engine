const PAGE_SIZE: u32 = 96u;
const CHUNK_SIZE: i32 = 32;

struct FaceInstance {
    @location(0) data0: vec4<u32>,  // x, y, z, w
    @location(1) data1: vec4<u32>,  // h, mat0, mat1, mat2
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

fn get_normal(dir: u32) -> vec3<f32> {
    switch dir {
        case 0u: { return vec3<f32>(1.0, 0.0, 0.0); }
        case 1u: { return vec3<f32>(-1.0, 0.0, 0.0); }
        case 2u: { return vec3<f32>(0.0, 1.0, 0.0); }
        case 3u: { return vec3<f32>(0.0, -1.0, 0.0); }
        case 4u: { return vec3<f32>(0.0, 0.0, 1.0); }
        case 5u: { return vec3<f32>(0.0, 0.0, -1.0); }
        default: { return vec3<f32>(0.0, 1.0, 0.0); }
    }
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
    face: FaceInstance,
) -> VertexOutput {
    var out: VertexOutput;

    let page_index = instance_index / PAGE_SIZE;
    let page_meta = metadata[page_index];

    // Unpack direction and LOD from packed field
    let direction = page_meta.direction_and_lod & 0xFFu;
    let lod = (page_meta.direction_and_lod >> 8u) & 0xFFu;
    let lod_scale = i32(1u << lod);
    let lod_scale_f = f32(lod_scale);

    // Camera-relative chunk origin (integer math, exact, result is small)
    let chunk_pos = vec3<i32>(page_meta.chunk_x, page_meta.chunk_y, page_meta.chunk_z);
    let rel_chunk = chunk_pos * lod_scale - camera.chunk_offset;
    let chunk_origin = vec3<f32>(rel_chunk * CHUNK_SIZE);

    // Unpack face data
    let pos = vec3<f32>(f32(face.data0.x), f32(face.data0.y), f32(face.data0.z));
    let w = f32(face.data0.w);
    let h = f32(face.data1.x);

    var u_factor: f32;
    var v_factor: f32;
    switch vertex_index {
        case 0u: { u_factor = 0.0; v_factor = 0.0; }
        case 1u: { u_factor = 1.0; v_factor = 0.0; }
        case 2u: { u_factor = 1.0; v_factor = 1.0; }
        case 3u: { u_factor = 0.0; v_factor = 0.0; }
        case 4u: { u_factor = 1.0; v_factor = 1.0; }
        case 5u: { u_factor = 0.0; v_factor = 1.0; }
        default: { u_factor = 0.0; v_factor = 0.0; }
    }

    var tangent_u: vec3<f32>;
    var tangent_v: vec3<f32>;
    var offset: vec3<f32>;

    switch direction {
        case 0u: { // +X
            offset = vec3<f32>(1.0, 0.0, 0.0);
            tangent_u = vec3<f32>(0.0, 1.0, 0.0);
            tangent_v = vec3<f32>(0.0, 0.0, 1.0);
        }
        case 1u: { // -X
            offset = vec3<f32>(0.0, 0.0, 0.0);
            tangent_u = vec3<f32>(0.0, 0.0, 1.0);
            tangent_v = vec3<f32>(0.0, 1.0, 0.0);
        }
        case 2u: { // +Y
            offset = vec3<f32>(0.0, 1.0, 0.0);
            tangent_u = vec3<f32>(0.0, 0.0, 1.0);
            tangent_v = vec3<f32>(1.0, 0.0, 0.0);
        }
        case 3u: { // -Y
            offset = vec3<f32>(0.0, 0.0, 0.0);
            tangent_u = vec3<f32>(1.0, 0.0, 0.0);
            tangent_v = vec3<f32>(0.0, 0.0, 1.0);
        }
        case 4u: { // +Z
            offset = vec3<f32>(0.0, 0.0, 1.0);
            tangent_u = vec3<f32>(1.0, 0.0, 0.0);
            tangent_v = vec3<f32>(0.0, 1.0, 0.0);
        }
        case 5u: { // -Z
            offset = vec3<f32>(0.0, 0.0, 0.0);
            tangent_u = vec3<f32>(0.0, 1.0, 0.0);
            tangent_v = vec3<f32>(1.0, 0.0, 0.0);
        }
        default: {
            offset = vec3<f32>(0.0, 0.0, 0.0);
            tangent_u = vec3<f32>(1.0, 0.0, 0.0);
            tangent_v = vec3<f32>(0.0, 1.0, 0.0);
        }
    }

    // Local position scaled by LOD
    let local_pos = (pos + offset + tangent_u * u_factor * w + tangent_v * v_factor * h) * lod_scale_f;
    let rel_pos = chunk_origin + local_pos;

    out.clip_position = camera.view_proj * vec4<f32>(rel_pos, 1.0);
    out.world_pos = rel_pos;
    out.normal = get_normal(direction);

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let abs_y = in.world_pos.y + f32(camera.chunk_offset.y * CHUNK_SIZE);
    let height_color = clamp(abs_y / 96.0, 0.0, 1.0);
    let base_color = vec3<f32>(
        0.3 + 0.4 * (1.0 - height_color),
        0.4 + 0.5 * height_color,
        0.2 + 0.2 * height_color,
    );

    let light_dir = normalize(vec3<f32>(0.3, 0.47, 0.5));
    let raw_ndotl = dot(in.normal, light_dir);
    let ndotl = max(raw_ndotl, 0.0);

    // Edge-aware shadow upscale: 2x2 bilinear with hard accept/reject
    let shadow_uv = in.clip_position.xy / camera.screen_size;
    let shadow_dims = vec2<f32>(textureDimensions(shadow_mask));
    let frag_height = dot(in.world_pos, in.normal);

    // Find the 2x2 texel footprint and sub-texel position
    let texel_pos = shadow_uv * shadow_dims - 0.5;
    let base = floor(texel_pos);
    let f = texel_pos - base; // sub-texel fraction [0,1]

    // Bilinear weights for the 4 texels
    let bw = vec4<f32>(
        (1.0 - f.x) * (1.0 - f.y), // top-left
        f.x * (1.0 - f.y),          // top-right
        (1.0 - f.x) * f.y,          // bottom-left
        f.x * f.y,                   // bottom-right
    );

    // Load the 4 texels directly (no interpolation)
    let base_i = vec2<i32>(base);
    let dims_i = vec2<i32>(shadow_dims);
    var shadow_vals: array<f32, 4>;
    var weights: array<f32, 4>;
    for (var i = 0; i < 4; i++) {
        let tc = clamp(base_i + vec2<i32>(i % 2, i / 2), vec2<i32>(0), dims_i - 1);
        let ss = textureLoad(shadow_mask, tc, 0);
        let n = textureLoad(shadow_normal, tc, 0).xyz * 2.0 - 1.0;
        let height_diff = abs(ss.g - frag_height);
        let accept = select(0.0, 1.0, dot(n, in.normal) > 0.9 && height_diff < 0.3);
        shadow_vals[i] = ss.r;
        weights[i] = bw[i] * accept;
    }

    let total_w = weights[0] + weights[1] + weights[2] + weights[3];
    var shadow_val: f32;
    if (total_w > 0.001) {
        shadow_val = (shadow_vals[0] * weights[0] + shadow_vals[1] * weights[1]
                    + shadow_vals[2] * weights[2] + shadow_vals[3] * weights[3]) / total_w;
    } else {
        // No texel matched — pick nearest
        let nearest = vec2<u32>(clamp(vec2<i32>(round(texel_pos)), vec2<i32>(0), vec2<i32>(shadow_dims) - 1));
        shadow_val = textureLoad(shadow_mask, nearest, 0).r;
    }

    // Faces pointing away from sun are always in shadow regardless of mask
    let shadow = select(shadow_val, 0.0, raw_ndotl <= 0.0);

    // Indirect sky light — surfaces facing up are brighter even in shadow
    let sky_light = max(in.normal.y * 0.5 + 0.5, 0.0) * 0.15;

    // Shadow doesn't fully kill diffuse — scattered light remains
    let effective_shadow = mix(0.15, 1.0, shadow);
    let ambient = 0.25;
    let diffuse = 0.7 * ndotl * effective_shadow;
    let lit_color = base_color * (ambient + sky_light + diffuse);

    return vec4<f32>(lit_color, 1.0);
}

@fragment
fn fs_normal(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.normal * 0.5 + 0.5, 1.0);
}
