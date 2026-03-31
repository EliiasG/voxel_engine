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
    @location(2) ao: f32,
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

    // Unpack AO corners: 2 bits each, packed in material[0] = data1.y
    let ao_packed = face.data1.y;
    let ao_00 = f32(ao_packed & 3u) / 3.0;
    let ao_10 = f32((ao_packed >> 2u) & 3u) / 3.0;
    let ao_01 = f32((ao_packed >> 4u) & 3u) / 3.0;
    let ao_11 = f32((ao_packed >> 6u) & 3u) / 3.0;

    // Flip quad diagonal when AO is asymmetric to fix interpolation artifacts
    let flip = ao_00 + ao_11 < ao_10 + ao_01;

    var u_factor: f32;
    var v_factor: f32;
    if flip {
        switch vertex_index {
            case 0u: { u_factor = 0.0; v_factor = 0.0; }
            case 1u: { u_factor = 1.0; v_factor = 0.0; }
            case 2u: { u_factor = 0.0; v_factor = 1.0; }
            case 3u: { u_factor = 1.0; v_factor = 0.0; }
            case 4u: { u_factor = 1.0; v_factor = 1.0; }
            case 5u: { u_factor = 0.0; v_factor = 1.0; }
            default: { u_factor = 0.0; v_factor = 0.0; }
        }
    } else {
        switch vertex_index {
            case 0u: { u_factor = 0.0; v_factor = 0.0; }
            case 1u: { u_factor = 1.0; v_factor = 0.0; }
            case 2u: { u_factor = 1.0; v_factor = 1.0; }
            case 3u: { u_factor = 0.0; v_factor = 0.0; }
            case 4u: { u_factor = 1.0; v_factor = 1.0; }
            case 5u: { u_factor = 0.0; v_factor = 1.0; }
            default: { u_factor = 0.0; v_factor = 0.0; }
        }
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
    out.ao = mix(mix(ao_00, ao_10, u_factor), mix(ao_01, ao_11, u_factor), v_factor);

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

    let light_dir = atmosphere.sun_direction;
    let raw_ndotl = dot(in.normal, light_dir);
    let ndotl = max(raw_ndotl, 0.0);

    // Edge-aware shadow upscale: 3x3 neighborhood with hard accept/reject
    let shadow_uv = in.clip_position.xy / camera.screen_size;
    let shadow_dims = vec2<f32>(textureDimensions(shadow_mask));
    let frag_height = dot(in.world_pos, in.normal);

    // Find the nearest texel and search a 3x3 neighborhood around it
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
            if (dot(n, in.normal) > 0.9 && height_diff < 0.3) {
                // Weight by distance from fragment's sub-texel position
                let d = vec2<f32>(tc) + 0.5 - (center_texel + 0.5);
                let w = 1.0 / (1.0 + dot(d, d)); // inverse distance weight
                total_shadow += ss.r * w;
                total_weight += w;
            }
            // Track closest texel as fallback
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

    // Indirect sky light — surfaces facing up are brighter even in shadow
    let sky_light = max(in.normal.y * 0.5 + 0.5, 0.0) * 0.15;

    // Shadow doesn't fully kill diffuse — scattered light remains
    let effective_shadow = mix(0.0, 1.0, shadow);
    let ambient = 0.25;
    let diffuse = 0.7 * ndotl * effective_shadow;
    let ao = mix(0.4, 1.0, in.ao);
    let lit_color = base_color * (ambient * ao + sky_light * ao + diffuse);

    // Exponential distance fog — blend toward sky horizon color
    let view_dir = normalize(in.world_pos - camera.camera_local_pos);
    let sun_yaw = atan2(atmosphere.sun_direction.x, atmosphere.sun_direction.z);
    let fog_yaw = atan2(view_dir.x, view_dir.z);
    let fog_rel_yaw = fog_yaw - sun_yaw;
    let fog_u = fog_rel_yaw / (2.0 * 3.14159265359) + 0.5;
    let fog_elevation = asin(clamp(view_dir.y, 0.15, 1.0)*0.8); // clamp to at/above horizon
    let fog_v = (fog_elevation) / 3.14159265359 + 0.5;
    let fog_color = textureSample(fog_lut, sky_sampler, vec3<f32>(fog_u, fog_v, atmosphere.lut_w)).rgb;
    let dist = length(in.world_pos);
    let fog_factor = 1.0 - exp(-dist * atmosphere.fog_density);
    let final_color = mix(lit_color, fog_color, fog_factor);

    return vec4<f32>(final_color, 1.0);
}

@fragment
fn fs_normal(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.normal * 0.5 + 0.5, 1.0);
}
