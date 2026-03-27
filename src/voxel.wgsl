struct CameraUniform {
    view_proj: mat4x4<f32>,
    chunk_offset: vec3<i32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct PageMetadata {
    chunk_x: i32,
    chunk_y: i32,
    chunk_z: i32,
    direction_and_lod: u32,
};

@group(1) @binding(0)
var<storage, read> metadata: array<PageMetadata>;

const PAGE_SIZE: u32 = 512u;
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

    let light_dir = normalize(vec3<f32>(0.3, 1.0, 0.5));
    let ndotl = max(dot(in.normal, light_dir), 0.0);
    let ambient = 0.3;
    let diffuse = 0.7 * ndotl;
    let lit_color = base_color * (ambient + diffuse);

    return vec4<f32>(lit_color, 1.0);
}
