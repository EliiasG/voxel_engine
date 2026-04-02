// Voxel vertex shader: shared between full and normal-only pipelines.
// Requires camera and metadata bind groups.

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
    @location(3) @interpolate(flat) material_id: u32,
    @location(4) @interpolate(flat) direction: u32,
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
    out.material_id = face.data1.z;
    out.direction = direction;

    return out;
}
