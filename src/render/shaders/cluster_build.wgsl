// Cluster build compute pass for slice 2 of clustered lighting.
//
// Topology: 16 × 9 × 24 = 3456 clusters. Standard Doom-2016 layout with
// log-Z depth slicing. One thread = one cluster. The thread iterates the
// chunk and dynamic light buffers, tests each light's bounding sphere
// against its cluster's view-frustum AABB (in camera-relative world
// space), and atomically appends hits into its pre-allocated slice of
// the cluster index list.
//
// Index encoding: bit 28 = buffer id (0 = chunk, 1 = dynamic),
// bits 27..0 = local index into that buffer. Lets the fragment shader
// dispatch a single index to the right buffer without sorting.
//
// Camera bind group is concatenated as group 0 (provides `camera`).

struct SimpleLight {
    chunk_pos: vec3<i32>,
    range: f32,
    local_pos: vec3<f32>,
    intensity: f32,
    color: vec3<f32>,
    inner_cos: f32,
    direction: vec3<f32>,
    outer_cos: f32,
};

struct LightCounts {
    chunk_light_count: u32,
    dynamic_light_count: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(1) @binding(0)
var<storage, read> chunk_lights: array<SimpleLight>;
@group(1) @binding(1)
var<storage, read> dynamic_lights: array<SimpleLight>;
@group(1) @binding(2)
var<uniform> light_counts: LightCounts;
@group(1) @binding(3)
var<storage, read_write> cluster_index_counts: array<atomic<u32>>;
@group(1) @binding(4)
var<storage, read_write> cluster_index_list: array<u32>;

const NUM_CLUSTERS_X: u32 = 16u;
const NUM_CLUSTERS_Y: u32 = 9u;
const NUM_CLUSTERS_Z: u32 = 24u;
const NUM_CLUSTERS: u32 = 3456u; // 16 * 9 * 24
const MAX_LIGHTS_PER_CLUSTER: u32 = 256u;

const CLUSTER_NEAR: f32 = 0.1;
const CLUSTER_FAR: f32 = 1024.0;

const CHUNK_SIZE_LIGHTS: i32 = 32;

const CLUSTER_INDEX_MASK: u32 = 0x0FFFFFFFu;
const BUFFER_ID_CHUNK: u32 = 0u;
const BUFFER_ID_DYNAMIC: u32 = 1u;

struct Aabb {
    min: vec3<f32>,
    max: vec3<f32>,
};

/// Unproject an NDC point to camera-relative world space.
fn ndc_to_world(ndc: vec3<f32>) -> vec3<f32> {
    let h = camera.inv_view_proj * vec4<f32>(ndc, 1.0);
    return h.xyz / h.w;
}

/// Convert a linear (positive) depth to NDC z under the project's
/// reverse-Z infinite-far perspective. Derived from the matrix in
/// `camera::perspective`: z_ndc = near / d.
fn depth_to_ndc_z(d: f32) -> f32 {
    return CLUSTER_NEAR / d;
}

/// Compute the camera-relative-world-space AABB for a single cluster.
fn compute_cluster_aabb(cluster_x: u32, cluster_y: u32, cluster_z: u32) -> Aabb {
    let nx = f32(NUM_CLUSTERS_X);
    let ny = f32(NUM_CLUSTERS_Y);
    let nz = f32(NUM_CLUSTERS_Z);

    // NDC x/y bounds. NDC ranges [-1, 1] in both axes.
    let ndc_x_min = -1.0 + 2.0 * f32(cluster_x) / nx;
    let ndc_x_max = -1.0 + 2.0 * f32(cluster_x + 1u) / nx;
    let ndc_y_min = -1.0 + 2.0 * f32(cluster_y) / ny;
    let ndc_y_max = -1.0 + 2.0 * f32(cluster_y + 1u) / ny;

    // Log-Z slicing in linear depth, converted to NDC z.
    let log_ratio = log(CLUSTER_FAR / CLUSTER_NEAR);
    let d_min = CLUSTER_NEAR * exp(log_ratio * f32(cluster_z) / nz);
    let d_max = CLUSTER_NEAR * exp(log_ratio * f32(cluster_z + 1u) / nz);
    // Reverse-Z: closer = higher ndc z.
    let ndc_z_max = depth_to_ndc_z(d_min);
    let ndc_z_min = depth_to_ndc_z(d_max);

    // Unproject the 8 corners and accumulate min/max in world space.
    var aabb_min = vec3<f32>(1e30);
    var aabb_max = vec3<f32>(-1e30);
    for (var i = 0u; i < 8u; i = i + 1u) {
        let nx_v = select(ndc_x_min, ndc_x_max, (i & 1u) != 0u);
        let ny_v = select(ndc_y_min, ndc_y_max, (i & 2u) != 0u);
        let nz_v = select(ndc_z_min, ndc_z_max, (i & 4u) != 0u);
        let world = ndc_to_world(vec3<f32>(nx_v, ny_v, nz_v));
        aabb_min = min(aabb_min, world);
        aabb_max = max(aabb_max, world);
    }
    return Aabb(aabb_min, aabb_max);
}

/// Camera-relative world-space center of a light, matching the
/// fragment-shader rebase math.
fn light_center(light: SimpleLight) -> vec3<f32> {
    let rel_chunk = light.chunk_pos - camera.chunk_offset;
    return vec3<f32>(rel_chunk * CHUNK_SIZE_LIGHTS) + light.local_pos;
}

/// Test whether a sphere of `radius` around `center` intersects the AABB.
fn sphere_aabb_intersect(center: vec3<f32>, radius: f32, aabb: Aabb) -> bool {
    let closest = clamp(center, aabb.min, aabb.max);
    let d = center - closest;
    return dot(d, d) <= radius * radius;
}

/// Atomic-append an encoded index to a cluster's slice. Drops the index
/// if the cluster cap is reached. The atomic counter still increments
/// past the cap so that fragment-side reads can clamp safely.
fn append_to_cluster(cluster_id: u32, encoded: u32) {
    let slot = atomicAdd(&cluster_index_counts[cluster_id], 1u);
    if (slot < MAX_LIGHTS_PER_CLUSTER) {
        cluster_index_list[cluster_id * MAX_LIGHTS_PER_CLUSTER + slot] = encoded;
    }
}

@compute @workgroup_size(64)
fn cs_cluster_build(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cluster_id = gid.x;
    if (cluster_id >= NUM_CLUSTERS) {
        return;
    }

    let cluster_z = cluster_id / (NUM_CLUSTERS_X * NUM_CLUSTERS_Y);
    let cluster_xy = cluster_id - cluster_z * NUM_CLUSTERS_X * NUM_CLUSTERS_Y;
    let cluster_y = cluster_xy / NUM_CLUSTERS_X;
    let cluster_x = cluster_xy - cluster_y * NUM_CLUSTERS_X;

    let aabb = compute_cluster_aabb(cluster_x, cluster_y, cluster_z);

    let chunk_n = light_counts.chunk_light_count;
    for (var i = 0u; i < chunk_n; i = i + 1u) {
        let light = chunk_lights[i];
        let center = light_center(light);
        if (sphere_aabb_intersect(center, light.range, aabb)) {
            append_to_cluster(cluster_id, (BUFFER_ID_CHUNK << 28u) | (i & CLUSTER_INDEX_MASK));
        }
    }

    let dyn_n = light_counts.dynamic_light_count;
    for (var i = 0u; i < dyn_n; i = i + 1u) {
        let light = dynamic_lights[i];
        let center = light_center(light);
        if (sphere_aabb_intersect(center, light.range, aabb)) {
            append_to_cluster(cluster_id, (BUFFER_ID_DYNAMIC << 28u) | (i & CLUSTER_INDEX_MASK));
        }
    }
}
