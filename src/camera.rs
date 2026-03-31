use bytemuck::{Pod, Zeroable};
use glam::IVec3;

use crate::chunk::CHUNK_SIZE;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
    /// Camera's chunk coordinate, subtracted from page metadata chunk_pos in the shader.
    pub chunk_offset: [i32; 3],
    pub _pad: i32,
    pub screen_size: [f32; 2],
    /// Sub-pixel jitter in pixel coordinates ([-0.5, +0.5] range)
    pub jitter_offset: [f32; 2],
    /// Inverse of the jittered view_proj (for depth → world position reconstruction)
    pub inv_view_proj: [[f32; 4]; 4],
    /// Previous frame's jittered view_proj (for TAA reprojection)
    pub prev_jittered_view_proj: [[f32; 4]; 4],
    /// Previous frame's chunk offset
    pub prev_chunk_offset: [i32; 3],
    /// Frame index for Halton jitter sequence
    pub frame_index: u32,
    pub _pad3: [f32; 3],
    pub _pad4: f32,
}

pub struct FlyCamera {
    /// World position in f64 for precision at large distances.
    pub position: [f64; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub speed: f32,
    pub sensitivity: f32,
    pub aspect: f32,
    pub fov_y: f32,
    pub near: f32,
    pub far: f32,
}

impl FlyCamera {
    pub fn new(position: [f64; 3]) -> Self {
        Self {
            position,
            yaw: 0.0,
            pitch: 0.0,
            speed: 50.0,
            sensitivity: 0.003,
            aspect: 16.0 / 9.0,
            fov_y: 70.0f32.to_radians(),
            near: 0.1,
            far: 1000.0,
        }
    }

    pub fn forward(&self) -> [f32; 3] {
        let (sy, cy) = self.yaw.sin_cos();
        [-sy, 0.0, -cy]
    }

    pub fn right(&self) -> [f32; 3] {
        let (sy, cy) = self.yaw.sin_cos();
        [cy, 0.0, -sy]
    }

    pub fn look_dir(&self) -> [f32; 3] {
        let (sp, cp) = self.pitch.sin_cos();
        let (sy, cy) = self.yaw.sin_cos();
        [-sy * cp, sp, -cy * cp]
    }

    pub fn rotate(&mut self, dx: f64, dy: f64) {
        self.yaw -= dx as f32 * self.sensitivity;
        self.pitch -= dy as f32 * self.sensitivity;
        let limit = std::f32::consts::FRAC_PI_2 - 0.01;
        self.pitch = self.pitch.clamp(-limit, limit);
    }

    pub fn move_dir(&mut self, forward: f32, right: f32, up: f32, dt: f32) {
        let fwd = self.forward();
        let rt = self.right();
        let speed = self.speed * dt;
        self.position[0] += (fwd[0] * forward + rt[0] * right) as f64 * speed as f64;
        self.position[1] += up as f64 * speed as f64;
        self.position[2] += (fwd[2] * forward + rt[2] * right) as f64 * speed as f64;
    }

    /// The chunk coordinate the camera is in.
    pub fn chunk_pos(&self) -> IVec3 {
        IVec3::new(
            (self.position[0] / CHUNK_SIZE as f64).floor() as i32,
            (self.position[1] / CHUNK_SIZE as f64).floor() as i32,
            (self.position[2] / CHUNK_SIZE as f64).floor() as i32,
        )
    }

    /// Build the camera uniform with camera-relative rendering.
    ///
    /// The view matrix translates by only the within-chunk offset (small f32).
    /// The chunk_offset is passed to the shader so it can compute
    /// `relative_origin = (chunk_pos - chunk_offset) * CHUNK_SIZE` in integers.
    pub fn uniform(&self) -> CameraUniform {
        let chunk = self.chunk_pos();

        // Camera position relative to its own chunk origin (always 0..CHUNK_SIZE, small f32)
        let local_eye = [
            (self.position[0] - chunk[0] as f64 * CHUNK_SIZE as f64) as f32,
            (self.position[1] - chunk[1] as f64 * CHUNK_SIZE as f64) as f32,
            (self.position[2] - chunk[2] as f64 * CHUNK_SIZE as f64) as f32,
        ];

        let dir = self.look_dir();
        let target = [
            local_eye[0] + dir[0],
            local_eye[1] + dir[1],
            local_eye[2] + dir[2],
        ];

        let view = look_at(local_eye, target, [0.0, 1.0, 0.0]);
        let proj = perspective(self.fov_y, self.aspect, self.near, self.far);

        CameraUniform {
            view_proj: mat4_mul(proj, view),
            chunk_offset: chunk.to_array(),
            _pad: 0,
            screen_size: [0.0; 2],
            jitter_offset: [0.0; 2],
            inv_view_proj: [[0.0; 4]; 4],
            prev_jittered_view_proj: [[0.0; 4]; 4],
            prev_chunk_offset: [0; 3],
            frame_index: 0,
            _pad3: [0.0; 3],
            _pad4: 0.0,
        }
    }
}

fn look_at(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> [[f32; 4]; 4] {
    let f = normalize(sub(target, eye));
    let s = normalize(cross(f, up));
    let u = cross(s, f);
    [
        [s[0], u[0], -f[0], 0.0],
        [s[1], u[1], -f[1], 0.0],
        [s[2], u[2], -f[2], 0.0],
        [-dot(s, eye), -dot(u, eye), dot(f, eye), 1.0],
    ]
}

fn perspective(fov_y: f32, aspect: f32, near: f32, _far: f32) -> [[f32; 4]; 4] {
    // Reversed-Z with infinite far plane: near maps to 1.0, infinity maps to 0.0.
    // This gives vastly better depth precision at all distances.
    let f = 1.0 / (fov_y / 2.0).tan();
    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, 0.0, -1.0],
        [0.0, 0.0, near, 0.0],
    ]
}

fn mat4_mul(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0f32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                out[i][j] += a[k][j] * b[i][k];
            }
        }
    }
    out
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len == 0.0 {
        return [0.0; 3];
    }
    [v[0] / len, v[1] / len, v[2] / len]
}

// --- TAA jitter helpers ---

fn halton_2(mut index: u32) -> f32 {
    let mut f = 0.5f32;
    let mut result = 0.0f32;
    while index > 0 {
        if index & 1 != 0 {
            result += f;
        }
        f *= 0.5;
        index >>= 1;
    }
    result
}

fn halton_3(mut index: u32) -> f32 {
    let mut f = 1.0f32 / 3.0;
    let mut result = 0.0f32;
    while index > 0 {
        result += f * (index % 3) as f32;
        index /= 3;
        f /= 3.0;
    }
    result
}

/// Returns (jx, jy) in [-0.5, +0.5] pixel range for the given frame index.
pub fn taa_jitter(frame_index: u32) -> (f32, f32) {
    let idx = (frame_index % 16) + 1;
    (halton_2(idx) - 0.5, halton_3(idx) - 0.5)
}

/// Apply sub-pixel jitter to a combined view_proj matrix.
/// jx, jy are in pixel coordinates; width, height are screen dimensions.
pub fn apply_jitter(
    view_proj: &mut [[f32; 4]; 4],
    jx: f32,
    jy: f32,
    width: f32,
    height: f32,
) {
    let jx_ndc = 2.0 * jx / width;
    let jy_ndc = 2.0 * jy / height;
    for col in 0..4 {
        view_proj[col][0] += jx_ndc * view_proj[col][3];
        view_proj[col][1] += jy_ndc * view_proj[col][3];
    }
}

/// 4x4 matrix inverse via Gaussian elimination (column-major storage).
pub fn invert_mat4(m: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut a = *m;
    let mut inv = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    for col in 0..4 {
        let mut max_val = 0.0f32;
        let mut max_row = col;
        for row in col..4 {
            let v = a[col][row].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_row != col {
            for c in 0..4 {
                let tmp = a[c][col];
                a[c][col] = a[c][max_row];
                a[c][max_row] = tmp;
                let tmp = inv[c][col];
                inv[c][col] = inv[c][max_row];
                inv[c][max_row] = tmp;
            }
        }
        let pivot = a[col][col];
        if pivot.abs() < 1e-12 {
            return inv;
        }
        for c in 0..4 {
            a[c][col] /= pivot;
            inv[c][col] /= pivot;
        }
        for row in 0..4 {
            if row == col {
                continue;
            }
            let factor = a[col][row];
            for c in 0..4 {
                a[c][row] -= factor * a[c][col];
                inv[c][row] -= factor * inv[c][col];
            }
        }
    }
    inv
}

/// Extract 6 frustum planes from a view-projection matrix (wgpu depth [0,1]).
/// Each plane [a, b, c, d] where ax + by + cz + d >= 0 means inside.
pub fn extract_frustum_planes(vp: &[[f32; 4]; 4]) -> [[f32; 4]; 6] {
    let row = |i: usize| -> [f32; 4] { [vp[0][i], vp[1][i], vp[2][i], vp[3][i]] };
    let r0 = row(0);
    let r1 = row(1);
    let r2 = row(2);
    let r3 = row(3);

    let add = |a: [f32; 4], b: [f32; 4]| [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]];
    let sub = |a: [f32; 4], b: [f32; 4]| [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]];

    let mut planes = [
        add(r3, r0), // Left
        sub(r3, r0), // Right
        add(r3, r1), // Bottom
        sub(r3, r1), // Top
        sub(r3, r2), // Near (reversed-Z: depth=1 at near)
        r2,          // Far (reversed-Z: depth=0 at far/infinity)
    ];

    for plane in &mut planes {
        let len = (plane[0] * plane[0] + plane[1] * plane[1] + plane[2] * plane[2]).sqrt();
        if len > 0.0 {
            plane[0] /= len;
            plane[1] /= len;
            plane[2] /= len;
            plane[3] /= len;
        }
    }

    planes
}

/// Test if an AABB is at least partially inside the frustum.
pub fn is_aabb_in_frustum(planes: &[[f32; 4]; 6], min: [f32; 3], max: [f32; 3]) -> bool {
    for plane in planes {
        let px = if plane[0] >= 0.0 { max[0] } else { min[0] };
        let py = if plane[1] >= 0.0 { max[1] } else { min[1] };
        let pz = if plane[2] >= 0.0 { max[2] } else { min[2] };

        if plane[0] * px + plane[1] * py + plane[2] * pz + plane[3] < 0.0 {
            return false;
        }
    }
    true
}
